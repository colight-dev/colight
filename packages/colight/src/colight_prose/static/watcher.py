"""File watching functionality for colight-prose."""

import asyncio
import fnmatch
import pathlib
import threading
from typing import List, Optional

from watchfiles import watch

import colight_prose.static.builder as builder
from colight_prose.static.server_watch import LiveReloadServer

from .builder import BuildConfig


def watch_and_build(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    config: Optional[BuildConfig] = None,
    include: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
    **kwargs,
):
    """Watch for changes and rebuild automatically.

    Args:
        input_path: Path to watch (file or directory)
        output_path: Where to write output
        config: BuildConfig object with build settings
        include: List of glob patterns to include
        ignore: List of glob patterns to ignore
        **kwargs: Additional keyword arguments to override config values
    """
    # Create config from provided config or kwargs
    if config is None:
        # Handle format -> formats conversion
        if "format" in kwargs and "formats" not in kwargs:
            kwargs["formats"] = {kwargs.pop("format")}

        config = BuildConfig(**kwargs)
    else:
        # If config provided but kwargs also given, create new config with overrides
        if kwargs:
            config_dict = {
                "verbose": config.verbose,
                "pragma": config.pragma.copy(),
                "formats": config.formats.copy(),
                "continue_on_error": config.continue_on_error,
                "colight_output_path": config.colight_output_path,
                "colight_embed_path": config.colight_embed_path,
                "inline_threshold": config.inline_threshold,
                "in_subprocess": config.in_subprocess,
            }
            config_dict.update(kwargs)
            config = BuildConfig(**config_dict)
    # Default include pattern
    if include is None:
        include = ["*.py"]

    print(f"Watching {input_path} for changes...")
    if config.verbose:
        print(f"Include patterns: {include}")
        if ignore:
            print(f"Ignore patterns: {ignore}")

    # Build initially
    if input_path.is_file():
        if output_path.suffix:
            builder.build_file(input_path, output_path, config=config)
        else:
            builder.build_file(input_path, output_dir=output_path, config=config)
    else:
        builder.build_directory(input_path, output_path, config=config)

    # Helper function to check if file matches patterns
    def matches_patterns(
        file_path: pathlib.Path,
        include_patterns: List[str],
        ignore_patterns: Optional[List[str]] = None,
    ) -> bool:
        """Check if file matches include patterns and doesn't match ignore patterns."""
        file_str = str(file_path)

        # Check if file matches any include pattern
        matches_include = any(
            fnmatch.fnmatch(file_str, pattern)
            or fnmatch.fnmatch(file_path.name, pattern)
            for pattern in include_patterns
        )

        if not matches_include:
            return False

        # Check if file matches any ignore pattern
        if ignore_patterns:
            matches_ignore = any(
                fnmatch.fnmatch(file_str, pattern)
                or fnmatch.fnmatch(file_path.name, pattern)
                for pattern in ignore_patterns
            )
            if matches_ignore:
                return False

        return True

    # Watch for changes
    for changes in watch(input_path):
        changed_files = {pathlib.Path(path) for _, path in changes}

        # Filter files based on include/ignore patterns
        matching_changes = {
            f for f in changed_files if matches_patterns(f, include, ignore)
        }

        if matching_changes:
            if config.verbose:
                print(
                    f"Changes detected: {', '.join(str(f) for f in matching_changes)}"
                )
                print(
                    {
                        "is_file": input_path.is_file(),
                        "in matching changes": input_path in matching_changes,
                        "matching changes": matching_changes,
                    }
                )
            try:
                if input_path.is_file():
                    if input_path in matching_changes:
                        if output_path.suffix:
                            builder.build_file(
                                input_path, output_file=output_path, config=config
                            )
                        else:
                            builder.build_file(
                                input_path, output_dir=output_path, config=config
                            )
                        if config.verbose:
                            print(f"Rebuilt {input_path}")
                else:
                    # Rebuild affected files
                    for changed_file in matching_changes:
                        # Try with absolute paths
                        try:
                            changed_file.absolute().is_relative_to(
                                input_path.absolute()
                            )
                        except Exception as e:
                            print(f"DEBUG: Error checking relative paths: {e}")

                        if changed_file.is_relative_to(input_path.resolve()):
                            rel_path = changed_file.relative_to(input_path.resolve())
                            suffix = ".html" if "html" in config.formats else ".md"
                            output_file = output_path / rel_path.with_suffix(suffix)
                            builder.build_file(
                                changed_file, output_file=output_file, config=config
                            )
                            if config.verbose:
                                print(f"Rebuilt {changed_file}")
            except Exception as e:
                print(f"Error during rebuild: {e}")
                if config.verbose:
                    import traceback

                    traceback.print_exc()


def watch_build_and_serve(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    config: Optional[BuildConfig] = None,
    include: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
    host: str = "127.0.0.1",
    http_port: int = 5500,
    ws_port: int = 5501,
    open_url: bool = True,
    **kwargs,
):
    """Watch for changes, rebuild, and serve with live reload.

    Args:
        input_path: Path to watch (file or directory)
        output_path: Where to write output
        config: BuildConfig object with build settings
        include: List of glob patterns to include
        ignore: List of glob patterns to ignore
        host: Host for the dev server
        http_port: Port for HTTP server
        ws_port: Port for WebSocket server
        open_url: Whether to open browser on start
        **kwargs: Additional keyword arguments to override config values
    """
    # Create config from provided config or kwargs
    if config is None:
        # Handle format -> formats conversion
        if "format" in kwargs and "formats" not in kwargs:
            kwargs["formats"] = {kwargs.pop("format")}

        # For serving, default to HTML format
        if "formats" not in kwargs:
            kwargs["formats"] = {"html"}

        config = BuildConfig(**kwargs)
    else:
        # If config provided but kwargs also given, create new config with overrides
        if kwargs:
            config_dict = {
                "verbose": config.verbose,
                "pragma": config.pragma.copy(),
                "formats": config.formats.copy(),
                "continue_on_error": config.continue_on_error,
                "colight_output_path": config.colight_output_path,
                "colight_embed_path": config.colight_embed_path,
                "inline_threshold": config.inline_threshold,
                "in_subprocess": config.in_subprocess,
            }
            config_dict.update(kwargs)
            config = BuildConfig(**config_dict)

    # Ensure HTML format for serving
    if "html" not in config.formats:
        config.formats.add("html")

    # Default include pattern
    if include is None:
        include = ["*.py"]

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Watching {input_path} for changes...")
    print(f"Output directory: {output_path}")

    # Build initially
    if input_path.is_file():
        # For single file, generate proper output file path
        output_file = output_path / input_path.with_suffix(".html").name
        builder.build_file(input_path, output_file=output_file, config=config)
    else:
        builder.build_directory(input_path, output_path, config=config)
        # Index is now generated as JSON via API endpoint

    # Start the live reload server
    # Set up roots like the original devserver
    roots = {
        "/": output_path,
    }

    # Also serve dist directory if it exists (for colight assets)
    dist_dir = pathlib.Path("dist").resolve()
    if dist_dir.exists():
        roots["/dist"] = dist_dir

    server = LiveReloadServer(
        roots=roots,
        host=host,
        http_port=http_port,
        ws_port=ws_port,
        open_url_delay=open_url,
    )

    # Run server in a separate thread
    server_thread = threading.Thread(
        target=lambda: asyncio.run(server.serve()), daemon=True
    )
    server_thread.start()

    # Helper function to check if file matches patterns
    def matches_patterns(
        file_path: pathlib.Path,
        include_patterns: List[str],
        ignore_patterns: Optional[List[str]] = None,
    ) -> bool:
        """Check if file matches include patterns and doesn't match ignore patterns."""
        file_str = str(file_path)

        # Check if file matches any include pattern
        matches_include = any(
            fnmatch.fnmatch(file_str, pattern)
            or fnmatch.fnmatch(file_path.name, pattern)
            for pattern in include_patterns
        )

        if not matches_include:
            return False

        # Check if file matches any ignore pattern
        if ignore_patterns:
            matches_ignore = any(
                fnmatch.fnmatch(file_str, pattern)
                or fnmatch.fnmatch(file_path.name, pattern)
                for pattern in ignore_patterns
            )
            if matches_ignore:
                return False

        return True

    # Watch for changes
    try:
        for changes in watch(input_path):
            changed_files = {pathlib.Path(path) for _, path in changes}

            # Filter files based on include/ignore patterns
            matching_changes = {
                f for f in changed_files if matches_patterns(f, include, ignore)
            }

            if matching_changes:
                if config.verbose:
                    print(
                        f"Changes detected: {', '.join(str(f) for f in matching_changes)}"
                    )
                try:
                    if input_path.is_file():
                        if input_path in matching_changes:
                            output_file = (
                                output_path / input_path.with_suffix(".html").name
                            )
                            builder.build_file(
                                input_path, output_file=output_file, config=config
                            )
                            if config.verbose:
                                print(f"Rebuilt {input_path}")
                    else:
                        # Rebuild affected files
                        for changed_file in matching_changes:
                            if changed_file.is_relative_to(input_path.resolve()):
                                rel_path = changed_file.relative_to(
                                    input_path.resolve()
                                )
                                suffix = ".html"  # Always HTML for serving
                                output_file = output_path / rel_path.with_suffix(suffix)
                                builder.build_file(
                                    changed_file, output_file=output_file, config=config
                                )
                                if config.verbose:
                                    print(f"Rebuilt {changed_file}")

                        # Index regeneration is handled by API endpoint
                        if config.verbose:
                            print("Files rebuilt successfully")

                except Exception as e:
                    print(f"Error during rebuild: {e}")
                    if config.verbose:
                        import traceback

                        traceback.print_exc()

    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
