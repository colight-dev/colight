"""File watching functionality for colight-site."""

import asyncio
import pathlib
from typing import Optional, List
from watchfiles import watch
import fnmatch
import threading

from . import api
from .builder import BuildConfig
from .server_live import LiveReloadServer
from .index_generator import generate_index_html


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
        # Handle legacy kwargs that might have individual flags
        if (
            "hide_statements" in kwargs
            or "hide_visuals" in kwargs
            or "hide_code" in kwargs
        ):
            pragma_tags = set()
            if kwargs.pop("hide_statements", False):
                pragma_tags.add("hide-statements")
            if kwargs.pop("hide_visuals", False):
                pragma_tags.add("hide-visuals")
            if kwargs.pop("hide_code", False):
                pragma_tags.add("hide-code")
            kwargs["pragma_tags"] = pragma_tags

        # Handle format -> formats conversion
        if "format" in kwargs and "formats" not in kwargs:
            kwargs["formats"] = {kwargs.pop("format")}

        config = BuildConfig(**kwargs)
    else:
        # If config provided but kwargs also given, create new config with overrides
        if kwargs:
            config_dict = {
                "verbose": config.verbose,
                "pragma_tags": config.pragma_tags.copy(),
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
        api.build_file(input_path, output_path, config=config)
    else:
        api.build_directory(input_path, output_path, config=config)

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
                        api.build_file(input_path, output_path, config=config)
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
                            suffix = ".html" if config.format == "html" else ".md"
                            output_file = output_path / rel_path.with_suffix(suffix)
                            api.build_file(changed_file, output_file, config=config)
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
        # Handle legacy kwargs that might have individual flags
        if (
            "hide_statements" in kwargs
            or "hide_visuals" in kwargs
            or "hide_code" in kwargs
        ):
            pragma_tags = set()
            if kwargs.pop("hide_statements", False):
                pragma_tags.add("hide-statements")
            if kwargs.pop("hide_visuals", False):
                pragma_tags.add("hide-visuals")
            if kwargs.pop("hide_code", False):
                pragma_tags.add("hide-code")
            kwargs["pragma_tags"] = pragma_tags

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
                "pragma_tags": config.pragma_tags.copy(),
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
        api.build_file(input_path, output_file, config=config)
    else:
        api.build_directory(input_path, output_path, config=config)
        # Generate index page for directory mode
        generate_index_html(input_path, output_path, include, ignore)

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
                            api.build_file(input_path, output_file, config=config)
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
                                api.build_file(changed_file, output_file, config=config)
                                if config.verbose:
                                    print(f"Rebuilt {changed_file}")

                        # Regenerate index page
                        generate_index_html(input_path, output_path, include, ignore)
                        if config.verbose:
                            print("Regenerated index.html")

                except Exception as e:
                    print(f"Error during rebuild: {e}")
                    if config.verbose:
                        import traceback

                        traceback.print_exc()

    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
