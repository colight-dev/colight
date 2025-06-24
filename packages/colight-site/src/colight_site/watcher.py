"""File watching functionality for colight-site."""

import pathlib
from typing import Optional, List
from watchfiles import watch
import fnmatch

from . import api
from .constants import DEFAULT_INLINE_THRESHOLD


def watch_and_build(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    verbose: bool = False,
    format: str = "markdown",
    hide_statements: bool = False,
    hide_visuals: bool = False,
    hide_code: bool = False,
    continue_on_error: bool = True,
    colight_output_path: Optional[str] = None,
    colight_embed_path: Optional[str] = None,
    inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    include: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
):
    """Watch for changes and rebuild automatically."""
    # Default include pattern
    if include is None:
        include = ["*.py"]

    print(f"Watching {input_path} for changes...")
    if verbose:
        print(f"Include patterns: {include}")
        if ignore:
            print(f"Ignore patterns: {ignore}")

    # Build initially
    if input_path.is_file():
        api.build_file(
            input_path,
            output_path,
            verbose=verbose,
            format=format,
            hide_statements=hide_statements,
            hide_visuals=hide_visuals,
            hide_code=hide_code,
            continue_on_error=continue_on_error,
            colight_output_path=colight_output_path,
            colight_embed_path=colight_embed_path,
            inline_threshold=inline_threshold,
        )
    else:
        api.build_directory(
            input_path,
            output_path,
            verbose=verbose,
            format=format,
            hide_statements=hide_statements,
            hide_visuals=hide_visuals,
            hide_code=hide_code,
            continue_on_error=continue_on_error,
            colight_output_path=colight_output_path,
            colight_embed_path=colight_embed_path,
            inline_threshold=inline_threshold,
        )

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
            if verbose:
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
                        api.build_file(
                            input_path,
                            output_path,
                            verbose=verbose,
                            format=format,
                            hide_statements=hide_statements,
                            hide_visuals=hide_visuals,
                            hide_code=hide_code,
                            continue_on_error=continue_on_error,
                            colight_output_path=colight_output_path,
                            colight_embed_path=colight_embed_path,
                        )
                        if verbose:
                            print(f"Rebuilt {input_path}")
                else:
                    # Rebuild affected files
                    for changed_file in matching_changes:
                        # Debug logging
                        print(f"DEBUG: Checking changed file: {changed_file}")
                        print(f"DEBUG: Input path: {input_path}")
                        print(f"DEBUG: Input path absolute: {input_path.absolute()}")
                        print(
                            f"DEBUG: Changed file absolute: {changed_file.absolute()}"
                        )
                        print(
                            f"DEBUG: Is relative to? {changed_file.is_relative_to(input_path)}"
                        )

                        # Try with absolute paths
                        try:
                            is_relative_abs = changed_file.absolute().is_relative_to(
                                input_path.absolute()
                            )
                            print(
                                f"DEBUG: Is relative to (absolute paths)? {is_relative_abs}"
                            )
                        except Exception as e:
                            print(f"DEBUG: Error checking relative paths: {e}")

                        if changed_file.is_relative_to(input_path.resolve()):
                            rel_path = changed_file.relative_to(input_path.resolve())
                            suffix = ".html" if format == "html" else ".md"
                            output_file = output_path / rel_path.with_suffix(suffix)
                            api.build_file(
                                changed_file,
                                output_file,
                                verbose=verbose,
                                format=format,
                                hide_statements=hide_statements,
                                hide_visuals=hide_visuals,
                                hide_code=hide_code,
                                continue_on_error=continue_on_error,
                                colight_output_path=colight_output_path,
                                colight_embed_path=colight_embed_path,
                                inline_threshold=inline_threshold,
                            )
                            if verbose:
                                print(f"Rebuilt {changed_file}")
            except Exception as e:
                print(f"Error during rebuild: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
