"""File watching functionality for colight-site."""

import pathlib
from typing import Optional
from watchfiles import watch

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
):
    """Watch for changes and rebuild automatically."""
    print(f"Watching {input_path} for changes...")

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

    # Watch for changes
    for changes in watch(input_path):
        changed_files = {pathlib.Path(path) for _, path in changes}

        # Filter for .colight.py files
        colight_changes = {
            f for f in changed_files if f.suffix == ".py" and ".colight" in f.name
        }

        if colight_changes:
            if verbose:
                print(f"Changes detected: {', '.join(str(f) for f in colight_changes)}")

            try:
                if input_path.is_file():
                    if input_path in colight_changes:
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
                    for changed_file in colight_changes:
                        if changed_file.is_relative_to(input_path):
                            rel_path = changed_file.relative_to(input_path)
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
