"""File watching functionality for colight-site."""

import pathlib
from watchfiles import watch

from . import builder


def watch_and_build(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    verbose: bool = False,
    format: str = "markdown",
):
    """Watch for changes and rebuild automatically."""
    print(f"Watching {input_path} for changes...")

    # Build initially
    if input_path.is_file():
        builder.build_file(input_path, output_path, verbose=verbose, format=format)
    else:
        builder.build_directory(input_path, output_path, verbose=verbose, format=format)

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
                        builder.build_file(
                            input_path, output_path, verbose=verbose, format=format
                        )
                        print(f"Rebuilt {input_path}")
                else:
                    # Rebuild affected files
                    for changed_file in colight_changes:
                        if changed_file.is_relative_to(input_path):
                            rel_path = changed_file.relative_to(input_path)
                            suffix = ".html" if format == "html" else ".md"
                            output_file = output_path / rel_path.with_suffix(suffix)
                            builder.build_file(
                                changed_file,
                                output_file,
                                verbose=verbose,
                                format=format,
                            )
                            print(f"Rebuilt {changed_file}")
            except Exception as e:
                print(f"Error during rebuild: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
