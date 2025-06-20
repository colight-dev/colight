"""Main builder module that coordinates parsing, execution, and generation."""

import pathlib
from typing import Optional, List

from .parser import parse_colight_file
from .executor import SafeFormExecutor
from .generator import MarkdownGenerator


def _get_output_path(input_path: pathlib.Path, format: str) -> pathlib.Path:
    """Convert Python file path to output path with correct extension."""
    if input_path.name.endswith(".colight.py"):
        # Remove .colight.py and add the new extension
        base_name = input_path.name[:-11]  # Remove ".colight.py"
        suffix = ".html" if format == "html" else ".md"
        return input_path.parent / (base_name + suffix)
    elif input_path.suffix == ".py":
        # For regular .py files, replace .py with the output extension
        suffix = ".html" if format == "html" else ".md"
        return input_path.with_suffix(suffix)
    else:
        # Fallback for other files
        suffix = ".html" if format == "html" else ".md"
        return input_path.with_suffix(suffix)


def build_file(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    verbose: bool = False,
    format: str = "markdown",
    hide_statements: bool = False,
    hide_visuals: bool = False,
    hide_code: bool = False,
    colight_output_path: Optional[str] = None,
    colight_embed_path: Optional[str] = None,
):
    """Build a single Python file."""
    if not input_path.suffix == ".py":
        raise ValueError(f"Not a Python file: {input_path}")

    if verbose:
        print(f"Building {input_path} -> {output_path}")

    try:
        # Parse the file
        forms, file_metadata = parse_colight_file(input_path)
        if verbose:
            print(f"Found {len(forms)} forms")
            if any(
                [
                    file_metadata.hide_statements,
                    file_metadata.hide_visuals,
                    file_metadata.hide_code,
                    file_metadata.format,
                ]
            ):
                print(f"File metadata: {file_metadata}")
    except Exception as e:
        if verbose:
            print(f"Parse error: {e}")
        # Create a minimal output file with error message
        output_path.parent.mkdir(parents=True, exist_ok=True)
        error_content = f"# Parse Error\n\nCould not parse {input_path.name}: {e}\n"
        output_path.write_text(error_content)
        return

    # Setup execution environment
    # Default templates if not provided
    output_template = (
        colight_output_path or "./{basename}_colight/form-{form:03d}.colight"
    )
    embed_template = colight_embed_path or "{basename}_colight/form-{form:03d}.colight"

    # For backward compatibility, create a directory for executor
    # This will be used as a base directory for relative paths
    colight_dir = output_path.parent / (output_path.stem + "_colight")
    executor = SafeFormExecutor(colight_dir, output_path_template=output_template)

    # Prepare path context for templates
    # Get relative path from build root (assumes we're building from a common root)
    try:
        # Try to get relative path from input's parent's parent (assuming docs/ or src/ structure)
        build_root = input_path.parent.parent
        rel_path = output_path.relative_to(build_root)
    except ValueError:
        # Fallback to just using the output path's parent
        build_root = output_path.parent
        rel_path = output_path.relative_to(output_path.parent)

    path_context = {
        "basename": output_path.stem,
        "filename": output_path.name,
        "reldir": str(rel_path.parent) if str(rel_path.parent) != "." else "",
        "relpath": str(rel_path.with_suffix("")),
        "abspath": str(output_path.absolute()),
        "absdir": str(output_path.parent.absolute()),
    }

    # Execute forms and collect visualizations
    colight_files = []
    for i, form in enumerate(forms):
        try:
            result = executor.execute_form(form, str(input_path))
            colight_file = executor.save_colight_visualization(
                result, i, output_path=output_path, path_context=path_context
            )
            colight_files.append(colight_file)

            if verbose and colight_file:
                print(f"  Form {i}: saved visualization to {colight_file.name}")
        except Exception as e:
            if verbose:
                print(f"  Form {i}: execution failed: {e}")
            colight_files.append(None)

    # Generate output
    generator = MarkdownGenerator(colight_dir, embed_path_template=embed_template)
    title = input_path.stem.replace(".colight", "").replace("_", " ").title()

    # Merge file metadata with CLI options (CLI takes precedence)
    merged_options = file_metadata.merge_with_cli_options(
        hide_statements=hide_statements,
        hide_visuals=hide_visuals,
        hide_code=hide_code,
        format=format,
    )

    # Extract the actual format to use
    final_format = merged_options.pop("format") or format

    if final_format == "html":
        html_content = generator.generate_html(
            forms, colight_files, title, output_path, path_context, **merged_options
        )
        generator.write_html_file(html_content, output_path)
    else:
        markdown_content = generator.generate_markdown(
            forms, colight_files, title, output_path, path_context, **merged_options
        )
        generator.write_markdown_file(markdown_content, output_path)

    if verbose:
        print(f"Generated {output_path}")


def build_directory(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    verbose: bool = False,
    format: str = "markdown",
    hide_statements: bool = False,
    hide_visuals: bool = False,
    hide_code: bool = False,
    colight_output_path: Optional[str] = None,
    colight_embed_path: Optional[str] = None,
    include_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
):
    """Build all Python files in a directory matching the patterns."""
    if verbose:
        print(f"Building directory {input_dir} -> {output_dir}")

    # Default to .colight.py files if no include patterns specified
    if include_patterns is None:
        include_patterns = ["*.colight.py"]
    if ignore_patterns is None:
        ignore_patterns = []

    # Find all Python files matching patterns
    python_files = []
    for include_pattern in include_patterns:
        for path in input_dir.rglob(include_pattern):
            if path.suffix == ".py":
                # Always ignore __pycache__ directories and __init__.py files
                if "__pycache__" in path.parts or path.name == "__init__.py":
                    continue

                # Check if file should be ignored by user patterns
                should_ignore = False
                for ignore_pattern in ignore_patterns:
                    if path.match(ignore_pattern):
                        should_ignore = True
                        break
                if not should_ignore:
                    python_files.append(path)

    # Remove duplicates and sort
    python_files = sorted(set(python_files))

    if verbose:
        print(f"Found {len(python_files)} Python files")

    # Build each file
    for python_file in python_files:
        try:
            # Calculate relative output path
            rel_path = python_file.relative_to(input_dir)
            output_file_rel = _get_output_path(rel_path, format)
            output_file = output_dir / output_file_rel

            build_file(
                python_file,
                output_file,
                verbose=verbose,
                format=format,
                hide_statements=hide_statements,
                hide_visuals=hide_visuals,
                hide_code=hide_code,
                colight_output_path=colight_output_path,
                colight_embed_path=colight_embed_path,
            )
        except Exception as e:
            print(f"Error building {python_file}: {e}")
            if verbose:
                import traceback

                traceback.print_exc()


def init_project(project_dir: pathlib.Path):
    """Initialize a new colight-site project."""
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    (project_dir / "src").mkdir(exist_ok=True)
    (project_dir / "build").mkdir(exist_ok=True)

    # Create example .colight.py file
    example_file = project_dir / "src" / "example.colight.py"
    example_content = """# My First Colight Document
# This is a simple example of mixing narrative text with executable code.

import numpy as np

# Let's create some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# This will create a visualization
x, y  # Return the data for visualization

# You can add more narrative here
# And more code blocks...

print("Hello from colight-site!")
"""

    example_file.write_text(example_content)

    # Create README
    readme_content = """# Colight Site Project

This project uses `colight-site` to generate static documentation with embedded visualizations.

## Usage

Build the site:
```bash
colight-site build src --output build
```

Watch for changes:
```bash
colight-site watch src --output build
```

## Files

- `src/` - Source .colight.py files
- `build/` - Generated markdown and HTML files
"""

    (project_dir / "README.md").write_text(readme_content)
