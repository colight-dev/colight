"""Main builder module that coordinates parsing, execution, and generation."""

import pathlib
from typing import Optional, List
import subprocess
import sys

from .parser import parse_colight_file
from .executor import SafeFormExecutor
from .generator import MarkdownGenerator
from .constants import DEFAULT_INLINE_THRESHOLD
from .pep723 import detect_pep723_metadata, parse_dependencies


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
    continue_on_error: bool = True,
    colight_output_path: Optional[str] = None,
    colight_embed_path: Optional[str] = None,
    inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    in_subprocess: bool = False,
):
    """Build a single Python file."""
    if not input_path.suffix == ".py":
        raise ValueError(f"Not a Python file: {input_path}")

    # Check if this is a PEP 723 file BEFORE doing any processing
    file_content = input_path.read_text(encoding="utf-8")
    pep723_metadata = detect_pep723_metadata(file_content)

    # Skip PEP 723 handling if we're already in a subprocess
    if pep723_metadata and not in_subprocess:
        if verbose:
            print("  Detected PEP 723 metadata - re-running with dependencies")

        # Parse the dependencies from PEP 723 metadata
        dependencies = parse_dependencies(pep723_metadata)

        # Build the uv run command
        # Get the absolute path to colight-site package
        colight_site_root = pathlib.Path(__file__).parent.parent.parent.resolve()
        cmd = [
            "uv",
            "run",
            "--with-editable",
            str(colight_site_root),  # colight-site package root
        ]

        # Add each PEP 723 dependency
        for dep in dependencies:
            cmd.extend(["--with", dep])

        # Add the script flag and colight-site command
        cmd.extend(
            [
                "--",
                sys.executable,
                "-m",
                "colight_site.cli",
                "build",
                str(input_path),
                "-o",
                str(output_path),
            ]
        )

        # Add all the flags
        if verbose:
            cmd.extend(["--verbose", "true"])
        if format != "markdown":
            cmd.extend(["--format", format])
        if hide_statements:
            cmd.append("--hide-statements")
        if hide_visuals:
            cmd.append("--hide-visuals")
        if hide_code:
            cmd.append("--hide-code")
        if not continue_on_error:
            cmd.extend(["--continue-on-error", "false"])
        if colight_output_path:
            cmd.extend(["--colight-output-path", colight_output_path])
        if colight_embed_path:
            cmd.extend(["--colight-embed-path", colight_embed_path])
        if inline_threshold != DEFAULT_INLINE_THRESHOLD:
            cmd.extend(["--inline-threshold", str(inline_threshold)])

        # Add flag to indicate we're in a subprocess
        cmd.append("--in-subprocess")

        if verbose:
            print(f"  Running: {' '.join(cmd)}")

        # Run the command and let output pass through in real-time
        result = subprocess.run(cmd)

        # Return instead of sys.exit to allow tests to continue
        # The subprocess has completed, so we just return to prevent further processing
        return

    # Not a PEP 723 file or already in PEP 723 environment - continue with normal execution
    if verbose:
        print(f"Building {input_path} -> {output_path}")
    try:
        # Parse the file
        forms, file_metadata = parse_colight_file(input_path)
        if verbose:
            print(f"Found {len(forms)} forms")
            if file_metadata.pragma_tags:
                print(f"  File metadata: {file_metadata}")
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
    if verbose:
        print(f"  Writing .colight files to: {colight_dir}")
    executor = SafeFormExecutor(verbose=verbose)

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
    colight_data = []  # Can be None, bytes, or Path
    execution_errors = []  # Track errors for reporting
    for i, form in enumerate(forms):
        try:
            result = executor.execute_form(form, str(input_path))
            colight_bytes = executor.get_colight_bytes(result)

            if colight_bytes is None:
                colight_data.append(None)
            elif len(colight_bytes) < inline_threshold:
                # Small file - keep in memory for inline embedding
                colight_data.append(colight_bytes)
                if verbose:
                    print(
                        f"  Form {i}: visualization will be inlined ({len(colight_bytes)} bytes)"
                    )
            else:
                # Large file - save to disk
                # Format the output path template
                context = {**path_context, "form": i}
                formatted_path = output_template.format(**context)

                # Resolve relative to output_path's parent or colight_dir
                if formatted_path.startswith("./"):
                    # Relative to the markdown file's location
                    base_dir = output_path.parent
                    formatted_path = formatted_path[2:]  # Remove ./
                else:
                    # Use the default colight_dir
                    base_dir = colight_dir

                output_file_path = base_dir / formatted_path
                # Ensure parent directory exists
                output_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Write bytes to file
                output_file_path.write_bytes(colight_bytes)
                colight_data.append(output_file_path)

                if verbose:
                    print(f"  Form {i}: saved visualization to {output_file_path.name}")

            execution_errors.append(None)
        except Exception as e:
            error_msg = f"Form {i} (line {form.start_line}): {type(e).__name__}: {e}"
            if continue_on_error:
                if verbose:
                    print(f"  {error_msg}")
                colight_data.append(None)
                execution_errors.append(error_msg)
            else:
                print(f"Error in {input_path}: {error_msg}")
                raise

    # Generate output
    generator = MarkdownGenerator(
        colight_dir,
        embed_path_template=embed_template,
        inline_threshold=inline_threshold,
    )
    title = input_path.stem.replace(".colight", "").replace("_", " ").title()

    # Merge file metadata with CLI options (CLI takes precedence)
    pragma_tags, formats = file_metadata.merge_with_cli_options(
        hide_statements=hide_statements,
        hide_visuals=hide_visuals,
        hide_code=hide_code,
        format=format,
    )

    # For now, use the first format (single output)
    # TODO: In the future, we could generate multiple formats
    final_format = next(iter(formats))

    if final_format == "html":
        html_content = generator.generate_html(
            forms,
            colight_data,
            title,
            output_path,
            path_context,
            pragma_tags=pragma_tags,
            execution_errors=execution_errors,
        )
        generator.write_html_file(html_content, output_path)
    else:
        markdown_content = generator.generate_markdown(
            forms,
            colight_data,
            title,
            output_path,
            path_context,
            pragma_tags=pragma_tags,
            execution_errors=execution_errors,
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
    continue_on_error: bool = True,
    colight_output_path: Optional[str] = None,
    colight_embed_path: Optional[str] = None,
    inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    include: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
):
    """Build all Python files in a directory matching the patterns."""
    if verbose:
        print(f"Building directory {input_dir} -> {output_dir}")

    # Default to .py files if no include patterns specified
    if include is None:
        include = ["*.py"]
    if ignore is None:
        ignore = []

    # Find all Python files matching patterns
    python_files = []
    for include_pattern in include:
        for path in input_dir.rglob(include_pattern):
            if path.suffix == ".py":
                # Always ignore __pycache__ directories and __init__.py files
                if "__pycache__" in path.parts or path.name == "__init__.py":
                    continue

                # Check if file should be ignored by user patterns
                should_ignore = False
                for ignore_pattern in ignore:
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
                continue_on_error=continue_on_error,
                colight_output_path=colight_output_path,
                colight_embed_path=colight_embed_path,
                inline_threshold=inline_threshold,
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
