"""Public API for colight-site - for use by plugins and external tools."""

import pathlib
from typing import Optional, List, Union
from dataclasses import dataclass

from .parser import parse_colight_file, is_colight_file, Form
from .executor import SafeFormExecutor
from .generator import MarkdownGenerator
from .constants import DEFAULT_INLINE_THRESHOLD
from . import builder  # For internal use only


@dataclass
class ProcessedForm:
    """Result of processing a single form."""

    form: Form
    visualization_data: Optional[
        Union[bytes, pathlib.Path]
    ]  # bytes if inlined, Path if saved
    error: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of processing a .colight.py file."""

    forms: List[ProcessedForm]
    markdown_content: str
    html_content: Optional[str] = None


def process_colight_file(
    input_path: pathlib.Path,
    *,
    output_dir: Optional[pathlib.Path] = None,
    inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    format: str = "markdown",
    verbose: bool = False,
    hide_statements: bool = False,
    hide_visuals: bool = False,
    hide_code: bool = False,
    embed_path_template: Optional[str] = None,
    output_path_template: Optional[str] = None,
) -> ProcessingResult:
    """
    Process a .colight.py file and return the results.

    This is the main public API for processing colight files. It handles:
    - Parsing the file
    - Executing forms
    - Generating visualizations
    - Creating markdown/HTML output

    Args:
        input_path: Path to the .colight.py file
        output_dir: Directory for saving .colight files (if needed)
        inline_threshold: Size threshold for inlining visualizations
        format: Output format ("markdown" or "html")
        verbose: Print verbose output
        hide_statements: Hide statement code blocks
        hide_visuals: Hide visualizations
        hide_code: Hide all code blocks
        embed_path_template: Template for embed paths in output
        output_path_template: Template for saving .colight files

    Returns:
        ProcessingResult with all processed data
    """
    # Parse the file
    forms, file_metadata = parse_colight_file(input_path)

    # Merge metadata with options
    pragma_tags, formats = file_metadata.merge_with_cli_options(
        hide_statements=hide_statements,
        hide_visuals=hide_visuals,
        hide_code=hide_code,
        format=format,
    )

    # Setup defaults
    if output_dir is None:
        output_dir = input_path.parent / (input_path.stem + "_colight")

    if embed_path_template is None:
        embed_path_template = f"{input_path.stem}_colight/form-{{form:03d}}.colight"

    if output_path_template is None:
        output_path_template = "form-{form:03d}.colight"

    # Execute forms
    executor = SafeFormExecutor(verbose=verbose)
    processed_forms = []

    for i, form in enumerate(forms):
        try:
            result = executor.execute_form(form, str(input_path))
            colight_bytes = executor.get_colight_bytes(result)

            if colight_bytes is None:
                processed_forms.append(ProcessedForm(form, None))
            elif len(colight_bytes) < inline_threshold:
                # Keep in memory for inlining
                processed_forms.append(ProcessedForm(form, colight_bytes))
                if verbose:
                    print(
                        f"  Form {i}: visualization will be inlined ({len(colight_bytes)} bytes)"
                    )
            else:
                # Save to disk
                output_dir.mkdir(parents=True, exist_ok=True)
                colight_path = output_dir / output_path_template.format(form=i)
                colight_path.write_bytes(colight_bytes)
                processed_forms.append(ProcessedForm(form, colight_path))
                if verbose:
                    print(f"  Form {i}: saved visualization to {colight_path.name}")

        except Exception as e:
            error_msg = f"Form {i} (line {form.start_line}): {type(e).__name__}: {e}"
            processed_forms.append(ProcessedForm(form, None, error=error_msg))
            if verbose:
                print(f"  {error_msg}")

    # Generate output
    generator = MarkdownGenerator(
        output_dir,
        embed_path_template=embed_path_template,
        inline_threshold=inline_threshold,
    )

    # Extract data for generator
    forms_list = [pf.form for pf in processed_forms]
    colight_data = [pf.visualization_data for pf in processed_forms]
    errors = [pf.error for pf in processed_forms]

    title = input_path.stem.replace(".colight", "").replace("_", " ").title()

    # Generate markdown
    markdown_content = generator.generate_markdown(
        forms_list,
        colight_data,
        title,
        pragma_tags=pragma_tags,
        execution_errors=errors,
    )

    # Generate HTML if requested
    html_content = None
    if format == "html" or "html" in formats:
        html_content = generator.generate_html(
            forms_list,
            colight_data,
            title,
            pragma_tags=pragma_tags,
            execution_errors=errors,
        )

    return ProcessingResult(
        forms=processed_forms,
        markdown_content=markdown_content,
        html_content=html_content,
    )


# Higher-level convenience functions that match CLI usage


def build_file(input_path: pathlib.Path, output_path: pathlib.Path, **kwargs) -> None:
    """Build a single .colight.py file to markdown/HTML. Convenience wrapper for CLI."""
    builder.build_file(input_path, output_path, **kwargs)


def build_directory(
    input_dir: pathlib.Path, output_dir: pathlib.Path, **kwargs
) -> None:
    """Build all .colight.py files in a directory. Convenience wrapper for CLI."""
    builder.build_directory(input_dir, output_dir, **kwargs)


def init_project(project_dir: pathlib.Path) -> None:
    """Initialize a new colight-site project. Convenience wrapper for CLI."""
    builder.init_project(project_dir)


def get_output_path(input_path: pathlib.Path, format: str) -> pathlib.Path:
    """Get the default output path for a given input file."""
    return builder._get_output_path(input_path, format)


# Re-export commonly used functions
__all__ = [
    # Core API
    "process_colight_file",
    "is_colight_file",
    "ProcessedForm",
    "ProcessingResult",
    # CLI convenience functions
    "build_file",
    "build_directory",
    "init_project",
    "get_output_path",
]
