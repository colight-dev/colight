"""Public API for colight-site - for use by plugins and external tools."""

import pathlib
from typing import Optional, List, Union
from dataclasses import dataclass

from .parser import parse_colight_file, is_colight_file
from .executor import DocumentExecutor
from .generator import MarkdownGenerator, HTMLGenerator
from .model import Block
from .constants import DEFAULT_INLINE_THRESHOLD
from .pragma import parse_pragma_arg
from . import builder  # For internal use only


@dataclass
class EvaluatedBlock:
    """Result of processing a single block."""

    block: Block
    visual_data: Optional[Union[bytes, pathlib.Path]]  # bytes if inlined, Path if saved
    error: Optional[str] = None


@dataclass
class EvaluatedPython:
    """Result of processing a python file."""

    blocks: List[EvaluatedBlock]
    markdown_content: str
    html_content: Optional[str] = None


def evaluate_python(
    input_path: pathlib.Path,
    *,
    output_dir: Optional[pathlib.Path] = None,
    inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    format: str = "markdown",
    verbose: bool = False,
    pragma: Optional[set[str] | str] = None,
    embed_path_template: Optional[str] = None,
    output_path_template: Optional[str] = None,
) -> EvaluatedPython:
    """
    Process a python file and return the results.

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
        EvaluatedPython with all processed data
    """
    # Parse the file
    document = parse_colight_file(input_path)

    # Parse pragma if provided
    if pragma:
        pragma_tags = parse_pragma_arg(pragma)
        # Merge with document tags
        from .model import TagSet

        document.tags = document.tags | TagSet(frozenset(pragma_tags))

    # Setup defaults
    if output_dir is None:
        output_dir = input_path.parent / (input_path.stem + "_colight")

    if embed_path_template is None:
        embed_path_template = f"{input_path.stem}_colight/block-{{block:03d}}.colight"

    if output_path_template is None:
        output_path_template = "block-{block:03d}.colight"

    # Execute document
    executor = DocumentExecutor(verbose=verbose)
    results, _ = executor.execute(document, str(input_path))

    # Process results into EvaluatedBlocks
    processed_blocks = []

    for i, (block, result) in enumerate(zip(document.blocks, results)):
        if result.error:
            error_msg = f"Block {i} (line {block.start_line}): {result.error.strip()}"
            processed_blocks.append(EvaluatedBlock(block, None, error=error_msg))
            if verbose:
                print(f"  {error_msg}")
        elif result.colight_bytes is None:
            processed_blocks.append(EvaluatedBlock(block, None))
        elif len(result.colight_bytes) < inline_threshold:
            # Keep in memory for inlining
            processed_blocks.append(EvaluatedBlock(block, result.colight_bytes))
            if verbose:
                print(
                    f"  Block {i}: visualization will be inlined ({len(result.colight_bytes)} bytes)"
                )
        else:
            # Save to disk
            output_dir.mkdir(parents=True, exist_ok=True)
            colight_path = output_dir / output_path_template.format(block=i)
            colight_path.write_bytes(result.colight_bytes)
            processed_blocks.append(EvaluatedBlock(block, colight_path))
            if verbose:
                print(f"  Block {i}: saved visualization to {colight_path.name}")

    # Generate output
    generator = MarkdownGenerator(
        output_dir,
        embed_path_template=embed_path_template,
        inline_threshold=inline_threshold,
    )
    # Generate markdown
    path_context = {"basename": input_path.stem}
    markdown_content = generator.generate(document, results, path_context)

    # Generate HTML if requested
    html_content = None
    if format == "html":
        title = input_path.stem.replace(".colight", "").replace("_", " ").title()
        html_generator = HTMLGenerator(
            output_dir,
            embed_path_template=embed_path_template,
            inline_threshold=inline_threshold,
        )
        html_content = html_generator.generate(document, results, title, path_context)

    return EvaluatedPython(
        blocks=processed_blocks,
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
    "evaluate_python",
    "is_colight_file",
    "EvaluatedBlock",
    "EvaluatedPython",
    # CLI convenience functions
    "build_file",
    "build_directory",
    "init_project",
    "get_output_path",
]
