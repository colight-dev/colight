"""CLI interface for colight-site."""

import click
import pathlib
from typing import Optional

from . import api
from . import watcher
from .constants import DEFAULT_INLINE_THRESHOLD


@click.group()
@click.version_option()
def main():
    """Static site generator for Colight visualizations."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Output file or directory",
)
@click.option(
    "--verbose", "-v", type=bool, default=False, help="Verbose output (default: False)"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "html"]),
    default="markdown",
    help="Output format",
)
@click.option(
    "--hide-statements",
    is_flag=True,
    help="Hide statements, only show expressions",
)
@click.option(
    "--hide-visuals",
    is_flag=True,
    help="Hide visuals, only show code",
)
@click.option(
    "--hide-code",
    is_flag=True,
    help="Hide code blocks",
)
@click.option(
    "--continue-on-error",
    type=bool,
    default=True,
    help="Continue building even if forms fail to execute (default: True)",
)
@click.option(
    "--colight-output-path",
    type=str,
    help="Template for colight file output paths (e.g., './{basename}/form-{form:03d}.colight')",
)
@click.option(
    "--colight-embed-path",
    type=str,
    help="Template for embed src paths in HTML (e.g., 'form-{form:03d}.colight')",
)
@click.option(
    "--inline-threshold",
    type=int,
    default=DEFAULT_INLINE_THRESHOLD,
    help=f"Embed .colight files smaller than this size (in bytes) as script tags (default: {DEFAULT_INLINE_THRESHOLD})",
)
def build(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    verbose: bool,
    format: str,
    hide_statements: bool,
    hide_visuals: bool,
    hide_code: bool,
    continue_on_error: bool,
    colight_output_path: Optional[str],
    colight_embed_path: Optional[str],
    inline_threshold: int,
):
    """Build a .colight.py file into markdown/HTML."""
    # Create options dict
    options = {
        "hide_statements": hide_statements,
        "hide_visuals": hide_visuals,
        "hide_code": hide_code,
        "continue_on_error": continue_on_error,
    }

    if input_path.is_file():
        # Single file
        if not output:
            output = api.get_output_path(input_path, format)
        api.build_file(
            input_path,
            output,
            verbose=verbose,
            format=format,
            colight_output_path=colight_output_path,
            colight_embed_path=colight_embed_path,
            inline_threshold=inline_threshold,
            **options,
        )
        if verbose:
            click.echo(f"Built {input_path} -> {output}")
    else:
        # Directory
        if not output:
            output = pathlib.Path("build")
        api.build_directory(
            input_path,
            output,
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
            click.echo(f"Built {input_path}/ -> {output}/")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Output directory",
    default="build",
)
@click.option(
    "--verbose", "-v", type=bool, default=False, help="Verbose output (default: False)"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "html"]),
    default="markdown",
    help="Output format",
)
@click.option(
    "--hide-statements",
    is_flag=True,
    help="Hide statements, only show expressions",
)
@click.option(
    "--hide-visuals",
    is_flag=True,
    help="Hide visuals, only show code",
)
@click.option(
    "--hide-code",
    is_flag=True,
    help="Hide code blocks",
)
@click.option(
    "--continue-on-error",
    type=bool,
    default=True,
    help="Continue building even if forms fail to execute (default: True)",
)
@click.option(
    "--colight-output-path",
    type=str,
    help="Template for colight file output paths (e.g., './{basename}/form-{form:03d}.colight')",
)
@click.option(
    "--colight-embed-path",
    type=str,
    help="Template for embed src paths in HTML (e.g., 'form-{form:03d}.colight')",
)
@click.option(
    "--inline-threshold",
    type=int,
    default=DEFAULT_INLINE_THRESHOLD,
    help=f"Embed .colight files smaller than this size (in bytes) as script tags (default: {DEFAULT_INLINE_THRESHOLD})",
)
def watch(
    input_path: pathlib.Path,
    output: pathlib.Path,
    verbose: bool,
    format: str,
    hide_statements: bool,
    hide_visuals: bool,
    hide_code: bool,
    continue_on_error: bool,
    colight_output_path: Optional[str],
    colight_embed_path: Optional[str],
    inline_threshold: int,
):
    """Watch for changes and rebuild automatically."""
    click.echo(f"Watching {input_path} for changes...")
    click.echo(f"Output: {output}")
    watcher.watch_and_build(
        input_path,
        output,
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


@main.command()
@click.argument("project_dir", type=click.Path(path_type=pathlib.Path))
def init(project_dir: pathlib.Path):
    """Initialize a new colight-site project."""
    api.init_project(project_dir)
    click.echo(f"Initialized project in {project_dir}")


if __name__ == "__main__":
    main()
