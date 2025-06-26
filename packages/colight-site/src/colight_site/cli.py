"""CLI interface for colight-site."""

import click
import pathlib
from typing import Optional

from . import api
from . import watcher
from .constants import DEFAULT_INLINE_THRESHOLD
from .builder import BuildConfig


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
    "--pragma",
    type=str,
    help="Comma-separated pragma tags (e.g., 'hide-statements,hide-visuals')",
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
@click.option(
    "--in-subprocess",
    is_flag=True,
    hidden=True,
    help="Internal flag to indicate we're already in a PEP 723 subprocess",
)
def build(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    verbose: bool,
    format: str,
    pragma: Optional[str],
    continue_on_error: bool,
    colight_output_path: Optional[str],
    colight_embed_path: Optional[str],
    inline_threshold: int,
    in_subprocess: bool,
):
    """Build a .colight.py file into markdown/HTML."""
    # Parse pragma tags from comma-separated string
    pragma_tags = set()
    if pragma:
        pragma_tags = {tag.strip() for tag in pragma.split(",") if tag.strip()}

    # Create BuildConfig from CLI args
    config = BuildConfig(
        verbose=verbose,
        formats={format},
        pragma_tags=pragma_tags,
        continue_on_error=continue_on_error,
        colight_output_path=colight_output_path,
        colight_embed_path=colight_embed_path,
        inline_threshold=inline_threshold,
        in_subprocess=in_subprocess,
    )

    if input_path.is_file():
        # Single file
        if not output:
            output = api.get_output_path(input_path, format)
        api.build_file(input_path, output, config=config)
        if verbose:
            click.echo(f"Built {input_path} -> {output}")
    else:
        # Directory
        if not output:
            output = pathlib.Path("build")
        api.build_directory(input_path, output, config=config)
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
    "--pragma",
    type=str,
    help="Comma-separated pragma tags (e.g., 'hide-statements,hide-visuals')",
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
@click.option(
    "--include",
    type=str,
    multiple=True,
    default=["*.py"],
    help="File patterns to include (default: *.py). Can be specified multiple times.",
)
@click.option(
    "--ignore",
    type=str,
    multiple=True,
    help="File patterns to ignore. Can be specified multiple times.",
)
def watch(
    input_path: pathlib.Path,
    output: pathlib.Path,
    verbose: bool,
    format: str,
    pragma: Optional[str],
    continue_on_error: bool,
    colight_output_path: Optional[str],
    colight_embed_path: Optional[str],
    inline_threshold: int,
    include: tuple,
    ignore: tuple,
):
    """Watch for changes and rebuild automatically."""
    click.echo(f"Watching {input_path} for changes...")
    click.echo(f"Output: {output}")

    # Parse pragma tags from comma-separated string
    pragma_tags = set()
    if pragma:
        pragma_tags = {tag.strip() for tag in pragma.split(",") if tag.strip()}

    # Create BuildConfig from CLI args
    config = BuildConfig(
        verbose=verbose,
        formats={format},
        pragma_tags=pragma_tags,
        continue_on_error=continue_on_error,
        colight_output_path=colight_output_path,
        colight_embed_path=colight_embed_path,
        inline_threshold=inline_threshold,
    )

    watcher.watch_and_build(
        input_path,
        output,
        config=config,
        include=list(include) if include else None,
        ignore=list(ignore) if ignore else None,
    )


@main.command()
@click.argument("project_dir", type=click.Path(path_type=pathlib.Path))
def init(project_dir: pathlib.Path):
    """Initialize a new colight-site project."""
    api.init_project(project_dir)
    click.echo(f"Initialized project in {project_dir}")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Output directory (default: .colight_cache)",
)
@click.option(
    "--verbose", "-v", type=bool, default=False, help="Verbose output (default: False)"
)
@click.option(
    "--pragma",
    type=str,
    help="Comma-separated pragma tags (e.g., 'hide-statements,hide-visuals')",
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
@click.option(
    "--include",
    type=str,
    multiple=True,
    default=["*.py"],
    help="File patterns to include (default: *.py). Can be specified multiple times.",
)
@click.option(
    "--ignore",
    type=str,
    multiple=True,
    help="File patterns to ignore. Can be specified multiple times.",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host for the dev server (default: 127.0.0.1)",
)
@click.option(
    "--port",
    type=int,
    default=5500,
    help="Port for the HTTP server (default: 5500)",
)
@click.option(
    "--no-open",
    is_flag=True,
    help="Don't open browser on start",
)
def watch_serve(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    verbose: bool,
    pragma: Optional[str],
    continue_on_error: bool,
    colight_output_path: Optional[str],
    colight_embed_path: Optional[str],
    inline_threshold: int,
    include: tuple,
    ignore: tuple,
    host: str,
    port: int,
    no_open: bool,
):
    """Watch for changes, rebuild, and serve with live reload."""
    # Default output to .colight_cache if not specified
    if not output:
        output = pathlib.Path(".colight_cache")

    click.echo(f"Watching {input_path} for changes...")
    click.echo(f"Output: {output}")
    click.echo(f"Server: http://{host}:{port}")

    # Parse pragma tags from comma-separated string
    pragma_tags = set()
    if pragma:
        pragma_tags = {tag.strip() for tag in pragma.split(",") if tag.strip()}

    # Create BuildConfig from CLI args
    config = BuildConfig(
        verbose=verbose,
        formats={"html"},  # Always HTML for serving
        pragma_tags=pragma_tags,
        continue_on_error=continue_on_error,
        colight_output_path=colight_output_path,
        colight_embed_path=colight_embed_path,
        inline_threshold=inline_threshold,
    )

    watcher.watch_build_and_serve(
        input_path,
        output,
        config=config,
        include=list(include) if include else None,
        ignore=list(ignore) if ignore else None,
        host=host,
        http_port=port,
        ws_port=port + 1,  # WebSocket port is HTTP port + 1
        open_url=not no_open,
    )


if __name__ == "__main__":
    main()
