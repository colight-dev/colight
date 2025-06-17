"""CLI interface for colight-site."""

import click
import pathlib
from typing import Optional

from . import builder
from . import watcher


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
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "html"]),
    default="markdown",
    help="Output format",
)
def build(
    input_path: pathlib.Path, output: Optional[pathlib.Path], verbose: bool, format: str
):
    """Build a .colight.py file into markdown/HTML."""
    if input_path.is_file():
        # Single file
        if not output:
            suffix = ".html" if format == "html" else ".md"
            output = input_path.with_suffix(suffix)
        builder.build_file(input_path, output, verbose=verbose, format=format)
        click.echo(f"Built {input_path} -> {output}")
    else:
        # Directory
        if not output:
            output = pathlib.Path("build")
        builder.build_directory(input_path, output, verbose=verbose, format=format)
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
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "html"]),
    default="markdown",
    help="Output format",
)
def watch(input_path: pathlib.Path, output: pathlib.Path, verbose: bool, format: str):
    """Watch for changes and rebuild automatically."""
    click.echo(f"Watching {input_path} for changes...")
    click.echo(f"Output: {output}")
    watcher.watch_and_build(input_path, output, verbose=verbose, format=format)


@main.command()
@click.argument("project_dir", type=click.Path(path_type=pathlib.Path))
def init(project_dir: pathlib.Path):
    """Initialize a new colight-site project."""
    builder.init_project(project_dir)
    click.echo(f"Initialized project in {project_dir}")


if __name__ == "__main__":
    main()
