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
def build(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    verbose: bool,
    format: str,
    hide_statements: bool,
    hide_visuals: bool,
    hide_code: bool,
):
    """Build a .colight.py file into markdown/HTML."""
    # Create options dict
    options = {
        "hide_statements": hide_statements,
        "hide_visuals": hide_visuals,
        "hide_code": hide_code,
    }

    if input_path.is_file():
        # Single file
        if not output:
            output = builder._get_output_path(input_path, format)
        builder.build_file(
            input_path, output, verbose=verbose, format=format, **options
        )
        click.echo(f"Built {input_path} -> {output}")
    else:
        # Directory
        if not output:
            output = pathlib.Path("build")
        builder.build_directory(
            input_path, output, verbose=verbose, format=format, **options
        )
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
def watch(
    input_path: pathlib.Path,
    output: pathlib.Path,
    verbose: bool,
    format: str,
    hide_statements: bool,
    hide_visuals: bool,
    hide_code: bool,
):
    """Watch for changes and rebuild automatically."""
    # Create options dict
    options = {
        "hide_statements": hide_statements,
        "hide_visuals": hide_visuals,
        "hide_code": hide_code,
    }

    click.echo(f"Watching {input_path} for changes...")
    click.echo(f"Output: {output}")
    watcher.watch_and_build(
        input_path, output, verbose=verbose, format=format, **options
    )


@main.command()
@click.argument("project_dir", type=click.Path(path_type=pathlib.Path))
def init(project_dir: pathlib.Path):
    """Initialize a new colight-site project."""
    builder.init_project(project_dir)
    click.echo(f"Initialized project in {project_dir}")


if __name__ == "__main__":
    main()
