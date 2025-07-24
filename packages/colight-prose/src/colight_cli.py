"""CLI interface for colight-prose."""

import asyncio
import pathlib
from typing import Optional

import click

import colight_prose.static.builder as builder
from colight_prose.constants import DEFAULT_INLINE_THRESHOLD
from colight_prose.server import LiveServer
from colight_prose.static import watcher


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
    "--formats",
    "-f",
    type=str,
    default="markdown",
    help="Comma-separated output formats (e.g., 'markdown,html')",
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
    formats: str,
    **kwargs,
):
    """Build a .py file into markdown/HTML."""

    if input_path.is_file():
        # Single file
        if not output:
            # Default to current directory
            output = pathlib.Path(".")

        # Pass formats as raw string to build_file
        try:
            if output.suffix:
                # Output is a file
                builder.build_file(input_path, output, formats=formats, **kwargs)
            else:
                # Output is a directory
                builder.build_file(
                    input_path, output_dir=output, formats=formats, **kwargs
                )
        except ValueError as e:
            click.echo(f"Error: {e}")
            return

        if kwargs.get("verbose"):
            if output.suffix:
                click.echo(f"Built {input_path} -> {output}")
            else:
                click.echo(f"Built {input_path} -> {output}/")
    else:
        # Directory
        if not output:
            output = pathlib.Path("build")
        try:
            builder.build_directory(input_path, output, formats=formats, **kwargs)
        except ValueError as e:
            click.echo(f"Error: {e}")
            return
        if kwargs.get("verbose"):
            click.echo(f"Built {input_path}/ -> {output}/")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Output directory (default: .colight_cache with dev server, build without)",
)
@click.option(
    "--verbose", "-v", type=bool, default=False, help="Verbose output (default: False)"
)
@click.option(
    "--formats",
    "-f",
    type=str,
    default="markdown",
    help="Comma-separated output formats (ignored when dev server is enabled)",
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
    "--dev-server",
    type=bool,
    default=True,
    help="Run development server with live reload (default: True)",
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
    help="Don't open browser on start (only with dev server)",
)
def watch(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    formats: str,
    include: tuple,
    ignore: tuple,
    dev_server: bool,
    host: str,
    port: int,
    no_open: bool,
    **kwargs,
):
    """Watch for changes and rebuild automatically, optionally with dev server."""

    if dev_server:
        # Default output to .colight_cache if not specified
        if not output:
            output = pathlib.Path(".colight_cache")

        click.echo(f"Watching {input_path} for changes...")
        click.echo(f"Output: {output}")
        click.echo(f"Server: http://{host}:{port}")

        # Ensure HTML is included in formats for serving
        if "html" not in formats:
            formats_with_html = f"{formats},html" if formats else "html"
        else:
            formats_with_html = formats

        watcher.watch_build_and_serve(
            input_path,
            output,
            formats=formats_with_html,  # Include HTML for serving
            include=list(include) if include else None,
            ignore=list(ignore) if ignore else None,
            host=host,
            http_port=port,
            ws_port=port + 1,  # WebSocket port is HTTP port + 1
            open_url=not no_open,
            **kwargs,
        )
    else:
        # Default output to build if not specified
        if not output:
            output = pathlib.Path("build")

        click.echo(f"Watching {input_path} for changes...")
        click.echo(f"Output: {output}")

        watcher.watch_and_build(
            input_path,
            output,
            formats=formats,  # Pass the formats string directly, builder will parse it
            include=list(include) if include else None,
            ignore=list(ignore) if ignore else None,
            **kwargs,
        )


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--verbose", "-v", type=bool, default=False, help="Verbose output (default: False)"
)
@click.option(
    "--pragma",
    type=str,
    help="Comma-separated pragma tags (e.g., 'hide-statements,hide-visuals')",
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
    "-p",
    type=int,
    default=5500,
    help="Port for the HTTP server (default: 5500)",
)
@click.option(
    "--no-open",
    is_flag=True,
    help="Don't open browser on start",
)
def live(
    input_path: pathlib.Path,
    verbose: bool,
    pragma: Optional[str],
    include: tuple,
    ignore: tuple,
    host: str,
    port: int,
    no_open: bool,
):
    """Start LiveServer for on-demand building and serving."""

    click.echo(f"Starting LiveServer for {input_path}")
    click.echo(f"Server: http://{host}:{port}")

    server = LiveServer(
        input_path,
        verbose=verbose,
        pragma=pragma,
        include=list(include) if include else ["*.py"],
        ignore=list(ignore) if ignore else None,
        host=host,
        http_port=port,
        ws_port=port + 1,  # WebSocket port is HTTP port + 1
        open_url=not no_open,
    )

    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        click.echo("\nStopping LiveServer...")
        server.stop()


if __name__ == "__main__":
    main()
