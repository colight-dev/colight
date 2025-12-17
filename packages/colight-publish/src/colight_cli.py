"""CLI entry point for the Colight publish workflow."""

from __future__ import annotations

import asyncio
import pathlib
from typing import Iterable, Optional

import click

import colight_publish.static.builder as builder
from colight_publish.constants import DEFAULT_INLINE_THRESHOLD
from colight_publish.server import LiveServer
from colight_publish.static import watcher
from colight_publish.static.site_builder import build_static_site

FORMAT_MAP = {"md": "markdown", "html": "html"}


def _clean_kwargs(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


def _build_documents(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    fmt: str,
    include: Iterable[str],
    ignore: Optional[Iterable[str]],
    config_kwargs: dict,
) -> None:
    format_value = FORMAT_MAP[fmt]
    config_kwargs = {**config_kwargs, "formats": format_value}

    if input_path.is_file():
        target = output or pathlib.Path(".")
        if target.suffix:
            builder.build_file(
                input_path,
                output_file=target,
                **config_kwargs,
            )
        else:
            builder.build_file(
                input_path,
                output_dir=target,
                **config_kwargs,
            )
        return

    dest = output or pathlib.Path("build")
    builder.build_directory(
        input_path,
        dest,
        include=list(include),
        ignore=list(ignore) if ignore else None,
        **config_kwargs,
    )


def _watch_documents(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    fmt: str,
    include: Iterable[str],
    ignore: Optional[Iterable[str]],
    config_kwargs: dict,
    host: str,
    port: int,
    no_open: bool,
) -> None:
    format_value = FORMAT_MAP[fmt]
    config_kwargs = {**config_kwargs, "formats": format_value}
    include_list = list(include)
    ignore_list = list(ignore) if ignore else None

    if fmt == "html":
        dest = output or pathlib.Path(".colight_cache")
        watcher.watch_build_and_serve(
            input_path,
            dest,
            include=include_list,
            ignore=ignore_list,
            host=host,
            http_port=port,
            ws_port=port + 1,
            open_url=not no_open,
            **config_kwargs,
        )
    else:
        dest = output or pathlib.Path("build")
        if fmt == "md":
            click.echo(
                "Watching for changes (Markdown is not served automatically).",
                err=True,
            )
        watcher.watch_and_build(
            input_path,
            dest,
            include=include_list,
            ignore=ignore_list,
            **config_kwargs,
        )


def _run_live_server(
    input_path: pathlib.Path,
    include: Iterable[str],
    ignore: Optional[Iterable[str]],
    pragma: Optional[str],
    host: str,
    port: int,
    no_open: bool,
    verbose: bool,
) -> None:
    server = LiveServer(
        input_path,
        verbose=verbose,
        pragma=pragma,
        include=list(include),
        ignore=list(ignore) if ignore else None,
        host=host,
        http_port=port,
        ws_port=port + 1,
        open_url=not no_open,
    )

    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        click.echo("\nStopping LiveServer...")
        server.stop()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--format",
    "format_",
    type=click.Choice(["md", "html", "site"]),
    required=True,
    help="Output format.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Output file or directory (defaults depend on format).",
)
@click.option("--watch", is_flag=True, help="Watch for changes.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option(
    "--pragma",
    type=str,
    help="Comma-separated pragma tags (e.g., 'hide-statements,hide-visuals').",
)
@click.option(
    "--continue-on-error/--stop-on-error",
    default=True,
    help="Continue building even if forms fail to execute (default: continue-on-error).",
)
@click.option(
    "--colight-output-path",
    type=str,
    help="Template for colight file output paths (e.g., './{basename}/form-{form:03d}.colight').",
)
@click.option(
    "--colight-embed-path",
    type=str,
    help="Template for embed src paths in HTML (e.g., 'form-{form:03d}.colight').",
)
@click.option(
    "--inline-threshold",
    type=int,
    default=DEFAULT_INLINE_THRESHOLD,
    show_default=True,
    help="Embed .colight files smaller than this size (in bytes).",
)
@click.option(
    "--include",
    type=str,
    multiple=True,
    default=["*.py", "*.md"],
    help="File patterns to include (defaults: *.py, *.md). Can be provided multiple times.",
)
@click.option(
    "--ignore",
    type=str,
    multiple=True,
    help="File patterns to ignore. Can be provided multiple times.",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Host for dev servers.",
)
@click.option(
    "--port",
    type=int,
    default=5500,
    show_default=True,
    help="Port for HTTP servers.",
)
@click.option(
    "--no-open",
    is_flag=True,
    help="Don't open a browser window when running dev servers.",
)
def publish(
    input_path: pathlib.Path,
    format_: str,
    output: Optional[pathlib.Path],
    watch: bool,
    verbose: bool,
    pragma: Optional[str],
    continue_on_error: bool,
    colight_output_path: Optional[str],
    colight_embed_path: Optional[str],
    inline_threshold: int,
    include: Iterable[str],
    ignore: Optional[Iterable[str]],
    host: str,
    port: int,
    no_open: bool,
):
    """Publish Colight documentation as Markdown, HTML, or a full site."""
    builder_kwargs = _clean_kwargs(
        verbose=verbose,
        pragma=pragma,
        continue_on_error=continue_on_error,
        colight_output_path=colight_output_path,
        colight_embed_path=colight_embed_path,
        inline_threshold=inline_threshold,
    )

    include_patterns = list(include) if include else ["*.py", "*.md"]
    ignore_patterns = list(ignore) if ignore else None

    if format_ == "site":
        if colight_output_path or colight_embed_path:
            click.echo(
                "Warning: colight output/embed options are ignored for site format.",
                err=True,
            )
        if watch:
            if output:
                click.echo(
                    "Ignoring --output; LiveServer manages its own cache.", err=True
                )
            _run_live_server(
                input_path,
                include_patterns,
                ignore_patterns,
                pragma,
                host,
                port,
                no_open,
                verbose,
            )
        else:
            site_output = output or pathlib.Path("site-build")
            build_static_site(
                input_path,
                site_output,
                include=include_patterns,
                ignore=ignore_patterns,
                pragma=pragma,
                inline_threshold=inline_threshold,
                verbose=verbose,
            )
            click.echo(f"Site built at {site_output}")
        return

    if watch:
        _watch_documents(
            input_path,
            output,
            format_,
            include_patterns,
            ignore_patterns,
            builder_kwargs,
            host,
            port,
            no_open,
        )
        return

    _build_documents(
        input_path,
        output,
        format_,
        include_patterns,
        ignore_patterns,
        builder_kwargs,
    )
    click.echo("Publish complete.")


if __name__ == "__main__":
    publish()
