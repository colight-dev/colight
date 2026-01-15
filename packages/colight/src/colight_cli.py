"""CLI interface for Colight."""

import asyncio
import base64
import pathlib
import subprocess
import tempfile
import webbrowser
from typing import Optional

import click

import colight.env as env
import colight.publish.static.builder as builder
from colight.publish.constants import DEFAULT_INLINE_THRESHOLD
from colight.live_server.server import LiveServer
from colight.publish.static import watcher
from colight.format import parse_file_with_updates
from colight.screenshots import StudioContext


@click.group()
@click.version_option()
def main():
    """Colight CLI for live docs and publishing."""
    pass


def _ensure_html_format(formats: str) -> str:
    if not formats:
        return "html"
    format_set = {fmt.strip() for fmt in formats.split(",") if fmt.strip()}
    if "html" not in format_set:
        format_set.add("html")
    return ",".join(sorted(format_set))


UPDATE_OPS = {"reset", "append", "concat", "setAt"}


def _load_embed_js() -> str:
    embed_path = env.DIST_LOCAL_PATH / "embed.js"
    try:
        return embed_path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"embed.js not found at {embed_path}") from e


def _build_inline_view_html(colight_base64: str, embed_js: str, title: str) -> str:
    safe_embed_js = embed_js.replace("</script>", "<\\/script>")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
</head>
<body>
  <script type="application/x-colight">
{colight_base64}
  </script>
  <script>
{safe_embed_js}
  </script>
</body>
</html>
"""


def _is_update_payload(payload, known_keys: set[str]) -> bool:
    if isinstance(payload, dict):
        if not known_keys:
            return False
        return set(payload.keys()).issubset(known_keys)
    if isinstance(payload, list):
        if not payload:
            return True
        if all(isinstance(item, dict) for item in payload):
            if not known_keys:
                return True
            return all(set(item.keys()).issubset(known_keys) for item in payload)
        if all(
            isinstance(item, list)
            and len(item) >= 2
            and isinstance(item[1], str)
            and item[1] in UPDATE_OPS
            for item in payload
        ):
            return True
    return False


def _extract_update_keys(payload) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        keys.update(payload.keys())
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                keys.update(item.keys())
            elif isinstance(item, list) and item:
                key = item[0]
                if isinstance(key, str):
                    keys.add(key)
    return keys


def _warn_unknown_keys(update_keys: set[str], known_keys: set[str], label: str) -> None:
    if not known_keys or not update_keys:
        return
    unknown = update_keys - known_keys
    if unknown:
        click.echo(
            f"Warning: {label} updates unknown state keys: {', '.join(sorted(unknown))}"
        )


def _collect_colight_entries(
    input_paths: tuple[pathlib.Path, ...],
) -> tuple[dict, list[bytes], list[dict]]:
    base_data = None
    base_buffers: list[bytes] = []
    update_entries: list[dict] = []

    for idx, path in enumerate(input_paths):
        data, buffers, updates = parse_file_with_updates(path)
        if idx == 0:
            if data is None:
                raise ValueError(f"First input must contain initial state: {path}")
            base_data = data
            base_buffers = buffers
        elif data is not None:
            click.echo(f"Warning: ignoring initial state from {path}")
        update_entries.extend(updates)

    assert base_data is not None
    return base_data, base_buffers, update_entries


def _state_updates_from_animate_by(data: dict) -> tuple[list[dict], Optional[int]]:
    animate_by = data.get("animateBy") or []
    if not animate_by:
        raise ValueError("No animateBy metadata found")
    if len(animate_by) > 1:
        raise ValueError(
            f"Multiple animated sliders found ({len(animate_by)}). "
            "Provide explicit updates instead."
        )
    meta = animate_by[0]
    range_val = meta.get("range")
    if isinstance(range_val, int):
        range_val = [0, range_val - 1]
    step = meta.get("step") or 1
    updates = [{meta["key"]: i} for i in range(range_val[0], range_val[1] + 1, step)]
    return updates, meta.get("fps")


def _apply_update_entry(
    studio: StudioContext,
    entry: dict,
    known_keys: set[str],
    label: str,
) -> bool:
    data = entry.get("data") or {}
    buffers = entry.get("buffers") or []
    ast_payload = data.get("ast")
    if ast_payload is not None and not _is_update_payload(ast_payload, known_keys):
        studio.load_plot(data=data, buffers=buffers, measure=False)
        return True

    applied = False
    state_payload = data.get("state") or {}
    if state_payload:
        _warn_unknown_keys(set(state_payload.keys()), known_keys, label)
        studio.apply_updates_json(state_payload, buffers)
        applied = True

    if ast_payload is not None:
        update_keys = _extract_update_keys(ast_payload)
        _warn_unknown_keys(update_keys, known_keys, label)
        studio.apply_updates_json(ast_payload, buffers)
        applied = True

    return applied


def _stream_video_from_updates(
    studio: StudioContext,
    update_entries: list[dict],
    filename: pathlib.Path,
    fps: int,
    known_keys: set[str],
    debug: bool,
) -> None:
    ext = filename.suffix.lower()
    if ext == ".gif":
        ffmpeg_cmd = (
            f"ffmpeg {'-v error' if not debug else ''} -y "
            f"-f image2pipe -vcodec png -framerate {fps} -i - "
            f'-vf "split [a][b];[b]palettegen=stats_mode=diff[p];[a][p]paletteuse=new=1" '
            f'-c:v gif -loop 0 "{filename}"'
        )
    else:
        ffmpeg_cmd = (
            f"ffmpeg {'-v error' if not debug else ''} -y "
            f"-f image2pipe -vcodec png -framerate {fps} -i - "
            f'-an -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow "{filename}"'
        )

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, shell=True)
    frames_written = 0

    try:
        for idx, entry in enumerate(update_entries):
            label = f"update[{idx}]"
            applied = _apply_update_entry(studio, entry, known_keys, label)
            if not applied:
                continue
            frame_bytes = studio.capture_bytes()
            if proc.stdin:
                proc.stdin.write(frame_bytes)
                frames_written += 1

        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    except Exception as e:
        if proc.stdin:
            proc.stdin.close()
        proc.terminate()
        raise e

    if frames_written == 0:
        raise ValueError("No frames captured from update entries")


@main.command()
@click.argument(
    "input_paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=pathlib.Path),
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=click.Path(path_type=pathlib.Path),
    help="Output file (.png, .webp, .pdf, .gif, .mp4)",
)
@click.option(
    "--fps",
    type=int,
    help="Frame rate for video output (default: 24 or from animateBy)",
)
@click.option(
    "--width",
    type=int,
    default=400,
    help="Browser width (default: 400)",
)
@click.option(
    "--height",
    type=int,
    help="Browser height (default: width)",
)
@click.option(
    "--scale",
    type=float,
    default=1.0,
    help="Device scale factor (default: 1.0)",
)
@click.option(
    "--quality",
    type=int,
    default=90,
    help="Image quality for WebP (default: 90)",
)
@click.option(
    "--frame",
    type=int,
    help="Apply updates up to this index (0-based) before rendering",
)
@click.option(
    "--last",
    is_flag=True,
    help="Apply all updates before rendering",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
@click.option(
    "--ready-timeout",
    type=float,
    default=10.0,
    show_default=True,
    help="Max seconds to wait for render readiness (0 to disable)",
)
def render(
    input_paths: tuple[pathlib.Path, ...],
    out: pathlib.Path,
    fps: Optional[int],
    width: int,
    height: Optional[int],
    scale: float,
    quality: int,
    frame: Optional[int],
    last: bool,
    debug: bool,
    ready_timeout: float,
):
    """Render .colight files into images or video."""
    if not input_paths:
        click.echo("Error: provide at least one .colight file")
        return
    if frame is not None and last:
        click.echo("Error: use either --frame or --last, not both")
        return

    try:
        base_data, base_buffers, update_entries = _collect_colight_entries(input_paths)
    except ValueError as e:
        click.echo(f"Error: {e}")
        return

    output_path = pathlib.Path(out)
    ext = output_path.suffix.lower()

    if ext in {".png", ".webp"}:
        mode = "image"
    elif ext == ".pdf":
        mode = "pdf"
    elif ext in {".gif", ".mp4"}:
        mode = "video"
    else:
        click.echo(f"Error: unsupported output extension: {ext}")
        return

    known_keys = set((base_data.get("state") or {}).keys())

    effective_timeout = None if ready_timeout <= 0 else ready_timeout
    with StudioContext(
        plot=None,
        width=width,
        height=height,
        scale=scale,
        debug=debug,
        ready_timeout=effective_timeout,
        reuse=True,
        keep_alive=1.0,
    ) as studio:
        studio.load_plot(data=base_data, buffers=base_buffers)

        if mode == "video":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if update_entries:
                fps_val = fps or 24
                try:
                    _stream_video_from_updates(
                        studio,
                        update_entries,
                        output_path,
                        fps_val,
                        known_keys,
                        debug,
                    )
                except ValueError as e:
                    click.echo(f"Error: {e}")
            else:
                try:
                    state_updates, auto_fps = _state_updates_from_animate_by(base_data)
                except ValueError as e:
                    click.echo(f"Error: {e}")
                    return
                fps_val = fps or auto_fps or 24
                studio.capture_video(state_updates, output_path, fps_val)
            return

        if update_entries:
            if frame is not None:
                if frame < 0 or frame >= len(update_entries):
                    click.echo("Error: --frame out of range for updates")
                    return
                selected_entries = update_entries[: frame + 1]
            elif last:
                selected_entries = update_entries
            else:
                selected_entries = []

            for idx, entry in enumerate(selected_entries):
                label = f"update[{idx}]"
                _apply_update_entry(studio, entry, known_keys, label)
        elif frame is not None or last:
            try:
                state_updates, _ = _state_updates_from_animate_by(base_data)
            except ValueError as e:
                click.echo(f"Error: {e}")
                return
            if frame is not None:
                if frame < 0 or frame >= len(state_updates):
                    click.echo("Error: --frame out of range for animateBy")
                    return
                selected_updates = state_updates[: frame + 1]
            else:
                selected_updates = state_updates
            for update in selected_updates:
                studio.update_state([update])

        if mode == "image":
            studio.save_image(output_path, quality=quality)
        else:
            studio.save_pdf(output_path)


def _publish_impl(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    formats: str,
    watch: bool,
    serve: bool,
    include: Optional[tuple],
    ignore: Optional[tuple],
    host: str,
    port: int,
    no_open: bool,
    **kwargs,
):
    if serve:
        watch = True

    if watch:
        if serve:
            if not output:
                output = pathlib.Path(".colight_cache")

            click.echo(f"Watching {input_path} for changes...")
            click.echo(f"Output: {output}")
            click.echo(f"Server: http://{host}:{port}")

            formats_with_html = _ensure_html_format(formats)

            watcher.watch_build_and_serve(
                input_path,
                output,
                formats=formats_with_html,
                include=list(include) if include else None,
                ignore=list(ignore) if ignore else None,
                host=host,
                http_port=port,
                ws_port=port + 1,
                open_url=not no_open,
                **kwargs,
            )
        else:
            if not output:
                output = pathlib.Path("build")

            click.echo(f"Watching {input_path} for changes...")
            click.echo(f"Output: {output}")

            watcher.watch_and_build(
                input_path,
                output,
                formats=formats,
                include=list(include) if include else None,
                ignore=list(ignore) if ignore else None,
                **kwargs,
            )
        return

    if input_path.is_file():
        if not output:
            output = pathlib.Path(".")

        try:
            if output.suffix:
                builder.build_file(input_path, output, formats=formats, **kwargs)
            else:
                builder.build_file(
                    input_path, output_dir=output, formats=formats, **kwargs
                )
        except ValueError as e:
            click.echo(f"Error: {e}")
            return

        if kwargs.get("verbose"):
            if output.suffix:
                click.echo(f"Published {input_path} -> {output}")
            else:
                click.echo(f"Published {input_path} -> {output}/")
    else:
        if not output:
            output = pathlib.Path("build")
        try:
            builder.build_directory(input_path, output, formats=formats, **kwargs)
        except ValueError as e:
            click.echo(f"Error: {e}")
            return
        if kwargs.get("verbose"):
            click.echo(f"Published {input_path}/ -> {output}/")


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Output file or directory (default: . for files, build/ for dirs, .colight_cache when serving)",
)
@click.option(
    "--verbose", "-v", type=bool, default=False, help="Verbose output (default: False)"
)
@click.option(
    "--format",
    "--formats",
    "-f",
    type=str,
    default="markdown",
    help="Comma-separated output formats (e.g., 'markdown,html')",
)
@click.option(
    "--watch",
    is_flag=True,
    help="Watch files and rebuild on changes",
)
@click.option(
    "--serve",
    is_flag=True,
    help="Serve with live reload (implies --watch)",
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
    help="Don't open browser on start (only with --serve)",
)
@click.option(
    "--in-subprocess",
    is_flag=True,
    hidden=True,
    help="Internal flag to indicate we're already in a PEP 723 subprocess",
)
def publish(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    formats: str,
    watch: bool,
    serve: bool,
    include: tuple,
    ignore: tuple,
    host: str,
    port: int,
    no_open: bool,
    **kwargs,
):
    """Publish a .py file or directory into markdown/HTML, optionally watching/serving."""
    _publish_impl(
        input_path,
        output,
        formats,
        watch,
        serve,
        include,
        ignore,
        host,
        port,
        no_open,
        **kwargs,
    )


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Output HTML file (default: temp file)",
)
@click.option(
    "--no-open",
    is_flag=True,
    help="Don't open browser after creating the HTML file",
)
def view(
    input_path: pathlib.Path,
    output: Optional[pathlib.Path],
    no_open: bool,
):
    """Open a .colight file in a browser using inline HTML."""
    try:
        colight_bytes = input_path.read_bytes()
    except OSError as e:
        click.echo(f"Error: {e}")
        return

    try:
        embed_js = _load_embed_js()
    except FileNotFoundError as e:
        click.echo(f"Error: {e}")
        return

    colight_base64 = base64.b64encode(colight_bytes).decode("ascii")
    title = f"Colight: {input_path.name}"
    html = _build_inline_view_html(colight_base64, embed_js, title)

    if output:
        output_path = pathlib.Path(output)
        try:
            output_path.write_text(html, encoding="utf-8")
        except OSError as e:
            click.echo(f"Error: {e}")
            return
    else:
        tmp_file = tempfile.NamedTemporaryFile(
            prefix="colight-view-",
            suffix=".html",
            delete=False,
        )
        tmp_file.write(html.encode("utf-8"))
        tmp_file.close()
        output_path = pathlib.Path(tmp_file.name)

    url = output_path.as_uri()
    click.echo(f"View: {url}")
    if not no_open:
        webbrowser.open(url)


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

    open_path = input_path.name if input_path.is_file() else None

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
        open_path=open_path,
    )

    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        click.echo("\nStopping LiveServer...")
        server.stop()


@main.command("eval")
@click.option(
    "--host",
    default="127.0.0.1",
    help="Host to bind server to (default: 127.0.0.1)",
)
@click.option(
    "--port",
    default=5510,
    type=int,
    help="HTTP port (WebSocket will be port+1, default: 5510)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def eval_server(
    host: str,
    port: int,
    verbose: bool,
):
    """Start eval server for VSCode integration.

    The eval server accepts code snippets via WebSocket and returns
    execution results with visualizations. Designed for integration
    with the Colight VSCode extension.
    """
    print(f"[colight eval] Starting eval server on {host}:{port}", flush=True)
    input_path = pathlib.Path.cwd()

    server = LiveServer(
        input_path,
        verbose=verbose,
        include=["*.py"],
        ignore=None,
        host=host,
        http_port=port,
        ws_port=port + 1,
        open_url=False,
        eval_mode=True,
    )

    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        click.echo("\nStopping eval server...")
        server.stop()


if __name__ == "__main__":
    main()
