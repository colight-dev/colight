"""CLI interface for Colight."""

import asyncio
import base64
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import webbrowser
from typing import Any, Dict, List, Optional

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


def _echo_json(payload: Dict[str, Any]) -> None:
    click.echo(json.dumps(payload, indent=1))


def _format_ids(ids: List[str]) -> str:
    return ",".join(ids) if ids else "-"


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def blocks(input_path: pathlib.Path, as_json: bool):
    """Dump the block graph of a notebook-style .py FILE.

    Blocks are the blank-line-separated units of a colight file. Each block
    gets a stable id (short hash of its own source, suffixed -N for
    duplicates) that survives edits to other blocks.

    \b
    JSON schema:
      {"file": str, "pragma": [str], "blocks": [
        {"id": str, "cache_key": str, "lines": [start, end],
         "kind": "prose"|"code"|"both",
         "provides": [str], "requires": [str],
         "depends_on": [block id], "dependents": [block id],
         "pragma": [str], "ends_with_expression": bool}]}
    """
    from colight.cli_tools import blocks as blocks_mod

    payload = blocks_mod.describe_file(input_path)
    if as_json:
        _echo_json(payload)
        return

    click.echo(f"{payload['file']} — {len(payload['blocks'])} blocks")
    if payload["pragma"]:
        click.echo(f"file pragma: {' '.join(payload['pragma'])}")
    header = f"{'ID':<15} {'LINES':<9} {'KIND':<6} {'PROVIDES':<24} {'REQUIRES':<24} {'DEPS':<20} TAGS"
    click.echo(header)
    for block in payload["blocks"]:
        lines = f"{block['lines'][0]}-{block['lines'][1]}"
        click.echo(
            f"{block['id']:<15} {lines:<9} {block['kind']:<6} "
            f"{_format_ids(block['provides']):<24} "
            f"{_format_ids(block['requires']):<24} "
            f"{_format_ids(block['depends_on']):<20} "
            f"{' '.join(block['pragma']) or '-'}"
        )


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--block",
    "focus_block",
    type=str,
    help="Restrict detail to this block id and its dependents "
    "(other blocks get one-line statuses).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-execute every block, ignoring the previous record's cache keys.",
)
def run(
    input_path: pathlib.Path, as_json: bool, focus_block: Optional[str], force: bool
):
    """Headlessly execute FILE.py and diff against the previous invocation.

    A compact fingerprint record is persisted per file (in
    .colight_cache/cli-run/ under the project root) so consecutive runs
    report what changed. Blocks whose transitive cache key (own source +
    upstream sources + local import mtimes) is unchanged are skipped and
    reported as cached; they re-execute only when an executed downstream
    block needs their symbols. Blocks tagged `# | pragma: always-eval`
    always re-execute; --force re-executes everything (statuses then
    compare against the stored fingerprints).

    Exit code is nonzero if any block errored.

    \b
    Statuses: cached | ran:unchanged | ran:changed | new | removed | error
    ("ran:changed" = result fingerprint differs from the previous run).

    \b
    JSON schema:
      {"file": str, "ok": bool, "errors": int, "blocks": [
        {"id": str, "lines": [start, end], "status": str, "executed": bool,
         "duration_ms": int?, "summary": {...}?, "stdout": str?,
         "error": {"type": str, "message": str,
                   "frames": [{"file", "line", "in", "code"}]}?}]}
    Summaries are token-frugal: arrays report dtype/shape/min/max, visuals
    report component types + counts, scalars report truncated reprs.
    """
    from colight.cli_tools import run as run_mod

    try:
        payload = run_mod.run_file(input_path, focus_block=focus_block, force=force)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if as_json:
        _echo_json(payload)
    else:
        click.echo(f"{payload['file']}")
        for block in payload["blocks"]:
            lines = block.get("lines")
            lines_str = f"{lines[0]}-{lines[1]}" if lines else "-"
            summary = block.get("summary") or {}
            if block["status"] == "error":
                error = block.get("error") or {}
                detail = f"{error.get('type')}: {error.get('message')}"
            elif summary.get("kind") == "visual":
                parts = [
                    f"{c['path']}×{c['count']}" for c in summary.get("components", [])
                ]
                detail = (
                    f"visual [{', '.join(parts)}]"
                    if parts
                    else f"visual ({summary.get('size', 0)} bytes)"
                )
            elif summary.get("kind") == "array":
                detail = f"array {summary.get('dtype')} {summary.get('shape')}"
            elif "repr" in summary:
                detail = summary["repr"]
            else:
                detail = summary.get("kind", "")
            duration = f" [{block['duration_ms']}ms]" if "duration_ms" in block else ""
            click.echo(
                f"{block['status']:<14} {block['id']:<15} {lines_str:<9} "
                f"{detail}{duration}"
            )
        if not payload["ok"]:
            click.echo(f"{payload['errors']} block(s) errored", err=True)

    if not payload["ok"]:
        sys.exit(1)


@main.command()
@click.argument("target", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def inspect(target: pathlib.Path, as_json: bool):
    """Inspect the structure of a visual without rendering it.

    TARGET is a .colight artifact (parsed directly) or a .py file (evaluated
    headlessly; every produced visual is inspected). Reports component
    structure, per-array schema (dtype/shape/min/max), state keys and
    callbacks, plus sanity warnings: empty arrays, NaN/Inf values, all
    alphas ~0, degenerate (zero-extent) bounds, and mismatched per-instance
    attribute lengths.

    \b
    JSON schema (.colight):
      {"file": str, "kind": "colight", "updates": int,
       "visual": VISUAL, "warnings": [WARNING]}
    JSON schema (.py):
      {"file": str, "kind": "py", "visuals": [
        {"block": str, "lines": [start, end],
         "visual": VISUAL, "warnings": [WARNING]}],
       "errors": [{"block", "lines", "error"}]?}
    VISUAL = {"components": [{"path", "count", "instances"}],
              "arrays": [{"path", "dtype", "shape", "min"?, "max"?,
                          "nan"?, "inf"?}],
              "legends"?: [{"component", "label"?, "cmap", "domain"?,
                            "categorical", "categories"?}],
              "state_keys": [str], "synced_keys": [str],
              "listeners": [str], "py_listeners": [str],
              "buffers": {"count": int, "total_bytes": int}}
    WARNING = {"code": str, "path": str, "message": str}

    ``legends`` reports what colormap-driven colors encode (components built
    with ``color_by``), e.g. "colors encode Cu % over [0, 2.5] via viridis".
    """
    from colight.cli_tools import inspect_tools

    try:
        payload = inspect_tools.inspect_target(target)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if as_json:
        _echo_json(payload)
        return

    def echo_visual(visual: Dict[str, Any], warnings: List[Dict[str, str]]) -> None:
        for component in visual["components"]:
            instances = (
                f" ({component['instances']} instances)"
                if component.get("instances") is not None
                else ""
            )
            click.echo(
                f"  component {component['path']}×{component['count']}{instances}"
            )
        for array in visual["arrays"]:
            stats = ""
            if "min" in array:
                stats = f" min={array['min']:.4g} max={array['max']:.4g}"
            click.echo(
                f"  array {array['path']}: {array['dtype']} {array['shape']}{stats}"
            )
        for legend in visual.get("legends", []):
            parts = [f"legend {legend['cmap']}"]
            if legend.get("domain") is not None:
                parts.append(f"domain {legend['domain']}")
            if legend.get("categories") is not None:
                parts.append(f"{len(legend['categories'])} categories")
            if legend.get("label"):
                parts.append(f"encodes {legend['label']!r}")
            click.echo(f"  {' '.join(parts)} on {legend['component']}")
        if visual["state_keys"]:
            click.echo(f"  state keys: {len(visual['state_keys'])}")
        if visual["listeners"] or visual["py_listeners"]:
            click.echo(
                f"  callbacks: listeners={visual['listeners']} "
                f"py_listeners={visual['py_listeners']}"
            )
        click.echo(
            f"  buffers: {visual['buffers']['count']} "
            f"({visual['buffers']['total_bytes']} bytes)"
        )
        for warning in warnings:
            click.echo(
                f"  WARNING [{warning['code']}] {warning['path']}: {warning['message']}"
            )

    click.echo(payload["file"])
    if payload["kind"] == "colight":
        if payload["updates"]:
            click.echo(f"  update entries: {payload['updates']}")
        echo_visual(payload["visual"], payload["warnings"])
    else:
        for item in payload["visuals"]:
            click.echo(
                f"block {item['block']} (lines {item['lines'][0]}-{item['lines'][1]}):"
            )
            echo_visual(item["visual"], item["warnings"])
        if not payload["visuals"]:
            click.echo("  no visuals produced")
        for error_item in payload.get("errors", []):
            error = error_item["error"]
            click.echo(
                f"block {error_item['block']} errored: "
                f"{error.get('type')}: {error.get('message')}",
                err=True,
            )


@main.command()
@click.argument("target_a", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("target_b", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--epsilon",
    type=float,
    default=1e-9,
    show_default=True,
    help="Elementwise threshold below which numeric changes count as identical.",
)
def diff(target_a: pathlib.Path, target_b: pathlib.Path, as_json: bool, epsilon: float):
    """Semantic diff of two visuals — change magnitude without rendering.

    TARGET_A and TARGET_B are each a .colight artifact or a .py file
    (evaluated headlessly; visuals are paired by position). Reports
    components added/removed/type-changed, per-array shape/dtype changes and
    magnitude stats (max/mean |delta|, fraction changed beyond epsilon,
    bounds drift), scalar value changes, state-key changes, buffer deltas
    and warnings introduced/resolved. Numeric leaf changes in nested JSON
    lists that differ only in list indices are aggregated into one
    array-style entry with a wildcarded path (e.g. "...args[1][*][0]") and
    a leaves changed/total count; itemized value lists are capped.

    Exit code: 0 identical (within epsilon), 1 differences found, 2 error.

    \b
    JSON schema:
      {"a": TARGET, "b": TARGET, "epsilon": float, "identical": bool,
       "pairs": [{"index": int, "a_block"?: str, "b_block"?: str,
         "identical": bool,
         "components": {"added": [{"path", "type"}], "removed": [...],
                        "changed": [{"path", "from", "to"}]},
         "arrays": {"added": [{"path", "dtype", "shape"}], "removed": [...],
           "changed": [{"path", "dtype"?: [a, b], "shape"?: [a, b],
                        "max_abs_delta"?, "mean_abs_delta"?,
                        "changed_fraction"?, "nan_mismatch"?,
                        "leaves"?: {"changed": int, "total": int},
                        "bounds"?: {"from": [min, max], "to": [min, max]}}]},
         "values": {"added": [path], "removed": [path],
                    "changed": [{"path", "from", "to"}],
                    "truncated"?: {"added"?, "removed"?, "changed"?: int}},
         "state": {"added": [key], "removed": [key], "changed": [key]},
         "buffers": {"count": [a, b], "total_bytes": [a, b]},
         "warnings": {"introduced": [WARNING], "resolved": [WARNING]}}],
       "unpaired"?: {"a": [...], "b": [...]},
       "summary": {"arrays_changed": int, "max_abs_delta"?: float,
                   "max_abs_delta_path"?: str}}
      TARGET = {"file": str, "kind": "colight"|"py", "visuals": int,
                "updates"?: int, "errors"?: [...]}
    """
    from colight.cli_tools import diff_tools

    try:
        payload = diff_tools.diff_targets(target_a, target_b, epsilon=epsilon)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if as_json:
        _echo_json(payload)
    else:
        click.echo(f"A: {payload['a']['file']} ({payload['a']['visuals']} visual(s))")
        click.echo(f"B: {payload['b']['file']} ({payload['b']['visuals']} visual(s))")
        for pair in payload["pairs"]:
            if pair["identical"]:
                continue
            label = f"pair {pair['index']}"
            if "a_block" in pair or "b_block" in pair:
                label += f" [{pair.get('a_block', '?')} -> {pair.get('b_block', '?')}]"
            click.echo(f"{label}:")
            components = pair["components"]
            for item in components["changed"]:
                click.echo(
                    f"  component {item['path']}: {item['from']} -> {item['to']}"
                )
            for item in components["removed"]:
                click.echo(f"  component removed {item['path']}")
            for item in components["added"]:
                click.echo(f"  component added {item['path']}")
            for item in pair["arrays"]["changed"]:
                detail_parts = []
                if "dtype" in item:
                    detail_parts.append(f"dtype {item['dtype'][0]}->{item['dtype'][1]}")
                if "shape" in item:
                    detail_parts.append(f"shape {item['shape'][0]}->{item['shape'][1]}")
                if "max_abs_delta" in item:
                    stats = (
                        f"max |Δ| {item['max_abs_delta']:.4g} "
                        f"mean {item['mean_abs_delta']:.4g} "
                        f"changed {item['changed_fraction']:.1%}"
                    )
                    if "leaves" in item:
                        stats += (
                            f" ({item['leaves']['changed']}/"
                            f"{item['leaves']['total']} leaves)"
                        )
                    detail_parts.append(stats)
                if "bounds" in item:
                    detail_parts.append(
                        f"bounds {item['bounds']['from']} -> {item['bounds']['to']}"
                    )
                click.echo(f"  array {item['path']}: {'; '.join(detail_parts)}")
            for item in pair["arrays"]["removed"]:
                click.echo(f"  array removed {item['path']}")
            for item in pair["arrays"]["added"]:
                click.echo(f"  array added {item['path']}")
            changed_values = pair["values"]["changed"]
            hidden_values = pair["values"].get("truncated", {}).get("changed", 0)
            for item in changed_values[:5]:
                click.echo(f"  value {item['path']}: {item['from']} -> {item['to']}")
            hidden_values += max(0, len(changed_values) - 5)
            if hidden_values:
                click.echo(f"  … {hidden_values} more value change(s)")
            state = pair["state"]
            for kind in ("added", "removed", "changed"):
                if state[kind]:
                    click.echo(f"  state {kind}: {', '.join(state[kind])}")
            counts = pair["buffers"]["count"]
            total = pair["buffers"]["total_bytes"]
            if counts[0] != counts[1] or total[0] != total[1]:
                click.echo(
                    f"  buffers: {counts[0]} -> {counts[1]} "
                    f"({total[0]} -> {total[1]} bytes)"
                )
            for warning in pair["warnings"]["introduced"]:
                click.echo(
                    f"  warning introduced [{warning['code']}] {warning['path']}: "
                    f"{warning['message']}"
                )
            for warning in pair["warnings"]["resolved"]:
                click.echo(f"  warning resolved [{warning['code']}] {warning['path']}")
        unpaired = payload.get("unpaired")
        if unpaired:
            for side, items in unpaired.items():
                for item in items:
                    click.echo(f"  visual only in {side.upper()}: {item}")
        for side in ("a", "b"):
            for error_item in payload[side].get("errors", []):
                error = error_item["error"]
                click.echo(
                    f"  error in {side.upper()} block {error_item['block']}: "
                    f"{error.get('type')}: {error.get('message')}",
                    err=True,
                )
        click.echo(diff_tools.verdict_line(payload))

    sys.exit(0 if payload["identical"] else 1)


@main.command()
@click.argument("target", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--out",
    "-o",
    required=True,
    type=click.Path(path_type=pathlib.Path),
    help="Output PNG path.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--block",
    type=str,
    help="Stable block id to screenshot (.py targets; default = last visual).",
)
@click.option(
    "--width",
    type=int,
    default=800,
    show_default=True,
    help="Viewport width in CSS pixels.",
)
@click.option(
    "--height",
    type=int,
    help="Viewport height in CSS pixels (default: measure rendered content).",
)
@click.option(
    "--dpr",
    type=float,
    default=1.0,
    show_default=True,
    help="Device pixel ratio (output pixels = CSS pixels * dpr).",
)
@click.option(
    "--check",
    is_flag=True,
    help="Render twice and byte-compare; report determinism (exit 1 on mismatch).",
)
@click.option(
    "--frame",
    type=str,
    help='Fit the camera on a scene3d selection "C[:A-B]" (component index '
    "or type name, optional inclusive instance ranges) before capture.",
)
@click.option(
    "--rulers",
    is_flag=True,
    help="Compose labeled coordinate rulers (page CSS px, the pick-at "
    "space) around the capture. Post-capture: the render is untouched.",
)
@click.option(
    "--views",
    type=str,
    help="Comma-separated camera presets (front,back,left,right,side,top,"
    "bottom,iso) composed into one labeled contact sheet. Scene3d only.",
)
@click.option(
    "--max-edge",
    type=int,
    help="Scale the viewport so the output PNG's long edge is exactly N "
    "pixels (preserves aspect; avoids double resampling for agents).",
)
@click.option(
    "--ready-timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Max seconds to wait for render readiness (0 to disable).",
)
@click.option(
    "--no-daemon",
    is_flag=True,
    help="Bypass a running colight daemon and render directly.",
)
@click.option("--debug", is_flag=True, help="Enable renderer debug logging.")
def screenshot(
    target: pathlib.Path,
    out: pathlib.Path,
    as_json: bool,
    block: Optional[str],
    width: int,
    height: Optional[int],
    dpr: float,
    check: bool,
    frame: Optional[str],
    rulers: bool,
    views: Optional[str],
    max_edge: Optional[int],
    ready_timeout: float,
    no_daemon: bool,
    debug: bool,
):
    """Render TARGET to a PNG with deterministic settings.

    TARGET is a .colight artifact or a .py file (evaluated headlessly;
    --block selects one visual by stable id, default = last visual in the
    file). Uses the same headless-Chrome path as `colight render` with a
    fixed viewport and device-pixel-ratio, waits for render completion, and
    renders at t=0 (no update entries applied). With --check the render is
    repeated in a fresh tab and byte-compared.

    Scene3d targets additionally report a `coverage` object (fraction of
    canvas pixels each component covers, from the GPU pick buffer) and
    accept --frame "C[:A-B]" to fit the camera on a component (or instance
    range) before capture — the agent's zoom loop.

    Machine legibility (all composed post-capture; the render stays
    untouched): --rulers surrounds the capture with big labeled coordinate
    rulers in the exact page-pixel space `pick-at` consumes (read a
    coordinate off the ruler, pass it to pick-at); --views renders a
    labeled contact sheet of camera presets (one image of four views costs
    an agent fewer tiles than four images) — rulers are single-view only
    and error when combined; --max-edge N sizes the render so the PNG's
    long edge is exactly N, letting agents that know their harness's
    native input size skip a second lossy resampling.

    Exit code: 0 success, 1 --check found nondeterminism, 2 error.

    \b
    JSON schema:
      {"target": str, "out": str, "width": int, "height": int, "dpr": float,
       "block"?: str, "sha256": str, "deterministic"?: bool,
       "sha256_recheck"?: str, "max_edge"?: int,
       "rulers"?: {"spacing": int, "margin": int},
       "views"?: [{"view": str, "camera": {...}}],
       "frame"?: {"component": int, "type": str, "instances"?: [[a, b]],
                  "camera": {"position", "target", "up", "fov", ...}},
       "coverage"?: {"width": int, "height": int, "rect": {...},
         "components": [{"component", "type", "instances", "pixels",
                         "fraction"}],
         "background": {"pixels": int, "fraction": float}},
       "legends"?: [{"component"?: int, "type"?: str, "label"?: str,
                     "cmap": str, "domain"?: [lo, hi],
                     "categorical": bool, "categories"?: [str]}]}
    (width/height are actual PNG pixel dimensions including any composed
    margin; rulers.spacing is CSS px between ticks, rulers.margin the
    composed band in PNG px — page coordinate = (png_px - margin) / dpr;
    coverage fractions are of the scene canvas; legends report what
    colormap-driven colors encode, read from the same DOM legends visible
    in the capture.)
    """
    from colight.cli_tools import daemon_client, scene_pick, screenshot_tools

    effective_timeout = None if ready_timeout <= 0 else ready_timeout

    def run_shot(
        run_width: int,
        run_height: Optional[int],
        run_out: pathlib.Path,
        run_check: bool,
        run_frame: Optional[str],
        run_rulers: bool,
        run_views: Optional[List[str]],
    ) -> dict:
        payload = None
        if not no_daemon:
            payload = daemon_client.try_screenshot(
                target,
                run_out,
                block=block,
                width=run_width,
                height=run_height,
                dpr=dpr,
                check=run_check,
                frame=run_frame,
                debug=debug,
                ready_timeout=effective_timeout,
                rulers=run_rulers,
                views=run_views,
            )
        if payload is None:
            payload = screenshot_tools.screenshot_target(
                target,
                run_out,
                block=block,
                width=run_width,
                height=run_height,
                dpr=dpr,
                check=run_check,
                frame=run_frame,
                debug=debug,
                ready_timeout=effective_timeout,
                rulers=run_rulers,
                views=run_views,
            )
        return payload

    try:
        if rulers and views:
            raise ValueError(
                "--rulers applies to single-view screenshots only "
                "(a contact sheet's cells have their own coordinate spaces)"
            )
        view_names = scene_pick.parse_views(views) if views else None

        if max_edge is not None:
            long_css = max_edge / dpr
            if abs(long_css - round(long_css)) > 1e-9:
                raise ValueError(
                    f"--max-edge {max_edge} is not reachable at --dpr {dpr:g} "
                    "(max-edge must be divisible by dpr)"
                )
            if height is None:
                # Measure the content's aspect with a probe render at the
                # requested width, then re-render at the fitted viewport.
                with tempfile.TemporaryDirectory() as tmp_dir:
                    probe = run_shot(
                        width,
                        None,
                        pathlib.Path(tmp_dir) / "probe.png",
                        False,
                        None,
                        False,
                        None,
                    )
                height = max(1, int(round(probe["height"] / dpr)))
            width, height = screenshot_tools.fit_max_edge(
                width, height, int(round(long_css))
            )

        payload = run_shot(width, height, out, check, frame, rulers, view_names)
        if max_edge is not None:
            payload["max_edge"] = max_edge
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if as_json:
        _echo_json(payload)
    else:
        parts = [
            f"{payload['out']}",
            f"{payload['width']}x{payload['height']}px",
            f"dpr={payload['dpr']:g}",
            f"sha256={payload['sha256'][:16]}…",
        ]
        if "block" in payload:
            parts.insert(1, f"block={payload['block']}")
        if "rulers" in payload:
            parts.append(
                f"rulers: every {payload['rulers']['spacing']}px, "
                f"margin {payload['rulers']['margin']}px"
            )
        if "views" in payload:
            parts.append("views: " + ", ".join(v["view"] for v in payload["views"]))
        if "frame" in payload:
            parts.append(
                f"framed={payload['frame']['type']}[{payload['frame']['component']}]"
            )
        if "coverage" in payload:
            fractions = ", ".join(
                f"{c['type']}[{c['component']}]={c['fraction']:.1%}"
                for c in payload["coverage"]["components"]
            )
            background = payload["coverage"]["background"]["fraction"]
            parts.append(f"coverage: {fractions} bg={background:.1%}")
        if "deterministic" in payload:
            parts.append(
                "deterministic" if payload["deterministic"] else "NONDETERMINISTIC"
            )
        click.echo("  ".join(parts))

    if payload.get("deterministic") is False:
        sys.exit(1)


@main.command("pick-at")
@click.argument("target", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("coords", type=str)
@click.option(
    "--radius",
    type=float,
    default=6.0,
    show_default=True,
    help="Sampling disc radius in CSS pixels.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--block",
    type=str,
    help="Stable block id to select a visual (.py targets; default = last visual).",
)
@click.option(
    "--width",
    type=int,
    default=800,
    show_default=True,
    help="Viewport width in CSS pixels (match your screenshot).",
)
@click.option(
    "--height",
    type=int,
    help="Viewport height in CSS pixels (default: measure rendered content).",
)
@click.option(
    "--dpr",
    type=float,
    default=1.0,
    show_default=True,
    help="Device pixel ratio.",
)
@click.option(
    "--ready-timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Max seconds to wait for render readiness (0 to disable).",
)
@click.option(
    "--no-daemon",
    is_flag=True,
    help="Bypass a running colight daemon and render directly.",
)
@click.option("--debug", is_flag=True, help="Enable renderer debug logging.")
def pick_at(
    target: pathlib.Path,
    coords: str,
    radius: float,
    as_json: bool,
    block: Optional[str],
    width: int,
    height: Optional[int],
    dpr: float,
    ready_timeout: float,
    no_daemon: bool,
    debug: bool,
):
    """What is at point X,Y? Re-render TARGET and query the GPU pick buffer.

    COORDS is "X,Y" in CSS pixels of the rendered page: origin at the
    top-left, y grows downward — the same coordinate space as a
    `colight screenshot` PNG taken with the same --width/--height (at
    --dpr 1; with a different dpr, divide PNG pixel coordinates by dpr).
    Hits within --radius are ranked by distance then coverage, and the top
    hits include the instance's dereferenced attribute values
    (center/color/size/... as actually rendered). Scene3d targets only:
    other visuals exit 2 with a clear message.

    Exit code: 0 hit found, 1 no hit within radius, 2 error.

    \b
    JSON schema:
      {"target": str, "block"?: str, "x": float, "y": float,
       "radius": float,
       "scene": {"rect": {"left", "top", "width", "height"},
                 "width": int, "height": int, "dpr": float, "scenes"?: int},
       "hits": [{"component": int, "type": str, "instance": int,
                 "distance": float, "pixels": int, "share": float,
                 "values"?: {"center": [x, y, z], "color"?, "alpha"?,
                             "half_size"?, "size"?, "quaternion"?, ...}}],
       "background_share": float}
    (distance in CSS px from the query point; share = fraction of the
    sampled disc covered by that instance; scene.rect maps page pixels to
    the canvas.)
    """
    from colight.cli_tools import daemon_client, scene_pick

    try:
        x_text, y_text = coords.split(",", 1)
        x, y = float(x_text), float(y_text)
    except ValueError:
        click.echo(f'Error: COORDS must be "X,Y", got {coords!r}', err=True)
        sys.exit(2)

    effective_timeout = None if ready_timeout <= 0 else ready_timeout
    try:
        payload = None
        if not no_daemon:
            payload = daemon_client.try_pick_at(
                target,
                x,
                y,
                radius=radius,
                block=block,
                width=width,
                height=height,
                dpr=dpr,
                debug=debug,
                ready_timeout=effective_timeout,
            )
        if payload is None:
            payload = scene_pick.pick_at_target(
                target,
                x,
                y,
                radius=radius,
                block=block,
                width=width,
                height=height,
                dpr=dpr,
                debug=debug,
                ready_timeout=effective_timeout,
            )
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if as_json:
        _echo_json(payload)
    else:
        if not payload["hits"]:
            click.echo(
                f"no hit within {payload['radius']:g}px of "
                f"({payload['x']:g}, {payload['y']:g})"
            )
        for hit in payload["hits"]:
            values = hit.get("values") or {}
            center = values.get("center")
            center_text = (
                " center=[" + ", ".join(f"{v:.3g}" for v in center) + "]"
                if center
                else ""
            )
            click.echo(
                f"{hit['type']}[{hit['component']}] instance {hit['instance']}"
                f"  dist={hit['distance']:g}px share={hit['share']:.1%}"
                f"{center_text}"
            )

    sys.exit(0 if payload["hits"] else 1)


@main.command("pick-where")
@click.argument("target", type=click.Path(exists=True, path_type=pathlib.Path))
@click.option(
    "--component",
    "component_selector",
    required=True,
    type=str,
    help="Component index or type name (as reported by coverage/pick-at).",
)
@click.option(
    "--instances",
    type=str,
    help='Inclusive instance ranges, e.g. "0-3,7" (default: all instances).',
)
@click.option(
    "--out",
    "-o",
    type=click.Path(path_type=pathlib.Path),
    help="Write a highlight-overlay PNG (selection emphasized, rest dimmed).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--block",
    type=str,
    help="Stable block id to select a visual (.py targets; default = last visual).",
)
@click.option(
    "--width",
    type=int,
    default=800,
    show_default=True,
    help="Viewport width in CSS pixels (match your screenshot).",
)
@click.option(
    "--height",
    type=int,
    help="Viewport height in CSS pixels (default: measure rendered content).",
)
@click.option(
    "--dpr",
    type=float,
    default=1.0,
    show_default=True,
    help="Device pixel ratio.",
)
@click.option(
    "--ready-timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Max seconds to wait for render readiness (0 to disable).",
)
@click.option(
    "--no-daemon",
    is_flag=True,
    help="Bypass a running colight daemon and render directly.",
)
@click.option("--debug", is_flag=True, help="Enable renderer debug logging.")
def pick_where(
    target: pathlib.Path,
    component_selector: str,
    instances: Optional[str],
    out: Optional[pathlib.Path],
    as_json: bool,
    block: Optional[str],
    width: int,
    height: Optional[int],
    dpr: float,
    ready_timeout: float,
    no_daemon: bool,
    debug: bool,
):
    """Where does a selection land on screen? Selection -> screen truth.

    Re-renders TARGET (scene3d only) and reports the selection's visible
    pixel count, bounding box and centroid (page CSS pixels, origin
    top-left, y down — the same space `pick-at` samples), plus a
    visibility fraction: visible pixels / the pixels the selection would
    cover with everything else hidden (its unoccluded projected footprint,
    measured by re-rendering only the selection). visibility < 1 means
    partially occluded; 0 with a nonzero projected footprint means fully
    occluded; 0 with no footprint means out of view. With --out, writes a
    PNG with the selection highlighted via scene3d's per-instance
    decorations (everything else dimmed) so the selection can be SEEN.

    Exit code: 0 selection visible, 1 selection entirely invisible,
    2 error.

    \b
    JSON schema:
      {"target": str, "block"?: str, "component": int, "type": str,
       "instances": [[a, b]] | "all",
       "scene": {"rect": {...}, "width": int, "height": int, "dpr": float},
       "selected": int, "visible_pixels": int, "projected_pixels": int,
       "visibility": float, "visible_instances": int,
       "hidden_instances": int,
       "bbox"?: [x0, y0, x1, y1], "centroid"?: [x, y],
       "projected_bbox"?: [x0, y0, x1, y1], "out"?: str}
    (bbox/centroid in page CSS pixels; projected_bbox appears when the
    selection is fully occluded but would land on screen.)
    """
    from colight.cli_tools import daemon_client, scene_pick

    effective_timeout = None if ready_timeout <= 0 else ready_timeout
    try:
        ranges = (
            scene_pick.parse_instance_ranges(instances)
            if instances is not None
            else None
        )
        payload = None
        if not no_daemon:
            payload = daemon_client.try_pick_where(
                target,
                component_selector,
                instances=ranges,
                out=out,
                block=block,
                width=width,
                height=height,
                dpr=dpr,
                debug=debug,
                ready_timeout=effective_timeout,
            )
        if payload is None:
            payload = scene_pick.pick_where_target(
                target,
                component_selector,
                instances=ranges,
                out=out,
                block=block,
                width=width,
                height=height,
                dpr=dpr,
                debug=debug,
                ready_timeout=effective_timeout,
            )
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if as_json:
        _echo_json(payload)
    else:
        selection = f"{payload['type']}[{payload['component']}]"
        if payload["instances"] != "all":
            selection += f" instances {payload['instances']}"
        click.echo(
            f"{selection}: {payload['visible_instances']}/{payload['selected']} "
            f"instance(s) visible, {payload['visible_pixels']}px "
            f"(visibility {payload['visibility']:.1%})"
        )
        if "bbox" in payload:
            click.echo(f"  bbox={payload['bbox']} centroid={payload['centroid']}")
        elif "projected_bbox" in payload:
            click.echo(f"  fully occluded; would cover {payload['projected_bbox']}")
        else:
            click.echo("  out of view (no projected footprint)")
        if "out" in payload:
            click.echo(f"  overlay: {payload['out']}")

    sys.exit(0 if payload["visible_pixels"] > 0 else 1)


@main.command()
@click.argument(
    "targets",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, path_type=pathlib.Path),
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--update",
    is_flag=True,
    help="Write/refresh goldens, reporting what changed vs the previous set.",
)
@click.option(
    "--goldens",
    "goldens_root",
    type=click.Path(path_type=pathlib.Path),
    help="Golden storage root (default: <project_root>/tests/goldens).",
)
@click.option(
    "--no-pixels",
    is_flag=True,
    help="Skip the screenshot layer (for Chrome-less environments).",
)
@click.option(
    "--epsilon",
    type=float,
    default=1e-9,
    show_default=True,
    help="Numeric threshold for the semantic diff reported on mismatch.",
)
@click.option(
    "--no-daemon",
    is_flag=True,
    help="Bypass a running colight daemon for the screenshot layer.",
)
def verify(
    targets: tuple[pathlib.Path, ...],
    as_json: bool,
    update: bool,
    goldens_root: Optional[pathlib.Path],
    no_pixels: bool,
    epsilon: float,
    no_daemon: bool,
):
    """Verify TARGETs against stored goldens (or pin them with --update).

    Each TARGET is a .py document (every visual-producing block gets a
    golden) or a .colight artifact. A golden pins three layers per visual:
    the .colight artifact bytes, the canonicalized-structure hash (per-run
    ids normalized), and the deterministic screenshot sha256 + dimensions.
    On mismatch the report says which layer changed; structure changes
    include a semantic diff summary (max/mean |delta|, changed paths).
    The screenshot layer is skipped with a warning when Chrome or the JS
    bundle is unavailable, or explicitly via --no-pixels.

    Goldens live in <project_root>/tests/goldens/<relpath-of-target>/
    (manifest.json + one .colight per visual); --goldens overrides.

    Exit code: 0 all match (or update succeeded), 1 mismatches, 2 error,
    3 no goldens found (run with --update to create them).

    \b
    JSON schema:
      {"ok": bool, "targets": [
        {"target": str, "goldens": str, "kind": "py"|"colight",
         "status": "match"|"mismatch"|"no-goldens"|"updated"|"error",
         "warnings"?: [str], "hint"?: str, "errors"?: [...],
         "blocks": [
           {"id": str, "lines"?: [start, end], "golden_id"?: str,
            "status": "match"|"structure-changed"|"pixels-changed"|"new"|
                      "removed"|"added"|"updated"|"unchanged",
            "layer"?: "structure"|"pixels",
            "structure"?: {"hash", "golden_hash", "match"},
            "pixels"?: {"sha256", "golden_sha256", "width", "height",
                        "match"},
            "diff"?: {"arrays_changed": int, "max_abs_delta"?: float,
                      "max_abs_delta_path"?: str, "mean_abs_delta"?: float,
                      "changed_paths"?: [str], "components": {...},
                      "values_changed": int, "state": {...}}}]}]}
    """
    from colight.cli_tools import daemon_client, verify_tools

    session_factory = None
    if not no_daemon and not no_pixels and targets:
        info = daemon_client.discover_for_target(targets[0])
        if info is not None:
            session_factory = lambda width, dpr: daemon_client.RemoteRenderSession(  # noqa: E731
                info, width, dpr
            )

    try:
        payload, exit_code = verify_tools.run_verify(
            list(targets),
            goldens_root=goldens_root,
            update=update,
            pixels=not no_pixels,
            epsilon=epsilon,
            session_factory=session_factory,
        )
    except daemon_client.DaemonUnavailable:
        # The daemon vanished mid-run; redo the whole verify directly.
        payload, exit_code = verify_tools.run_verify(
            list(targets),
            goldens_root=goldens_root,
            update=update,
            pixels=not no_pixels,
            epsilon=epsilon,
        )

    if as_json:
        _echo_json(payload)
        sys.exit(exit_code)

    for result in payload["targets"]:
        click.echo(f"{result['target']} — {result['status']}")
        for warning in result.get("warnings", []):
            click.echo(f"  warning: {warning}")
        if "hint" in result:
            click.echo(f"  {result['hint']}")
        for block in result.get("blocks", []):
            line = f"  {block['status']:<18} {block['id']}"
            if block.get("layer"):
                line += f" [{block['layer']}]"
            diff = block.get("diff")
            if diff:
                parts = [f"{diff['arrays_changed']} array(s)"]
                if "max_abs_delta" in diff:
                    parts.append(
                        f"max |Δ| {diff['max_abs_delta']:.4g} "
                        f"in {diff['max_abs_delta_path']}"
                    )
                if diff.get("values_changed"):
                    parts.append(f"{diff['values_changed']} value(s)")
                line += f" — {'; '.join(parts)}"
            elif block["status"] == "pixels-changed" and "pixels" in block:
                line += (
                    f" — sha {block['pixels']['golden_sha256'][:12]}… -> "
                    f"{block['pixels']['sha256'][:12]}…"
                )
            click.echo(line)
        for error_item in result.get("errors", []):
            error = error_item.get("error", {})
            block_id = error_item.get("block", "?")
            click.echo(
                f"  error in block {block_id}: "
                f"{error.get('type')}: {error.get('message')}",
                err=True,
            )

    sys.exit(exit_code)


@main.group()
def daemon():
    """Manage the colight render daemon (keeps headless Chrome warm).

    The daemon serves the render-path commands (screenshot, pick-at,
    pick-where, verify pixels) from a pool of warm Chrome instances plus a
    small cache of loaded scenes, so tight agent loops skip the ~1-2s
    browser launch per invocation. Discovery is automatic: it writes
    <project_root>/.colight_cache/daemon.json and commands use it whenever
    that file points at a live, version-matched daemon — otherwise they
    silently run direct. Pass --no-daemon to any routed command to bypass.
    """


@daemon.command("start")
@click.option(
    "--idle-timeout",
    type=float,
    default=1800.0,
    show_default=True,
    help="Self-shutdown after this many seconds without a tool request.",
)
@click.option(
    "--pool",
    "pool_size",
    type=int,
    default=2,
    show_default=True,
    help="Maximum concurrent isolated Chrome instances.",
)
@click.option(
    "--scene-cache",
    type=int,
    default=4,
    show_default=True,
    help="Loaded scenes kept warm (LRU).",
)
@click.option(
    "--foreground",
    is_flag=True,
    help="Run in the foreground instead of detaching.",
)
@click.option("--verbose", "-v", is_flag=True, help="Log requests to stderr.")
def daemon_start(
    idle_timeout: float,
    pool_size: int,
    scene_cache: int,
    foreground: bool,
    verbose: bool,
):
    """Start a daemon for the current project root (no-op if one runs)."""
    from colight.cli_tools import daemon as daemon_mod
    from colight.cli_tools import daemon_client
    from colight.runtime.parser import find_project_root

    root = find_project_root(pathlib.Path.cwd())
    existing = daemon_client.read_daemon_file(daemon_client.daemon_file_path(root))
    if existing is not None and daemon_client.validate_info(existing):
        click.echo(f"daemon already running (pid {existing.pid}, port {existing.port})")
        return

    if foreground:
        server = daemon_mod.DaemonServer(
            root,
            idle_timeout=idle_timeout,
            pool_size=pool_size,
            scene_cache=scene_cache,
            verbose=verbose,
        )
        server.start()
        click.echo(
            f"colight daemon on 127.0.0.1:{server.port} "
            f"(root {root}, pool {pool_size}, idle-timeout {idle_timeout:g}s)"
        )
        server.run_until_shutdown()
        return

    log_path = root / ".colight_cache" / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "colight_cli",
        "daemon",
        "start",
        "--foreground",
        "--idle-timeout",
        str(idle_timeout),
        "--pool",
        str(pool_size),
        "--scene-cache",
        str(scene_cache),
    ]
    if verbose:
        command.append("--verbose")
    with open(log_path, "ab") as log:
        subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.time() + 20.0
    while time.time() < deadline:
        info = daemon_client.read_daemon_file(daemon_client.daemon_file_path(root))
        if info is not None and daemon_client.validate_info(info):
            click.echo(f"daemon started (pid {info.pid}, port {info.port})")
            return
        time.sleep(0.1)
    click.echo(f"Error: daemon did not come up (see {log_path})", err=True)
    sys.exit(1)


@daemon.command("stop")
def daemon_stop():
    """Stop the project's daemon (reads the discovery file, signals it)."""
    import signal as signal_mod

    from colight.cli_tools import daemon_client
    from colight.runtime.parser import find_project_root

    root = find_project_root(pathlib.Path.cwd())
    path = daemon_client.daemon_file_path(root)
    info = daemon_client.read_daemon_file(path)
    if info is None or not daemon_client.pid_alive(info.pid):
        if info is not None:
            path.unlink(missing_ok=True)
        click.echo("no daemon running")
        return
    os.kill(info.pid, signal_mod.SIGTERM)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not daemon_client.pid_alive(info.pid):
            path.unlink(missing_ok=True)
            click.echo(f"daemon stopped (pid {info.pid})")
            return
        time.sleep(0.1)
    click.echo(f"Error: daemon (pid {info.pid}) did not exit within 10s", err=True)
    sys.exit(1)


@daemon.command("status")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def daemon_status(as_json: bool):
    """Report the project daemon's health, pool and warm-scene stats."""
    from colight.cli_tools import daemon_client
    from colight.runtime.parser import find_project_root

    root = find_project_root(pathlib.Path.cwd())
    info = daemon_client.read_daemon_file(daemon_client.daemon_file_path(root))
    if info is None or not daemon_client.validate_info(info):
        if as_json:
            _echo_json({"running": False})
        else:
            click.echo("no daemon running")
        sys.exit(1)
    status = daemon_client.request(info, "GET", "/status")
    if as_json:
        _echo_json({"running": True, **status})
        return
    pool = status["pool"]
    warm = status["warm"]
    requests = status["requests"]
    click.echo(
        f"daemon pid {status['pid']} port {status['port']} "
        f"uptime {status['uptime']:g}s idle-timeout {status['idle_timeout']:g}s"
    )
    click.echo(
        f"  chrome pool: {pool['instances']}/{pool['max']} instances "
        f"({pool['busy']} busy, {pool['launches']} launched)"
    )
    click.echo(
        f"  warm scenes: {warm['entries']}/{warm['capacity']} "
        f"(hits {warm['hits']}, misses {warm['misses']}, "
        f"evictions {warm['evictions']})"
    )
    click.echo(f"  requests: {requests['total']}")


if __name__ == "__main__":
    main()
