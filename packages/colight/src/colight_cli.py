"""CLI interface for Colight."""

import asyncio
import base64
import json
import pathlib
import subprocess
import sys
import tempfile
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
              "state_keys": [str], "synced_keys": [str],
              "listeners": [str], "py_listeners": [str],
              "buffers": {"count": int, "total_bytes": int}}
    WARNING = {"code": str, "path": str, "message": str}
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
        if "error" in payload:
            click.echo(f"Error: {payload['error']['message']}", err=True)
            sys.exit(1)
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
    "--ready-timeout",
    type=float,
    default=30.0,
    show_default=True,
    help="Max seconds to wait for render readiness (0 to disable).",
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
    ready_timeout: float,
    debug: bool,
):
    """Render TARGET to a PNG with deterministic settings.

    TARGET is a .colight artifact or a .py file (evaluated headlessly;
    --block selects one visual by stable id, default = last visual in the
    file). Uses the same headless-Chrome path as `colight render` with a
    fixed viewport and device-pixel-ratio, waits for render completion, and
    renders at t=0 (no update entries applied). With --check the render is
    repeated in a fresh tab and byte-compared.

    Exit code: 0 success, 1 --check found nondeterminism, 2 error.

    \b
    JSON schema:
      {"target": str, "out": str, "width": int, "height": int, "dpr": float,
       "block"?: str, "sha256": str, "deterministic"?: bool,
       "sha256_recheck"?: str}
    (width/height are actual PNG pixel dimensions.)
    """
    from colight.cli_tools import screenshot_tools

    try:
        payload = screenshot_tools.screenshot_target(
            target,
            out,
            block=block,
            width=width,
            height=height,
            dpr=dpr,
            check=check,
            debug=debug,
            ready_timeout=None if ready_timeout <= 0 else ready_timeout,
        )
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
        if "deterministic" in payload:
            parts.append(
                "deterministic" if payload["deterministic"] else "NONDETERMINISTIC"
            )
        click.echo("  ".join(parts))

    if payload.get("deterministic") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
