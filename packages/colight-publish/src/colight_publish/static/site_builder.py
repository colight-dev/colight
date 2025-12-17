"""Tools for building the interactive documentation site as a static artifact."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from typing import Iterable, Optional

import click
from colight.env import DIST_LOCAL_PATH, VERSIONED_CDN_DIST_URL
from time import perf_counter

from ..constants import DEFAULT_INLINE_THRESHOLD
from ..file_resolver import find_files
from ..json_api import JsonDocumentGenerator, build_file_tree_json
from ..logging import build_rich_progress
from ..pragma import parse_pragma_arg
from ..utils import merge_ignore_patterns


def _write_index_html(output_dir: pathlib.Path, script_src: str) -> None:
    """Write the SPA entry point that boots the live explorer."""
    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Colight Publish</title>
    <style>
      html, body {{
        margin: 0;
        padding: 0;
        min-height: 100%;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      body {{
        background-color: #f8f9fb;
      }}
    </style>
  </head>
  <body>
    <div id="root"></div>
    <script>
      window.__COLIGHT_STATIC_MODE__ = true;
    </script>
    <script src="{script_src}"></script>
  </body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _copy_live_bundle(dist_dir: pathlib.Path) -> str:
    """Copy the live explorer bundle locally and return the script src."""
    dist_dir.mkdir(parents=True, exist_ok=True)
    live_bundle = DIST_LOCAL_PATH / "live.js"
    if not live_bundle.exists():
        raise FileNotFoundError(f"Could not find live.js at {live_bundle}")

    shutil.copy2(live_bundle, dist_dir / "live.js")
    live_map = live_bundle.with_suffix(".js.map")
    if live_map.exists():
        shutil.copy2(live_map, dist_dir / "live.js.map")
    return "/dist/live.js"


def _prepare_bundle(output_dir: pathlib.Path) -> str:
    """Decide whether to use the CDN bundle or copy the local bundle."""
    if VERSIONED_CDN_DIST_URL:
        return f"{VERSIONED_CDN_DIST_URL}/live.js"
    return _copy_live_bundle(output_dir / "dist")


def _write_visuals(visual_dir: pathlib.Path, store: dict[str, bytes]) -> None:
    """Persist collected .colight blobs to disk."""
    if not store:
        return
    visual_dir.mkdir(parents=True, exist_ok=True)
    for key, data in store.items():
        (visual_dir / f"{key}.colight").write_bytes(data)


def _relative_output_path(
    input_path: pathlib.Path, file_path: pathlib.Path
) -> pathlib.Path:
    try:
        return file_path.relative_to(input_path)
    except ValueError:
        return pathlib.Path(file_path.name)


def _markdown_document_json(file_path: pathlib.Path) -> str:
    content = file_path.read_text(encoding="utf-8").strip()
    block_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    block = {
        "id": block_hash[:16],
        "interface": {"provides": [], "requires": []},
        "interface_hash": "e3b0c44298fc1c14",
        "content_hash": block_hash[:16],
        "line": 1,
        "ordinal": 0,
        "elements": [
            {
                "type": "prose",
                "value": content,
                "show": True,
            }
        ],
        "error": None,
        "stdout": None,
        "showsVisual": False,
    }
    doc = {
        "file": file_path.name,
        "metadata": {"pragma": [], "title": file_path.stem},
        "blocks": [block],
    }
    return json.dumps(doc, indent=2)


def build_static_site(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    include: Optional[Iterable[str]] = None,
    ignore: Optional[Iterable[str]] = None,
    pragma: Optional[str] = None,
    inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    verbose: bool = False,
) -> None:
    """Build the interactive site as a static artifact."""
    include_patterns = list(include) if include else ["*.py", "*.md"]
    ignore_patterns = merge_ignore_patterns(list(ignore) if ignore else None)

    if input_path.is_file():
        files = [input_path]
    else:
        files = find_files(input_path, include_patterns, ignore_patterns)

    if not files:
        raise click.ClickException("No matching Python files found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    api_dir = output_dir / "api"
    document_dir = api_dir / "document"
    visual_dir = api_dir / "visual"
    document_dir.mkdir(parents=True, exist_ok=True)

    visual_store: dict[str, bytes] = {}
    generator = JsonDocumentGenerator(
        verbose=verbose,
        pragma=parse_pragma_arg(pragma) if pragma else set(),
        visual_store=visual_store,
        inline_threshold=inline_threshold,
    )

    progress = build_rich_progress()
    if progress is None:

        class _FallbackProgress:
            console = None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def add_task(self, *args, **kwargs):
                return None

            def update(self, *args, **kwargs):
                pass

            def advance(self, *args, **kwargs):
                pass

            def log(self, message):
                click.echo(message)

        progress = _FallbackProgress()
    console = getattr(progress, "console", None)
    timings = []

    start_all = perf_counter()
    with progress:
        task = progress.add_task("Building site", total=len(files))
        for file_path in files:
            rel_path = _relative_output_path(input_path, file_path)
            display_name = str(rel_path) if str(rel_path) != "." else file_path.name
            target = document_dir / f"{rel_path}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            progress.update(task, description=display_name)
            progress.log(f"[cyan]→[/] Building {display_name}...")

            start = perf_counter()
            try:
                if file_path.suffix.lower() == ".md":
                    json_content = _markdown_document_json(file_path)
                else:
                    json_content = generator.generate_json(file_path)
                target.write_text(json_content, encoding="utf-8")
            except Exception as exc:  # pragma: no cover - propagated
                raise
            finally:
                duration_ms = (perf_counter() - start) * 1000
                timings.append((display_name, duration_ms))
                progress.log(f"    [green]✓ {duration_ms:.1f} ms[/]")
                progress.advance(task)

    tree = build_file_tree_json(files, input_path)
    (api_dir / "index.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")

    _write_visuals(visual_dir, visual_store)
    script_src = _prepare_bundle(output_dir)
    _write_index_html(output_dir, script_src)

    total_ms = (perf_counter() - start_all) * 1000
    summary = f"Site built in {total_ms:.1f} ms ({len(files)} documents)."
    if console:
        console.print(f"[bold green]{summary}[/]")
        if verbose:
            console.print("[bold]Timings:[/]")
            for name, duration in timings:
                console.print(f"  [dim]{name}[/]: {duration:.1f} ms")
    else:
        click.echo(summary)
