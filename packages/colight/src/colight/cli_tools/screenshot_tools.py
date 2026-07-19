"""Deterministic screenshots of visuals via the existing headless renderer.

Builds on the same rendering path as ``colight render`` (StudioContext /
headless Chrome), adding agent-facing ergonomics and a determinism pass:
fixed viewport and device-pixel-ratio, wait-for-render-complete, no update
entries applied (render at t=0 — animated state stays at its initial
values), and an optional double-render byte-hash check.
"""

import hashlib
import pathlib
import struct
from typing import Any, Dict, List, Optional, Tuple

import colight.format as colight_format
from colight.screenshots import StudioContext

from . import inspect_tools

DEFAULT_WIDTH = 800
DEFAULT_DPR = 1.0
DEFAULT_READY_TIMEOUT = 30.0


def _png_size(png: bytes) -> Tuple[int, int]:
    """Pixel dimensions from a PNG's IHDR chunk."""
    width, height = struct.unpack(">II", png[16:24])
    return int(width), int(height)


def resolve_visual(
    target: pathlib.Path, block: Optional[str] = None
) -> Tuple[Dict[str, Any], List[bytes], Optional[str]]:
    """Resolve a target to a single visual's (data, buffers, block id).

    ``.colight`` artifacts are parsed directly (initial state only — update
    entries are not applied, so animated visuals render at t=0). ``.py``
    files are evaluated headlessly; ``block`` selects the visual produced by
    that stable block id, defaulting to the last visual in the file.

    Args:
        target: A ``.colight`` artifact or notebook-style ``.py`` file.
        block: Stable block id (``.py`` targets only).

    Returns:
        Tuple of (data, buffers, block id or None).

    Raises:
        ValueError: Unsupported target, no visual produced, or unknown block.
    """
    if target.suffix == ".colight":
        if block is not None:
            raise ValueError("--block applies only to .py targets")
        data, buffers, _updates = colight_format.parse_file(target)
        if data is None:
            raise ValueError(f"file contains no initial state entry: {target}")
        return data, buffers, None
    if target.suffix == ".py":
        visuals, errors = inspect_tools.evaluate_python_visuals(target)
        if block is not None:
            matches = [v for v in visuals if v["block"] == block]
            if not matches:
                known = ", ".join(v["block"] for v in visuals) or "none"
                raise ValueError(
                    f"block {block} produced no visual (blocks with visuals: {known})"
                )
            selected = matches[0]
        else:
            if not visuals:
                detail = ""
                if errors:
                    first = errors[0]["error"]
                    detail = f" ({first.get('type')}: {first.get('message')})"
                raise ValueError(f"no visuals produced by {target}{detail}")
            selected = visuals[-1]
        return selected["data"], selected["buffers"], selected["block"]
    raise ValueError(f"Unsupported target (expected .colight or .py): {target}")


def render_png(
    data: Dict[str, Any],
    buffers: List[bytes],
    width: int = DEFAULT_WIDTH,
    height: Optional[int] = None,
    dpr: float = DEFAULT_DPR,
    debug: bool = False,
    ready_timeout: Optional[float] = DEFAULT_READY_TIMEOUT,
) -> bytes:
    """Render a visual to PNG bytes in a fresh tab, waiting for readiness.

    Args:
        data: Visual JSON envelope.
        buffers: Binary buffers.
        width: CSS-pixel viewport width.
        height: CSS-pixel viewport height; None measures the rendered
            content (deterministic for a given input).
        dpr: Device pixel ratio (output pixels = CSS pixels * dpr).
        debug: Verbose renderer logging.
        ready_timeout: Max seconds to wait for render readiness.

    Returns:
        PNG bytes.
    """
    with StudioContext(
        width=width,
        height=height,
        scale=dpr,
        debug=debug,
        ready_timeout=ready_timeout,
        reuse=True,
        keep_alive=1.0,
    ) as studio:
        studio.load_plot(data=data, buffers=buffers, measure=height is None)
        return studio.capture_bytes(format="png")


def screenshot_target(
    target: pathlib.Path,
    out: pathlib.Path,
    block: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: Optional[int] = None,
    dpr: float = DEFAULT_DPR,
    check: bool = False,
    debug: bool = False,
    ready_timeout: Optional[float] = DEFAULT_READY_TIMEOUT,
) -> Dict[str, Any]:
    """Screenshot a target deterministically.

    Args:
        target: A ``.colight`` artifact or ``.py`` file.
        out: Output PNG path (parent dirs are created).
        block: Stable block id to select a visual (``.py`` only; default =
            last visual).
        width: CSS-pixel viewport width.
        height: CSS-pixel viewport height (None = measure content).
        dpr: Device pixel ratio.
        check: Render twice in fresh tabs and byte-compare, reporting
            ``deterministic`` in the payload.
        debug: Verbose renderer logging.
        ready_timeout: Max seconds to wait for render readiness.

    Returns:
        Payload with ``target``, ``out``, ``width``/``height`` (actual PNG
        pixels), ``dpr``, ``block`` (when a .py block was selected),
        ``sha256``, and ``deterministic`` (only when ``check`` is set;
        ``sha256_recheck`` is added when the two renders differ).
    """
    data, buffers, block_id = resolve_visual(target, block)
    png = render_png(
        data,
        buffers,
        width=width,
        height=height,
        dpr=dpr,
        debug=debug,
        ready_timeout=ready_timeout,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)

    pixel_width, pixel_height = _png_size(png)
    payload: Dict[str, Any] = {
        "target": str(target),
        "out": str(out),
        "width": pixel_width,
        "height": pixel_height,
        "dpr": dpr,
        "sha256": hashlib.sha256(png).hexdigest(),
    }
    if block_id is not None:
        payload["block"] = block_id
    if check:
        png_recheck = render_png(
            data,
            buffers,
            width=width,
            height=height,
            dpr=dpr,
            debug=debug,
            ready_timeout=ready_timeout,
        )
        payload["deterministic"] = png_recheck == png
        if not payload["deterministic"]:
            payload["sha256_recheck"] = hashlib.sha256(png_recheck).hexdigest()
    return payload


__all__ = [
    "DEFAULT_DPR",
    "DEFAULT_READY_TIMEOUT",
    "DEFAULT_WIDTH",
    "render_png",
    "resolve_visual",
    "screenshot_target",
]
