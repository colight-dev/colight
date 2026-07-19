"""Deterministic screenshots of visuals via the existing headless renderer.

Builds on the same rendering path as ``colight render`` (StudioContext /
headless Chrome), adding agent-facing ergonomics and a determinism pass:
fixed viewport and device-pixel-ratio, wait-for-render-complete, no update
entries applied (render at t=0 — animated state stays at its initial
values), and an optional double-render byte-hash check.
"""

import hashlib
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from colight.screenshots import StudioContext

from . import targets

DEFAULT_WIDTH = 800
DEFAULT_DPR = 1.0
DEFAULT_READY_TIMEOUT = 30.0


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
    if target.suffix == ".colight" and block is not None:
        raise ValueError("--block applies only to .py targets")
    loaded = targets.load_target(target)
    if loaded.kind == "colight":
        visual = loaded.visuals[0]
        return visual["data"], visual["buffers"], None
    if block is not None:
        matches = [v for v in loaded.visuals if v["block"] == block]
        if not matches:
            known = ", ".join(v["block"] for v in loaded.visuals) or "none"
            raise ValueError(
                f"block {block} produced no visual (blocks with visuals: {known})"
            )
        selected = matches[0]
    else:
        if not loaded.visuals:
            detail = ""
            if loaded.errors:
                first = loaded.errors[0]["error"]
                detail = f" ({first.get('type')}: {first.get('message')})"
            raise ValueError(f"no visuals produced by {target}{detail}")
        selected = loaded.visuals[-1]
    return selected["data"], selected["buffers"], selected["block"]


class RenderSession:
    """Render several visuals sequentially through one shared StudioContext.

    The context (Chrome tab + HTTP server + JS bundle load) is opened lazily
    on the first render and reused for subsequent visuals; the viewport is
    reset before every render, so output matches a fresh context.
    """

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        dpr: float = DEFAULT_DPR,
        debug: bool = False,
        ready_timeout: Optional[float] = DEFAULT_READY_TIMEOUT,
    ) -> None:
        self._width = width
        self._dpr = dpr
        self._debug = debug
        self._ready_timeout = ready_timeout
        self._studio: Optional[StudioContext] = None

    def __enter__(self) -> "RenderSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying context (no-op if nothing was rendered)."""
        if self._studio is not None:
            self._studio.stop()
            self._studio = None

    def render(
        self,
        data: Dict[str, Any],
        buffers: List[bytes],
        height: Optional[int] = None,
    ) -> Tuple[bytes, int, int]:
        """Render one visual, waiting for readiness.

        Args:
            data: Visual JSON envelope.
            buffers: Binary buffers.
            height: CSS-pixel viewport height; None measures the rendered
                content (deterministic for a given input).

        Returns:
            Tuple of (png bytes, pixel width, pixel height).
        """
        if self._studio is None:
            self._studio = StudioContext(
                width=self._width,
                height=height,
                scale=self._dpr,
                debug=self._debug,
                ready_timeout=self._ready_timeout,
                reuse=True,
                keep_alive=1.0,
            )
            self._studio.start()
        studio = self._studio
        # Reset the viewport: a previous render's measure pass may have
        # resized it. Matches what a fresh context applies before loading.
        studio.set_size(width=self._width, height=height or self._width)
        studio.load_plot(data=data, buffers=buffers, measure=height is None)
        png = studio.capture_bytes(format="png")
        pixel_width = int(round(studio.width * studio.scale))
        pixel_height = int(round((studio.height or studio.width) * studio.scale))
        return png, pixel_width, pixel_height


def render_png(
    data: Dict[str, Any],
    buffers: List[bytes],
    width: int = DEFAULT_WIDTH,
    height: Optional[int] = None,
    dpr: float = DEFAULT_DPR,
    debug: bool = False,
    ready_timeout: Optional[float] = DEFAULT_READY_TIMEOUT,
) -> Tuple[bytes, int, int]:
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
        Tuple of (png bytes, pixel width, pixel height).
    """
    with RenderSession(
        width=width, dpr=dpr, debug=debug, ready_timeout=ready_timeout
    ) as session:
        return session.render(data, buffers, height=height)


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
    png, pixel_width, pixel_height = render_png(
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
        # Intentionally a second fresh session: the determinism claim is
        # about independent renders, not renders sharing a tab.
        png_recheck, _w, _h = render_png(
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
    "RenderSession",
    "render_png",
    "resolve_visual",
    "screenshot_target",
]
