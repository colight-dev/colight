"""Deterministic screenshots of visuals via the existing headless renderer.

Builds on the same rendering path as ``colight render`` (StudioContext /
headless Chrome), adding agent-facing ergonomics and a determinism pass:
fixed viewport and device-pixel-ratio, wait-for-render-complete, no update
entries applied (render at t=0 — animated state stays at its initial
values), and an optional double-render byte-hash check.
"""

import contextlib
import hashlib
import pathlib
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

from colight.screenshots import StudioContext

from . import targets

DEFAULT_WIDTH = 800
DEFAULT_DPR = 1.0
DEFAULT_READY_TIMEOUT = 30.0


class SceneLike(Protocol):
    """A loaded, render-ready visual (fresh tab or daemon warm scene)."""

    @property
    def studio(self) -> StudioContext: ...

    def capture(self) -> Tuple[bytes, int, int]: ...

    def mark_mutated(self) -> None:
        """Flag that the loaded scene is about to be mutated (camera framing,
        highlight decorations, ...) so a caching provider reloads it before
        the next reuse. No-op for single-use sessions."""
        ...


class SceneSource(Protocol):
    """Provides loaded scenes for one resolved target visual.

    The direct implementation (:class:`DirectSceneSource`) resolves the
    target once and loads it into a fresh render session per ``scene()``
    call; the daemon's implementation may serve an already-loaded warm
    scene instead. ``fresh=True`` demands a genuinely independent render
    (used by ``screenshot --check``).
    """

    @property
    def block_id(self) -> Optional[str]: ...

    def scene(
        self, fresh: bool = False
    ) -> "contextlib.AbstractContextManager[SceneLike]": ...


def load_visual(
    studio: StudioContext,
    data: Dict[str, Any],
    buffers: List[bytes],
    width: int,
    height: Optional[int],
) -> None:
    """Load one visual into a studio, resetting the viewport first.

    A previous render's measure pass may have resized the viewport; this
    matches what a fresh context applies before loading.
    """
    studio.set_size(width=width, height=height or width)
    studio.load_plot(data=data, buffers=buffers, measure=height is None)


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
        chrome_port: Optional[int] = None,
    ) -> None:
        self._width = width
        self._dpr = dpr
        self._debug = debug
        self._ready_timeout = ready_timeout
        self._chrome_port = chrome_port
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

    @property
    def studio(self) -> StudioContext:
        """The live StudioContext (raises if nothing was loaded yet)."""
        if self._studio is None:
            raise RuntimeError("RenderSession has not loaded a visual yet")
        return self._studio

    def load(
        self,
        data: Dict[str, Any],
        buffers: List[bytes],
        height: Optional[int] = None,
    ) -> None:
        """Load one visual and wait for render readiness (no capture).

        Args:
            data: Visual JSON envelope.
            buffers: Binary buffers.
            height: CSS-pixel viewport height; None measures the rendered
                content (deterministic for a given input).
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
                port=self._chrome_port,
            )
            self._studio.start()
        load_visual(self._studio, data, buffers, self._width, height)

    def mark_mutated(self) -> None:
        """No-op: a RenderSession's scene is never reused across requests."""

    def capture(self) -> Tuple[bytes, int, int]:
        """Capture the loaded visual as PNG.

        Returns:
            Tuple of (png bytes, pixel width, pixel height).
        """
        studio = self.studio
        png = studio.capture_bytes(format="png")
        pixel_width = int(round(studio.width * studio.scale))
        pixel_height = int(round((studio.height or studio.width) * studio.scale))
        return png, pixel_width, pixel_height

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
        self.load(data, buffers, height=height)
        return self.capture()


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


class DirectSceneSource:
    """Scene source that resolves a target once and renders it in fresh tabs.

    This is the direct (non-daemon) implementation of :class:`SceneSource`:
    every ``scene()`` call opens a fresh render session, exactly like the
    pre-daemon CLI did.
    """

    def __init__(
        self,
        target: pathlib.Path,
        block: Optional[str] = None,
        width: int = DEFAULT_WIDTH,
        height: Optional[int] = None,
        dpr: float = DEFAULT_DPR,
        debug: bool = False,
        ready_timeout: Optional[float] = DEFAULT_READY_TIMEOUT,
    ) -> None:
        self._data, self._buffers, self._block_id = resolve_visual(target, block)
        self._width = width
        self._height = height
        self._dpr = dpr
        self._debug = debug
        self._ready_timeout = ready_timeout

    @property
    def block_id(self) -> Optional[str]:
        return self._block_id

    @contextlib.contextmanager
    def scene(self, fresh: bool = False) -> Iterator[SceneLike]:
        with RenderSession(
            width=self._width,
            dpr=self._dpr,
            debug=self._debug,
            ready_timeout=self._ready_timeout,
        ) as session:
            session.load(self._data, self._buffers, height=self._height)
            yield session


def _capture_scene(
    scene: SceneLike, frame: Optional[str], want_coverage: bool
) -> Tuple[bytes, int, int, Dict[str, Any]]:
    """Capture a loaded scene (optionally camera-framed) and gather extras.

    Returns:
        Tuple of (png bytes, pixel width, pixel height, extras) where
        extras may carry ``frame`` (fitted camera + selection) and
        ``coverage`` (scene3d targets only).
    """
    from . import scene_pick

    extras: Dict[str, Any] = {}
    is_scene = False
    if frame is not None or want_coverage:
        is_scene = scene_pick.scene_count(scene.studio) > 0
    if frame is not None:
        if not is_scene:
            raise ValueError(
                "--frame requires a scene3d visual "
                "(target's visual contains no scene3d scene)"
            )
        selector, ranges = scene_pick.parse_frame_selector(frame)
        snapshot = scene_pick.take_snapshot(scene.studio)
        resolved = scene_pick.resolve_component(snapshot.components, selector)
        # Framing moves the scene camera: a caching provider must reload
        # this scene before serving it again.
        scene.mark_mutated()
        camera = scene_pick.frame_selection(scene.studio, resolved["component"], ranges)
        extras["frame"] = {
            "component": resolved["component"],
            "type": resolved["type"],
            "camera": camera,
        }
        if ranges:
            extras["frame"]["instances"] = [list(pair) for pair in ranges]

    png, pixel_width, pixel_height = scene.capture()

    if want_coverage and is_scene:
        snapshot = scene_pick.take_snapshot(scene.studio)
        extras["coverage"] = scene_pick.coverage_payload(snapshot)
    return png, pixel_width, pixel_height, extras


def screenshot_source(
    source: SceneSource,
    target_label: str,
    out: pathlib.Path,
    dpr: float = DEFAULT_DPR,
    check: bool = False,
    frame: Optional[str] = None,
    out_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Screenshot an already-resolved scene source (shared CLI/daemon core).

    Args:
        source: Scene source (direct or daemon-backed).
        target_label: Target path string for the payload.
        out: Output PNG path (parent dirs are created).
        dpr: Device pixel ratio (reported in the payload).
        check: Render twice (second render fresh) and byte-compare.
        frame: Scene3d selection to fit the camera on before capture.
        out_label: Payload ``out`` value (defaults to ``str(out)``).

    Returns:
        The ``colight screenshot`` payload (see :func:`screenshot_target`).
    """
    with source.scene() as scene:
        png, pixel_width, pixel_height, extras = _capture_scene(
            scene, frame=frame, want_coverage=True
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)

    payload: Dict[str, Any] = {
        "target": target_label,
        "out": out_label or str(out),
        "width": pixel_width,
        "height": pixel_height,
        "dpr": dpr,
        "sha256": hashlib.sha256(png).hexdigest(),
    }
    payload.update(extras)
    if source.block_id is not None:
        payload["block"] = source.block_id
    if check:
        # Intentionally a fresh, independent render: the determinism claim
        # is about independent renders, not renders sharing a tab.
        with source.scene(fresh=True) as scene:
            png_recheck, _w, _h, _extras = _capture_scene(
                scene, frame=frame, want_coverage=False
            )
        payload["deterministic"] = png_recheck == png
        if not payload["deterministic"]:
            payload["sha256_recheck"] = hashlib.sha256(png_recheck).hexdigest()
    return payload


def screenshot_target(
    target: pathlib.Path,
    out: pathlib.Path,
    block: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: Optional[int] = None,
    dpr: float = DEFAULT_DPR,
    check: bool = False,
    frame: Optional[str] = None,
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
        frame: Scene3d selection ``C[:A-B]`` (component index or type name,
            optional instance ranges) to fit the camera on before capture.
        debug: Verbose renderer logging.
        ready_timeout: Max seconds to wait for render readiness.

    Returns:
        Payload with ``target``, ``out``, ``width``/``height`` (actual PNG
        pixels), ``dpr``, ``block`` (when a .py block was selected),
        ``sha256``, ``frame`` (fitted camera, with ``--frame``),
        ``coverage`` (scene3d targets only: per-component pixel fractions),
        and ``deterministic`` (only when ``check`` is set;
        ``sha256_recheck`` is added when the two renders differ).
    """
    source = DirectSceneSource(
        target,
        block=block,
        width=width,
        height=height,
        dpr=dpr,
        debug=debug,
        ready_timeout=ready_timeout,
    )
    return screenshot_source(
        source, str(target), out, dpr=dpr, check=check, frame=frame
    )


__all__ = [
    "DEFAULT_DPR",
    "DEFAULT_READY_TIMEOUT",
    "DEFAULT_WIDTH",
    "DirectSceneSource",
    "RenderSession",
    "SceneLike",
    "SceneSource",
    "load_visual",
    "render_png",
    "resolve_visual",
    "screenshot_source",
    "screenshot_target",
]
