"""Agent-facing scene3d pick queries (pick-at, pick-where, coverage, frame).

Every query re-renders the target through the deterministic screenshot path
and reads the GPU pick buffer in memory via the ``window.colight.scene3d``
bridge (see ``js/scene3d/canvasSnapshot.ts``). Nothing is persisted: the
contract is the queries. The pick-id legend is serialized in the browser
from the same component-offset registry interactive picking decodes with,
so ids can never drift between the CLI and the live scene.

Pixel-coordinate convention: X,Y are CSS pixels in the rendered page
(origin top-left, y down) — the same space as a ``colight screenshot`` PNG
taken with the same --width/--height at dpr 1. The scene canvas's page rect
is reported in every payload so callers can map between the two.
"""

import base64
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from colight.screenshots import StudioContext

from . import screenshot_tools

DEFAULT_RADIUS = 6
MAX_DEREFERENCED_HITS = 8

InstanceRanges = List[Tuple[int, int]]


def parse_instance_ranges(text: str) -> InstanceRanges:
    """Parse instance ranges like ``"0-3,7,10-12"`` (inclusive bounds).

    Args:
        text: Comma-separated instance indices or ``A-B`` ranges.

    Returns:
        List of (start, end) inclusive tuples.

    Raises:
        ValueError: On malformed or descending ranges.
    """
    ranges: InstanceRanges = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(part)
        if start < 0 or end < start:
            raise ValueError(f"invalid instance range: {part}")
        ranges.append((start, end))
    if not ranges:
        raise ValueError(f"no instance ranges in: {text!r}")
    return ranges


def parse_frame_selector(text: str) -> Tuple[str, Optional[InstanceRanges]]:
    """Parse a ``--frame`` selector ``C[:A-B[,C-D]]``.

    ``C`` is a component index or type name; the optional suffix restricts
    the selection to instance ranges.
    """
    if ":" in text:
        component, ranges_text = text.split(":", 1)
        return component.strip(), parse_instance_ranges(ranges_text)
    return text.strip(), None


def resolve_component(
    components: Sequence[Dict[str, Any]], selector: str
) -> Dict[str, Any]:
    """Resolve a component selector (index or type name) against a summary.

    Args:
        components: Component summaries with ``component`` (index) and
            ``type`` keys (as reported by the scene legend / info).
        selector: Integer index or type name (case-insensitive).

    Returns:
        The matching component summary.

    Raises:
        ValueError: Unknown selector or ambiguous type name.
    """
    known = ", ".join(f"{c['component']}:{c['type']}" for c in components) or "none"
    try:
        index = int(selector)
    except ValueError:
        matches = [c for c in components if c["type"].lower() == selector.lower()]
        if not matches:
            raise ValueError(f"no component of type {selector!r} (components: {known})")
        if len(matches) > 1:
            indices = ", ".join(str(c["component"]) for c in matches)
            raise ValueError(
                f"type {selector!r} matches several components "
                f"(indices {indices}); select one by index"
            )
        return matches[0]
    for component in components:
        if component["component"] == index:
            return component
    raise ValueError(f"no component with index {index} (components: {known})")


@dataclass
class SceneSnapshot:
    """Full-frame pick readback for one scene.

    Attributes:
        ids: (height, width) uint32 array of raw pick ids (0 = background;
            otherwise instance id = ids - 1, decoded via ``legend``).
        legend: Pick-id ranges per component
            ({component, type, count, idBase}).
        width/height: Buffer dimensions in device pixels.
        rect: Canvas rect in page CSS pixels ({left, top, width, height}).
        dpr: Device pixel ratio (device px = CSS px * dpr).
        scenes: Number of scene3d scenes in the visual.
        components: Component summaries ({component, type, count}).
        camera: Camera parameters the buffer was rendered with.
    """

    ids: np.ndarray
    legend: List[Dict[str, Any]]
    width: int
    height: int
    rect: Dict[str, float]
    dpr: float
    scenes: int
    components: List[Dict[str, Any]]
    camera: Optional[Dict[str, Any]]


def _api_call(
    studio: StudioContext, expression: str, await_promise: bool = False
) -> Any:
    """Call the scene3d agent API, raising ``ValueError`` on {'error'}."""
    guarded = (
        "(() => { if (!window.colight || !window.colight.scene3d) "
        "return { error: 'scene3d agent API unavailable (bundle too old?)' }; "
        f"return {expression}; }})()"
    )
    result = studio.evaluate(guarded, await_promise=await_promise)
    if isinstance(result, dict) and "error" in result:
        raise ValueError(result["error"])
    return result


def scene_count(studio: StudioContext) -> int:
    """Number of mounted scene3d scenes in the loaded visual."""
    count = studio.evaluate(
        "(window.colight && window.colight.scene3d) "
        "? window.colight.scene3d.count() : 0"
    )
    return int(count or 0)


def require_scene(studio: StudioContext) -> int:
    """Assert the loaded visual contains a scene3d scene.

    Returns:
        The scene count.

    Raises:
        ValueError: When the visual contains no scene3d scene.
    """
    count = scene_count(studio)
    if count == 0:
        raise ValueError(
            "target's visual contains no scene3d scene "
            "(pick queries are scene3d-only)"
        )
    return count


def _ranges_json(ranges: Optional[InstanceRanges]) -> str:
    if not ranges:
        return "null"
    return json.dumps([[start, end] for start, end in ranges])


def take_snapshot(
    studio: StudioContext,
    component: Optional[int] = None,
    instances: Optional[InstanceRanges] = None,
) -> SceneSnapshot:
    """Render the pick pass full-frame and read back the id buffer.

    Args:
        studio: Studio context with a scene3d visual loaded and ready.
        component: When given, draw ONLY this component (optionally only
            ``instances``) — the selection's unoccluded footprint.
        instances: Inclusive instance ranges (with ``component``).

    Returns:
        The decoded snapshot.
    """
    options = (
        f"{{scene: 0, component: {json.dumps(component)}, "
        f"instances: {_ranges_json(instances)}}}"
    )
    payload = _api_call(
        studio,
        f"window.colight.scene3d.snapshot({options})",
        await_promise=True,
    )
    ids = np.frombuffer(base64.b64decode(payload["ids"]), dtype="<u4").reshape(
        payload["height"], payload["width"]
    )
    scene = payload.get("scene") or {}
    return SceneSnapshot(
        ids=ids,
        legend=payload["legend"],
        width=payload["width"],
        height=payload["height"],
        rect=scene.get("rect") or {"left": 0.0, "top": 0.0, "width": 0, "height": 0},
        dpr=float(scene.get("dpr") or 1.0),
        scenes=int(payload.get("scenes") or 1),
        components=scene.get("components") or [],
        camera=scene.get("camera"),
    )


def instance_values(
    studio: StudioContext, component: int, instance: int
) -> Dict[str, Any]:
    """Dereferenced attribute values (center/color/size/...) for an instance.

    Values are read from the compiled component configs the renderer
    consumes, so they reflect what is actually drawn.
    """
    payload = _api_call(
        studio,
        "window.colight.scene3d.instanceInfo("
        f"{{component: {component}, instance: {instance}}})",
    )
    return payload["values"]


def frame_selection(
    studio: StudioContext,
    component: Optional[int],
    instances: Optional[InstanceRanges] = None,
) -> Dict[str, Any]:
    """Fit the scene camera on a selection and re-render.

    Args:
        studio: Studio context with a scene3d visual loaded.
        component: Component index (None = frame the whole scene).
        instances: Inclusive instance ranges.

    Returns:
        The fitted camera parameters.
    """
    payload = _api_call(
        studio,
        "window.colight.scene3d.frame("
        f"{{component: {json.dumps(component)}, "
        f"instances: {_ranges_json(instances)}}})",
        await_promise=True,
    )
    return payload["camera"]


def highlight_selection(
    studio: StudioContext,
    component: int,
    instances: Optional[InstanceRanges] = None,
) -> None:
    """Re-render with the selection highlighted via instance decorations."""
    _api_call(
        studio,
        "window.colight.scene3d.highlight("
        f"{{component: {component}, instances: {_ranges_json(instances)}}})",
        await_promise=True,
    )


def clear_highlight(studio: StudioContext) -> None:
    """Restore the un-highlighted scene."""
    _api_call(
        studio,
        "window.colight.scene3d.clearHighlight()",
        await_promise=True,
    )


# ========== Buffer analysis (pure numpy; unit-testable) ==========


def component_mask(snapshot: SceneSnapshot, entry: Dict[str, Any]) -> np.ndarray:
    """Boolean mask of pixels belonging to one legend entry."""
    element = snapshot.ids.astype(np.int64) - 1
    return (
        (snapshot.ids != 0)
        & (element >= entry["idBase"])
        & (element < entry["idBase"] + entry["count"])
    )


def selection_mask(
    snapshot: SceneSnapshot,
    entry: Dict[str, Any],
    instances: Optional[InstanceRanges] = None,
) -> np.ndarray:
    """Boolean mask of pixels belonging to selected instances of an entry."""
    mask = component_mask(snapshot, entry)
    if not instances:
        return mask
    element = snapshot.ids.astype(np.int64) - 1 - entry["idBase"]
    in_ranges = np.zeros_like(mask)
    for start, end in instances:
        in_ranges |= (element >= start) & (element <= end)
    return mask & in_ranges


def legend_entry(snapshot: SceneSnapshot, component: int) -> Dict[str, Any]:
    """The legend entry for a component index.

    Raises:
        ValueError: When the component contributed no pickable elements.
    """
    for entry in snapshot.legend:
        if entry["component"] == component:
            return entry
    raise ValueError(f"component {component} has no pickable elements")


def coverage_payload(snapshot: SceneSnapshot) -> Dict[str, Any]:
    """Per-component pixel coverage of the scene canvas.

    Returns:
        ``{"width", "height", "rect", "scenes", "components": [{"component",
        "type", "instances", "pixels", "fraction"}], "background":
        {"pixels", "fraction"}}`` — fractions are of the canvas pixel count.
    """
    total = int(snapshot.ids.size) or 1
    components = []
    for entry in snapshot.legend:
        pixels = int(component_mask(snapshot, entry).sum())
        components.append(
            {
                "component": entry["component"],
                "type": entry["type"],
                "instances": entry["count"],
                "pixels": pixels,
                "fraction": round(pixels / total, 6),
            }
        )
    background = int((snapshot.ids == 0).sum())
    payload: Dict[str, Any] = {
        "width": snapshot.width,
        "height": snapshot.height,
        "rect": snapshot.rect,
        "components": components,
        "background": {
            "pixels": background,
            "fraction": round(background / total, 6),
        },
    }
    if snapshot.scenes > 1:
        payload["scenes"] = snapshot.scenes
    return payload


def _page_to_device(snapshot: SceneSnapshot, x: float, y: float) -> Tuple[float, float]:
    """Page CSS coordinates -> pick-buffer device pixels."""
    return (
        (x - snapshot.rect["left"]) * snapshot.dpr,
        (y - snapshot.rect["top"]) * snapshot.dpr,
    )


def _device_to_page(snapshot: SceneSnapshot, x: float, y: float) -> Tuple[float, float]:
    """Pick-buffer device pixels -> page CSS coordinates."""
    return (
        x / snapshot.dpr + snapshot.rect["left"],
        y / snapshot.dpr + snapshot.rect["top"],
    )


def hits_at(
    snapshot: SceneSnapshot, x: float, y: float, radius: float
) -> List[Dict[str, Any]]:
    """Ranked pick hits within a radius disc around a page CSS point.

    Args:
        snapshot: A full-frame snapshot.
        x: Page CSS x (origin top-left).
        y: Page CSS y (y down).
        radius: Disc radius in CSS pixels.

    Returns:
        Hits sorted by distance then coverage, each with ``component``,
        ``type``, ``instance``, ``distance`` (CSS px from the query point),
        ``pixels`` and ``share`` (fraction of the sampled disc).
    """
    cx, cy = _page_to_device(snapshot, x, y)
    device_radius = radius * snapshot.dpr
    x0 = max(0, int(math.floor(cx - device_radius)))
    x1 = min(snapshot.width - 1, int(math.ceil(cx + device_radius)))
    y0 = max(0, int(math.floor(cy - device_radius)))
    y1 = min(snapshot.height - 1, int(math.ceil(cy + device_radius)))
    if x0 > x1 or y0 > y1:
        return []

    ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    distances = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy)
    disc = distances <= device_radius
    disc_pixels = int(disc.sum())
    if disc_pixels == 0:
        return []
    ids = snapshot.ids[y0 : y1 + 1, x0 : x1 + 1]

    hits: List[Dict[str, Any]] = []
    for entry in snapshot.legend:
        element = ids.astype(np.int64) - 1 - entry["idBase"]
        in_entry = (ids != 0) & (element >= 0) & (element < entry["count"]) & disc
        if not in_entry.any():
            continue
        for instance in np.unique(element[in_entry]):
            mask = in_entry & (element == instance)
            pixels = int(mask.sum())
            hits.append(
                {
                    "component": entry["component"],
                    "type": entry["type"],
                    "instance": int(instance),
                    "distance": round(float(distances[mask].min()) / snapshot.dpr, 2),
                    "pixels": pixels,
                    "share": round(pixels / disc_pixels, 4),
                }
            )
    hits.sort(key=lambda hit: (hit["distance"], -hit["pixels"]))
    return hits


def selection_metrics(
    full: SceneSnapshot,
    solo: SceneSnapshot,
    entry: Dict[str, Any],
    instances: Optional[InstanceRanges] = None,
) -> Dict[str, Any]:
    """Screen-truth metrics for an instance selection.

    Args:
        full: Snapshot of the whole scene (occlusion applies).
        solo: Snapshot with only the selection drawn (its unoccluded
            projected footprint).
        entry: The selection's legend entry.
        instances: Inclusive instance ranges (None = all instances).

    Returns:
        ``{"selected", "visible_pixels", "projected_pixels", "visibility",
        "visible_instances", "hidden_instances", "bbox"?, "centroid"?,
        "projected_bbox"?}`` — bbox/centroid in page CSS pixels;
        ``visibility`` = visible / projected (occlusion-free) pixels.
    """
    visible = selection_mask(full, entry, instances)
    projected = selection_mask(solo, entry, instances)
    visible_pixels = int(visible.sum())
    projected_pixels = int(projected.sum())

    if instances:
        selected = sum(
            min(end, entry["count"] - 1) - start + 1
            for start, end in instances
            if start < entry["count"]
        )
    else:
        selected = entry["count"]

    element = full.ids.astype(np.int64) - 1 - entry["idBase"]
    visible_instances = np.unique(element[visible]) if visible_pixels else []

    payload: Dict[str, Any] = {
        "selected": selected,
        "visible_pixels": visible_pixels,
        "projected_pixels": projected_pixels,
        "visibility": round(visible_pixels / projected_pixels, 4)
        if projected_pixels
        else 0.0,
        "visible_instances": int(len(visible_instances)),
        "hidden_instances": selected - int(len(visible_instances)),
    }

    def bbox_of(mask: np.ndarray, snapshot: SceneSnapshot) -> List[float]:
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        x0, y0 = _device_to_page(snapshot, float(cols[0]), float(rows[0]))
        x1, y1 = _device_to_page(snapshot, float(cols[-1]) + 1, float(rows[-1]) + 1)
        return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]

    if visible_pixels:
        payload["bbox"] = bbox_of(visible, full)
        ys, xs = np.nonzero(visible)
        centroid_x, centroid_y = _device_to_page(
            full, float(xs.mean()) + 0.5, float(ys.mean()) + 0.5
        )
        payload["centroid"] = [round(centroid_x, 1), round(centroid_y, 1)]
    elif projected_pixels:
        payload["projected_bbox"] = bbox_of(projected, solo)
    return payload


# ========== Command implementations ==========


def pick_at_source(
    source: "screenshot_tools.SceneSource",
    target_label: str,
    x: float,
    y: float,
    radius: float = DEFAULT_RADIUS,
) -> Dict[str, Any]:
    """Pick hits around a point on an already-resolved scene source.

    Shared CLI/daemon core of :func:`pick_at_target`.
    """
    with source.scene() as scene:
        studio = scene.studio
        require_scene(studio)
        snapshot = take_snapshot(studio)
        hits = hits_at(snapshot, x, y, radius)
        for hit in hits[:MAX_DEREFERENCED_HITS]:
            hit["values"] = instance_values(studio, hit["component"], hit["instance"])

        cx, cy = _page_to_device(snapshot, x, y)
        device_radius = radius * snapshot.dpr
        x0 = max(0, int(math.floor(cx - device_radius)))
        x1 = min(snapshot.width - 1, int(math.ceil(cx + device_radius)))
        y0 = max(0, int(math.floor(cy - device_radius)))
        y1 = min(snapshot.height - 1, int(math.ceil(cy + device_radius)))
        background_share = 1.0
        if x0 <= x1 and y0 <= y1:
            ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
            disc = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy) <= device_radius
            if disc.any():
                ids = snapshot.ids[y0 : y1 + 1, x0 : x1 + 1]
                background_share = round(
                    float(((ids == 0) & disc).sum() / disc.sum()), 4
                )

        payload: Dict[str, Any] = {
            "target": target_label,
            "x": x,
            "y": y,
            "radius": radius,
            "scene": {
                "rect": snapshot.rect,
                "width": snapshot.width,
                "height": snapshot.height,
                "dpr": snapshot.dpr,
            },
            "hits": hits,
            "background_share": background_share,
        }
        if snapshot.scenes > 1:
            payload["scene"]["scenes"] = snapshot.scenes
        if source.block_id is not None:
            payload["block"] = source.block_id
        return payload


def pick_at_target(
    target: pathlib.Path,
    x: float,
    y: float,
    radius: float = DEFAULT_RADIUS,
    block: Optional[str] = None,
    width: int = 800,
    height: Optional[int] = None,
    dpr: float = 1.0,
    debug: bool = False,
    ready_timeout: Optional[float] = 30.0,
) -> Dict[str, Any]:
    """Re-render TARGET and report ranked pick hits around a point.

    Coordinates are page CSS pixels (origin top-left, y down) — the same
    space as a ``colight screenshot`` PNG at the same --width/--height and
    dpr 1.

    Returns:
        Payload with ``target``, ``block``?, ``x``/``y``/``radius``,
        ``scene`` ({rect, width, height, dpr, scenes?}), ``hits`` (ranked;
        the top hits carry dereferenced ``values``) and
        ``background_share``.
    """
    source = screenshot_tools.DirectSceneSource(
        target,
        block=block,
        width=width,
        height=height,
        dpr=dpr,
        debug=debug,
        ready_timeout=ready_timeout,
    )
    return pick_at_source(source, str(target), x, y, radius=radius)


def pick_where_source(
    source: "screenshot_tools.SceneSource",
    target_label: str,
    component_selector: str,
    instances: Optional[InstanceRanges] = None,
    out: Optional[pathlib.Path] = None,
    out_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Selection -> screen truth on an already-resolved scene source.

    Shared CLI/daemon core of :func:`pick_where_target`.
    """
    with source.scene() as scene:
        studio = scene.studio
        require_scene(studio)
        full = take_snapshot(studio)
        resolved = resolve_component(full.components, component_selector)
        component = resolved["component"]
        entry = legend_entry(full, component)
        solo = take_snapshot(studio, component=component, instances=instances)
        metrics = selection_metrics(full, solo, entry, instances)

        payload: Dict[str, Any] = {
            "target": target_label,
            "component": component,
            "type": entry["type"],
            "instances": [[start, end] for start, end in instances]
            if instances
            else "all",
            "scene": {
                "rect": full.rect,
                "width": full.width,
                "height": full.height,
                "dpr": full.dpr,
            },
            **metrics,
        }
        if source.block_id is not None:
            payload["block"] = source.block_id

        if out is not None:
            # Highlight decorations mutate the loaded scene; a caching
            # provider must reload it before serving it again (the explicit
            # clear below restores this render, but the conservative marking
            # keeps warm reuse byte-honest even if a capture fails midway).
            scene.mark_mutated()
            highlight_selection(studio, component, instances)
            png, _w, _h = scene.capture()
            clear_highlight(studio)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(png)
            payload["out"] = out_label or str(out)
        return payload


def pick_where_target(
    target: pathlib.Path,
    component_selector: str,
    instances: Optional[InstanceRanges] = None,
    out: Optional[pathlib.Path] = None,
    block: Optional[str] = None,
    width: int = 800,
    height: Optional[int] = None,
    dpr: float = 1.0,
    debug: bool = False,
    ready_timeout: Optional[float] = 30.0,
) -> Dict[str, Any]:
    """Re-render TARGET and report where a selection lands on screen.

    Returns:
        Payload with ``target``, ``block``?, ``component``/``type``,
        ``instances``, the metrics from :func:`selection_metrics`, the
        ``scene`` rect, and ``out`` when an overlay PNG was written.
    """
    source = screenshot_tools.DirectSceneSource(
        target,
        block=block,
        width=width,
        height=height,
        dpr=dpr,
        debug=debug,
        ready_timeout=ready_timeout,
    )
    return pick_where_source(
        source,
        str(target),
        component_selector,
        instances=instances,
        out=out,
    )


__all__ = [
    "DEFAULT_RADIUS",
    "SceneSnapshot",
    "clear_highlight",
    "component_mask",
    "coverage_payload",
    "frame_selection",
    "highlight_selection",
    "hits_at",
    "instance_values",
    "legend_entry",
    "parse_frame_selector",
    "parse_instance_ranges",
    "pick_at_source",
    "pick_at_target",
    "pick_where_source",
    "pick_where_target",
    "require_scene",
    "resolve_component",
    "scene_count",
    "selection_mask",
    "selection_metrics",
    "take_snapshot",
]
