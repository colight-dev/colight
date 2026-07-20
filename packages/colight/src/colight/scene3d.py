from typing import Any, Dict, Literal, Optional, Sequence, TypedDict, Union

import numpy as np

import colight.plot as Plot
from colight import colormaps
from colight.layout import JSExpr, is_js_expr

# Move Array type definition after imports
ArrayLike = Union[list, np.ndarray, JSExpr]
NumberLike = Union[int, float, np.number, JSExpr]

# Import helpers - these are re-exported below
from colight.helpers import (
    CameraFrustum,
    GridHelper,
    ImageProjection,
    coerce_image_array,
    flatten_array,
)


# =============================================================================
# Drag Constraints
# =============================================================================


class DragConstraint(TypedDict, total=False):
    """Configuration for drag operations on scene components."""

    type: Literal["plane", "axis", "surface", "screen", "free"]
    direction: ArrayLike  # For axis constraint: [x, y, z]
    normal: ArrayLike  # For plane constraint: [x, y, z]
    point: ArrayLike  # Origin point for axis/plane constraints


def drag_axis(
    direction: ArrayLike,
    point: Optional[ArrayLike] = None,
) -> DragConstraint:
    """Create an axis constraint for dragging.

    Constrains drag movement to a single axis line.

    Args:
        direction: Axis direction vector [x, y, z]
        point: Optional origin point for the axis. If not provided,
               uses the instance center at drag start.

    Returns:
        DragConstraint configuration

    Example:
        >>> Cuboid(centers=[[0,0,0]], on_drag=handle_drag, drag_constraint=drag_axis([0, 1, 0]))
    """
    result: DragConstraint = {"type": "axis", "direction": direction}
    if point is not None:
        result["point"] = point
    return result


def drag_plane(
    normal: ArrayLike,
    point: Optional[ArrayLike] = None,
) -> DragConstraint:
    """Create a plane constraint for dragging.

    Constrains drag movement to a plane.

    Args:
        normal: Plane normal vector [x, y, z]
        point: Optional point on the plane. If not provided,
               uses the instance center at drag start.

    Returns:
        DragConstraint configuration

    Example:
        >>> # Drag on the ground plane (XZ plane at y=0)
        >>> Cuboid(centers=[[0,0,0]], on_drag=handle_drag, drag_constraint=drag_plane([0, 1, 0], [0, 0, 0]))
    """
    result: DragConstraint = {"type": "plane", "normal": normal}
    if point is not None:
        result["point"] = point
    return result


# Convenience constants for common axis constraints
DRAG_AXIS_X: DragConstraint = {"type": "axis", "direction": [1, 0, 0]}
DRAG_AXIS_Y: DragConstraint = {"type": "axis", "direction": [0, 1, 0]}
DRAG_AXIS_Z: DragConstraint = {"type": "axis", "direction": [0, 0, 1]}

# Convenience constants for common plane constraints
DRAG_PLANE_XY: DragConstraint = {"type": "plane", "normal": [0, 0, 1]}
DRAG_PLANE_XZ: DragConstraint = {"type": "plane", "normal": [0, 1, 0]}
DRAG_PLANE_YZ: DragConstraint = {"type": "plane", "normal": [1, 0, 0]}

# Other drag constraint modes
DRAG_SURFACE: DragConstraint = {"type": "surface"}  # Drag along hit surface (default)
DRAG_SCREEN: DragConstraint = {"type": "screen"}  # Screen-space drag (fixed depth)
DRAG_FREE: DragConstraint = {"type": "free"}  # Camera-facing plane through start point


class ColorBy(TypedDict, total=False):
    """Colormap-driven coloring for instanced primitives.

    ``values`` is required; everything else has defaults. Colors are
    computed in Python via :mod:`colight.colormaps`; the colormap spec
    (cmap, domain, label, ...) travels with the component so the scene can
    render a legend and ``colight inspect`` / ``screenshot --json`` can
    report what the colors encode.
    """

    values: ArrayLike  # scalar values (continuous) or category codes
    cmap: str  # colormap name, default "viridis"
    domain: Sequence[float]  # (min, max) for continuous maps
    label: str  # what the colors encode (legend title)
    # Categorical display: either ordinal display names (["ore", "waste"]) or a
    # first-class category table ([{value, label, color?}]) — the xmi id-maps
    # idiom with arbitrary codes, per-category colors, and a fallback slot.
    categories: Sequence[Any]
    fallback: Dict[str, Any]  # {label?, color?} for unmatched/NaN (category table)
    palette: str  # categorical palette for auto-assigned colors (default tab10)
    nan_color: Sequence[float]  # RGB for NaN/invalid values
    legend: Union[bool, str]  # False hides; or a dock corner like "top-left"


class ColorChannel(TypedDict, total=False):
    """One named color channel (a ``color_by``-shaped spec).

    Channels ship their raw ``values`` once; the active channel is colorized
    client-side (JS applies a LUT / category table), so an artifact can switch
    which attribute drives the colors without re-exporting.
    """

    values: ArrayLike
    cmap: str
    domain: Sequence[float]
    label: str
    categories: Sequence[Any]
    fallback: Dict[str, Any]
    palette: str
    nan_color: Sequence[float]


_LEGEND_POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


def _apply_color_by(
    data: Dict[str, Any],
    color_by: Optional[Union[ColorBy, Dict[str, Any]]],
) -> None:
    """Resolve a ``color_by`` spec into per-instance colors + legend metadata.

    Mutates ``data``: sets ``colors`` (float32 RGB) and ``color_by`` (the
    JSON-safe colormap spec consumed by the JS legend and CLI reporting).

    Raises:
        ValueError: When explicit ``colors``/``color`` are also present, or
            the spec is malformed.
    """
    if color_by is None:
        return
    if "colors" in data or "color" in data:
        raise ValueError("pass either color_by or colors/color, not both")
    spec: Dict[str, Any] = dict(color_by)
    legend: Union[bool, str] = spec.pop("legend", True)
    if isinstance(legend, str) and legend not in _LEGEND_POSITIONS:
        raise ValueError(
            f"invalid legend position {legend!r} (one of {', '.join(_LEGEND_POSITIONS)})"
        )
    colors, meta = colormaps.resolve_color_by(spec)
    if legend is False:
        meta["legend"] = False
    elif isinstance(legend, str):
        meta["position"] = legend
    data["colors"] = flatten_array(colors, dtype=np.float32)
    data["color_by"] = meta


def _apply_color_channels(
    data: Dict[str, Any],
    color_channels: Optional[Dict[str, Any]],
    active_channel: Optional[Union[str, JSExpr]],
    legend: Union[bool, str] = True,
) -> None:
    """Resolve named ``color_channels`` for client-side switching.

    Each channel ships its raw ``values`` once plus a compact colorizer (a
    256-entry RGB LUT for continuous, or the resolved category table for
    categorical) so JS recolors the active channel without reimplementing
    colormaps. Python also bakes the initially-active channel's colors so the
    first render (and non-JS paths) are correct.

    Mutates ``data``: sets ``colors`` (baked active channel), ``color_by`` (the
    active channel's legend), ``color_channels`` (per-channel colorizer +
    legend + values), and ``active_channel`` (literal name or the JSExpr JS
    resolves against ``$state``).

    Args:
        active_channel: A literal channel name, a ``Plot.js("$state...")``
            expression, or None (defaults to the first channel).
        legend: False hides the legend; a corner string docks it.

    Raises:
        ValueError: When ``color_by``/``colors``/``color`` are also present,
            channels are empty, or a literal ``active_channel`` is unknown.
    """
    if not color_channels:
        return
    if "colors" in data or "color" in data or "color_by" in data:
        raise ValueError(
            "color_channels is mutually exclusive with color_by / colors / color"
        )
    if isinstance(legend, str) and legend not in _LEGEND_POSITIONS:
        raise ValueError(
            f"invalid legend position {legend!r} (one of {', '.join(_LEGEND_POSITIONS)})"
        )

    names = list(color_channels.keys())
    if not names:
        raise ValueError("color_channels must declare at least one channel")

    # Which channel to bake for the initial render. A JSExpr active_channel is
    # resolved by JS at runtime; Python bakes a concrete default (the literal
    # if given, else the first channel).
    is_expr = is_js_expr(active_channel)
    default_name: str
    if active_channel is None or is_expr:
        default_name = names[0]
    else:
        default_name = str(active_channel)
        if default_name not in color_channels:
            raise ValueError(
                f"active_channel {default_name!r} not in color_channels " f"{names}"
            )

    channels_meta: Dict[str, Any] = {}
    baked_colors: Optional[np.ndarray] = None
    baked_legend: Optional[Dict[str, Any]] = None
    for name, spec in color_channels.items():
        colors, chan_legend, colorizer = colormaps.resolve_channel(dict(spec))
        n = colors.shape[0]
        channels_meta[name] = {
            "label": chan_legend.get("label", name),
            "legend": chan_legend,
            "colorizer": colorizer,
            "values": flatten_array(
                np.asarray(spec["values"], dtype=np.float32), dtype=np.float32
            ),
            "count": n,
        }
        if name == default_name:
            baked_colors = colors
            baked_legend = dict(chan_legend)

    assert baked_colors is not None and baked_legend is not None
    if legend is False:
        baked_legend["legend"] = False
    elif isinstance(legend, str):
        baked_legend["position"] = legend

    data["colors"] = flatten_array(baked_colors, dtype=np.float32)
    data["color_by"] = baked_legend
    data["color_channels"] = channels_meta
    # JS resolves a JSExpr against $state; a literal (or default) passes through.
    data["active_channel"] = active_channel if is_expr else default_name


class FilterBy(TypedDict, total=False):
    """Per-instance threshold filter for instanced primitives.

    Instances whose scalar ``values`` fall outside ``[min, max]`` are hidden
    (collapsed in the vertex shader and made unpickable). ``NaN`` values are
    always hidden. ``values`` is required; ``min``/``max`` default to unbounded.

    ``values`` is uploaded once as a per-instance attribute; ``min``/``max``
    live in a small per-component uniform, so a threshold change (e.g. a
    ``Plot.js("$state.cutoff")`` slider) updates only the uniform and does not
    re-upload the instance data. ``min``/``max`` may be literals or ``$state``
    references (``Plot.js(...)``).
    """

    values: ArrayLike  # per-instance scalar values
    min: Optional[NumberLike]  # inclusive lower bound (default: unbounded)
    max: Optional[NumberLike]  # inclusive upper bound (default: unbounded)
    label: str  # what the filter encodes (for inspect / screenshot --json)


def _normalize_filter_by(
    filter_by: Optional[Union[FilterBy, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Normalize a ``filter_by`` spec for the JS boundary.

    Flattens ``values`` to a float32 array and passes ``min``/``max``/``label``
    through unchanged (``min``/``max`` may be numbers or ``$state`` JSExprs).

    Raises:
        ValueError: When ``values`` is missing.
    """
    if filter_by is None:
        return None
    spec = dict(filter_by)
    if spec.get("values") is None:
        raise ValueError("filter_by requires 'values'")
    out: Dict[str, Any] = {"values": flatten_array(spec["values"], dtype=np.float32)}
    if spec.get("min") is not None:
        out["min"] = spec["min"]
    if spec.get("max") is not None:
        out["max"] = spec["max"]
    if spec.get("label") is not None:
        out["label"] = spec["label"]
    return out


class Decoration(TypedDict, total=False):
    indexes: ArrayLike
    color: Optional[ArrayLike]  # [r,g,b]
    alpha: Optional[NumberLike]  # 0-1
    scale: Optional[NumberLike]  # scale factor
    outline: Optional[bool]  # enable outline effect
    outline_color: Optional[ArrayLike]  # [r,g,b]
    outline_width: Optional[NumberLike]  # pixels


class HoverProps(TypedDict, total=False):
    """Properties to apply automatically when an instance is hovered."""

    color: Optional[ArrayLike]  # [r,g,b]
    alpha: Optional[NumberLike]  # 0-1
    scale: Optional[NumberLike]  # scale factor
    outline: Optional[bool]  # enable outline effect
    outline_color: Optional[ArrayLike]  # [r,g,b]
    outline_width: Optional[NumberLike]  # pixels


def deco(
    indexes: Union[int, np.integer, ArrayLike],
    *,
    color: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,
    scale: Optional[NumberLike] = None,
    outline: Optional[bool] = None,
    outline_color: Optional[ArrayLike] = None,
    outline_width: Optional[NumberLike] = None,
) -> Decoration:
    """Create a decoration for scene components.

    Args:
        indexes: Single index or list of indices to decorate
        color: Optional RGB color override [r,g,b]
        alpha: Optional opacity value (0-1)
        scale: Optional scale factor
        outline: Optional flag to enable outline effect
        outline_color: Optional outline RGB color [r,g,b]
        outline_width: Optional outline width in pixels

    Returns:
        Dictionary containing decoration settings
    """
    # Convert single index to list
    if isinstance(indexes, (int, np.integer)):
        indexes = np.array([indexes])

    # Create base decoration dict with Any type to avoid type conflicts
    decoration: Dict[str, Any] = {"indexes": indexes}

    # Add optional parameters if provided
    if color is not None:
        decoration["color"] = color
    if alpha is not None:
        decoration["alpha"] = alpha
    if scale is not None:
        decoration["scale"] = scale
    if outline is not None:
        decoration["outline"] = outline
    if outline_color is not None:
        decoration["outline_color"] = outline_color
    if outline_width is not None:
        decoration["outline_width"] = outline_width

    return decoration  # type: ignore


# Props consumed as camelCase by the JS scene3d framework layer (interaction
# callbacks, hover/outline styling, caching keys, render options, and helper
# props). ONLY these keys are renamed at the Python->JS boundary. Everything
# else -- the data props (centers, half_sizes, fill_mode, ...) -- crosses the
# boundary unchanged in snake_case: the JS coercion layer consumes snake_case
# keys and warns about keys it does not recognize.
_JS_PROP_NAMES = {
    # Interaction & framework props (BaseComponentConfig / GroupConfig)
    "on_hover": "onHover",
    "on_click": "onClick",
    "on_drag": "onDrag",
    "on_drag_start": "onDragStart",
    "on_drag_end": "onDragEnd",
    "drag_constraint": "dragConstraint",
    "hover_props": "hoverProps",
    "picking_scale": "pickingScale",
    "outline_color": "outlineColor",
    "outline_width": "outlineWidth",
    "child_defaults": "childDefaults",
    "child_overrides": "childOverrides",
    # Caching / render options
    "image_key": "imageKey",
    "texture_key": "textureKey",
    "geometry_key": "geometryKey",
    "cull_mode": "cullMode",
    # Helper props (GridHelper / CameraFrustum / ImageProjection)
    "center_color": "centerColor",
    "line_width": "lineWidth",
    "show_frustum": "showFrustum",
    "frustum_color": "frustumColor",
}


def _convert_to_js(obj: Any) -> Any:
    """Recursively rename framework props to their JS (camelCase) names.

    Data props are passed through unchanged; the JS side consumes them in
    snake_case and warns loudly about unknown keys.
    """
    if isinstance(obj, SceneComponent):
        # Convert SceneComponent to a config object with type and props
        return {"type": obj.type, **_convert_to_js(obj.props)}
    if isinstance(obj, dict):
        return {_JS_PROP_NAMES.get(k, k): _convert_to_js(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_js(item) for item in obj]
    return obj


class SceneComponent(Plot.LayoutItem):
    """Base class for all 3D scene components."""

    def __init__(self, type_name: str, data: Dict[str, Any], **kwargs):
        super().__init__()
        self.type = type_name
        self.props = {**data, **kwargs}

    def to_js_call(self) -> Any:
        """Convert the element to a JSCall representation."""
        return Plot.JSCall(f"scene3d.{self.type}", [_convert_to_js(self.props)])

    def for_json(self) -> Dict[str, Any]:
        """Convert the element to a JSON-compatible dictionary."""
        return Scene(self).for_json()

    def __add__(
        self, other: Union["SceneComponent", "Scene", Dict[str, Any]]
    ) -> "Scene":
        """Allow combining components with + operator."""
        if isinstance(other, Scene):
            return other + self
        elif isinstance(other, SceneComponent):
            return Scene(self, other)
        elif isinstance(other, dict):
            return Scene(self, other)
        else:
            raise TypeError(f"Cannot add SceneComponent with {type(other)}")

    def __radd__(self, other: Dict[str, Any]) -> "Scene":
        """Allow combining components with + operator when dict is on the left."""
        return Scene(self, other)

    def merge(
        self, new_props: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> "SceneComponent":
        """Return a new SceneComponent with updated properties.

        This method does not modify the current SceneComponent instance. Instead, it creates and returns a *new* SceneComponent instance.
        The new instance's properties are derived by merging the properties of the current instance with the provided `new_props` and `kwargs`.
        If there are any conflicts in property keys, the values in `kwargs` take precedence over `new_props`, and `new_props` take precedence over the original properties of this SceneComponent.

        Args:
            new_props: An optional dictionary of new properties to merge. These properties will override the existing properties of this SceneComponent if there are key conflicts.
            **kwargs: Additional keyword arguments representing properties to merge. These properties will take the highest precedence in case of key conflicts, overriding both `new_props` and the existing properties.

        Returns:
            A new SceneComponent instance with all properties merged.
        """
        merged_props = {**self.props}  # Start with existing props
        if new_props:
            merged_props.update(new_props)  # Update with new_props
        merged_props.update(kwargs)  # Update with kwargs, overriding if necessary
        return SceneComponent(
            self.type, merged_props
        )  # Create and return a new instance


def flatten_layers(layers):
    flattened = []
    for layer in layers:
        if isinstance(layer, Scene):
            flattened.extend(flatten_layers(layer.layers))
        else:
            flattened.append(layer)
    return flattened


# Position-typed component attributes: arrays of world-space coordinates that
# must be translated by -origin when a scene declares an ``origin`` (see
# ``Scene(origin=...)``). Everything else -- colors, sizes, quaternions, uvs --
# is origin-invariant. ``points`` (LineBeams) packs [x,y,z,lineIndex] quads, so
# only the first three of every four scalars are translated.
#
# Mesh geometry is handled specially (``_recenter_mesh_props``): a mesh renders
# each vertex at ``center + R*(S*vertex)``, so simply subtracting origin from
# the instance ``centers`` alone would leave the large-magnitude geometry
# vertices (e.g. UTM eastings ~4.4e5) in float32 GPU buffers -- the precision
# problem the origin is meant to solve. Instead we fold the geometry centroid
# into the instance centers (leaving small local geometry) and then shift the
# centers by -origin uniformly, exactly like every other position array. The
# composite world position is unchanged; only the *representation* becomes
# float32-safe.
_POSITION_ATTRS_STRIDE3 = ("centers", "starts", "ends")
_POSITION_ATTRS_STRIDE4 = ("points",)


def _quat_rotate(quat: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Rotate row vectors by quaternion ``[w, x, y, z]`` (mesh convention).

    Args:
        quat: (4,) quaternion in ``[w, x, y, z]`` order (the mesh instance
            convention; identity when absent).
        vecs: (N, 3) vectors to rotate.

    Returns:
        (N, 3) rotated vectors: ``v + 2*w*(q x v) + 2*(q x (q x v))``.
    """
    w = float(quat[0])
    q = quat[1:4].astype(np.float64)
    cross1 = np.cross(np.broadcast_to(q, vecs.shape), vecs)
    cross2 = np.cross(np.broadcast_to(q, vecs.shape), cross1)
    return vecs + 2.0 * w * cross1 + 2.0 * cross2


def _recenter_mesh_props(props: Dict[str, Any]) -> Dict[str, Any]:
    """Fold a mesh's geometry centroid into its instance centers.

    Returns a copy of ``props`` with ``geometry.positions`` re-expressed
    relative to their centroid (small, local coordinates) and each instance
    ``center`` moved by ``R*(S*centroid)`` so the composite world position of
    every vertex is unchanged. This keeps large world-space vertex arrays out
    of float32 GPU buffers; the (now small) centers are subsequently shifted
    by -origin like all other positions.

    A no-op unless ``geometry.positions`` is a translatable numeric array.
    """
    geometry = props.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("positions") is None:
        return props
    positions = geometry["positions"]
    if isinstance(positions, JSExpr):
        return props
    verts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if verts.size == 0:
        return props
    centroid = verts.mean(axis=0)

    out = dict(props)
    new_geometry = dict(geometry)
    new_geometry["positions"] = (verts - centroid).reshape(-1).astype(np.float32)
    out["geometry"] = new_geometry

    centers = np.asarray(out.get("centers"), dtype=np.float64).reshape(-1, 3)
    # Per-instance transform: quaternion is [w,x,y,z], scale is scalar or
    # [x,y,z]. Offset each center by the transformed centroid so the composite
    # (center + R*(S*(vertex-centroid))) matches the original.
    quats = out.get("quaternions")
    quat = out.get("quaternion")
    scales = out.get("scales")
    scale = out.get("scale")
    offsets = np.empty_like(centers)
    for i in range(len(centers)):
        if scales is not None:
            s = np.asarray(scales, dtype=np.float64).reshape(-1, 3)[i]
        elif scale is not None:
            s = np.asarray(scale, dtype=np.float64)
            if s.ndim == 0:
                s = np.repeat(s, 3)
        else:
            s = np.ones(3)
        if quats is not None:
            q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)[i]
        elif quat is not None:
            q = np.asarray(quat, dtype=np.float64)
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        scaled = (s * centroid).reshape(1, 3)
        offsets[i] = _quat_rotate(q, scaled)[0]
    # Keep centers in float64 here: they may still be at world magnitude
    # (~4.4e5) and only become float32-safe after the -origin shift that the
    # caller applies next. Casting to float32 now would reintroduce the
    # precision loss the origin machinery exists to avoid.
    out["centers"] = (centers + offsets).reshape(-1)
    return out


def _translate_flat(arr: Any, origin: np.ndarray, stride: int) -> Any:
    """Subtract ``origin`` from every position packed in a flat array.

    Args:
        arr: A flat float array (or nested list) of packed coordinates.
        origin: (3,) offset to subtract from each x,y,z triple.
        stride: 3 for plain xyz packing, 4 for [x,y,z,extra] packing (the
            4th slot, e.g. LineBeams line index, is left untouched).

    Returns:
        A new float32 array with the offset applied, or ``arr`` unchanged
        when it is not a translatable numeric array (e.g. a JSExpr).
    """
    if isinstance(arr, JSExpr):
        return arr
    # Subtract in float64 and only cast to float32 afterwards: the inputs may
    # be at world magnitude (~4.4e5), where a float32 representation before the
    # subtraction would already have lost the sub-metre detail. Doing the
    # subtraction first keeps the shifted (small) result precise.
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == stride:
        flat = values.reshape(-1).copy()
    elif values.ndim == 1 and values.size % stride == 0:
        flat = values.copy()
    else:
        # Unexpected shape (e.g. a scalar or ragged input): leave untouched
        # rather than corrupt the data.
        return arr
    flat = flat.reshape(-1, stride)
    flat[:, 0:3] -= origin.astype(np.float64)
    return flat.reshape(-1).astype(np.float32)


def _translate_component_props(
    props: Dict[str, Any], origin: np.ndarray
) -> Dict[str, Any]:
    """Return a copy of a component's props with positions shifted by -origin."""
    # Fold any mesh geometry centroid into its instance centers first, so the
    # large-magnitude vertex array becomes small/local and the world offset it
    # carried lands on ``centers`` -- which the loop below then shifts by
    # -origin like every other position array (single, uniform shift). Mesh
    # ``geometry.positions`` are deliberately NOT in the shift lists: shifting
    # them in addition to the centers would double-shift the composite.
    out = _recenter_mesh_props(props)
    for key in _POSITION_ATTRS_STRIDE3:
        if key in out and out[key] is not None:
            out[key] = _translate_flat(out[key], origin, 3)
    for key in _POSITION_ATTRS_STRIDE4:
        if key in out and out[key] is not None:
            out[key] = _translate_flat(out[key], origin, 4)
    # Group children are nested components with their own position attrs; the
    # group's own ``position`` offset is a world-space translation too.
    if "children" in out and isinstance(out["children"], list):
        out["children"] = [_translate_layer(child, origin) for child in out["children"]]
    if "position" in out and out["position"] is not None:
        pos = np.asarray(out["position"], dtype=np.float64)
        out["position"] = (pos - origin).tolist()
    return out


def _translate_layer(layer: Any, origin: np.ndarray) -> Any:
    """Translate a single scene layer (component or nested config) by -origin."""
    if isinstance(layer, SceneComponent):
        return SceneComponent(
            layer.type, _translate_component_props(layer.props, origin)
        )
    if isinstance(layer, dict) and "type" in layer:
        return _translate_component_props(layer, origin)
    return layer


# =============================================================================
# Section / clipping planes
# =============================================================================

# Fixed-size uniform array on the GPU (see clipPlanesStruct in shaders.ts). More
# than this many planes cannot be uploaded, so we raise loudly rather than
# silently drop.
MAX_CLIP_PLANES = 8


class ClipPlane(TypedDict, total=False):
    """A scene-level section / clipping plane.

    A fragment at world position ``p`` is KEPT where
    ``dot(p, normal) <= offset`` and discarded otherwise; multiple planes
    intersect (a fragment must pass every plane). Give the plane either as
    ``normal`` + ``offset`` or the anchored ``normal`` + ``point`` form.
    """

    normal: ArrayLike  # plane normal [nx, ny, nz] (normalized internally)
    offset: NumberLike  # signed distance along normal; may be a $state ref
    point: ArrayLike  # a point the plane passes through (anchored form)


def _is_js_expr(value: Any) -> bool:
    return isinstance(value, JSExpr)


def _normalize_clip_planes(
    clip_planes: Optional[Sequence[Union[ClipPlane, Dict[str, Any]]]],
    origin: Optional[np.ndarray],
) -> Optional[list]:
    """Normalize ``clip_planes`` into ``[{normal, offset}, ...]`` for the JS boundary.

    - ``normal`` is normalized to unit length (a zero normal is an error).
    - The anchored ``point`` form is converted to an ``offset`` AFTER the origin
      re-centering: positions are shifted by ``-origin`` at serialization time,
      so a plane through world point ``P`` becomes ``offset = dot(P - origin,
      normal)`` in the shifted space the shader sees. The direct ``offset`` form
      is already given in that post-origin space (offsets are origin-relative).
    - ``offset`` (or ``point`` components) may be ``$state`` refs (``Plot.js``)
      for a slider-driven section sweep. A state-ref ``offset`` passes through
      unchanged; a ``point`` with any state-ref component is converted to a
      ``Plot.js`` dot-product expression so the sweep still tracks state.

    Raises:
        ValueError: on too many planes, a missing/zero normal, or a plane that
            supplies neither ``offset`` nor ``point``.
    """
    if clip_planes is None:
        return None
    planes = list(clip_planes)
    if len(planes) == 0:
        return None
    if len(planes) > MAX_CLIP_PLANES:
        raise ValueError(
            f"Scene supports at most {MAX_CLIP_PLANES} clip_planes, "
            f"got {len(planes)}. Reduce the number of section planes."
        )

    out: list = []
    origin_vec = None if origin is None else np.asarray(origin, dtype=np.float64)
    for i, plane in enumerate(planes):
        if "normal" not in plane:
            raise ValueError(f"clip_planes[{i}] requires a 'normal'")
        normal = np.asarray(plane["normal"], dtype=np.float64).reshape(-1)
        if normal.shape[0] != 3:
            raise ValueError(f"clip_planes[{i}] 'normal' must have 3 components")
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError(f"clip_planes[{i}] 'normal' must be non-zero")
        unit = (normal / norm).tolist()

        raw_offset = plane.get("offset")
        raw_point = plane.get("point")
        has_offset = raw_offset is not None
        has_point = raw_point is not None
        if has_offset and has_point:
            raise ValueError(
                f"clip_planes[{i}] must specify either 'offset' or 'point', not both"
            )

        offset: Any
        if has_offset:
            # Offset is already in the post-origin (origin-relative) space; pass
            # scalars and $state refs through unchanged.
            offset = raw_offset
        elif has_point:
            point = raw_point
            point_is_ref = _is_js_expr(point) or (
                isinstance(point, (list, tuple)) and any(_is_js_expr(c) for c in point)
            )
            if point_is_ref:
                # A state-driven anchor: emit dot((point - origin), normal) as a
                # JS expression so the section still tracks state on the client.
                ox, oy, oz = (
                    (0.0, 0.0, 0.0)
                    if origin_vec is None
                    else (
                        float(origin_vec[0]),
                        float(origin_vec[1]),
                        float(origin_vec[2]),
                    )
                )
                nx, ny, nz = unit
                px, py, pz = point  # type: ignore[misc]
                offset = Plot.js(
                    f"(($p) => ($p[0]-({ox}))*({nx}) + ($p[1]-({oy}))*({ny}) "
                    f"+ ($p[2]-({oz}))*({nz}))([{_js_scalar(px)}, "
                    f"{_js_scalar(py)}, {_js_scalar(pz)}])"
                )
            else:
                pt = np.asarray(point, dtype=np.float64).reshape(-1)
                if pt.shape[0] != 3:
                    raise ValueError(f"clip_planes[{i}] 'point' must have 3 components")
                shifted = pt if origin_vec is None else pt - origin_vec
                offset = float(np.dot(shifted, unit))
        else:
            raise ValueError(f"clip_planes[{i}] requires either 'offset' or 'point'")

        out.append({"normal": unit, "offset": offset})
    return out


def _js_scalar(value: Any) -> str:
    """Render a clip-plane point component as inline JS (number or Plot.js code).

    A ``point`` component may be a literal number or a ``Plot.js(...)`` state
    ref. For the latter we inline the expression's raw source (wrapped in
    parens) so the surrounding dot-product expression evaluates against live
    ``$state``.
    """
    if _is_js_expr(value):
        code = getattr(value, "code", None)
        if code is None:
            raise ValueError(
                "clip_planes 'point' state refs must be Plot.js(...) expressions"
            )
        return f"({code})"
    return repr(float(value))


class Scene(Plot.LayoutItem):
    """A 3D scene visual component using WebGPU.

    This class creates an interactive 3D scene that can contain multiple types of components:

    - Point clouds
    - Ellipsoids
    - Ellipsoid bounds (wireframe)
    - Cuboids

    The component supports:

    - Orbit camera control (left mouse drag)
    - Pan camera control (shift + left mouse drag or middle mouse drag)
    - Zoom control (mouse wheel)
    - Component hover highlighting
    - Component click selection
    - Optional FPS display (set controls=['fps'])
    """

    def __init__(
        self,
        *layers: Union[SceneComponent, Dict[str, Any], JSExpr],
        origin: Optional[Sequence[float]] = None,
        background: Optional[Sequence[float]] = None,
        clip_planes: Optional[Sequence[Union[ClipPlane, Dict[str, Any]]]] = None,
        primitive_specs: Optional[Dict[str, Dict[str, Any]]] = None,
        meshes: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize the scene.

        Args:
            *layers: Scene components and optional properties.
                Properties can include:
                - controls: List of controls to show. Currently supports ['fps']
            origin: Optional world-space offset ``[x, y, z]`` subtracted from
                every position-typed attribute (centers/starts/ends/points/
                positions and Group positions) at serialization time. Use it
                for scenes whose coordinates are far from the origin (e.g.
                UTM eastings ~445,000 m) so positions fit float32 GPU
                precision. The offset travels as scene metadata; ``pick-at`` /
                ``pick-where`` add it back so dereferenced positions are in
                the caller's original coordinate space, and camera auto-fit
                works in the shifted space transparently.
            background: Optional RGB clear color ``[r, g, b]`` (each 0-1) for
                the WebGPU render pass. Defaults to opaque black. This is the
                canvas clear color *behind* the geometry; DOM overlays drawn
                over the canvas (legends, FPS) keep their own styling.
            clip_planes: Optional scene-level section / clipping planes. Each is
                ``{"normal": [nx, ny, nz], "offset": d}`` (keep the half-space
                ``dot(p, normal) <= d``) or the anchored
                ``{"normal": n, "point": [x, y, z]}`` form (converted to an
                offset that respects ``origin``). ``offset`` or ``point``
                components may be ``Plot.js("$state...")`` refs to drive a
                section-sweep slider. Planes apply to ALL components in both the
                render and pick passes; interior structure behind the cut
                becomes visible AND pickable. Max 8 planes. Hollow shells are
                visible on the cut (v1 does not cap/fill the section surface).
                For geographic scenes with an ``origin``, prefer the ``point``
                form — it converts the anchor into the origin-shifted space
                automatically.
            primitive_specs: Dictionary of custom primitive definitions.
            meshes: Deprecated alias for primitive_specs.
        """

        self.layers = flatten_layers(layers)
        self.origin = (
            np.asarray(origin, dtype=np.float64) if origin is not None else None
        )
        self.background = list(background) if background is not None else None
        # Normalize eagerly so validation errors (bad normal, >8 planes) surface
        # at construction, not deep in the JS boundary. Point->offset conversion
        # uses origin, matching the -origin re-centering applied to positions.
        self.clip_planes = _normalize_clip_planes(clip_planes, self.origin)
        if meshes and primitive_specs:
            merged_specs = {**meshes, **primitive_specs}
        else:
            merged_specs = primitive_specs or meshes
        self.primitive_specs = merged_specs
        super().__init__()

    def _scene_kwargs(self) -> Dict[str, Any]:
        """Scene-level kwargs (origin/background/clip_planes) preserved across +."""
        kwargs: Dict[str, Any] = {}
        if self.origin is not None:
            kwargs["origin"] = self.origin
        if self.background is not None:
            kwargs["background"] = self.background
        if self.clip_planes is not None:
            # Already normalized ({normal, offset}); re-normalization on the
            # receiving Scene is idempotent (unit normal, scalar/ref offset).
            kwargs["clip_planes"] = self.clip_planes
        return kwargs

    def __add__(self, other: Union[SceneComponent, "Scene", Dict[str, Any]]) -> "Scene":
        """Allow combining scenes with + operator."""
        new_specs = self.primitive_specs.copy() if self.primitive_specs else {}
        if isinstance(other, Scene) and other.primitive_specs:
            new_specs.update(other.primitive_specs)

        kwargs = self._scene_kwargs()
        if isinstance(other, Scene):
            # The right-hand scene's origin/background win if it declares them.
            kwargs.update(other._scene_kwargs())
        if new_specs:
            kwargs["primitive_specs"] = new_specs

        if isinstance(other, Scene):
            return Scene(*self.layers, *other.layers, **kwargs)
        else:
            return Scene(*self.layers, other, **kwargs)

    def __radd__(self, other: Union[Dict[str, Any], JSExpr]) -> "Scene":
        """Allow combining scenes with + operator when dict or JSExpr is on the left."""
        kwargs = self._scene_kwargs()
        if self.primitive_specs:
            kwargs["primitive_specs"] = self.primitive_specs
        return Scene(other, *self.layers, **kwargs)

    def for_json(self) -> Any:
        """Convert to JSON representation for JavaScript."""
        layers = self.layers
        if self.origin is not None:
            layers = [_translate_layer(layer, self.origin) for layer in layers]
        components = [
            e.to_js_call() if isinstance(e, SceneComponent) else e for e in layers
        ]
        props: Dict[str, Any] = {"layers": components}
        if self.primitive_specs:
            props["primitiveSpecs"] = self.primitive_specs
        if self.origin is not None:
            props["origin"] = self.origin.tolist()
        if self.background is not None:
            props["background"] = self.background
        if self.clip_planes is not None:
            # camelCase for the JS boundary. Offsets that are Plot.js state refs
            # serialize through their own for_json (js_source), so a slider
            # sweep re-resolves on every state change — the light render path.
            props["clipPlanes"] = self.clip_planes
        # Named selections are resident in $state.selections (see Selection /
        # select). Passing a live reference lets the JS side re-resolve them on
        # every state change (Python or human click), and it costs nothing when
        # no selections are set (undefined resolves to no-op).
        props["selections"] = Plot.js("$state.selections")
        return [Plot.JSRef("scene3d.Scene"), props]


# =============================================================================
# Named selections
# =============================================================================


class SelectionStyle(TypedDict, total=False):
    """Highlight style applied to a named selection's instances."""

    color: ArrayLike  # [r,g,b]
    alpha: NumberLike
    scale: NumberLike
    outline: bool
    outline_color: ArrayLike  # [r,g,b]
    outline_width: NumberLike


def Selection(
    name: str,
    component: int,
    *,
    instances: Optional[Sequence[int]] = None,
    values: Optional[ArrayLike] = None,
    values_ref: Optional[str] = None,
    min: Optional[NumberLike] = None,
    max: Optional[NumberLike] = None,
    style: Optional[Union[SelectionStyle, str]] = None,
) -> Dict[str, Any]:
    """Define a named selection over one scene component.

    A selection is a NAMED per-instance mask — the same abstraction as
    ``filter_by`` — consumed as a highlight decoration plus addressability: it
    is a *shared referent* that both a human (via clicks) and an agent (via
    predicates, ``pick-where --selection NAME``, ``screenshot --frame NAME``)
    can name in conversation. Selections live in ``$state.selections`` (seed
    them with :func:`select`), so they sync Python<->JS and persist into
    ``.colight`` artifacts.

    The mask is either an explicit ``instances`` list, or a threshold predicate:
    ``values`` (inline scalars) or ``values_ref`` (a per-instance attribute name
    on the component) tested against ``[min, max]``.

    Args:
        name: Stable selection name (the shared referent).
        component: Compiled-component index the selection targets.
        instances: Explicit instance indices to select.
        values: Inline per-instance scalar values for a threshold predicate.
        values_ref: Name of a per-instance attribute to read values from.
        min: Inclusive lower threshold for the predicate.
        max: Inclusive upper threshold for the predicate.
        style: Highlight style dict, or ``"default"`` for the built-in
            highlight. Defaults to the built-in highlight.

    Returns:
        ``(name, spec)`` where ``spec`` is the ``$state.selections[name]``
        entry. Pass the results to :func:`select` to seed state.
    """
    source: Dict[str, Any]
    if instances is not None:
        source = {"instances": [int(i) for i in instances]}
    elif values is not None or values_ref is not None:
        source = {}
        if values is not None:
            source["values"] = flatten_array(values, dtype=np.float32)
        if values_ref is not None:
            source["values_ref"] = values_ref
        if min is not None:
            source["min"] = min
        if max is not None:
            source["max"] = max
    else:
        raise ValueError(
            "Selection requires either 'instances' or 'values'/'values_ref'"
        )

    spec: Dict[str, Any] = {"component": int(component), "source": source}
    if style is not None:
        spec["style"] = _convert_to_js(style) if isinstance(style, dict) else style
    return {name: spec}


def select(*selections: Dict[str, Any]) -> Any:
    """Seed ``$state.selections`` with one or more :func:`Selection` specs.

    Returns a ``Plot.initialState`` marker (synced) that makes the selections
    resident, shared state. Combine it with a scene:

        >>> scene | select(Selection("hi", 0, instances=[1, 2]))

    Args:
        *selections: Dicts returned by :func:`Selection` (each ``{name: spec}``).

    Returns:
        A ``Plot.initialState`` layout marker seeding ``$state.selections``.
    """
    merged: Dict[str, Any] = {}
    for sel in selections:
        merged.update(sel)
    return Plot.initialState({"selections": merged}, sync=True)


def toggle_selection(name: str, component: int) -> Any:
    """An ``on_click`` handler that toggles the picked instance in a selection.

    Human clicks and agent predicates converge on the same named object: a
    click adds (or removes) the picked instance to ``$state.selections[name]``
    (creating an ``instances`` selection if absent). Attach it to a component:

        >>> Cuboid(centers=..., on_click=toggle_selection("hi", 0))

    Args:
        name: Selection name to toggle membership in.
        component: Component index the selection targets.

    Returns:
        A ``Plot.js`` click handler expression.
    """
    return Plot.js(
        """(e) => {
  const sels = {...($state.selections || {})};
  const cur = sels[%1] || {component: %2, source: {instances: []}};
  const src = cur.source || {instances: []};
  const list = (src.instances || []).slice();
  const i = e.instanceIndex;
  const at = list.indexOf(i);
  if (at >= 0) { list.splice(at, 1); } else { list.push(i); }
  sels[%1] = {...cur, component: %2, source: {...src, instances: list}};
  $state.selections = sels;
}""",
        name,
        component,
    )


def coerce_index_array(arr: Any) -> Any:
    """Normalize index data to uint16/uint32 based on max index."""
    if arr is None:
        return None
    array = np.asarray(arr)
    max_val = int(array.max()) if array.size else 0
    dtype = np.uint32 if max_val > 65535 else np.uint16
    return flatten_array(array, dtype=dtype)


def PointCloud(
    centers: Optional[ArrayLike] = None,
    *,
    center: Optional[ArrayLike] = None,  # Singular form for a single point
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,  # Default RGB color for all points
    color_by: Optional[ColorBy] = None,  # Colormap-driven per-point colors
    color_channels: Optional[Dict[str, ColorChannel]] = None,  # Switchable channels
    active_channel: Optional[Union[str, JSExpr]] = None,  # Active channel name/$state
    sizes: Optional[ArrayLike] = None,
    size: Optional[NumberLike] = None,  # Default size for all points
    alphas: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,  # Default alpha for all points
    filter_by: Optional[FilterBy] = None,  # Per-instance threshold filter
    layer: Optional[Literal["scene", "overlay"]] = None,
    hover_props: Optional[HoverProps] = None,
    picking_scale: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a point cloud element.

    Args:
        centers: Nx3 array of point centers or flattened array
        center: Single point center [x, y, z] (convenience for single-point case)
        colors: Nx3 array of RGB colors or flattened array (optional)
        color: Default RGB color [r,g,b] for all points if colors not provided
        color_by: Colormap spec {values, cmap, domain, label, ...} mapping
            scalar values to per-point colors (see ColorBy); adds a legend
        sizes: N array of point sizes or flattened array (optional)
        size: Default size for all points if sizes not provided
        alphas: Array of alpha values per point (optional)
        alpha: Default alpha value for all points if alphas not provided
        layer: Render layer - "scene" (default) or "overlay" (renders in front, always visible)
        hover_props: Properties to apply on hover (color, alpha, scale)
        picking_scale: Scale factor for picking hit area (values > 1 make clicking easier)
        **kwargs: Additional arguments like decorations, on_hover, on_click
    """
    if centers is None:
        if center is not None:
            centers = [center]
        else:
            raise ValueError("Either 'centers' or 'center' must be provided")
    centers = flatten_array(centers, dtype=np.float32)
    data: Dict[str, Any] = {"centers": centers}

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)
    if color is not None:
        data["color"] = color
    _apply_color_by(data, color_by)
    _apply_color_channels(data, color_channels, active_channel)

    if sizes is not None:
        data["sizes"] = flatten_array(sizes, dtype=np.float32)
    if size is not None:
        data["size"] = size

    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    if alpha is not None:
        data["alpha"] = alpha

    if filter_by is not None:
        data["filter_by"] = _normalize_filter_by(filter_by)

    if layer is not None:
        data["layer"] = layer

    if hover_props is not None:
        data["hover_props"] = hover_props

    if picking_scale is not None:
        data["picking_scale"] = picking_scale

    return SceneComponent("PointCloud", data, **kwargs)


def Ellipsoid(
    centers: Optional[ArrayLike] = None,
    *,
    center: Optional[ArrayLike] = None,  # Singular form for a single ellipsoid
    half_sizes: Optional[ArrayLike] = None,
    half_size: Optional[Union[NumberLike, ArrayLike]] = None,  # Single value or [x,y,z]
    quaternions: Optional[ArrayLike] = None,  # Nx4 array of quaternions [x,y,z,w]
    quaternion: Optional[ArrayLike] = None,  # Default orientation quaternion [x,y,z,w]
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,  # Default RGB color for all ellipsoids
    color_by: Optional[ColorBy] = None,  # Colormap-driven per-instance colors
    color_channels: Optional[Dict[str, ColorChannel]] = None,  # Switchable channels
    active_channel: Optional[Union[str, JSExpr]] = None,  # Active channel name/$state
    alphas: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,  # Default alpha for all ellipsoids
    fill_mode: str
    | None = None,  # How the shape is drawn ("Solid" or "MajorWireframe")
    filter_by: Optional[FilterBy] = None,  # Per-instance threshold filter
    layer: Optional[Literal["scene", "overlay"]] = None,
    hover_props: Optional[HoverProps] = None,
    picking_scale: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create an ellipsoid element.

    Args:
        centers: Nx3 array of ellipsoid centers or flattened array
        center: Single ellipsoid center [x, y, z] (convenience for single-ellipsoid case)
        half_sizes: Nx3 array of half_sizes (x,y,z) or flattened array (optional)
        half_size: Default half_size (sphere) or [x,y,z] half_sizes (ellipsoid) if half_sizes not provided
        quaternions: Nx4 array of orientation quaternions [x,y,z,w] (optional)
        quaternion: Default orientation quaternion [x,y,z,w] if quaternions not provided
        colors: Nx3 array of RGB colors or flattened array (optional)
        color: Default RGB color [r,g,b] for all ellipsoids if colors not provided
        color_by: Colormap spec {values, cmap, domain, label, ...} mapping
            scalar values to per-instance colors (see ColorBy); adds a legend
        alphas: Array of alpha values per ellipsoid (optional)
        alpha: Default alpha value for all ellipsoids if alphas not provided
        fill_mode: How the shape is drawn. One of:
            - "Solid": Filled surface with solid color
            - "MajorWireframe": Three axis-aligned ellipse cross-sections
        layer: Render layer - "scene" (default) or "overlay" (renders in front, always visible)
        hover_props: Properties to apply on hover (color, alpha, scale)
        picking_scale: Scale factor for picking hit area (values > 1 make clicking easier)
        **kwargs: Additional arguments like decorations, on_hover, on_click
    """
    if centers is None:
        if center is not None:
            centers = [center]
        else:
            raise ValueError("Either 'centers' or 'center' must be provided")
    centers = flatten_array(centers, dtype=np.float32)
    data: Dict[str, Any] = {"centers": centers}

    if half_sizes is not None:
        data["half_sizes"] = flatten_array(half_sizes, dtype=np.float32)
    elif half_size is not None:
        data["half_size"] = half_size

    if quaternions is not None:
        data["quaternions"] = flatten_array(quaternions, dtype=np.float32)
    elif quaternion is not None:
        data["quaternion"] = quaternion

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)
    elif color is not None:
        data["color"] = color
    _apply_color_by(data, color_by)
    _apply_color_channels(data, color_channels, active_channel)

    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    elif alpha is not None:
        data["alpha"] = alpha

    if fill_mode is not None:
        data["fill_mode"] = fill_mode

    if filter_by is not None:
        data["filter_by"] = _normalize_filter_by(filter_by)

    if layer is not None:
        data["layer"] = layer

    if hover_props is not None:
        data["hover_props"] = hover_props

    if picking_scale is not None:
        data["picking_scale"] = picking_scale

    return SceneComponent("Ellipsoid", data, **kwargs)


def Cuboid(
    centers: Optional[ArrayLike] = None,
    *,
    center: Optional[ArrayLike] = None,  # Singular form for a single cuboid
    half_sizes: Optional[ArrayLike] = None,
    half_size: Optional[Union[ArrayLike, NumberLike]] = None,
    quaternions: Optional[ArrayLike] = None,  # Nx4 array of quaternions [x,y,z,w]
    quaternion: Optional[ArrayLike] = None,  # Default orientation quaternion [x,y,z,w]
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,  # Default RGB color for all cuboids
    color_by: Optional[ColorBy] = None,  # Colormap-driven per-instance colors
    color_channels: Optional[Dict[str, ColorChannel]] = None,  # Switchable channels
    active_channel: Optional[Union[str, JSExpr]] = None,  # Active channel name/$state
    alphas: Optional[ArrayLike] = None,  # Per-cuboid alpha values
    alpha: Optional[NumberLike] = None,  # Default alpha for all cuboids
    filter_by: Optional[FilterBy] = None,  # Per-instance threshold filter
    layer: Optional[Literal["scene", "overlay"]] = None,
    hover_props: Optional[HoverProps] = None,
    picking_scale: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a cuboid element.

    Args:
        centers: Nx3 array of cuboid centers or flattened array
        center: Single cuboid center [x, y, z] (convenience for single-cuboid case)
        half_sizes: Nx3 array of half sizes (width,height,depth) or flattened array (optional)
        half_size: Default half size [w,h,d] for all cuboids if half_sizes not provided
        quaternions: Nx4 array of orientation quaternions [x,y,z,w] (optional)
        quaternion: Default orientation quaternion [x,y,z,w] if quaternions not provided
        colors: Nx3 array of RGB colors or flattened array (optional)
        color: Default RGB color [r,g,b] for all cuboids if colors not provided
        color_by: Colormap spec {values, cmap, domain, label, ...} mapping
            scalar values to per-instance colors (see ColorBy); adds a legend
        alphas: Array of alpha values per cuboid (optional)
        alpha: Default alpha value for all cuboids if alphas not provided
        layer: Render layer - "scene" (default) or "overlay" (renders in front, always visible)
        hover_props: Properties to apply on hover (color, alpha, scale)
        picking_scale: Scale factor for picking hit area (values > 1 make clicking easier)
        **kwargs: Additional arguments like decorations, on_hover, on_click
    """
    if centers is None:
        if center is not None:
            centers = [center]
        else:
            raise ValueError("Either 'centers' or 'center' must be provided")
    centers = flatten_array(centers, dtype=np.float32)
    data: Dict[str, Any] = {"centers": centers}

    if half_sizes is not None:
        data["half_sizes"] = flatten_array(half_sizes, dtype=np.float32)
    elif half_size is not None:
        data["half_size"] = half_size

    if quaternions is not None:
        data["quaternions"] = flatten_array(quaternions, dtype=np.float32)
    elif quaternion is not None:
        data["quaternion"] = quaternion

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)
    elif color is not None:
        data["color"] = color
    _apply_color_by(data, color_by)
    _apply_color_channels(data, color_channels, active_channel)

    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    elif alpha is not None:
        data["alpha"] = alpha

    if filter_by is not None:
        data["filter_by"] = _normalize_filter_by(filter_by)

    if layer is not None:
        data["layer"] = layer

    if hover_props is not None:
        data["hover_props"] = hover_props

    if picking_scale is not None:
        data["picking_scale"] = picking_scale

    return SceneComponent("Cuboid", data, **kwargs)


def LineBeams(
    points: ArrayLike,  # Array of quadruples [x,y,z,i, x,y,z,i, ...]
    color: Optional[ArrayLike] = None,  # Default RGB color for all beams
    size: Optional[NumberLike] = None,  # Default size for all beams
    colors: Optional[ArrayLike] = None,  # Per-line colors
    color_by: Optional[ColorBy] = None,  # Colormap-driven per-line colors
    color_channels: Optional[Dict[str, ColorChannel]] = None,  # Switchable channels
    active_channel: Optional[Union[str, JSExpr]] = None,  # Active channel name/$state
    sizes: Optional[ArrayLike] = None,  # Per-line sizes
    alpha: Optional[NumberLike] = None,  # Default alpha for all beams
    alphas: Optional[ArrayLike] = None,  # Per-line alpha values
    filter_by: Optional[FilterBy] = None,  # Per-instance (per-segment) filter
    layer: Optional[Literal["scene", "overlay"]] = None,
    hover_props: Optional[HoverProps] = None,
    picking_scale: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a line beams element.

    Args:
        points: Array of quadruples [x,y,z,i, x,y,z,i, ...] where points sharing the same i value are connected in sequence
        color: Default RGB color [r,g,b] for all beams if colors not provided
        size: Default size for all beams if sizes not provided
        colors: Array of RGB colors per line (optional)
        color_by: Colormap spec {values, cmap, domain, label, ...} mapping
            scalar values to per-line colors (see ColorBy); adds a legend
        sizes: Array of sizes per line (optional)
        alpha: Default alpha value for all beams if alphas not provided
        alphas: Array of alpha values per line (optional)
        layer: Render layer - "scene" (default) or "overlay" (renders in front, always visible)
        hover_props: Properties to apply on hover (color, alpha, scale)
        picking_scale: Scale factor for picking hit area (values > 1 make clicking easier)
        **kwargs: Additional arguments like on_hover, on_click

    Returns:
        A LineBeams scene component that renders connected beam segments.
        Points are connected in sequence within groups sharing the same i value.
    """
    data: Dict[str, Any] = {"points": flatten_array(points, dtype=np.float32)}

    if color is not None:
        data["color"] = color
    if size is not None:
        data["size"] = size
    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)
    _apply_color_by(data, color_by)
    _apply_color_channels(data, color_channels, active_channel)
    if sizes is not None:
        data["sizes"] = flatten_array(sizes, dtype=np.float32)
    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    elif alpha is not None:
        data["alpha"] = alpha

    if filter_by is not None:
        data["filter_by"] = _normalize_filter_by(filter_by)

    if layer is not None:
        data["layer"] = layer

    if hover_props is not None:
        data["hover_props"] = hover_props

    if picking_scale is not None:
        data["picking_scale"] = picking_scale

    return SceneComponent("LineBeams", data, **kwargs)


def LineSegments(
    starts: ArrayLike,
    ends: ArrayLike,
    color: Optional[ArrayLike] = None,
    size: Optional[NumberLike] = None,
    colors: Optional[ArrayLike] = None,
    color_by: Optional[ColorBy] = None,  # Colormap-driven per-segment colors
    color_channels: Optional[Dict[str, ColorChannel]] = None,  # Switchable channels
    active_channel: Optional[Union[str, JSExpr]] = None,  # Active channel name/$state
    sizes: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,
    alphas: Optional[ArrayLike] = None,
    filter_by: Optional[FilterBy] = None,  # Per-instance (per-segment) filter
    layer: Optional[Literal["scene", "overlay"]] = None,
    hover_props: Optional[HoverProps] = None,
    picking_scale: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a line segments element.

    Args:
        starts: Nx3 array of segment start positions
        ends: Nx3 array of segment end positions
        color: Default RGB color [r,g,b] for all segments if colors not provided
        size: Default size for all segments if sizes not provided
        colors: Array of RGB colors per segment (optional)
        color_by: Colormap spec {values, cmap, domain, label, ...} mapping
            scalar values to per-segment colors (see ColorBy); adds a legend
        sizes: Array of sizes per segment (optional)
        alpha: Default alpha value for all segments if alphas not provided
        alphas: Array of alpha values per segment (optional)
        layer: Render layer - "scene" (default) or "overlay"
        hover_props: Properties to apply on hover (color, alpha, scale)
        picking_scale: Scale factor for picking hit area (values > 1 make clicking easier)
        **kwargs: Additional arguments like on_hover, on_click

    Returns:
        A LineSegments scene component that renders independent segments.
    """
    data: Dict[str, Any] = {
        "starts": flatten_array(starts, dtype=np.float32),
        "ends": flatten_array(ends, dtype=np.float32),
    }

    if color is not None:
        data["color"] = color
    if size is not None:
        data["size"] = size
    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)
    _apply_color_by(data, color_by)
    _apply_color_channels(data, color_channels, active_channel)
    if sizes is not None:
        data["sizes"] = flatten_array(sizes, dtype=np.float32)
    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    elif alpha is not None:
        data["alpha"] = alpha

    if filter_by is not None:
        data["filter_by"] = _normalize_filter_by(filter_by)

    if layer is not None:
        data["layer"] = layer

    if hover_props is not None:
        data["hover_props"] = hover_props

    if picking_scale is not None:
        data["picking_scale"] = picking_scale

    return SceneComponent("LineSegments", data, **kwargs)


def ImagePlane(
    image: Any,
    *,
    centers: Optional[ArrayLike] = None,
    position: Optional[ArrayLike] = None,
    quaternions: Optional[ArrayLike] = None,
    quaternion: Optional[ArrayLike] = None,
    sizes: Optional[ArrayLike] = None,
    size: Optional[ArrayLike] = None,
    width: Optional[NumberLike] = None,
    height: Optional[NumberLike] = None,
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,
    alphas: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,
    opacity: Optional[NumberLike] = None,
    image_key: Optional[Union[str, int]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    layer: Optional[Literal["scene", "overlay"]] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a textured image plane."""
    if centers is None:
        if position is not None:
            centers = [position]
        else:
            centers = [[0, 0, 0]]

    data: Dict[str, Any] = {
        "image": coerce_image_array(image, image_width, image_height),
        "centers": flatten_array(centers, dtype=np.float32),
    }

    if quaternions is not None:
        data["quaternions"] = flatten_array(quaternions, dtype=np.float32)
    elif quaternion is not None:
        data["quaternions"] = flatten_array([quaternion], dtype=np.float32)

    if sizes is not None:
        data["sizes"] = flatten_array(sizes, dtype=np.float32)
    else:
        resolved_size = size
        if resolved_size is None and (width is not None or height is not None):
            resolved_size = [
                width if width is not None else 1,
                height if height is not None else 1,
            ]
        if resolved_size is not None:
            data["size"] = resolved_size

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)
    if color is not None:
        data["color"] = color
    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    elif alpha is not None:
        data["alpha"] = alpha
    elif opacity is not None:
        data["alpha"] = opacity

    if image_key is not None:
        data["image_key"] = image_key
    if layer is not None:
        data["layer"] = layer

    return SceneComponent("ImagePlane", data, **kwargs)


# =============================================================================
# Groups (Hierarchical Transforms)
# =============================================================================


def Group(
    children: list,
    position: Optional[ArrayLike] = None,
    quaternion: Optional[ArrayLike] = None,
    scale: Optional[Union[NumberLike, ArrayLike]] = None,
    name: Optional[str] = None,
    child_defaults: Optional[Dict[str, Any]] = None,
    child_overrides: Optional[Dict[str, Any]] = None,
    hover_props: Optional[HoverProps] = None,
    on_hover: Optional[Any] = None,
    on_click: Optional[Any] = None,
    on_drag: Optional[Any] = None,
    on_drag_start: Optional[Any] = None,
    on_drag_end: Optional[Any] = None,
    drag_constraint: Optional[DragConstraint] = None,
) -> SceneComponent:
    """Create a group component for hierarchical scene composition.

    Groups apply a transform (position, rotation, scale) to all their children.
    At render time, groups are flattened into transformed primitives.

    Group-level event handlers receive events bubbled up from any child component.
    Group-level hover_props apply to ALL children when ANY child is hovered.

    Args:
        children: List of child components (can include nested groups)
        position: Position offset [x, y, z] in parent space
        quaternion: Rotation as quaternion [x, y, z, w]
        scale: Scale factor (uniform number or [x, y, z] per-axis)
        name: Optional name for identifying this group in pick info
        child_defaults: Props to apply to children (child values take precedence)
        child_overrides: Props to apply to children (group values take precedence)
        hover_props: Props to apply to ALL children when ANY child is hovered
        on_hover: Handler called when any child is hovered (bubbles up)
        on_click: Handler called when any child is clicked (bubbles up)
        on_drag: Handler called when any child is dragged (bubbles up)
        on_drag_start: Handler called when drag starts on any child
        on_drag_end: Handler called when drag ends on any child
        drag_constraint: Constraint for drag operations (applies to all children)

    Returns:
        A Group scene component

    Example:
        >>> # Create an interactive group that highlights all children on hover
        >>> Group(
        ...     name="robot_arm",
        ...     children=[
        ...         Ellipsoid(centers=[[0, 0, 0]], half_size=0.1),
        ...         Cuboid(centers=[[0.5, 0, 0]], half_size=0.05),
        ...     ],
        ...     hover_props={"outline": True},
        ...     on_click=lambda info: print(f"Clicked: {info}"),
        ... )
    """

    def _coerce_group_value(value: ArrayLike | NumberLike) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    # Snake_case framework keys (hover_props, on_hover, child_defaults, ...)
    # are renamed to their JS names centrally by _convert_to_js at
    # serialization time.
    data: Dict[str, Any] = {"children": children}

    if position is not None:
        data["position"] = _coerce_group_value(position)
    if quaternion is not None:
        data["quaternion"] = _coerce_group_value(quaternion)
    if scale is not None:
        data["scale"] = _coerce_group_value(scale)
    if name is not None:
        data["name"] = name
    if child_defaults is not None:
        data["child_defaults"] = child_defaults
    if child_overrides is not None:
        data["child_overrides"] = child_overrides
    if hover_props is not None:
        data["hover_props"] = hover_props
    if on_hover is not None:
        data["on_hover"] = on_hover
    if on_click is not None:
        data["on_click"] = on_click
    if on_drag is not None:
        data["on_drag"] = on_drag
    if on_drag_start is not None:
        data["on_drag_start"] = on_drag_start
    if on_drag_end is not None:
        data["on_drag_end"] = on_drag_end
    if drag_constraint is not None:
        data["drag_constraint"] = drag_constraint

    return SceneComponent("Group", data)


def Mesh(
    positions: ArrayLike,
    *,
    normals: Optional[ArrayLike] = None,
    vertex_colors: Optional[ArrayLike] = None,
    uvs: Optional[ArrayLike] = None,
    indices: Optional[ArrayLike] = None,
    texture: Optional[Any] = None,
    texture_key: Optional[Union[str, int]] = None,
    centers: Optional[ArrayLike] = None,
    center: Optional[ArrayLike] = None,  # Singular form for a single mesh instance
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,
    scales: Optional[ArrayLike] = None,
    scale: Optional[Union[NumberLike, ArrayLike]] = None,
    quaternions: Optional[ArrayLike] = None,
    quaternion: Optional[ArrayLike] = None,
    shading: Optional[Literal["lit", "unlit"]] = None,
    cull_mode: Optional[Literal["none", "front", "back"]] = None,
    geometry_key: Optional[Union[str, int]] = None,
    layer: Optional[Literal["scene", "overlay"]] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a mesh component with inline geometry.

    Supports flexible vertex formats with optional normals, per-vertex colors, UVs, and textures.

    Args:
        positions: Nx3 array of vertex positions (required)
        normals: Nx3 array of vertex normals (optional, auto-computed for lit shading)
        vertex_colors: Nx3 (RGB) or Nx4 (RGBA) array of per-vertex colors (optional)
        uvs: Nx2 array of texture coordinates (required for textured meshes)
        indices: Array of triangle indices (optional for non-indexed meshes)
        texture: Image for texturing (numpy array, PIL Image, etc). Requires uvs.
        texture_key: Optional key to force texture reupload when image data changes
        centers: Nx3 array of mesh instance centers
        center: Single mesh instance center [x, y, z] (convenience for single-instance case)
        colors: Nx3 array of per-instance RGB colors (optional, multiplied with texture)
        color: Default RGB color [r,g,b] for all instances if colors not provided
        scales: Nx3 array of per-instance scales (optional)
        scale: Default scale for all instances if scales not provided
        quaternions: Nx4 array of orientation quaternions [w,x,y,z] (optional)
        quaternion: Default orientation quaternion [w,x,y,z] if quaternions not provided
        shading: "lit" (default) or "unlit"
        cull_mode: "back" (default), "front", or "none"
        geometry_key: Optional key to force geometry reupload
        layer: Render layer - "scene" (default) or "overlay"
        **kwargs: Additional arguments like decorations, onHover, onClick

    Returns:
        A Mesh scene component using inline geometry.

    Example:
        # Simple triangle with auto-computed normals
        Mesh(
            positions=[[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
            indices=[0, 1, 2],
            center=[0, 0, 0],
        )

        # Mesh with per-vertex colors (e.g., from a 3D scan)
        Mesh(
            positions=scan_vertices,
            vertex_colors=scan_colors,  # RGB from camera
            indices=scan_faces,
            center=[0, 0, 0],
            shading="unlit",
        )

        # Textured quad
        Mesh(
            positions=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
            uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
            indices=[0, 1, 2, 0, 2, 3],
            texture=my_image,
            center=[0, 0, 0],
            shading="unlit",
        )
    """
    if centers is None:
        # A plain world-space mesh needs no instancing; default to a single
        # instance at the origin so callers don't have to pass center=[0,0,0].
        centers = [center] if center is not None else [[0.0, 0.0, 0.0]]

    geometry: Dict[str, Any] = {
        "positions": flatten_array(positions, dtype=np.float32),
    }
    if normals is not None:
        geometry["normals"] = flatten_array(normals, dtype=np.float32)
    if vertex_colors is not None:
        geometry["colors"] = flatten_array(vertex_colors, dtype=np.float32)
    if uvs is not None:
        geometry["uvs"] = flatten_array(uvs, dtype=np.float32)
    indices_arr = coerce_index_array(indices)
    if indices_arr is not None:
        geometry["indices"] = indices_arr

    data: Dict[str, Any] = {
        "geometry": geometry,
        "centers": flatten_array(centers, dtype=np.float32),
    }

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)
    elif color is not None:
        data["color"] = color

    if scales is not None:
        data["scales"] = flatten_array(scales, dtype=np.float32)
    elif scale is not None:
        data["scale"] = scale

    if quaternions is not None:
        data["quaternions"] = flatten_array(quaternions, dtype=np.float32)
    elif quaternion is not None:
        data["quaternion"] = quaternion

    if texture is not None:
        data["texture"] = coerce_image_array(texture)
    if texture_key is not None:
        data["texture_key"] = texture_key

    if shading:
        data["shading"] = shading
    if cull_mode:
        data["cull_mode"] = cull_mode
    if geometry_key is not None:
        data["geometry_key"] = geometry_key
    if layer is not None:
        data["layer"] = layer

    return SceneComponent("Mesh", data, **kwargs)


def CustomPrimitive(
    type_name: str,
    centers: Optional[ArrayLike] = None,
    *,
    center: Optional[ArrayLike] = None,  # Singular form for a single instance
    layer: Optional[Literal["scene", "overlay"]] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a custom primitive instance.

    Args:
        type_name: Name of the mesh type (must match a Mesh definition)
        centers: Nx3 array of centers
        center: Single center [x, y, z] (convenience for single-instance case)
        layer: "scene" or "overlay"
        **kwargs: Additional properties (colors, scales, quaternions, etc.)

    Returns:
        SceneComponent for the custom primitive
    """
    if centers is None:
        if center is not None:
            centers = [center]
        else:
            raise ValueError("Either 'centers' or 'center' must be provided")
    centers = flatten_array(centers, dtype=np.float32)
    data: Dict[str, Any] = {"type": type_name, "centers": centers}
    if layer:
        data["layer"] = layer
    data.update(kwargs)

    # We use "CustomPrimitive" as the JS component type, which acts as a pass-through
    # factory that returns the config with the correct 'type' field.
    return SceneComponent("CustomPrimitive", data)


# =============================================================================
# Legend
# =============================================================================


class Legend(Plot.LayoutItem):
    """A standalone colormap legend, usable anywhere in a layout.

    Scenes render legends automatically for components with ``color_by``;
    this class renders the same legend UI outside a scene (e.g. next to a
    plot, or in a layout row).

    Example:
        >>> scene | Legend(cmap="viridis", domain=(0, 2.5), label="Cu %")
    """

    def __init__(
        self,
        cmap: str = "viridis",
        domain: Optional[Sequence[float]] = None,
        label: Optional[str] = None,
        categories: Optional[Sequence[str]] = None,
    ):
        """Initialize the legend.

        Args:
            cmap: Colormap name (continuous ramp or categorical palette).
            domain: (min, max) the colors span (continuous maps only).
            label: What the colors encode (legend title).
            categories: Display names for categorical codes 0..K-1.
        """
        super().__init__()
        self.spec = colormaps.colormap_metadata(
            cmap,
            domain=domain,
            label=label,
            categories=list(categories) if categories is not None else None,
        )

    def for_json(self) -> Any:
        """Convert to JSON representation for JavaScript."""
        return [Plot.JSRef("scene3d.Legend"), {"spec": self.spec}]


# =============================================================================
# Gizmo (Prototype - API may change)
# =============================================================================

# Type alias for gizmo parts
GizmoPart = Literal["x", "y", "z", "xy", "xz", "yz", "center"]

# Gizmo colors (matching JS GIZMO_COLORS)
GIZMO_COLORS: Dict[str, list] = {
    "x": [1.0, 0.2, 0.2],  # Red
    "y": [0.2, 1.0, 0.2],  # Green
    "z": [0.2, 0.4, 1.0],  # Blue
    "xy": [1.0, 1.0, 0.2],  # Yellow
    "xz": [0.2, 1.0, 1.0],  # Cyan
    "yz": [1.0, 0.2, 1.0],  # Magenta
    "center": [0.9, 0.9, 0.9],  # White
}

# Axis direction vectors
_AXIS_DIRECTIONS: Dict[str, list] = {
    "x": [1, 0, 0],
    "y": [0, 1, 0],
    "z": [0, 0, 1],
}

# Plane normal vectors
_PLANE_NORMALS: Dict[str, list] = {
    "xy": [0, 0, 1],
    "xz": [0, 1, 0],
    "yz": [1, 0, 0],
}


def _get_hover_props(part: str) -> HoverProps:
    """Get hoverProps for a gizmo part (brightens color on hover)."""
    base_color = GIZMO_COLORS[part]
    brightened = [min(1.0, c * 1.5) for c in base_color]
    return {"color": brightened}


def TranslateGizmo(
    position: ArrayLike,
    on_drag: Optional[Any] = None,
    axes: Optional[tuple[str, ...]] = None,
    planes: Optional[tuple[str, ...]] = None,
    show_center: bool = True,
    scale: float = 1.0,
) -> Scene:
    """Create a translate gizmo for interactive object translation.

    The gizmo consists of axis arrows, plane handles, and a center sphere
    that allow dragging objects along constrained axes or planes.

    Hover highlighting is automatic via hoverProps - no state management needed.

    NOTE: This is a prototype API and may change. Screen-space sizing is not
    yet implemented - use the `scale` parameter to adjust gizmo size manually.

    Args:
        position: World position [x, y, z] of the gizmo center
        on_drag: Callback fired during drag. Receives DragInfo with delta.position.
            Use info.delta.position to get the world-space translation.
        axes: Which axis handles to show. Default: ("x", "y", "z")
        planes: Which plane handles to show. Default: ("xy", "xz", "yz")
        show_center: Show center sphere for free drag. Default: True
        scale: Scale factor for gizmo size. Default: 1.0

    Returns:
        A Scene containing the gizmo components

    Example:
        >>> def handle_drag(info):
        ...     # Move object by drag delta
        ...     new_pos = [p + d for p, d in zip(obj_pos, info.delta.position)]
        ...     set_obj_pos(new_pos)
        ...
        >>> gizmo = TranslateGizmo(position=obj_pos, on_drag=handle_drag)
        >>> Scene(my_object, gizmo)
    """
    if axes is None:
        axes = ("x", "y", "z")
    if planes is None:
        planes = ("xy", "xz", "yz")

    pos = np.asarray(position, dtype=np.float32)
    components: list = []

    # Geometry dimensions (scaled)
    axis_length = 1.0 * scale
    shaft_width = 0.02 * scale
    cone_length = 0.15 * scale
    cone_radius = 0.06 * scale
    plane_size = 0.25 * scale
    plane_offset = 0.35 * scale
    center_radius = 0.08 * scale

    # Create axis arrows
    for axis in axes:
        direction = np.array(_AXIS_DIRECTIONS[axis], dtype=np.float32)
        base_color = GIZMO_COLORS[axis]
        hover_props = _get_hover_props(axis)

        # Axis shaft (LineBeams)
        shaft_end = pos + direction * (axis_length - cone_length)
        # LineBeams format: [x, y, z, lineIndex, ...]
        points = np.array(
            [pos[0], pos[1], pos[2], 0, shaft_end[0], shaft_end[1], shaft_end[2], 0],
            dtype=np.float32,
        )

        components.append(
            LineBeams(
                points=points,
                size=shaft_width,
                color=base_color,
                layer="overlay",
                drag_constraint=drag_axis(direction.tolist(), pos.tolist()),
                on_drag=on_drag,
                hover_props=hover_props,
                picking_scale=3.0,  # 3x larger hit area for thin axis shafts
            )
        )

        # Cone tip (Ellipsoid stretched along axis)
        cone_center = pos + direction * (axis_length - cone_length / 2)

        # Determine cone half-sizes based on axis
        if axis == "x":
            cone_half_size = [cone_length / 2, cone_radius, cone_radius]
        elif axis == "y":
            cone_half_size = [cone_radius, cone_length / 2, cone_radius]
        else:
            cone_half_size = [cone_radius, cone_radius, cone_length / 2]

        components.append(
            Ellipsoid(
                centers=[cone_center.tolist()],
                half_size=cone_half_size,
                color=base_color,
                layer="overlay",
                drag_constraint=drag_axis(direction.tolist(), pos.tolist()),
                on_drag=on_drag,
                hover_props=hover_props,
            )
        )

    # Create plane handles
    for plane in planes:
        normal = _PLANE_NORMALS[plane]
        base_color = GIZMO_COLORS[plane]
        hover_props = _get_hover_props(plane)

        # Position the plane handle at the intersection of the two axes
        if plane == "xy":
            handle_center = pos + np.array([plane_offset, plane_offset, 0])
            handle_half_size = [plane_size / 2, plane_size / 2, plane_size / 10]
        elif plane == "xz":
            handle_center = pos + np.array([plane_offset, 0, plane_offset])
            handle_half_size = [plane_size / 2, plane_size / 10, plane_size / 2]
        else:  # yz
            handle_center = pos + np.array([0, plane_offset, plane_offset])
            handle_half_size = [plane_size / 10, plane_size / 2, plane_size / 2]

        components.append(
            Ellipsoid(
                centers=[handle_center.tolist()],
                half_size=handle_half_size,
                color=base_color,
                layer="overlay",
                drag_constraint=drag_plane(normal, pos.tolist()),
                on_drag=on_drag,
                hover_props=hover_props,
            )
        )

    # Create center sphere
    if show_center:
        base_color = GIZMO_COLORS["center"]
        hover_props = _get_hover_props("center")
        components.append(
            Ellipsoid(
                centers=[pos.tolist()],
                half_size=[center_radius, center_radius, center_radius],
                color=base_color,
                layer="overlay",
                drag_constraint={"type": "free"},
                on_drag=on_drag,
                hover_props=hover_props,
            )
        )

    return Scene(*components)


__all__ = [
    "Scene",
    "PointCloud",
    "Ellipsoid",
    "Cuboid",
    "LineBeams",
    "LineSegments",
    "GridHelper",
    "CameraFrustum",
    "ImagePlane",
    "ImageProjection",
    "Group",
    "Mesh",
    "CustomPrimitive",
    "deco",
    # Colormaps & legends
    "ColorBy",
    "Legend",
    # Hover props
    "HoverProps",
    # Drag constraints
    "DragConstraint",
    "drag_axis",
    "drag_plane",
    "DRAG_AXIS_X",
    "DRAG_AXIS_Y",
    "DRAG_AXIS_Z",
    "DRAG_PLANE_XY",
    "DRAG_PLANE_XZ",
    "DRAG_PLANE_YZ",
    "DRAG_SURFACE",
    "DRAG_SCREEN",
    "DRAG_FREE",
    # Gizmo (prototype)
    "TranslateGizmo",
    "GIZMO_COLORS",
]
