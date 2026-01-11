from typing import Any, Dict, Literal, Optional, TypedDict, Union

import numpy as np

import colight.plot as Plot
from colight.layout import JSExpr

# Move Array type definition after imports
ArrayLike = Union[list, np.ndarray, JSExpr]
NumberLike = Union[int, float, np.number, JSExpr]


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
        >>> Cuboid(centers=[[0,0,0]], onDrag=handle_drag, dragConstraint=drag_axis([0, 1, 0]))
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
        >>> Cuboid(centers=[[0,0,0]], onDrag=handle_drag, dragConstraint=drag_plane([0, 1, 0], [0, 0, 0]))
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


class Decoration(TypedDict, total=False):
    indexes: ArrayLike
    color: Optional[ArrayLike]  # [r,g,b]
    alpha: Optional[NumberLike]  # 0-1
    scale: Optional[NumberLike]  # scale factor


class HoverProps(TypedDict, total=False):
    """Properties to apply automatically when an instance is hovered."""

    color: Optional[ArrayLike]  # [r,g,b]
    alpha: Optional[NumberLike]  # 0-1
    scale: Optional[NumberLike]  # scale factor


def deco(
    indexes: Union[int, np.integer, ArrayLike],
    *,
    color: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,
    scale: Optional[NumberLike] = None,
) -> Decoration:
    """Create a decoration for scene components.

    Args:
        indexes: Single index or list of indices to decorate
        color: Optional RGB color override [r,g,b]
        alpha: Optional opacity value (0-1)
        scale: Optional scale factor

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

    return decoration  # type: ignore


class SceneComponent(Plot.LayoutItem):
    """Base class for all 3D scene components."""

    def __init__(self, type_name: str, data: Dict[str, Any], **kwargs):
        super().__init__()
        self.type = type_name
        self.props = {**data, **kwargs}

    def to_js_call(self) -> Any:
        """Convert the element to a JSCall representation."""
        return Plot.JSCall(f"scene3d.{self.type}", [self.props])

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
    ):
        """Initialize the scene.

        Args:
            *layers: Scene components and optional properties.
                Properties can include:
                - controls: List of controls to show. Currently supports ['fps']
        """

        self.layers = flatten_layers(layers)
        super().__init__()

    def __add__(self, other: Union[SceneComponent, "Scene", Dict[str, Any]]) -> "Scene":
        """Allow combining scenes with + operator."""
        if isinstance(other, Scene):
            return Scene(*self.layers, *other.layers)
        else:
            return Scene(*self.layers, other)

    def __radd__(self, other: Union[Dict[str, Any], JSExpr]) -> "Scene":
        """Allow combining scenes with + operator when dict or JSExpr is on the left."""
        return Scene(other, *self.layers)

    def for_json(self) -> Any:
        """Convert to JSON representation for JavaScript."""
        components = [
            e.to_js_call() if isinstance(e, SceneComponent) else e for e in self.layers
        ]
        return [Plot.JSRef("scene3d.Scene"), {"layers": components}]


def flatten_array(arr: Any, dtype: Any = np.float32) -> Any:
    """Flatten an array if it is a 2D array, otherwise return as is.

    Args:
        arr: The array to flatten.
        dtype: The desired data type of the array.

    Returns:
        A flattened array if input is 2D, otherwise the original array.
    """
    if isinstance(arr, (np.ndarray, list)):
        arr = np.asarray(arr, dtype=dtype)
        if arr.ndim == 2:
            return arr.flatten()
    return arr


def PointCloud(
    centers: ArrayLike,
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,  # Default RGB color for all points
    sizes: Optional[ArrayLike] = None,
    size: Optional[NumberLike] = None,  # Default size for all points
    alphas: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,  # Default alpha for all points
    layer: Optional[Literal["scene", "overlay"]] = None,
    hoverProps: Optional[HoverProps] = None,
    picking_scale: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a point cloud element.

    Args:
        centers: Nx3 array of point centers or flattened array
        colors: Nx3 array of RGB colors or flattened array (optional)
        color: Default RGB color [r,g,b] for all points if colors not provided
        sizes: N array of point sizes or flattened array (optional)
        size: Default size for all points if sizes not provided
        alphas: Array of alpha values per point (optional)
        alpha: Default alpha value for all points if alphas not provided
        layer: Render layer - "scene" (default) or "overlay" (renders in front, always visible)
        hoverProps: Properties to apply on hover (color, alpha, scale)
        picking_scale: Scale factor for picking hit area (values > 1 make clicking easier)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
    centers = flatten_array(centers, dtype=np.float32)
    data: Dict[str, Any] = {"centers": centers}

    if colors is not None:
        data["colors"] = flatten_array(colors, dtype=np.float32)
    if color is not None:
        data["color"] = color

    if sizes is not None:
        data["sizes"] = flatten_array(sizes, dtype=np.float32)
    if size is not None:
        data["size"] = size

    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    if alpha is not None:
        data["alpha"] = alpha

    if layer is not None:
        data["layer"] = layer

    if hoverProps is not None:
        data["hoverProps"] = hoverProps

    if picking_scale is not None:
        data["pickingScale"] = picking_scale

    return SceneComponent("PointCloud", data, **kwargs)


def Ellipsoid(
    centers: ArrayLike,
    half_sizes: Optional[ArrayLike] = None,
    half_size: Optional[Union[NumberLike, ArrayLike]] = None,  # Single value or [x,y,z]
    quaternions: Optional[ArrayLike] = None,  # Nx4 array of quaternions [x,y,z,w]
    quaternion: Optional[ArrayLike] = None,  # Default orientation quaternion [x,y,z,w]
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,  # Default RGB color for all ellipsoids
    alphas: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,  # Default alpha for all ellipsoids
    fill_mode: str
    | None = None,  # How the shape is drawn ("Solid" or "MajorWireframe")
    layer: Optional[Literal["scene", "overlay"]] = None,
    hoverProps: Optional[HoverProps] = None,
    picking_scale: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create an ellipsoid element.

    Args:
        centers: Nx3 array of ellipsoid centers or flattened array
        half_sizes: Nx3 array of half_sizes (x,y,z) or flattened array (optional)
        half_size: Default half_size (sphere) or [x,y,z] half_sizes (ellipsoid) if half_sizes not provided
        quaternions: Nx4 array of orientation quaternions [x,y,z,w] (optional)
        quaternion: Default orientation quaternion [x,y,z,w] if quaternions not provided
        colors: Nx3 array of RGB colors or flattened array (optional)
        color: Default RGB color [r,g,b] for all ellipsoids if colors not provided
        alphas: Array of alpha values per ellipsoid (optional)
        alpha: Default alpha value for all ellipsoids if alphas not provided
        fill_mode: How the shape is drawn. One of:
            - "Solid": Filled surface with solid color
            - "MajorWireframe": Three axis-aligned ellipse cross-sections
        layer: Render layer - "scene" (default) or "overlay" (renders in front, always visible)
        hoverProps: Properties to apply on hover (color, alpha, scale)
        picking_scale: Scale factor for picking hit area (values > 1 make clicking easier)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
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

    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    elif alpha is not None:
        data["alpha"] = alpha

    if fill_mode is not None:
        data["fill_mode"] = fill_mode

    if layer is not None:
        data["layer"] = layer

    if hoverProps is not None:
        data["hoverProps"] = hoverProps

    if picking_scale is not None:
        data["pickingScale"] = picking_scale

    return SceneComponent("Ellipsoid", data, **kwargs)


def Cuboid(
    centers: ArrayLike,
    half_sizes: Optional[ArrayLike] = None,
    half_size: Optional[Union[ArrayLike, NumberLike]] = None,
    quaternions: Optional[ArrayLike] = None,  # Nx4 array of quaternions [x,y,z,w]
    quaternion: Optional[ArrayLike] = None,  # Default orientation quaternion [x,y,z,w]
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,  # Default RGB color for all cuboids
    alphas: Optional[ArrayLike] = None,  # Per-cuboid alpha values
    alpha: Optional[NumberLike] = None,  # Default alpha for all cuboids
    layer: Optional[Literal["scene", "overlay"]] = None,
    hoverProps: Optional[HoverProps] = None,
    picking_scale: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a cuboid element.

    Args:
        centers: Nx3 array of cuboid centers or flattened array
        half_sizes: Nx3 array of half sizes (width,height,depth) or flattened array (optional)
        half_size: Default half size [w,h,d] for all cuboids if half_sizes not provided
        quaternions: Nx4 array of orientation quaternions [x,y,z,w] (optional)
        quaternion: Default orientation quaternion [x,y,z,w] if quaternions not provided
        colors: Nx3 array of RGB colors or flattened array (optional)
        color: Default RGB color [r,g,b] for all cuboids if colors not provided
        alphas: Array of alpha values per cuboid (optional)
        alpha: Default alpha value for all cuboids if alphas not provided
        layer: Render layer - "scene" (default) or "overlay" (renders in front, always visible)
        hoverProps: Properties to apply on hover (color, alpha, scale)
        picking_scale: Scale factor for picking hit area (values > 1 make clicking easier)
        **kwargs: Additional arguments like decorations, onHover, onClick
    """
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

    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    elif alpha is not None:
        data["alpha"] = alpha

    if layer is not None:
        data["layer"] = layer

    if hoverProps is not None:
        data["hoverProps"] = hoverProps

    if picking_scale is not None:
        data["pickingScale"] = picking_scale

    return SceneComponent("Cuboid", data, **kwargs)


def LineBeams(
    points: ArrayLike,  # Array of quadruples [x,y,z,i, x,y,z,i, ...]
    color: Optional[ArrayLike] = None,  # Default RGB color for all beams
    size: Optional[NumberLike] = None,  # Default size for all beams
    colors: Optional[ArrayLike] = None,  # Per-line colors
    sizes: Optional[ArrayLike] = None,  # Per-line sizes
    alpha: Optional[NumberLike] = None,  # Default alpha for all beams
    alphas: Optional[ArrayLike] = None,  # Per-line alpha values
    layer: Optional[Literal["scene", "overlay"]] = None,
    hoverProps: Optional[HoverProps] = None,
    picking_scale: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a line beams element.

    Args:
        points: Array of quadruples [x,y,z,i, x,y,z,i, ...] where points sharing the same i value are connected in sequence
        color: Default RGB color [r,g,b] for all beams if colors not provided
        size: Default size for all beams if sizes not provided
        colors: Array of RGB colors per line (optional)
        sizes: Array of sizes per line (optional)
        alpha: Default alpha value for all beams if alphas not provided
        alphas: Array of alpha values per line (optional)
        layer: Render layer - "scene" (default) or "overlay" (renders in front, always visible)
        hoverProps: Properties to apply on hover (color, alpha, scale)
        picking_scale: Scale factor for picking hit area (values > 1 make clicking easier)
        **kwargs: Additional arguments like onHover, onClick

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
    if sizes is not None:
        data["sizes"] = flatten_array(sizes, dtype=np.float32)
    if alphas is not None:
        data["alphas"] = flatten_array(alphas, dtype=np.float32)
    elif alpha is not None:
        data["alpha"] = alpha

    if layer is not None:
        data["layer"] = layer

    if hoverProps is not None:
        data["hoverProps"] = hoverProps

    if picking_scale is not None:
        data["pickingScale"] = picking_scale

    return SceneComponent("LineBeams", data, **kwargs)


# =============================================================================
# Groups (Hierarchical Transforms)
# =============================================================================


def Group(
    children: list,
    position: Optional[ArrayLike] = None,
    quaternion: Optional[ArrayLike] = None,
    scale: Optional[Union[NumberLike, ArrayLike]] = None,
    name: Optional[str] = None,
) -> SceneComponent:
    """Create a group component for hierarchical scene composition.

    Groups apply a transform (position, rotation, scale) to all their children.
    At render time, groups are flattened into transformed primitives.

    Args:
        children: List of child components (can include nested groups)
        position: Position offset [x, y, z] in parent space
        quaternion: Rotation as quaternion [x, y, z, w]
        scale: Scale factor (uniform number or [x, y, z] per-axis)
        name: Optional name for identifying this group in pick info

    Returns:
        A Group scene component

    Example:
        >>> # Create a group of objects offset by [1, 0, 0]
        >>> Group(
        ...     children=[
        ...         Ellipsoid(centers=[[0, 0, 0]], half_size=0.1),
        ...         Cuboid(centers=[[0.5, 0, 0]], half_size=0.05),
        ...     ],
        ...     position=[1, 0, 0],
        ... )
    """

    def _coerce_group_value(value: ArrayLike | NumberLike) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, tuple):
            return list(value)
        return value

    data: Dict[str, Any] = {"children": children}

    if position is not None:
        data["position"] = _coerce_group_value(position)
    if quaternion is not None:
        data["quaternion"] = _coerce_group_value(quaternion)
    if scale is not None:
        data["scale"] = _coerce_group_value(scale)
    if name is not None:
        data["name"] = name

    return SceneComponent("Group", data)


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
                dragConstraint=drag_axis(direction.tolist(), pos.tolist()),
                onDrag=on_drag,
                hoverProps=hover_props,
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
                dragConstraint=drag_axis(direction.tolist(), pos.tolist()),
                onDrag=on_drag,
                hoverProps=hover_props,
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
                dragConstraint=drag_plane(normal, pos.tolist()),
                onDrag=on_drag,
                hoverProps=hover_props,
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
                dragConstraint={"type": "free"},
                onDrag=on_drag,
                hoverProps=hover_props,
            )
        )

    return Scene(*components)


__all__ = [
    "Scene",
    "PointCloud",
    "Ellipsoid",
    "Cuboid",
    "LineBeams",
    "Group",
    "deco",
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
    # Gizmo (prototype)
    "TranslateGizmo",
    "GIZMO_COLORS",
]
