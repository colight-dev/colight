"""R3F (React Three Fiber) 3D visualization module for Colight.

This module provides similar primitives to colight.scene3d but uses
React Three Fiber (Three.js) instead of custom WebGPU rendering.

Example usage:
    from colight import r3f
    import numpy as np

    # Create a point cloud
    points = np.random.randn(100, 3).astype(np.float32)
    r3f.Scene(
        r3f.PointCloud(points, color=[1, 0, 0]),
        r3f.Cuboid([[0, 0, 0]], half_size=0.5, color=[0, 1, 0]),
    )
"""

from typing import Any, Dict, Optional, TypedDict, Union

import numpy as np

import colight.plot as Plot
from colight.layout import JSExpr

# Type aliases
ArrayLike = Union[list, np.ndarray, JSExpr]
NumberLike = Union[int, float, np.number, JSExpr]


class Decoration(TypedDict, total=False):
    indexes: ArrayLike
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
    """Create a decoration for R3F scene components.

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

    # Create base decoration dict
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
    """Base class for all R3F scene components."""

    def __init__(self, type_name: str, data: Dict[str, Any], **kwargs):
        super().__init__()
        self.type = type_name
        self.props = {**data, **kwargs}

    def to_js_call(self) -> Any:
        """Convert the element to a JSCall representation."""
        return Plot.JSCall(f"r3f.{self.type}", [self.props])

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

        This method does not modify the current SceneComponent instance.
        Instead, it creates and returns a *new* SceneComponent instance.

        Args:
            new_props: An optional dictionary of new properties to merge.
            **kwargs: Additional keyword arguments representing properties to merge.

        Returns:
            A new SceneComponent instance with all properties merged.
        """
        merged_props = {**self.props}
        if new_props:
            merged_props.update(new_props)
        merged_props.update(kwargs)
        return SceneComponent(self.type, merged_props)


def flatten_layers(layers):
    flattened = []
    for layer in layers:
        if isinstance(layer, Scene):
            flattened.extend(flatten_layers(layer.layers))
        else:
            flattened.append(layer)
    return flattened


class Scene(Plot.LayoutItem):
    """A 3D scene visual component using React Three Fiber.

    This class creates an interactive 3D scene that can contain multiple types of components:

    - Point clouds
    - Ellipsoids
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
        return [Plot.JSRef("r3f.SceneWithLayers"), {"layers": components}]


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
    color: Optional[ArrayLike] = None,
    sizes: Optional[ArrayLike] = None,
    size: Optional[NumberLike] = None,
    alphas: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,
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

    return SceneComponent("PointCloud", data, **kwargs)


def Ellipsoid(
    centers: ArrayLike,
    half_sizes: Optional[ArrayLike] = None,
    half_size: Optional[Union[NumberLike, ArrayLike]] = None,
    quaternions: Optional[ArrayLike] = None,
    quaternion: Optional[ArrayLike] = None,
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,
    alphas: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create an ellipsoid element.

    Args:
        centers: Nx3 array of ellipsoid centers or flattened array
        half_sizes: Nx3 array of half_sizes (x,y,z) or flattened array (optional)
        half_size: Default half_size (sphere) or [x,y,z] half_sizes (ellipsoid)
        quaternions: Nx4 array of orientation quaternions [x,y,z,w] (optional)
        quaternion: Default orientation quaternion [x,y,z,w]
        colors: Nx3 array of RGB colors or flattened array (optional)
        color: Default RGB color [r,g,b] for all ellipsoids
        alphas: Array of alpha values per ellipsoid (optional)
        alpha: Default alpha value for all ellipsoids
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

    return SceneComponent("Ellipsoid", data, **kwargs)


def Cuboid(
    centers: ArrayLike,
    half_sizes: Optional[ArrayLike] = None,
    half_size: Optional[Union[ArrayLike, NumberLike]] = None,
    quaternions: Optional[ArrayLike] = None,
    quaternion: Optional[ArrayLike] = None,
    colors: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,
    alphas: Optional[ArrayLike] = None,
    alpha: Optional[NumberLike] = None,
    **kwargs: Any,
) -> SceneComponent:
    """Create a cuboid element.

    Args:
        centers: Nx3 array of cuboid centers or flattened array
        half_sizes: Nx3 array of half sizes (width,height,depth) or flattened array
        half_size: Default half size [w,h,d] for all cuboids
        quaternions: Nx4 array of orientation quaternions [x,y,z,w] (optional)
        quaternion: Default orientation quaternion [x,y,z,w]
        colors: Nx3 array of RGB colors or flattened array (optional)
        color: Default RGB color [r,g,b] for all cuboids
        alphas: Array of alpha values per cuboid (optional)
        alpha: Default alpha value for all cuboids
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

    return SceneComponent("Cuboid", data, **kwargs)


__all__ = [
    "Scene",
    "PointCloud",
    "Ellipsoid",
    "Cuboid",
    "deco",
]
