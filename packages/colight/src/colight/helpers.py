"""Composite helper components built from primitives.

These are convenience functions that generate common 3D visualization aids
like grids, camera frustums, and image projections.
"""

from typing import Any, Dict, Literal, Optional, Union

import numpy as np

from colight.layout import JSExpr

# Type aliases
ArrayLike = Union[list, np.ndarray, JSExpr]
NumberLike = Union[int, float, np.number, JSExpr]


def flatten_array(arr: Any, dtype: Any = np.float32) -> Any:
    """Flatten an array if it is a 2D array, otherwise return as is."""
    if isinstance(arr, (np.ndarray, list)):
        arr = np.asarray(arr, dtype=dtype)
        if arr.ndim == 2:
            return arr.flatten()
    return arr


def coerce_image_array(
    image: Any,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Any:
    """Normalize image data for ImagePlane/ImageProjection.

    Returns a dict with {data, width, height, channels} or passes through JSExpr.
    """
    if isinstance(image, JSExpr):
        return image
    if (
        isinstance(image, dict)
        and "data" in image
        and "width" in image
        and "height" in image
    ):
        return image
    if isinstance(image, (np.ndarray, list)):
        arr = np.asarray(image)
        if arr.ndim == 1:
            if image_width is None or image_height is None:
                raise ValueError(
                    "Flat image data requires image_width and image_height."
                )
            channels = (
                int(arr.size / (image_width * image_height)) if image_width else 0
            )
            arr = arr.reshape((image_height, image_width, channels))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3:
            raise ValueError("Image data must be HxW or HxWxC.")

        height, width = arr.shape[:2]
        channels = arr.shape[2]
        if channels not in (1, 3, 4):
            raise ValueError("Image data must have 1, 3, or 4 channels.")
        if channels == 1:
            arr = np.repeat(arr, 3, axis=-1)
            channels = 3

        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = arr * 255.0

        arr = arr.astype(np.uint8)
        return {
            "data": arr.flatten(),
            "width": width,
            "height": height,
            "channels": channels,
        }
    return image


# Import SceneComponent here to avoid circular imports
# We use a late import pattern
def _get_scene_component():
    from colight.scene3d import SceneComponent

    return SceneComponent


# =============================================================================
# GridHelper
# =============================================================================


def GridHelper(
    size: NumberLike = 10,
    divisions: int = 10,
    color: Optional[ArrayLike] = None,
    center_color: Optional[ArrayLike] = None,
    line_width: NumberLike = 0.005,
    layer: Optional[Literal["scene", "overlay"]] = None,
    **kwargs: Any,
) -> Any:
    """Create an XZ grid helper.

    Args:
        size: Total size of the grid
        divisions: Number of divisions
        color: Grid line color [r,g,b]
        center_color: Center line color [r,g,b]
        line_width: Width of grid lines
        layer: Render layer - "scene" (default) or "overlay"

    Returns:
        A GridHelper scene component
    """
    SceneComponent = _get_scene_component()

    data: Dict[str, Any] = {
        "size": size,
        "divisions": divisions,
        "lineWidth": line_width,
    }
    if color is not None:
        data["color"] = list(color) if hasattr(color, "__iter__") else color  # type: ignore[arg-type]
    if center_color is not None:
        data["centerColor"] = (
            list(center_color) if hasattr(center_color, "__iter__") else center_color  # type: ignore[arg-type]
        )
    if layer is not None:
        data["layer"] = layer

    return SceneComponent("GridHelper", data, **kwargs)


# =============================================================================
# CameraFrustum
# =============================================================================


def CameraFrustum(
    intrinsics: Dict[str, Any],
    extrinsics: Dict[str, Any],
    near: NumberLike = 0.1,
    far: NumberLike = 1.0,
    color: Optional[ArrayLike] = None,
    line_width: NumberLike = 0.005,
    layer: Optional[Literal["scene", "overlay"]] = None,
    **kwargs: Any,
) -> Any:
    """Create a camera frustum helper.

    Args:
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy, width, height
        extrinsics: Camera extrinsics dict with position [x,y,z] and quaternion [x,y,z,w]
        near: Near plane distance
        far: Far plane distance
        color: Frustum line color [r,g,b]
        line_width: Width of frustum lines
        layer: Render layer - "scene" (default) or "overlay"

    Returns:
        A CameraFrustum scene component
    """
    SceneComponent = _get_scene_component()

    data: Dict[str, Any] = {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "near": near,
        "far": far,
        "lineWidth": line_width,
    }
    if color is not None:
        data["color"] = list(color) if hasattr(color, "__iter__") else color  # type: ignore[arg-type]
    if layer is not None:
        data["layer"] = layer

    return SceneComponent("CameraFrustum", data, **kwargs)


# =============================================================================
# ImageProjection
# =============================================================================


def ImageProjection(
    image: Any,
    intrinsics: Dict[str, Any],
    extrinsics: Dict[str, Any],
    depth: NumberLike = 1.0,
    color: Optional[ArrayLike] = None,
    opacity: Optional[NumberLike] = None,
    show_frustum: bool = False,
    frustum_color: Optional[ArrayLike] = None,
    line_width: NumberLike = 0.005,
    image_key: Optional[Union[str, int]] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
    layer: Optional[Literal["scene", "overlay"]] = None,
    **kwargs: Any,
) -> Any:
    """Create an image projection (plane + optional frustum lines).

    Args:
        image: Image data (numpy array, PIL image, etc.)
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy, width, height
        extrinsics: Camera extrinsics dict with position [x,y,z] and quaternion [x,y,z,w]
        depth: Distance from camera to image plane
        color: Tint color for the image [r,g,b]
        opacity: Opacity of the image plane (0-1)
        show_frustum: Whether to show frustum lines from camera to image
        frustum_color: Color of frustum lines [r,g,b]
        line_width: Width of frustum lines
        image_key: Key for caching/updating image
        image_width: Width if providing flat image data
        image_height: Height if providing flat image data
        layer: Render layer - "scene" (default) or "overlay"

    Returns:
        An ImageProjection scene component
    """
    SceneComponent = _get_scene_component()

    data: Dict[str, Any] = {
        "image": coerce_image_array(image, image_width, image_height),
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "depth": depth,
        "showFrustum": show_frustum,
        "lineWidth": line_width,
    }
    if color is not None:
        data["color"] = list(color) if hasattr(color, "__iter__") else color  # type: ignore[arg-type]
    if opacity is not None:
        data["opacity"] = opacity
    if frustum_color is not None:
        data["frustumColor"] = (
            list(frustum_color) if hasattr(frustum_color, "__iter__") else frustum_color  # type: ignore[arg-type]
        )
    if image_key is not None:
        data["imageKey"] = image_key
    if layer is not None:
        data["layer"] = layer

    return SceneComponent("ImageProjection", data, **kwargs)


__all__ = [
    "GridHelper",
    "CameraFrustum",
    "ImageProjection",
    # Utility functions
    "flatten_array",
    "coerce_image_array",
]
