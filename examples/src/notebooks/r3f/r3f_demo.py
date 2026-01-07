"""R3F (React Three Fiber) Demo

Demonstrates the r3f module which provides similar primitives to scene3d
but uses React Three Fiber (Three.js) instead of custom WebGPU rendering.
"""

import numpy as np
from colight import r3f
import colight.plot as Plot
import math


def create_demo_scene():
    """Create a demo scene with examples of all element types."""
    # 1. Create a point cloud in a spiral pattern
    n_points = 1000
    t = np.linspace(0, 10 * np.pi, n_points)
    r = t / 30
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = t / 10

    # Create positions array
    centers = np.column_stack([x, y, z]).astype(np.float32)

    # Create rainbow colors
    hue = t / t.max()
    colors = np.zeros((n_points, 3), dtype=np.float32)
    # Red component
    colors[:, 0] = np.clip(1.5 - abs(3.0 * hue - 1.5), 0, 1)
    # Green component
    colors[:, 1] = np.clip(1.5 - abs(3.0 * hue - 3.0), 0, 1)
    # Blue component
    colors[:, 2] = np.clip(1.5 - abs(3.0 * hue - 4.5), 0, 1)

    # Create varying sizes for points
    sizes = (0.01 + 0.02 * np.sin(t)).astype(np.float32)

    # Create quaternion rotations for ellipsoids
    def axis_angle_to_quat(axis, angle):
        axis = np.array(axis) / np.linalg.norm(axis)
        s = math.sin(angle / 2)
        return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])

    # Different rotation quaternions for each ellipsoid
    ellipsoid_quats = np.array(
        [
            axis_angle_to_quat([1, 1, 0], math.pi / 4),  # 45 degrees around [1,1,0]
            axis_angle_to_quat([0, 1, 1], math.pi / 3),  # 60 degrees around [0,1,1]
            axis_angle_to_quat([1, 0, 1], math.pi / 6),  # 30 degrees around [1,0,1]
        ],
        dtype=np.float32,
    )

    scene = r3f.Scene(
        # Point cloud with rainbow colors
        r3f.PointCloud(
            centers,
            colors,
            sizes,
            onHover=Plot.js("(i) => $state.update({hover_point: i})"),
            decorations=[
                {
                    "indexes": Plot.js("$state.hover_point != null ? [$state.hover_point] : []"),
                    "color": [1, 1, 0],
                    "scale": 1.5,
                }
            ],
        ),
        # Ellipsoids with rotations
        r3f.Ellipsoid(
            centers=np.array(
                [[0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32
            ),
            half_sizes=np.array(
                [[0.1, 0.2, 0.1], [0.2, 0.1, 0.1], [0.15, 0.15, 0.15]], dtype=np.float32
            ),
            colors=np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            ),
            quaternions=ellipsoid_quats,
            decorations=[r3f.deco([1], color=[1, 1, 0], alpha=0.8)],
        ),
        # Cuboids with rotations
        r3f.Cuboid(
            centers=np.array([[0.0, -0.8, 0.0], [0.0, -0.8, 0.3]], dtype=np.float32),
            half_sizes=np.array([[0.15, 0.05, 0.1], [0.1, 0.05, 0.1]], dtype=np.float32),
            colors=np.array([[0.8, 0.2, 0.8], [0.2, 0.8, 0.8]], dtype=np.float32),
            quaternions=np.array(
                [
                    axis_angle_to_quat([0, 0, 1], math.pi / 6),
                    axis_angle_to_quat([1, 1, 1], math.pi / 4),
                ],
                dtype=np.float32,
            ),
            decorations=[r3f.deco([0], scale=1.2)],
        ),
        # Scene options
        {
            "defaultCamera": {
                "position": [1.5, 1.0, 1.5],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 45,
            }
        },
    )

    return scene


create_demo_scene()
