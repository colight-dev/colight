import numpy as np
from colight.scene3d import PointCloud, Ellipsoid, Cuboid, deco
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
    centers = np.column_stack([x, y, z])

    # Create rainbow colors
    hue = t / t.max()
    colors = np.zeros((n_points, 3))
    # Red component
    colors[:, 0] = np.clip(1.5 - abs(3.0 * hue - 1.5), 0, 1)
    # Green component
    colors[:, 1] = np.clip(1.5 - abs(3.0 * hue - 3.0), 0, 1)
    # Blue component
    colors[:, 2] = np.clip(1.5 - abs(3.0 * hue - 4.5), 0, 1)

    # Create varying scales for points
    sizes = 0.01 + 0.02 * np.sin(t)

    # Create quaternion rotations for ellipsoids
    def axis_angle_to_quat(axis, angle):
        axis = axis / np.linalg.norm(axis)
        s = math.sin(angle / 2)
        return np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])

    # Different rotation quaternions for each ellipsoid
    ellipsoid_quats = np.array(
        [
            axis_angle_to_quat([1, 1, 0], math.pi / 4),  # 45 degrees around [1,1,0]
            axis_angle_to_quat([0, 1, 1], math.pi / 3),  # 60 degrees around [0,1,1]
            axis_angle_to_quat([1, 0, 1], math.pi / 6),  # 30 degrees around [1,0,1]
        ]
    )

    # Create the base scene with shared elements
    base_scene = (
        PointCloud(
            centers,
            colors,
            sizes,
            onHover=Plot.js("(i) => $state.update({hover_point: i})"),
            decorations=[
                {
                    "indexes": Plot.js(
                        "$state.hover_point ? [$state.hover_point] : []"
                    ),
                    "color": [1, 1, 0],
                    "scale": 1.5,
                }
            ],
        )
        +
        # Ellipsoids with one highlighted and rotations
        Ellipsoid(
            centers=[0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0],
            half_sizes=[0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.15, 0.15, 0.15],
            colors=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            quaternions=ellipsoid_quats,
            decorations=[deco([1], color=[1, 1, 0], alpha=0.8)],
        )
        +
        # Ellipsoid bounds with transparency and rotations
        Ellipsoid(
            fill_mode="MajorWireframe",
            centers=[0.8, 0.0, 0.0, -0.8, 0.0, 0.0],
            half_sizes=[0.2, 0.1, 0.1, 0.1, 0.2, 0.1],
            colors=[1.0, 0.5, 0.0, 0.0, 0.5, 1.0],
            quaternions=[
                axis_angle_to_quat([1, 0, 0], math.pi / 3),
                axis_angle_to_quat([0, 1, 0], math.pi / 4),
            ],
            decorations=[deco([0, 1], alpha=0.5)],
        )
        +
        # Cuboids with one enlarged
        Cuboid(
            centers=[0.0, -0.8, 0.0, 0.0, -0.8, 0.3],
            half_sizes=[0.15, 0.05, 0.1, 0.1, 0.05, 0.1],
            colors=[0.8, 0.2, 0.8, 0.2, 0.8, 0.8],
            quaternions=[
                axis_angle_to_quat([0, 0, 1], math.pi / 6),
                axis_angle_to_quat([1, 1, 1], math.pi / 4),
            ],
            decorations=[deco([0], scale=1.2)],
        )
    )
    controlled_camera = {
        "camera": Plot.js("$state.camera"),
        "onCameraChange": Plot.js("(camera) => $state.update({camera})"),
    }

    # Create a layout with two scenes side by side
    scene = (
        (base_scene + controlled_camera & base_scene + controlled_camera)
        | base_scene
        + {
            "defaultCamera": {
                "position": [1.388039, 0.091859, 0.189095],
                "target": [0.000000, 0.000000, 0.000000],
                "up": [0.000000, 1.000000, 0.000000],
                "fov": 45,
            }
        }
        | Plot.initialState(
            {
                "camera": {
                    "position": [
                        1.5 * math.sin(0.2) * math.sin(1.0),  # x
                        1.5 * math.cos(1.0),  # y
                        1.5 * math.sin(0.2) * math.cos(1.0),  # z
                    ],
                    "target": [0, 0, 0],
                    "up": [0, 1, 0],
                    "fov": math.degrees(math.pi / 3),
                    "near": 0.01,
                    "far": 100.0,
                }
            }
        )
    )

    return scene


create_demo_scene()
