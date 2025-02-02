import numpy as np
from genstudio.scene3d import Ellipsoid, EllipsoidAxes, PointCloud, LineBeams, Cuboid
import genstudio.plot as Plot
import math


def create_depth_test_scene():
    """Create a scene with predictable depth test cases"""

    # Create a row of ellipsoids with different alphas, same component
    ellipsoid_centers = np.array(
        [
            [-2, 0, 0],  # Alpha 1.0
            [-2, 0, 0.5],  # Alpha 0.1
            [-2, 0, 1],  # Alpha 0.5
            [-2, 0, 1.5],  # Alpha 0.95
        ]
    )

    ellipsoid_colors = np.array(
        [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
        ]
    )

    ellipsoid_alphas = np.array([1.0, 0.1, 0.5, 0.95])

    # Create a row of cuboids with different alphas, same component
    cuboid_centers = np.array(
        [
            [-4, 0, 0],  # Alpha 1.0
            [-4, 0, 0.5],  # Alpha 0.1
            [-4, 0, 1],  # Alpha 0.5
            [-4, 0, 1.5],  # Alpha 0.95
        ]
    )

    cuboid_colors = np.array(
        [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
        ]
    )

    scene = (
        # First set of ellipsoids in same component
        Ellipsoid(
            centers=ellipsoid_centers,
            colors=ellipsoid_colors,
            alphas=ellipsoid_alphas,
            radius=[0.2, 0.2, 0.2],
        )
        +
        # First set of cuboids in same component
        Cuboid(
            centers=cuboid_centers,
            colors=cuboid_colors,
            alphas=ellipsoid_alphas,
            size=[0.4, 0.4, 0.4],
        )
        +
        # Second set - identical ellipsoids in separate components
        EllipsoidAxes(
            centers=np.array([[0, 0, 0]]),
            colors=np.array([[1, 0, 0]]),  # Red alpha=1.0
            radius=[0.2, 0.2, 0.2],
        )
        + EllipsoidAxes(
            centers=np.array([[0, 0, 0.5]]),
            colors=np.array([[0, 1, 0]]),  # Green alpha=0.5
            alphas=np.array([0.5]),
            radius=[0.2, 0.2, 0.2],
        )
        +
        # Second set - identical cuboids in separate components
        Cuboid(
            centers=np.array([[1, 0, 0]]),
            colors=np.array([[1, 0, 0]]),  # Red alpha=1.0
            size=[0.4, 0.4, 0.4],
        )
        + Cuboid(
            centers=np.array([[1, 0, 0.5]]),
            colors=np.array([[0, 1, 0]]),  # Green alpha=0.5
            alphas=np.array([0.5]),
            size=[0.4, 0.4, 0.4],
        )
        +
        # Third set - line beams passing through
        LineBeams(
            positions=np.array(
                [
                    2,
                    0,
                    -0.5,
                    0,  # Start point
                    2,
                    0,
                    2.0,
                    0,  # End point
                ],
                dtype=np.float32,
            ),
            color=np.array([1.0, 1.0, 1.0]),  # White
            radius=0.05,
        )
        +
        # Fourth set - point cloud passing through
        PointCloud(
            positions=np.array([[2, 0, 0], [2, 0, 0.5], [2, 0, 1.0], [2, 0, 1.5]]),
            colors=np.array(
                [
                    [1, 0, 0],  # Red, no alpha
                    [0, 1, 0],  # Green, alpha 0.1
                    [0, 0, 1],  # Blue, alpha 0.5
                    [1, 1, 0],  # Yellow, alpha 0.95
                ]
            ),
            alphas=np.array([1.0, 0.1, 0.5, 0.95]),
            size=0.2,
        )
    ) | Plot.initialState(
        {
            "camera": {
                "position": [5, 5, 5],
                "target": [0, 0, 0],
                "up": [0, 1, 0],
                "fov": math.degrees(math.pi / 3),
                "near": 0.01,
                "far": 100.0,
            }
        }
    )

    return scene


create_depth_test_scene()
