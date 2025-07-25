# Common imports and configuration
import numpy as np
from colight.scene3d import Ellipsoid, Cuboid, LineBeams, PointCloud, deco

# Common camera parameters
DEFAULT_CAMERA = {"up": [0, 0, 1], "fov": 45, "near": 0.1, "far": 100}

# 1) Opaque Occlusion (Single Component)
print(
    "Test 1: Opaque Occlusion.\n"
    "Two overlapping ellipsoids: green (higher z) should occlude red."
)

scene_1 = Ellipsoid(
    centers=np.array([[-8, 1, 0], [-8, 1, 0.5]]),
    colors=np.array(
        [
            [1, 0, 0],  # red
            [0, 1, 0],  # green
        ]
    ),
    half_size=0.5,
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            # Place the camera diagonally so the two ellipsoids overlap in projection.
            "position": [-12, 2, 3],
            "target": [-8, 1, 0.25],
        }
    }
)

# 2) Transparent Blending (Single Component)
print(
    "Test 2: Transparent Blending.\n"
    "Overlapping blue and semi-transparent yellow ellipsoids should blend."
)

scene_2 = Ellipsoid(
    centers=np.array([[-8, -1, 0], [-8, -1, 0.5]]),
    colors=np.array(
        [
            [0, 0, 1],  # blue
            [1, 1, 0],
        ]
    ),
    alphas=np.array([1.0, 0.5]),
    half_size=0.5,
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [-12, -2, 3],
            "target": [-8, -1, 0.25],
        }
    }
)

# 3) Inter-Component Ordering
print(
    "Test 3: Inter-Component Ordering.\n"
    "Cyan point cloud should appear on top of orange cuboid due to render order."
)

cuboid_component = Cuboid(
    centers=[[-4, 0, 0.5]],
    colors=[[1, 0.5, 0]],
    alphas=[0.5],
    half_size=0.5,
)
pointcloud_component = PointCloud(
    centers=[[-4, 0, 0]],
    colors=[[0, 1, 1]],
    alphas=[0.8],
    size=0.2,
)
scene_3 = (
    cuboid_component
    + pointcloud_component
    + (
        {
            "defaultCamera": {
                **DEFAULT_CAMERA,
                "position": [-8, -2, 4],
                "target": [-4, 0, 0.5],
            }
        }
    )
)

# 4) Per-Instance Alpha Overrides (Point Cloud)
# ISSUE - blending only visible from one direction?
print(
    "Test 4: Per-Instance Alpha Overrides.\n"
    "Four vertical points with alternating opaque/semi-transparent alphas."
)

scene_4 = PointCloud(
    centers=[0, 2, 0, 0, 2, 0.5, 0, 2, 1.0, 0, 2, 1.5],
    colors=[1, 0, 1, 0, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5],
    alphas=[1.0, 0.5, 1.0, 0.5],
    size=0.1,
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [0.076180, 2.110072, -0.497107],
            "target": [0.000000, 2.000000, 0.750000],
        }
    }
)

# 5) Decoration Overrides (Cuboid)
print(
    "Test 5: Decoration Overrides.\n"
    "Three cuboids: middle one red, semi-transparent, and scaled up."
)

cuboid_centers = np.array([[2, -1, 0], [2, -1, 0.5], [2, -1, 1.0]])
cuboid_colors = np.array([[0.8, 0.8, 0.8], [0.8, 0.8, 0.8], [0.8, 0.8, 0.8]])
scene_5 = Cuboid(
    centers=cuboid_centers,
    colors=cuboid_colors,
    alphas=np.array([1.0, 1.0, 1.0]),
    half_size=0.4,
    decorations=[
        # Override instance index 1: change color to red, set alpha to 0.5, and scale up by 1.2.
        deco(1, color=[1.0, 0.0, 0.0], alpha=0.5, scale=1.2)
    ],
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [2.053780, 1.688640, 0.635353],
            "target": [2.000000, -1.000000, 0.500000],
        }
    }
)

# 6) Extreme Alpha Values (Ellipsoids)
# ISSUE: nearly invisible shows background color (occluding) rather than showing the other item
print(
    "Test 6: Extreme Alpha Values.\n"
    "Two ellipsoids: one nearly invisible (α=0.01), one almost opaque (α=0.99)."
)

scene_6 = Ellipsoid(
    centers=np.array([[-6, -2, 0], [-6, -2, 0.5]]),
    colors=np.array([[0.2, 0.2, 0.2], [0.9, 0.9, 0.9]]),
    alphas=np.array([0.5, 0.99]),
    half_size=0.4,
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [-6.101873, -4.035756, -2.471731],
            "target": [-6.000000, -2.000000, 0.250000],
        }
    }
)

# 7) Mixed Primitive Types Overlapping
print(
    "Test 7: Mixed Primitive Types.\n"
    "Overlapping red ellipsoid, green cuboid, yellow beam, and blue/magenta points."
)

mixed_ellipsoids = Ellipsoid(
    centers=np.array([[2, 2, 0.25]]),
    colors=np.array([[1, 0, 0]]),
    alphas=np.array([0.7]),
    half_size=[0.5, 0.5, 0.5],
)
mixed_cuboids = Cuboid(
    centers=np.array([[2, 2, 0]]),
    colors=np.array([[0, 1, 0]]),
    alphas=np.array([0.7]),
    half_size=[1, 1, 1],
)
mixed_linebeams = LineBeams(
    points=np.array([2, 2, -0.5, 0, 2, 2, 1, 0], dtype=np.float32),
    color=np.array([1, 1, 0]),
    size=0.05,
)
mixed_pointcloud = PointCloud(
    centers=np.array([[2, 2, -0.25], [2, 2, 0.75]]),
    colors=np.array([[0, 0, 1], [1, 0, 1]]),
    alphas=np.array([1.0, 0.8]),
    size=0.1,
)

scene_7 = (mixed_ellipsoids + mixed_cuboids + mixed_linebeams + mixed_pointcloud) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [6, 6, 4],
            "target": [2, 2, 0.25],
        }
    }
)

# 8) Insertion Order Conflict (Two PointClouds)
print(
    "Test 8: Insertion Order.\n"
    "Two overlapping point clouds: second one (purple) should appear on top of first (green)."
)

pointcloud_component1 = PointCloud(
    centers=np.array([[-2, 2, 0.2]]),
    colors=np.array([[0, 0.5, 0]]),
    size=0.15,
    alpha=0.5,
)
pointcloud_component2 = PointCloud(
    centers=np.array([[-2, 2, 0]]),
    colors=np.array([[0.5, 0, 0.5]]),
    size=0.15,
    alpha=0.5,
)

scene_8 = (pointcloud_component1 + pointcloud_component2) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [-2, 4, 3],
            "target": [-2, 2, 0.1],
        }
    }
)

(scene_1 & scene_2 & scene_3 | scene_4 & scene_5 & scene_6 | scene_7 & scene_8)
# %%
