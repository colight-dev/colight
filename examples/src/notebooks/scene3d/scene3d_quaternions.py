# %% Common imports and configuration
import numpy as np
from colight.scene3d import (
    Ellipsoid,
    Cuboid,
    deco,
)
from colight.plot import js

# Common camera parameters
DEFAULT_CAMERA = {"up": [0, 0, 1], "fov": 45, "near": 0.1, "far": 100}

# %% Helper functions for quaternion operations


def axis_angle_to_quat(axis, angle):
    """Convert axis-angle representation to quaternion [x,y,z,w]

    Args:
        axis: 3D vector representing rotation axis (will be normalized)
        angle: Rotation angle in radians

    Returns:
        Quaternion as [x,y,z,w]
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)

    qx = axis[0] * np.sin(angle / 2)
    qy = axis[1] * np.sin(angle / 2)
    qz = axis[2] * np.sin(angle / 2)
    qw = np.cos(angle / 2)

    return np.array([qx, qy, qz, qw])


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler angles to quaternion [x,y,z,w]

    Args:
        roll: Rotation around x-axis in radians
        pitch: Rotation around y-axis in radians
        yaw: Rotation around z-axis in radians

    Returns:
        Quaternion as [x,y,z,w]
    """
    # Roll (x-axis rotation)
    sinr_cosp = np.sin(roll / 2) * np.cos(pitch / 2)
    cosr_cosp = np.cos(roll / 2) * np.cos(pitch / 2)

    # Pitch (y-axis rotation)
    sinp = np.sin(pitch / 2)
    cosp = np.cos(pitch / 2)

    # Yaw (z-axis rotation)
    siny_cosp = np.sin(yaw / 2) * cosp

    qx = sinr_cosp * np.cos(yaw / 2) + cosr_cosp * siny_cosp
    qy = cosr_cosp * siny_cosp - sinr_cosp * np.cos(yaw / 2)
    qz = cosr_cosp * np.sin(yaw / 2) - sinr_cosp * sinp
    qw = cosr_cosp * np.cos(yaw / 2) + sinr_cosp * sinp

    return np.array([qx, qy, qz, qw])


# %% 1) Ellipsoids with different quaternion rotations
print("Demo 1: Ellipsoids with different quaternion rotations")

# Create a grid of ellipsoids with different orientations
x, y = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7))
centers = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))

# Create different quaternions for each ellipsoid
quaternions = np.zeros((centers.shape[0], 4))
for i in range(centers.shape[0]):
    # Create quaternions based on position
    # Rotate around different axes based on position
    x_pos = centers[i, 0]
    y_pos = centers[i, 1]

    # Normalize position to -1 to 1 range for angle calculation
    x_norm = x_pos / 3.0
    y_norm = y_pos / 3.0

    # Create quaternion from axis-angle
    # Vary the axis based on position
    axis = np.array([y_norm, x_norm, 1.0])
    angle = np.pi * (x_norm**2 + y_norm**2)

    quaternions[i] = axis_angle_to_quat(axis, angle)

# Generate colors based on quaternion values
colors = np.zeros((centers.shape[0], 3))
colors[:, 0] = np.abs(quaternions[:, 0])  # Red from x component
colors[:, 1] = np.abs(quaternions[:, 1])  # Green from y component
colors[:, 2] = np.abs(quaternions[:, 2])  # Blue from z component

# Create ellipsoids with different orientations
scene_ellipsoids = Ellipsoid(
    centers=centers,
    half_size=[0.4, 0.2, 0.1],  # Non-uniform half-size to show rotation clearly
    quaternions=quaternions,
    colors=colors,
    alpha=0.7,
    onHover=js(
        "(i) => $state.update({hover_ellipsoid: typeof i === 'number' ? [i] : []})"
    ),
    decorations=[
        deco(
            js("$state.hover_ellipsoid"),
            color=[1, 1, 0],
            scale=1.2,
        ),
    ],
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [0, 0, 10],
            "target": [0, 0, 0],
        }
    }
)

# %% 2) Ellipsoid Axes with quaternion rotations
print("Demo 2: Ellipsoid Axes with quaternion rotations")

# Create a row of ellipsoid axes with different orientations
centers = np.array([[-3, 0, 0], [-1, 0, 0], [1, 0, 0], [3, 0, 0]])

# Create different quaternions for each ellipsoid
quaternions = np.array(
    [
        axis_angle_to_quat([1, 0, 0], np.pi / 4),  # X-axis rotation
        axis_angle_to_quat([0, 1, 0], np.pi / 4),  # Y-axis rotation
        axis_angle_to_quat([0, 0, 1], np.pi / 4),  # Z-axis rotation
        euler_to_quat(np.pi / 6, np.pi / 4, np.pi / 3),  # Combined rotation
    ]
)

# Create ellipsoid axes with different orientations
scene_ellipsoid_axes = Ellipsoid(
    fill_mode="MajorWireframe",
    centers=centers,
    half_size=[0.5, 0.3, 0.2],  # Non-uniform half-size to show rotation clearly
    quaternions=quaternions,
    alpha=0.9,
    onHover=js("(i) => $state.update({hover_axes: typeof i === 'number' ? [i] : []})"),
    decorations=[
        deco(
            js("$state.hover_axes"),
            color=[1, 1, 0],
            scale=1.2,
        ),
    ],
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [0, -5, 2],
            "target": [0, 0, 0],
        }
    }
)

# %% 3) Cuboids with quaternion rotations
print("Demo 3: Cuboids with quaternion rotations")

# Create a spiral of cuboids with progressive rotations
t = np.linspace(0, 4 * np.pi, 20)
x = t * np.cos(t) * 0.2
y = t * np.sin(t) * 0.2
z = t * 0.1
centers = np.column_stack((x, y, z))

# Create progressively changing quaternions
quaternions = np.zeros((centers.shape[0], 4))
for i in range(centers.shape[0]):
    # Rotate around z-axis with increasing angle
    angle = i * np.pi / 10
    quaternions[i] = axis_angle_to_quat([1, 1, 1], angle)

# Generate colors based on position in spiral
colors = np.zeros((centers.shape[0], 3))
colors[:, 0] = np.linspace(0, 1, centers.shape[0])  # Red increases along spiral
colors[:, 1] = np.linspace(1, 0, centers.shape[0])  # Green decreases along spiral
colors[:, 2] = np.abs(
    np.sin(np.linspace(0, 3 * np.pi, centers.shape[0]))
)  # Blue oscillates

# Create cuboids with different orientations
scene_cuboids = Cuboid(
    centers=centers,
    half_size=[0.2, 0.1, 0.3],  # Non-uniform half-size to show rotation clearly
    quaternions=quaternions,
    colors=colors,
    alpha=0.8,
    onHover=js(
        "(i) => $state.update({hover_cuboid: typeof i === 'number' ? [i] : []})"
    ),
    decorations=[
        deco(
            js("$state.hover_cuboid"),
            color=[1, 1, 0],
            scale=1.2,
        ),
    ],
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [3, 3, 5],
            "target": [0, 0, 2],
        }
    }
)

# %% 4) Comparison of different rotation representations
print("Demo 4: Comparison of different rotation representations")

# Create a grid of cuboids to compare different rotation methods
centers = np.array(
    [
        [-2, -2, 0],  # No rotation (identity quaternion)
        [-2, 0, 0],  # X-axis rotation
        [-2, 2, 0],  # Y-axis rotation
        [0, -2, 0],  # Z-axis rotation
        [0, 0, 0],  # Axis-angle rotation
        [0, 2, 0],  # Euler angles rotation
        [2, -2, 0],  # Combined rotation 1
        [2, 0, 0],  # Combined rotation 2
        [2, 2, 0],  # Combined rotation 3
    ]
)

# Create different quaternions for each cuboid
quaternions = np.array(
    [
        [0, 0, 0, 1],  # Identity quaternion (no rotation)
        axis_angle_to_quat([1, 0, 0], np.pi / 2),  # 90° around X-axis
        axis_angle_to_quat([0, 1, 0], np.pi / 2),  # 90° around Y-axis
        axis_angle_to_quat([0, 0, 1], np.pi / 2),  # 90° around Z-axis
        axis_angle_to_quat([1, 1, 1], np.pi / 3),  # 60° around [1,1,1] axis
        euler_to_quat(np.pi / 4, np.pi / 4, np.pi / 4),  # 45° Euler angles
        euler_to_quat(np.pi / 2, 0, 0),  # 90° roll only
        euler_to_quat(0, np.pi / 2, 0),  # 90° pitch only
        euler_to_quat(0, 0, np.pi / 2),  # 90° yaw only
    ]
)

# Create labels for each rotation type
labels = [
    "Identity",
    "X-axis 90°",
    "Y-axis 90°",
    "Z-axis 90°",
    "Axis-angle [1,1,1] 60°",
    "Euler 45°,45°,45°",
    "Roll 90°",
    "Pitch 90°",
    "Yaw 90°",
]

# Generate colors based on rotation type
colors = np.zeros((centers.shape[0], 3))
colors[0] = [0.5, 0.5, 0.5]  # Gray for identity
colors[1:4] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB for X,Y,Z rotations
colors[4:6] = [[1, 0, 1], [0, 1, 1]]  # Magenta and cyan for other methods
colors[6:9] = [[1, 1, 0], [0.5, 0, 1], [1, 0.5, 0]]  # Various colors for combined

# Create cuboids with different orientations
scene_comparison = Cuboid(
    centers=centers,
    half_size=[0.3, 0.1, 0.5],  # Non-uniform half-size to show rotation clearly
    quaternions=quaternions,
    colors=colors,
    alpha=0.8,
    onHover=js(
        "(i) => $state.update({hover_comparison: typeof i === 'number' ? [i] : []})"
    ),
    decorations=[
        deco(
            js("$state.hover_comparison"),
            color=[1, 1, 1],
            scale=1.2,
        ),
    ],
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [0, -5, 5],
            "target": [0, 0, 0],
        }
    }
)

# %% 5) Combined scene with all demonstrations
print("Demo 5: Combined scene with all quaternion demonstrations")

# Combine all scenes with appropriate positioning
(
    # Ellipsoids grid (moved to left side)
    Ellipsoid(
        centers=centers,
        half_size=[0.4, 0.2, 0.1],
        quaternions=quaternions,
        colors=colors,
        alpha=0.7,
        onHover=js(
            "(i) => $state.update({hover_combined: typeof i === 'number' ? [i] : []})"
        ),
        decorations=[
            deco(
                js("$state.hover_combined"),
                color=[1, 1, 0],
                scale=1.2,
            ),
        ],
    )
    +
    # Cuboid spiral (moved to right side)
    Cuboid(
        centers=centers + np.array([0, 0, 2]),  # Shift right
        half_size=[0.3, 0.1, 0.5],
        quaternions=quaternions,
        colors=colors,
        alpha=0.8,
        onHover=js(
            "(i) => $state.update({hover_combined2: typeof i === 'number' ? [i] : []})"
        ),
        decorations=[
            deco(
                js("$state.hover_combined2"),
                color=[1, 1, 0],
                scale=1.2,
            ),
        ],
    )
    +
    # Camera settings for the combined scene
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [0, -10, 5],
            "target": [0, 0, 0],
        }
    }
)

# %% 6) Cubes rotating 180 degrees in small increments
print("Demo 6: Cubes rotating 180 degrees in small increments")

# Create a line of cubes with incremental rotations
num_cubes = 12  # Number of cubes in the line
centers = np.zeros((num_cubes, 3))
centers[:, 0] = np.linspace(-5, 5, num_cubes)  # Place cubes along x-axis

# Create quaternions for each cube, rotating from 0 to 180 degrees around the z-axis
quaternions = np.zeros((num_cubes, 4))
for i in range(num_cubes):
    # Calculate angle from 0 to 180 degrees (π radians)
    angle = (i / (num_cubes - 1)) * np.pi
    # Create quaternion for rotation around z-axis
    quaternions[i] = axis_angle_to_quat([0, 0, 1], angle)

# Generate colors that transition from blue to red based on rotation angle
colors = np.zeros((num_cubes, 3))
colors[:, 0] = np.linspace(0, 1, num_cubes)  # Red increases with rotation
colors[:, 2] = np.linspace(1, 0, num_cubes)  # Blue decreases with rotation

# Create cuboids with incremental rotations
scene_rotation_sequence = Cuboid(
    centers=centers,
    half_size=[0.2, 0.5, 0.1],  # Non-uniform half-size to show rotation clearly
    quaternions=quaternions,
    colors=colors,
    alpha=0.9,
    onHover=js(
        "(i) => $state.update({hover_rotation: typeof i === 'number' ? [i] : []})"
    ),
    decorations=[
        deco(
            js("$state.hover_rotation"),
            color=[1, 1, 0],
            scale=1.2,
        ),
    ],
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [0, -5, 3],
            "target": [0, 0, 0],
        }
    }
)

# %% 7) Cubes rotating 180 degrees around different axes
print("Demo 7: Cubes rotating 180 degrees around different axes")

# Create three rows of cubes, each rotating around a different axis
num_cubes_per_row = 8
total_cubes = num_cubes_per_row * 3
centers = np.zeros((total_cubes, 3))

# First row: rotation around x-axis
centers[0:num_cubes_per_row, 0] = np.linspace(-3.5, 3.5, num_cubes_per_row)
centers[0:num_cubes_per_row, 1] = -2

# Second row: rotation around y-axis
centers[num_cubes_per_row : 2 * num_cubes_per_row, 0] = np.linspace(
    -3.5, 3.5, num_cubes_per_row
)
centers[num_cubes_per_row : 2 * num_cubes_per_row, 1] = 0

# Third row: rotation around z-axis
centers[2 * num_cubes_per_row :, 0] = np.linspace(-3.5, 3.5, num_cubes_per_row)
centers[2 * num_cubes_per_row :, 1] = 2

# Create quaternions for each cube
quaternions = np.zeros((total_cubes, 4))
for i in range(num_cubes_per_row):
    # Calculate angle from 0 to 180 degrees (π radians)
    angle = (i / (num_cubes_per_row - 1)) * np.pi

    # X-axis rotation (first row)
    quaternions[i] = axis_angle_to_quat([1, 0, 0], angle)

    # Y-axis rotation (second row)
    quaternions[i + num_cubes_per_row] = axis_angle_to_quat([0, 1, 0], angle)

    # Z-axis rotation (third row)
    quaternions[i + 2 * num_cubes_per_row] = axis_angle_to_quat([0, 0, 1], angle)

# Generate colors based on row (axis of rotation)
colors = np.zeros((total_cubes, 3))
# X-axis row: red gradient
colors[0:num_cubes_per_row, 0] = 1
colors[0:num_cubes_per_row, 1] = np.linspace(0.2, 0.8, num_cubes_per_row)

# Y-axis row: green gradient
colors[num_cubes_per_row : 2 * num_cubes_per_row, 1] = 1
colors[num_cubes_per_row : 2 * num_cubes_per_row, 0] = np.linspace(
    0.2, 0.8, num_cubes_per_row
)

# Z-axis row: blue gradient
colors[2 * num_cubes_per_row :, 2] = 1
colors[2 * num_cubes_per_row :, 1] = np.linspace(0.2, 0.8, num_cubes_per_row)

# Create cuboids with incremental rotations around different axes
Cuboid(
    centers=centers,
    half_size=[0.2, 0.5, 0.1],  # Non-uniform half-size to show rotation clearly
    quaternions=quaternions,
    colors=colors,
    alpha=0.9,
    onHover=js(
        "(i) => $state.update({hover_multi_axis: typeof i === 'number' ? [i] : []})"
    ),
    decorations=[
        deco(
            js("$state.hover_multi_axis"),
            color=[1, 1, 0],
            scale=1.2,
        ),
    ],
) + (
    {
        "defaultCamera": {
            **DEFAULT_CAMERA,
            "position": [0, -8, 4],
            "target": [0, 0, 0],
        }
    }
)
