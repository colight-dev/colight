"""
Demo: Group Features

This notebook demonstrates all Group capabilities in a single interactive scene:
- child_overrides: Force props on all children (group wins)
- child_defaults: Provide default props (child wins if specified)
- hover_props: Apply to ALL children when ANY is hovered
- on_click/on_hover: Event handlers bubble up from children
- transforms: position, quaternion, scale
"""

import numpy as np
from colight import plot as Plot
from colight.plot import js
from colight.scene3d import PointCloud, Ellipsoid, Cuboid, Group, Scene

# Drag constraint: screen-space (plane facing camera)
DRAG_SCREEN = {"type": "screen"}


# Helper: create a small cluster of points
def point_cluster(offset, n=16):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 0.15
    return np.column_stack(
        [r * np.cos(t) + offset[0], r * np.sin(t) + offset[1], np.zeros(n) + offset[2]]
    )


# =============================================================================
# Interactive Demo: Draggable, Selectable Groups
# =============================================================================

# Each group:
# - Draggable in screen-space (plane facing camera)
# - Click to toggle selection (outline appears)
# - Hover to highlight (color change + scale)
# - Demonstrates child_overrides for group-wide color


# Initial state: no selection, initial positions
initial_state = {
    "selected": [],
    "positions": {
        "red": [-0.5, 0, 0],
        "green": [0, 0, 0],
        "blue": [0.5, 0, 0],
    },
}

# Build scene with state-driven positions and selection outlines
scene = (
    (
        Plot.State(initial_state)
        | Scene(
            # Red group: uses child_overrides (all children are red)
            Group(
                name="red",
                children=[
                    PointCloud(centers=point_cluster([0, 0, 0]), size=0.04),
                    Ellipsoid(center=[0, 0.3, 0], half_size=0.08),
                ],
                position=js("$state.positions.red"),
                child_overrides={
                    "color": [1, 0.3, 0.3],
                    "outline": js("$state.selected.includes('red')"),
                    "outline_color": [1, 1, 1],
                    "outline_width": 2,
                },
                hover_props={"color": [1, 1, 0.3], "scale": 1.15},
                on_click=js("""(info) => {
                const sel = new Set($state.selected);
                sel.has('red') ? sel.delete('red') : sel.add('red');
                $state.selected = [...sel];
            }"""),
                on_drag=js("(info) => info.applyDelta($state, 'positions.red')"),
                drag_constraint=DRAG_SCREEN,
            ),
            # Green group: uses child_overrides
            Group(
                name="green",
                children=[
                    PointCloud(centers=point_cluster([0, 0, 0]), size=0.04),
                    Ellipsoid(center=[0, 0.3, 0], half_size=0.08),
                ],
                position=js("$state.positions.green"),
                child_overrides={
                    "color": [0.3, 1, 0.3],
                    "outline": js("$state.selected.includes('green')"),
                    "outline_color": [1, 1, 1],
                    "outline_width": 2,
                },
                hover_props={"color": [1, 1, 0.3], "scale": 1.15},
                on_click=js("""(info) => {
                const sel = new Set($state.selected);
                sel.has('green') ? sel.delete('green') : sel.add('green');
                $state.selected = [...sel];
            }"""),
                on_drag=js("(info) => info.applyDelta($state, 'positions.green')"),
                drag_constraint=DRAG_SCREEN,
            ),
            # Blue group: uses child_overrides
            Group(
                name="blue",
                children=[
                    PointCloud(centers=point_cluster([0, 0, 0]), size=0.04),
                    Ellipsoid(center=[0, 0.3, 0], half_size=0.08),
                ],
                position=js("$state.positions.blue"),
                child_overrides={
                    "color": [0.3, 0.5, 1],
                    "outline": js("$state.selected.includes('blue')"),
                    "outline_color": [1, 1, 1],
                    "outline_width": 2,
                },
                hover_props={"color": [1, 1, 0.3], "scale": 1.15},
                on_click=js("""(info) => {
                const sel = new Set($state.selected);
                sel.has('blue') ? sel.delete('blue') : sel.add('blue');
                $state.selected = [...sel];
            }"""),
                on_drag=js("(info) => info.applyDelta($state, 'positions.blue')"),
                drag_constraint=DRAG_SCREEN,
            ),
        )
    )
    | "Interactive scene. Click to toggle selection, drag to move across the screen plane."
)
scene


# =============================================================================
# Demo: child_defaults vs child_overrides
# =============================================================================

# child_defaults: children CAN override the group's default color
# child_overrides: children CANNOT override - group wins

orange = [1, 0.5, 0.2]
yellow = [1, 1, 0]
green = [0.2, 1, 0.2]
(
    Group(
        children=[
            # Left side: uses group default (orange)
            PointCloud(centers=point_cluster([-1, 0, 0]), size=0.05),
            Ellipsoid(center=[-0.3, 0.3, 0], half_size=0.08),
            # Right side: child explicitly sets green - child wins with defaults!
            PointCloud(centers=point_cluster([0.3, 0, 0]), size=0.05, color=green),
            Ellipsoid(center=[1, 0.3, 0], half_size=0.08, color=green),
        ],
        child_defaults={
            "color": orange,  # Orange default - children can override
            "hover_props": {"scale": 1.2, "color": yellow},
        },
    )
    | "Group default is orange. Left children specify no color, right children specify green."
)


# =============================================================================
# Demo: Group transform (rotation + scale)
# =============================================================================

# Rotating and scaling an entire group
Group(
    children=[
        Cuboid(center=[-0.2, 0, 0], half_size=0.1, color=[1, 0.3, 0.3]),
        Cuboid(center=[0, 0, 0], half_size=0.1, color=[0.3, 1, 0.3]),
        Cuboid(center=[0.2, 0, 0], half_size=0.1, color=[0.3, 0.3, 1]),
    ],
    position=[0, 0, 0],
    quaternion=[0, 0, 0.383, 0.924],  # 45 degree rotation around Z
    scale=1.5,
    hover_props={
        "outline": True,
        "outline_color": [1, 1, 0],
        "outline_width": 8,
    },
)


# =============================================================================
# Demo: All Primitive Types with GPU Transforms
# =============================================================================

# This demonstrates that ALL primitive types properly support group transforms
# (position, rotation, scale) computed on the GPU. Drag any group to move it!

from colight.scene3d import (
    CameraFrustum,
    ImagePlane,
    ImageProjection,
    LineBeams,
    LineSegments,
    Mesh,
)


# Create a simple checkerboard image for ImagePlane
def make_checkerboard(size=64, squares=8):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    sq_size = size // squares
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                img[
                    i * sq_size : (i + 1) * sq_size, j * sq_size : (j + 1) * sq_size
                ] = [
                    200,
                    200,
                    200,
                ]
            else:
                img[
                    i * sq_size : (i + 1) * sq_size, j * sq_size : (j + 1) * sq_size
                ] = [
                    50,
                    50,
                    50,
                ]
    return img


checkerboard = make_checkerboard()

# Camera intrinsics/extrinsics for frustum + image projection helpers
camera_intrinsics = {
    "fx": 480.0,
    "fy": 480.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640.0,
    "height": 480.0,
}
camera_extrinsics = {
    "position": [0.0, 0.0, 0.0],
    "quaternion": [0.0, 0.0, 0.0, 1.0],
}
projection_extrinsics = {
    "position": [0.0, 0.0, 0.0],
    "quaternion": [0.0, 0.0, 0.0, 1.0],
}

# Line data: a small arc
arc_points = []
for i in range(12):
    theta = i * np.pi / 11
    arc_points.extend([0.15 * np.cos(theta), 0.15 * np.sin(theta), 0, i])
arc_points = np.array(arc_points, dtype=np.float32)

# Line segments: a small X shape
seg_starts = [[-0.1, -0.1, 0], [0.1, -0.1, 0]]
seg_ends = [[0.1, 0.1, 0], [-0.1, 0.1, 0]]

# Simple triangle mesh
triangle_pos = [[0, 0.15, 0], [-0.12, -0.08, 0], [0.12, -0.08, 0]]
triangle_idx = [0, 1, 2]

# State for draggable positions
all_types_state = {
    "positions": {
        "points": [-1.2, 0.6, 0],
        "ellipsoid": [-0.4, 0.6, 0],
        "wireframe": [0.4, 0.6, 0],
        "cuboid": [1.2, 0.6, 0],
        "linebeams": [-1.2, -0.4, 0],
        "lineseg": [-0.4, -0.4, 0],
        "imageplane": [0.4, -0.4, 0],
        "mesh": [1.2, -0.4, 0],
        "camera_frustum": [-0.8, -1.15, -0.9],
        "projection": [0.6, -1.15, -0.9],
    }
}

(
    Plot.State(all_types_state)
    | Scene(
        # 1. PointCloud
        Group(
            name="points",
            children=[PointCloud(centers=point_cluster([0, 0, 0]), size=0.03)],
            position=js("$state.positions.points"),
            child_overrides={"color": [1, 0.5, 0.2]},
            hover_props={"scale": 1.2, "outline": True, "outline_color": [1, 1, 1]},
            on_drag=js("(info) => info.applyDelta($state, 'positions.points')"),
            drag_constraint=DRAG_SCREEN,
        ),
        # 2. Ellipsoid (Solid)
        Group(
            name="ellipsoid",
            children=[Ellipsoid(center=[0, 0, 0], half_size=0.12)],
            position=js("$state.positions.ellipsoid"),
            child_overrides={"color": [0.3, 0.8, 0.3]},
            hover_props={"scale": 1.2, "outline": True, "outline_color": [1, 1, 1]},
            on_drag=js("(info) => info.applyDelta($state, 'positions.ellipsoid')"),
            drag_constraint=DRAG_SCREEN,
        ),
        # 3. Ellipsoid (MajorWireframe) - uses EllipsoidAxes internally
        Group(
            name="wireframe",
            children=[
                Ellipsoid(
                    center=[0, 0, 0],
                    half_size=[0.15, 0.1, 0.08],
                    fill_mode="MajorWireframe",
                )
            ],
            position=js("$state.positions.wireframe"),
            child_overrides={"color": [0.8, 0.3, 0.8]},
            hover_props={"scale": 1.2, "outline": True, "outline_color": [1, 1, 1]},
            on_drag=js("(info) => info.applyDelta($state, 'positions.wireframe')"),
            drag_constraint=DRAG_SCREEN,
        ),
        # 4. Cuboid
        Group(
            name="cuboid",
            children=[Cuboid(center=[0, 0, 0], half_size=0.1)],
            position=js("$state.positions.cuboid"),
            child_overrides={"color": [0.3, 0.5, 1]},
            hover_props={"scale": 1.2, "outline": True, "outline_color": [1, 1, 1]},
            on_drag=js("(info) => info.applyDelta($state, 'positions.cuboid')"),
            drag_constraint=DRAG_SCREEN,
        ),
        # 5. LineBeams
        Group(
            name="linebeams",
            children=[LineBeams(points=arc_points, size=0.015)],
            position=js("$state.positions.linebeams"),
            child_overrides={"color": [1, 0.8, 0.2]},
            hover_props={"scale": 1.3},
            on_drag=js("(info) => info.applyDelta($state, 'positions.linebeams')"),
            drag_constraint=DRAG_SCREEN,
        ),
        # 6. LineSegments
        Group(
            name="lineseg",
            children=[LineSegments(starts=seg_starts, ends=seg_ends, size=0.015)],
            position=js("$state.positions.lineseg"),
            child_overrides={"color": [0.2, 1, 0.8]},
            hover_props={"scale": 1.3},
            on_drag=js("(info) => info.applyDelta($state, 'positions.lineseg')"),
            drag_constraint=DRAG_SCREEN,
        ),
        # 7. ImagePlane - uses custom shaders
        Group(
            name="imageplane",
            children=[ImagePlane(image=checkerboard, size=[0.25, 0.25])],
            position=js("$state.positions.imageplane"),
            hover_props={"scale": 1.2, "outline": True, "outline_color": [1, 1, 1]},
            on_drag=js("(info) => info.applyDelta($state, 'positions.imageplane')"),
            drag_constraint=DRAG_SCREEN,
        ),
        # 8. Mesh - uses custom shader generation
        Group(
            name="mesh",
            children=[
                Mesh(
                    positions=triangle_pos,
                    indices=triangle_idx,
                    center=[0, 0, 0],
                    color=[1, 0.3, 0.5],
                )
            ],
            position=js("$state.positions.mesh"),
            hover_props={"scale": 1.2, "outline": True, "outline_color": [1, 1, 1]},
            on_drag=js("(info) => info.applyDelta($state, 'positions.mesh')"),
            drag_constraint=DRAG_SCREEN,
        ),
        # 9. Camera frustum helper (draggable)
        Group(
            name="camera_frustum",
            children=[
                CameraFrustum(
                    intrinsics=camera_intrinsics,
                    extrinsics=camera_extrinsics,
                    near=0.2,
                    far=0.8,
                    color=[0.9, 0.9, 0.2],
                    line_width=0.01,
                )
            ],
            position=js("$state.positions.camera_frustum"),
            hover_props={"scale": 1.1, "outline": True, "outline_color": [1, 1, 1]},
            on_drag=js("(info) => info.applyDelta($state, 'positions.camera_frustum')"),
            drag_constraint=DRAG_SCREEN,
        ),
        # 10. Image projection (draggable, no frustum)
        Group(
            name="projection",
            children=[
                ImageProjection(
                    checkerboard,
                    intrinsics=camera_intrinsics,
                    extrinsics=projection_extrinsics,
                    depth=0.6,
                    opacity=0.9,
                    show_frustum=True,
                )
            ],
            position=js("$state.positions.projection"),
            hover_props={"scale": 1.1, "outline": True, "outline_color": [1, 1, 1]},
            on_drag=js("(info) => info.applyDelta($state, 'positions.projection')"),
            drag_constraint=DRAG_SCREEN,
        ),
    )
    | "All 8 primitive types in draggable groups plus a camera frustum and image projection."
)
