import numpy as np

from colight.scene3d import Mesh, Cuboid, LineSegments, GridHelper, CameraFrustum
from colight.screenshots import save_image

# Define a simple triangle mesh (unlit, double-sided)
triangle_vertex_data = np.array(
    [
        # x, y, z, nx, ny, nz
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        -0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)
triangle_index_data = np.array([0, 1, 2], dtype=np.uint16)

# Define a quad mesh (lit, back-face culled)
quad_vertex_data = np.array(
    [
        # x, y, z, nx, ny, nz
        -0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        -0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0,
        0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)
quad_index_data = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint16)

# Inline mesh instances
triangles = Mesh(
    triangle_vertex_data,
    triangle_index_data,
    centers=[[0.0, 0.0, 0.0], [-0.8, 0.3, 0.0]],
    colors=[[1.0, 0.2, 0.2], [1.0, 0.6, 0.2]],
    scales=[[1.0, 1.0, 1.0], [0.6, 0.6, 0.6]],
    shading="unlit",
    cull_mode="none",
    hoverProps={'outline': True}
)
quads = Mesh(
    quad_vertex_data,
    quad_index_data,
    centers=[[0.9, -0.1, 0.0], [0.9, 0.5, 0.2]],
    colors=[[0.2, 0.6, 1.0], [0.2, 1.0, 0.6]],
    scales=[[0.8, 0.8, 0.8], [0.4, 0.4, 0.4]],
    shading="lit",
    cull_mode="back",
)

# Standard primitives for contrast
cuboid = Cuboid(centers=[[0.0, -0.6, 0.0]], half_size=0.18, color=[0, 1, 0],
    outline=True)
cuboid


# Line segment examples (axes)
line_starts = np.array(
    [
        [-1.2, 0.0, 0.0],
        [0.0, -1.2, 0.0],
        [0.0, 0.0, -1.2],
    ],
    dtype=np.float32,
)
line_ends = np.array(
    [
        [1.2, 0.0, 0.0],
        [0.0, 1.2, 0.0],
        [0.0, 0.0, 1.2],
    ],
    dtype=np.float32,
)
line_colors = np.array(
    [
        [1.0, 0.2, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 0.4, 1.0],
    ],
    dtype=np.float32,
)
line_segments = LineSegments(
    starts=line_starts,
    ends=line_ends,
    colors=line_colors,
    size=0.02
)

# Grid helper example
grid = GridHelper(size=2.4, divisions=12, line_width=0.002, )

# Camera frustum example
intrinsics = {
    "fx": 480.0,
    "fy": 480.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640.0,
    "height": 480.0,
}
extrinsics = {
    "position": [0.0, 0.3, -1.0],
    "quaternion": [0.0, 0.0, 0.0, 1.0],
}
frustum = CameraFrustum(line_width=0.002, intrinsics=intrinsics, extrinsics=extrinsics, near=0.2, far=1.0)

# Scene properties
scene_props = {
    "defaultCamera": {
        "position": [0, 0, 3],
        "target": [0, 0, 0],
        "up": [0, 1, 0],
    },
}

# Compose the scene using + operator
scene = triangles + quads + cuboid + line_segments + grid + frustum + scene_props
scene