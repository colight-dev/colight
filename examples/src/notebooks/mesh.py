import numpy as np

from colight.scene3d import Mesh, LineSegments, GridHelper

# =============================================================================
# Basic Triangle - auto-computed normals
# =============================================================================

# Simple triangle with positions only - normals computed automatically for lit shading
triangle = Mesh(
    positions=[
        [0.0, 0.5, 0.0],  # top
        [-0.5, -0.5, 0.0],  # bottom-left
        [0.5, -0.5, 0.0],  # bottom-right
    ],
    indices=[0, 1, 2],
    centers=[[0.0, 0.0, 0.0]],
    color=[1.0, 0.3, 0.3],
    shading="lit",
)
triangle

# =============================================================================
# Quad with explicit normals
# =============================================================================

quad = Mesh(
    positions=[
        [-0.5, 0.5, 0.0],  # top-left
        [-0.5, -0.5, 0.0],  # bottom-left
        [0.5, 0.5, 0.0],  # top-right
        [0.5, -0.5, 0.0],  # bottom-right
    ],
    normals=[
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ],
    indices=[0, 1, 2, 2, 1, 3],
    centers=[[1.2, 0.0, 0.0]],
    color=[0.3, 0.6, 1.0],
    shading="lit",
)
quad

# =============================================================================
# Per-vertex colors (unlit) - like a 3D scan
# =============================================================================

# Triangle with RGB vertex colors
vertex_colored_triangle = Mesh(
    positions=[
        [0.0, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, -0.5, 0.0],
    ],
    vertex_colors=[
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
        [0.0, 0.0, 1.0],  # blue
    ],
    indices=[0, 1, 2],
    centers=[[-1.2, 0.0, 0.0]],
    shading="unlit",
    cull_mode="none",
)
vertex_colored_triangle

# =============================================================================
# Multiple instances with different colors
# =============================================================================

# Same geometry, multiple instances with per-instance colors
instanced_triangles = Mesh(
    positions=[
        [0.0, 0.4, 0.0],
        [-0.3, -0.2, 0.0],
        [0.3, -0.2, 0.0],
    ],
    indices=[0, 1, 2],
    centers=[
        [0.0, 1.0, 0.0],
        [-0.6, 1.0, 0.2],
        [0.6, 1.0, -0.2],
    ],
    colors=[
        [1.0, 0.8, 0.2],
        [0.8, 0.2, 1.0],
        [0.2, 1.0, 0.8],
    ],
    scales=[
        [0.8, 0.8, 0.8],
        [0.6, 0.6, 0.6],
        [0.5, 0.5, 0.5],
    ],
    shading="lit",
    hover_props={"outline": True},
)
instanced_triangles

# =============================================================================
# Height-colored terrain (vertex colors + lit shading)
# =============================================================================


def create_terrain_grid(size=10, scale=2.0, height_scale=0.3):
    """Create a simple terrain grid with height-based coloring."""
    positions = []
    colors = []

    for i in range(size):
        for j in range(size):
            x = (i / (size - 1) - 0.5) * scale
            z = (j / (size - 1) - 0.5) * scale
            # Simple height function
            y = np.sin(x * 3) * np.cos(z * 3) * height_scale

            positions.append([x, y, z])

            # Color by height: blue (low) -> green (mid) -> red (high)
            t = (y / height_scale + 1) / 2  # normalize to 0-1
            if t < 0.5:
                # blue to green
                colors.append([0.0, t * 2, 1.0 - t * 2])
            else:
                # green to red
                colors.append([(t - 0.5) * 2, 1.0 - (t - 0.5) * 2, 0.0])

    # Create triangle indices for grid
    indices = []
    for i in range(size - 1):
        for j in range(size - 1):
            v0 = i * size + j
            v1 = v0 + 1
            v2 = v0 + size
            v3 = v2 + 1
            indices.extend([v0, v2, v1, v1, v2, v3])

    return np.array(positions), np.array(colors), np.array(indices)


terrain_pos, terrain_colors, terrain_indices = create_terrain_grid()

terrain = Mesh(
    positions=terrain_pos,
    vertex_colors=terrain_colors,
    indices=terrain_indices,
    centers=[[0.0, -0.8, 0.0]],
    shading="lit",  # lit shading with auto-computed normals
    cull_mode="none",
)
terrain

# =============================================================================
# Textured quad
# =============================================================================


def create_checkerboard(size=64, squares=8):
    """Create a checkerboard texture as a numpy array."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    square_size = size // squares
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                y0, y1 = i * square_size, (i + 1) * square_size
                x0, x1 = j * square_size, (j + 1) * square_size
                img[y0:y1, x0:x1] = [255, 200, 100, 255]  # orange
            else:
                y0, y1 = i * square_size, (i + 1) * square_size
                x0, x1 = j * square_size, (j + 1) * square_size
                img[y0:y1, x0:x1] = [50, 80, 140, 255]  # blue
    return img


checkerboard = create_checkerboard()

# Textured quad with UVs
textured_quad = Mesh(
    positions=[
        [-0.5, 0.5, 0.0],  # top-left
        [-0.5, -0.5, 0.0],  # bottom-left
        [0.5, 0.5, 0.0],  # top-right
        [0.5, -0.5, 0.0],  # bottom-right
    ],
    uvs=[
        [0.0, 1.0],  # top-left
        [0.0, 0.0],  # bottom-left
        [1.0, 1.0],  # top-right
        [1.0, 0.0],  # bottom-right
    ],
    indices=[0, 1, 2, 2, 1, 3],
    texture=checkerboard,
    centers=[[2.4, 0.0, 0.0]],
    shading="unlit",
    cull_mode="none",
)
textured_quad

# =============================================================================
# Helpers and scene composition
# =============================================================================

# Line segment examples (axes)
line_starts = np.array([[-1.5, 0.0, 0.0], [0.0, -1.5, 0.0], [0.0, 0.0, -1.5]])
line_ends = np.array([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]])
line_colors = np.array([[1.0, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.4, 1.0]])
axes = LineSegments(starts=line_starts, ends=line_ends, colors=line_colors, size=0.015)

# Grid helper
grid = GridHelper(size=3.0, divisions=15, line_width=0.002)

# Scene properties
scene_props = {
    "defaultCamera": {
        "position": [2.5, 2.0, 3.5],
        "target": [0, 0, 0],
        "up": [0, 1, 0],
    },
}

# Compose the scene
scene = (
    triangle
    + quad
    + vertex_colored_triangle
    + instanced_triangles
    + terrain
    + textured_quad
    + axes
    + grid
    + scene_props
)
scene
