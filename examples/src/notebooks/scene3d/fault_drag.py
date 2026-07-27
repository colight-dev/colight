# # Fault drag on a gridded horizon
#
# A mapped surface crossed by a normal fault. The two blocks move rigidly;
# the surface between them does not tear, it *drags* — bending through a
# damage zone whose width is a property of the rock, not of the mesh.
#
# The same primitive as any other coarse-control-structure deformation: two
# `Group` transforms, and per-vertex weights blending between them.

import numpy as np

import colight.plot as Plot
from colight import scene3d

# ## The horizon
#
# A 40x40 grid over a 2 km square, with gentle regional dip and a little
# structural relief. The fault trace runs N-S at x = 0; distance from it is
# what the drag weights are a function of.

N = 40
EXTENT = 1000.0  # metres, half-width
DRAG_HALF_WIDTH = 220.0  # metres either side of the trace

gx = np.linspace(-EXTENT, EXTENT, N)
gy = np.linspace(-EXTENT, EXTENT, N)
X, Y = np.meshgrid(gx, gy, indexing="ij")
Z = -0.08 * X + 40.0 * np.sin(Y / 420.0)  # regional dip + relief

horizon = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)


def grid_indices(n: int) -> np.ndarray:
    """Two triangles per grid cell, row-major."""
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel()
    b, c, d = a + 1, a + n, a + n + 1
    return np.stack([a, c, b, b, c, d], axis=1).ravel().astype(np.uint32)


faces = grid_indices(N)

# ## The drag weights
#
# Each vertex blends between the footwall's transform and the hanging wall's.
# Far from the trace a vertex belongs wholly to its own block; across the
# damage zone the weight ramps smoothly, which is what makes the horizon bend
# into the fault instead of stepping across it.

u = np.clip((X.ravel() + DRAG_HALF_WIDTH) / (2.0 * DRAG_HALF_WIDTH), 0.0, 1.0)
drag = u * u * (3.0 - 2.0 * u)  # smoothstep: 0 in the footwall, 1 in the hanging wall

SLOTS = np.tile([0, 1], (len(horizon), 1))
WEIGHTS = np.stack([1.0 - drag, drag], axis=1)

# ## Slip
#
# The hanging wall drops. `Plot.channel` declares its offset as a table
# indexed by a `$state` scalar, so the slider resamples it in the browser: the
# horizon's 1600 vertices and their weights ship once and never move again.
# Only the hanging wall's transform is repacked per frame.

THROWS = np.linspace(0.0, 160.0, 9)
OFFSETS = np.stack([np.array([0.0, 0.0, -t], dtype=np.float32) for t in THROWS])

# A low, oblique viewpoint looking along strike: the throw and the flexure
# that accommodates it read in profile, which a map view flattens away.

CAMERA = {
    "defaultCamera": {
        "position": [1100.0, -3400.0, 1500.0],
        "target": [0.0, 0.0, -60.0],
        "up": [0.0, 0.0, 1.0],
        "fov": 32,
        "near": 1.0,
        "far": 12000.0,
    }
}

section = scene3d.Group(
    name="footwall",
    children=[
        scene3d.Mesh(
            positions=horizon,
            indices=faces,
            transform_refs=["footwall", "hangingwall"],
            transform_indices=SLOTS,
            transform_weights=WEIGHTS,
            color=[0.78, 0.62, 0.42],
            shading="lit",
            cull_mode="none",
        ),
        scene3d.Group(
            name="hangingwall",
            position=Plot.channel("throw", values=OFFSETS, at=THROWS, rule="linear"),
        ),
    ],
)

(
    scene3d.Scene(section, CAMERA)
    | Plot.Slider(
        "throw",
        init=90.0,
        range=[0.0, 160.0],
        step=2.0,
        label=Plot.js("`Throw: ${$state.throw.toFixed(0)} m`"),
    )
)

# Sweeping the throw costs no geometry upload: the vertex positions and the
# drag weights are both resident, and the only thing that changes per frame is
# one block's transform. The width of the drag zone is a separate declaration
# from the displacement, which is the point — `DRAG_HALF_WIDTH` is a property
# of the rock and the throw is a property of the structure, and neither is
# baked into the other.
