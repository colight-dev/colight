# # A two-segment tentacle that bends
#
# A tapered tube driven by a single bend angle, built from the primitives
# Scene3D has today: a `Mesh` with inline vertex arrays, nested `Group`s as
# the control structure, and a `$state` scalar wired to `Plot.Slider`.
#
# This document is a measurement fixture, not a rig. The same shape is a
# drill-hole trace deforming under a structural model, a linkage, or a strip
# of cloth — nothing here is named for anatomy. Its purpose is to show,
# concretely, how "a coarse control structure drives a dense surface" is
# expressed: three versions of one bend, ending in the one that gets the
# control structure and the surface at the same time.

import math

import numpy as np

import colight.plot as Plot
from colight import scene3d

# ## The tube
#
# A tapered tube of revolution: `N_RINGS` rings of `N_SIDES` vertices each,
# swept along +Z from radius `R0` down to `R1`. 24 rings x 8 sides = 192
# vertices, which is the "~200 vertices" scale a control structure of two
# transforms is meant to drive.

N_RINGS = 24
N_SIDES = 8
LENGTH = 4.0
R0, R1 = 0.55, 0.12

ring_t = np.linspace(0.0, 1.0, N_RINGS)  # arc-length parameter along the tube
ring_z = ring_t * LENGTH
ring_r = R0 + (R1 - R0) * ring_t

theta = np.linspace(0.0, 2.0 * math.pi, N_SIDES, endpoint=False)
cos_t, sin_t = np.cos(theta), np.sin(theta)


def tube_vertices(zs: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """Rings of `N_SIDES` vertices around the Z axis, one ring per (z, r) pair.

    Args:
        zs: (R,) ring centre positions along Z.
        rs: (R,) ring radii.

    Returns:
        (R * N_SIDES, 3) float32 vertex positions, ring-major.
    """
    x = rs[:, None] * cos_t[None, :]
    y = rs[:, None] * sin_t[None, :]
    z = np.broadcast_to(zs[:, None], x.shape)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)


def tube_indices(n_rings: int) -> np.ndarray:
    """Two triangles per quad between consecutive rings, wrapping around."""
    quads = []
    for i in range(n_rings - 1):
        for j in range(N_SIDES):
            a = i * N_SIDES + j
            b = i * N_SIDES + (j + 1) % N_SIDES
            c = (i + 1) * N_SIDES + j
            d = (i + 1) * N_SIDES + (j + 1) % N_SIDES
            quads.extend([a, c, b, b, c, d])
    return np.array(quads, dtype=np.uint32)


rest_positions = tube_vertices(ring_z, ring_r)
rest_positions.shape

# ## The control structure: two nested Groups
#
# `Group` composes a TRS transform onto every child, and nests. Two groups —
# a root at the tube's base and a child at the elbow — are the whole control
# structure. The child's `position` is the elbow in the root's frame; its
# `quaternion` is the bend.

ELBOW_RING = N_RINGS // 2
ELBOW_Z = float(ring_z[ELBOW_RING])


def bend_quaternion(angle_deg: float) -> list:
    """Rotation about +X by `angle_deg`, as the [x, y, z, w] quaternion Group takes."""
    half = math.radians(angle_deg) / 2.0
    return [math.sin(half), 0.0, 0.0, math.cos(half)]


# ## Splitting the vertices between the two Groups
#
# A `Group` transform applies to a whole child *component*. There is no
# per-vertex reference from geometry into the transform palette, so "these
# vertices belong to the lower group, those to the upper" can only be said by
# emitting **two Mesh components** — one per group — and cutting the tube's
# vertex array in half at the elbow.
#
# The cut is a real topological cut: the quad band that spanned the elbow ring
# belongs to neither mesh, so the two halves share no triangles. Each half is
# authored in the local frame of its own group (the upper half's Z is measured
# from the elbow, not from the base), because the group transform is what puts
# it back in place.

lower_z, lower_r = ring_z[: ELBOW_RING + 1], ring_r[: ELBOW_RING + 1]
upper_z, upper_r = ring_z[ELBOW_RING:] - ELBOW_Z, ring_r[ELBOW_RING:]

lower_positions = tube_vertices(lower_z, lower_r)
upper_positions = tube_vertices(upper_z, upper_r)

lower_indices = tube_indices(len(lower_z))
upper_indices = tube_indices(len(upper_z))

(lower_positions.shape, upper_positions.shape)


def tentacle(angle_deg: float) -> scene3d.SceneComponent:
    """The two-Group control structure holding the two mesh halves."""
    return scene3d.Group(
        position=[0.0, 0.0, 0.0],
        name="root",
        children=[
            scene3d.Mesh(
                positions=lower_positions,
                indices=lower_indices,
                color=[0.36, 0.62, 0.86],
                shading="lit",
                cull_mode="none",
            ),
            scene3d.Group(
                position=[0.0, 0.0, ELBOW_Z],
                quaternion=bend_quaternion(angle_deg),
                name="elbow",
                children=[
                    scene3d.Mesh(
                        positions=upper_positions,
                        indices=upper_indices,
                        color=[0.93, 0.53, 0.30],
                        shading="lit",
                        cull_mode="none",
                    )
                ],
            ),
        ],
    )


# Scene props, not bare camera params: a dict layer is merged into the scene's
# top-level props, so the camera has to travel under `defaultCamera` to be read.
CAMERA = {
    "defaultCamera": {
        "position": [7.0, -6.5, 4.5],
        "target": [0.0, 0.0, 2.0],
        "up": [0.0, 0.0, 1.0],
        "fov": 40,
        "near": 0.1,
        "far": 60.0,
    }
}

# ## Static bent pose: the hinge is visible
#
# At 45 degrees the two halves separate at the elbow — the surface tears open
# because the seam ring exists twice, once in each frame, and nothing blends
# between them. This is the deformation the control structure can actually
# express: rigid per-component, discontinuous at every group boundary.

scene3d.Scene(tentacle(45.0), CAMERA)

# ## Driving the angle from `$state`
#
# `Plot.channel` declares the elbow's rotation as a table of sampled
# quaternions indexed by a `$state` scalar. Nine samples across the slider's
# range are enough: `rule="qlerp"` interpolates each pair on the short arc and
# renormalizes, tracking a true slerp to within 0.01° at this spacing. The
# table travels once; the browser resamples it on every slider move with no
# Python round trip. Only the flattened transform palette is repacked; the
# vertex buffers are untouched.

BEND_SAMPLES = np.linspace(-80.0, 80.0, 9)
BEND_QUATS = np.array([bend_quaternion(a) for a in BEND_SAMPLES], dtype=np.float32)

hinged = scene3d.Group(
    name="root",
    children=[
        scene3d.Mesh(
            positions=lower_positions,
            indices=lower_indices,
            color=[0.36, 0.62, 0.86],
            shading="lit",
            cull_mode="none",
        ),
        scene3d.Group(
            position=[0.0, 0.0, ELBOW_Z],
            quaternion=Plot.channel(
                "bend", values=BEND_QUATS, at=BEND_SAMPLES, rule="qlerp"
            ),
            name="elbow",
            children=[
                scene3d.Mesh(
                    positions=upper_positions,
                    indices=upper_indices,
                    color=[0.93, 0.53, 0.30],
                    shading="lit",
                    cull_mode="none",
                )
            ],
        ),
    ],
)

(
    scene3d.Scene(hinged, CAMERA)
    | Plot.Slider(
        "bend",
        init=45.0,
        range=[-80.0, 80.0],
        step=1.0,
        label=Plot.js("`Bend: ${$state.bend.toFixed(0)}°`"),
    )
)

# Sweeping this slider is cheap: the tube's 192 vertices never move, the two
# meshes keep their identity, and the elbow's TRS is the only thing repacked.
# It is also the wrong picture — the tube tears at the elbow rather than
# curving through it.

# ## The continuous bend, computed in Python
#
# The shape the control structure *should* produce is a smooth arc: each ring
# rotates by a fraction of the bend angle that ramps across a blend zone
# straddling the elbow. Expressed as a single mesh whose vertices are
# recomputed per angle, this is a full-rank deformation — every vertex moves,
# and the whole positions array is rebuilt.

BLEND_HALF_WIDTH = 0.22  # fraction of tube length over which the bend is spread


def bent_positions(angle_deg: float) -> np.ndarray:
    """Every ring rotated about +X by a smoothly ramped fraction of `angle_deg`.

    The ramp is a smoothstep centred on the elbow, so curvature is spread over
    a zone rather than concentrated at one ring. Each ring's frame is carried
    forward along the arc so the tube stays connected.

    Args:
        angle_deg: Total bend angle in degrees.

    Returns:
        (N_RINGS * N_SIDES, 3) float32 vertex positions.
    """
    t_elbow = ring_t[ELBOW_RING]
    u = np.clip(
        (ring_t - (t_elbow - BLEND_HALF_WIDTH)) / (2.0 * BLEND_HALF_WIDTH), 0.0, 1.0
    )
    ramp = u * u * (3.0 - 2.0 * u)  # smoothstep
    phi = math.radians(angle_deg) * ramp  # cumulative bend at each ring

    # Integrate ring centres along the bent centreline in the YZ plane.
    ds = np.diff(ring_z, prepend=ring_z[0])
    dy = -np.sin(phi) * ds
    dz = np.cos(phi) * ds
    cy = np.cumsum(dy)
    cz = np.cumsum(dz)

    # Each ring's local frame: X unchanged (the bend axis), the ring plane
    # tilted by phi about X.
    ex = np.stack(
        [np.ones_like(phi), np.zeros_like(phi), np.zeros_like(phi)], axis=-1
    )  # (R, 3)
    ey = np.stack([np.zeros_like(phi), np.cos(phi), np.sin(phi)], axis=-1)

    centres = np.stack([np.zeros_like(cy), cy, cz], axis=-1)  # (R, 3)
    offs = ring_r[:, None, None] * (
        cos_t[None, :, None] * ex[:, None, :] + sin_t[None, :, None] * ey[:, None, :]
    )  # (R, N_SIDES, 3)
    return (centres[:, None, :] + offs).reshape(-1, 3).astype(np.float32)


indices_full = tube_indices(N_RINGS)

# The bent pose at 45 degrees, as one continuous mesh. This is what the
# hinged version above is trying and failing to be — but it is produced by
# recomputing the vertex array in Python, not by the control structure. The
# two `Group`s play no part in it.

scene3d.Scene(
    scene3d.Mesh(
        positions=bent_positions(45.0),
        indices=indices_full,
        color=[0.42, 0.72, 0.52],
        shading="lit",
        cull_mode="none",
    ),
    CAMERA,
)

# ## Sweeping the continuous bend
#
# The same declaration drives the dense case. A pose table sampled every 4°
# becomes the channel's `values`; the slider steps in 1° increments, finer
# than the sample spacing, so most positions the mesh receives are interpolated
# between two poses rather than read off one. Every vertex moves — this is the
# full-rank deformation, riding the geometry contents-write path: the mesh
# keeps its identity and pipelines, and the resampled array is written into the
# vertex buffer that already exists.

N_POSES = 41
ANGLES = np.linspace(-80.0, 80.0, N_POSES)
POSES = np.stack([bent_positions(a) for a in ANGLES]).reshape(N_POSES, -1)
POSES.shape

(
    scene3d.Scene(
        scene3d.Mesh(
            positions=Plot.channel("bend2", values=POSES, at=ANGLES, rule="linear"),
            indices=indices_full,
            color=[0.42, 0.72, 0.52],
            shading="lit",
            cull_mode="none",
        ),
        CAMERA,
    )
    | Plot.Slider(
        "bend2",
        init=20.0,
        range=[-80.0, 80.0],
        step=1.0,
        label=Plot.js("`Bend: ${$state.bend2.toFixed(0)}°`"),
    )
)

# A channel fixes what happens per frame, not what crosses the wire: the pose
# table is still O(poses × vertices) and still ships in full. What it buys is
# that the sweep is a declaration — legible to `colight inspect` as a parameter
# with a domain and a rule, and resolved entirely in the browser, so a saved
# `.colight` sweeps with no Python attached.

# ## The continuous bend, driven by the control structure
#
# The two versions above are the two halves of one thing: the hinged one has
# the right control structure and the wrong surface, the swept one has the
# right surface and no control structure. What unifies them is letting a
# vertex be positioned by a *weighted combination* of the group transforms
# rather than rigidly by one.
#
# `transform_refs` names the Groups a mesh's vertices reference;
# `transform_indices` says which of them each vertex uses, and
# `transform_weights` in what proportion. Here every vertex references both
# groups (`[0, 1]`) and the weight is the *same smoothstep ramp* the swept
# version integrates — so the two computations are answering the same
# question, one in Python and one on the GPU.
#
# One mesh, the full unsplit 192-vertex tube, authored in the root's frame.
# No cut at the elbow, no pose table.

full_positions = tube_vertices(ring_z, ring_r)

_t_elbow = ring_t[ELBOW_RING]
_u = np.clip(
    (ring_t - (_t_elbow - BLEND_HALF_WIDTH)) / (2.0 * BLEND_HALF_WIDTH), 0.0, 1.0
)
ring_ramp = _u * _u * (3.0 - 2.0 * _u)  # smoothstep, per ring

# One row per vertex: ring-major, so every vertex in a ring shares its ramp.
vertex_ramp = np.repeat(ring_ramp, N_SIDES)
REF_SLOTS = np.tile([0, 1], (len(full_positions), 1))
REF_WEIGHTS = np.stack([1.0 - vertex_ramp, vertex_ramp], axis=1)

REF_WEIGHTS.shape

# When a mesh declares `transform_refs`, the blend **replaces** its own
# component-level group transform: the referenced entries are already composed
# through the Group nesting, so the mesh may sit anywhere in the hierarchy and
# only the Groups it names move its vertices.
#
# That has one consequence worth stating plainly. Every referenced transform
# is applied to the *same* vertex coordinates, so all of them must be written
# in one frame — here the root's. The hinged version could author its upper
# half in elbow-local Z because only one transform touched it; a blended mesh
# cannot. So the elbow is expressed as a **rotation about the elbow point**
# rather than a translate-then-rotate: translate to the pivot, rotate,
# translate back. Nested `Group`s compose exactly that, and the composed
# result is what lands in the palette under the name `elbow`.

blended = scene3d.Group(
    name="root",
    children=[
        scene3d.Mesh(
            positions=full_positions,
            indices=indices_full,
            transform_refs=["root", "elbow"],
            transform_indices=REF_SLOTS,
            transform_weights=REF_WEIGHTS,
            color=[0.42, 0.72, 0.52],
            shading="lit",
            cull_mode="none",
        ),
        scene3d.Group(
            position=[0.0, 0.0, ELBOW_Z],
            children=[
                scene3d.Group(
                    quaternion=Plot.channel(
                        "bend3", values=BEND_QUATS, at=BEND_SAMPLES, rule="qlerp"
                    ),
                    children=[
                        scene3d.Group(name="elbow", position=[0.0, 0.0, -ELBOW_Z])
                    ],
                )
            ],
        ),
    ],
)

(
    scene3d.Scene(blended, CAMERA)
    | Plot.Slider(
        "bend3",
        init=45.0,
        range=[-80.0, 80.0],
        step=1.0,
        label=Plot.js("`Bend: ${$state.bend3.toFixed(0)}°`"),
    )
)

# This is the picture both earlier versions were reaching for: a continuous
# bend *through* the control structure. The elbow is still a `Group`
# quaternion driven by the same `Plot.channel` table the hinged version uses —
# the only thing that ships per frame is that quaternion, repacked into the
# transform palette. The 192 vertices never move and the weights never change,
# so sweeping the slider costs zero geometry uploads: the surface deforms
# entirely from the palette.
#
# The gap this document was built to expose is closed. The hinged and swept
# versions stay above as the two halves it unified — the first shows what
# rigid per-component transforms can and cannot say, the second shows what
# a full-rank pose table costs to say it. Neither is obsolete: a deformation
# that moves every vertex independently (an FEM step, a solver output) is
# still the swept version's territory. "Which rank is this deformation?" is
# the question that routes between them.
