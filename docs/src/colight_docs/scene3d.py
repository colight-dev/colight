# %% [markdown]
#
# Scene3D builds on the same data and composition paradigms as Colight Plot but adds support for WebGPU–powered 3D primitives.
#
# %%
import colight.plot as Plot
from colight.scene3d import (
    Cuboid,
    Ellipsoid,
    LineBeams,
    PointCloud,
    Scene,
    deco,
)
import numpy as np

# %% [markdown]
# ## A Basic Point Cloud
#
# Let’s start by creating a simple point cloud. Our point cloud takes an array of 3D coordinates and an array of colors.

# %%
# Define some 3D positions and corresponding colors.

centers = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)

colors = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [0.5, 0.5, 0.5],
    ],
    dtype=np.float32,
)

# Create the point cloud component.
point_cloud = PointCloud(
    centers=centers,
    colors=colors,
    size=0.1,  # Default size for all points
)

# %% [markdown]
# Next, we combine the point cloud with a camera configuration. The camera is specified in a properties dictionary using the key `"defaultCamera"`.

# %%
scene_pc = point_cloud + {
    "defaultCamera": {
        "position": [2.029898, 2.039866, 2.034882],
        "target": [-0.004984, 0.004984, 0.000000],
        "up": [0.000000, 0.000000, 1.000000],
        "fov": 45,
    }
}

scene_pc

# %% [markdown]
# ## Camera auto-fit, scene origin, and background
#
# **Auto-fit.** When you don't provide a camera, the scene fits its own
# world-space bounds on first render (deriving `near`/`far` from the scene
# extent). A far-from-unit-scale scene — a 3 km deposit, a UTM tile — frames
# correctly out of the box instead of rendering as empty background under the
# default unit-scale camera. An explicit `"defaultCamera"` (as above) still
# wins, and auto-fit is deterministic: the same scene yields the same camera,
# so `colight screenshot --check` stays byte-identical.
#
# **`Scene(origin=[x, y, z])`.** For scenes whose coordinates are far from the
# origin (e.g. UTM eastings ~445,000 m, which exceed float32 GPU precision),
# pass an `origin`. Every position-typed attribute (`centers`, `starts`,
# `ends`, `points`, and mesh geometry) is shifted by `-origin` at
# serialization, so the GPU sees small, float32-safe coordinates. World-space
# meshes are additionally re-centered — their geometry centroid is folded into
# the instance center — so large vertex arrays never reach the GPU buffers.
# The offset travels as scene metadata: `colight pick-at` / `pick-where` add it
# back, so reported positions stay in your original coordinate space.
#
# ```python
# scene3d.Scene(
#     topo.mesh(color=[0.82, 0.76, 0.65]),
#     drillholes.line_segments(color=[0.75, 0.75, 0.78]),
#     origin=[445500.0, 493500.0, 2942.0],  # UTM midpoint
# )
# ```
#
# **`Scene(background=[r, g, b])`.** Sets the WebGPU clear color (each channel
# 0–1) behind the geometry; it defaults to opaque black. This is the canvas
# clear color — DOM overlays drawn over the canvas (legends, FPS) keep their
# own styling.

# %% [markdown]
# ## Other Primitives
#
# Scene3D provides these primitives:
#
# - `PointCloud` — instanced points (squares oriented to face the camera)
# - `Ellipsoid` — spheres/ellipsoids, solid or `fill_mode="MajorWireframe"`
# - `Cuboid` — boxes with optional per-instance orientation quaternions
# - `LineBeams` — connected beam segments (points sharing an `i` value form a polyline)
# - `LineSegments` — independent segments given `starts` and `ends` arrays
# - `Mesh` — arbitrary triangle meshes with optional normals, per-vertex colors, UVs, and textures
# - `ImagePlane` — a textured quad displaying an image (numpy array or PIL image)
# - `ImageProjection` — an image plane positioned in space from camera intrinsics/extrinsics
# - `CameraFrustum` — wireframe camera frustum from intrinsics/extrinsics
# - `GridHelper` — a reference grid of line segments
# - `Group` — hierarchical transform (position/quaternion/scale) applied to child components
# - `CustomPrimitive` — instances of a custom mesh type defined elsewhere in the scene
#
# The examples below overlay several of them in one scene.

# %%

# Create a point cloud of 100 particles in a tight 3D Gaussian distribution
gaussian_centers = np.random.normal(loc=[1.0, 1.5, 0], scale=0.2, size=(100, 3)).astype(
    np.float32
)
# Generate random colors between purple [1,0,1] and cyan [0,1,1]
gaussian_colors = np.random.uniform(
    low=[0, 0, 1], high=[1, 1, 1], size=(100, 3)
).astype(np.float32)

gaussian_cloud = PointCloud(centers=gaussian_centers, colors=gaussian_colors, size=0.03)

# Create an ellipsoid component
(
    ellipsoid := Ellipsoid(
        centers=[
            [0, 0, 0],
            [1.5, 0, 0],
        ],
        half_sizes=[0.5, 0.5, 0.5],  # Can be a single value or a list per instance
        colors=np.array(
            [
                [0, 1, 1],  # cyan
                [1, 0, 1],  # magenta
            ],
            dtype=np.float32,
        ),
        alphas=np.array([1.0, 0.5]),  # Opaque and semi-transparent
    )
)

# Create a wireframe ellipsoid
wireframe = Ellipsoid(
    fill_mode="MajorWireframe",
    centers=[[0, 0, 0]],
    half_sizes=[0.7, 0.7, 0.7],
    color=[1, 1, 1],  # white
)

# Create a cuboid component
cuboid = Cuboid(
    centers=np.array([[0, 2, 0.5]], dtype=np.float32),
    half_sizes=[0.5, 0.5, 0.5],
    color=[1, 0.5, 0],  # orange
    alpha=0.8,
)

# Create line beams connecting points to form letter A
beams = LineBeams(
    points=np.array(
        [
            # Outer segments of A (i=0)
            1.5,
            1,
            -1,
            0,  # bottom left
            2.0,
            1,
            1,
            0,  # top
            2.5,
            1,
            -1,
            0,  # bottom right
            # Crossbar (i=1)
            1.75,
            1,
            0,
            1,  # left
            2.25,
            1,
            0,
            1,  # right
        ],
        dtype=np.float32,
    ),
    color=[0, 1, 0],  # green
    size=0.05,
)

# %% [markdown]
# ## Composition
#
# Use the `+` operator to overlay multiple scene components.

# %%

(
    gaussian_cloud
    + ellipsoid
    + wireframe
    + cuboid
    + beams
    + {
        "defaultCamera": {
            "position": [3.915157, 4.399701, 3.023268],
            "target": [0.401950, 0.815510, -0.408825],
            "up": [0.000000, 0.000000, 1.000000],
            "fov": 45,
        }
    }
)
# %% [markdown]

# ## Decorations
#
# Decorations allow you to modify the appearance of specific instances in a component. You can decorate instances by providing:
#
# - `color`: Override the color for decorated instances (RGB array, e.g. [1.0, 0.0, 0.0] for red)
# - `alpha`: Set transparency (0.0 = fully transparent, 1.0 = opaque)
# - `scale`: Scale the size of decorated instances relative to their base size
#
# The `deco()` function takes an array of indices to decorate and the desired appearance properties.

# %%
from colight.scene3d import PointCloud
import numpy as np

# Create a point cloud with 100 points
centers = np.random.normal(0, 1, (100, 3))
cloud = PointCloud(
    centers=centers,
    color=[0.5, 0.5, 0.5],  # Default gray color
    size=0.05,  # Default size
    decorations=[
        # Make points 0-9 red, 3x size
        deco(np.arange(10), color=[1.0, 0.0, 0.0], scale=3.0),
        # Make points 10-19 semi-transparent blue, 10x size
        deco(np.arange(10, 20), color=[0.0, 0.0, 1.0], alpha=0.5, scale=10.0),
    ],
)

cloud

# %% [markdown]

# ## Filtering instances (`filter_by`)
#
# `filter_by` hides instances whose per-instance scalar `values` fall outside a
# `[min, max]` threshold. It works on every instanced primitive (`PointCloud`,
# `Ellipsoid`, `Cuboid`, `LineSegments`, `LineBeams`). The essential property:
# `values` uploads **once** as instance data, while `min`/`max` live in a small
# per-component uniform — so `min`/`max` may be `Plot.js("$state.cutoff")`
# state references and a slider re-thresholds the scene client-side without
# re-uploading the (potentially large) instance data. `NaN` values are always
# hidden.
#
# Filtered-out instances are collapsed in the vertex shader and are also
# **unpickable**, so `pick-at` / `pick-where` / coverage report the visible
# instances honestly. `colight inspect` and `screenshot --json` report the
# active filters as `{component, label?, min, max}`.

# %%
from colight.scene3d import Cuboid

_grades = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
(
    Scene(
        Cuboid(
            centers=[[-3, 0, 0], [-1, 0, 0], [1, 0, 0], [3, 0, 0]],
            half_size=0.5,
            color=[0.2, 0.6, 0.9],
            # Only cells at/above the slider cutoff stay visible.
            filter_by={
                "values": _grades,
                "min": Plot.js("$state.cutoff"),
                "label": "grade",
            },
        )
    )
    | Plot.Slider("cutoff", init=0.0, range=[0.0, 1.0], step=0.05, label="cutoff")
    | Plot.initialState({"cutoff": 0.0})
)

# %% [markdown]
# ## Section / clipping planes (`Scene(clip_planes=...)`)
#
# `clip_planes` slices the **whole scene** with one or more half-space planes so
# interior structure becomes visible — the section view a geologist reads a
# block model or drillhole set through. Unlike `filter_by` (a per-instance mask),
# a clip plane cuts *through* geometry per-fragment, so it exposes the inside of
# solid shells. Each plane keeps the half-space `dot(p, normal) <= offset`;
# multiple planes intersect. Give a plane as `{"normal": n, "offset": d}` or the
# anchored `{"normal": n, "point": [x, y, z]}` form (the point is converted to an
# offset, respecting `Scene(origin=...)` — prefer it for geographic scenes).
#
# The plane offset lives in a scene-level uniform, so `offset` (or a `point`
# component) may be a `Plot.js("$state...")` reference and a slider sweeps the
# section client-side with no re-upload. Clipping applies in the **pick pass**
# too: `pick-at` on the exposed cut face reports the interior instance behind the
# section, not the outer shell. `inspect` / `screenshot --json` report the active
# `clip_planes` (`{normal, offset}` or `{normal, state_key}`), and both warn with
# `section-excludes-scene` if the planes clip away the entire scene. Up to 8
# planes; v1 does not cap/fill the cut surface (hollow shells show on the
# section).

# %%
_section_centers = np.array(
    [[x, y, z] for x in (-2, 0, 2) for y in (-2, 0, 2) for z in (-2, 0, 2)],
    dtype=np.float32,
)
(
    Scene(
        Cuboid(centers=_section_centers, half_size=0.4, color=[0.2, 0.6, 0.9]),
        # Keep everything below the sweeping plane (northing <= section_y).
        clip_planes=[{"normal": [0, 1, 0], "offset": Plot.js("$state.section_y")}],
    )
    | Plot.Slider("section_y", init=0.0, range=[-3.0, 3.0], step=0.5, label="section")
    | Plot.initialState({"section_y": 0.0})
    | {
        "defaultCamera": {
            "position": [8, 8, 8],
            "target": [0, 0, 0],
            "up": [0, 1, 0],
            "fov": 45,
        }
    }
)

# %% [markdown]

# ## Named selections (`Selection`)
#
# A **selection** is the same per-instance mask as a filter, but *named* and
# consumed differently: it highlights its instances (via the decoration system)
# and becomes a **shared referent** that both a human (clicking) and an agent
# (predicates, `pick-where --selection NAME`, `screenshot --frame NAME`) can
# name in conversation. Selections live in `$state.selections`, so they sync
# Python↔JS and persist into `.colight` artifacts.
#
# Build a selection with `scene3d.Selection(name, component, ...)` — either an
# explicit `instances=[...]` list or a threshold predicate
# (`values`/`values_ref` + `min`/`max`) — and seed it into state with
# `scene3d.select(...)`. `scene3d.toggle_selection(name, component)` is an
# `on_click` handler that adds/removes the picked instance, so human clicks and
# agent predicates converge on the same named object.

# %%
from colight import scene3d

(
    Scene(
        Cuboid(
            centers=[[-3, 0, 0], [-1, 0, 0], [1, 0, 0], [3, 0, 0]],
            half_size=0.5,
            color=[0.3, 0.3, 0.6],
            on_click=scene3d.toggle_selection("picked", 0),
        )
    )
    | scene3d.select(scene3d.Selection("picked", 0, instances=[1, 3]))
)

# %% [markdown]

# ## Picking
#
# A picking system allows for selecting elements in a scene using the `onHover` callback.
#
# In the example below, we decorate the hovered cube.

# %%
from colight.scene3d import Cuboid, deco

# Define centers for three non-overlapping cubes
cuboid_centers = np.array(
    [
        [-1.5, 0, 0],  # Left cube
        [0, 0, 0],  # Middle cube
        [1.5, 0, 0],  # Right cube
    ],
    dtype=np.float32,
)

# Create interactive cubes with hover effect
(
    Plot.State({"hovered": 1})  # Middle cube initially selected
    | Cuboid(
        centers=cuboid_centers,
        color=[1.0, 1.0, 0.0],  # yellow
        half_size=[0.4, 0.4, 0.4],
        alpha=0.5,
        onHover=Plot.js("(index) => $state.update({'hovered': index})"),
        decorations=[
            # Make hovered cube red and translucent
            deco(
                Plot.js("typeof $state.hovered === 'number' ? [$state.hovered] : []"),
                color=[1.0, 0.0, 0.0],
            )
        ],
    )
)

# %% [markdown]

# ## Meshes and Groups
#
# `Mesh` renders arbitrary triangle geometry (with optional normals, per-vertex
# colors, UVs, and textures), and `Group` applies a transform — `position`,
# `quaternion`, `scale` — to a list of child components, which may include
# nested groups. Groups can also bubble up events from their children
# (`on_hover`, `on_click`, `on_drag`) and apply `hover_props` or
# `child_defaults`/`child_overrides` to all children at once.

# %%
from colight.scene3d import Group, GridHelper, Mesh

triangle = Mesh(
    positions=[[0, 0, 0], [1, 0, 0], [0.5, 1, 0]],
    indices=[0, 1, 2],
    center=[0, 0, 0],
    color=[1, 0.5, 0],
)

(
    Group(
        [triangle, Cuboid(center=[1.2, 0.5, 0], half_size=0.2, color=[0, 0.7, 1])],
        position=[0, 0, 0.5],
        scale=0.8,
        name="assembly",
    )
    + GridHelper(size=4, divisions=8)
    + {
        "defaultCamera": {
            "position": [2.5, 2.5, 2.5],
            "target": [0.5, 0.5, 0.5],
            "up": [0, 0, 1],
        }
    }
)

# %% [markdown]

# ## Hover and Drag Interactions
#
# Every primitive accepts interaction props:
#
# - `hover_props`: appearance applied automatically while an instance is hovered
#   — `color`, `alpha`, `scale`, or `outline` (with `outline_color` /
#   `outline_width`) for an outline overlay, with no state management required.
# - `on_hover` / `on_click`: callbacks receiving the picked instance. Rich pick
#   info (instance index, world position, component and group names) is
#   available to click and drag handlers.
# - `on_drag`, `on_drag_start`, `on_drag_end` with a `drag_constraint`:
#   `drag_axis(direction, origin)` and `drag_plane(normal, origin)` build
#   constraints, and the constants `DRAG_AXIS_X/Y/Z`, `DRAG_PLANE_XY/XZ/YZ`,
#   `DRAG_SURFACE`, `DRAG_SCREEN`, and `DRAG_FREE` cover common cases.
# - `picking_scale`: enlarges the picking hit area of thin geometry.
# - `layer="overlay"`: renders a component in front of the scene, always
#   visible — useful for handles and manipulators.
#
# `TranslateGizmo(position, on_drag=...)` composes these into a ready-made
# translation gizmo with axis arrows, plane handles, and a center sphere
# (prototype; API may change). See the `drag.py`, `gizmo.py`, `mesh.py`, and
# `image_plane.py` notebooks in `examples/src/notebooks/` for full
# interactive examples.
