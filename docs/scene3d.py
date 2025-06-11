# %% [markdown]
# # Scene3D
#
# Scene3D builds on the same data and composition paradigms as Colight Plot but adds support for WebGPU–powered 3D primitives.
#
# %%
from colight.scene3d import (
    Cuboid,
    Ellipsoid,
    LineBeams,
    PointCloud,
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
# ## Other Primitives
#
# Scene3D supports several primitive types: PointCloud, Ellipsoid, Cuboid, and LineBeams.

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
import colight.plot as Plot
from colight.scene3d import PointCloud
import numpy as np

# Create a point cloud with 100 points
centers = np.random.normal(0, 1, (100, 3))
cloud = PointCloud(
    centers=centers,
    color=[0.5, 0.5, 0.5],  # Default gray color
    size=0.05,  # Default size
    decorations=[
        # Make points 0-9 red, 3x size size
        deco(np.arange(10), color=[1.0, 0.0, 0.0], scale=3.0),
        # Make points 10-19 semi-transparent blue, 10x size
        deco(np.arange(10, 20), color=[0.0, 0.0, 1.0], alpha=0.5, scale=10.0),
    ],
)

cloud

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
    Plot.initialState({"hovered": 1})  # Middle cube initially selected
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
