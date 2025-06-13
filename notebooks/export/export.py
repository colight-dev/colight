# %% [markdown]
# ## Saving Plots as Images and Videos
#
# This notebook shows how to save plots as static images and videos.

# %%
import colight.plot as Plot
from colight.scene3d import Ellipsoid
from pathlib import Path
import numpy as np


# Create output directory
output_dir = Path("scratch/export_examples")
output_dir.mkdir(exist_ok=True, parents=True)
# Remove any existing files in output directory
for file in output_dir.glob("*"):
    file.unlink()


# %% [markdown]
# ### Save as PDF
#

# %%

# Create interesting pattern of points using numpy

t = np.linspace(0, 4 * np.pi, 40)
points = np.column_stack(
    [
        t * np.cos(t),  # Spiral x coordinates
        t * np.sin(t),  # Spiral y coordinates
    ]
)
dots = Plot.dot(points, r=10, fill="steelblue")

dots.save_html(str(output_dir / "dots.html"), use_cdn=False)

# %%

ellipsoid_plot = (
    Plot.initialState({})
    | Ellipsoid(
        [0, 0, 0, 1, 1, 1],  # Center position
        half_size=0.5,
        color=[1, 0, 0],  # Red color
    )
    & Ellipsoid(
        [0, 0, 0, 1, 1, 1],  # Center position
        half_size=0.5,
        color=[1, 1, 0],
    )
    | dots
    | Plot.html(["div.text-lg", "Hello"])
)


# Save PDFs at different sizes
ellipsoid_plot.save_pdf(str(output_dir / "ellipsoid_1x500.pdf"), width=500, scale=1)
ellipsoid_plot.save_pdf(str(output_dir / "ellipsoid_9x1000.pdf"), width=1000, scale=9)

# Save PNGs at different sizes
ellipsoid_plot.save_image(str(output_dir / "ellipsoid_1x500.png"), width=500, scale=1)
ellipsoid_plot.save_image(str(output_dir / "ellipsoid_2x1000.png"), width=1000, scale=2)


# %% [markdown]
# ### PNG Images
#
# Save a plot as a static image:

# %%
# Create and display a simple scatter plot
dots = Plot.dot([[1, 1], [2, 2], [3, 3]], r=10, fill="steelblue")
dots.save_image(str(output_dir / "scatter.png"), width=400)

# %% [markdown]
# ### Image Sequences
#
# Save a plot in different states as a sequence of images. Here's a plot that
# arranges points in a circle, with the number of points controlled by state:

# %%
# Create a plot with state
circle_plot = Plot.initialState({"count": 5}) | [
    "div",
    {"style": {"padding": "20px", "backgroundColor": "#f0f0f0"}},
    ["h3", Plot.js("`Points: ${$state.count}`")],
    Plot.dot(
        {"length": Plot.js("$state.count")},
        x=Plot.js("(d,i) => Math.cos(i * Math.PI * 2 / $state.count)"),
        y=Plot.js("(d,i) => Math.sin(i * Math.PI * 2 / $state.count)"),
        r=8,
        fill="steelblue",
    ),
]

# Display the initial state
circle_plot

# %% [markdown]
# Save images with different numbers of points:

# %%
# Save multiple states as separate images
paths = circle_plot.save_images(
    state_updates=[{"count": i} for i in [3, 6, 12, 24]],
    output_dir=output_dir,  # Convert Path to str
    filename_base="circle",
    width=500,
)

print("Created images:")
for path in paths:
    print(f"  {path}")

# %% [markdown]
# ### Videos
#
# Create a video by animating state transitions. This example shows animated
# points in 3D space:

# %%
# Create a 3D scene with animated points
animated_scene = Plot.initialState({"t": 0}) | Ellipsoid(
    Plot.js("""
            Array.from({length: 50}, (_, i) => {
                const angle = i * Math.PI * 2 / 50;
                const x = Math.cos(angle + $state.t);
                const y = Math.sin(angle + $state.t);
                const z = Math.sin($state.t * 2 + i * 0.1);
                return [x, y, z];
            }).flat()
        """),
    half_size=0.1,
    color=Plot.js("""
            (d, i) => {
                const j = Math.floor(i / 3);
                return [
                    0.5 + 0.5 * Math.sin($state.t + j * 0.1),
                    0.5 + 0.5 * Math.cos($state.t + j * 0.2),
                    0.5 + 0.5 * Math.sin($state.t + j * 0.3)
                ];
            }
        """),
)

# Display the initial state
animated_scene

(animated_scene | Plot.Slider("t", range=60, fps=10)).save_html(
    str(output_dir / "points.html"), use_cdn=False
)

# %%
# Save as video if ffmpeg is available
import shutil

if shutil.which("ffmpeg"):
    video_path = animated_scene.save_video(
        state_updates=[{"t": i * 0.1} for i in range(60)],
        filename=str(output_dir / "points.mp4"),  # Convert Path to str
        fps=30,
        width=800,
        height=600,
    )
    video_path = animated_scene.save_video(
        state_updates=[{"t": i * 0.1} for i in range(60)],
        filename=str(output_dir / "points.gif"),  # Convert Path to str
        fps=30,
        width=800,
        height=600,
    )
    print(f"Video saved to: {video_path}")
else:
    print("Note: Video creation requires ffmpeg to be installed")
