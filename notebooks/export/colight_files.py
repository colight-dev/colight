import colight.plot as Plot
from colight.plot import js
from colight.scene3d import PointCloud
from notebooks.scene3d.scene3d_ripple import create_ripple_grid
from notebooks.save_and_embed_file import create_embed_example
from pathlib import Path

# Create output directory if it doesn't exist
output_dir = Path("scratch")
output_dir.mkdir(exist_ok=True)

print("Creating a visual with 3D scene and KaTeX formula...")

# Create a simple 3D scene with a ripple grid
n_frames = 30
grid_xyz_frames, grid_rgb = create_ripple_grid(50, 50, n_frames=n_frames)

# Create the scene
scene = PointCloud(
    centers=js("$state.grid_xyz[$state.frame]"),
    colors=js("$state.grid_rgb"),
    size=0.04,
) + {
    "camera": js("$state.camera"),
    "onCameraChange": js("(camera) => $state.update({'camera': camera})"),
    "controls": ["fps"],
} | Plot.State(
    {
        "camera": {
            "position": [4.421623, -0.563180, 1.317901],
            "target": [-0.003753, -0.008899, 0.008920],
            "up": [0.000000, 0.000000, 1.000000],
            "fov": 35,
        }
    }
)

# Create a layout with the 3D scene and KaTeX formula
p = (
    Plot.State(
        {
            "frame": 0,
            "grid_xyz": grid_xyz_frames,
            "grid_rgb": grid_rgb,
        }
    )
    | Plot.Slider("frame", range=n_frames, fps=30)
    | (
        scene
        & Plot.md(r"""
The ripple pattern follows this equation:

$$z = A \sin(\omega(x + y) + \phi(t))$$

where:
- $A$ is the amplitude
- $\omega$ is the wave frequency
- $\phi(t)$ is the time-dependent phase
        """)
    )
)

colight_path = p.save_file("scratch/embed_example.colight")
create_embed_example(colight_path)
print(f"✓ Created .colight file at: {colight_path}")
