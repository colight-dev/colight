import colight.plot as Plot
from colight.plot import js
from colight.scene3d import PointCloud
from notebooks.scene3d.scene3d_ripple import create_ripple_grid
from colight.html import export_colight
from notebooks.embed_examples import create_embed_example
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
} | Plot.initialState(
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
    Plot.initialState(
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

p

# Method 1: Export with automatic example creation (recommended)
print("\n1. Exporting visual with automatic example creation...")
colight_path, example_path = export_colight(
    p,
    "scratch/embed_example.colight",
    create_example=True,
    use_local_embed=True,  # Use local embed.mjs for testing
)

print(f"✓ Created .colight file at: {colight_path}")
print(f"✓ Created example HTML at: {example_path}")

# Method 2: Export just the .colight file and create example separately
print("\n2. Alternative method with separate example creation...")
# Export as .colight file only
colight_path2 = export_colight(
    p, "scratch/embed_example2.colight", create_example=False
)

# Create an example HTML file showing how to embed it
example_path2 = create_embed_example(
    "scratch/embed_example2.colight",  # Path to the .colight file
    use_local_embed=True,  # Use local embed.mjs for testing
)

print(f"✓ Created .colight file at: {colight_path2}")
print(f"✓ Created example HTML at: {example_path2}")

# Instructions for the user
print("\n" + "=" * 70)
print("VIEWING THE EXAMPLES")
print("=" * 70)
print("The example HTML files should work directly in your browser when opened:")
print(f"1. {example_path}")
print(f"2. {example_path2}")
