import os
from pathlib import Path
from typing import Union

import colight.env as env
import colight.plot as Plot
from colight.plot import js
from colight.scene3d import PointCloud
from notebooks.scene3d.scene3d_ripple import create_ripple_grid


def create_embed_example(
    colight_path: Union[str, Path],
    use_cdn: bool = True,
) -> str:
    """Create a minimal HTML example demonstrating .colight embedding."""
    colight_path = Path(colight_path)
    output_dir = colight_path.parent

    rel_path = colight_path.name

    if not use_cdn:
        local_embed_path = env.DIST_PATH / "embed.mjs"
        if local_embed_path.exists():
            script_url = f"./{os.path.relpath(local_embed_path, output_dir)}"
        else:
            raise FileNotFoundError("Local embed.js not found. Run `yarn dev`")
    elif isinstance(env.WIDGET_URL, str):
        script_url = str(env.WIDGET_URL).replace("widget.mjs", "embed.mjs")
    else:
        script_url = "https://cdn.jsdelivr.net/npm/@colight/core/embed.js"

    example_html = f"""<!DOCTYPE html>
<html>
<body>
    <div class="colight-embed" data-src="{rel_path}"></a>
    <script type="module">
        import {{loadVisuals}} from "{script_url}"
        loadVisuals()
    </script>
</body>
</html>"""

    example_path = output_dir / f"{colight_path.stem}.html"
    with open(example_path, "w") as f:
        f.write(example_html)
    return str(example_path)


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


path = create_embed_example(p.save_file("scratch/embed_example.colight"), False)

print(f"✓ Created .colight file at: {path}")
