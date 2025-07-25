# ## Saving Plots as Images and Videos
#
# This notebook shows how to save plots as static images and videos.


import colight.plot as Plot
from colight.scene3d import Ellipsoid
from pathlib import Path
import numpy as np

output_dir = Path("scratch/export_examples")
output_dir.mkdir(exist_ok=True, parents=True)


def multi_scene(num_plots, num_spheres=50):
    return Plot.html(
        [
            "div.grid.grid-cols-6.gap-4",
            *[
                Ellipsoid(
                    np.random.normal(
                        0, 0.5, (num_spheres, 3)
                    ).flatten(),  # 50 random positions in cluster
                    half_size=0.2,  # Smaller size for denser cluster
                    color=np.random.random(
                        (num_spheres, 3)
                    ).flatten(),  # 50 random RGB colors
                )
                for _ in range(num_plots)
            ],
        ]
    )


def multi_pdf(n, num_spheres=50, scale=1):
    multi_scene(n, num_spheres).save_pdf(
        str(output_dir / f"multi_{n}_{num_spheres}_{scale}.pdf"),
        width=500,
        scale=scale,
        debug=True,
    )


multi_pdf(40, 10000, 4)
