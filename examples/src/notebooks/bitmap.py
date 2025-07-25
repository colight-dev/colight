import colight.plot as Plot
import numpy as np

import os
import shutil

# Remove the scratch/bitmap directory if it exists
if os.path.exists("scratch/bitmap"):
    shutil.rmtree("scratch/bitmap")

Plot.bitmap(np.random.rand(8, 8)).save_pdf("scratch/bitmap/single.pdf", debug=True)


# Create bitmaps of different sizes
sizes = [4, 8, 16, 32]
bitmaps = [Plot.bitmap(np.random.rand(size, size)) for size in sizes]
Plot.html(["div", *bitmaps])

# Arrange them in a flex container with padding and gap
p = Plot.html(
    [
        "div.p-3.grid.grid-cols-2.gap-2",  # 2x2 grid container with padding and gap
        [
            ["div.text-sm.text-gray-600", f"{size}x{size}", bitmap]
            for size, bitmap in zip(sizes, bitmaps)
        ],
    ]
)

[
    p.save_pdf(f"scratch/bitmap/grid_{width}.pdf", width=width, debug=True)
    for width in [200]  # [200, 400, 500, 600, 700, 800, 1200, 1600]
]
