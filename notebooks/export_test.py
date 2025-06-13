"""
Quick test script for the .colight export functionality.
"""

import colight.plot as Plot
import numpy as np
from colight.html import export_colight
from pathlib import Path

# Create output directory
output_dir = Path("scratch")
output_dir.mkdir(exist_ok=True)

# Create a simple visual
print("Creating a visual...")
data = np.random.rand(10, 10)
p = Plot.raster(data)

# Export with local embed
print("Exporting to .colight with local development viewer...")
colight_path, example_path = export_colight(
    p, "scratch/test_export.colight", create_example=True, use_local_embed=True
)

print("Success! Files created:")
print(f"- .colight file: {colight_path}")
print(f"- Example HTML: {example_path}")
print(f"\nOpen {example_path} in your browser to view the visual.")
print("It should work directly with the file:// protocol, no server needed.")
