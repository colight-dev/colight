"""
Quick test script for the .colight export functionality.
"""

import colight.plot as Plot
import numpy as np
from pathlib import Path
from notebooks.save_and_embed_file import create_embed_example

# Create output directory
output_dir = Path("scratch")
output_dir.mkdir(exist_ok=True)

# Create a simple visual
print("Creating a visual...")
data = np.random.rand(10, 10)
p = Plot.raster(data)

# Export with local embed
print("Exporting to .colight with local development viewer...")
colight_path = p.save_file("scratch/test_export.colight")
example_path = create_embed_example(colight_path, False)

print("Success! Files created:")
print(f"- .colight file: {colight_path}")
print(f"- Example HTML: {example_path}")
print(f"\nOpen {example_path} in your browser to view the visual.")
print("It should work directly with the file:// protocol, no server needed.")
