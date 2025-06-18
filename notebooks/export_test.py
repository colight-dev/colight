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
# Create a 1D flat numpy array with interleaved x,y coordinates
data = np.random.rand(2000)  # 1000 points * 2 coordinates = 2000 values
p = Plot.dot(
    {"x": np.random.rand(1000), "y": np.random.rand(1000)},
)

# Export with local embed
print("Exporting to .colight with local development viewer...")
colight_path = p.save_file("scratch/test_export.colight")
example_path = create_embed_example(colight_path)

print("Success! Files created:")
print(f"- .colight file: {colight_path}")
print(f"- Example HTML: {example_path}")
print(f"\nOpen {example_path} in your browser to view the visual.")
print("It should work directly with the file:// protocol, no server needed.")
