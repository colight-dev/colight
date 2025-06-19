# | hide-statements hide-code

# Basic NumPy Visualization

# This example demonstrates simple data visualization with numpy.

# | colight: show-code
import numpy as np


# Create sample data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
z = 0

# Generate visualization data
x, y

# Let's also look at some statistics:
f"Data range: {np.min(y):.3f} to {np.max(y):.3f}"

# Mean and standard deviation:
np.mean(y), np.std(y)
