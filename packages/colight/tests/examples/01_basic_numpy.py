# | hide-statements

# Basic NumPy Visualization

# This example demonstrates simple data visualization with numpy.

# | colight: show-code
import colight.plot as Plot
import numpy as np

Plot.katex(r"""
y(\theta) = \mathbb{E}_{x\sim P(\theta)}[x] = \int_{\mathbb{R}}\left[\theta^2\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x}{\sigma}\right)^2} + (1-\theta)\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x-0.5\theta}{\sigma}\right)^2}\right]dx =\frac{\theta-\theta^2}{2}
        """)

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
