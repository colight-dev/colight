# Multiple Visualizations
# This example shows several different types of data.

import numpy as np

# Linear data
x_linear = np.linspace(0, 10, 50)
y_linear = 2 * x_linear + 1

# First visualization: linear relationship
x_linear, y_linear

# Trigonometric data
x_trig = np.linspace(0, 4 * np.pi, 80)
y_sin = np.sin(x_trig)
y_cos = np.cos(x_trig)

# Second visualization: sin and cos
x_trig, y_sin, y_cos

# Random data
np.random.seed(42)
random_data = np.random.normal(0, 1, 200)

# Third visualization: random distribution
random_data

# Some analysis
print(f"Random data mean: {np.mean(random_data):.3f}")
print(f"Random data std: {np.std(random_data):.3f}")

# Combined analysis
summary_stats = {
    "linear_slope": 2.0,
    "trig_amplitude": 1.0,
    "random_mean": np.mean(random_data),
    "random_std": np.std(random_data),
}

# Fourth visualization: summary
summary_stats
