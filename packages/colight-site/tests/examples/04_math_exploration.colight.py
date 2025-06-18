# Mathematical Function Exploration
# Exploring different mathematical functions and their properties.

import numpy as np

# Domain for our functions
x = np.linspace(-2 * np.pi, 2 * np.pi, 200)


# Polynomial functions
def polynomial_family(x, degree):
    """Generate polynomial functions of different degrees."""
    results = {}
    for d in range(1, degree + 1):
        results[f"x^{d}"] = x**d
    return results


# Generate polynomial data
poly_data = polynomial_family(x, 4)

# Visualize polynomial progression
poly_data

# Trigonometric functions
trig_functions = {
    "sin(x)": np.sin(x),
    "cos(x)": np.cos(x),
    "tan(x)": np.tan(x),
    "sin(2x)": np.sin(2 * x),
    "cos(x/2)": np.cos(x / 2),
}

# Visualize trigonometric family
x, trig_functions

# Exponential and logarithmic (positive domain)
x_pos = np.linspace(0.1, 5, 100)
exp_log_functions = {
    "exp(x)": np.exp(x_pos),
    "log(x)": np.log(x_pos),
    "sqrt(x)": np.sqrt(x_pos),
    "x^2": x_pos**2,
}

# Visualize exponential/log family
x_pos, exp_log_functions

# Function composition
composite = np.sin(x) * np.exp(-(x**2) / 10)

# Visualize composite function
x, composite


# Fourier-like series approximation
def fourier_approx(x, n_terms):
    """Approximate a square wave using Fourier series."""
    result = np.zeros_like(x)
    for n in range(1, n_terms + 1, 2):  # Odd harmonics only
        result += (4 / (np.pi * n)) * np.sin(n * x)
    return result


# Compare different Fourier approximations
fourier_terms = [1, 3, 5, 10, 20]
fourier_data = {}
for n in fourier_terms:
    fourier_data[f"{n}_terms"] = fourier_approx(x, n)

# Visualize Fourier series convergence
x, fourier_data
