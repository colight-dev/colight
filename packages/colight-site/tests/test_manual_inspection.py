"""Generate test artifacts for manual inspection."""

import pathlib
from colight_site.builder import build_file


def test_generate_artifacts_for_inspection():
    """Generate various colight-site examples in test-artifacts for manual inspection."""

    # Create test-artifacts directory
    artifacts_dir = (
        pathlib.Path(__file__).parent.parent.parent.parent
        / "test-artifacts"
        / "colight-site"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any existing files
    for file in artifacts_dir.rglob("*"):
        if file.is_file():
            file.unlink()
    for dir in artifacts_dir.rglob("*"):
        if dir.is_dir() and dir != artifacts_dir:
            try:
                dir.rmdir()
            except OSError:
                pass

    # Example 1: Basic numpy visualization
    example1_path = artifacts_dir / "01_basic_numpy.colight.py"
    example1_content = """# Basic NumPy Visualization
# This example demonstrates simple data visualization with numpy.

import numpy as np

# Create sample data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Generate visualization data
x, y

# Let's also look at some statistics
print(f"Data range: {np.min(y):.3f} to {np.max(y):.3f}")

# Mean and standard deviation
np.mean(y), np.std(y)
"""

    example1_path.write_text(example1_content)
    build_file(example1_path, artifacts_dir / "01_basic_numpy.md", verbose=True)

    # Example 2: Multiple visualizations
    example2_path = artifacts_dir / "02_multiple_viz.colight.py"
    example2_content = """# Multiple Visualizations
# This example shows several different types of data.

import numpy as np

# Linear data
x_linear = np.linspace(0, 10, 50)
y_linear = 2 * x_linear + 1

# First visualization: linear relationship
x_linear, y_linear

# Trigonometric data
x_trig = np.linspace(0, 4*np.pi, 80)
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
    'linear_slope': 2.0,
    'trig_amplitude': 1.0,
    'random_mean': np.mean(random_data),
    'random_std': np.std(random_data)
}

# Fourth visualization: summary
summary_stats
"""

    example2_path.write_text(example2_content)
    build_file(example2_path, artifacts_dir / "02_multiple_viz.md", verbose=True)

    # Example 3: Data analysis workflow
    example3_path = artifacts_dir / "03_data_analysis.colight.py"
    example3_content = """# Data Analysis Workflow
# This example demonstrates a complete data analysis pipeline.

import numpy as np

# Generate synthetic dataset
np.random.seed(123)
n_samples = 1000

# Features
temperature = np.random.normal(20, 5, n_samples)  # Temperature in Celsius
humidity = np.random.normal(60, 15, n_samples)    # Humidity percentage

# Dependent variable with some noise
ice_cream_sales = (
    50 +  # Base sales
    2 * temperature +  # Temperature effect
    0.5 * humidity +   # Humidity effect
    np.random.normal(0, 10, n_samples)  # Random noise
)

# Data exploration
print(f"Dataset size: {n_samples} samples")
print(f"Temperature range: {np.min(temperature):.1f}°C to {np.max(temperature):.1f}°C")
print(f"Humidity range: {np.min(humidity):.1f}% to {np.max(humidity):.1f}%")

# Visualize the relationships
temperature, ice_cream_sales

# Correlation analysis
correlation_temp = np.corrcoef(temperature, ice_cream_sales)[0, 1]
correlation_humidity = np.corrcoef(humidity, ice_cream_sales)[0, 1]

print(f"Temperature correlation: {correlation_temp:.3f}")
print(f"Humidity correlation: {correlation_humidity:.3f}")

# Create correlation matrix
correlation_matrix = np.corrcoef([temperature, humidity, ice_cream_sales])

# Visualize correlation matrix
correlation_matrix

# Summary statistics
stats_summary = {
    'temperature_mean': np.mean(temperature),
    'temperature_std': np.std(temperature),
    'humidity_mean': np.mean(humidity), 
    'humidity_std': np.std(humidity),
    'sales_mean': np.mean(ice_cream_sales),
    'sales_std': np.std(ice_cream_sales),
    'temp_sales_corr': correlation_temp,
    'humidity_sales_corr': correlation_humidity
}

# Final summary visualization
stats_summary
"""

    example3_path.write_text(example3_content)
    build_file(example3_path, artifacts_dir / "03_data_analysis.md", verbose=True)

    # Example 4: Mathematical exploration
    example4_path = artifacts_dir / "04_math_exploration.colight.py"
    example4_content = '''# Mathematical Function Exploration
# Exploring different mathematical functions and their properties.

import numpy as np

# Domain for our functions
x = np.linspace(-2*np.pi, 2*np.pi, 200)

# Polynomial functions
def polynomial_family(x, degree):
    """Generate polynomial functions of different degrees."""
    results = {}
    for d in range(1, degree + 1):
        results[f'x^{d}'] = x**d
    return results

# Generate polynomial data
poly_data = polynomial_family(x, 4)

# Visualize polynomial progression
poly_data

# Trigonometric functions
trig_functions = {
    'sin(x)': np.sin(x),
    'cos(x)': np.cos(x), 
    'tan(x)': np.tan(x),
    'sin(2x)': np.sin(2*x),
    'cos(x/2)': np.cos(x/2)
}

# Visualize trigonometric family
x, trig_functions

# Exponential and logarithmic (positive domain)
x_pos = np.linspace(0.1, 5, 100)
exp_log_functions = {
    'exp(x)': np.exp(x_pos),
    'log(x)': np.log(x_pos),
    'sqrt(x)': np.sqrt(x_pos),
    'x^2': x_pos**2
}

# Visualize exponential/log family
x_pos, exp_log_functions

# Function composition
composite = np.sin(x) * np.exp(-x**2 / 10)

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
    fourier_data[f'{n}_terms'] = fourier_approx(x, n)

# Visualize Fourier series convergence
x, fourier_data
'''

    example4_path.write_text(example4_content)
    build_file(example4_path, artifacts_dir / "04_math_exploration.md", verbose=True)

    # Example 5: Error handling demonstration
    example5_path = artifacts_dir / "05_mixed_content.colight.py"
    example5_content = """# Mixed Content Example
# This example shows various types of content and edge cases.

import numpy as np

# Start with some narrative
# This is just explanatory text that becomes markdown.
# 
# We can have multiple paragraphs.

# Simple computation
result = 2 + 2
print(f"2 + 2 = {result}")

# Some data that will be visualized
data = [1, 4, 9, 16, 25, 36]

# Visualize the data
data

# More narrative
# Here we continue with more explanation.
# This demonstrates how narrative and code are interwoven.

# Non-visualizable return values
"This is just a string"

42

True

None

# These won't create .colight files, just return values

# Arrays that will be visualized
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([1, 4, 9, 16, 25])

# Multiple arrays
arr1, arr2

# Dictionary with mixed data
mixed_data = {
    'numbers': [1, 2, 3, 4, 5],
    'squares': [1, 4, 9, 16, 25],
    'description': 'Numbers and their squares'
}

# Visualize dictionary
mixed_data

# Final section
# This concludes our mixed content example.
# Notice how both text and code flow together naturally.
"""

    example5_path.write_text(example5_content)
    build_file(example5_path, artifacts_dir / "05_mixed_content.md", verbose=True)

    # Create an index file
    index_content = """# Colight-Site Test Artifacts

This directory contains test artifacts generated by colight-site for manual inspection.

## Examples

1. **01_basic_numpy.md** - Simple numpy visualization example
2. **02_multiple_viz.md** - Multiple visualizations in one document  
3. **03_data_analysis.md** - Complete data analysis workflow
4. **04_math_exploration.md** - Mathematical function exploration
5. **05_mixed_content.md** - Mixed content with various data types

## Files Generated

For each `.colight.py` source file, colight-site generates:
- A `.md` markdown file with the narrative and code blocks
- A `_colight/` directory containing `.colight` visualization files
- Each visualization is embedded using `<div class="colight-embed" data-src="...">` tags

## Viewing

You can view the generated markdown files and inspect the `.colight` files in the `*_colight/` directories to see how visualizations are serialized.
"""

    (artifacts_dir / "README.md").write_text(index_content)

    print(f"\nGenerated test artifacts in: {artifacts_dir}")
    print("Files created:")
    for file in sorted(artifacts_dir.rglob("*")):
        if file.is_file():
            print(f"  {file.relative_to(artifacts_dir)}")


if __name__ == "__main__":
    test_generate_artifacts_for_inspection()
