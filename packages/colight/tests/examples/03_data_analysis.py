# Data Analysis Workflow
# This example demonstrates a complete data analysis pipeline.

import numpy as np

# Generate synthetic dataset
np.random.seed(123)
n_samples = 1000

# Features
temperature = np.random.normal(20, 5, n_samples)  # Temperature in Celsius
humidity = np.random.normal(60, 15, n_samples)  # Humidity percentage

# Dependent variable with some noise
ice_cream_sales = (
    50  # Base sales
    + 2 * temperature  # Temperature effect
    + 0.5 * humidity  # Humidity effect
    + np.random.normal(0, 10, n_samples)  # Random noise
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
    "temperature_mean": np.mean(temperature),
    "temperature_std": np.std(temperature),
    "humidity_mean": np.mean(humidity),
    "humidity_std": np.std(humidity),
    "sales_mean": np.mean(ice_cream_sales),
    "sales_std": np.std(ice_cream_sales),
    "temp_sales_corr": correlation_temp,
    "humidity_sales_corr": correlation_humidity,
}

# Final summary visualization
stats_summary
