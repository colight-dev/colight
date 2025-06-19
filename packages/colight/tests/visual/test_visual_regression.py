"""
Visual regression tests for Colight 2D plots.

This demonstrates comprehensive visual testing with multiple chart types,
KaTeX mathematics, and markdown formatting.
"""

from pathlib import Path
import pytest
import numpy as np
import colight.plot as Plot
from colight.chrome_devtools import find_chrome, check_chrome_version
from tests.visual.utils import compare_images


def chrome_available() -> bool:
    try:
        chrome_path = find_chrome()
        check_chrome_version(chrome_path)
        return True
    except Exception:
        return False


# Create output directory for test artifacts
TEST_OUTPUT_DIR = Path("./packages/colight/test-artifacts/visual/")
TEST_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def create_comprehensive_plot():
    """
    Create a comprehensive plot showcasing multiple chart types,
    KaTeX equations, and markdown formatting.

    This serves as both a visual regression test and a good example
    of Colight's capabilities.
    """
    # Generate sample data
    np.random.seed(42)  # Deterministic for testing

    # Scatter plot data
    scatter_data = np.random.multivariate_normal(
        [0.3, 0.7], [[0.01, 0.005], [0.005, 0.01]], 50
    )

    # Line plot data
    x = np.linspace(0, 1, 20)
    line_data = np.column_stack([x, 0.5 + 0.2 * np.sin(x * 6)])

    # Bar chart data
    categories = ["Alpha", "Beta", "Gamma", "Delta"]
    values = [0.8, 0.6, 0.9, 0.4]
    bar_data = [{"category": c, "value": v} for c, v in zip(categories, values)]

    # Create the three chart types in a row
    scatter_plot = (
        Plot.dot(scatter_data, fill="steelblue", r=4)
        + Plot.title("Scatter Plot")
        + Plot.domain([0, 1], [0, 1])
    )

    line_plot = (
        Plot.line(line_data, stroke="darkgreen", strokeWidth=2)
        + Plot.title("Line Chart")
        + Plot.domain([0, 1], [0, 1])
    )

    bar_plot = (
        Plot.barY(bar_data, x="category", y="value", fill="coral")
        + Plot.title("Bar Chart")
        + {"marginBottom": 40}
    )

    # Combine charts horizontally
    charts_row = scatter_plot & line_plot & bar_plot

    # Add KaTeX equation
    equation = Plot.md(
        r"""### Mathematical Example

The **Gaussian distribution** is fundamental in statistics:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$$

where $\mu$ is the mean and $\sigma$ is the standard deviation.""",
        className="bg-gray-50 p-4 rounded",
    )

    # Add descriptive markdown
    description = Plot.md(
        """## Visual Regression Test Example

This plot demonstrates:
- **Multiple chart types**: scatter, line, and bar charts
- **Mathematical notation**: rendered with KaTeX
- **Consistent styling**: using Colight's layout system

This serves as both a regression test and a usage example.""",
        className="bg-blue-50 p-4 rounded mt-4",
    )

    # Combine everything vertically
    return charts_row | equation | description


def get_test_paths(test_name: str, output_dir: Path) -> tuple[Path, Path, Path]:
    """
    Get the standard paths for a visual test.

    Args:
        test_name: Name of the test (without extension)
        output_dir: Directory for test outputs

    Returns:
        Tuple of (baseline_path, actual_path, diff_path)
    """
    baseline_path = Path("packages/colight/tests/visual/baselines") / f"{test_name}.png"
    actual_path = output_dir / f"{test_name}.png"
    diff_path = output_dir / f"{test_name}_diff.png"

    return baseline_path, actual_path, diff_path


@pytest.mark.skipif(not chrome_available(), reason="Chrome not installed")
def test_comprehensive_visual_regression():
    """
    Test comprehensive plot with multiple chart types, KaTeX, and markdown.

    This test demonstrates:
    - Pixel-perfect comparison for visual regression testing
    - Before/after/diff generation for failed tests
    - Good example of comprehensive plot testing
    """
    plot = create_comprehensive_plot()

    # Get test paths
    baseline_path, actual_path, diff_path = get_test_paths(
        "comprehensive_plot", TEST_OUTPUT_DIR
    )

    # Generate the actual image
    plot.save_image(str(actual_path), width=1200, height=800, debug=True)

    # Compare against baseline
    if baseline_path.exists():
        images_match = compare_images(baseline_path, actual_path, diff_path)

        if not images_match:
            pytest.fail(
                f"Visual regression detected!\n"
                f"Baseline: {baseline_path}\n"
                f"Actual: {actual_path}\n"
                f"Diff: {diff_path}\n"
                f"To update baseline: uv run python scripts/update_baselines.py"
            )
    else:
        pytest.fail(
            f"Baseline image not found: {baseline_path}\n"
            f"To create baseline: uv run python scripts/update_baselines.py"
        )


if __name__ == "__main__":
    """Allow running test directly for development"""
    if chrome_available():
        plot = create_comprehensive_plot()
        test_path = TEST_OUTPUT_DIR / "comprehensive_plot_dev.png"
        plot.save_image(str(test_path), width=1200, height=800, debug=True)
        print(f"Generated test image: {test_path}")
    else:
        print("Chrome not available - cannot generate test image")
