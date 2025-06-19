"""
Visual testing utilities for pixel-perfect comparison and baseline management.
"""

import shutil
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw


def compare_images(baseline_path: Path, actual_path: Path, diff_path: Path) -> bool:
    """
    Compare two images pixel-by-pixel and generate a diff image.

    Args:
        baseline_path: Path to the reference image
        actual_path: Path to the generated image
        diff_path: Path where diff image will be saved

    Returns:
        True if images are identical, False otherwise
    """
    if not baseline_path.exists():
        print(f"Baseline image not found: {baseline_path}")
        return False

    if not actual_path.exists():
        print(f"Actual image not found: {actual_path}")
        return False

    # Load images
    baseline = Image.open(baseline_path).convert("RGB")
    actual = Image.open(actual_path).convert("RGB")

    # Check if dimensions match
    if baseline.size != actual.size:
        print(f"Image size mismatch: baseline {baseline.size} vs actual {actual.size}")
        return False

    # Pixel-perfect comparison
    diff = ImageChops.difference(baseline, actual)

    # Check if images are identical
    if diff.getbbox() is None:
        return True

    # Generate diff visualization
    diff_visual = diff.copy()
    diff_visual = ImageChops.multiply(
        diff_visual, Image.new("RGB", diff_visual.size, (4, 4, 4))
    )

    # Create a composite showing baseline | actual | diff
    width, height = baseline.size
    composite = Image.new("RGB", (width * 3, height))
    composite.paste(baseline, (0, 0))
    composite.paste(actual, (width, 0))
    composite.paste(diff_visual, (width * 2, 0))

    # Add labels
    draw = ImageDraw.Draw(composite)
    try:
        # Use default font
        draw.text((10, 10), "BASELINE", fill="white")
        draw.text((width + 10, 10), "ACTUAL", fill="white")
        draw.text((width * 2 + 10, 10), "DIFF", fill="white")
    except Exception:
        # If font loading fails, continue without labels
        pass

    composite.save(diff_path)
    return False


def save_baseline(actual_path: Path, baseline_path: Path) -> None:
    """
    Save an actual image as a new baseline.

    Args:
        actual_path: Path to the generated image
        baseline_path: Path where baseline will be saved
    """
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(actual_path, baseline_path)
    print(f"Saved baseline: {baseline_path}")
