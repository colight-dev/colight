#!/usr/bin/env python3
"""
Script to update visual test baselines.

Run this when you've intentionally changed something visual and need to
update the reference images for regression testing.

Usage (from repository root):
    uv run python scripts/update_baselines.py

This will:
1. Generate all visual test plots
2. Save them as new baselines
3. Report what was updated
"""

import sys
import shutil
import os
from pathlib import Path

# Change to colight package directory and add to Python path
colight_dir = Path(__file__).parent.parent / "packages" / "colight"
os.chdir(colight_dir)
sys.path.insert(0, str(colight_dir))

from tests.visual.test_visual_regression import (  # pyright: ignore
    create_comprehensive_plot,
    chrome_available,
)
from tests.visual.utils import save_baseline, get_test_paths  # pyright: ignore


def update_baselines():
    """Update all visual test baselines."""

    if not chrome_available():
        print("❌ Chrome not available - cannot update baselines")
        return False

    print("🔄 Updating visual test baselines...")

    # Create output directory
    output_dir = Path("./scratch/baseline_update/")
    output_dir.mkdir(exist_ok=True, parents=True)

    updated_baselines = []

    # Update comprehensive plot baseline
    try:
        print("  📊 Generating comprehensive plot...")
        plot = create_comprehensive_plot()

        baseline_path, actual_path, diff_path = get_test_paths(
            "comprehensive_plot", output_dir
        )

        # Generate the image
        plot.save_image(str(actual_path), width=1200, height=800, debug=True)

        # Save as baseline
        save_baseline(actual_path, baseline_path)
        updated_baselines.append(baseline_path)

    except Exception as e:
        print(f"❌ Failed to update comprehensive_plot baseline: {e}")
        return False

    # Clean up temporary files
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Report results
    print(f"\n✅ Updated {len(updated_baselines)} baseline(s):")
    for baseline in updated_baselines:
        print(f"   - {baseline}")

    print("\n💡 Next steps:")
    print("   1. Review the updated baselines")
    print("   2. Run tests: uv run pytest tests/visual/")
    print("   3. Commit the baseline changes if they look correct")

    return True


if __name__ == "__main__":
    success = update_baselines()
    exit(0 if success else 1)
