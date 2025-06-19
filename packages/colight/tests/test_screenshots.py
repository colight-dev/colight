"""
Tests for WebGPU screenshot functionality in Colight
"""

import shutil
from pathlib import Path
import pytest
from colight.screenshots import ChromeContext
from colight.chrome_devtools import find_chrome, check_chrome_version


def chrome_available() -> bool:
    try:
        chrome_path = find_chrome()
        check_chrome_version(chrome_path)
        return True
    except Exception:
        return False


import colight.plot as Plot
from colight.scene3d import Ellipsoid

# Create an artifacts directory for screenshots
ARTIFACTS_DIR = Path("./tests/test-artifacts/")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)


def basic_scene():
    return (
        Plot.initialState({"test": "hello", "count": 3})
        | [
            "div",
            {"style": {"padding": "20px"}},
            Plot.js("$state.test"),
        ]
        | Ellipsoid(
            Plot.js("""
                Array.from({length: $state.count}, (_, i) => {
                    const t = i * Math.PI / 10;
                    return [
                        Math.cos(t),
                        Math.sin(t),
                        i / $state.count
                    ];
                }).flat()
            """),
            half_size=0.1,
            color=[1, 0, 0],  # Red color for all ellipsoids
        )
    )


@pytest.mark.skipif(not chrome_available(), reason="Chrome not installed")
def test_basic_screenshot():
    """Test basic screenshot functionality"""
    test_plot = basic_scene()

    test_plot.save_image(ARTIFACTS_DIR / "test.png", debug=True)
    test_plot.save_pdf(ARTIFACTS_DIR / "test.pdf", debug=True)


@pytest.mark.skipif(not chrome_available(), reason="Chrome not installed")
def test_counter_plot():
    """Test more complex plot with state updates"""
    counter_plot = (
        Plot.initialState({"count": 1})
        | [
            "div.bg-yellow-200.p-4",
            {"onClick": Plot.js("(e) => $state.clicks = ($state.clicks || 0) + 1")},
            Plot.js("`Count: ${$state.count}`"),
        ]
        | Plot.dot({"length": Plot.js("$state.count")}, x=Plot.index, y=Plot.index)
        + {"height": 200}
        | Ellipsoid(
            Plot.js("""
                Array.from({length: $state.count}, (_, i) => {
                    const t = i * Math.PI / 10;
                    return [
                        Math.cos(t),
                        Math.sin(t),
                        i / $state.count
                    ];
                }).flat()
            """),
            half_size=0.1,
            color=[1, 0, 0],  # Red color for all ellipsoids
        )
    )

    # Test single screenshot
    single_path = ARTIFACTS_DIR / "_single.png"
    counter_plot.save_image(single_path, debug=True)
    assert single_path.exists()

    # Test screenshot sequence
    paths = counter_plot.save_images(
        state_updates=[{"count": i} for i in [1, 10, 100]],
        output_dir=ARTIFACTS_DIR,
        filename_base="count",
        debug=True,
        width=2000,
    )
    for path in paths:
        assert path.exists()

    # Test video generation
    if shutil.which("ffmpeg"):
        video_path = ARTIFACTS_DIR / "counter.mp4"
        counter_plot.save_video(
            state_updates=[{"count": i} for i in range(30)],  # 30 frames
            filename=video_path,
            fps=12,
            debug=True,
        )
        assert video_path.exists()
        assert video_path.stat().st_size > 0


if __name__ == "__main__x":
    if chrome_available():
        test_basic_screenshot()
        test_counter_plot()

        with ChromeContext(debug=True, width=1024, height=768) as chrome:
            chrome.check_webgpu_support()
            chrome.save_gpu_info(ARTIFACTS_DIR / "gpu_status.pdf")
