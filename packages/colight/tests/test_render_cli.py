import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import colight.env as env
import colight.plot as Plot
from colight.chrome_devtools import find_chrome
from colight.format import save_updates


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _has_js_dist() -> bool:
    widget_path = env.WIDGET_PATH
    if isinstance(widget_path, Path):
        return widget_path.exists()
    return False


def _require_runtime(video: bool = False) -> None:
    if not _has_js_dist():
        pytest.skip("colight JS bundle not built (js-dist missing)")
    try:
        chrome_path = find_chrome()
    except FileNotFoundError:
        chrome_path = None
    if not chrome_path:
        pytest.skip("Chrome not found for render tests")
    if video and not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not available for video render tests")


def _run_render(args: list[str]) -> None:
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = os.pathsep.join(
        [str(_repo_root()), env_vars.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, "-m", "colight_cli", "render", *args],
        check=True,
        env=env_vars,
        cwd=str(_repo_root()),
    )


def _scratch_output(filename: str) -> Path:
    scratch_dir = _repo_root() / "scratch" / "render-cli"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return scratch_dir / filename


def test_render_image_with_updates(tmp_path: Path) -> None:
    _require_runtime(video=False)

    plot = Plot.State({"value": 0}) | Plot.dot(
        {"x": np.array([1, 2]), "y": np.array([3, 4])}
    )
    base_path = tmp_path / "base.colight"
    plot.save_file(base_path)

    updates_path = tmp_path / "updates.colight"
    save_updates(
        updates_path,
        [Plot.State({"value": 1}), Plot.State({"value": 2})],
    )

    output_path = _scratch_output("render_image_with_updates.png")
    _run_render(
        [
            str(base_path),
            str(updates_path),
            "--out",
            str(output_path),
            "--last",
        ]
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_video_from_animate_by(tmp_path: Path) -> None:
    _require_runtime(video=True)

    plot = Plot.dot([1, 2, 3], x=Plot.js("$state.frame")) + Plot.Slider(
        "frame", range=[0, 2], fps=5
    )
    base_path = tmp_path / "base.colight"
    plot.save_file(base_path)

    output_path = _scratch_output("render_video_from_animate_by.mp4")
    _run_render(
        [
            str(base_path),
            "--out",
            str(output_path),
        ]
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
