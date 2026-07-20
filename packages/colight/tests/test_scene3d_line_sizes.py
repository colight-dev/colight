"""Tests for per-segment LineSegments radii (sizes).

Pure tests assert the Python surface flattens per-segment ``sizes`` onto the
component and that they cross the serialization boundary intact. The Chrome-
gated tests render two otherwise-identical line segments — one thick, one thin —
and show the difference in the GPU pick buffer: the thick segment covers many
more pickable pixels (coverage), and a pixel off the thin segment's axis picks
the thick segment but is background for the thin one.
"""

import json
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner

import colight.env as env
import colight.scene3d as scene3d
from colight.chrome_devtools import find_chrome
from colight.widget import to_json_with_state
from colight.cli_tools.structure import collect_structure
from colight_cli import main as cli_main


# =============================================================================
# Python surface: per-segment sizes flatten and serialize
# =============================================================================


def test_line_segments_sizes_flattened():
    seg = scene3d.LineSegments(
        starts=[[0, 0, 0], [1, 0, 0]],
        ends=[[1, 0, 0], [2, 0, 0]],
        sizes=[0.05, 0.5],
    )
    sizes = np.asarray(seg.props["sizes"]).reshape(-1)
    assert sizes.dtype == np.float32
    np.testing.assert_allclose(sizes, [0.05, 0.5])


def test_line_segments_sizes_reach_serialized_buffer():
    seg = scene3d.LineSegments(
        starts=[[0, 0, 0], [1, 0, 0]],
        ends=[[1, 0, 0], [2, 0, 0]],
        sizes=[0.05, 0.5],
    )
    scene = scene3d.Scene(seg)
    data, buffers = to_json_with_state(scene)
    arrays = {
        r.key: r
        for r in collect_structure(data, buffers).arrays
        if r.values is not None
    }
    assert "sizes" in arrays, f"sizes not serialized; had {sorted(arrays)}"
    np.testing.assert_allclose(
        np.asarray(arrays["sizes"].values).reshape(-1), [0.05, 0.5], atol=1e-6
    )


def test_line_segments_scalar_size_still_supported():
    seg = scene3d.LineSegments(starts=[[0, 0, 0]], ends=[[1, 0, 0]], size=0.3)
    assert seg.props["size"] == 0.3
    assert "sizes" not in seg.props


# =============================================================================
# End-to-end (Chrome-gated): thick vs thin segments differ in the pick buffer
# =============================================================================

# Two single-segment LineSegments components, both running along X at the same
# place, viewed head-on from +Z. Component 0 is thick (radius 0.6), component 1
# is thin (radius 0.02). The thick beam paints far more pickable pixels.
SCENE = """import colight.scene3d as S

(
    S.Scene(
        S.LineSegments(
            starts=[[-2.0, 1.0, 0.0]], ends=[[2.0, 1.0, 0.0]],
            sizes=[0.6], color=[0.9, 0.2, 0.2],
        )
        + S.LineSegments(
            starts=[[-2.0, -1.0, 0.0]], ends=[[2.0, -1.0, 0.0]],
            sizes=[0.02], color=[0.2, 0.4, 0.9],
        )
    )
    + {"defaultCamera": {"position": [0, 0, 8], "target": [0, 0, 0],
                         "up": [0, 1, 0], "fov": 45}}
)
"""

SIZE_ARGS = ["--width", "500", "--height", "500"]


def _require_renderer() -> None:
    widget_path = env.WIDGET_PATH
    if not (isinstance(widget_path, pathlib.Path) and widget_path.exists()):
        pytest.skip("colight JS bundle not built (js-dist missing)")
    try:
        chrome_path = find_chrome()
    except FileNotFoundError:
        chrome_path = None
    if not chrome_path:
        pytest.skip("Chrome not found for line-size tests")


@pytest.fixture
def scene_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "lines.py"
    path.write_text(SCENE)
    return path


def test_thick_segment_covers_more_pixels(scene_file: pathlib.Path, tmp_path):
    _require_renderer()
    result = CliRunner().invoke(
        cli_main,
        [
            "screenshot",
            str(scene_file),
            "--out",
            str(tmp_path / "shot.png"),
            *SIZE_ARGS,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    coverage = json.loads(result.output)["coverage"]
    # Both components are LineSegments; coverage lists them in scene order:
    # component 0 = thick, component 1 = thin.
    comps = coverage["components"]
    assert [c["type"] for c in comps] == ["LineSegments", "LineSegments"]
    thick_px, thin_px = comps[0]["pixels"], comps[1]["pixels"]
    assert thick_px > 0 and thin_px > 0
    # A 30x-radius difference must be plainly visible in coverage; require a
    # large margin so this is not a rounding artifact.
    assert thick_px > thin_px * 5, f"thick {thick_px} vs thin {thin_px}"


def _pick_component_at(scene_path: pathlib.Path, x: int, y: int):
    # pick-at exits 0 on a hit and 1 on background (2 is an actual error).
    result = CliRunner().invoke(
        cli_main,
        ["pick-at", str(scene_path), f"{x},{y}", *SIZE_ARGS, "--json"],
    )
    assert result.exit_code in (0, 1), result.output
    hits = json.loads(result.output).get("hits") or []
    return hits[0]["component"] if hits else None


def test_pick_scan_thick_band_is_wider(scene_file: pathlib.Path):
    _require_renderer()
    # Scan a vertical column of pixels through both horizontal beams and measure
    # how many rows each component's pick id occupies. The thick beam (radius
    # 0.6) must occupy a far taller band than the thin one (radius 0.02) — the
    # per-segment radius reaching the pick buffer, row by row.
    thick_rows = 0
    thin_rows = 0
    thick_ys = []
    thin_ys = []
    for y in range(120, 380):
        comp = _pick_component_at(scene_file, 250, y)
        if comp == 0:
            thick_rows += 1
            thick_ys.append(y)
        elif comp == 1:
            thin_rows += 1
            thin_ys.append(y)
    assert thick_rows > 0 and thin_rows > 0, (thick_rows, thin_rows)
    # The 30x radius difference shows up as a much taller pickable band.
    assert thick_rows > thin_rows * 4, f"thick {thick_rows} rows vs thin {thin_rows}"

    # Direct off-axis evidence: a pixel ~15px off the thick beam's own axis is
    # still inside its fat radius (picks comp 0); the same offset from the thin
    # beam's axis is background (the thin radius can't reach it).
    thick_axis = (min(thick_ys) + max(thick_ys)) // 2
    thin_axis = (min(thin_ys) + max(thin_ys)) // 2
    assert _pick_component_at(scene_file, 250, thick_axis) == 0
    assert _pick_component_at(scene_file, 250, thick_axis + 15) == 0
    assert _pick_component_at(scene_file, 250, thin_axis) == 1
    assert _pick_component_at(scene_file, 250, thin_axis + 15) is None
