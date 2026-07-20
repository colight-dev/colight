"""Tests for per-instance filtering on scene3d primitives (filter_by).

Covers the Python API (values coerced once, thresholds passed through),
inspect / screenshot --json reporting of active filters, and the GPU truth:
filtered-out instances are hidden AND unpickable, so pick-where visible-instance
counts change with the threshold.

Pure tests run everywhere; end-to-end tests drive the real CLI through headless
Chrome (skipped when Chrome or the JS bundle is missing).
"""

import json
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner

import colight.env as env
import colight.plot as Plot
import colight.scene3d as scene3d
from colight.chrome_devtools import find_chrome
from colight.cli_tools import inspect_tools
from colight.inspect import inspect as colight_inspect
from colight_cli import main as cli_main

# =============================================================================
# API: values coerced once, thresholds pass through
# =============================================================================

centers4 = np.array([[i, 0, 0] for i in range(4)], dtype=np.float32)
FVALUES = [0.1, 0.4, 0.6, 0.9]


def props_of(component):
    return component.to_js_call().args[0]


def assert_filter_props(props, *, expect_min, expect_max, label="grade"):
    fb = props["filter_by"]
    assert fb["values"].dtype == np.float32
    np.testing.assert_allclose(fb["values"], FVALUES, atol=1e-6)
    assert fb.get("min") == expect_min
    assert fb.get("max") == expect_max
    assert fb["label"] == label
    # The raw values must NOT be duplicated as a top-level data prop.
    assert "values" not in props


def test_point_cloud_filter_by():
    c = scene3d.PointCloud(
        centers4,
        filter_by={"values": FVALUES, "min": 0.3, "max": 0.8, "label": "grade"},
    )
    assert_filter_props(props_of(c), expect_min=0.3, expect_max=0.8)


def test_ellipsoid_filter_by():
    c = scene3d.Ellipsoid(
        centers4,
        half_size=0.4,
        filter_by={"values": FVALUES, "min": 0.3, "max": 0.8, "label": "grade"},
    )
    assert_filter_props(props_of(c), expect_min=0.3, expect_max=0.8)


def test_cuboid_filter_by():
    c = scene3d.Cuboid(
        centers4,
        half_size=0.4,
        filter_by={"values": FVALUES, "min": 0.3, "max": 0.8, "label": "grade"},
    )
    assert_filter_props(props_of(c), expect_min=0.3, expect_max=0.8)


def test_line_segments_filter_by():
    c = scene3d.LineSegments(
        starts=centers4,
        ends=centers4 + 1,
        filter_by={"values": FVALUES, "min": 0.3, "max": 0.8, "label": "grade"},
    )
    assert_filter_props(props_of(c), expect_min=0.3, expect_max=0.8)


def test_line_beams_filter_by():
    pts = np.array([[x, 0, 0, 0] for x in range(5)], dtype=np.float32).reshape(-1)
    c = scene3d.LineBeams(
        pts,
        filter_by={"values": FVALUES, "min": 0.3, "max": 0.8, "label": "grade"},
    )
    assert_filter_props(props_of(c), expect_min=0.3, expect_max=0.8)


def test_filter_by_min_only():
    c = scene3d.Cuboid(centers4, filter_by={"values": FVALUES, "min": 0.5})
    fb = props_of(c)["filter_by"]
    assert fb.get("min") == 0.5
    assert "max" not in fb


def test_filter_by_state_ref_min():
    # A $state threshold reference passes through unchanged (resolved on JS).
    ref = Plot.js("$state.cutoff")
    c = scene3d.Cuboid(centers4, filter_by={"values": FVALUES, "min": ref})
    assert props_of(c)["filter_by"]["min"] is ref


def test_filter_by_requires_values():
    with pytest.raises(ValueError, match="filter_by requires 'values'"):
        scene3d.Cuboid(centers4, filter_by={"min": 0.5})  # type: ignore[typeddict-item]


# =============================================================================
# colight inspect: active filters reported without rendering
# =============================================================================


def make_artifact(tmp_path: pathlib.Path, scene) -> pathlib.Path:
    visual = colight_inspect(scene)
    assert visual is not None
    target = tmp_path / "scene.colight"
    target.write_bytes(visual.to_bytes())
    return target


def test_inspect_reports_filters(tmp_path):
    scene = scene3d.Scene(
        scene3d.Cuboid(
            centers4,
            half_size=0.5,
            filter_by={"values": FVALUES, "min": 0.3, "max": 0.8, "label": "grade"},
        ),
        scene3d.PointCloud(centers4, color=[1, 0, 0]),
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    filters = payload["visual"]["filters"]
    assert filters == [
        {
            "component": "scene3d.Cuboid",
            "label": "grade",
            "min": 0.3,
            "max": 0.8,
        }
    ]
    # The filter's per-instance values must not leak into the array records.
    assert not any("filter_by" in a["path"] for a in payload["visual"]["arrays"])


def test_inspect_omits_filters_when_absent(tmp_path):
    target = make_artifact(
        tmp_path, scene3d.Scene(scene3d.PointCloud(centers4, color=[1, 0, 0]))
    )
    payload = inspect_tools.inspect_target(target)
    assert "filters" not in payload["visual"]


# =============================================================================
# End-to-end: filtered-out instances are hidden AND unpickable (Chrome-gated)
# =============================================================================

# Four cuboids on a row at x = -3, -1, 1, 3, grade values 0.1..0.9. A slider
# state cutoff drives filter_by.min; instances below the cutoff are collapsed.
SCENE_TEMPLATE = """import colight.scene3d as S

(
    S.Cuboid(
        centers=[[-3.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]],
        half_size=0.5,
        color=[0.2, 0.6, 0.9],
        filter_by={{"values": [0.1, 0.4, 0.6, 0.9], "min": {cutoff}, "label": "grade"}},
    )
    + {{"defaultCamera": {{"position": [0, 0, 12], "target": [0, 0, 0],
                          "up": [0, 1, 0], "fov": 45}}}}
)
"""

SIZE_ARGS = ["--width", "400", "--height", "400"]


def _require_renderer() -> None:
    widget_path = env.WIDGET_PATH
    if not (isinstance(widget_path, pathlib.Path) and widget_path.exists()):
        pytest.skip("colight JS bundle not built (js-dist missing)")
    try:
        chrome_path = find_chrome()
    except FileNotFoundError:
        chrome_path = None
    if not chrome_path:
        pytest.skip("Chrome not found for filter tests")


def _scene_at(tmp_path: pathlib.Path, cutoff: float, name: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(SCENE_TEMPLATE.format(cutoff=cutoff))
    return path


def _visible_count(scene_path: pathlib.Path) -> int:
    """pick-where over all four instances -> how many are visible/pickable."""
    result = CliRunner().invoke(
        cli_main,
        [
            "pick-where",
            str(scene_path),
            "--component",
            "0",
            "--instances",
            "0-3",
            *SIZE_ARGS,
            "--json",
        ],
    )
    payload = json.loads(result.output)
    return payload.get("visible_instances", 0)


def test_pick_where_visible_count_changes_with_cutoff(tmp_path):
    _require_renderer()
    # cutoff 0.0 keeps all four; cutoff 0.5 hides the two below 0.5 (0.1, 0.4).
    low = _visible_count(_scene_at(tmp_path, 0.0, "low.py"))
    high = _visible_count(_scene_at(tmp_path, 0.5, "high.py"))
    assert low == 4, f"expected all 4 visible at cutoff 0.0, got {low}"
    assert high == 2, f"expected 2 visible at cutoff 0.5, got {high}"
    assert high < low


def test_screenshot_json_reports_filters_and_stays_deterministic(tmp_path):
    _require_renderer()
    scene_path = _scene_at(tmp_path, 0.5, "shot_scene.py")
    out = tmp_path / "shot.png"
    result = CliRunner().invoke(
        cli_main,
        [
            "screenshot",
            str(scene_path),
            "--out",
            str(out),
            *SIZE_ARGS,
            "--check",
            "--json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["deterministic"] is True
    assert payload["filters"] == [
        {"component": 0, "type": "Cuboid", "label": "grade", "min": 0.5, "max": None}
    ]
