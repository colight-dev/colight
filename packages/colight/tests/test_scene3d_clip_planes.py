"""Tests for scene-level section / clipping planes (Scene(clip_planes=...)).

Covers the Python API (normal normalization, origin-aware point->offset
conversion, state-ref offsets, validation, the 8-plane cap), inspect reporting
of active clip_planes, and — Chrome-gated — the GPU truth: a sweeping plane
hides geometry (pick-where visible count drops), pick-at in the clipped region
reaches the interior instance exposed BEHIND the cut, and screenshot --check
stays byte-deterministic with an active plane.

Pure tests run everywhere; end-to-end tests drive the real CLI through headless
Chrome (skipped when Chrome or the JS bundle is missing).
"""

import json
import math
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
# API: normal normalization, offset / point forms, validation
# =============================================================================


def planes_of(scene: scene3d.Scene):
    """The normalized clip_planes as they cross the JS boundary."""
    return scene.for_json()[1].get("clipPlanes")


def test_normal_is_normalized():
    s = scene3d.Scene(
        scene3d.Cuboid([0, 0, 0], half_size=0.4),
        clip_planes=[{"normal": [0, 2, 0], "offset": 5.0}],
    )
    planes = planes_of(s)
    assert planes is not None
    np.testing.assert_allclose(planes[0]["normal"], [0, 1, 0], atol=1e-9)
    assert planes[0]["offset"] == 5.0


def test_point_form_no_origin():
    # offset = dot(point, normal)
    s = scene3d.Scene(
        scene3d.Cuboid([0, 0, 0], half_size=0.4),
        clip_planes=[{"normal": [0, 1, 0], "point": [0, 3, 0]}],
    )
    assert planes_of(s)[0]["offset"] == pytest.approx(3.0)


def test_point_form_respects_origin():
    # Positions are shifted by -origin; the anchor must be too:
    # offset = dot(point - origin, normal).
    s = scene3d.Scene(
        scene3d.Cuboid([0, 0, 0], half_size=0.4),
        origin=[0, 10, 0],
        clip_planes=[{"normal": [0, 1, 0], "point": [0, 13, 0]}],
    )
    assert planes_of(s)[0]["offset"] == pytest.approx(3.0)


def test_point_form_oblique_normal_with_origin():
    origin = np.array([100.0, 200.0, 300.0])
    normal = np.array([1.0, 1.0, 0.0])
    unit = normal / np.linalg.norm(normal)
    point = np.array([105.0, 207.0, 300.0])
    s = scene3d.Scene(
        scene3d.Cuboid([0, 0, 0], half_size=0.4),
        origin=origin.tolist(),
        clip_planes=[{"normal": normal.tolist(), "point": point.tolist()}],
    )
    expected = float(np.dot(point - origin, unit))
    assert planes_of(s)[0]["offset"] == pytest.approx(expected)


def test_state_ref_offset_passes_through():
    ref = Plot.js("$state.section_y")
    s = scene3d.Scene(
        scene3d.Cuboid([0, 0, 0], half_size=0.4),
        clip_planes=[{"normal": [0, 1, 0], "offset": ref}],
    )
    assert planes_of(s)[0]["offset"] is ref


def test_state_ref_point_becomes_js_dot_expression():
    s = scene3d.Scene(
        scene3d.Cuboid([0, 0, 0], half_size=0.4),
        origin=[0, 10, 0],
        clip_planes=[
            {"normal": [0, 1, 0], "point": [0, Plot.js("$state.sy"), 0]},
        ],
    )
    offset = planes_of(s)[0]["offset"]
    # A JSCode expression that folds in origin + normal and reads $state.
    assert offset.for_json()["__type__"] == "js_source"
    assert "$state.sy" in offset.code
    assert "10" in offset.code  # origin subtracted


def test_multiple_planes_preserved():
    s = scene3d.Scene(
        scene3d.Cuboid([0, 0, 0], half_size=0.4),
        clip_planes=[
            {"normal": [0, 1, 0], "offset": 1.0},
            {"normal": [1, 0, 0], "offset": 2.0},
        ],
    )
    planes = planes_of(s)
    assert len(planes) == 2
    assert planes[1]["offset"] == 2.0


def test_over_cap_raises():
    with pytest.raises(ValueError, match="at most 8 clip_planes"):
        scene3d.Scene(
            scene3d.Cuboid([0, 0, 0], half_size=0.4),
            clip_planes=[{"normal": [0, 1, 0], "offset": float(i)} for i in range(9)],
        )


def test_exactly_eight_ok():
    s = scene3d.Scene(
        scene3d.Cuboid([0, 0, 0], half_size=0.4),
        clip_planes=[{"normal": [0, 1, 0], "offset": float(i)} for i in range(8)],
    )
    assert len(planes_of(s)) == 8


def test_zero_normal_raises():
    with pytest.raises(ValueError, match="non-zero"):
        scene3d.Scene(
            scene3d.Cuboid([0, 0, 0], half_size=0.4),
            clip_planes=[{"normal": [0, 0, 0], "offset": 1.0}],
        )


def test_missing_normal_raises():
    with pytest.raises(ValueError, match="requires a 'normal'"):
        scene3d.Scene(
            scene3d.Cuboid([0, 0, 0], half_size=0.4),
            clip_planes=[{"offset": 1.0}],
        )


def test_offset_and_point_both_raises():
    with pytest.raises(ValueError, match="not both"):
        scene3d.Scene(
            scene3d.Cuboid([0, 0, 0], half_size=0.4),
            clip_planes=[{"normal": [0, 1, 0], "offset": 1.0, "point": [0, 1, 0]}],
        )


def test_neither_offset_nor_point_raises():
    with pytest.raises(ValueError, match="either 'offset' or 'point'"):
        scene3d.Scene(
            scene3d.Cuboid([0, 0, 0], half_size=0.4),
            clip_planes=[{"normal": [0, 1, 0]}],
        )


def test_no_clip_planes_absent_from_json():
    s = scene3d.Scene(scene3d.Cuboid([0, 0, 0], half_size=0.4))
    assert planes_of(s) is None


def test_clip_planes_preserved_across_addition():
    left = scene3d.Scene(
        scene3d.Cuboid([0, 0, 0], half_size=0.4),
        clip_planes=[{"normal": [0, 1, 0], "offset": 2.0}],
    )
    combined = left + scene3d.Cuboid([1, 1, 1], half_size=0.4)
    planes = planes_of(combined)
    assert planes is not None
    assert planes[0]["offset"] == 2.0


# =============================================================================
# inspect reporting (structural, no rendering)
# =============================================================================


def make_artifact(tmp_path: pathlib.Path, scene) -> pathlib.Path:
    visual = colight_inspect(scene)
    assert visual is not None
    target = tmp_path / "scene.colight"
    target.write_bytes(visual.to_bytes())
    return target


def test_inspect_reports_clip_planes(tmp_path):
    scene = scene3d.Scene(
        scene3d.Cuboid(
            np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32).flatten(),
            half_size=0.4,
        ),
        clip_planes=[{"normal": [0, 1, 0], "offset": 5.0}],
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    clip = payload["visual"]["clip_planes"]
    assert clip == [{"normal": [0.0, 1.0, 0.0], "offset": 5.0}]


def test_inspect_reports_state_key_for_state_ref(tmp_path):
    scene = scene3d.Scene(
        scene3d.Cuboid(
            np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32).flatten(),
            half_size=0.4,
        ),
        clip_planes=[{"normal": [0, 1, 0], "offset": Plot.js("$state.section_y")}],
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    clip = payload["visual"]["clip_planes"]
    assert clip[0]["normal"] == [0.0, 1.0, 0.0]
    assert clip[0]["state_key"] == "$state.section_y"
    assert "offset" not in clip[0]


def test_inspect_omits_clip_planes_when_absent(tmp_path):
    scene = scene3d.Scene(
        scene3d.Cuboid(
            np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32).flatten(),
            half_size=0.4,
        )
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    assert "clip_planes" not in payload["visual"]


def test_inspect_warns_when_section_excludes_scene(tmp_path):
    # A plane keeping y <= -1000 removes the whole scene (centers at y=0).
    scene = scene3d.Scene(
        scene3d.Cuboid(
            np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32).flatten(),
            half_size=0.4,
        ),
        clip_planes=[{"normal": [0, 1, 0], "offset": -1000.0}],
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    codes = {w.get("code") for w in payload.get("warnings", [])}
    assert "section-excludes-scene" in codes


def test_inspect_no_exclusion_warning_when_plane_keeps_scene(tmp_path):
    scene = scene3d.Scene(
        scene3d.Cuboid(
            np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32).flatten(),
            half_size=0.4,
        ),
        clip_planes=[{"normal": [0, 1, 0], "offset": 1000.0}],
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    codes = {w.get("code") for w in payload.get("warnings", [])}
    assert "section-excludes-scene" not in codes


# =============================================================================
# End-to-end (Chrome-gated): sweeping section hides + exposes interior geometry
# =============================================================================

# A 3x3x3 grid of cuboids centered on the origin (x,y,z in {-2, 0, 2}), one
# Cuboid component (27 instances). A $state offset drives a plane with normal
# +Y keeping y <= offset: sweeping the offset from +3 (keep all) down to -3
# (keep none) hides rows of the grid.
GRID_SCENE = """import colight.scene3d as S
import numpy as np

coords = [-2.0, 0.0, 2.0]
centers = np.array(
    [[x, y, z] for x in coords for y in coords for z in coords],
    dtype=np.float32,
)

(
    S.Scene(
        S.Cuboid(centers=centers, half_size=0.4, color=[0.2, 0.6, 0.9]),
        clip_planes=[{{"normal": [0, 1, 0], "offset": {offset}}}],
    )
    + {{"defaultCamera": {{"position": [10, 10, 10], "target": [0, 0, 0],
                          "up": [0, 1, 0], "fov": 45}}}}
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
        pytest.skip("Chrome not found for clip-plane tests")


def _grid_scene(tmp_path: pathlib.Path, offset: float, name: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(GRID_SCENE.format(offset=offset))
    return path


def _visible_count(scene_path: pathlib.Path) -> int:
    """pick-where over all 27 grid instances -> how many survive the section."""
    result = CliRunner().invoke(
        cli_main,
        [
            "pick-where",
            str(scene_path),
            "--component",
            "0",
            "--instances",
            "0-26",
            *SIZE_ARGS,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.output).get("visible_instances", 0)


def test_pick_where_visible_count_drops_as_plane_sweeps(tmp_path):
    _require_renderer()
    # pick-where counts pixel-visible (pickable) instances; some grid cuboids
    # occlude others, so the unsectioned baseline is <= 27. What matters is that
    # sweeping the +Y plane in monotonically HIDES instances (fewer pickable).
    keep_all = _visible_count(_grid_scene(tmp_path, 3.0, "all.py"))  # keep y<=3
    keep_mid = _visible_count(_grid_scene(tmp_path, 0.0, "mid.py"))  # cut y=2 row
    keep_low = _visible_count(_grid_scene(tmp_path, -1.0, "low.py"))  # only y=-2
    assert keep_all > 0, "expected some instances visible with an open section"
    assert (
        keep_mid < keep_all
    ), f"section should hide instances: {keep_mid} vs {keep_all}"
    assert keep_low < keep_mid, f"lower section hides more: {keep_low} vs {keep_mid}"
    # A section keeping only the bottom y-layer leaves at most its 9 cuboids.
    assert keep_low <= 9, f"only the bottom 9 cuboids should survive, got {keep_low}"


# Two cuboids on the camera's central ray (looking down -Z from +Z). The front
# cuboid at z=+2 occludes the back cuboid at z=-2 at the center pixel. A plane
# with normal +Z keeping z <= 0 cuts the FRONT cuboid away, exposing the back
# one — pick-at at the center must then report the interior instance.
INTERIOR_SCENE = """import colight.scene3d as S

(
    S.Scene(
        S.Cuboid(
            centers=[[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]],
            half_size=0.6,
            color=[[0.9, 0.2, 0.2], [0.2, 0.4, 0.9]],
        ),
        clip_planes=[{{"normal": [0, 0, 1], "offset": {offset}}}],
    )
    + {{"defaultCamera": {{"position": [0, 0, 12], "target": [0, 0, 0],
                          "up": [0, 1, 0], "fov": 45}}}}
)
"""


def _interior_scene(tmp_path: pathlib.Path, offset: float, name: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(INTERIOR_SCENE.format(offset=offset))
    return path


def _pick_center_instance(scene_path: pathlib.Path):
    """pick-at the center pixel; return the top hit's instance index or None."""
    result = CliRunner().invoke(
        cli_main,
        ["pick-at", str(scene_path), "250,250", *SIZE_ARGS, "--json"],
    )
    payload = json.loads(result.output)
    hits = payload.get("hits") or []
    return hits[0]["instance"] if hits else None


def test_pick_at_exposes_interior_instance_behind_section(tmp_path):
    _require_renderer()
    # offset 3: nothing clipped -> the FRONT cuboid (instance 0) is picked.
    front = _pick_center_instance(_interior_scene(tmp_path, 3.0, "front.py"))
    assert front == 0, f"expected front instance 0 with no section, got {front}"
    # offset 0: the plane cuts z>0, removing the front cuboid; pick-at must now
    # reach the interior BACK cuboid (instance 1) newly exposed behind the cut.
    behind = _pick_center_instance(_interior_scene(tmp_path, 0.0, "behind.py"))
    assert (
        behind == 1
    ), f"expected exposed interior instance 1 behind the section, got {behind}"


def test_screenshot_check_deterministic_with_active_plane(tmp_path):
    _require_renderer()
    scene_path = _grid_scene(tmp_path, 0.0, "shot_scene.py")
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
    # The active section is reported so an agent knows the view is clipped.
    reported = payload.get("clip_planes")
    assert reported and math.isclose(reported[0]["offset"], 0.0, abs_tol=1e-6)
