"""Tests for named selections on scene3d — the second consumer of the shared
per-instance mask abstraction.

Covers the Python Selection/select/toggle_selection API, that selections are
resident $state (so they persist into .colight artifacts and inspect reports
them), and the addressability the design requires: pick-at membership,
pick-where --selection NAME, and screenshot --frame NAME.

Pure tests run everywhere; end-to-end tests drive the real CLI through headless
Chrome (skipped when Chrome or the JS bundle is missing).
"""

import json
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner

import colight.env as env
import colight.scene3d as scene3d
from colight.chrome_devtools import find_chrome
from colight.cli_tools import inspect_tools
from colight.inspect import inspect as colight_inspect
from colight_cli import main as cli_main

# =============================================================================
# API: Selection specs + state seeding
# =============================================================================


def test_selection_explicit_instances():
    sel = scene3d.Selection("sel-hi", 2, instances=[3, 1, 5])
    assert sel == {"sel-hi": {"component": 2, "source": {"instances": [3, 1, 5]}}}


def test_selection_predicate_values_coerced():
    sel = scene3d.Selection("grade", 0, values=[0.1, 0.9, 0.5], min=0.4, max=1.0)
    src = sel["grade"]["source"]
    assert src["values"].dtype == np.float32
    np.testing.assert_allclose(src["values"], [0.1, 0.9, 0.5], atol=1e-6)
    assert src["min"] == 0.4
    assert src["max"] == 1.0


def test_selection_values_ref():
    sel = scene3d.Selection("hi", 1, values_ref="CU_pct", min=0.5)
    assert sel["hi"]["source"] == {"values_ref": "CU_pct", "min": 0.5}


def test_selection_style_renamed_to_camelcase():
    sel = scene3d.Selection(
        "hi", 0, instances=[1], style={"color": [1, 0, 0], "outline_width": 4}
    )
    # outline_width -> outlineWidth at the Python->JS boundary.
    assert sel["hi"]["style"] == {"color": [1, 0, 0], "outlineWidth": 4}


def test_selection_requires_a_source():
    with pytest.raises(ValueError, match="requires either 'instances'"):
        scene3d.Selection("hi", 0)


def test_select_seeds_state_selections(tmp_path):
    # select() seeds $state.selections; verify it lands in the serialized
    # visual's state (the artifact path is the source of truth).
    centers = np.array([[0, 0, 0]], dtype=np.float32)
    scene = scene3d.Scene(scene3d.Cuboid(centers=centers)) | scene3d.select(
        scene3d.Selection("a", 0, instances=[1]),
        scene3d.Selection("b", 1, instances=[2]),
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    assert "selections" in payload["visual"]["state_keys"]
    names = {s["name"] for s in payload["visual"].get("selections", [])}
    assert names == {"a", "b"}


def test_scene_passes_live_selections_ref():
    centers = np.array([[0, 0, 0]], dtype=np.float32)
    scene = scene3d.Scene(scene3d.Cuboid(centers=centers))
    _ref, props = scene.for_json()
    # The Scene always passes a live $state.selections reference so state-
    # resident selections resolve on every render.
    assert "selections" in props


# =============================================================================
# Artifacts + inspect: selections are just state, so they persist
# =============================================================================


def make_artifact(tmp_path: pathlib.Path, scene) -> pathlib.Path:
    visual = colight_inspect(scene)
    assert visual is not None
    target = tmp_path / "scene.colight"
    target.write_bytes(visual.to_bytes())
    return target


centers4 = np.array([[i, 0, 0] for i in range(4)], dtype=np.float32)


def test_saved_artifact_reports_its_selections(tmp_path):
    scene = scene3d.Scene(
        scene3d.Cuboid(centers=centers4, half_size=0.4)
    ) | scene3d.select(
        scene3d.Selection("sel-hi", 0, instances=[1, 3]),
        scene3d.Selection("grade-hi", 0, values=[0.1, 0.9, 0.5, 0.2], min=0.4),
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    selections = payload["visual"]["selections"]
    assert {"name": "sel-hi", "component": 0, "count": 2} in selections
    predicate = next(s for s in selections if s["name"] == "grade-hi")
    assert predicate["component"] == 0
    assert predicate["predicate"] == {"min": 0.4}


def test_inspect_omits_selections_when_absent(tmp_path):
    target = make_artifact(
        tmp_path, scene3d.Scene(scene3d.Cuboid(centers=centers4, half_size=0.4))
    )
    payload = inspect_tools.inspect_target(target)
    assert "selections" not in payload["visual"]


# =============================================================================
# End-to-end addressability (Chrome-gated)
# =============================================================================

SCENE = """import colight.scene3d as S

(
    S.Scene(
        S.Cuboid(
            centers=[[-3.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]],
            half_size=0.5,
            color=[0.3, 0.3, 0.6],
        )
    )
    | S.select(S.Selection("hi", 0, instances=[1, 3]))
    | {"defaultCamera": {"position": [0, 0, 12], "target": [0, 0, 0],
                         "up": [0, 1, 0], "fov": 45}}
)
"""

SIZE_ARGS = ["--width", "400", "--height", "400"]

# Cuboid instance 1 (center [-1,0,0]) projects near page x ~ 160, y ~ 180.
HIT_X, HIT_Y = 160, 180


def _require_renderer() -> None:
    widget_path = env.WIDGET_PATH
    if not (isinstance(widget_path, pathlib.Path) and widget_path.exists()):
        pytest.skip("colight JS bundle not built (js-dist missing)")
    try:
        chrome_path = find_chrome()
    except FileNotFoundError:
        chrome_path = None
    if not chrome_path:
        pytest.skip("Chrome not found for selection tests")


@pytest.fixture()
def scene_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "sel_scene.py"
    path.write_text(SCENE)
    return path


def test_pick_at_reports_selection_membership(scene_file):
    _require_renderer()
    result = CliRunner().invoke(
        cli_main,
        [
            "pick-at",
            str(scene_file),
            f"{HIT_X},{HIT_Y}",
            *SIZE_ARGS,
            "--json",
            "--no-daemon",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    hits = payload["hits"]
    assert hits, "expected a hit on instance 1"
    assert hits[0]["component"] == 0
    assert hits[0]["instance"] == 1
    assert hits[0]["selections"] == ["hi"]


def test_pick_where_selection_resolves_name(scene_file):
    _require_renderer()
    result = CliRunner().invoke(
        cli_main,
        [
            "pick-where",
            str(scene_file),
            "--selection",
            "hi",
            *SIZE_ARGS,
            "--json",
            "--no-daemon",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # "hi" resolves to component 0, instances [1, 3]; both are visible.
    assert payload["component"] == 0
    assert payload["instances"] == [[1, 1], [3, 3]]
    assert payload["visible_instances"] == 2


def test_pick_where_rejects_both_selectors(scene_file):
    _require_renderer()
    result = CliRunner().invoke(
        cli_main,
        [
            "pick-where",
            str(scene_file),
            "--component",
            "0",
            "--selection",
            "hi",
            *SIZE_ARGS,
            "--no-daemon",
        ],
    )
    assert result.exit_code == 2
    assert "exactly one of --component or --selection" in result.output


def test_screenshot_frame_resolves_selection_name(scene_file, tmp_path):
    _require_renderer()
    out = tmp_path / "framed.png"
    result = CliRunner().invoke(
        cli_main,
        [
            "screenshot",
            str(scene_file),
            "--out",
            str(out),
            *SIZE_ARGS,
            "--frame",
            "hi",
            "--json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["frame"]["selection"] == "hi"
    assert payload["frame"]["component"] == 0
    # The selection reporting marker is present too.
    assert any(s["name"] == "hi" for s in payload.get("selections", []))
