"""Regression tests for prop-key naming at the scene3d Python->JS boundary.

Data props (centers, half_sizes, fill_mode, ...) must cross the boundary in
snake_case — the JS coercion layer consumes snake_case keys and silently
ignores unknown ones, so a blanket snake->camel conversion made every
multi-word data prop render at its default (e.g. half_size was dropped and
ellipsoids always rendered at the 0.5 default).

Framework props (callbacks, hover/outline styling, caching keys) are the
exception: the JS side consumes those as camelCase, and only those are
renamed.

The e2e test drives the real render + GPU pick path through headless Chrome
(skipped when Chrome or the JS bundle is missing, same mechanism as the
other visual tests).
"""

import json
import pathlib

import pytest
from click.testing import CliRunner

import colight.env as env
from colight.chrome_devtools import find_chrome
from colight.scene3d import Cuboid, Ellipsoid, Group, deco
from colight_cli import main as cli_main

# =============================================================================
# Serialization: which keys does Python actually send?
# =============================================================================


def props_of(component):
    """Extract the JS-bound props dict from a SceneComponent."""
    return component.to_js_call().args[0]


def test_data_props_stay_snake_case():
    props = props_of(
        Ellipsoid(
            centers=[[0, 0, 0]],
            half_sizes=[[0.1, 0.2, 0.3]],
            quaternions=[[1, 0, 0, 0]],
            fill_mode="MajorWireframe",
        )
    )
    assert "half_sizes" in props
    assert "quaternions" in props
    assert "fill_mode" in props
    # The camelCased spellings must NOT be produced — JS would ignore them.
    assert "halfSizes" not in props
    assert "fillMode" not in props


def test_singular_data_props_stay_snake_case():
    props = props_of(Cuboid(center=[0, 0, 0], half_size=0.25))
    assert props["half_size"] == 0.25
    assert "halfSize" not in props


def test_framework_props_are_renamed_to_camel_case():
    props = props_of(
        Ellipsoid(
            center=[0, 0, 0],
            half_size=0.1,
            hover_props={"outline": True, "outline_color": [1, 0, 0]},
            picking_scale=2.0,
            on_hover="cb",
            decorations=[deco(0, outline_color=[0, 1, 0], outline_width=3)],
        )
    )
    assert props["pickingScale"] == 2.0
    assert props["onHover"] == "cb"
    assert props["hoverProps"]["outlineColor"] == [1, 0, 0]
    assert "hover_props" not in props
    assert "picking_scale" not in props
    decoration = props["decorations"][0]
    assert decoration["outlineColor"] == [0, 1, 0]
    assert decoration["outlineWidth"] == 3
    assert "outline_color" not in decoration


def test_group_props_are_renamed_but_child_data_props_kept():
    props = props_of(
        Group(
            children=[Ellipsoid(center=[0, 0, 0], half_size=0.1)],
            child_defaults={"half_size": 0.5, "hover_props": {"outline": True}},
            hover_props={"outline_width": 4},
            on_click="cb",
        )
    )
    assert props["onClick"] == "cb"
    assert props["hoverProps"]["outlineWidth"] == 4
    assert props["childDefaults"]["half_size"] == 0.5
    assert props["childDefaults"]["hoverProps"] == {"outline": True}
    child = props["children"][0]
    assert child["type"] == "Ellipsoid"
    assert child["half_size"] == 0.1


# =============================================================================
# End-to-end: half_size must actually take effect in the rendered scene
# =============================================================================

# Camera at z=6, fov 45, 400x400 => focal length ~482.8 px. The red
# ellipsoid (half_size 0.1) at x=-1.2 projects to page x ~103 with an
# ~8 px radius; the blue one (half_size 0.8) at x=+1.2 projects to
# x ~297 with a ~64 px radius. With the historical bug both rendered at
# the 0.5 default (~40 px radius).
E2E_SCENE = """import colight.scene3d as S

(
    S.Ellipsoid(center=[-1.2, 0, 0], half_size=0.1, color=[1, 0, 0])
    + S.Ellipsoid(center=[1.2, 0, 0], half_size=0.8, color=[0, 0, 1])
    + {"defaultCamera": {"position": [0, 0, 6], "target": [0, 0, 0],
                         "up": [0, 1, 0], "fov": 45}}
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
        pytest.skip("Chrome not found for pick tests")


def _pick_at(scene_path: pathlib.Path, x: int, y: int) -> dict:
    """Run pick-at; exit code 0 means hit, 1 means background-only miss."""
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["pick-at", str(scene_path), f"{x},{y}", *SIZE_ARGS, "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code in (0, 1), result.output
    return json.loads(result.output)


def test_half_size_takes_effect_end_to_end(tmp_path):
    _require_renderer()
    scene_path = tmp_path / "half_size_scene.py"
    scene_path.write_text(E2E_SCENE)

    # Small red ellipsoid: hit at its center, with the half_size we sent.
    payload = _pick_at(scene_path, 103, 200)
    hits = payload["hits"]
    assert hits, payload
    assert hits[0]["type"] == "Ellipsoid"
    assert hits[0]["values"]["half_size"] == pytest.approx([0.1, 0.1, 0.1])

    # 30 px right of the small ellipsoid's center: outside its ~8 px
    # radius, but well inside the ~40 px radius it wrongly rendered with
    # when half_size was dropped at the boundary.
    payload = _pick_at(scene_path, 133, 200)
    assert payload["background_share"] == pytest.approx(1.0), payload

    # Large blue ellipsoid: different half_size, also honored.
    payload = _pick_at(scene_path, 297, 200)
    hits = payload["hits"]
    assert hits, payload
    assert hits[0]["values"]["half_size"] == pytest.approx([0.8, 0.8, 0.8])
