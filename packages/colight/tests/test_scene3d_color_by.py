"""Tests for first-class colormaps on scene3d primitives (color_by) and the
legend pipeline: Python color computation, metadata serialization, legends in
``colight inspect``/``screenshot --json``, the rendered legend overlay, and
value->color verification through the GPU pick path.

Pure tests run everywhere; end-to-end tests drive the real CLI through
headless Chrome (skipped when Chrome or the JS bundle is missing, same
mechanism as the other visual tests).
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
from colight.colormaps import apply_colormap
from colight.inspect import inspect as colight_inspect
from colight_cli import main as cli_main

# =============================================================================
# Serialization: colors computed + spec metadata attached
# =============================================================================

VALUES = [0.0, 1.0, 2.0, 3.0]
SPEC = {"values": VALUES, "cmap": "viridis", "domain": (0, 3), "label": "Cu %"}
EXPECTED_COLORS = apply_colormap(VALUES, "viridis", domain=(0, 3))


def props_of(component):
    return component.to_js_call().args[0]


def assert_color_by_props(props, count=4):
    colors = props["colors"]
    assert colors.dtype == np.float32
    np.testing.assert_allclose(
        colors.reshape(-1, 3), EXPECTED_COLORS[:count], atol=1e-6
    )
    meta = props["color_by"]
    assert meta["cmap"] == "viridis"
    assert meta["domain"] == [0.0, 3.0]
    assert meta["label"] == "Cu %"
    assert meta["categorical"] is False
    assert len(meta["stops"]) == 17
    # values must NOT travel in the metadata (colors already carry them)
    assert "values" not in meta


centers4 = np.array([[i, 0, 0] for i in range(4)], dtype=np.float32)


def test_point_cloud_color_by():
    assert_color_by_props(props_of(scene3d.PointCloud(centers4, color_by=SPEC)))


def test_ellipsoid_color_by():
    assert_color_by_props(props_of(scene3d.Ellipsoid(centers4, color_by=SPEC)))


def test_cuboid_color_by():
    assert_color_by_props(props_of(scene3d.Cuboid(centers4, color_by=SPEC)))


def test_line_segments_color_by():
    assert_color_by_props(
        props_of(
            scene3d.LineSegments(starts=centers4, ends=centers4 + 1, color_by=SPEC)
        )
    )


def test_line_beams_color_by():
    points = np.array(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 1], [1, 1, 0, 1]], dtype=np.float32
    )
    props = props_of(scene3d.LineBeams(points=points, color_by=SPEC))
    assert_color_by_props(props)


def test_color_by_conflicts_with_colors():
    with pytest.raises(ValueError, match="not both"):
        scene3d.PointCloud(centers4, colors=np.zeros((4, 3)), color_by=SPEC)
    with pytest.raises(ValueError, match="not both"):
        scene3d.Cuboid(centers4, color=[1, 0, 0], color_by=SPEC)


def test_color_by_legend_flag_and_position():
    props = props_of(scene3d.PointCloud(centers4, color_by={**SPEC, "legend": False}))
    assert props["color_by"]["legend"] is False

    props = props_of(
        scene3d.PointCloud(centers4, color_by={**SPEC, "legend": "bottom-left"})
    )
    assert props["color_by"]["position"] == "bottom-left"

    with pytest.raises(ValueError, match="legend position"):
        scene3d.PointCloud(centers4, color_by={**SPEC, "legend": "middle"})


def test_categorical_color_by():
    props = props_of(
        scene3d.PointCloud(
            centers4,
            color_by={
                "values": [0, 1, 2, 0],
                "cmap": "tab10",
                "categories": ["ore", "waste", "cover"],
                "label": "rock type",
            },
        )
    )
    meta = props["color_by"]
    assert meta["categorical"] is True
    assert meta["categories"] == ["ore", "waste", "cover"]
    assert len(meta["colors"]) == 3
    assert "domain" not in meta


def test_standalone_legend_layout_item():
    legend = scene3d.Legend(cmap="viridis", domain=(0, 2.5), label="Cu %")
    ref, props = legend.for_json()
    assert ref.path == "scene3d.Legend"
    assert props["spec"]["domain"] == [0.0, 2.5]
    assert props["spec"]["label"] == "Cu %"


# =============================================================================
# colight inspect: legends reported without rendering
# =============================================================================


def make_artifact(tmp_path: pathlib.Path, scene) -> pathlib.Path:
    visual = colight_inspect(scene)
    assert visual is not None
    target = tmp_path / "scene.colight"
    target.write_bytes(visual.to_bytes())
    return target


def test_inspect_reports_legends(tmp_path):
    scene = scene3d.Scene(
        scene3d.Cuboid(centers4, color_by=SPEC, half_size=0.5),
        scene3d.PointCloud(centers4, color=[1, 0, 0]),
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    legends = payload["visual"]["legends"]
    assert legends == [
        {
            "component": "scene3d.Cuboid",
            "label": "Cu %",
            "cmap": "viridis",
            "domain": [0.0, 3.0],
            "categorical": False,
        }
    ]
    # The legend metadata (stops etc.) must not leak into the array records.
    assert not any("color_by" in a["path"] for a in payload["visual"]["arrays"])


def test_inspect_omits_legends_when_absent(tmp_path):
    target = make_artifact(
        tmp_path, scene3d.Scene(scene3d.PointCloud(centers4, color=[1, 0, 0]))
    )
    payload = inspect_tools.inspect_target(target)
    assert "legends" not in payload["visual"]


def test_inspect_cli_human_output_mentions_legend(tmp_path):
    target = make_artifact(
        tmp_path, scene3d.Scene(scene3d.Cuboid(centers4, color_by=SPEC))
    )
    result = CliRunner().invoke(cli_main, ["inspect", str(target)])
    assert result.exit_code == 0
    assert "legend viridis" in result.output
    assert "'Cu %'" in result.output


# =============================================================================
# End-to-end: rendered legend + value->color verification (Chrome-gated)
# =============================================================================

# Camera on +z at 400x400 / fov 45 (focal ~482.8 px). Cuboids (half_size
# 0.5, ~20 px on screen) sit at x = -3, -1, 1, 3 -> page x ~ 79, 160, 240,
# 321, y ~ 200. Values 0..3 over domain (0, 3) via viridis.
E2E_SCENE = """import colight.scene3d as S

(
    S.Cuboid(
        centers=[[-3.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]],
        half_size=0.5,
        color_by={
            "values": [0.0, 1.0, 2.0, 3.0],
            "cmap": "viridis",
            "domain": (0, 3),
            "label": "Cu %",
        },
    )
    + {"defaultCamera": {"position": [0, 0, 12], "target": [0, 0, 0],
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
        pytest.skip("Chrome not found for color_by tests")


@pytest.fixture()
def e2e_scene(tmp_path):
    _require_renderer()
    scene_path = tmp_path / "color_by_scene.py"
    scene_path.write_text(E2E_SCENE)
    return scene_path


def test_screenshot_json_reports_legends_and_stays_deterministic(e2e_scene, tmp_path):
    out = tmp_path / "shot.png"
    result = CliRunner().invoke(
        cli_main,
        [
            "screenshot",
            str(e2e_scene),
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
    assert payload["legends"] == [
        {
            "cmap": "viridis",
            "categorical": False,
            "label": "Cu %",
            "domain": [0, 3],
            "component": 0,
            "type": "Cuboid",
        }
    ]


def test_legend_appears_in_screenshot_pixels(e2e_scene, tmp_path):
    from PIL import Image

    out = tmp_path / "legend_shot.png"
    result = CliRunner().invoke(
        cli_main,
        ["screenshot", str(e2e_scene), "--out", str(out), *SIZE_ARGS],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    image = np.asarray(Image.open(out).convert("RGB"), dtype=np.int64)
    # Legend docks top-right; cuboids sit at y~200, so the region is
    # exclusively background + legend.
    region = image[:140, 200:]

    background = np.array([255, 255, 255])
    non_background = np.abs(region - background).sum(axis=-1) > 30
    assert non_background.sum() > 300, "legend region is blank"

    def present(rgb01, tol=40):
        target = np.array([round(c * 255) for c in rgb01])
        return bool((np.abs(region - target).sum(axis=-1) < tol).any())

    # Both viridis gradient endpoints must be visible in the legend bar.
    assert present([0.267004, 0.004874, 0.329415]), "dark gradient end missing"
    assert present([0.993248, 0.906157, 0.143936]), "bright gradient end missing"


def test_pick_at_dereferences_colormapped_colors(e2e_scene):
    """Strong assertion: the rendered per-instance colors are exactly the
    colormap of the values we passed, verified through the pick path."""
    expected = apply_colormap([0.0, 1.0, 2.0, 3.0], "viridis", domain=(0, 3))
    runner = CliRunner()
    for x, instance in [(79, 0), (160, 1), (240, 2), (321, 3)]:
        result = runner.invoke(
            cli_main,
            ["pick-at", str(e2e_scene), f"{x},200", *SIZE_ARGS, "--json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        hits = payload["hits"]
        assert hits, payload
        hit = hits[0]
        assert hit["type"] == "Cuboid"
        assert hit["instance"] == instance
        assert hit["values"]["color"] == pytest.approx(
            list(expected[instance]), abs=1e-4
        )
