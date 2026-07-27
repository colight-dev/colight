"""Tests for switchable named color channels on scene3d primitives.

Channels ship their raw ``values`` once plus a compact colorizer (a 256-entry
RGB LUT for continuous, or a category table for categorical); JS recolors the
active channel client-side so an artifact switches which attribute drives the
colors without re-exporting. Pure tests run everywhere; end-to-end tests drive
the real CLI through headless Chrome (skipped when Chrome or the JS bundle is
missing).
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

centers3 = np.array([[i, 0, 0] for i in range(3)], dtype=np.float32)

CHANNELS = {
    "CU_pct": {
        "values": [0.0, 1.0, 2.0],
        "cmap": "viridis",
        "domain": (0, 2),
        "label": "Cu %",
    },
    "Lithology": {
        "values": [0, 1, 2],
        "categories": [
            {"value": 0, "label": "not logged", "color": [0.5, 0.5, 0.5]},
            {"value": 1, "label": "Dacite"},
            {"value": 2, "label": "Andesite"},
        ],
    },
}


def props_of(component):
    return component.to_js_call().args[0]


# =============================================================================
# Serialization
# =============================================================================


def test_channels_bake_default_and_ship_colorizers():
    props = props_of(scene3d.Cuboid(centers3, color_channels=CHANNELS))
    assert "color_channels" in props
    assert "colors" in props  # first channel baked for the initial render
    cc = props["color_channels"]
    assert set(cc) == {"CU_pct", "Lithology"}

    cu = cc["CU_pct"]["colorizer"]
    assert cu["kind"] == "continuous"
    assert len(cu["lut"]) == 256
    assert cu["domain"] == [0.0, 2.0]

    litho = cc["Lithology"]["colorizer"]
    assert litho["kind"] == "categorical"
    assert [c["label"] for c in litho["categories"]] == [
        "not logged",
        "Dacite",
        "Andesite",
    ]
    # Default active channel = first declared -> its legend is baked as color_by.
    assert props["color_by"]["label"] == "Cu %"
    assert props["active_channel"] == "CU_pct"


def test_literal_active_channel_bakes_that_channel():
    props = props_of(
        scene3d.Cuboid(centers3, color_channels=CHANNELS, active_channel="Lithology")
    )
    assert props["active_channel"] == "Lithology"
    assert props["color_by"]["categorical"] is True
    colors = props["colors"].reshape(-1, 3)
    np.testing.assert_allclose(colors[0], [0.5, 0.5, 0.5], atol=1e-6)


def test_state_ref_active_channel_passes_through():
    props = props_of(
        scene3d.Cuboid(
            centers3,
            color_channels=CHANNELS,
            active_channel=Plot.js("$state.color_channel"),
        )
    )
    # A $state ref is resolved by JS; Python bakes the first channel meanwhile.
    from colight.layout import is_js_expr

    assert is_js_expr(props["active_channel"])
    assert props["active_channel"] is not None
    assert props["color_by"]["label"] == "Cu %"


def test_unknown_literal_active_channel_raises():
    with pytest.raises(ValueError, match="active_channel"):
        scene3d.Cuboid(centers3, color_channels=CHANNELS, active_channel="nope")


def test_channels_mutually_exclusive_with_color_by():
    with pytest.raises(ValueError, match="mutually exclusive"):
        scene3d.Cuboid(
            centers3,
            color_by={"values": [0, 1, 2], "cmap": "viridis"},
            color_channels=CHANNELS,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        scene3d.Cuboid(centers3, color=[1, 0, 0], color_channels=CHANNELS)


def test_channels_work_on_line_segments():
    props = props_of(
        scene3d.LineSegments(
            starts=centers3, ends=centers3 + 1, color_channels=CHANNELS
        )
    )
    # Per-segment values expand exactly like color_by (1:1 with instances).
    assert props["colors"].reshape(-1, 3).shape[0] == 3
    assert set(props["color_channels"]) == {"CU_pct", "Lithology"}


# =============================================================================
# Payload sanity: channels artifact ~ values arrays, not N x RGB
# =============================================================================


def test_payload_size_scales_with_values_not_colors(tmp_path):
    n = 5000
    values = np.random.default_rng(0).random(n).astype(np.float32)
    litho = np.random.default_rng(1).integers(0, 3, n)
    scene = scene3d.Scene(
        scene3d.PointCloud(
            np.random.default_rng(2).random((n, 3)).astype(np.float32),
            color_channels={
                "CU_pct": {"values": values, "cmap": "viridis", "domain": (0, 1)},
                "AG_ppm": {"values": values * 40, "cmap": "magma"},
                "Litho": {
                    "values": litho,
                    "categories": [
                        {"value": 0, "label": "a"},
                        {"value": 1, "label": "b"},
                        {"value": 2, "label": "c"},
                    ],
                },
            },
            active_channel=Plot.js("$state.color_channel"),
        )
    )
    visual = colight_inspect(scene)
    assert visual is not None
    size = len(visual.to_bytes())
    # 3 channels x N float32 values (~60KB) + one baked N x 3 colors buffer +
    # 3 x N centers. A per-channel N x RGB baking would add ~2 more full color
    # buffers; assert we stayed well under that ceiling.
    baked_colors_bytes = n * 3 * 4
    values_bytes = 3 * n * 4
    assert size < baked_colors_bytes * 4 + values_bytes + 200_000


# =============================================================================
# inspect: channel roster reported without rendering
# =============================================================================


def test_inspect_reports_channel_roster(tmp_path):
    scene = scene3d.Scene(
        scene3d.Cuboid(
            centers3,
            half_size=0.5,
            color_channels=CHANNELS,
            active_channel="Lithology",
        )
    )
    visual = colight_inspect(scene)
    assert visual is not None
    target = tmp_path / "scene.colight"
    target.write_bytes(visual.to_bytes())
    payload = inspect_tools.inspect_target(target)
    channels = payload["visual"]["color_channels"]
    assert len(channels) == 1
    entry = channels[0]
    assert entry["active"] == "Lithology"
    roster = {c["name"]: c["kind"] for c in entry["channels"]}
    assert roster == {"CU_pct": "continuous", "Lithology": "categorical"}
    # The per-channel value/LUT buffers must NOT leak into the array records.
    assert not any("colorizer" in a["path"] for a in payload["visual"]["arrays"])


# =============================================================================
# End-to-end: channel switching + pick-at channels row (Chrome-gated)
# =============================================================================

E2E_SCENE = """import colight.scene3d as S
import colight.plot as Plot

(
    S.Cuboid(
        centers=[[-3.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]],
        half_size=0.5,
        color_channels={
            "CU_pct": {
                "values": [0.0, 1.0, 2.0, 3.0],
                "cmap": "viridis",
                "domain": (0, 3),
                "label": "Cu %",
            },
            "Lithology": {
                "values": [0, 1, 2, 1],
                "categories": [
                    {"value": 0, "label": "not logged", "color": [0.5, 0.5, 0.5]},
                    {"value": 1, "label": "Dacite"},
                    {"value": 2, "label": "Andesite"},
                ],
                "label": "Lithology",
            },
        },
        active_channel=Plot.js("$state.color_channel"),
    )
    + {"defaultCamera": {"position": [0, 0, 12], "target": [0, 0, 0],
                         "up": [0, 1, 0], "fov": 45}}
) | Plot.initialState({"color_channel": CHANNEL})
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
        pytest.skip("Chrome not found for color_channels tests")


def _scene_for_channel(tmp_path, channel: str) -> pathlib.Path:
    scene_path = tmp_path / f"channel_{channel}.py"
    scene_path.write_text(E2E_SCENE.replace("CHANNEL", json.dumps(channel)))
    return scene_path


def test_switching_active_channel_changes_pixels(tmp_path):
    """Two screenshots at different $state.color_channel differ."""
    from PIL import Image

    _require_renderer()
    runner = CliRunner()
    images = {}
    for channel in ("CU_pct", "Lithology"):
        scene_path = _scene_for_channel(tmp_path, channel)
        out = tmp_path / f"{channel}.png"
        result = runner.invoke(
            cli_main,
            ["screenshot", str(scene_path), "--out", str(out), *SIZE_ARGS],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        images[channel] = np.asarray(Image.open(out).convert("RGB"), dtype=np.int64)

    # The instance region (cuboids at y~200) must differ between channels:
    # viridis(0..3) vs the lithology swatches are different colors.
    a = images["CU_pct"][180:220, :]
    b = images["Lithology"][180:220, :]
    assert np.abs(a - b).sum() > 5000, "channel switch did not change pixels"


def test_screenshot_json_reports_active_channel(tmp_path):
    _require_renderer()
    scene_path = _scene_for_channel(tmp_path, "Lithology")
    out = tmp_path / "shot.png"
    result = CliRunner().invoke(
        cli_main,
        ["screenshot", str(scene_path), "--out", str(out), *SIZE_ARGS, "--json"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # The active channel's legend is reported (categorical Lithology).
    legends = payload["legends"]
    assert any(leg.get("categorical") for leg in legends)


def test_pick_at_reports_full_channels_row(tmp_path):
    """Headline: pick-at on a channel component reports ALL channel values for
    the picked instance (categorical as labels)."""
    _require_renderer()
    scene_path = _scene_for_channel(tmp_path, "CU_pct")
    runner = CliRunner()
    # Cuboids at x = -3, -1, 1, 3 -> page x ~ 79, 160, 240, 321.
    expected = {
        0: {"CU_pct": 0.0, "Lithology": "not logged"},
        1: {"CU_pct": 1.0, "Lithology": "Dacite"},
        2: {"CU_pct": 2.0, "Lithology": "Andesite"},
        3: {"CU_pct": 3.0, "Lithology": "Dacite"},
    }
    for x, instance in [(79, 0), (160, 1), (240, 2), (321, 3)]:
        result = runner.invoke(
            cli_main,
            ["pick-at", str(scene_path), f"{x},200", *SIZE_ARGS, "--json"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        hits = payload["hits"]
        assert hits, payload
        hit = hits[0]
        assert hit["instance"] == instance
        channels = hit["values"]["channels"]
        assert channels["Lithology"] == expected[instance]["Lithology"]
        assert channels["CU_pct"] == pytest.approx(expected[instance]["CU_pct"])
        assert hit["values"]["activeChannel"] == "CU_pct"


def _channel_scene_object(channel: str):
    import colight.plot as P

    scene = (
        scene3d.Cuboid(
            centers=[[-3.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]],
            half_size=0.5,
            color_channels={
                "CU_pct": {
                    "values": [0.0, 1.0, 2.0, 3.0],
                    "cmap": "viridis",
                    "domain": (0, 3),
                    "label": "Cu %",
                },
                "Lithology": {
                    "values": [0, 1, 2, 1],
                    "categories": [
                        {"value": 0, "label": "not logged", "color": [0.5, 0.5, 0.5]},
                        {"value": 1, "label": "Dacite"},
                        {"value": 2, "label": "Andesite"},
                    ],
                    "label": "Lithology",
                },
            },
            active_channel=P.js("$state.color_channel"),
        )
        + {
            "defaultCamera": {
                "position": [0, 0, 12],
                "target": [0, 0, 0],
                "up": [0, 1, 0],
                "fov": 45,
            }
        }
    ) | P.initialState({"color_channel": channel})
    return scene


def test_artifact_round_trip_keeps_switching(tmp_path):
    """Serialize a .colight artifact and re-screenshot at a different channel:
    the switch still works because it is entirely client-side (values +
    colorizer travel in the artifact; JS recolors on load)."""
    from PIL import Image

    _require_renderer()
    runner = CliRunner()
    imgs = {}
    for channel in ("CU_pct", "Lithology"):
        visual = colight_inspect(_channel_scene_object(channel))
        assert visual is not None
        artifact = tmp_path / f"{channel}.colight"
        artifact.write_bytes(visual.to_bytes())
        out = tmp_path / f"{channel}_rt.png"
        shot = runner.invoke(
            cli_main,
            ["screenshot", str(artifact), "--out", str(out), *SIZE_ARGS],
            catch_exceptions=False,
        )
        assert shot.exit_code == 0, shot.output
        imgs[channel] = np.asarray(Image.open(out).convert("RGB"), dtype=np.int64)
    a = imgs["CU_pct"][180:220, :]
    b = imgs["Lithology"][180:220, :]
    assert np.abs(a - b).sum() > 5000, "round-tripped artifact lost channel switch"
