"""Tests for `Plot.channel`: declared samples resampled client-side.

A channel ships its sample rows once and names the ``$state`` key that indexes
them; the browser resamples on every parameter change. Pure tests run
everywhere; end-to-end tests drive the real CLI through headless Chrome
(skipped when Chrome or the JS bundle is missing).
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


def bend_quaternion(angle_deg: float) -> list:
    """Rotation about +X by ``angle_deg``, as an [x, y, z, w] quaternion."""
    half = math.radians(angle_deg) / 2.0
    return [math.sin(half), 0.0, 0.0, math.cos(half)]


ANGLES = np.linspace(-80.0, 80.0, 9)
QUATS = np.array([bend_quaternion(a) for a in ANGLES], dtype=np.float32)


# =============================================================================
# Validation
# =============================================================================


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="rule must be one of"):
        Plot.channel("t", values=[0.0, 1.0], rule="cubic")


@pytest.mark.parametrize(
    "parameter", ["", "not an identifier", "$state.t", "t;alert(1)", "1t", 42]
)
def test_bad_parameter_name_raises(parameter):
    with pytest.raises(ValueError, match="parameter must be an identifier"):
        Plot.channel(parameter, values=[0.0, 1.0])


def test_non_increasing_at_raises():
    with pytest.raises(ValueError, match="strictly increasing"):
        Plot.channel("t", values=[0.0, 1.0, 2.0], at=[0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        Plot.channel("t", values=[0.0, 1.0, 2.0], at=[0.0, 1.0, 1.0])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="3 coordinates but values has 2 rows"):
        Plot.channel("t", values=[0.0, 1.0], at=[0.0, 1.0, 2.0])


def test_qlerp_requires_quaternion_rows():
    with pytest.raises(ValueError, match=r"qlerp' requires values of shape \(N, 4\)"):
        Plot.channel("t", values=np.zeros((3, 3)), rule="qlerp")
    with pytest.raises(ValueError, match=r"qlerp' requires values of shape \(N, 4\)"):
        Plot.channel("t", values=[0.0, 1.0, 2.0], rule="qlerp")
    # (N, 4) is accepted.
    Plot.channel("t", values=QUATS, at=ANGLES, rule="qlerp")


def test_empty_and_scalar_values_raise():
    with pytest.raises(ValueError, match="at least one row"):
        Plot.channel("t", values=np.zeros((0, 3)))
    with pytest.raises(ValueError, match="got a scalar"):
        Plot.channel("t", values=1.0)


def test_single_sample_is_allowed():
    # One sample is a constant channel, not an error: `at` cannot be
    # non-increasing with a single coordinate.
    call = Plot.channel("t", values=np.zeros((1, 3)))
    assert call.for_json()["args"][0]["at"].tolist() == [0.0]


# =============================================================================
# Serialization
# =============================================================================


def config_of(call):
    node = call.for_json()
    assert node["__type__"] == "function"
    assert node["path"] == "colight.resampleChannel"
    assert len(node["args"]) == 1
    return node["args"][0]


def test_serializes_as_one_jscall_with_ndarray_samples():
    config = config_of(Plot.channel("bend", values=QUATS, at=ANGLES, rule="qlerp"))
    # The parameter NAME rides as a plain string so inspect can report it...
    assert config["parameter"] == "bend"
    assert config["rule"] == "qlerp"
    # ...beside a $state read that resolves during AST evaluation.
    assert config["value"].for_json() == {
        "__type__": "js_source",
        "value": '$state["bend"]',
        "params": (),
        "expression": True,
        "scope": {},
    }
    # Values ship once, as ONE array of the declared shape.
    assert isinstance(config["values"], np.ndarray)
    assert config["values"].shape == (9, 4)
    assert isinstance(config["at"], np.ndarray)
    assert config["at"].shape == (9,)
    np.testing.assert_allclose(config["at"], ANGLES)


def test_at_defaults_to_arange():
    config = config_of(Plot.channel("frame", values=np.zeros((4, 3))))
    np.testing.assert_allclose(config["at"], [0, 1, 2, 3])


def test_wide_rows_ship_as_one_array():
    poses = np.zeros((5, 192 * 3), dtype=np.float32)
    config = config_of(Plot.channel("bend", values=poses, at=np.arange(5)))
    assert config["values"].shape == (5, 576)


def test_integer_values_are_coerced_to_float():
    config = config_of(Plot.channel("t", values=[[1, 2], [3, 4]]))
    assert config["values"].dtype.kind == "f"


def test_non_numeric_values_raise():
    with pytest.raises(ValueError, match="must be numeric"):
        Plot.channel("t", values=["a", "b"])


# =============================================================================
# Inspect legibility
# =============================================================================


def channel_scene():
    """A Group quaternion and a Mesh's positions, both channel-driven."""
    poses = np.zeros((5, 9), dtype=np.float32)
    poses[:, 0] = np.arange(5)
    return scene3d.Scene(
        scene3d.Group(
            quaternion=Plot.channel("bend", values=QUATS, at=ANGLES, rule="qlerp"),
            children=[
                scene3d.Mesh(
                    positions=Plot.channel(
                        "bend2",
                        values=poses,
                        at=np.linspace(-80.0, 80.0, 5),
                        rule="linear",
                    ),
                    indices=np.array([0, 1, 2], dtype=np.uint32),
                )
            ],
        )
    )


def inspect_scene(tmp_path) -> dict:
    visual = colight_inspect(channel_scene())
    assert visual is not None
    artifact = tmp_path / "channels.colight"
    artifact.write_bytes(visual.to_bytes())
    return inspect_tools.inspect_target(artifact)["visual"]


def test_channel_is_not_a_component(tmp_path):
    payload = inspect_scene(tmp_path)
    paths = [c["path"] for c in payload["components"]]
    assert "colight.resampleChannel" not in paths
    assert paths == ["scene3d.Scene", "scene3d.Group", "scene3d.Mesh"]


def test_inspect_reports_channels_on_their_component(tmp_path):
    payload = inspect_scene(tmp_path)
    by_parameter = {c["parameter"]: c for c in payload["channels"]}
    assert set(by_parameter) == {"bend", "bend2"}

    bend = by_parameter["bend"]
    assert bend["component"] == "scene3d.Group"
    assert bend["rule"] == "qlerp"
    assert bend["prop"] == "quaternion"
    assert bend["domain"] == [-80.0, 80.0]
    assert bend["samples"] == 9

    bend2 = by_parameter["bend2"]
    assert bend2["component"] == "scene3d.Mesh"
    assert bend2["rule"] == "linear"
    assert bend2["prop"] == "positions"
    assert bend2["domain"] == [-80.0, 80.0]
    assert bend2["samples"] == 5


def test_channel_sample_arrays_stay_in_the_array_records(tmp_path):
    # Unlike filter_by's thresholds, a channel's `values` IS the shipped data
    # (a dense pose table is the dominant cost of the artifact), so it stays
    # visible in the arrays list under a stable path.
    payload = inspect_scene(tmp_path)
    by_path = {a["path"]: a for a in payload["arrays"]}
    values = [p for p in by_path if p.endswith("quaternion.values")]
    ats = [p for p in by_path if p.endswith("quaternion.at")]
    assert len(values) == 1 and len(ats) == 1
    assert by_path[values[0]]["shape"] == [9, 4]
    assert by_path[ats[0]]["shape"] == [9]
    positions = [p for p in by_path if p.endswith("positions.values")]
    assert by_path[positions[0]]["shape"] == [5, 9]


def test_inspect_cli_text_reports_the_parameter(tmp_path):
    visual = colight_inspect(channel_scene())
    assert visual is not None
    artifact = tmp_path / "channels.colight"
    artifact.write_bytes(visual.to_bytes())
    result = CliRunner().invoke(
        cli_main, ["inspect", str(artifact)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "channel 'bend'" in result.output
    assert "rule qlerp" in result.output
    assert "drives quaternion" in result.output
    assert "domain [-80.0, 80.0]" in result.output


# =============================================================================
# End-to-end: the parameter sweeps entirely client-side (Chrome-gated)
# =============================================================================

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
        pytest.skip("Chrome not found for channel tests")


E2E_SCENE = """import math

import numpy as np

import colight.plot as Plot
import colight.scene3d as scene3d


def bend_quaternion(angle_deg):
    half = math.radians(angle_deg) / 2.0
    return [math.sin(half), 0.0, 0.0, math.cos(half)]


ANGLES = np.linspace(-80.0, 80.0, 9)
QUATS = np.array([bend_quaternion(a) for a in ANGLES], dtype=np.float32)

positions = np.array(
    [[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.4, 0.4, 0.0], [-0.4, 0.4, 0.0]],
    dtype=np.float32,
)
indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

(
    scene3d.Scene(
        scene3d.Group(
            position=[0.0, 0.0, 0.0],
            quaternion=Plot.channel("bend", values=QUATS, at=ANGLES, rule="qlerp"),
            children=[
                scene3d.Mesh(
                    positions=positions,
                    indices=indices,
                    color=[0.93, 0.53, 0.30],
                    shading="lit",
                    cull_mode="none",
                )
            ],
        ),
        {
            "defaultCamera": {
                "position": [0.0, -4.0, 0.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 40,
            }
        },
    )
    | Plot.initialState({"bend": BEND})
)
"""


def _scene_at_bend(tmp_path, bend: float) -> pathlib.Path:
    scene_path = tmp_path / f"bend_{bend:.0f}.py"
    scene_path.write_text(E2E_SCENE.replace("BEND", json.dumps(bend)))
    return scene_path


def test_two_parameter_values_render_differently(tmp_path):
    """The headline: only $state.bend differs, and the pixels follow."""
    from PIL import Image

    _require_renderer()
    runner = CliRunner()
    images = {}
    for bend in (-80.0, 80.0):
        scene_path = _scene_at_bend(tmp_path, bend)
        out = tmp_path / f"bend_{bend:.0f}.png"
        result = runner.invoke(
            cli_main,
            ["screenshot", str(scene_path), "--out", str(out), *SIZE_ARGS],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        images[bend] = np.asarray(Image.open(out).convert("RGB"), dtype=np.int64)

    diff = np.abs(images[-80.0] - images[80.0]).sum()
    assert diff > 5000, "channel parameter change did not move the geometry"


SCALAR_SCENE = """import numpy as np

import colight.plot as Plot

(
    Plot.dot(
        {"x": [1, 2, 3], "y": [1, 2, 3]},
        {"r": Plot.channel("size", values=np.array([2.0, 30.0]), at=[0.0, 1.0])},
    )
    + Plot.domain([0, 4])
    + {"height": 250, "width": 400}
) | Plot.initialState({"size": SIZE})
"""


def test_scalar_channel_drives_a_plot_mark(tmp_path):
    """A channel is not scene3d-specific, and a scalar one must survive AST
    evaluation as a number (not a constructed object) to reach a mark."""
    from PIL import Image

    _require_renderer()
    runner = CliRunner()
    images = {}
    for size in (0.0, 1.0):
        scene_path = tmp_path / f"size_{size:.0f}.py"
        scene_path.write_text(SCALAR_SCENE.replace("SIZE", json.dumps(size)))
        out = tmp_path / f"size_{size:.0f}.png"
        result = runner.invoke(
            cli_main,
            ["screenshot", str(scene_path), "--out", str(out), "--width", "450"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        image = np.asarray(Image.open(out).convert("RGB"), dtype=np.int64)
        images[size] = image
        # Dots really are drawn: the plot is not blank at either radius.
        assert (image < 200).any(), f"nothing rendered at size={size}"

    # r = 2 vs r = 30 is a large, unmistakable difference in ink.
    ink_small = int((images[0.0] < 200).sum())
    ink_large = int((images[1.0] < 200).sum())
    assert ink_large > ink_small * 2, (ink_small, ink_large)


def _artifact_scene(bend: float):
    """The same scene as an object, for a standalone artifact round trip."""
    poses = np.stack(
        [
            np.array(
                [
                    [-0.4, -0.4, 0.0],
                    [0.4, -0.4, 0.0],
                    [0.4, 0.4 + shift, 0.0],
                    [-0.4, 0.4 + shift, 0.0],
                ],
                dtype=np.float32,
            ).reshape(-1)
            for shift in np.linspace(-1.5, 1.5, 9)
        ]
    )
    return scene3d.Scene(
        scene3d.Mesh(
            positions=Plot.channel(
                "bend", values=poses, at=np.linspace(-80.0, 80.0, 9), rule="linear"
            ),
            indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32),
            color=[0.42, 0.72, 0.52],
            shading="lit",
            cull_mode="none",
        ),
        {
            "defaultCamera": {
                "position": [0.0, 0.0, 6.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0],
                "fov": 40,
            }
        },
    ) | Plot.initialState({"bend": bend})


def test_standalone_artifact_resamples_client_side(tmp_path):
    """A saved `.colight` still resamples: the table and the rule travel with
    it, so a different parameter value renders a different shape with no
    Python attached."""
    from PIL import Image

    _require_renderer()
    runner = CliRunner()
    images = {}
    for bend in (-80.0, 80.0):
        visual = colight_inspect(_artifact_scene(bend))
        assert visual is not None
        artifact = tmp_path / f"swept_{bend:.0f}.colight"
        artifact.write_bytes(visual.to_bytes())
        out = tmp_path / f"swept_{bend:.0f}.png"
        shot = runner.invoke(
            cli_main,
            ["screenshot", str(artifact), "--out", str(out), *SIZE_ARGS],
            catch_exceptions=False,
        )
        assert shot.exit_code == 0, shot.output
        images[bend] = np.asarray(Image.open(out).convert("RGB"), dtype=np.int64)

    diff = np.abs(images[-80.0] - images[80.0]).sum()
    assert diff > 5000, "round-tripped artifact lost the channel sweep"
