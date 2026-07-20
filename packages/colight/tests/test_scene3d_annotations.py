"""Tests for named annotation callouts on scene3d.

Covers the Python Annotation/annotate API, that annotations are resident
$state (so they persist into .colight artifacts and inspect reports them), and
the machine legibility the design requires: screenshot --json reports each
callout's projected screen position (in pick-at pixel space) and its labels are
captured in the screenshot; the projection tracks the camera; pick-at reports
instance-anchored membership; and a position anchor lands on origin-shifted
geometry.

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
# API: Annotation specs + state seeding
# =============================================================================


def test_annotation_position_anchor():
    ann = scene3d.Annotation("note", "hello", position=[1, 2, 3])
    assert ann == {"note": {"text": "hello", "anchor": {"position": [1.0, 2.0, 3.0]}}}


def test_annotation_instance_anchor():
    ann = scene3d.Annotation("note", "hi", component=2, instance=7)
    assert ann == {"note": {"text": "hi", "anchor": {"component": 2, "instance": 7}}}


def test_annotation_style_renamed_to_camelcase():
    ann = scene3d.Annotation(
        "note", "hi", position=[0, 0, 0], style={"color": [1, 0, 0]}
    )
    assert ann["note"]["style"] == {"color": [1, 0, 0]}


def test_annotation_position_must_be_3d():
    with pytest.raises(ValueError, match="3 components"):
        scene3d.Annotation("note", "hi", position=[1, 2])


def test_annotation_rejects_both_anchor_forms():
    with pytest.raises(ValueError, match="not both"):
        scene3d.Annotation("note", "hi", position=[1, 2, 3], component=0)


def test_annotation_instance_anchor_requires_instance():
    with pytest.raises(ValueError, match="requires both"):
        scene3d.Annotation("note", "hi", component=0)


def test_annotation_requires_an_anchor():
    with pytest.raises(ValueError, match="requires either"):
        scene3d.Annotation("note", "hi")


def test_annotate_seeds_state_annotations(tmp_path):
    centers = np.array([[0, 0, 0]], dtype=np.float32)
    scene = scene3d.Scene(scene3d.Cuboid(centers=centers)) | scene3d.annotate(
        scene3d.Annotation("a", "text a", position=[1, 0, 0]),
        scene3d.Annotation("b", "text b", component=0, instance=0),
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    assert "annotations" in payload["visual"]["state_keys"]
    names = {a["name"] for a in payload["visual"].get("annotations", [])}
    assert names == {"a", "b"}


def test_scene_passes_live_annotations_ref():
    centers = np.array([[0, 0, 0]], dtype=np.float32)
    scene = scene3d.Scene(scene3d.Cuboid(centers=centers))
    _ref, props = scene.for_json()
    # The Scene always passes a live $state.annotations reference so
    # state-resident annotations resolve on every render.
    assert "annotations" in props


# =============================================================================
# Artifacts + inspect: annotations are just state, so they persist
# =============================================================================


def make_artifact(tmp_path: pathlib.Path, scene) -> pathlib.Path:
    visual = colight_inspect(scene)
    assert visual is not None
    target = tmp_path / "scene.colight"
    target.write_bytes(visual.to_bytes())
    return target


centers4 = np.array([[i, 0, 0] for i in range(4)], dtype=np.float32)


def test_saved_artifact_reports_its_annotations(tmp_path):
    scene = scene3d.Scene(
        scene3d.Cuboid(centers=centers4, half_size=0.4)
    ) | scene3d.annotate(
        scene3d.Annotation("inst", "check twin hole", component=0, instance=3),
        scene3d.Annotation("pos", "high-grade shoot", position=[1.5, 2.0, 0.0]),
    )
    target = make_artifact(tmp_path, scene)
    payload = inspect_tools.inspect_target(target)
    annotations = payload["visual"]["annotations"]
    by_name = {a["name"]: a for a in annotations}
    assert by_name["inst"]["text"] == "check twin hole"
    assert by_name["inst"]["anchor"] == {"component": 0, "instance": 3}
    assert by_name["pos"]["anchor"] == {"position": [1.5, 2.0, 0.0]}


def test_inspect_omits_annotations_when_absent(tmp_path):
    target = make_artifact(
        tmp_path, scene3d.Scene(scene3d.Cuboid(centers=centers4, half_size=0.4))
    )
    payload = inspect_tools.inspect_target(target)
    assert "annotations" not in payload["visual"]


# =============================================================================
# End-to-end machine legibility (Chrome-gated)
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
    | S.annotate(
        S.Annotation("inst", "instance 1", component=0, instance=1),
        S.Annotation("pos", "at origin", position=[0.0, 0.0, 0.0]),
    )
    | {"defaultCamera": {"position": [0, 0, 12], "target": [0, 0, 0],
                         "up": [0, 1, 0], "fov": 45}}
)
"""

# Origin-shifted (UTM-scale) scene: geometry sits at ~445000 easting, the
# position anchor is given in the same world coords and must land on it.
SCENE_ORIGIN = """import colight.scene3d as S

(
    S.Scene(
        S.Cuboid(
            centers=[[445000.0, 0, 0], [445002.0, 0, 0]],
            half_size=0.5,
            color=[0.3, 0.3, 0.6],
        ),
        origin=[445001.0, 0.0, 0.0],
    )
    | S.annotate(
        S.Annotation("utm", "on geometry", position=[445000.0, 0.0, 0.0]),
    )
    | {"defaultCamera": {"position": [-1, 0, 12], "target": [-1, 0, 0],
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
        pytest.skip("Chrome not found for annotation tests")


@pytest.fixture()
def scene_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "ann_scene.py"
    path.write_text(SCENE)
    return path


def _screenshot_json(scene_path, out, extra=None):
    args = [
        "screenshot",
        str(scene_path),
        "--out",
        str(out),
        *SIZE_ARGS,
        "--json",
    ]
    if extra:
        args.extend(extra)
    result = CliRunner().invoke(cli_main, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_screenshot_reports_annotation_screen_positions(scene_file, tmp_path):
    _require_renderer()
    payload = _screenshot_json(scene_file, tmp_path / "ann.png")
    annotations = payload.get("annotations")
    assert annotations, "screenshot --json should report annotations"
    by_name = {a["name"]: a for a in annotations}
    assert set(by_name) == {"inst", "pos"}
    # Both anchors are on-screen and in front of the camera.
    for a in annotations:
        assert a["visible"] is True
        assert a["screen"] is not None
        assert 0 <= a["screen"]["x"] <= 400
        assert 0 <= a["screen"]["y"] <= 400
    # The origin anchor projects to the center of the 400x400 canvas.
    assert by_name["pos"]["screen"]["x"] == pytest.approx(200, abs=8)
    assert by_name["pos"]["screen"]["y"] == pytest.approx(200, abs=8)
    # The instance anchor (center x=-1) projects left of the origin anchor.
    assert by_name["inst"]["screen"]["x"] < by_name["pos"]["screen"]["x"]
    # Resolved world position is reported too.
    assert by_name["inst"]["world"][0] == pytest.approx(-1, abs=1e-3)


def test_screenshot_captures_annotation_labels(scene_file, tmp_path):
    """The DOM overlay (marker + label) must land in the captured PNG — the
    whole point for agents. Evidence: the region around a projected label has
    materially more non-background pixels with annotations than without."""
    _require_renderer()
    from PIL import Image

    with_out = tmp_path / "with.png"
    payload = _screenshot_json(scene_file, with_out)
    pos = next(a for a in payload["annotations"] if a["name"] == "pos")
    sx, sy = int(pos["screen"]["x"]), int(pos["screen"]["y"])

    # Same scene without annotations.
    plain = scene_file.parent / "plain.py"
    plain.write_text(
        SCENE.split("| S.annotate")[0]
        + '| {"defaultCamera": {"position": [0, 0, 12], "target": [0, 0, 0],'
        ' "up": [0, 1, 0], "fov": 45}}\n)\n'
    )
    plain_out = tmp_path / "plain.png"
    _screenshot_json(plain, plain_out)

    # The label sits up-and-right of the marker; sample that box.
    box = (sx + 6, sy - 40, sx + 90, sy - 6)

    def bright_pixels(path):
        img = Image.open(path).convert("RGB")
        crop = img.crop(box)
        arr = np.asarray(crop).astype(np.int32)
        # The label is a near-white card; count bright pixels.
        return int((arr.sum(axis=2) > 600).sum())

    annotated = bright_pixels(with_out)
    plain_bright = bright_pixels(plain_out)
    # The white label card adds many bright pixels the plain scene lacks.
    assert annotated > plain_bright + 50, (annotated, plain_bright)


def test_annotation_projection_tracks_camera(scene_file):
    """Two different cameras -> the callout's projected screen position moves
    consistently with the projection. Driven through the same snapshot API the
    CLI uses; the off-center instance anchor (instance 1, center x=-1) is
    sensitive to the view direction (unlike an origin anchor, which stays
    centered under any framing)."""
    _require_renderer()
    from colight.cli_tools import scene_pick, screenshot_tools

    source = screenshot_tools.DirectSceneSource(scene_file, width=400, height=400)
    with source.scene() as scene:
        studio = scene.studio

        def project_inst(direction):
            scene_pick.frame_view(studio, direction, [0, 1, 0])
            info = studio.evaluate("window.colight.scene3d.info()[0].annotations")
            return next(a for a in info if a["name"] == "inst")

        front = project_inst([0, 0, 1])
        oblique = project_inst([1, 0.4, 0.7])

    assert front["visible"] and oblique["visible"]
    # The projected screen position changes with the camera.
    assert front["screen"] != oblique["screen"]
    assert abs(front["screen"]["x"] - oblique["screen"]["x"]) > 5


def test_annotation_position_anchor_lands_on_origin_shifted_geometry(tmp_path):
    _require_renderer()
    origin_scene = tmp_path / "origin.py"
    origin_scene.write_text(SCENE_ORIGIN)
    payload = _screenshot_json(origin_scene, tmp_path / "origin.png")
    utm = next(a for a in payload["annotations"] if a["name"] == "utm")
    # Camera targets the first cuboid at UTM x=445000; the anchor is at the
    # same world coords, so after the origin shift it projects on-screen near
    # the geometry (left-of-center where the cuboid sits), not off in space.
    assert utm["visible"] is True
    assert utm["screen"] is not None
    assert 0 <= utm["screen"]["x"] <= 400
    assert 0 <= utm["screen"]["y"] <= 400


def test_pick_at_reports_annotation_membership(scene_file):
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
    # The "inst" callout is anchored to instance 1; "pos" (a position anchor)
    # is not instance-bound and must not appear.
    assert hits[0]["annotations"] == ["inst"]


def test_screenshot_check_deterministic_with_annotations(scene_file, tmp_path):
    _require_renderer()
    payload = _screenshot_json(scene_file, tmp_path / "det.png", extra=["--check"])
    assert payload["deterministic"] is True
