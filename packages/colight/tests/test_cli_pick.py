"""Tests for agent-facing scene3d pick queries: pick-at, pick-where,
screenshot coverage and --frame.

Pure-analysis tests run everywhere; end-to-end tests drive the real CLI
through headless Chrome (skipped when Chrome or the JS bundle is missing,
same mechanism as the other visual tests).
"""

import json
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner

import colight.env as env
from colight.chrome_devtools import find_chrome
from colight.cli_tools import scene_pick
from colight_cli import main as cli_main

# Camera on +z looking at the origin; ellipsoids use the default half
# size 0.5. Instance 1 sits at x=2; instance 2 sits exactly behind it on
# the camera ray through instance 1's center (x = 2 * 14/12), so it is
# fully occluded. The cuboid floats above at y=1.5.
SCENE = """import colight.scene3d as S

scene = (
    S.Ellipsoid(
        centers=[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.3333, 0.0, -2.0]],
        colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    + S.Cuboid(centers=[[0.0, 1.5, 0.0]], color=[1, 1, 0])
    + {"defaultCamera": {"position": [0, 0, 12], "target": [0, 0, 0],
                         "up": [0, 1, 0], "fov": 45}}
)
scene
"""

PLOT_2D = """import colight.plot as Plot

Plot.dot({"x": [1.0, 2.0, 3.0], "y": [2.0, 1.0, 3.0]})
"""

SIZE_ARGS = ["--width", "400", "--height", "400"]

# Ellipsoid instance 0 (center [-2, 0, 0]) projects near page (120, 200)
# at 400x400 / fov 45 with the camera above.
HIT_X, HIT_Y = 120, 200


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


@pytest.fixture
def scene_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "scene.py"
    path.write_text(SCENE)
    return path


class TestParsing:
    def test_instance_ranges(self):
        assert scene_pick.parse_instance_ranges("0-3,7") == [(0, 3), (7, 7)]
        assert scene_pick.parse_instance_ranges("5") == [(5, 5)]
        with pytest.raises(ValueError, match="invalid instance range"):
            scene_pick.parse_instance_ranges("5-2")
        with pytest.raises(ValueError):
            scene_pick.parse_instance_ranges("")

    def test_frame_selector(self):
        assert scene_pick.parse_frame_selector("Cuboid") == ("Cuboid", None)
        assert scene_pick.parse_frame_selector("0:2-4,7") == (
            "0",
            [(2, 4), (7, 7)],
        )

    def test_resolve_component(self):
        components = [
            {"component": 0, "type": "Ellipsoid", "count": 3},
            {"component": 1, "type": "Cuboid", "count": 1},
            {"component": 2, "type": "Cuboid", "count": 2},
        ]
        assert scene_pick.resolve_component(components, "1")["component"] == 1
        assert scene_pick.resolve_component(components, "ellipsoid")["component"] == 0
        with pytest.raises(ValueError, match="matches several"):
            scene_pick.resolve_component(components, "Cuboid")
        with pytest.raises(ValueError, match="no component"):
            scene_pick.resolve_component(components, "PointCloud")
        with pytest.raises(ValueError, match="no component with index"):
            scene_pick.resolve_component(components, "9")


def make_snapshot(ids: np.ndarray, rect_left=0.0, rect_top=0.0, dpr=1.0):
    height, width = ids.shape
    return scene_pick.SceneSnapshot(
        ids=ids.astype(np.uint32),
        legend=[
            {"component": 0, "type": "Cuboid", "count": 2, "idBase": 0},
            {"component": 1, "type": "PointCloud", "count": 3, "idBase": 2},
        ],
        width=width,
        height=height,
        rect={
            "left": rect_left,
            "top": rect_top,
            "width": width / dpr,
            "height": height / dpr,
        },
        dpr=dpr,
        scenes=1,
        components=[
            {"component": 0, "type": "Cuboid", "count": 2},
            {"component": 1, "type": "PointCloud", "count": 3},
        ],
        camera=None,
    )


class TestBufferAnalysis:
    """Pure numpy analysis on synthetic pick buffers (raw id = element+1)."""

    def test_coverage_payload(self):
        ids = np.zeros((10, 10), dtype=np.uint32)
        ids[0:2, 0:5] = 1  # component 0, instance 0 -> 10 px
        ids[5:8, 5:9] = 3  # component 1, instance 0 -> 12 px
        payload = scene_pick.coverage_payload(make_snapshot(ids))
        by_type = {c["type"]: c for c in payload["components"]}
        assert by_type["Cuboid"]["pixels"] == 10
        assert by_type["Cuboid"]["fraction"] == 0.1
        assert by_type["PointCloud"]["pixels"] == 12
        assert payload["background"]["pixels"] == 78
        assert payload["background"]["fraction"] == 0.78

    def test_hits_at_ranks_by_distance(self):
        ids = np.zeros((20, 20), dtype=np.uint32)
        ids[10, 10] = 2  # component 0, instance 1 at the query point
        ids[10, 13] = 4  # component 1, instance 1 three px away
        hits = scene_pick.hits_at(make_snapshot(ids), 10.5, 10.5, 5)
        assert [(h["component"], h["instance"]) for h in hits] == [
            (0, 1),
            (1, 1),
        ]
        assert hits[0]["distance"] < hits[1]["distance"]
        assert hits[0]["share"] > 0

    def test_hits_at_respects_radius_and_rect(self):
        ids = np.zeros((20, 20), dtype=np.uint32)
        ids[10, 13] = 1
        snapshot = make_snapshot(ids, rect_left=100.0, rect_top=50.0)
        # Page coordinates offset by the canvas rect.
        assert scene_pick.hits_at(snapshot, 110.5, 60.5, 2) == []
        hits = scene_pick.hits_at(snapshot, 110.5, 60.5, 4)
        assert len(hits) == 1
        # Far outside the canvas: no crash, no hits.
        assert scene_pick.hits_at(snapshot, 500, 500, 6) == []

    def test_selection_metrics_occlusion(self):
        full = np.zeros((10, 10), dtype=np.uint32)
        solo = np.zeros((10, 10), dtype=np.uint32)
        # Instance 0 of component 0 projects to 8 px, 4 visible.
        solo[2, 2:10] = 1
        full[2, 2:6] = 1
        metrics = scene_pick.selection_metrics(
            make_snapshot(full),
            make_snapshot(solo),
            {"component": 0, "type": "Cuboid", "count": 2, "idBase": 0},
            [(0, 0)],
        )
        assert metrics["selected"] == 1
        assert metrics["visible_pixels"] == 4
        assert metrics["projected_pixels"] == 8
        assert metrics["visibility"] == 0.5
        assert metrics["bbox"] == [2.0, 2.0, 6.0, 3.0]

    def test_selection_metrics_fully_occluded(self):
        full = np.zeros((10, 10), dtype=np.uint32)
        solo = np.zeros((10, 10), dtype=np.uint32)
        solo[4, 4:6] = 2
        metrics = scene_pick.selection_metrics(
            make_snapshot(full),
            make_snapshot(solo),
            {"component": 0, "type": "Cuboid", "count": 2, "idBase": 0},
            [(1, 1)],
        )
        assert metrics["visible_pixels"] == 0
        assert metrics["visibility"] == 0.0
        assert metrics["hidden_instances"] == 1
        assert "bbox" not in metrics
        assert metrics["projected_bbox"] == [4.0, 4.0, 6.0, 5.0]


class TestPickCli:
    """End-to-end through the real CLI + headless Chrome."""

    def test_coverage_in_screenshot_json(
        self, scene_file: pathlib.Path, tmp_path: pathlib.Path
    ):
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
        payload = json.loads(result.output)
        coverage = payload["coverage"]
        types = [c["type"] for c in coverage["components"]]
        assert types == ["Ellipsoid", "Cuboid"]
        assert coverage["components"][0]["instances"] == 3
        assert coverage["components"][0]["fraction"] > 0.005
        total = sum(c["pixels"] for c in coverage["components"])
        total += coverage["background"]["pixels"]
        assert total == coverage["width"] * coverage["height"]

    def test_coverage_omitted_for_non_scene(self, tmp_path: pathlib.Path):
        _require_renderer()
        target = tmp_path / "plot.py"
        target.write_text(PLOT_2D)
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(target),
                "--out",
                str(tmp_path / "plot.png"),
                "--width",
                "300",
                "--height",
                "200",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "coverage" not in payload

    def test_pick_at_hit_dereferences_values(self, scene_file: pathlib.Path):
        _require_renderer()
        result = CliRunner().invoke(
            cli_main,
            [
                "pick-at",
                str(scene_file),
                f"{HIT_X},{HIT_Y}",
                *SIZE_ARGS,
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["block"]
        hit = payload["hits"][0]
        assert (hit["component"], hit["type"], hit["instance"]) == (
            0,
            "Ellipsoid",
            0,
        )
        assert hit["share"] > 0.5
        # Dereferenced values come from the arrays actually rendered.
        assert hit["values"]["center"] == [-2.0, 0.0, 0.0]
        assert hit["values"]["color"] == [1.0, 0.0, 0.0]

    def test_pick_at_miss_and_radius(self, scene_file: pathlib.Path):
        _require_renderer()
        # ~10 px right of instance 0's silhouette: a tight radius misses...
        miss = CliRunner().invoke(
            cli_main,
            [
                "pick-at",
                str(scene_file),
                "150,200",
                "--radius",
                "4",
                *SIZE_ARGS,
                "--json",
            ],
        )
        assert miss.exit_code == 1, miss.output
        assert json.loads(miss.output)["hits"] == []
        # ...a wider radius reaches the same instance.
        hit = CliRunner().invoke(
            cli_main,
            [
                "pick-at",
                str(scene_file),
                "150,200",
                "--radius",
                "15",
                *SIZE_ARGS,
                "--json",
            ],
        )
        assert hit.exit_code == 0, hit.output
        hits = json.loads(hit.output)["hits"]
        assert (hits[0]["component"], hits[0]["instance"]) == (0, 0)
        assert hits[0]["distance"] <= 15

    def test_pick_at_non_scene_exits_2(self, tmp_path: pathlib.Path):
        _require_renderer()
        target = tmp_path / "plot.py"
        target.write_text(PLOT_2D)
        result = CliRunner().invoke(
            cli_main, ["pick-at", str(target), "50,50", *SIZE_ARGS]
        )
        assert result.exit_code == 2
        assert "no scene3d scene" in result.output

    def test_pick_where_visible_instance(self, scene_file: pathlib.Path):
        _require_renderer()
        result = CliRunner().invoke(
            cli_main,
            [
                "pick-where",
                str(scene_file),
                "--component",
                "0",
                "--instances",
                "1",
                *SIZE_ARGS,
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["visible_instances"] == 1
        assert payload["visibility"] > 0.9
        # Instance 1 (center x=+2) lands right of the canvas midline.
        x0, _y0, x1, _y1 = payload["bbox"]
        assert x0 > 200
        assert payload["centroid"][0] > 200

    def test_pick_where_occluded_instance(self, scene_file: pathlib.Path):
        _require_renderer()
        result = CliRunner().invoke(
            cli_main,
            [
                "pick-where",
                str(scene_file),
                "--component",
                "0",
                "--instances",
                "2",
                *SIZE_ARGS,
                "--json",
            ],
        )
        # Fully occluded behind instance 1 -> exit 1 with a projected bbox.
        assert result.exit_code == 1, result.output
        payload = json.loads(result.output)
        assert payload["visible_pixels"] == 0
        assert payload["projected_pixels"] > 0
        assert payload["hidden_instances"] == 1
        assert "projected_bbox" in payload

    def test_pick_where_overlay(self, scene_file: pathlib.Path, tmp_path: pathlib.Path):
        _require_renderer()
        plain = tmp_path / "plain.png"
        CliRunner().invoke(
            cli_main,
            ["screenshot", str(scene_file), "--out", str(plain), *SIZE_ARGS],
        )
        overlay = tmp_path / "overlay.png"
        result = CliRunner().invoke(
            cli_main,
            [
                "pick-where",
                str(scene_file),
                "--component",
                "Cuboid",
                "--out",
                str(overlay),
                *SIZE_ARGS,
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["out"] == str(overlay)
        assert overlay.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        # The highlight overlay must actually differ from the plain render.
        assert overlay.read_bytes() != plain.read_bytes()

    def test_frame_zooms_selection(
        self, scene_file: pathlib.Path, tmp_path: pathlib.Path
    ):
        _require_renderer()
        unframed_out = tmp_path / "unframed.png"
        unframed = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "--out",
                str(unframed_out),
                *SIZE_ARGS,
                "--json",
            ],
        )
        assert unframed.exit_code == 0, unframed.output
        unframed_payload = json.loads(unframed.output)

        framed_out = tmp_path / "framed.png"
        framed = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "--out",
                str(framed_out),
                *SIZE_ARGS,
                "--frame",
                "Cuboid",
                "--json",
            ],
        )
        assert framed.exit_code == 0, framed.output
        framed_payload = json.loads(framed.output)
        assert framed_payload["frame"]["component"] == 1
        assert framed_payload["frame"]["camera"]["target"] == [0.0, 1.5, 0.0]

        # The framed image is a genuinely different view...
        assert framed_payload["sha256"] != unframed_payload["sha256"]

        # ...and the selection's coverage fraction strictly increases:
        # that is what "zoom" means.
        def fraction(payload, component):
            for entry in payload["coverage"]["components"]:
                if entry["component"] == component:
                    return entry["fraction"]
            return 0.0

        assert fraction(framed_payload, 1) > 5 * fraction(unframed_payload, 1)

    def test_frame_deterministic(
        self, scene_file: pathlib.Path, tmp_path: pathlib.Path
    ):
        _require_renderer()
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "--out",
                str(tmp_path / "framed.png"),
                *SIZE_ARGS,
                "--frame",
                "0:2",
                "--check",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["deterministic"] is True
