"""Tests for the `declarations` section shared by inspect and screenshot.

`colight inspect` derives the declarative facts a visual carries (filter_by,
color_by, switchable color_channels, `Plot.channel` parameters) from the
payload. `colight screenshot --json` reports the SAME facts from the SAME
structure walk, so an agent reading a screenshot report can discover what is
sweepable without a second command.

Assembly tests are pure Python; end-to-end tests drive the real CLI through
headless Chrome (skipped when Chrome or the JS bundle is missing).
"""

import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
from click.testing import CliRunner

import colight.env as env
from colight import scene3d
from colight.chrome_devtools import find_chrome
from colight.cli_tools import inspect_tools, screenshot_tools, structure
from colight.cli_tools import daemon as daemon_mod
from colight.inspect import inspect as colight_inspect
from colight_cli import main as cli_main

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
        pytest.skip("Chrome not found for screenshot tests")


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return tmp_path


# A scene declaring every discoverable fact at once: a filtered + color_by
# Cuboid and a PointCloud with switchable channels driven by a Plot.channel.
DECLARATIVE_SCENE = """import numpy as np
import colight.plot as Plot
from colight import scene3d

c = np.array([[-3.0, 0, 0], [-1, 0, 0], [1, 0, 0], [3, 0, 0]], dtype=np.float32)
grade = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

(
    scene3d.Scene(
        scene3d.Cuboid(
            centers=c,
            half_size=0.5,
            color_by={"values": grade, "cmap": "viridis",
                      "domain": (0, 3), "label": "Cu %"},
            filter_by={"values": grade, "min": 0.5, "label": "Cu cutoff"},
        ),
        scene3d.PointCloud(
            centers=c + np.array([0, 2.0, 0], dtype=np.float32),
            size=Plot.channel(
                "scale",
                values=np.array([0.1, 0.4, 0.9], dtype=np.float32),
                at=np.array([0.0, 1.0, 2.0]),
                rule="linear",
            ),
            color_channels={
                "CU_pct": {"values": grade, "cmap": "viridis",
                           "domain": (0, 3), "label": "Cu %"},
                "Lithology": {"values": [0, 1, 2, 1], "categories": [
                    {"value": 0, "label": "not logged"},
                    {"value": 1, "label": "Dacite"},
                    {"value": 2, "label": "Andesite"}]},
            },
            active_channel="Lithology",
        ),
        {"defaultCamera": {"position": [0, 0, 14], "target": [0, 1, 0],
                           "up": [0, 1, 0], "fov": 45}},
    )
    | Plot.initialState({"scale": 1.0})
)
"""


def _entry_by_label(declarations: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    matches = [d for d in declarations if d["label"] == label]
    assert len(matches) == 1, f"{label} not uniquely present in {declarations}"
    return matches[0]


class _StubScene:
    """A SceneLike that only supplies a payload (no renderer involved)."""

    def __init__(self, data: Dict[str, Any], buffers: List[bytes]) -> None:
        self._visual = (data, buffers)

    @property
    def studio(self) -> Any:  # pragma: no cover - never used by these tests
        raise AssertionError("declarations must not touch the renderer")

    def visual(self) -> Optional[Tuple[Dict[str, Any], List[bytes]]]:
        return self._visual

    def capture(self) -> Tuple[bytes, int, int]:  # pragma: no cover
        raise AssertionError("declarations must not capture")

    def mark_mutated(self) -> None:  # pragma: no cover
        pass


class TestDeclarationsAssembly:
    """The shared payload-derived assembly, no renderer."""

    def visual_bytes(self, project: pathlib.Path) -> Tuple[Dict[str, Any], List[bytes]]:
        path = project / "declarative.py"
        path.write_text(DECLARATIVE_SCENE)
        data, buffers, _block = screenshot_tools.resolve_visual(path)
        return data, buffers

    def test_reports_filter_color_and_channels(self, project: pathlib.Path):
        data, buffers = self.visual_bytes(project)
        declarations = inspect_tools.declarations_payload(data, buffers)

        cuboid = _entry_by_label(declarations, "scene3d.Cuboid")
        assert cuboid["filter_by"] == {"label": "Cu cutoff", "min": 0.5, "max": None}
        assert cuboid["color_by"]["cmap"] == "viridis"
        assert cuboid["color_by"]["domain"] == [0.0, 3.0]
        assert cuboid["color_by"]["categorical"] is False

        points = _entry_by_label(declarations, "scene3d.PointCloud")
        assert points["color_channels"]["active"] == "Lithology"
        roster = {c["name"]: c["kind"] for c in points["color_channels"]["channels"]}
        assert roster == {"CU_pct": "continuous", "Lithology": "categorical"}
        assert points["channels"] == [
            {
                "parameter": "scale",
                "rule": "linear",
                "prop": "size",
                "domain": [0.0, 2.0],
                "samples": 3,
            }
        ]

    def test_paths_are_unique_join_keys(self, project: pathlib.Path):
        data, buffers = self.visual_bytes(project)
        declarations = inspect_tools.declarations_payload(data, buffers)
        paths = [d["path"] for d in declarations]
        assert len(paths) == len(set(paths))

    def test_components_declaring_nothing_are_omitted(self, project: pathlib.Path):
        scene = scene3d.Scene(
            scene3d.Cuboid(centers=np.zeros((2, 3), dtype=np.float32), half_size=0.5)
        )
        visual = colight_inspect(scene)
        assert visual is not None
        target = project / "plain.colight"
        target.write_bytes(visual.to_bytes())
        data, buffers, _block = screenshot_tools.resolve_visual(target)
        assert inspect_tools.declarations_payload(data, buffers) == []

    def test_collect_declarations_uses_the_scene_payload(self, project: pathlib.Path):
        data, buffers = self.visual_bytes(project)
        via_scene = screenshot_tools.collect_declarations(_StubScene(data, buffers))
        assert via_scene == inspect_tools.declarations_payload(data, buffers)

    def test_no_payload_yields_no_declarations(self):
        class _Blind(_StubScene):
            def __init__(self) -> None:
                pass

            def visual(self):
                return None

        assert screenshot_tools.collect_declarations(_Blind()) == []

    def test_inspect_reports_the_same_section(self, project: pathlib.Path):
        data, buffers = self.visual_bytes(project)
        payload, _warnings = inspect_tools.inspect_visual_data(data, buffers)
        assert payload["declarations"] == inspect_tools.declarations_payload(
            data, buffers
        )


class TestCoverageAlignment:
    """Coverage indices are attached only where the mapping is unambiguous."""

    def walk(self, source: str, project: pathlib.Path) -> structure.WalkState:
        path = project / "scene.py"
        path.write_text(source)
        data, buffers, _block = screenshot_tools.resolve_visual(path)
        return structure.collect_structure(data, buffers)

    FLAT = """import numpy as np
from colight import scene3d

pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
scene3d.Scene(
    scene3d.PointCloud(centers=pts, size=0.2,
                       color_by={"values": np.array([0.0, 1.0, 2.0]),
                                 "cmap": "viridis"}),
    scene3d.Ellipsoid(centers=pts + 2.0, half_size=0.4,
                      color_by={"values": np.array([0.0, 1.0, 2.0]),
                                "cmap": "viridis"}),
)
"""

    GROUPED = """import numpy as np
from colight import scene3d

pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
scene3d.Scene(
    scene3d.PointCloud(centers=pts, size=0.2,
                       color_by={"values": np.array([0.0, 1.0, 2.0]),
                                 "cmap": "viridis"}),
    scene3d.Group(name="g", children=[
        scene3d.Ellipsoid(centers=pts + 2.0, half_size=0.4,
                          color_by={"values": np.array([0.0, 1.0, 2.0]),
                                    "cmap": "viridis"}),
    ]),
)
"""

    def coverage(self, *types: str) -> Dict[str, Any]:
        return {
            "components": [
                {"component": i, "type": t, "instances": 3} for i, t in enumerate(types)
            ]
        }

    def test_flat_scene_aligns_by_drawable_order(self, project: pathlib.Path):
        state = self.walk(self.FLAT, project)
        declarations = inspect_tools.component_declarations(state)
        inspect_tools.align_declarations_to_coverage(
            state, declarations, self.coverage("PointCloud", "Ellipsoid")
        )
        assert {d["label"]: d["component"] for d in declarations} == {
            "scene3d.PointCloud": 0,
            "scene3d.Ellipsoid": 1,
        }

    def test_groups_are_flattened_away_not_counted(self, project: pathlib.Path):
        """A Group is a walker component but never a compiled one: its child
        must still land on the right index."""
        state = self.walk(self.GROUPED, project)
        assert [c.path for c in state.components] == [
            "scene3d.Scene",
            "scene3d.PointCloud",
            "scene3d.Group",
            "scene3d.Ellipsoid",
        ]
        declarations = inspect_tools.component_declarations(state)
        inspect_tools.align_declarations_to_coverage(
            state, declarations, self.coverage("PointCloud", "Ellipsoid")
        )
        assert {d["label"]: d["component"] for d in declarations} == {
            "scene3d.PointCloud": 0,
            "scene3d.Ellipsoid": 1,
        }

    def test_count_mismatch_omits_the_index(self, project: pathlib.Path):
        """A compiled scene that dropped or expanded components is ambiguous:
        report the path only rather than guess an index."""
        state = self.walk(self.FLAT, project)
        declarations = inspect_tools.component_declarations(state)
        inspect_tools.align_declarations_to_coverage(
            state, declarations, self.coverage("PointCloud")
        )
        assert all("component" not in d for d in declarations)

    def test_non_contiguous_coverage_omits_the_index(self, project: pathlib.Path):
        """Only components that produced a render object appear in coverage; a
        gap means the indices are not a positional match."""
        state = self.walk(self.FLAT, project)
        declarations = inspect_tools.component_declarations(state)
        coverage = {
            "components": [
                {"component": 1, "type": "PointCloud"},
                {"component": 3, "type": "Ellipsoid"},
            ]
        }
        inspect_tools.align_declarations_to_coverage(state, declarations, coverage)
        assert all("component" not in d for d in declarations)

    def test_absent_coverage_is_harmless(self, project: pathlib.Path):
        state = self.walk(self.FLAT, project)
        declarations = inspect_tools.component_declarations(state)
        inspect_tools.align_declarations_to_coverage(state, declarations, None)
        assert all("component" not in d for d in declarations)
        assert len(declarations) == 2


class TestScreenshotDeclarationsCli:
    """End-to-end through the real CLI + headless Chrome."""

    def test_json_reports_declarations_with_coverage_indices(
        self, project: pathlib.Path
    ):
        _require_renderer()
        scene_path = project / "declarative.py"
        scene_path.write_text(DECLARATIVE_SCENE)
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_path),
                "--out",
                str(project / "shot.png"),
                *SIZE_ARGS,
                "--json",
                "--no-daemon",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        declarations = payload["declarations"]

        cuboid = _entry_by_label(declarations, "scene3d.Cuboid")
        assert cuboid["filter_by"]["min"] == 0.5
        assert cuboid["color_by"]["cmap"] == "viridis"

        points = _entry_by_label(declarations, "scene3d.PointCloud")
        assert points["channels"][0]["parameter"] == "scale"
        assert points["color_channels"]["active"] == "Lithology"

        # Both components render, so the coverage index join is unambiguous.
        covered = {c["component"]: c["type"] for c in payload["coverage"]["components"]}
        assert covered[cuboid["component"]] == "Cuboid"
        assert covered[points["component"]] == "PointCloud"

    def test_matches_inspect_on_the_same_target(self, project: pathlib.Path):
        """The headline: screenshot reports exactly what inspect knows."""
        _require_renderer()
        scene_path = project / "declarative.py"
        scene_path.write_text(DECLARATIVE_SCENE)
        runner = CliRunner()

        shot = runner.invoke(
            cli_main,
            [
                "screenshot",
                str(scene_path),
                "--out",
                str(project / "shot.png"),
                *SIZE_ARGS,
                "--json",
                "--no-daemon",
            ],
            catch_exceptions=False,
        )
        assert shot.exit_code == 0, shot.output
        inspected = runner.invoke(
            cli_main, ["inspect", str(scene_path), "--json"], catch_exceptions=False
        )
        assert inspected.exit_code == 0, inspected.output

        from_shot = json.loads(shot.output)["declarations"]
        from_inspect = json.loads(inspected.output)["visuals"][-1]["visual"][
            "declarations"
        ]
        # Screenshot additionally resolves coverage indices; the declarative
        # facts themselves are identical.
        stripped = [
            {k: v for k, v in entry.items() if k != "component"} for entry in from_shot
        ]
        assert stripped == from_inspect

    def test_channel_parameter_from_a_real_example(self, project: pathlib.Path):
        """fault_drag declares a Plot.channel sweep on a Group transform: the
        parameter, rule, domain and driven prop must all be discoverable."""
        _require_renderer()
        example = (
            pathlib.Path(__file__).resolve().parents[3]
            / "examples"
            / "src"
            / "notebooks"
            / "scene3d"
            / "fault_drag.py"
        )
        if not example.exists():
            pytest.skip(f"example not present: {example}")
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(example),
                "--out",
                str(project / "fault.png"),
                *SIZE_ARGS,
                "--json",
                "--no-daemon",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        declarations = json.loads(result.output)["declarations"]
        channels = [c for entry in declarations for c in entry.get("channels", [])]
        assert {
            "parameter": "throw",
            "rule": "linear",
            "prop": "position",
            "domain": [0.0, 160.0],
            "samples": 9,
        } in channels
        # The channel drives a Group, which is flattened away at compile time
        # and has no coverage component: no index may be invented for it.
        group = _entry_by_label(declarations, "scene3d.Group")
        assert "component" not in group


class TestDaemonParity:
    """Declarations are assembled in the shared screenshot_source path, so
    daemon and direct modes must report exactly the same section."""

    def test_daemon_and_direct_declarations_identical(self, project: pathlib.Path):
        _require_renderer()
        scene_path = project / "declarative.py"
        scene_path.write_text(DECLARATIVE_SCENE)
        runner = CliRunner()

        def shot(out_name: str, *extra: str) -> dict:
            result = runner.invoke(
                cli_main,
                [
                    "screenshot",
                    str(scene_path),
                    "--out",
                    str(project / out_name),
                    *SIZE_ARGS,
                    *extra,
                    "--json",
                ],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            return json.loads(result.output)

        direct = shot("direct.png", "--no-daemon")

        daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0, pool_size=2)
        daemon.start()
        try:
            # Cold (visual shipped to the daemon) and warm (served from the
            # scene cache) must both carry the payload-derived section.
            cold = shot("cold.png")
            warm = shot("warm.png")
            assert daemon.request_counts.get("/screenshot") == 2
        finally:
            daemon.shutdown()

        assert cold["declarations"] == direct["declarations"]
        assert warm["declarations"] == direct["declarations"]
