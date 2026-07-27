"""Tests for the frame-time probe: colight screenshot --probe-param."""

import json
import pathlib

import pytest
from click.testing import CliRunner

import colight.env as env
from colight.chrome_devtools import find_chrome
from colight.cli_tools import probe_tools
from colight_cli import main as cli_main


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


# A small swept scene: two poses of a 4-vertex quad behind a Plot.channel, so
# moving the slider exercises the same evaluate -> compile -> equality-gate ->
# render path a real scene does, at a size that renders in milliseconds.
SMALL_SWEEP = """
import numpy as np
import colight.plot as Plot
from colight import scene3d

POSES = np.array(
    [
        [-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.5, 1.0, -1.0, 0.0, 1.0, 1.0, 0.5, -1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

(
    scene3d.Scene(
        scene3d.Mesh(
            positions=Plot.channel("t", values=POSES, at=[0.0, 1.0], rule="linear"),
            indices=INDICES,
            color=[0.4, 0.7, 0.5],
            shading="lit",
            cull_mode="none",
        ),
        {
            "defaultCamera": {
                "position": [3.0, -3.0, 2.0],
                "target": [0.0, 0.0, 0.0],
                "up": [0.0, 0.0, 1.0],
                "fov": 45.0,
            }
        },
    )
    | Plot.Slider("t", init=0.0, range=[0.0, 1.0], step=0.01)
)
"""


class TestSweepMath:
    """The sweep's pure helpers need no renderer."""

    def test_parse_range(self):
        assert probe_tools.parse_range("0,1") == (0.0, 1.0)
        assert probe_tools.parse_range(" -80 , 80 ") == (-80.0, 80.0)

    def test_parse_range_rejects_malformed(self):
        with pytest.raises(ValueError, match="lo,hi"):
            probe_tools.parse_range("5")
        with pytest.raises(ValueError, match="two numbers"):
            probe_tools.parse_range("a,b")

    def test_sweep_values_spans_endpoints(self):
        values = probe_tools.sweep_values(0.0, 1.0, 5)
        assert len(values) == 5
        assert values[0] == 0.0
        assert values[-1] == 1.0

    def test_sweep_single_frame(self):
        assert probe_tools.sweep_values(2.0, 9.0, 1) == [2.0]

    def test_sweep_rejects_zero_frames(self):
        with pytest.raises(ValueError, match=">= 1"):
            probe_tools.sweep_values(0.0, 1.0, 0)


class TestSummarize:
    """Aggregation is a pure function of a client snapshot."""

    def test_stage_stats(self):
        snapshot = {
            "enabled": True,
            "stages": {
                "evaluate": {"durations": [1.0, 2.0, 3.0, 4.0], "count": 4},
                "compile": {"durations": [10.0], "count": 1},
            },
            "writes": {"calls": [2, 4], "bytes": [100, 300]},
            "frameIntervals": [16.0, 17.0],
            "frames": 2,
        }
        stats = probe_tools.summarize(snapshot)
        evaluate = stats["stages"]["evaluate"]
        assert evaluate["count"] == 4
        assert evaluate["median_ms"] == 2.5
        assert evaluate["max_ms"] == 4.0
        assert evaluate["total_ms"] == 10.0
        assert stats["writes"]["bytes_total"] == 400
        assert stats["writes"]["calls_total"] == 6
        assert stats["frame"]["interval_median_ms"] == 16.5
        # Frame rate is derived from the median interval.
        assert 58.0 < stats["frame"]["fps_median"] < 62.0

    def test_known_stages_come_first(self):
        snapshot = {
            "stages": {
                "zzz_custom": {"durations": [1.0], "count": 1},
                "render": {"durations": [1.0], "count": 1},
                "evaluate": {"durations": [1.0], "count": 1},
            },
            "writes": {"calls": [], "bytes": []},
            "frameIntervals": [],
            "frames": 0,
        }
        names = list(probe_tools.summarize(snapshot)["stages"])
        assert names.index("evaluate") < names.index("render")
        assert names[-1] == "zzz_custom"

    def test_empty_snapshot_is_all_zeros(self):
        stats = probe_tools.summarize(
            {"stages": {}, "writes": {}, "frameIntervals": [], "frames": 0}
        )
        assert stats["stages"] == {}
        assert stats["writes"]["bytes_total"] == 0
        assert stats["frame"]["fps_median"] == 0.0

    def test_format_summary_mentions_each_stage(self):
        probe = {
            "param": "t",
            "range": [0.0, 1.0],
            "frames": 3,
            "measured_frames": 3,
            **probe_tools.summarize(
                {
                    "stages": {"evaluate": {"durations": [1.0], "count": 1}},
                    "writes": {"calls": [1], "bytes": [64]},
                    "frameIntervals": [16.0],
                    "frames": 1,
                }
            ),
        }
        text = "\n".join(probe_tools.format_summary(probe))
        assert "evaluate" in text
        assert "writeBuffer" in text
        assert "frame interval" in text


class TestProbeCli:
    """End-to-end: the flags produce a stats section on a real render."""

    def test_probe_reports_stages(self, project: pathlib.Path):
        _require_renderer()
        path = project / "sweep.py"
        path.write_text(SMALL_SWEEP)
        out = project / "shot.png"
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(path),
                "-o",
                str(out),
                "--probe-param",
                "t",
                "--probe-range",
                "0,1",
                "--probe-frames",
                "8",
                "--probe-warmup",
                "1",
                "--no-daemon",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert out.exists()

        probe = payload["probe"]
        assert probe["param"] == "t"
        assert probe["range"] == [0.0, 1.0]
        assert probe["frames"] == 8
        # Nearly every swept value produces a frame. A value equal to the one
        # already set is a genuine no-op for the renderer (nothing changed), so
        # allow a small shortfall rather than pinning the exact count.
        assert probe["measured_frames"] >= 6
        assert probe["frames_without_render"] <= 2

        stages = probe["stages"]
        # The three CPU stages on the state path must all have been observed.
        for name in ("evaluate", "compile", "render"):
            assert name in stages, f"missing stage {name}: {list(stages)}"
            assert stages[name]["occurrences"] >= 6
            # Sane values: non-negative, and nothing takes a whole second.
            assert 0.0 <= stages[name]["median_ms"] < 1000.0
            assert stages[name]["p95_ms"] >= stages[name]["median_ms"]

        # This scene is 4 vertices; the equality walk must be trivially cheap.
        assert stages["evaluate"]["median_ms"] < 50.0

        writes = probe["writes"]
        assert writes["frames"] >= 6
        # Rendering a mesh always writes at least camera uniforms.
        assert writes["calls_total"] > 0
        assert writes["bytes_total"] > 0

        frame = probe["frame"]
        assert frame["count"] >= 6
        assert frame["interval_median_ms"] > 0.0
        assert frame["budget_60fps_ms"] == pytest.approx(16.67)

    def test_probe_absent_without_flag(self, project: pathlib.Path):
        _require_renderer()
        path = project / "sweep.py"
        path.write_text(SMALL_SWEEP)
        out = project / "shot.png"
        result = CliRunner().invoke(
            cli_main,
            ["screenshot", str(path), "-o", str(out), "--no-daemon", "--json"],
        )
        assert result.exit_code == 0, result.output
        assert "probe" not in json.loads(result.output)

    def test_probe_rejects_views(self, project: pathlib.Path):
        _require_renderer()
        path = project / "sweep.py"
        path.write_text(SMALL_SWEEP)
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(path),
                "-o",
                str(project / "shot.png"),
                "--probe-param",
                "t",
                "--views",
                "front,top",
                "--no-daemon",
            ],
        )
        assert result.exit_code == 2
        assert "probe-param" in result.output

    def test_probe_rejects_bad_range(self, project: pathlib.Path):
        path = project / "sweep.py"
        path.write_text(SMALL_SWEEP)
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(path),
                "-o",
                str(project / "shot.png"),
                "--probe-param",
                "t",
                "--probe-range",
                "nope",
                "--no-daemon",
            ],
        )
        assert result.exit_code == 2
        assert "probe-range" in result.output
