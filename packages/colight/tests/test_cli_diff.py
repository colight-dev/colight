"""Tests for the semantic artifact diff: colight diff."""

import json
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner

import colight.plot as Plot
from colight.cli_tools import diff_tools
from colight.inspect import inspect as colight_inspect
from colight_cli import main as cli_main


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    """A temp project dir anchored as a project root."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return tmp_path


def save_artifact(tmp_path: pathlib.Path, name: str, plot) -> pathlib.Path:
    visual = colight_inspect(plot)
    assert visual is not None
    target = tmp_path / name
    target.write_bytes(visual.to_bytes())
    return target


def dot_artifact(tmp_path: pathlib.Path, name: str, ys: np.ndarray) -> pathlib.Path:
    xs = np.arange(len(ys), dtype=np.float64)
    return save_artifact(tmp_path, name, Plot.dot({"x": xs, "y": ys}))


class TestArtifactDiff:
    def test_magnitude_stats(self, tmp_path: pathlib.Path):
        ys_a = np.array([0.0, 1.0, 2.0, 3.0])
        ys_b = np.array([0.0, 1.5, 2.0, 3.1])
        a = dot_artifact(tmp_path, "a.colight", ys_a)
        b = dot_artifact(tmp_path, "b.colight", ys_b)

        payload = diff_tools.diff_targets(a, b)
        assert payload["identical"] is False
        changed = payload["pairs"][0]["arrays"]["changed"]
        entries = [e for e in changed if e["path"].endswith(".y")]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["max_abs_delta"] == pytest.approx(0.5)
        assert entry["mean_abs_delta"] == pytest.approx(0.6 / 4)
        assert entry["changed_fraction"] == pytest.approx(0.5)
        assert entry["bounds"] == {"from": [0.0, 3.0], "to": [0.0, 3.1]}
        # Unchanged x array is not reported.
        assert not any(e["path"].endswith(".x") for e in changed)

        summary = payload["summary"]
        assert summary["arrays_changed"] == 1
        assert summary["max_abs_delta"] == pytest.approx(0.5)
        assert summary["max_abs_delta_path"] == entry["path"]

    def test_identical_artifacts(self, tmp_path: pathlib.Path):
        ys = np.array([1.0, 2.0])
        a = dot_artifact(tmp_path, "a.colight", ys)
        b = dot_artifact(tmp_path, "b.colight", ys)
        payload = diff_tools.diff_targets(a, b)
        assert payload["identical"] is True
        assert payload["pairs"][0]["identical"] is True
        assert payload["summary"]["arrays_changed"] == 0

    def test_shape_and_dtype_change(self, tmp_path: pathlib.Path):
        a = dot_artifact(tmp_path, "a.colight", np.array([1.0, 2.0]))
        b = dot_artifact(tmp_path, "b.colight", np.array([1, 2, 3], dtype=np.int64))
        payload = diff_tools.diff_targets(a, b)
        entries = [
            e
            for e in payload["pairs"][0]["arrays"]["changed"]
            if e["path"].endswith(".y")
        ]
        assert entries[0]["shape"] == [[2], [3]]
        assert entries[0]["dtype"] == ["float64", "int64"]
        # No magnitude stats for shape-mismatched arrays.
        assert "max_abs_delta" not in entries[0]

    def test_component_type_change(self, tmp_path: pathlib.Path):
        data = {"x": [1.0, 2.0], "y": [3.0, 4.0]}
        a = save_artifact(tmp_path, "a.colight", Plot.dot(data))
        b = save_artifact(tmp_path, "b.colight", Plot.line(data))
        payload = diff_tools.diff_targets(a, b)
        changed = payload["pairs"][0]["components"]["changed"]
        assert changed[0]["from"] == "MarkSpec:dot"
        assert changed[0]["to"] == "MarkSpec:line"

    def test_warning_delta(self, tmp_path: pathlib.Path):
        ys_good = np.array([1.0, 2.0, 3.0])
        ys_nan = np.array([1.0, np.nan, 3.0])
        a = dot_artifact(tmp_path, "a.colight", ys_good)
        b = dot_artifact(tmp_path, "b.colight", ys_nan)
        payload = diff_tools.diff_targets(a, b)
        warnings = payload["pairs"][0]["warnings"]
        assert any(w["code"] == "nan-values" for w in warnings["introduced"])
        assert warnings["resolved"] == []
        # Reverse direction resolves it.
        reverse = diff_tools.diff_targets(b, a)
        assert any(
            w["code"] == "nan-values"
            for w in reverse["pairs"][0]["warnings"]["resolved"]
        )

    def test_state_key_changes(self, tmp_path: pathlib.Path):
        a = save_artifact(
            tmp_path, "a.colight", Plot.State({"n": 1, "gone": 2}) | Plot.dot([1, 2])
        )
        b = save_artifact(
            tmp_path, "b.colight", Plot.State({"n": 5, "fresh": 3}) | Plot.dot([1, 2])
        )
        state = diff_tools.diff_targets(a, b)["pairs"][0]["state"]
        assert "fresh" in state["added"]
        assert "gone" in state["removed"]
        assert "n" in state["changed"]


class TestNumericDelta:
    def test_equal_infinities_and_nans_are_unchanged(self):
        from colight.cli_tools.diff_tools import _numeric_delta

        inf = np.array([np.inf, 1.0])
        assert _numeric_delta(inf, inf.copy(), 1e-9) is None
        nan = np.array([np.nan, 1.0])
        assert _numeric_delta(nan, nan.copy(), 1e-9) is None

    def test_nan_mismatch_counts_as_changed(self):
        from colight.cli_tools.diff_tools import _numeric_delta

        stats = _numeric_delta(np.array([1.0, np.nan]), np.array([1.0, 2.0]), 1e-9)
        assert stats is not None
        assert stats["nan_mismatch"] == 1
        assert stats["changed_fraction"] == pytest.approx(0.5)


class TestEpsilon:
    def test_below_epsilon_is_identical(self, tmp_path: pathlib.Path):
        a = dot_artifact(tmp_path, "a.colight", np.array([1.0, 2.0]))
        b = dot_artifact(tmp_path, "b.colight", np.array([1.0, 2.0 + 1e-12]))
        assert diff_tools.diff_targets(a, b, epsilon=1e-9)["identical"] is True
        assert diff_tools.diff_targets(a, b, epsilon=1e-15)["identical"] is False

    def test_epsilon_applies_to_scalar_leaves(self, tmp_path: pathlib.Path):
        a = save_artifact(tmp_path, "a.colight", Plot.State({"k": 1.0}))
        b = save_artifact(tmp_path, "b.colight", Plot.State({"k": 1.0 + 1e-12}))
        assert diff_tools.diff_targets(a, b, epsilon=1e-9)["identical"] is True
        assert diff_tools.diff_targets(a, b, epsilon=0.0)["identical"] is False


class TestPythonTargets:
    SOURCE = (
        "import colight.plot as Plot\n"
        "import numpy as np\n\n"
        "xs = np.linspace(0, 1, 20)\n\n"
        "Plot.dot({'x': xs, 'y': np.sin(xs)})\n"
    )

    def test_py_vs_edited_py(self, project: pathlib.Path):
        a = project / "a.py"
        b = project / "b.py"
        a.write_text(self.SOURCE)
        b.write_text(self.SOURCE.replace("np.sin(xs)", "np.sin(xs) * 2"))

        payload = diff_tools.diff_targets(a, b)
        assert payload["a"]["kind"] == "py"
        assert payload["identical"] is False
        pair = payload["pairs"][0]
        assert "a_block" in pair and "b_block" in pair
        entries = [e for e in pair["arrays"]["changed"] if e["path"].endswith(".y")]
        assert entries[0]["max_abs_delta"] == pytest.approx(float(np.sin(1.0)))

    def test_py_self_diff_identical_despite_regenerated_ids(
        self, project: pathlib.Path
    ):
        path = project / "nb.py"
        path.write_text(self.SOURCE)
        # Two separate evaluations regenerate widget/state uuids.
        assert diff_tools.diff_targets(path, path)["identical"] is True

    def test_visual_count_mismatch(self, project: pathlib.Path):
        a = project / "a.py"
        b = project / "b.py"
        a.write_text(self.SOURCE)
        b.write_text(self.SOURCE + "\nPlot.dot({'x': [1], 'y': [2]})\n")
        payload = diff_tools.diff_targets(a, b)
        assert payload["identical"] is False
        assert len(payload["unpaired"]["b"]) == 1
        assert payload["unpaired"]["a"] == []

    def test_evaluation_errors_reported(self, project: pathlib.Path):
        a = project / "a.py"
        b = project / "b.py"
        a.write_text(self.SOURCE)
        b.write_text(self.SOURCE + "\nboom = 1 / 0\n")
        payload = diff_tools.diff_targets(a, b)
        assert payload["identical"] is False
        errors = payload["b"]["errors"]
        assert errors[0]["error"]["type"] == "ZeroDivisionError"


class TestCli:
    def test_exit_codes(self, tmp_path: pathlib.Path):
        a = dot_artifact(tmp_path, "a.colight", np.array([1.0, 2.0]))
        b = dot_artifact(tmp_path, "b.colight", np.array([1.0, 9.0]))
        runner = CliRunner()

        identical = runner.invoke(cli_main, ["diff", str(a), str(a)])
        assert identical.exit_code == 0
        assert "identical" in identical.output

        different = runner.invoke(cli_main, ["diff", str(a), str(b)])
        assert different.exit_code == 1
        assert "max |Δ|" in different.output

        bad = tmp_path / "bad.txt"
        bad.write_text("nope")
        error = runner.invoke(cli_main, ["diff", str(a), str(bad)])
        assert error.exit_code == 2

    def test_json_output(self, tmp_path: pathlib.Path):
        a = dot_artifact(tmp_path, "a.colight", np.array([1.0, 2.0]))
        b = dot_artifact(tmp_path, "b.colight", np.array([1.0, 3.0]))
        result = CliRunner().invoke(cli_main, ["diff", str(a), str(b), "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert {"a", "b", "epsilon", "identical", "pairs", "summary"} <= set(payload)
        assert payload["identical"] is False

    def test_epsilon_option(self, tmp_path: pathlib.Path):
        a = dot_artifact(tmp_path, "a.colight", np.array([1.0]))
        b = dot_artifact(tmp_path, "b.colight", np.array([1.0001]))
        runner = CliRunner()
        assert (
            runner.invoke(cli_main, ["diff", str(a), str(b), "--epsilon", "1e-3"])
        ).exit_code == 0
        assert (
            runner.invoke(cli_main, ["diff", str(a), str(b), "--epsilon", "1e-6"])
        ).exit_code == 1
