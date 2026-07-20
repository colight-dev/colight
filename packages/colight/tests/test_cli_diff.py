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

    def test_added_mark_ahead_keeps_existing_state_key(self, tmp_path: pathlib.Path):
        """Inserting a stateful node must not shift existing canonical ids."""
        data = {"x": [1.0, 2.0], "y": [3.0, 4.0]}
        a = save_artifact(tmp_path, "a.colight", Plot.dot(data))
        b = save_artifact(
            tmp_path,
            "b.colight",
            Plot.line({"x": [0.0], "y": [9.0]}) + Plot.dot(data),
        )
        pair = diff_tools.diff_targets(a, b)["pairs"][0]
        # The untouched dot mark's state key is unchanged: only the new
        # line mark's state key is added.
        assert len(pair["state"]["added"]) == 1
        assert pair["state"]["removed"] == []
        assert pair["state"]["changed"] == []
        assert pair["arrays"]["changed"] == []
        assert [c["type"] for c in pair["components"]["added"]] == ["MarkSpec:line"]

    def test_added_ref_ahead_keeps_existing_state_key(self, tmp_path: pathlib.Path):
        a = save_artifact(
            tmp_path, "a.colight", Plot.html(["div", Plot.ref([1.0, 2.0, 3.0])])
        )
        b = save_artifact(
            tmp_path,
            "b.colight",
            Plot.html(["div", Plot.ref({"mode": "fast"}), Plot.ref([1.0, 2.0, 3.0])]),
        )
        pair = diff_tools.diff_targets(a, b)["pairs"][0]
        assert len(pair["state"]["added"]) == 1
        assert pair["state"]["removed"] == []
        assert pair["state"]["changed"] == []

    def test_edited_state_entry_keeps_its_key(self, tmp_path: pathlib.Path):
        """Editing an entry's data must keep its canonical id (pairing)."""
        a = dot_artifact(tmp_path, "a.colight", np.array([1.0, 2.0]))
        b = dot_artifact(tmp_path, "b.colight", np.array([1.0, 5.0]))
        pair = diff_tools.diff_targets(a, b)["pairs"][0]
        assert pair["state"]["added"] == []
        assert pair["state"]["removed"] == []
        assert len(pair["state"]["changed"]) == 1

    def test_removed_first_array_arg_reports_removed_only(self, tmp_path: pathlib.Path):
        """Sibling array args have distinct paths: no positional mispairing."""
        from colight.layout import JSCall

        arr1 = np.arange(6, dtype=np.float32)
        arr2 = np.arange(6, dtype=np.float32) * 10
        a = save_artifact(tmp_path, "a.colight", JSCall("View", [arr1, arr2]))
        b = save_artifact(tmp_path, "b.colight", JSCall("View", [None, arr2]))
        arrays = diff_tools.diff_targets(a, b)["pairs"][0]["arrays"]
        assert len(arrays["removed"]) == 1
        assert arrays["removed"][0]["path"].endswith("View[0]")
        # The surviving array is not mispaired against the removed one.
        assert arrays["changed"] == []
        assert arrays["added"] == []

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


class TestCategoricalArrays:
    """Integer-dtype arrays lead with changed count/fraction, demote |Δ|."""

    def test_integer_array_is_categorical(self):
        from colight.cli_tools.diff_tools import _numeric_delta

        a = np.array([0, 1, 2, 3], dtype=np.int64)
        b = np.array([0, 5, 2, 9], dtype=np.int64)
        stats = _numeric_delta(a, b, 1e-9, categorical=True)
        assert stats is not None
        # Change count/fraction present and lead the payload.
        assert stats["categorical"] is True
        assert stats["changed_count"] == 2
        assert stats["changed_fraction"] == pytest.approx(0.5)
        keys = list(stats)
        assert keys.index("changed_count") < keys.index("max_abs_delta")
        # Magnitude stats retained but demoted.
        assert stats["max_abs_delta"] == pytest.approx(6.0)

    def test_float_array_not_categorical(self):
        from colight.cli_tools.diff_tools import _numeric_delta

        a = np.array([0.0, 1.0], dtype=np.float64)
        b = np.array([0.0, 1.5], dtype=np.float64)
        stats = _numeric_delta(a, b, 1e-9)
        assert stats is not None
        assert "categorical" not in stats
        assert stats["changed_count"] == 1

    def test_integer_artifact_diff_reports_categorical(self, tmp_path: pathlib.Path):
        a = save_artifact(
            tmp_path,
            "a.colight",
            Plot.State({"grid": np.array([0, 1, 2, 3], dtype=np.int64)}),
        )
        b = save_artifact(
            tmp_path,
            "b.colight",
            Plot.State({"grid": np.array([0, 5, 2, 3], dtype=np.int64)}),
        )
        payload = diff_tools.diff_targets(a, b)
        changed = payload["pairs"][0]["arrays"]["changed"]
        entries = [e for e in changed if e["path"].endswith("grid")]
        assert len(entries) == 1
        assert entries[0]["categorical"] is True
        assert entries[0]["changed_count"] == 1


def _arc_fixtures() -> pathlib.Path:
    from notebooks.arc_agi import trajectory as arc

    return pathlib.Path(arc.__file__).parent / "fixtures"


def _build_arc_artifacts(tmp_path: pathlib.Path):
    """Build the two committed ls20 run artifacts (real multi-update files)."""
    from notebooks.arc_agi.trajectory import build_artifact, load_fixture

    fixtures = _arc_fixtures()
    run_a = load_fixture(fixtures / "ls20-run-a.json.gz")
    run_b = load_fixture(fixtures / "ls20-run-b.json.gz")
    a = build_artifact(run_a, tmp_path / "ls20-run-a.colight")
    b = build_artifact(run_b, tmp_path / "ls20-run-b.colight")
    return a, b


class TestUpdateEntryDiff:
    """Aligned per-step diffing of .colight artifacts with update entries."""

    def test_arc_fixture_runs_diverge(self, tmp_path: pathlib.Path):
        """Acceptance: two REAL fixture-built runs must diff sensibly."""
        a, b = _build_arc_artifacts(tmp_path)
        payload = diff_tools.diff_targets(a, b)

        # The study's bug: identical:true for behaviorally different runs.
        assert payload["identical"] is False

        updates = payload["updates"]
        # 81 steps -> 80 update entries per run.
        assert updates["count"] == [80, 80]
        assert updates["aligned"] == 80
        # Both runs share the RESET initial frame; they diverge from step 1
        # (update index 0), where actions differ (ACTION2 vs ACTION1).
        assert payload["pairs"][0]["identical"] is True
        assert updates["first_diverging_update"] == 0
        assert updates["updates_differing"] == 80

        summary = payload["summary"]
        assert summary["first_diverging_update"] == 0
        assert summary["updates_differing"] == 80

        # First differing step leads with a categorical pixel-grid change and
        # a state action change - not a hollow magnitude stat.
        step0 = updates["steps"][0]
        assert step0["index"] == 0
        pixel_entries = [
            e for e in step0["arrays"]["changed"] if e["path"].endswith("pixels")
        ]
        assert pixel_entries and pixel_entries[0]["categorical"] is True
        assert pixel_entries[0]["changed_count"] > 0
        assert "action" in step0["state"]["changed"]

    def test_identical_runs_report_identical(self, tmp_path: pathlib.Path):
        from notebooks.arc_agi.trajectory import build_artifact, load_fixture

        run = load_fixture(_arc_fixtures() / "ls20-run-a.json.gz")
        a = build_artifact(run, tmp_path / "a.colight")
        b = build_artifact(run, tmp_path / "b.colight")
        payload = diff_tools.diff_targets(a, b)
        assert payload["identical"] is True
        assert payload["updates"]["updates_differing"] == 0
        assert payload["updates"]["first_diverging_update"] is None

    def test_trailing_length_mismatch_reported(self, tmp_path: pathlib.Path):
        from colight import format as colight_format

        base = save_artifact(tmp_path, "a.colight", Plot.State({"step": 0}))
        b = tmp_path / "b.colight"
        b.write_bytes(base.read_bytes())
        # A has 2 updates, B has 4 - identical on the shared prefix.
        colight_format.append_updates(base, [{"step": 1}, {"step": 2}])
        colight_format.append_updates(
            b, [{"step": 1}, {"step": 2}, {"step": 3}, {"step": 4}]
        )

        payload = diff_tools.diff_targets(base, b)
        assert payload["identical"] is False
        updates = payload["updates"]
        assert updates["aligned"] == 2
        assert updates["updates_differing"] == 0
        assert updates["first_diverging_update"] is None
        assert updates["trailing"] == {"side": "b", "from": 2, "count": 2}

    def test_verdict_line_reports_first_divergence(self, tmp_path: pathlib.Path):
        a, b = _build_arc_artifacts(tmp_path)
        payload = diff_tools.diff_targets(a, b)
        line = diff_tools.verdict_line(payload)
        assert "first divergence at update 0" in line
        assert "80/80 updates differ" in line

    def test_cli_human_output_and_exit(self, tmp_path: pathlib.Path):
        a, b = _build_arc_artifacts(tmp_path)
        result = CliRunner().invoke(cli_main, ["diff", str(a), str(b)])
        assert result.exit_code == 1
        assert "updates:" in result.output
        assert "first divergence at update 0" in result.output
        # Per-step listing leads with cell counts for the categorical grid.
        assert "cells" in result.output


class TestLeafAggregation:
    """Nested-JSON-list changes collapse into one array-style entry."""

    def make_pair(self, tmp_path: pathlib.Path):
        points_a = [[i * 0.1, float(i % 7)] for i in range(40)]
        points_b = [
            [x, y + (0.5 if i % 2 == 0 else 0.0)] for i, (x, y) in enumerate(points_a)
        ]
        a = save_artifact(tmp_path, "a.colight", Plot.line(points_a))
        b = save_artifact(tmp_path, "b.colight", Plot.line(points_b))
        return a, b

    def test_many_leaf_changes_aggregate(self, tmp_path: pathlib.Path):
        a, b = self.make_pair(tmp_path)
        payload = diff_tools.diff_targets(a, b)
        pair = payload["pairs"][0]
        # One wildcarded entry instead of 20 itemized value lines; the
        # constant column index stays literal.
        entries = [e for e in pair["arrays"]["changed"] if "[*]" in e["path"]]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["path"].endswith("[*][1]")
        assert entry["leaves"] == {"changed": 20, "total": 40}
        assert entry["changed_fraction"] == pytest.approx(0.5)
        assert entry["max_abs_delta"] == pytest.approx(0.5)
        assert entry["mean_abs_delta"] == pytest.approx(0.25)
        assert entry["bounds"]["from"] == [0.0, 6.0]
        assert entry["bounds"]["to"] == [0.0, 6.5]
        # Consumed leaves are not itemized, and the summary sees the group.
        assert pair["values"]["changed"] == []
        assert payload["summary"]["arrays_changed"] == 1
        assert payload["summary"]["max_abs_delta_path"] == entry["path"]

    def test_small_groups_stay_itemized(self, tmp_path: pathlib.Path):
        points_a = [[float(i), float(i)] for i in range(10)]
        points_b = [row[:] for row in points_a]
        points_b[3][1] = 9.0
        points_b[7][1] = 9.0  # two changes < AGGREGATE_MIN_CHANGED
        a = save_artifact(tmp_path, "a.colight", Plot.line(points_a))
        b = save_artifact(tmp_path, "b.colight", Plot.line(points_b))
        pair = diff_tools.diff_targets(a, b)["pairs"][0]
        assert not any("[*]" in e["path"] for e in pair["arrays"]["changed"])
        assert len(pair["values"]["changed"]) == 2

    def test_itemized_values_are_capped(self, tmp_path: pathlib.Path):
        count = diff_tools.MAX_ITEMIZED_VALUES + 10
        a = save_artifact(
            tmp_path,
            "a.colight",
            Plot.State({f"k{i:03d}": f"a{i}" for i in range(count)}),
        )
        b = save_artifact(
            tmp_path,
            "b.colight",
            Plot.State({f"k{i:03d}": f"b{i}" for i in range(count)}),
        )
        payload = diff_tools.diff_targets(a, b)
        values = payload["pairs"][0]["values"]
        assert len(values["changed"]) == diff_tools.MAX_ITEMIZED_VALUES
        assert values["truncated"] == {"changed": 10}

        result = CliRunner().invoke(cli_main, ["diff", str(a), str(b)])
        assert result.exit_code == 1
        itemized = [ln for ln in result.output.splitlines() if "  value " in ln]
        assert len(itemized) == 5
        assert f"… {count - 5} more value change(s)" in result.output
        # Verdict counts the full number of changes, not the capped list.
        assert f"{count} value change(s)" in result.output


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
