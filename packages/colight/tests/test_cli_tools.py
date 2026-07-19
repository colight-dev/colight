"""Tests for the agent-facing CLI: colight blocks / run / inspect."""

import json
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner

import colight.scene3d as scene3d
from colight.cli_tools import blocks as blocks_mod
from colight.cli_tools import inspect_tools
from colight.cli_tools import run as run_mod
from colight.inspect import inspect as colight_inspect
from colight_cli import main as cli_main

FIXTURE = """\
# # Title prose
# Some description.

x = 1

y = x + 1
y

y * 2

z = 42
z
"""


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    """A temp project dir anchored as a project root."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return tmp_path


def write_fixture(project: pathlib.Path, source: str = FIXTURE) -> pathlib.Path:
    path = project / "nb.py"
    path.write_text(source, encoding="utf-8")
    return path


def statuses(payload: dict) -> dict:
    return {b["id"]: b["status"] for b in payload["blocks"]}


class TestBlocks:
    def test_block_graph(self, project: pathlib.Path):
        path = write_fixture(project)
        payload = blocks_mod.describe_file(path)

        assert payload["file"] == str(path)
        blocks = payload["blocks"]
        assert len(blocks) == 5

        prose, bx, by, bexpr, bz = blocks
        assert prose["kind"] == "prose"
        assert prose["lines"] == [1, 2]
        assert bx["kind"] == "code"
        assert bx["provides"] == ["x"]
        assert bx["lines"] == [4, 4]
        assert by["provides"] == ["y"]
        assert by["requires"] == ["x"]
        assert by["depends_on"] == [bx["id"]]
        assert by["lines"] == [6, 7]
        assert by["ends_with_expression"] is True
        assert bexpr["requires"] == ["y"]
        assert bexpr["depends_on"] == [by["id"]]
        assert by["dependents"] == [bexpr["id"]]
        assert bz["depends_on"] == []
        assert bz["provides"] == ["z"]

    def test_stable_ids_survive_edits_to_other_blocks(self, project: pathlib.Path):
        path = write_fixture(project)
        before = {b["id"] for b in blocks_mod.describe_file(path)["blocks"]}

        write_fixture(project, FIXTURE.replace("x = 1", "x = 100"))
        after = blocks_mod.describe_file(path)["blocks"]

        # Only the edited block's id changed; all others survive, including
        # the dependent `y` block (unlike the runtime's transitive cache key).
        changed = [b["id"] for b in after if b["id"] not in before]
        assert len(changed) == 1
        assert [b for b in after if b["provides"] == ["x"]][0]["id"] == changed[0]

    def test_duplicate_blocks_get_distinct_ids(self, project: pathlib.Path):
        path = write_fixture(project, "a = 1\n\nprint(a)\n\nprint(a)\n")
        ids = [b["id"] for b in blocks_mod.describe_file(path)["blocks"]]
        assert len(set(ids)) == 3
        assert ids[2] == f"{ids[1]}-2"

    def test_cli_json(self, project: pathlib.Path):
        path = write_fixture(project)
        result = CliRunner().invoke(cli_main, ["blocks", str(path), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert {"file", "pragma", "blocks"} <= set(payload.keys())

    def test_pragma_tags_reported(self, project: pathlib.Path):
        path = write_fixture(project, "# | hide-code\nx = 1\n")
        payload = blocks_mod.describe_file(path)
        assert payload["blocks"][0]["pragma"] == ["hide-code"]


class TestRun:
    def test_first_run_all_new(self, project: pathlib.Path):
        path = write_fixture(project)
        payload = run_mod.run_file(path)
        assert payload["ok"] is True
        assert set(statuses(payload).values()) == {"new"}

    def test_second_run_all_cached_nothing_executes(self, project: pathlib.Path):
        path = write_fixture(project)
        run_mod.run_file(path)
        payload = run_mod.run_file(path)
        assert set(statuses(payload).values()) == {"cached"}
        assert all(b["executed"] is False for b in payload["blocks"])

    def test_edit_reports_exact_statuses(self, project: pathlib.Path):
        path = write_fixture(project)
        first = run_mod.run_file(path)
        by_provides = {
            tuple(info["provides"]): info["id"]
            for info in blocks_mod.describe_file(path)["blocks"]
        }

        # Edit the y block: its own result and its dependent's result change.
        write_fixture(project, FIXTURE.replace("y = x + 1", "y = x + 10"))
        payload = run_mod.run_file(path)
        status = statuses(payload)
        blocks = {b["id"]: b for b in payload["blocks"]}

        # x re-executes (the edited block needs it) but its result is unchanged.
        assert status[by_provides[("x",)]] == "ran:unchanged"
        # Edited block keeps its identity via positional pairing: ran:changed.
        old_y_id = by_provides[("y",)]
        new_y = [b for b in payload["blocks"] if b["status"] == "ran:changed"]
        assert len(new_y) == 2  # edited block + its dependent
        assert old_y_id not in status  # id changed with the source
        # Dependent expression block re-ran and changed.
        dependent = [
            info
            for info in blocks_mod.describe_file(path)["blocks"]
            if info["requires"] == ["y"] and not info["provides"]
        ][0]
        assert status[dependent["id"]] == "ran:changed"
        # Untouched independent block did not re-run.
        z_id = by_provides[("z",)]
        assert status[z_id] == "cached"
        assert blocks[z_id]["executed"] is False
        # Prose block stays cached.
        prose_id = first["blocks"][0]["id"]
        assert status[prose_id] == "cached"
        # No spurious new/removed entries.
        assert "new" not in status.values()
        assert "removed" not in status.values()

    def test_removed_block_reported(self, project: pathlib.Path):
        path = write_fixture(project)
        run_mod.run_file(path)
        write_fixture(project, FIXTURE.replace("z = 42\nz\n", ""))
        payload = run_mod.run_file(path)
        assert "removed" in statuses(payload).values()

    def test_error_block_structured_and_exit_code(self, project: pathlib.Path):
        path = write_fixture(project, "a = 1\n\nboom = 1 / 0\n")
        payload = run_mod.run_file(path)
        assert payload["ok"] is False
        error_block = [b for b in payload["blocks"] if b["status"] == "error"][0]
        error = error_block["error"]
        assert error["type"] == "ZeroDivisionError"
        assert "division by zero" in error["message"]
        assert error["frames"][0]["file"] == str(path)
        assert error["frames"][0]["line"] == 3
        assert error["frames"][0]["code"] == "boom = 1 / 0"

        result = CliRunner().invoke(cli_main, ["run", str(path)])
        assert result.exit_code == 1

    def test_errors_always_rerun(self, project: pathlib.Path):
        path = write_fixture(project, "boom = 1 / 0\n")
        run_mod.run_file(path)
        payload = run_mod.run_file(path)  # unchanged source, cache key matches
        assert statuses(payload) != {}
        assert set(statuses(payload).values()) == {"error"}
        assert payload["ok"] is False

    def test_always_eval_pragma_never_cached(self, project: pathlib.Path):
        source = (
            "x = 1\n"
            "\n"
            "# | pragma: always-eval\n"
            "import random\n"
            "value = random.random()\n"
            "value\n"
        )
        path = write_fixture(project, source)
        run_mod.run_file(path)
        payload = run_mod.run_file(path)  # unchanged source
        status = statuses(payload)
        blocks = {b["id"]: b for b in payload["blocks"]}
        always_id = [
            info["id"]
            for info in blocks_mod.describe_file(path)["blocks"]
            if "always-eval" in info["pragma"]
        ][0]
        # The always-eval block must never report cached.
        assert status[always_id] in ("ran:unchanged", "ran:changed")
        assert blocks[always_id]["executed"] is True
        # The untouched block stays cached.
        other = [sid for sid in status if sid != always_id][0]
        assert status[other] == "cached"

    def test_force_reexecutes_everything(self, project: pathlib.Path):
        path = write_fixture(project)
        run_mod.run_file(path)
        payload = run_mod.run_file(path, force=True)
        assert all(b["executed"] is True for b in payload["blocks"])
        # Same source, same results: statuses compare against stored
        # fingerprints, so everything reports ran:unchanged.
        assert set(statuses(payload).values()) == {"ran:unchanged"}
        # A forced run after an edit reports the change.
        write_fixture(project, FIXTURE.replace("z = 42", "z = 43"))
        payload = run_mod.run_file(path, force=True)
        assert "ran:changed" in statuses(payload).values()
        assert "cached" not in statuses(payload).values()

    def test_force_cli_flag(self, project: pathlib.Path):
        path = write_fixture(project)
        CliRunner().invoke(cli_main, ["run", str(path)])
        result = CliRunner().invoke(cli_main, ["run", str(path), "--force", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert all(b["executed"] is True for b in payload["blocks"])

    def test_focus_block_restricts_detail(self, project: pathlib.Path):
        path = write_fixture(project)
        infos = blocks_mod.describe_file(path)["blocks"]
        y_id = [i["id"] for i in infos if i["provides"] == ["y"]][0]
        dependent_id = [i["id"] for i in infos if i["requires"] == ["y"]][0]
        z_id = [i["id"] for i in infos if i["provides"] == ["z"]][0]

        payload = run_mod.run_file(path, focus_block=y_id)
        blocks = {b["id"]: b for b in payload["blocks"]}
        assert "summary" in blocks[y_id]
        assert "summary" in blocks[dependent_id]
        assert "summary" not in blocks[z_id]

    def test_focus_block_unknown_id(self, project: pathlib.Path):
        path = write_fixture(project)
        with pytest.raises(ValueError, match="Unknown block id"):
            run_mod.run_file(path, focus_block="nope")

    def test_record_location_gitignored(self, project: pathlib.Path):
        path = write_fixture(project)
        run_mod.run_file(path)
        cache_dir = project / ".colight_cache"
        assert (cache_dir / ".gitignore").read_text().strip() == "*"
        assert list((cache_dir / "cli-run").glob("nb-*.json"))

    def test_visual_summary_is_token_frugal(self, project: pathlib.Path):
        path = write_fixture(
            project,
            "import colight.plot as Plot\n\n"
            "Plot.dot({'x': [1, 2, 3], 'y': [4, 5, 6]})\n",
        )
        payload = run_mod.run_file(path)
        summary = payload["blocks"][-1]["summary"]
        assert summary["kind"] == "visual"
        paths = {c["path"] for c in summary["components"]}
        assert "MarkSpec:dot" in paths
        # Never serialize full data into summaries.
        assert "[4, 5, 6]" not in json.dumps(summary)


class TestInspect:
    def make_broken_scene(self, tmp_path: pathlib.Path) -> pathlib.Path:
        centers = np.zeros((10, 3), dtype=np.float32)
        centers[3] = np.nan
        scene = scene3d.PointCloud(
            centers=centers,
            colors=np.empty((0, 3), dtype=np.float32),
            alphas=np.zeros(8, dtype=np.float32),
            size=0.1,
        )
        visual = colight_inspect(scene)
        assert visual is not None
        target = tmp_path / "broken.colight"
        target.write_bytes(visual.to_bytes())
        return target

    def test_inspect_colight_warnings(self, tmp_path: pathlib.Path):
        target = self.make_broken_scene(tmp_path)
        payload = inspect_tools.inspect_colight_file(target)

        assert payload["kind"] == "colight"
        codes = {w["code"] for w in payload["warnings"]}
        assert {
            "nan-values",
            "empty-array",
            "alphas-zero",
            "length-mismatch",
            "degenerate-bounds",
        } <= codes

        visual = payload["visual"]
        component_paths = {c["path"] for c in visual["components"]}
        assert "scene3d.PointCloud" in component_paths
        centers = [a for a in visual["arrays"] if a["path"].endswith("centers")][0]
        assert centers["dtype"] == "float32"
        assert centers["nan"] == 3

    def test_inspect_cli_json(self, tmp_path: pathlib.Path):
        target = self.make_broken_scene(tmp_path)
        result = CliRunner().invoke(cli_main, ["inspect", str(target), "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["kind"] == "colight"
        assert payload["warnings"]

    def test_inspect_python_file(self, project: pathlib.Path):
        path = write_fixture(
            project,
            "import colight.plot as Plot\n\n"
            "Plot.dot({'x': [1, 2, 3], 'y': [4, 5, 6]})\n",
        )
        payload = inspect_tools.inspect_python_file(path)
        assert payload["kind"] == "py"
        assert len(payload["visuals"]) == 1
        visual = payload["visuals"][0]["visual"]
        assert {c["path"] for c in visual["components"]} >= {"MarkSpec:dot"}
        xs = [a for a in visual["arrays"] if a["path"].endswith(".x")]
        assert xs and xs[0]["min"] == 1 and xs[0]["max"] == 3

    def test_inspect_unsupported_target(self, tmp_path: pathlib.Path):
        target = tmp_path / "foo.txt"
        target.write_text("hi")
        with pytest.raises(ValueError, match="Unsupported target"):
            inspect_tools.inspect_target(target)


class TestFingerprints:
    def test_visual_fingerprint_ignores_volatile_ids(self):
        import colight.plot as Plot
        from colight.cli_tools import summaries

        def make_bytes() -> bytes:
            visual = colight_inspect(Plot.dot({"x": [1, 2], "y": [3, 4]}))
            assert visual is not None
            return visual.to_bytes()

        # Two evaluations regenerate widget/state uuids; fingerprints match.
        assert summaries.visual_fingerprint(make_bytes()) == (
            summaries.visual_fingerprint(make_bytes())
        )

    def test_value_summaries(self):
        from colight.cli_tools import summaries

        array_summary = summaries.summarize_value(np.arange(6, dtype=np.float32))
        assert array_summary == {
            "kind": "array",
            "dtype": "float32",
            "shape": [6],
            "min": 0.0,
            "max": 5.0,
        }
        scalar = summaries.summarize_value("a" * 500)
        assert scalar["kind"] == "scalar"
        assert len(scalar["repr"]) <= summaries.MAX_REPR
