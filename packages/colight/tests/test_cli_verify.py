"""Tests for golden verification: colight verify."""

import json
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner

import colight.env as env
import colight.plot as Plot
from colight.chrome_devtools import find_chrome
from colight.cli_tools import verify_tools
from colight.inspect import inspect as colight_inspect
from colight_cli import main as cli_main

NB_TEMPLATE = """\
import colight.plot as Plot

xs = list(range(10))
ys = [float(x) * {scale} for x in xs]
Plot.dot({{'x': xs, 'y': ys}})

Plot.line({{'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 4.0]}})
"""


def _require_renderer() -> None:
    widget_path = env.WIDGET_PATH
    if not (isinstance(widget_path, pathlib.Path) and widget_path.exists()):
        pytest.skip("colight JS bundle not built (js-dist missing)")
    try:
        find_chrome()
    except FileNotFoundError:
        pytest.skip("Chrome not found for pixel tests")


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return tmp_path


def write_nb(project: pathlib.Path, scale: float = 1.0) -> pathlib.Path:
    path = project / "nb.py"
    path.write_text(NB_TEMPLATE.format(scale=scale), encoding="utf-8")
    return path


def save_artifact(tmp_path: pathlib.Path, name: str, plot) -> pathlib.Path:
    visual = colight_inspect(plot)
    assert visual is not None
    target = tmp_path / name
    target.write_bytes(visual.to_bytes())
    return target


def block_statuses(result: dict) -> dict:
    return {b["id"]: b["status"] for b in result["blocks"]}


class TestGoldenLayout:
    def test_default_goldens_root_is_project_tests_goldens(self, project: pathlib.Path):
        path = write_nb(project)
        assert verify_tools.default_goldens_root(path) == (
            project / "tests" / "goldens"
        )
        directory = verify_tools.golden_dir(path, project / "tests" / "goldens")
        assert directory == project / "tests" / "goldens" / "nb.py"

    def test_update_writes_manifest_and_artifacts(self, project: pathlib.Path):
        path = write_nb(project)
        result = verify_tools.update_target(path, pixels=False)
        assert result["status"] == "updated"
        assert set(block_statuses(result).values()) == {"added"}

        directory = project / "tests" / "goldens" / "nb.py"
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["version"] == 1
        assert manifest["kind"] == "py"
        assert len(manifest["blocks"]) == 2
        for entry in manifest["blocks"]:
            assert (directory / entry["artifact"]).exists()
            assert entry["structure_hash"]
            assert "screenshot" not in entry  # pixels disabled


class TestVerify:
    def test_no_goldens_exit_code_and_hint(self, project: pathlib.Path):
        path = write_nb(project)
        payload, exit_code = verify_tools.run_verify([path], pixels=False)
        assert exit_code == 3
        result = payload["targets"][0]
        assert result["status"] == "no-goldens"
        assert "--update" in result["hint"]

    def test_match_after_update(self, project: pathlib.Path):
        path = write_nb(project)
        verify_tools.update_target(path, pixels=False)
        # Re-evaluation regenerates widget/state uuids; canonicalization
        # keeps the structure hashes stable.
        payload, exit_code = verify_tools.run_verify([path], pixels=False)
        assert exit_code == 0
        result = payload["targets"][0]
        assert result["status"] == "match"
        assert set(block_statuses(result).values()) == {"match"}

    def test_structure_mismatch_reports_semantic_diff(self, project: pathlib.Path):
        path = write_nb(project)
        verify_tools.update_target(path, pixels=False)
        write_nb(project, scale=3.0)  # edit the dot block's data

        payload, exit_code = verify_tools.run_verify([path], pixels=False)
        assert exit_code == 1
        result = payload["targets"][0]
        assert result["status"] == "mismatch"
        changed = [b for b in result["blocks"] if b["status"] == "structure-changed"]
        assert len(changed) == 1
        block = changed[0]
        # The edited block's id changed with its source: positional pairing
        # still matches it to the old golden for a semantic diff.
        assert "golden_id" in block
        assert block["structure"]["match"] is False
        diff = block["diff"]
        # ys went from x*1 to x*3: max |delta| = 9 * 2 = 18.
        assert diff["max_abs_delta"] == pytest.approx(18.0)
        assert diff["changed_paths"]
        # The untouched line block still matches.
        untouched = [b for b in result["blocks"] if b["status"] == "match"]
        assert len(untouched) == 1

    def test_removed_visual_reported(self, project: pathlib.Path):
        path = write_nb(project)
        verify_tools.update_target(path, pixels=False)
        path.write_text(
            "import colight.plot as Plot\n\n"
            "Plot.line({'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 4.0]})\n",
            encoding="utf-8",
        )
        payload, exit_code = verify_tools.run_verify([path], pixels=False)
        assert exit_code == 1
        statuses = set(block_statuses(payload["targets"][0]).values())
        assert "removed" in statuses

    def test_new_visual_reported(self, project: pathlib.Path):
        path = write_nb(project)
        verify_tools.update_target(path, pixels=False)
        path.write_text(
            path.read_text() + "\nPlot.dot({'x': [5.0], 'y': [5.0]})\n",
            encoding="utf-8",
        )
        payload, exit_code = verify_tools.run_verify([path], pixels=False)
        assert exit_code == 1
        statuses = set(block_statuses(payload["targets"][0]).values())
        assert "new" in statuses

    def test_evaluation_error_exit_code(self, project: pathlib.Path):
        path = project / "nb.py"
        path.write_text("boom = 1 / 0\n", encoding="utf-8")
        payload, exit_code = verify_tools.run_verify([path], pixels=False)
        assert exit_code == 2
        assert payload["targets"][0]["status"] == "error"

    def test_colight_artifact_target(self, project: pathlib.Path):
        target = save_artifact(
            project, "plot.colight", Plot.dot({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        )
        verify_tools.update_target(target, pixels=False)
        payload, exit_code = verify_tools.run_verify([target], pixels=False)
        assert exit_code == 0
        assert payload["targets"][0]["kind"] == "colight"

        # Replace the artifact with different data: structure mismatch.
        save_artifact(
            project, "plot.colight", Plot.dot({"x": [1.0, 2.0], "y": [30.0, 40.0]})
        )
        payload, exit_code = verify_tools.run_verify([target], pixels=False)
        assert exit_code == 1

    def test_goldens_override_dir(self, project: pathlib.Path, tmp_path_factory):
        path = write_nb(project)
        alt = tmp_path_factory.mktemp("alt-goldens")
        verify_tools.update_target(path, goldens_root=alt, pixels=False)
        assert (alt / "nb.py" / "manifest.json").exists()
        assert not (project / "tests" / "goldens").exists()
        _payload, exit_code = verify_tools.run_verify(
            [path], goldens_root=alt, pixels=False
        )
        assert exit_code == 0


class TestUpdateReporting:
    def test_update_reports_changes_vs_previous_goldens(self, project: pathlib.Path):
        path = write_nb(project)
        verify_tools.update_target(path, pixels=False)
        write_nb(project, scale=3.0)
        result = verify_tools.update_target(path, pixels=False)
        statuses = block_statuses(result)
        assert "updated" in statuses.values()
        updated = [b for b in result["blocks"] if b["status"] == "updated"][0]
        assert updated["layer"] == "structure"
        assert updated["diff"]["max_abs_delta"] == pytest.approx(18.0)
        assert "unchanged" in statuses.values()
        # Goldens now reflect the new state.
        _payload, exit_code = verify_tools.run_verify([path], pixels=False)
        assert exit_code == 0


class TestCli:
    def test_cli_json_and_exit_codes(self, project: pathlib.Path):
        path = write_nb(project)
        runner = CliRunner()

        result = runner.invoke(cli_main, ["verify", str(path), "--no-pixels", "--json"])
        assert result.exit_code == 3
        payload = json.loads(result.output)
        assert payload["ok"] is False
        assert payload["targets"][0]["status"] == "no-goldens"

        result = runner.invoke(
            cli_main, ["verify", str(path), "--update", "--no-pixels", "--json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["targets"][0]["status"] == "updated"

        result = runner.invoke(cli_main, ["verify", str(path), "--no-pixels", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["ok"] is True

        write_nb(project, scale=3.0)
        result = runner.invoke(cli_main, ["verify", str(path), "--no-pixels"])
        assert result.exit_code == 1
        assert "structure-changed" in result.output

    def test_cli_human_output_is_token_frugal(self, project: pathlib.Path):
        path = write_nb(project)
        runner = CliRunner()
        runner.invoke(cli_main, ["verify", str(path), "--update", "--no-pixels"])
        result = runner.invoke(cli_main, ["verify", str(path), "--no-pixels"])
        assert result.exit_code == 0
        # No raw data values leak into the summary output.
        assert "[0.0, 1.0, 4.0]" not in result.output


class TestPixels:
    def test_pixel_layer_roundtrip_and_mismatch(self, project: pathlib.Path):
        _require_renderer()
        xs = np.linspace(0.0, 1.0, 20)
        target = save_artifact(
            project, "plot.colight", Plot.dot({"x": xs, "y": np.sin(xs)})
        )

        result = verify_tools.update_target(target)
        assert result["status"] == "updated"
        directory = project / "tests" / "goldens" / "plot.colight"
        manifest = json.loads((directory / "manifest.json").read_text())
        shot = manifest["blocks"][0]["screenshot"]
        assert shot["sha256"] and shot["width"] > 0 and shot["height"] > 0

        # Deterministic renderer: verify matches including pixels.
        payload, exit_code = verify_tools.run_verify([target])
        assert exit_code == 0, payload
        block = payload["targets"][0]["blocks"][0]
        assert block["pixels"]["match"] is True

        # Simulate a pixel-only regression by tampering the golden sha.
        manifest["blocks"][0]["screenshot"]["sha256"] = "0" * 64
        (directory / "manifest.json").write_text(json.dumps(manifest))
        payload, exit_code = verify_tools.run_verify([target])
        assert exit_code == 1
        block = payload["targets"][0]["blocks"][0]
        assert block["status"] == "pixels-changed"
        assert block["structure"]["match"] is True

    def test_pixels_skipped_without_chrome(
        self, project: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ):
        path = write_nb(project)
        monkeypatch.setattr(
            verify_tools, "pixels_unavailable_reason", lambda: "Chrome not found"
        )
        result = verify_tools.update_target(path)
        assert any("pixels skipped" in w for w in result["warnings"])
        payload, exit_code = verify_tools.run_verify([path])
        assert exit_code == 0
        assert any(
            "pixels skipped" in w for w in payload["targets"][0].get("warnings", [])
        )
