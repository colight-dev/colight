"""Tests for deterministic screenshots: colight screenshot."""

import hashlib
import json
import pathlib

import numpy as np
import pytest
from click.testing import CliRunner

import colight.env as env
import colight.plot as Plot
from colight.chrome_devtools import find_chrome
from colight.cli_tools import screenshot_tools
from colight.inspect import inspect as colight_inspect
from colight_cli import main as cli_main

TWO_VISUALS = (
    "import colight.plot as Plot\n\n"
    "first = Plot.dot({'x': [1.0, 2.0], 'y': [3.0, 4.0]})\n"
    "first\n\n"
    "Plot.line({'x': [1.0, 2.0], 'y': [3.0, 4.0]})\n"
)


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


def save_artifact(tmp_path: pathlib.Path, name: str, plot) -> pathlib.Path:
    visual = colight_inspect(plot)
    assert visual is not None
    target = tmp_path / name
    target.write_bytes(visual.to_bytes())
    return target


class TestResolveVisual:
    """Visual selection needs no renderer."""

    def test_py_defaults_to_last_visual(self, project: pathlib.Path):
        path = project / "nb.py"
        path.write_text(TWO_VISUALS)
        data, _buffers, block = screenshot_tools.resolve_visual(path)
        assert block is not None
        # The last visual is the line plot.
        assert "line" in json.dumps(data.get("state"))

    def test_py_block_selection(self, project: pathlib.Path):
        from colight.cli_tools import inspect_tools

        path = project / "nb.py"
        path.write_text(TWO_VISUALS)
        visuals, _errors = inspect_tools.evaluate_python_visuals(path)
        assert len(visuals) == 2
        first_block = visuals[0]["block"]
        data, _buffers, block = screenshot_tools.resolve_visual(path, first_block)
        assert block == first_block
        assert "dot" in json.dumps(data.get("state"))

    def test_py_unknown_block(self, project: pathlib.Path):
        path = project / "nb.py"
        path.write_text(TWO_VISUALS)
        with pytest.raises(ValueError, match="blocks with visuals"):
            screenshot_tools.resolve_visual(path, "nope")

    def test_py_no_visuals(self, project: pathlib.Path):
        path = project / "nb.py"
        path.write_text("x = 1\n")
        with pytest.raises(ValueError, match="no visuals produced"):
            screenshot_tools.resolve_visual(path)

    def test_block_rejected_for_colight(self, tmp_path: pathlib.Path):
        target = save_artifact(tmp_path, "a.colight", Plot.dot([1, 2]))
        with pytest.raises(ValueError, match="only to .py targets"):
            screenshot_tools.resolve_visual(target, "some-id")

    def test_unsupported_target(self, tmp_path: pathlib.Path):
        target = tmp_path / "foo.txt"
        target.write_text("hi")
        with pytest.raises(ValueError, match="Unsupported target"):
            screenshot_tools.resolve_visual(target)


class TestScreenshot:
    def make_artifact(self, tmp_path: pathlib.Path) -> pathlib.Path:
        xs = np.linspace(0.0, 1.0, 30)
        return save_artifact(
            tmp_path, "plot.colight", Plot.dot({"x": xs, "y": np.sin(xs)})
        )

    def test_deterministic_double_render(self, tmp_path: pathlib.Path):
        _require_renderer()
        target = self.make_artifact(tmp_path)
        out = tmp_path / "shot.png"
        payload = screenshot_tools.screenshot_target(
            target, out, width=400, height=300, check=True
        )
        assert out.exists()
        assert payload["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
        assert payload["width"] == 400
        assert payload["height"] == 300
        # Acceptance: two renders of the same input are byte-identical.
        assert payload["deterministic"] is True

    def test_dpr_scales_output_pixels(self, tmp_path: pathlib.Path):
        _require_renderer()
        target = self.make_artifact(tmp_path)
        out = tmp_path / "shot2x.png"
        payload = screenshot_tools.screenshot_target(
            target, out, width=400, height=300, dpr=2.0
        )
        assert (payload["width"], payload["height"]) == (800, 600)

    def test_cli_json(self, tmp_path: pathlib.Path):
        _require_renderer()
        target = self.make_artifact(tmp_path)
        out = tmp_path / "cli.png"
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(target),
                "--out",
                str(out),
                "--width",
                "300",
                "--height",
                "200",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert {"target", "out", "width", "height", "dpr", "sha256"} <= set(payload)
        assert out.exists()

    def test_cli_error_exit_code(self, tmp_path: pathlib.Path):
        bad = tmp_path / "bad.txt"
        bad.write_text("nope")
        result = CliRunner().invoke(
            cli_main, ["screenshot", str(bad), "--out", str(tmp_path / "x.png")]
        )
        assert result.exit_code == 2
