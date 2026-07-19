"""Tests for machine-legible screenshots: --rulers, --views, --max-edge.

Pure composition/arithmetic tests run everywhere; end-to-end tests drive
the real CLI through headless Chrome (skipped when Chrome or the JS bundle
is missing, same mechanism as the other visual tests).
"""

import io
import json
import pathlib
from typing import Tuple

import pytest
from click.testing import CliRunner
from PIL import Image

import colight.env as env
from colight.chrome_devtools import find_chrome
from colight.cli_tools import compose, scene_pick, screenshot_tools
from colight.cli_tools import daemon as daemon_mod
from colight_cli import main as cli_main

# Same scene as the pick tests: camera on +z, red ellipsoid left, green
# right, blue fully occluded behind green, yellow cuboid above.
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
        pytest.skip("Chrome not found for legibility tests")


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    return tmp_path


@pytest.fixture
def scene_file(project: pathlib.Path) -> pathlib.Path:
    path = project / "scene.py"
    path.write_text(SCENE)
    return path


def _png(width: int, height: int, color: Tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


class TestRulerSpacing:
    def test_nice_spacings_by_size(self):
        assert compose.ruler_spacing(100) == 10
        assert compose.ruler_spacing(400) == 50
        assert compose.ruler_spacing(800) == 100
        assert compose.ruler_spacing(1600) == 200
        assert compose.ruler_spacing(4000) == 500

    def test_at_most_ten_ticks(self):
        for size in (120, 333, 800, 1234, 5000):
            assert size / compose.ruler_spacing(size) <= 10


class TestFitMaxEdge:
    def test_landscape_and_portrait(self):
        assert screenshot_tools.fit_max_edge(400, 300, 1000) == (1000, 750)
        assert screenshot_tools.fit_max_edge(300, 400, 1000) == (750, 1000)
        assert screenshot_tools.fit_max_edge(500, 500, 200) == (200, 200)

    def test_long_edge_always_exact(self):
        for w, h in ((123, 457), (800, 799), (10, 3000)):
            fitted = screenshot_tools.fit_max_edge(w, h, 768)
            assert max(fitted) == 768

    def test_invalid_inputs(self):
        with pytest.raises(ValueError, match="positive"):
            screenshot_tools.fit_max_edge(400, 300, 0)
        with pytest.raises(ValueError, match="invalid dimensions"):
            screenshot_tools.fit_max_edge(0, 300, 100)


class TestParseViews:
    def test_valid_list(self):
        assert scene_pick.parse_views("front,top, side ,iso") == [
            "front",
            "top",
            "side",
            "iso",
        ]

    def test_errors(self):
        with pytest.raises(ValueError, match="unknown view"):
            scene_pick.parse_views("front,diagonal")
        with pytest.raises(ValueError, match="duplicate view"):
            scene_pick.parse_views("front,front")
        with pytest.raises(ValueError, match="no views"):
            scene_pick.parse_views(" , ")

    def test_presets_cover_axes_and_iso(self):
        assert {"front", "back", "left", "right", "top", "bottom", "iso"} <= set(
            scene_pick.VIEW_PRESETS
        )


class TestComposeRulers:
    """Pure composition on synthetic PNGs (no renderer needed)."""

    def test_expands_canvas_by_margin(self):
        base = _png(200, 150, (40, 40, 40))
        composed, meta = compose.compose_rulers(base)
        image = Image.open(io.BytesIO(composed))
        assert image.size == (200 + meta["margin"], 150 + meta["margin"])
        assert meta["spacing"] == compose.ruler_spacing(200)

    def test_margin_band_and_ticks_present(self):
        base = _png(400, 400, (40, 40, 40))
        composed, meta = compose.compose_rulers(base)
        image = Image.open(io.BytesIO(composed)).convert("RGB")
        margin, spacing = meta["margin"], meta["spacing"]
        # The margin band is non-background (white, distinct from content).
        assert image.getpixel((2, 2)) == (255, 255, 255)
        # Tick marks sit exactly at composed = margin + coordinate for
        # every labeled coordinate; between ticks the band stays white.
        for tick in range(0, 400, spacing):
            x = margin + tick
            assert image.getpixel((x, margin - 2)) == (0, 0, 0), f"x tick {tick}"
            assert image.getpixel((margin - 2, x)) == (0, 0, 0), f"y tick {tick}"
        assert image.getpixel((margin + spacing // 2, margin - 2)) == (255, 255, 255)

    def test_gridlines_faint_and_content_preserved(self):
        color = (40, 40, 40)
        base = _png(400, 400, color)
        composed, meta = compose.compose_rulers(base)
        image = Image.open(io.BytesIO(composed)).convert("RGB")
        margin, spacing = meta["margin"], meta["spacing"]
        # A gridline crosses the content but only faintly (blended, not
        # opaque) ...
        on_grid = image.getpixel((margin + spacing, margin + spacing // 2))
        assert on_grid != color
        assert all(abs(a - b) < 60 for a, b in zip(on_grid, color))
        # ... and off-grid scene pixels are byte-identical to the capture.
        off_grid = image.getpixel((margin + spacing // 2, margin + spacing // 2))
        assert off_grid == color

    def test_composition_is_deterministic(self):
        base = _png(300, 200, (10, 120, 200))
        first, _ = compose.compose_rulers(base)
        second, _ = compose.compose_rulers(base)
        assert first == second

    def test_dpr_scales_tick_positions(self):
        base = _png(800, 800, (40, 40, 40))
        composed, meta = compose.compose_rulers(base, dpr=2.0)
        image = Image.open(io.BytesIO(composed)).convert("RGB")
        margin, spacing = meta["margin"], meta["spacing"]
        # 800px at dpr 2 covers 400 CSS px; ticks land at margin + css*2.
        assert meta["spacing"] == compose.ruler_spacing(400)
        assert image.getpixel((margin + spacing * 2, margin - 2)) == (0, 0, 0)


class TestComposeGrid:
    def test_cells_labeled_and_distinct(self):
        cells = [
            ("front", _png(100, 80, (200, 0, 0))),
            ("top", _png(100, 80, (0, 200, 0))),
            ("side", _png(100, 80, (0, 0, 200))),
            ("iso", _png(100, 80, (200, 200, 0))),
        ]
        composed = compose.compose_grid(cells)
        image = Image.open(io.BytesIO(composed)).convert("RGB")
        columns, rows = compose.grid_cell_count([name for name, _ in cells])
        assert (columns, rows) == (2, 2)
        gutter = (image.width - columns * 100) // (columns + 1)
        band = (image.height - rows * 80 - (rows + 1) * gutter) // rows
        assert band > 10  # label band exists above every cell
        seen = set()
        for index, (_name, _png_bytes) in enumerate(cells):
            col, row = index % columns, index // columns
            x0 = gutter + col * (100 + gutter)
            y0 = gutter + row * (80 + band + gutter)
            # Label band is white with dark text pixels in it.
            band_region = image.crop((x0, y0, x0 + 100, y0 + band))
            colors = {pixel for pixel in band_region.getdata()}
            assert (255, 255, 255) in colors
            assert any(sum(c) < 200 for c in colors)  # text ink
            # Cell content is pasted unscaled and cells differ.
            content = image.crop((x0, y0 + band, x0 + 100, y0 + band + 80))
            digest = content.tobytes()
            assert digest not in seen
            seen.add(digest)

    def test_deterministic_and_rejects_empty(self):
        cells = [("a", _png(50, 50, (1, 2, 3)))]
        assert compose.compose_grid(cells) == compose.compose_grid(cells)
        with pytest.raises(ValueError, match="at least one view"):
            compose.compose_grid([])


class TestExclusivity:
    def test_rulers_and_views_rejected_without_render(self, tmp_path: pathlib.Path):
        class _Source:
            block_id = None

        with pytest.raises(ValueError, match="single-view"):
            screenshot_tools.screenshot_source(
                _Source(),  # type: ignore[arg-type]  (raises before use)
                "t",
                tmp_path / "x.png",
                rulers=True,
                views=["front"],
            )


class TestLegibilityCli:
    """End-to-end through the real CLI + headless Chrome."""

    def test_ruler_coordinates_align_with_pick_at(
        self, scene_file: pathlib.Path, tmp_path: pathlib.Path
    ):
        """THE critical contract: a coordinate read off the ruler is a
        valid pick-at input for the same instance."""
        _require_renderer()
        runner = CliRunner()
        out = tmp_path / "rulers.png"
        shot = runner.invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "-o",
                str(out),
                *SIZE_ARGS,
                "--rulers",
                "--no-daemon",
                "--json",
            ],
        )
        assert shot.exit_code == 0, shot.output
        payload = json.loads(shot.output)
        margin = payload["rulers"]["margin"]
        spacing = payload["rulers"]["spacing"]
        assert payload["width"] == 400 + margin
        assert payload["height"] == 400 + margin

        # Ground truth: where instance 1 (green ellipsoid) actually is.
        where = runner.invoke(
            cli_main,
            [
                "pick-where",
                str(scene_file),
                "--component",
                "0",
                "--instances",
                "1",
                *SIZE_ARGS,
                "--no-daemon",
                "--json",
            ],
        )
        assert where.exit_code == 0, where.output
        centroid_x, centroid_y = json.loads(where.output)["centroid"]

        # pick-at accepts that centroid (page CSS px) and hits instance 1.
        pick = runner.invoke(
            cli_main,
            [
                "pick-at",
                str(scene_file),
                f"{centroid_x},{centroid_y}",
                *SIZE_ARGS,
                "--no-daemon",
                "--json",
            ],
        )
        assert pick.exit_code == 0, pick.output
        hit = json.loads(pick.output)["hits"][0]
        assert (hit["component"], hit["instance"]) == (0, 1)

        image = Image.open(out).convert("RGB")
        # Ruler-space convention: page coordinate v is drawn at composed
        # pixel margin + v. Tick marks prove the labeled axis is the page
        # pixel axis...
        for tick in range(0, 400, spacing):
            assert image.getpixel((margin + tick, margin - 2)) == (0, 0, 0)
            assert image.getpixel((margin - 2, margin + tick)) == (0, 0, 0)
        # ...so the hit's ruler-space position must land on the instance's
        # actual pixels: the green ellipsoid, on a black background.
        r, g, b = image.getpixel(
            (margin + round(centroid_x), margin + round(centroid_y))
        )
        assert g > 100 and g > r and g > b, (r, g, b)

    def test_contact_sheet_cells_differ(
        self, scene_file: pathlib.Path, tmp_path: pathlib.Path
    ):
        _require_renderer()
        out = tmp_path / "sheet.png"
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "-o",
                str(out),
                *SIZE_ARGS,
                "--views",
                "front,top,side,iso",
                "--no-daemon",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        views = payload["views"]
        assert [v["view"] for v in views] == ["front", "top", "side", "iso"]
        # Per-view cameras are distinct (genuinely different directions).
        positions = [tuple(v["camera"]["position"]) for v in views]
        assert len(set(positions)) == 4
        # "coverage" is a single-view concept.
        assert "coverage" not in payload

        image = Image.open(out).convert("RGB")
        columns, rows = compose.grid_cell_count(["a", "b", "c", "d"])
        gutter = (image.width - columns * 400) // (columns + 1)
        band = (image.height - rows * 400 - (rows + 1) * gutter) // rows
        hashes = set()
        for index in range(4):
            col, row = index % columns, index // columns
            x0 = gutter + col * (400 + gutter)
            y0 = gutter + row * (400 + band + gutter) + band
            hashes.add(image.crop((x0, y0, x0 + 400, y0 + 400)).tobytes())
        assert len(hashes) == 4  # every view rendered a different image

    def test_views_deterministic_check(self, scene_file: pathlib.Path):
        _require_renderer()
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "-o",
                str(scene_file.parent / "sheet.png"),
                *SIZE_ARGS,
                "--views",
                "front,iso",
                "--check",
                "--no-daemon",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["deterministic"] is True

    def test_rulers_deterministic_composed_bytes(
        self, scene_file: pathlib.Path, tmp_path: pathlib.Path
    ):
        _require_renderer()
        runner = CliRunner()
        outputs = []
        for name in ("a.png", "b.png"):
            out = tmp_path / name
            result = runner.invoke(
                cli_main,
                [
                    "screenshot",
                    str(scene_file),
                    "-o",
                    str(out),
                    *SIZE_ARGS,
                    "--rulers",
                    *(["--check"] if name == "a.png" else []),
                    "--no-daemon",
                    "--json",
                ],
            )
            assert result.exit_code == 0, result.output
            payload = json.loads(result.output)
            if "--check" in result.output or "deterministic" in payload:
                # --check verified the underlying render byte-identity.
                assert payload["deterministic"] is True
            outputs.append(out.read_bytes())
        # Independent composed outputs are also byte-identical.
        assert outputs[0] == outputs[1]

    def test_max_edge_exact_long_edge(
        self, scene_file: pathlib.Path, tmp_path: pathlib.Path
    ):
        _require_renderer()
        out = tmp_path / "big.png"
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "-o",
                str(out),
                "--width",
                "400",
                "--height",
                "300",
                "--max-edge",
                "600",
                "--no-daemon",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert (payload["width"], payload["height"]) == (600, 450)
        assert payload["max_edge"] == 600
        assert Image.open(out).size == (600, 450)

    def test_max_edge_measured_height(
        self, scene_file: pathlib.Path, tmp_path: pathlib.Path
    ):
        _require_renderer()
        out = tmp_path / "measured.png"
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "-o",
                str(out),
                "--width",
                "400",
                "--max-edge",
                "512",
                "--no-daemon",
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert max(payload["width"], payload["height"]) == 512
        assert max(Image.open(out).size) == 512

    def test_max_edge_indivisible_dpr_errors(self, scene_file: pathlib.Path):
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "-o",
                "/tmp/never.png",
                "--max-edge",
                "501",
                "--dpr",
                "2",
            ],
        )
        assert result.exit_code == 2
        assert "divisible" in result.output

    def test_rulers_and_views_conflict_exits_2(self, scene_file: pathlib.Path):
        result = CliRunner().invoke(
            cli_main,
            [
                "screenshot",
                str(scene_file),
                "-o",
                "/tmp/never.png",
                "--rulers",
                "--views",
                "front",
            ],
        )
        assert result.exit_code == 2
        assert "single-view" in result.output


class TestDaemonParity:
    """Composition lives in the shared screenshot_source path, so daemon
    and direct modes must produce byte-identical composed output."""

    def test_rulers_and_views_bytes_identical(
        self, scene_file: pathlib.Path, project: pathlib.Path
    ):
        _require_renderer()
        runner = CliRunner()

        def shot(out_name: str, *extra: str) -> dict:
            result = runner.invoke(
                cli_main,
                [
                    "screenshot",
                    str(scene_file),
                    "-o",
                    str(project / out_name),
                    *SIZE_ARGS,
                    *extra,
                    "--json",
                ],
            )
            assert result.exit_code == 0, result.output
            return json.loads(result.output)

        direct_rulers = shot("direct-rulers.png", "--rulers", "--no-daemon")
        direct_views = shot("direct-views.png", "--views", "front,iso", "--no-daemon")

        daemon = daemon_mod.DaemonServer(project, idle_timeout=300.0, pool_size=2)
        daemon.start()
        try:
            routed_rulers = shot("routed-rulers.png", "--rulers")
            routed_views = shot("routed-views.png", "--views", "front,iso")
            assert daemon.request_counts.get("/screenshot") == 2
        finally:
            daemon.shutdown()

        assert (project / "routed-rulers.png").read_bytes() == (
            project / "direct-rulers.png"
        ).read_bytes()
        assert routed_rulers["sha256"] == direct_rulers["sha256"]
        assert routed_rulers["rulers"] == direct_rulers["rulers"]

        assert (project / "routed-views.png").read_bytes() == (
            project / "direct-views.png"
        ).read_bytes()
        assert routed_views["sha256"] == direct_views["sha256"]
        assert routed_views["views"] == direct_views["views"]
