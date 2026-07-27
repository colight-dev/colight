"""
End-to-end: a JavaScript producer, then the colight CLI.

`examples/streaming/deforming_surface.mjs` writes a `.colight` with no Python
involved. These tests run it and then drive `inspect`, `diff` and `render`
against its output, asserting what each reports -- the acceptance case for the
format actually being neutral, checked rather than eyeballed.

Skipped unless `yarn build` has produced `@colight/format`'s JavaScript.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from colight.format import parse_file_with_updates

REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCER = REPO_ROOT / "examples" / "streaming" / "deforming_surface.mjs"
FORMAT_DIST = REPO_ROOT / "packages" / "colight-format" / "dist" / "node.js"

pytestmark = pytest.mark.skipif(
    not (PRODUCER.exists() and FORMAT_DIST.exists()),
    reason="@colight/format is not built; run `yarn build`",
)


def produce(out: Path, *, ticks: int = 6, grid: int = 8, phase: float = 0.0):
    """Runs the JavaScript producer. No Python touches the artifact."""
    subprocess.run(
        [
            "node",
            str(PRODUCER),
            "--out",
            str(out),
            "--ticks",
            str(ticks),
            "--grid",
            str(grid),
            "--phase",
            str(phase),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "colight_cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_inspect_reports_the_entries_and_arrays(tmp_path):
    artifact = produce(tmp_path / "surface.colight", ticks=6, grid=8)

    result = cli("inspect", str(artifact), "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    assert report["kind"] == "colight"
    assert report["updates"] == 6

    arrays = {a["path"]: a for a in report["visual"]["arrays"]}
    # The deforming field, as the producer declared it.
    assert arrays["state.positions"]["dtype"] == "float32"
    assert arrays["state.positions"]["shape"] == [64, 3]
    # The static triangle list.
    indices = next(a for p, a in arrays.items() if p.endswith("geometry.indices"))
    assert indices["dtype"] == "uint32"
    assert indices["shape"] == [(8 - 1) * (8 - 1) * 6]

    assert report["visual"]["buffers"]["count"] == 3
    assert not report["warnings"], report["warnings"]


def test_diff_reports_per_array_deltas_between_two_runs(tmp_path):
    """Two runs differing only in phase: same shape, different motion."""
    a = produce(tmp_path / "a.colight", ticks=6, grid=8, phase=0.0)
    b = produce(tmp_path / "b.colight", ticks=6, grid=8, phase=0.5)

    result = cli("diff", str(a), str(b), "--json")
    assert result.returncode == 1, result.stderr  # 1 == differences found
    report = json.loads(result.stdout)

    updates = report["updates"]
    assert updates["count"] == [6, 6]
    assert updates["aligned"] == 6
    assert updates["updates_differing"] == 6
    assert updates["first_diverging_update"] == 0

    # Every aligned step reports a delta on the deforming array, and only on
    # it: the geometry and the camera are untouched, only the surface moves.
    for step in updates["steps"]:
        arrays = step["arrays"]
        assert arrays["added"] == [] and arrays["removed"] == []
        assert [entry["path"] for entry in arrays["changed"]] == [
            "state.positions"
        ], step
        delta = arrays["changed"][0]
        assert delta["max_abs_delta"] > 0
        assert delta["changed_count"] > 0
        assert step["state"]["changed"] == ["positions"]


def test_diff_reports_a_trailing_length_mismatch(tmp_path):
    """A shorter run against a longer one: the tail has no counterpart."""
    short = produce(tmp_path / "short.colight", ticks=4, grid=8)
    long = produce(tmp_path / "long.colight", ticks=9, grid=8)

    result = cli("diff", str(short), str(long), "--json")
    report = json.loads(result.stdout)

    updates = report["updates"]
    assert updates["count"] == [4, 9]
    assert updates["aligned"] == 4
    # Deterministic producer: the aligned prefix is identical.
    assert updates["updates_differing"] == 0
    assert updates["trailing"] == {"side": "b", "from": 4, "count": 5}


def test_render_at_two_update_indices_produces_different_images(tmp_path):
    """
    `colight render --frame N` applies the first N updates before rendering.

    This is the CLI's update-index affordance -- `colight screenshot` has none
    (its `--frame` selects a camera framing, not a point in the stream).
    """
    artifact = produce(tmp_path / "surface.colight", ticks=40, grid=16)

    early = tmp_path / "early.png"
    late = tmp_path / "late.png"
    for frame, out in ((3, early), (30, late)):
        result = cli(
            "render",
            str(artifact),
            "--frame",
            str(frame),
            "-o",
            str(out),
            "--width",
            "200",
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()

    early_bytes = early.read_bytes()
    late_bytes = late.read_bytes()
    # Both rendered something (a blank frame compresses to almost nothing).
    assert len(early_bytes) > 2000
    assert len(late_bytes) > 2000
    # And the wave moved between them.
    assert early_bytes != late_bytes


def test_a_reader_follows_the_producer_while_it_is_still_appending(tmp_path):
    """
    The demo's `--delay` mode: read the artifact mid-production.

    Every read must succeed and see a prefix that only grows -- the artifact is
    never in a state a reader cannot consume.
    """
    artifact = tmp_path / "live.colight"
    producer = subprocess.Popen(
        [
            "node",
            str(PRODUCER),
            "--out",
            str(artifact),
            "--ticks",
            "40",
            "--grid",
            "8",
            "--delay",
            "25",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    counts = []
    try:
        deadline = time.time() + 60
        while time.time() < deadline and producer.poll() is None:
            if artifact.exists() and artifact.stat().st_size > 0:
                _, _, entries = parse_file_with_updates(artifact)
                counts.append(len(entries))
            time.sleep(0.02)
        assert producer.wait(timeout=60) == 0
    finally:
        if producer.poll() is None:
            producer.kill()

    assert counts, "never observed the artifact while it was being written"
    assert counts == sorted(counts), "the visible prefix shrank"
    # The reader really did catch the producer mid-stream.
    assert min(counts) < 40
    _, _, final = parse_file_with_updates(artifact)
    assert len(final) == 40
