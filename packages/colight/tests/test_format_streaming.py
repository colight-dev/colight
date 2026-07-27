"""
Streaming contract: appending while another process reads.

A .colight file is an append-only stream whose readers tolerate a torn tail, so
a producer may hold it open and a consumer may read it at any moment. These
tests hold both ends of that promise to the fire: a producer appends with
pauses, and a reader polling concurrently must always see a prefix that is
complete, correct, and never shrinks -- including when it lands mid-append.
"""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

import colight.plot as Plot
from colight.format import (
    ColightWriter,
    append_update,
    create_update_bytes,
    parse_file_with_updates,
)

TICKS = 12


def positions(tick: int) -> np.ndarray:
    """A distinct, checkable array per tick."""
    return np.full((4, 3), float(tick), dtype=np.float32)


def update(tick: int):
    return Plot.State({"tick": tick, "positions": positions(tick)})


def read_positions(path: Path) -> list[np.ndarray]:
    """Every update entry's positions array, decoded from its own buffer."""
    _, _, entries = parse_file_with_updates(path)
    out = []
    for entry in entries:
        ref = entry["data"]["state"]["positions"]
        buffer = entry["buffers"][ref["__buffer_index__"]]
        out.append(np.frombuffer(buffer, dtype=ref["dtype"]).reshape(ref["shape"]))
    return out


# ---------------------------------------------------------------------------
# The writer API itself


def test_writer_appends_are_readable_after_each_append(tmp_path):
    """After every single append the file parses, with one more entry."""
    path = tmp_path / "grow.colight"
    with ColightWriter.create(path, Plot.State({"positions": positions(0)})) as w:
        for tick in range(1, TICKS + 1):
            w.append(update(tick))

            initial, _, entries = parse_file_with_updates(path)
            assert initial is not None
            assert len(entries) == tick

    assert [int(p[0, 0]) for p in read_positions(path)] == list(range(1, TICKS + 1))


def test_writer_open_reopens_for_further_appends(tmp_path):
    path = tmp_path / "reopen.colight"
    with ColightWriter.create(path, Plot.State({"positions": positions(0)})) as w:
        w.append(update(1))

    with ColightWriter.open(path) as w:
        w.append(update(2))

    assert [int(p[0, 0]) for p in read_positions(path)] == [1, 2]


def test_writer_create_without_initial_state_yields_updates_only_file(tmp_path):
    path = tmp_path / "updates-only.colight"
    with ColightWriter.create(path) as w:
        w.append_all([update(1), update(2)])

    initial, _, entries = parse_file_with_updates(path)
    assert initial is None
    assert len(entries) == 2


def test_holding_open_and_reopening_produce_the_same_artifact(tmp_path):
    """
    The two strategies differ only in how the file is opened.

    Same entry count, same entry sizes, same decoded values -- so a producer can
    choose on throughput grounds alone. (The bytes are not literally identical:
    each serialization mints a fresh widget id.)
    """
    held = tmp_path / "held.colight"
    reopened = tmp_path / "reopened.colight"

    with ColightWriter.create(held, Plot.State({"positions": positions(0)})) as w:
        for tick in range(1, 4):
            w.append(update(tick))

    ColightWriter.create(reopened, Plot.State({"positions": positions(0)})).close()
    for tick in range(1, 4):
        append_update(reopened, update(tick))

    assert held.stat().st_size == reopened.stat().st_size
    for a, b in zip(read_positions(held), read_positions(reopened)):
        assert np.array_equal(a, b)


def test_appending_to_a_misaligned_file_is_refused(tmp_path):
    """Alignment is a precondition, not something to silently break."""
    path = tmp_path / "misaligned.colight"
    with ColightWriter.create(path, Plot.State({"positions": positions(0)})) as w:
        w.append(update(1))
    path.write_bytes(path.read_bytes() + b"\x00")  # length no longer % 8

    with pytest.raises(ValueError, match="multiple of 8"):
        append_update(path, update(2))
    with pytest.raises(ValueError, match="multiple of 8"):
        ColightWriter.open(path)


def test_open_rejects_a_non_colight_file(tmp_path):
    path = tmp_path / "not.colight"
    path.write_bytes(b"NOTMAGIC" + b"\x00" * 88)
    with pytest.raises(ValueError, match="magic bytes"):
        ColightWriter.open(path)


def test_appending_after_close_raises(tmp_path):
    path = tmp_path / "closed.colight"
    w = ColightWriter.create(path)
    w.close()
    assert w.closed
    w.close()  # idempotent
    with pytest.raises(ValueError, match="closed"):
        w.append(update(1))


# ---------------------------------------------------------------------------
# Reading while writing


def test_reader_sees_a_monotonically_growing_prefix_while_writing(tmp_path):
    """
    A reader polling during a paced write never regresses and never errors.

    The producer runs in this process with pauses between appends; between each
    pair of appends the reader parses the file from scratch, exactly as a
    separate process would.
    """
    path = tmp_path / "concurrent.colight"
    seen = []

    with ColightWriter.create(path, Plot.State({"positions": positions(0)})) as w:
        for tick in range(1, TICKS + 1):
            w.append(update(tick))
            time.sleep(0.002)

            decoded = read_positions(path)
            seen.append(len(decoded))
            # Every entry visible so far is complete and correct...
            assert [int(p[0, 0]) for p in decoded] == list(range(1, len(decoded) + 1))
            # ...and the visible prefix never shrinks.
            assert seen == sorted(seen)

    assert seen[-1] == TICKS


def test_reader_lands_mid_append_and_sees_the_complete_prefix(tmp_path):
    """
    Reads that land *inside* an entry, byte by byte.

    Rather than racing a real writer (which would be flaky), this replays a
    real append one byte-count at a time -- the technique the JS torn-tail
    suite uses -- so every possible mid-append instant is covered, not just the
    ones a scheduler happens to produce.
    """
    path = tmp_path / "torn.colight"
    with ColightWriter.create(path, Plot.State({"positions": positions(0)})) as w:
        w.append(update(1))
        w.append(update(2))
    complete = path.read_bytes()

    tail = create_update_bytes(update(3))
    partial_path = path.parent / "partial.colight"

    # Every prefix of the in-flight third entry: the reader must see exactly
    # the two complete updates, never an error and never a phantom third.
    for cut in range(0, len(tail), 8):
        partial_path.write_bytes(complete + tail[:cut])
        decoded = read_positions(partial_path)
        assert [int(p[0, 0]) for p in decoded] == [1, 2], f"cut={cut}"

    # And once the last byte lands, the third entry appears whole.
    partial_path.write_bytes(complete + tail)
    assert [int(p[0, 0]) for p in read_positions(partial_path)] == [1, 2, 3]


def test_a_separate_process_reads_while_this_one_writes(tmp_path):
    """
    The real two-process case: an external reader tails a file we are writing.

    The reader is a subprocess polling with `parse_file_with_updates`; it
    records the entry count it observes and asserts monotonicity itself, so a
    corrupt or shrinking read fails in the reader, not by inference here.
    """
    path = tmp_path / "two-process.colight"
    report = tmp_path / "report.txt"

    reader_source = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        from colight.format import parse_file_with_updates

        path = Path({str(path)!r})
        counts = []
        deadline = time.time() + 30
        while time.time() < deadline:
            if path.exists() and path.stat().st_size > 0:
                try:
                    _, _, entries = parse_file_with_updates(path)
                except Exception as exc:  # a torn tail must never raise
                    raise SystemExit(f"reader error: {{exc}}")
                if counts and len(entries) < counts[-1]:
                    raise SystemExit("prefix shrank")
                counts.append(len(entries))
                if len(entries) >= {TICKS}:
                    break
            time.sleep(0.005)
        Path({str(report)!r}).write_text(",".join(map(str, counts)))
        """
    )

    reader = subprocess.Popen([sys.executable, "-c", reader_source])
    try:
        with ColightWriter.create(path, Plot.State({"positions": positions(0)})) as w:
            for tick in range(1, TICKS + 1):
                w.append(update(tick))
                time.sleep(0.02)
        assert reader.wait(timeout=30) == 0, "the concurrent reader failed"
    finally:
        if reader.poll() is None:
            reader.kill()

    counts = [int(n) for n in report.read_text().split(",") if n]
    assert counts, "the reader never observed the file"
    assert counts == sorted(counts), "the reader saw the prefix shrink"
    assert counts[-1] == TICKS, "the reader never caught up to the writer"
    # The point of the exercise: the reader observed the file mid-stream, not
    # only once it was finished.
    assert min(counts) < TICKS


def test_a_reader_sees_partial_data_from_a_writer_that_never_closes(tmp_path):
    """
    A producer killed mid-run leaves a readable artifact.

    No close, no flush -- just a process that stops existing. Everything it
    finished writing must still be there.
    """
    path = tmp_path / "abandoned.colight"
    writer_source = textwrap.dedent(
        f"""
        import time
        import numpy as np
        import colight.plot as Plot
        from colight.format import ColightWriter

        w = ColightWriter.create({str(path)!r}, Plot.State({{"tick": 0}}))
        for tick in range(1, 1000):
            w.append(Plot.State({{"tick": tick}}))
            time.sleep(0.01)
        """
    )
    writer = subprocess.Popen([sys.executable, "-c", writer_source])
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if path.exists() and path.stat().st_size > 0:
                _, _, entries = parse_file_with_updates(path)
                if len(entries) >= 3:
                    break
            time.sleep(0.005)
    finally:
        writer.kill()
        writer.wait(timeout=10)

    initial, _, entries = parse_file_with_updates(path)
    assert initial is not None
    assert len(entries) >= 3
    ticks = [e["data"]["state"]["tick"] for e in entries]
    assert ticks == list(range(1, len(ticks) + 1))


def test_a_python_reader_consumes_a_javascript_written_stream(tmp_path):
    """
    The whole point of format neutrality: no Python in the producer.

    Skipped when the `@colight/format` build output is absent (`yarn build`).
    """
    repo_root = Path(__file__).resolve().parents[3]
    producer = repo_root / "examples" / "streaming" / "deforming_surface.mjs"
    dist = repo_root / "packages" / "colight-format" / "dist" / "node.js"
    if not producer.exists() or not dist.exists():
        pytest.skip("@colight/format is not built; run `yarn build`")

    path = tmp_path / "from-js.colight"
    subprocess.run(
        [
            "node",
            str(producer),
            "--out",
            str(path),
            "--ticks",
            "5",
            "--grid",
            "8",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    initial, buffers, entries = parse_file_with_updates(path)
    assert initial is not None
    assert len(entries) == 5
    assert initial["state"]["positions"]["dtype"] == "float32"
    assert initial["state"]["positions"]["shape"] == [64, 3]
    assert [e["data"]["state"]["tick"] for e in entries] == [1, 2, 3, 4, 5]
