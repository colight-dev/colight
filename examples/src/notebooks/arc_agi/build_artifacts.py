"""Convert ARC-AGI-3 trajectories into `.colight` replay artifacts.

Each artifact holds the initial frame as state plus one update entry per step,
so `colight render <artifact> --out replay.mp4` produces a replay film and
`colight inspect` / `colight diff` operate on the run's state.

Usage:
    uv run python -m notebooks.arc_agi.build_artifacts SRC [SRC ...] --out DIR

SRC may be a harness `.recording.jsonl` or a `.json.gz` fixture.
"""

import argparse
from pathlib import Path

from notebooks.arc_agi.trajectory import (
    Trajectory,
    build_artifact,
    load_fixture,
    load_recording,
)


def load_any(path: Path) -> Trajectory:
    """Load a trajectory from a harness recording or a fixture file."""
    if path.name.endswith(".recording.jsonl"):
        return load_recording(path)
    if path.name.endswith(".json.gz"):
        return load_fixture(path)
    raise ValueError(f"Unrecognized trajectory file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("output/arc-agi"))
    args = parser.parse_args()

    for src in args.sources:
        traj = load_any(src)
        stem = src.name.removesuffix(".recording.jsonl").removesuffix(".json.gz")
        dest = build_artifact(traj, args.out / f"{stem}.colight")
        print(f"{src} -> {dest} ({len(traj)} steps)")


if __name__ == "__main__":
    main()
