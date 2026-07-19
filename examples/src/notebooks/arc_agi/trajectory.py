"""ARC-AGI-3 trajectory loading and conversion to colight artifacts.

ARC-AGI-3 (https://docs.arcprize.org) is an interactive reasoning benchmark:
agents act in 64x64 grid game environments over time. The official agents
harness (arcprize/ARC-AGI-3-Agents) records each run as a `.recording.jsonl`
file - one JSON event per action, each carrying the resulting frame(s)
(64x64 grids of 4-bit color indices), the echoed action, game state, and
levels completed.

This module:

- loads harness recordings into a `Trajectory` of `Step`s,
- round-trips compact gzipped JSON fixtures (hex-row encoded grids) so docs
  and tests run without the ARC API,
- maps grids to RGB via the official 16-color palette,
- computes per-step behavioral comparisons between two runs,
- converts a trajectory into a `.colight` artifact: initial state plus one
  update entry per step (renderable to mp4 via `colight render --out x.mp4`).

Fixtures in `fixtures/` were recorded by us with the official random agent
against public games (ls20, lf52); they contain only our own agent's
observations.
"""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

GRID_SIZE = 64

# Official 16-color palette used by the ARC-AGI-3 harness (RGB).
PALETTE: np.ndarray = np.array(
    [
        (0xFF, 0xFF, 0xFF),  # 0 White
        (0xCC, 0xCC, 0xCC),  # 1 Off-white
        (0x99, 0x99, 0x99),  # 2 Neutral light
        (0x66, 0x66, 0x66),  # 3 Neutral
        (0x33, 0x33, 0x33),  # 4 Off-black
        (0x00, 0x00, 0x00),  # 5 Black
        (0xE5, 0x3A, 0xA3),  # 6 Magenta
        (0xFF, 0x7B, 0xCC),  # 7 Magenta light
        (0xF9, 0x3C, 0x31),  # 8 Red
        (0x1E, 0x93, 0xFF),  # 9 Blue
        (0x88, 0xD8, 0xF1),  # 10 Blue light
        (0xFF, 0xDC, 0x00),  # 11 Yellow
        (0xFF, 0x85, 0x1B),  # 12 Orange
        (0x92, 0x12, 0x31),  # 13 Maroon
        (0x4F, 0xCC, 0x30),  # 14 Green
        (0xA3, 0x56, 0xD6),  # 15 Purple
    ],
    dtype=np.uint8,
)

ACTION_NAMES: dict[int, str] = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "CLICK",
    7: "ACTION7",
}

FIXTURE_FORMAT = "arc-agi-3-trajectory/v1"


@dataclass
class Step:
    """One agent action and the settled frame that resulted from it."""

    index: int
    action_id: int
    x: Optional[int]
    y: Optional[int]
    state: str
    levels: int
    grid: np.ndarray  # (64, 64) uint8, values 0-15

    @property
    def action_name(self) -> str:
        return ACTION_NAMES.get(self.action_id, f"ACTION{self.action_id}")

    @property
    def label(self) -> str:
        if self.action_id == 6 and self.x is not None:
            return f"CLICK({self.x},{self.y})"
        return self.action_name


@dataclass
class Trajectory:
    """A recorded run: one agent playing one game from RESET onwards."""

    game_id: str
    agent: str
    guid: str
    win_levels: int
    steps: list[Step]

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def grids(self) -> np.ndarray:
        """All settled grids, shape (n_steps, 64, 64) uint8."""
        return np.stack([s.grid for s in self.steps])

    @property
    def actions(self) -> np.ndarray:
        """Action ids per step, shape (n_steps,)."""
        return np.array([s.action_id for s in self.steps], dtype=np.int64)

    @property
    def levels(self) -> np.ndarray:
        """Cumulative levels completed per step, shape (n_steps,)."""
        return np.array([s.levels for s in self.steps], dtype=np.int64)

    def rgb(self) -> np.ndarray:
        """Palette-mapped frames, shape (n_steps, 64, 64, 3) uint8."""
        return PALETTE[self.grids]

    def flat_rgb(self) -> np.ndarray:
        """Frames flattened for `Plot.pixels`/`bitmap`, shape (n_steps, 64*64*3)."""
        return self.rgb().reshape(len(self), -1)


def load_recording(path: Union[str, Path]) -> Trajectory:
    """Load a harness `.recording.jsonl` file into a Trajectory.

    Args:
        path: Path to a `<game>.<agent>.<guid>.recording.jsonl` file produced
            by the official ARC-AGI-3 agents harness.

    Returns:
        Trajectory with one Step per recorded action (settled/last frame).
    """
    path = Path(path)
    name_parts = path.name.split(".")
    agent = ".".join(name_parts[1:-3]) if len(name_parts) >= 5 else "unknown"

    steps: list[Step] = []
    game_id = ""
    guid = ""
    win_levels = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line).get("data", {})
            frames = data.get("frame")
            if not frames:
                continue  # scorecard/summary entries carry no frame
            game_id = data.get("game_id", game_id)
            guid = data.get("guid") or guid
            win_levels = data.get("win_levels", win_levels)
            action = data.get("action_input") or {}
            action_data = action.get("data") or {}
            # The first event is the RESET response; the server's action_input
            # echo on that frame is not meaningful, so normalize it to RESET.
            steps.append(
                Step(
                    index=len(steps),
                    action_id=0 if not steps else action.get("id", 0),
                    x=action_data.get("x"),
                    y=action_data.get("y"),
                    state=data.get("state", "NOT_FINISHED"),
                    levels=data.get("levels_completed", 0),
                    grid=np.array(frames[-1], dtype=np.uint8),
                )
            )
    return Trajectory(
        game_id=game_id, agent=agent, guid=guid, win_levels=win_levels, steps=steps
    )


def save_fixture(traj: Trajectory, path: Union[str, Path]) -> Path:
    """Save a Trajectory as a compact gzipped JSON fixture.

    Grids are encoded as 64 rows of 64 hex digits (one digit per cell).

    Args:
        traj: Trajectory to save.
        path: Output path, conventionally ending in `.json.gz`.

    Returns:
        The output path.
    """
    path = Path(path)
    doc = {
        "format": FIXTURE_FORMAT,
        "game_id": traj.game_id,
        "agent": traj.agent,
        "guid": traj.guid,
        "win_levels": traj.win_levels,
        "steps": [
            {
                "action": s.action_id,
                "x": s.x,
                "y": s.y,
                "state": s.state,
                "levels": s.levels,
                "grid": ["".join(f"{v:x}" for v in row) for row in s.grid.tolist()],
            }
            for s in traj.steps
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    return path


def load_fixture(path: Union[str, Path]) -> Trajectory:
    """Load a Trajectory from a fixture written by `save_fixture`."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != FIXTURE_FORMAT:
        raise ValueError(f"Unknown fixture format: {doc.get('format')!r}")
    steps = [
        Step(
            index=i,
            action_id=s["action"],
            x=s.get("x"),
            y=s.get("y"),
            state=s["state"],
            levels=s["levels"],
            grid=np.array(
                [[int(c, 16) for c in row] for row in s["grid"]], dtype=np.uint8
            ),
        )
        for i, s in enumerate(doc["steps"])
    ]
    return Trajectory(
        game_id=doc["game_id"],
        agent=doc["agent"],
        guid=doc["guid"],
        win_levels=doc["win_levels"],
        steps=steps,
    )


def grid_change_counts(traj: Trajectory) -> np.ndarray:
    """Cells changed between consecutive steps, shape (n_steps,); index 0 is 0."""
    grids = traj.grids
    changes = np.zeros(len(traj), dtype=np.int64)
    if len(traj) > 1:
        changes[1:] = (grids[1:] != grids[:-1]).sum(axis=(1, 2))
    return changes


def grid_diff_counts(a: Trajectory, b: Trajectory) -> np.ndarray:
    """Per-step cell differences between two runs (over the shared prefix)."""
    n = min(len(a), len(b))
    return (a.grids[:n] != b.grids[:n]).sum(axis=(1, 2))


def action_divergence(a: Trajectory, b: Trajectory) -> dict[str, Any]:
    """Compare the action sequences of two runs.

    Returns:
        Dict with `first_divergence` (step index or None), `equal_mask`
        (bool array over the shared prefix; CLICK actions compare x/y too),
        and `n_shared` (length of the shared prefix).
    """
    n = min(len(a), len(b))
    equal = np.zeros(n, dtype=bool)
    for i in range(n):
        sa, sb = a.steps[i], b.steps[i]
        equal[i] = (sa.action_id, sa.x, sa.y) == (sb.action_id, sb.x, sb.y)
    diverging = np.nonzero(~equal)[0]
    return {
        "first_divergence": int(diverging[0]) if len(diverging) else None,
        "equal_mask": equal,
        "n_shared": n,
    }


def replay_visual(traj: Trajectory):
    """Single-step replay view driven entirely by `$state` (artifact-friendly).

    State keys: `pixels` (flat RGB of the current frame), `step`, `action`,
    `levels`, `game_state`. `step_updates` below emits per-step update entries
    against exactly these keys.
    """
    import colight.plot as Plot
    from colight.components.bitmap import bitmap
    from colight.plot import js

    first = traj.steps[0]
    return Plot.State(
        {
            "pixels": PALETTE[first.grid].reshape(-1),
            "step": 0,
            "action": first.label,
            "levels": first.levels,
            "game_state": first.state,
        }
    ) | [
        "div.flex.flex-col.gap-2.p-2.max-w-[400px]",
        bitmap(js("$state.pixels"), width=GRID_SIZE, height=GRID_SIZE),
        [
            "div.font-mono.text-sm",
            js(
                "`"
                + f"{traj.game_id} [{traj.agent}] "
                + "step ${$state.step}  ${$state.action}  "
                + "levels ${$state.levels}"
                + f"/{traj.win_levels}"
                + "  ${$state.game_state}`"
            ),
        ],
    ]


def step_updates(traj: Trajectory) -> list[dict[str, Any]]:
    """One state-update dict per step (skipping step 0, the initial state)."""
    rgb = traj.rgb()
    return [
        {
            "pixels": rgb[s.index].reshape(-1),
            "step": s.index,
            "action": s.label,
            "levels": s.levels,
            "game_state": s.state,
        }
        for s in traj.steps[1:]
    ]


def build_artifact(traj: Trajectory, out_path: Union[str, Path]) -> Path:
    """Write a trajectory as a `.colight` artifact: initial state + one update
    entry per step.

    The result renders as a replay film via
    `colight render <artifact> --out replay.mp4`.

    Args:
        traj: Trajectory to convert.
        out_path: Destination `.colight` path.

    Returns:
        The output path.
    """
    from colight import format as colight_format

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    replay_visual(traj).save_file(str(out_path))
    colight_format.append_updates(out_path, step_updates(traj))
    return out_path
