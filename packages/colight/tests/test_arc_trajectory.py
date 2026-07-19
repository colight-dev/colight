"""Tests for the ARC-AGI-3 trajectory loader/conversion example module.

These run entirely offline: they use small committed fixtures under
examples/src/notebooks/arc_agi/fixtures plus synthetic trajectories.
"""

from pathlib import Path

import numpy as np
import pytest

from colight.format import parse_file_with_updates
from notebooks.arc_agi import trajectory as arc
from notebooks.arc_agi.trajectory import (
    Step,
    Trajectory,
    action_divergence,
    build_artifact,
    grid_change_counts,
    grid_diff_counts,
    load_fixture,
    save_fixture,
    step_updates,
)

FIXTURES = Path(arc.__file__).parent / "fixtures"


def make_traj(actions: list[tuple[int, int | None, int | None]], seed: int = 0):
    """Build a synthetic trajectory with random 64x64 grids."""
    rng = np.random.default_rng(seed)
    steps = [
        Step(
            index=i,
            action_id=action_id,
            x=x,
            y=y,
            state="NOT_FINISHED",
            levels=0,
            grid=rng.integers(0, 16, size=(64, 64), dtype=np.uint8),
        )
        for i, (action_id, x, y) in enumerate(actions)
    ]
    return Trajectory(
        game_id="test-game", agent="test", guid="g", win_levels=3, steps=steps
    )


def test_load_committed_fixture():
    traj = load_fixture(FIXTURES / "ls20-run-a.json.gz")
    assert traj.game_id == "ls20-9607627b"
    assert len(traj) == 81
    assert traj.steps[0].action_id == 0  # normalized RESET
    grids = traj.grids
    assert grids.shape == (81, 64, 64)
    assert grids.dtype == np.uint8
    assert grids.max() <= 15
    assert traj.rgb().shape == (81, 64, 64, 3)
    assert traj.flat_rgb().shape == (81, 64 * 64 * 3)


def test_fixture_roundtrip(tmp_path):
    traj = make_traj([(0, None, None), (1, None, None), (6, 10, 20)])
    path = save_fixture(traj, tmp_path / "t.json.gz")
    loaded = load_fixture(path)
    assert len(loaded) == len(traj)
    assert np.array_equal(loaded.grids, traj.grids)
    assert np.array_equal(loaded.actions, traj.actions)
    assert loaded.steps[2].x == 10 and loaded.steps[2].y == 20
    assert loaded.win_levels == traj.win_levels


def test_rgb_uses_palette():
    traj = make_traj([(0, None, None)])
    rgb = traj.rgb()
    grid = traj.grids[0]
    assert np.array_equal(rgb[0][grid == 5], np.tile([0, 0, 0], ((grid == 5).sum(), 1)))
    assert np.array_equal(
        rgb[0][grid == 0], np.tile([255, 255, 255], ((grid == 0).sum(), 1))
    )


def test_action_divergence():
    a = make_traj([(0, None, None), (1, None, None), (6, 5, 5), (2, None, None)])
    b = make_traj([(0, None, None), (1, None, None), (6, 5, 6), (2, None, None)])
    div = action_divergence(a, b)
    assert div["n_shared"] == 4
    assert div["first_divergence"] == 2  # CLICK coordinates differ
    assert div["equal_mask"].tolist() == [True, True, False, True]


def test_action_divergence_identical():
    a = make_traj([(0, None, None), (3, None, None)])
    b = make_traj([(0, None, None), (3, None, None)])
    div = action_divergence(a, b)
    assert div["first_divergence"] is None
    assert div["equal_mask"].all()


def test_grid_diff_and_change_counts():
    a = make_traj([(0, None, None), (1, None, None)], seed=1)
    b = make_traj([(0, None, None), (1, None, None)], seed=1)
    assert grid_diff_counts(a, b).tolist() == [0, 0]
    changes = grid_change_counts(a)
    assert changes[0] == 0
    assert changes[1] == (a.grids[0] != a.grids[1]).sum()


def test_build_artifact_has_update_entries(tmp_path):
    traj = make_traj(
        [(0, None, None), (1, None, None), (6, 3, 4), (2, None, None)], seed=2
    )
    path = build_artifact(traj, tmp_path / "run.colight")
    data, buffers, updates = parse_file_with_updates(path)
    assert data is not None
    state_keys = set((data.get("state") or {}).keys())
    assert state_keys == {"pixels", "step", "action", "levels", "game_state"}
    assert len(updates) == len(traj) - 1
    # Every update entry must target known state keys only - the contract
    # `colight render` relies on to apply updates as state patches.
    for entry in updates:
        ast = entry["data"].get("ast")
        assert isinstance(ast, dict)
        assert set(ast.keys()) <= state_keys
    # Each update carries the frame's RGB buffer.
    assert all(len(entry["buffers"]) == 1 for entry in updates)


def test_step_updates_align_with_steps():
    traj = make_traj([(0, None, None), (5, None, None), (6, 1, 2)])
    updates = step_updates(traj)
    assert [u["step"] for u in updates] == [1, 2]
    assert updates[1]["action"] == "CLICK(1,2)"
    assert updates[0]["pixels"].shape == (64 * 64 * 3,)


def test_loader_tolerates_missing_fixture():
    with pytest.raises(FileNotFoundError):
        load_fixture(FIXTURES / "does-not-exist.json.gz")
