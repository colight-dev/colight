# # ARC-AGI-3 behavioral comparison: two runs, side by side
#
# Two runs of the official random agent (different seeds) on the public game
# `ls20`, loaded from committed fixtures. The point of this document is
# trajectory understanding: where do the two runs' actions diverge, how far
# apart do their world states drift, and how do their scores evolve?

from pathlib import Path

import colight.plot as Plot
from colight.components.bitmap import bitmap
from colight.plot import js

from notebooks.arc_agi.trajectory import (
    action_divergence,
    grid_diff_counts,
    load_fixture,
)

FIXTURES = Path(__file__).parent / "fixtures"

run_a = load_fixture(FIXTURES / "ls20-run-a.json.gz")

run_b = load_fixture(FIXTURES / "ls20-run-b.json.gz")

div = action_divergence(run_a, run_b)

state_diff = grid_diff_counts(run_a, run_b)

# ## Synced side-by-side replay
#
# One slider drives both runs. The HUD under each frame shows that run's
# action at the current step; the line chart shows how many grid cells differ
# between the two runs' world states at each step, with a rule at the current
# step and at the first action divergence.


def run_panel(name: str, key: str, run):
    return [
        "div.flex.flex-col.gap-1",
        ["div.font-bold.text-sm", f"{name} ({run.game_id})"],
        bitmap(js(f"$state.{key}[$state.step]"), width=64, height=64),
        [
            "div.font-mono.text-xs",
            js(f"`${{$state.{key}Labels[$state.step]}}`"),
        ],
    ]


(
    Plot.State(
        {
            "step": 0,
            "a": run_a.flat_rgb(),
            "b": run_b.flat_rgb(),
            "aLabels": [s.label for s in run_a.steps],
            "bLabels": [s.label for s in run_b.steps],
        }
    )
    | [
        "div.flex.flex-row.gap-4.max-w-[700px]",
        run_panel("run A", "a", run_a),
        run_panel("run B", "b", run_b),
    ]
    | Plot.Slider(
        "step", range=min(len(run_a), len(run_b)), controls=["play", "slider"], fps=8
    )
    | Plot.line(
        [{"step": i, "cells": int(v)} for i, v in enumerate(state_diff)],
        x="step",
        y="cells",
        stroke="firebrick",
    )
    + Plot.ruleX(js("[$state.step]"), stroke="black", strokeWidth=1.5)
    + Plot.ruleX(
        [div["first_divergence"]] if div["first_divergence"] is not None else [],
        stroke="orange",
        strokeDasharray="4 2",
    )
    + Plot.title("world-state divergence (grid cells differing, A vs B)")
    + {"height": 150, "width": 700}
)

# ## Where actions diverge
#
# Each lane is one run's action sequence; the bottom lane marks steps where
# the two runs chose the same action (for CLICK, same coordinates too).
# The dashed orange rule is the first divergence.

lanes = [
    {"step": s.index, "run": name, "action": s.action_name}
    for name, run in [("A", run_a), ("B", run_b)]
    for s in run.steps
] + [
    {"step": i, "run": "A = B", "action": "same" if eq else "different"}
    for i, eq in enumerate(div["equal_mask"])
    if eq
]

(
    Plot.dot(lanes, x="step", y="run", fill="action", r=3)
    + Plot.ruleX(
        [div["first_divergence"]] if div["first_divergence"] is not None else [],
        stroke="orange",
        strokeDasharray="4 2",
    )
    + Plot.colorLegend()
    + {"height": 140, "width": 700, "marginLeft": 50}
)

Plot.md(
    f"First action divergence at step **{div['first_divergence']}** "
    f"of {div['n_shared']} shared steps; "
    f"{int(div['equal_mask'].sum())} steps chose identical actions."
)

# ## Score trajectories
#
# Cumulative levels completed per step for both runs. (Random agents rarely
# complete a level, so flat lines at zero are the expected baseline here -
# the chart earns its keep once a heuristic starts clearing levels.)

(
    Plot.line(
        [
            {"step": s.index, "levels": s.levels, "run": name}
            for name, run in [("A", run_a), ("B", run_b)]
            for s in run.steps
        ],
        x="step",
        y="levels",
        stroke="run",
    )
    + Plot.colorLegend()
    + {"height": 140, "width": 700}
)
