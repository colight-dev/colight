# # ARC-AGI-3 replay browser
#
# Scrub through a recorded agent run: the game frame, the action taken at each
# step, and how much of the grid changed. The trajectory was recorded with the
# official random agent on the public game `ls20` and is loaded from a small
# committed fixture (no API access required).

from pathlib import Path

import colight.plot as Plot
from colight.components.bitmap import bitmap
from colight.plot import js

from notebooks.arc_agi.trajectory import grid_change_counts, load_fixture

FIXTURES = Path(__file__).parent / "fixtures"

traj = load_fixture(FIXTURES / "ls20-run-a.json.gz")

# ## The replay
#
# All frames are preloaded into state as flat RGB arrays; the slider drives
# which frame the bitmap shows. The timeline underneath plots the action taken
# at each step (CLICK actions are coordinate clicks; the rest are keyboard
# actions) plus the number of grid cells that changed, with a rule marking the
# current step.

changes = grid_change_counts(traj)

action_rows = [
    {"step": s.index, "action": s.action_name, "changed": int(changes[s.index])}
    for s in traj.steps
]

(
    Plot.State(
        {
            "step": 0,
            "frames": traj.flat_rgb(),
            "labels": [s.label for s in traj.steps],
            "states": [s.state for s in traj.steps],
            "levels": [int(v) for v in traj.levels],
        }
    )
    | [
        "div.flex.flex-col.gap-2.max-w-[440px]",
        bitmap(js("$state.frames[$state.step]"), width=64, height=64),
        [
            "div.font-mono.text-sm",
            js(
                "`step ${$state.step}  ${$state.labels[$state.step]}  "
                + "levels ${$state.levels[$state.step]}"
                + f"/{traj.win_levels}"
                + "  ${$state.states[$state.step]}`"
            ),
        ],
    ]
    | Plot.Slider("step", range=len(traj), controls=["play", "slider"], fps=8)
    | Plot.dot(
        action_rows,
        x="step",
        y="action",
        fill="action",
        r=3,
    )
    + Plot.ruleX(js("[$state.step]"), stroke="black", strokeWidth=1.5)
    + Plot.colorLegend()
    + {"height": 180, "width": 440, "marginLeft": 70}
    | Plot.line(action_rows, x="step", y="changed", stroke="steelblue")
    + Plot.ruleX(js("[$state.step]"), stroke="black", strokeWidth=1.5)
    + Plot.title("grid cells changed per step")
    + {"height": 140, "width": 440}
)
