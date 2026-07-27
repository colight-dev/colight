# Sliders
#
# Sliders allow users to dynamically adjust parameters. Each slider is bound to a reactive variable in `$state`, accessible in Plot.js functions as `$state.{key}`.
#
# Here's an example of a sine wave with an adjustable frequency:

import colight.plot as Plot

slider = Plot.Slider(
    key="frequency",
    label="Frequency:",
    showValue=True,
    range=[0.5, 5],
    step=0.1,
    init=1,
)

line = (
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js(
                """(d, i) => {
                    console.log($state, Math.sin(i * 2 * Math.PI / 100 * $state.frequency))
                return Math.sin(i * 2 * Math.PI / 100 * $state.frequency)
            }"""
            )
        },
    )
    + Plot.domain([0, 99], [-1, 1])
    + {"height": 300, "width": 500}
)

line | slider

# ## Animated Sliders
#
# Sliders can also be used to create animations. When a slider is given an [fps](bylight?match=fps=30) (frames per second) parameter, it automatically animates by updating [its value](bylight?match=$state.frame,key="frame") over time. This approach is useful when all frame differences can be expressed using JavaScript functions that read from $state variables.

(
    Plot.line(
        {"x": range(100)},
        {
            "y": Plot.js(
                """(d, i) => Math.sin(
                        i * 2 * Math.PI / 100 + 2 * Math.PI * $state.frame / 60
                    )"""
            )
        },
    )
    + Plot.domain([0, 99], [-1, 1])
) | Plot.Slider(
    key="frame", label="frame:", showValue=True, fps=30, showFps=True, range=[0, 59]
)

# ## Resampling declared values: `Plot.channel`
#
# A `Plot.js` expression computes a prop from `$state`, which is opaque: only
# the browser knows what it does. `Plot.channel` instead *declares* the same
# relationship — a table of sampled `values`, the coordinates `at` which they
# were sampled, the `$state` key that indexes them, and the `rule` for values
# in between. The samples travel once, as one array; the browser resamples them
# on every parameter change, so the sweep works with no Python attached and
# survives into a standalone `.colight` artifact.
#
# The rules are `"nearest"`, `"step"` (hold the lower sample), `"linear"`
# (elementwise lerp) and `"qlerp"` (normalized quaternion lerp with antipodal
# correction, for (N, 4) xyzw rows — the right rule for rotations, which must
# not be lerped componentwise). A channel carries no notion of time:
# `parameter="t"` is a clock, `parameter="grade"` a cutoff, `parameter="blend"`
# a morph. `colight inspect` reports each one — its parameter, sample domain,
# rule and the prop it drives — so what is sweepable is discoverable without
# reading the source.

import numpy as np

radii = Plot.channel("size", values=np.array([2.0, 8.0, 30.0]), at=[0.0, 0.5, 1.0])

(
    Plot.dot({"x": [1, 2, 3], "y": [1, 2, 3]}, {"r": radii})
    + Plot.domain([0, 4])
    + {"height": 250, "width": 400}
) | Plot.Slider(
    key="size", label="size:", showValue=True, init=0.5, range=[0, 1], step=0.01
)

# Channels fix what happens per frame, not what crosses the wire: a dense table
# — say one flattened set of vertex positions per pose — still ships in full,
# O(poses × vertices), once. What the declaration buys is that resampling it
# costs no Python round trip and no rebuild of what the renderer already holds.
