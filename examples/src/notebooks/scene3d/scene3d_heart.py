# %%
import colight.plot as Plot
import colight.scene3d as Scene3D
from colight.plot import js

# %% [markdown]
# # Interactive Heart-Shaped Particle System
#
# This example demonstrates how to create an interactive 3D particle system
# where particles form a heart shape. We'll use JavaScript to generate the
# particles dynamically based on state parameters controlled by UI elements.

# %%
# Create the scene with interactive controls
(
    Plot.State(
        {
            "num_particles": 100000,
            "alpha": 1.0,
            "frame": 0,
            "point_size": 0.003,
            # Pre-generate frames using JavaScript.
            #
            # `n` is the per-frame particle budget, precomputed `num_frames`
            # times up front. Two separate costs scale with it:
            #
            #  - Load: n * num_frames points to generate and hold resident.
            #    The original 1M x 30 was ~5s of generation and ~343MB.
            #  - Per animation frame: the `frame` slider below runs at
            #    fps="raf", so every tick re-slices `n` centers and re-uploads
            #    them. Each render registers a pending update with the ready
            #    tracker, and if a tick's upload cannot finish before the next
            #    one starts, the pending count never returns to zero and the
            #    page never reports itself loaded -- headless capture then
            #    fails on the load timeout rather than rendering slowly.
            #    100k keeps a raf tick comfortably ahead of the next one.
            "frames": js(
                """
                const n = 100000;
                const num_frames = 30;
                const frames = [];  // Use regular array to store Float32Arrays

                // Pre-compute some values to speed up generation
                const uSteps = 50;
                const vSteps = 25;
                const uStep = (2 * Math.PI) / uSteps;
                const vStep = Math.PI / vSteps;
                const jitter = 0.1;
                const scale = 0.04;

                for (let frame = 0; frame < num_frames; frame++) {
                    const t = frame * (2 * Math.PI / num_frames); // Normalize t to complete one full cycle
                    const frameData = new Float32Array(n * 3);
                    let idx = 0;

                    // Generate points uniformly through the volume
                    for (let i = 0; i < n; i++) {
                        let x, y, z;
                        // Generate points in parametric space with volume filling
                        const u = Math.random() * 2 * Math.PI;
                        const v = Math.random() * Math.PI;
                        const r = Math.cbrt(Math.random()); // Cube root for volume filling

                        // Heart shape parametric equations with radial scaling
                        x = 16 * Math.pow(Math.sin(u), 3) * r;
                        y = (13 * Math.cos(u) - 5 * Math.cos(2*u) - 2 * Math.cos(3*u) - Math.cos(4*u)) * r;
                        z = 8 * Math.sin(v) * r;

                        // Add subtle jitter for more natural look
                        const rx = (Math.random() - 0.5) * jitter * 0.5;
                        const ry = (Math.random() - 0.5) * jitter * 0.5;
                        const rz = (Math.random() - 0.5) * jitter * 0.5;

                        // Scale and animate
                        x = (x * scale + rx) * (1 + 0.1 * Math.sin(t + u));
                        y = (y * scale + ry) * (1 + 0.1 * Math.sin(t + v));
                        z = (z * scale + rz) * (1 + 0.1 * Math.cos(t + u));

                        frameData[idx++] = x;
                        frameData[idx++] = y;
                        frameData[idx++] = z;
                    }

                    frames.push(frameData);
                }

                return frames;
            """,
                expression=False,
            ),
            # Pre-generate colors (these don't change per frame).
            # Must cover the same per-frame budget as `frames` above.
            "colors": js(
                """
                const n = 100000;
                const colors = new Float32Array(n * 3);

                for (let i = 0; i < n; i++) {
                    // Create a gradient from red to pink based on height
                    const y = i / n;
                    colors[i*3] = 1.0;  // Red
                    colors[i*3 + 1] = 0.2 + y * 0.3;  // Green
                    colors[i*3 + 2] = 0.4 + y * 0.4;  // Blue
                }

                return colors;
            """,
                expression=False,
            ),
        }
    )
    | [
        "div.flex.gap-4.mb-4",
        # Particle count slider
        [
            "label.flex.items-center.gap-2",
            "Particles: ",
            [
                "input",
                {
                    "type": "range",
                    "min": 100,
                    # Bounded by the per-frame budget generated above: slicing
                    # past it would hand the scene a short/undefined array.
                    "max": 100000,
                    "step": 1000,
                    "value": js("$state.num_particles"),
                    "onChange": js(
                        """(e) => $state.update({num_particles: parseInt(e.target.value)})"""
                    ),
                },
            ],
            js("$state.num_particles"),
        ],
        # Alpha control
        [
            "label.flex.items-center.gap-2",
            "Alpha: ",
            [
                "input",
                {
                    "type": "range",
                    "min": 0,
                    "max": 1,
                    "step": 0.1,
                    "value": js("$state.alpha"),
                    "onChange": js(
                        "(e) => $state.update({alpha: parseFloat(e.target.value)})"
                    ),
                },
            ],
            js("$state.alpha"),
        ],
        # Point size control
        [
            "label.flex.items-center.gap-2",
            "Point Size: ",
            [
                "input",
                {
                    "type": "range",
                    "min": 0.00015,
                    "max": 0.02,
                    "step": 0.00001,
                    "value": js("$state.point_size"),
                    "onChange": js(
                        "(e) => $state.update({point_size: parseFloat(e.target.value)})"
                    ),
                },
            ],
            js("$state.point_size"),
        ],
    ]
    | Scene3D.Cuboid(
        # Use pre-generated frames based on animation state
        centers=js("$state.frames[$state.frame % 30].slice(0, $state.num_particles*3)"),
        colors=js("$state.colors.slice(0, $state.num_particles* 3 )"),
        half_size=js("$state.point_size"),
        alpha=js("$state.alpha"),
    )
    + {
        "defaultCamera": {
            "position": [0.1, 0.1, 2],  # Adjusted position to view the heart head-on
            "target": [0, 0, 0],  # Keeping the target at the center
            "up": [0, 1, 0],  # Adjusted up vector for correct orientation
            "fov": 45,
            "near": 0.1,
            "far": 100,
        },
        "controls": ["fps"],
    }
    | Plot.Slider("frame", range=120, fps="raf")
)
