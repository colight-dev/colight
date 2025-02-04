# %%
import genstudio.plot as Plot
import genstudio.scene3d as Scene3D
from genstudio.plot import js

# %% [markdown]
# # Interactive Heart-Shaped Particle System
#
# This example demonstrates how to create an interactive 3D particle system
# where particles form a heart shape. We'll use JavaScript to generate the
# particles dynamically based on state parameters controlled by UI elements.

# %%
# Create the scene with interactive controls
(
    Plot.initialState(
        {
            "num_particles": 1000,
            "alpha": 0.8,
            "frame": 0,
            # Pre-generate frames using JavaScript
            "frames": js("""(() => {
                const n = $state.num_particles;
                const num_frames = 30;
                const frames = [];  // Use regular array to store Float32Arrays

                for (let frame = 0; frame < num_frames; frame++) {
                    const t = frame * 0.05;
                    const frameData = new Float32Array(n * 3);

                    for (let i = 0; i < n; i++) {
                        // Generate points in a heart shape using parametric equations
                        const u = Math.random() * 2 * Math.PI;
                        const v = Math.random() * Math.PI;
                        const jitter = 0.1;

                        // Heart shape parametric equations
                        const x = 16 * Math.pow(Math.sin(u), 3);
                        const y = 13 * Math.cos(u) - 5 * Math.cos(2*u) - 2 * Math.cos(3*u) - Math.cos(4*u);
                        const z = 8 * Math.sin(v);

                        // Add some random jitter and animation
                        const rx = (Math.random() - 0.5) * jitter;
                        const ry = (Math.random() - 0.5) * jitter;
                        const rz = (Math.random() - 0.5) * jitter;

                        // Scale down the heart and add animation
                        frameData[i*3] = (x * 0.04 + rx) * (1 + 0.1 * Math.sin(t + u));
                        frameData[i*3 + 1] = (y * 0.04 + ry) * (1 + 0.1 * Math.sin(t + v));
                        frameData[i*3 + 2] = (z * 0.04 + rz) * (1 + 0.1 * Math.cos(t + u));
                    }

                    frames.push(frameData);
                }

                return frames;
            })()"""),
            # Pre-generate colors (these don't change per frame)
            "colors": js("""(() => {
                const n = $state.num_particles;
                const colors = new Float32Array(n * 3);

                for (let i = 0; i < n; i++) {
                    // Create a gradient from red to pink based on height
                    const y = i / n;
                    colors[i*3] = 1.0;  // Red
                    colors[i*3 + 1] = 0.2 + y * 0.3;  // Green
                    colors[i*3 + 2] = 0.4 + y * 0.4;  // Blue
                }

                return colors;
            })()"""),
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
                    "max": 500000,
                    "step": 100,
                    "value": js("$state.num_particles"),
                    "onChange": js("""(e) => {
                        const n = parseInt(e.target.value);
                        // When particle count changes, regenerate frames and colors
                        $state.update({
                            num_particles: n,
                            frames: (() => {
                                const num_frames = 30;
                                const frames = [];  // Use regular array to store Float32Arrays

                                for (let frame = 0; frame < num_frames; frame++) {
                                    const t = frame * 0.05;
                                    const frameData = new Float32Array(n * 3);

                                    for (let i = 0; i < n; i++) {
                                        const u = Math.random() * 2 * Math.PI;
                                        const v = Math.random() * Math.PI;
                                        const jitter = 0.1;

                                        const x = 16 * Math.pow(Math.sin(u), 3);
                                        const y = 13 * Math.cos(u) - 5 * Math.cos(2*u) - 2 * Math.cos(3*u) - Math.cos(4*u);
                                        const z = 8 * Math.sin(v);

                                        const rx = (Math.random() - 0.5) * jitter;
                                        const ry = (Math.random() - 0.5) * jitter;
                                        const rz = (Math.random() - 0.5) * jitter;

                                        frameData[i*3] = (x * 0.04 + rx) * (1 + 0.1 * Math.sin(t + u));
                                        frameData[i*3 + 1] = (y * 0.04 + ry) * (1 + 0.1 * Math.sin(t + v));
                                        frameData[i*3 + 2] = (z * 0.04 + rz) * (1 + 0.1 * Math.cos(t + u));
                                    }

                                    frames.push(frameData);
                                }

                                return frames;
                            })(),
                            colors: (() => {
                                const colors = new Float32Array(n * 3);
                                for (let i = 0; i < n; i++) {
                                    const y = i / n;
                                    colors[i*3] = 1.0;
                                    colors[i*3 + 1] = 0.2 + y * 0.3;
                                    colors[i*3 + 2] = 0.4 + y * 0.4;
                                }
                                return colors;
                            })()
                        });
                    }"""),
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
    ]
    | Scene3D.PointCloud(
        # Use pre-generated frames based on animation state
        positions=js("$state.frames[$state.frame % 30]"),
        colors=js("$state.colors"),
        size=0.01,
        alpha=js("$state.alpha"),
    )
    + {
        "defaultCamera": {
            "position": [2, 2, 2],
            "target": [0, 0, 0],
            "up": [0, 0, 1],
            "fov": 45,
            "near": 0.1,
            "far": 100,
        },
        "controls": ["fps"],
    }
    | Plot.Slider("frame", range=120, fps="raf")
)
