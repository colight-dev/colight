# | hide-code hide-prose

# # Colight Comprehensive Feature Showcase 102
#
# This example demonstrates most key features of the Colight library on one page:
#
# - Layout operators (| for vertical, & for horizontal)
# - 3D scenes with multiple primitives
# - Observable plots with various chart types
# - Animation and interactivity
# - State management between Python and JavaScript

import colight.plot as Plot
from colight.plot import js
from colight.scene3d import PointCloud, Ellipsoid, Cuboid
import numpy as np

counter_section = (
    Plot.State({"clicks": 0, "hover_count": 0})
    | Plot.html(
        [
            "div",
            {
                "class": "bg-blue-200 p-8 rounded cursor-pointer select-none text-center font-bold",
                "onClick": js("(e) => $state.clicks = ($state.clicks || 0) + 1"),
                "onMouseEnter": js(
                    "(e) => $state.hover_count = ($state.hover_count || 0) + 1"
                ),
            },
            [
                "div.flex.flex-col.gap-4",
                js("`Clicked ${$state.clicks} times`"),
                js("`Hovered ${$state.hover_count} times`"),
            ],
        ]
    )
    &  # Horizontal layout operator
    # Right side: info panel
    [
        "div.bg-gray-100.p-4.rounded.ml-4",
        ["h4.font-semibold", "Layout Demo"],
        ["p", "This demonstrates the & (horizontal) layout operator."],
        ["p.text-sm.text-gray-600", "Click the blue box to increment counter."],
    ]
)
counter_section


np.random.seed(42)
spiral_t = np.linspace(0, 4 * np.pi, 100)
spiral_centers = np.column_stack(
    [
        0.5 * spiral_t / (4 * np.pi) * np.cos(spiral_t),
        0.5 * spiral_t / (4 * np.pi) * np.sin(spiral_t),
        spiral_t / (4 * np.pi) - 0.5,
    ]
)
spiral_colors = np.column_stack(
    [
        np.sin(spiral_t) * 0.5 + 0.5,
        np.cos(spiral_t) * 0.5 + 0.5,
        np.ones_like(spiral_t) * 0.8,
    ]
)

scene_3d = (
    Plot.State({"rotation": 0})
    | (
        # Static spiral point cloud
        PointCloud(centers=spiral_centers, colors=spiral_colors, size=0.02)
        +
        # Static ellipsoids
        Ellipsoid(
            centers=[[-0.5, 0, 0], [0.5, 0, 0]],
            half_size=[0.1, 0.1, 0.1],
            colors=[[1, 0.5, 0], [0, 0.5, 1]],
            alphas=[0.8, 0.8],
        )
        +
        # Static cuboids
        Cuboid(
            centers=[[-0.3, 0, -0.3], [0.3, 0, 0.3]],
            half_sizes=[0.05, 0.05, 0.05],
            colors=[[1, 0, 1], [0, 1, 0]],
            alpha=0.9,
        )
        + {
            "camera": {
                "position": js("""[
                    Math.cos($state.rotation * Math.PI / 180) * 1.5,
                    Math.sin($state.rotation * Math.PI / 180) * 1.5,
                    1.5
                ]"""),
                "target": [0, 0, 0],
                "up": [0, 0, 1],
            },
            "height": 250,
        }
    )
    | Plot.Slider("rotation", range=[0, 360], fps=30, controls=[])
)
scene_3d


x_data = np.linspace(0, 10, 50)
y_data = np.sin(x_data) + np.random.normal(0, 0.1, 50)
scatter_data = np.column_stack([x_data, y_data])

categories = ["Alpha", "Beta", "Gamma", "Delta"]
values = [23, 45, 12, 38]
bar_data = [{"category": c, "value": v} for c, v in zip(categories, values)]

plot_gallery = (
    Plot.State({"wave_phase": 0})
    | Plot.Grid(
        # Animated sine wave
        (
            Plot.line(
                {"x": range(100)},
                {
                    "y": js(
                        "(d, i) => Math.sin(i * 2 * Math.PI / 100 + $state.wave_phase)"
                    )
                },
            )
            + Plot.domain([0, 99], [-1.2, 1.2])
        ),
        # Scatter plot with trend line
        (
            Plot.dot(scatter_data, {"fill": "steelblue", "r": 4})
            + Plot.line(scatter_data, {"stroke": "red", "strokeWidth": 2})
        ),
        # Bar chart
        (Plot.barY(bar_data, {"x": "category", "y": "value", "fill": "steelblue"})),
        # Area chart
        (
            Plot.area(
                {"x": range(50)}, {"y": js("(d, i) => Math.exp(-i/20) * Math.cos(i/5)")}
            )
            + {"height": 200}
        ),
        cols=2,
    )
    | Plot.Slider("wave_phase", range=[0, 2 * np.pi], fps=20, controls=[])
)
plot_gallery


def generate_ripple_pixels(width=60, height=60, num_frames=30):
    x, y = np.meshgrid(np.linspace(-2, 2, width), np.linspace(-2, 2, height))
    frames = []

    for frame in range(num_frames):
        t = frame * 2 * np.pi / num_frames
        r = np.sqrt(x**2 + y**2)

        # Create ripple effect
        intensity = np.sin(r * 5 - t) * np.exp(-r)

        # Convert to RGB
        red = np.clip((intensity + 1) * 127, 0, 255)
        green = np.clip((np.sin(r * 3 - t * 0.7) + 1) * 127, 0, 255)
        blue = np.clip((np.cos(r * 4 - t * 1.3) + 1) * 127, 0, 255)

        # Stack RGB channels and flatten
        rgb = np.stack([red, green, blue], axis=-1)
        frames.append(rgb.reshape(-1).astype(np.uint8))

    return np.array(frames)


pixel_data = generate_ripple_pixels()

pixel_section = (
    Plot.State({"frame": 0, "pixels": pixel_data, "width": 60, "height": 60})
    | Plot.pixels(
        js("$state.pixels[$state.frame]"),
        imageWidth=js("$state.width"),
        imageHeight=js("$state.height"),
    )
    | Plot.Slider(
        "frame",
        rangeFrom=js("$state.pixels"),
        fps=15,
        controls=[],
        label="Animation Frame",
    )
)
pixel_section

drag_plot = (
    Plot.State({"points": [[0.2, 0.3], [0.5, 0.7], [0.8, 0.4]]})
    | ["h3", "Interactive Drag & Drop"]
    | (
        Plot.dot(
            js("$state.points"),
            {
                "r": 8,
                "fill": "orange",
                "stroke": "darkorange",
                "strokeWidth": 2,
                "render": Plot.renderChildEvents(
                    {
                        "onDrag": js("""(e) => {
                        $state.update(['points', 'setAt', [e.index, [e.x, e.y]]])
                    }""")
                    }
                ),
            },
        )
        + Plot.line(js("$state.points"), {"stroke": "blue", "strokeWidth": 2})
        + Plot.events(
            {"onClick": js("(e) => $state.update(['points', 'append', [e.x, e.y]])")}
        )
        + Plot.domain([0, 1], [0, 1])
        + Plot.grid()
        + {"height": 300}
    )
    | Plot.md("**Instructions:** Click to add points, drag to move them")
)
