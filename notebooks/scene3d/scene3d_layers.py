from colight.scene3d import Scene, PointCloud, Ellipsoid
import colight.plot as Plot

# Create some example primitives
point_cloud = PointCloud(
    centers=[[0, 0, 0], [1, 1, 1], [-1, -1, -1]], color=[1, 0, 0], size=0.1
)

ellipsoid1 = Ellipsoid(centers=[[0.5, 0.5, 0.5]], half_size=0.2, color=[0, 1, 0])

ellipsoid2 = Ellipsoid(centers=[[-0.5, -0.5, -0.5]], half_size=0.15, color=[0, 0, 1])

# Create dictionary of layers
layers = {
    "Points": point_cloud,
    "Green Ellipsoid": ellipsoid1,
    "Blue Ellipsoid": ellipsoid2,
}


# Create a checkbox for each rendering
checkboxes = Plot.html(
    [
        "div.flex.flex-col",
        *[
            [
                "div.flex.gap-2",
                [
                    "input",
                    {
                        "type": "checkbox",
                        "checked": Plot.js("$state.layers.has(%1)", key),
                        "onChange": Plot.js("() => $state.toggleLayer(%1)", key),
                        "id": key,
                        "name": key,
                    },
                ],
                ["label", {"for": key}, key],
            ]
            for key in layers.keys()
        ],
    ]
)

# Create conditional components for each rendering
components = [
    Plot.cond(Plot.js("$state.layers.has(%1)", key), value)
    for key, value in layers.items()
]

# Combine into a scene with controls
(
    Scene(
        *components,
        {
            "defaultCamera": {
                "position": [-0.814902, -0.116913, -2.228022],
                "target": [0.016641, -0.314990, 1.128954],
                "up": [0.081127, -0.981787, 0.171791],
                "fov": 45,
            }
        },
    )
    | checkboxes
    | Plot.initialState(
        {
            "layers": Plot.js("new Set(%1)", layers.keys()),
            "toggleLayer": Plot.js(
                "(key) => { const s = new Set($state.layers); if (s.has(key)) s.delete(key); else s.add(key); $state.layers = s }"
            ),
        }
    )
)
