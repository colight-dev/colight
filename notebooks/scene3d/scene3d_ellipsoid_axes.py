from colight.scene3d import Ellipsoid, deco
import numpy as np
from colight.plot import js

Ellipsoid(
    # fill_mode="MajorWireframe",
    centers=np.array([[1, 0, 0], [1, 1, 0], [1, 0.5, 1]]),  # Offset by 1 in x direction
    color=[0, 1, 0],
    half_sizes=[0.4, 0.4, 0.4],
    alpha=0.8,
    quaternions=np.array(
        [[0.866, 0, 0.5, 0], [0.707, 0.707, 0, 0], [0.5, 0.5, 0.5, 0.5]]
    ),
    onHover=js("(i) => $state.update({hover_axes1: typeof i === 'number' ? [i] : []})"),
    decorations=[
        deco([2], color=[1, 0, 0]),
        deco(
            js("$state.hover_axes1"),
            color=[1, 1, 0],
        ),
        deco([0], scale=1.5),
        deco([1], scale=0.5),
    ],
) & Ellipsoid(
    fill_mode="MajorWireframe",
    centers=np.array([[1, 0, 0], [1, 1, 0], [1, 0.5, 1]]),  # Offset by 1 in x direction
    color=[0, 1, 0],
    half_sizes=[0.4, 0.4, 0.4],
    alpha=0.8,
    quaternions=np.array(
        [[0.866, 0, 0.5, 0], [0.707, 0.707, 0, 0], [0.5, 0.5, 0.5, 0.5]]
    ),
    onHover=js("(i) => $state.update({hover_axes1: typeof i === 'number' ? [i] : []})"),
    decorations=[
        deco([2], color=[1, 0, 0]),
        deco(
            js("$state.hover_axes1"),
            color=[1, 1, 0],
        ),
        deco([0], scale=1.5),
        deco([1], scale=0.5),
    ],
)


(
    Ellipsoid(
        fill_mode="MajorWireframe",
        centers=np.array(
            [
                [2, 0, 0],  # Row 1: Small shapes
                [4, 0, 0],
                [6, 0, 0],
                [8, 0, 0],
                [2, 2, 0],  # Row 2: Medium shapes
                [4, 2, 0],
                [6, 2, 0],
                [8, 2, 0],
                [2, 4, 0],  # Row 3: Large shapes
                [4, 4, 0],
                [6, 4, 0],
                [8, 4, 0],
            ]
        ),
        color=[1, 0, 1],
        half_sizes=[
            [0.2, 0.2, 0.05],  # Tiny pancake
            [0.1, 0.1, 0.4],  # Small needle
            [0.3, 0.3, 0.3],  # Small sphere
            [0.4, 0.2, 0.2],  # Small squashed
            [1.0, 1.0, 0.2],  # Medium disk
            [0.5, 0.5, 1.2],  # Medium elongated
            [0.8, 0.8, 0.8],  # Medium sphere
            [1.0, 0.4, 0.4],  # Medium flattened
            [2.5, 2.5, 0.3],  # Giant pancake
            [0.4, 0.4, 3.0],  # Tall needle
            [1.8, 1.8, 1.8],  # Large sphere
            [2.0, 1.0, 0.5],  # Large ellipsoid
        ],
        quaternions=np.array(
            [
                [1, 0, 0, 0],  # No rotation
                [0.707, 0.707, 0, 0],  # 90° around Y
                [1, 0, 0, 0],  # No rotation
                [0.866, 0, 0.5, 0],  # Slight tilt
                [0.5, 0.5, 0.5, 0.5],  # Complex rotation
                [0.707, 0, 0, 0.707],  # 90° around X
                [0.866, 0.5, 0, 0],  # Mixed rotation
                [0.5, 0.866, 0, 0],  # Another angle
                [0.707, -0.707, 0, 0],  # -90° around Y
                [0, 0.866, 0.5, 0],  # Complex tilt
                [1, 0, 0, 0],  # No rotation
                [0.383, 0.924, 0, 0],  # About 110° rotation
            ]
        ),
        onHover=js(
            "(i) => $state.update({hover_axes2: typeof i === 'number' ? [i] : []})"
        ),
        decorations=[
            deco([2], color=[1, 0, 0]),
            deco(
                js("$state.hover_axes2"),
                color=[1, 1, 0],
            ),
            deco([0], scale=1.5),
            deco([1], scale=0.5),
        ],
    )
    + {
        "defaultCamera": {
            "position": [6.261824, 2.239899, 1.435801],
            "target": [0.000000, 0.000000, 0.000000],
            "up": [0.000000, 1.000000, 0.000000],
            "fov": 45,
        }
    }
)
