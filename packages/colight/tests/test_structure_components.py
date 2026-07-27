"""Component roster collection for nested scene3d components.

Scene components nested inside another component's props (e.g. a Group's
``children``) serialize as plain ``{"type": ...}`` config dicts rather than
function nodes; the structure walker must still report them as components.
"""

import numpy as np

from colight import scene3d
from colight.cli_tools.structure import collect_structure
from colight.widget import to_json_with_state

POSITIONS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
INDICES = np.array([0, 1, 2], dtype=np.uint32)


def _mesh() -> scene3d.SceneComponent:
    return scene3d.Mesh(positions=POSITIONS, indices=INDICES)


def _roster(scene: scene3d.Scene) -> list:
    data, buffers = to_json_with_state(scene)
    return [c.path for c in collect_structure(data, buffers).components]


def test_group_children_appear_in_component_roster():
    scene = scene3d.Scene(
        scene3d.Group(
            name="root",
            children=[
                _mesh(),
                scene3d.Group(
                    name="elbow",
                    position=[0.0, 0.0, 2.0],
                    children=[_mesh()],
                ),
            ],
        )
    )
    roster = _roster(scene)
    assert roster.count("scene3d.Mesh") == 2
    assert roster.count("scene3d.Group") >= 2  # root + nested elbow


def test_array_paths_unchanged_by_component_recognition():
    # Array paths are the diff/inspect addressing scheme; recognizing nested
    # components must not rename anything beneath them.
    scene = scene3d.Scene(
        scene3d.Group(name="root", children=[_mesh()]),
    )
    data, buffers = to_json_with_state(scene)
    state = collect_structure(data, buffers)
    position_paths = [
        r.path for r in state.arrays if r.key == "positions" and r.values is not None
    ]
    assert position_paths, "mesh positions must be collected"
    for path in position_paths:
        assert "children[0]" in path
        assert "/scene3d.Mesh" not in path


def test_unrelated_type_key_is_not_a_component():
    # A plain dict carrying a `type` key outside a scene3d subtree must not be
    # misread as a component.
    import colight.plot as Plot

    plot = Plot.dot({"x": [1, 2], "y": [3, 4], "type": ["a", "b"]})
    data, buffers = to_json_with_state(plot)
    roster = [c.path for c in collect_structure(data, buffers).components]
    assert not any(p.startswith("scene3d.") for p in roster)
