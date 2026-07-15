# Translate Gizmo Prototype Demo
#
# This notebook demonstrates the translate gizmo prototype.
# The gizmo allows interactive translation of objects along
# constrained axes or planes.

from colight.scene3d import Scene, Cuboid, TranslateGizmo

# A cube that we want to manipulate
cube = Cuboid(centers=[[0, 0, 0]], half_size=0.5, color=[0.5, 0.5, 0.8])

# Create a translate gizmo at the cube's position
# The gizmo renders as an overlay (always visible, even when behind objects)
gizmo = TranslateGizmo(
    position=[0, 0, 0],
    scale=0.8,  # Adjust size to fit the scene
)

# Compose the scene
Scene(cube, gizmo)

# To make the gizmo interactive, you would use state management:
#
# import colight.state as State
#
# cube_pos = State.var([0, 0, 0])
#
# def handle_drag(info):
#     # Update cube position by drag delta
#     new_pos = [p + d for p, d in zip(cube_pos.value, info["delta"]["position"])]
#     cube_pos.set(new_pos)
#
# Scene(
#     Cuboid(centers=[cube_pos], half_size=0.5, color=[0.5, 0.5, 0.8]),
#     TranslateGizmo(position=cube_pos, on_drag=handle_drag, scale=0.8),
# )
