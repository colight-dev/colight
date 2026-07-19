from colight import scene3d
from colight import plot


def handle_drag(_widget, info):
    # info.delta["position"] contains the world-space movement
    # Note: nested dicts need bracket access, top-level uses attribute access
    print(f"Dragged to {info.current['position']}")


scene3d.Cuboid(
    centers=[[0, 0, 0]],
    half_size=0.5,
    on_drag=plot.js("console.log"),
    drag_constraint=scene3d.DRAG_PLANE_XZ,  # Drag on ground plane
)
