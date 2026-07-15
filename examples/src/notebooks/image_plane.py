import numpy as np

from colight.scene3d import ImagePlane, ImageProjection, GridHelper, Cuboid

# Create a simple RGB gradient image (values 0-1)
height, width = 64, 96
gradient = np.zeros((height, width, 3), dtype=np.float32)
gradient[:, :, 0] = np.linspace(0.0, 1.0, width)[None, :]
gradient[:, :, 1] = np.linspace(0.0, 1.0, height)[:, None]
gradient[:, :, 2] = 0.25

# Create a small checkerboard image (uint8)
checker = (np.indices((48, 48)).sum(axis=0) % 2) * 255
checker = np.stack([checker, checker, checker], axis=-1).astype(np.uint8)

plane_a = ImagePlane(
    gradient,
    position=[-0.9, 0.3, 0.0],
    width=1.2,
    height=0.8,
    opacity=0.95,
)
plane_b = ImagePlane(
    checker,
    position=[0.9, -0.2, 0.0],
    width=0.9,
    height=0.9,
    opacity=0.9,
)

intrinsics = {
    "fx": 480.0,
    "fy": 480.0,
    "cx": 320.0,
    "cy": 240.0,
    "width": 640.0,
    "height": 480.0,
}
extrinsics = {
    "position": [0.0, 0.4, -1.1],
    "quaternion": [0.0, 0.0, 0.0, 1.0],
}
projection = ImageProjection(
    gradient,
    intrinsics=intrinsics,
    extrinsics=extrinsics,
    depth=1.0,
    opacity=0.75,
    show_frustum=True,
    line_width=0.01,
)

grid = GridHelper(size=3.0, divisions=12)
cuboid = Cuboid(centers=[[0.0, -0.7, 0.0]], half_size=0.15, color=[0, 0.8, 0.2])

scene_props = {
    "defaultCamera": {
        "position": [0, 0, 3.2],
        "target": [0, 0, 0],
        "up": [0, 1, 0],
    },
}

scene = grid + cuboid + plane_a + plane_b + projection + scene_props

scene
