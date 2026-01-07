# Picking test scene - a simple cube at origin for testing pick info
# Run with: uv run python -m colight_cli serve packages/colight/tests/e2e/fixtures --port 8000

from colight.scene3d import Scene, Cuboid

# Create a simple axis-aligned cube at origin
Cuboid(
    centers=[0, 0, 0],
    half_size=[1, 1, 1],
    color=[0.2, 0.6, 1.0],
)
