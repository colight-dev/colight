# Scene3D Live Cube Demo

Minimal demo showing a draggable cube with per-session Python state.

## Run

```bash
python examples/scene3d_live/server.py
```

Open the printed URL in a browser with WebGPU enabled.
If you want to copy/paste, it should look like:

```
http://127.0.0.1:8000/
```

## Implementation Notes

### Screen-to-World Coordinate Conversion

When implementing raycasting for interactive 3D picking and dragging:

1. **Y-Axis Convention**: Screen coordinates have Y increasing downward (0 at top), but NDC (Normalized Device Coordinates) have Y increasing upward (-1 at bottom, +1 at top). Always flip the Y coordinate:
   ```javascript
   const ndcY = -((y / rect.height) * 2 - 1);
   ```

2. **WebGPU vs WebGL**: While WebGPU and WebGL both use the same NDC convention (Y up), be aware of framebuffer Y-axis differences when dealing with texture coordinates or framebuffer operations.

3. **Perspective Matrix**: The standard perspective projection matrix expects `near` and `far` clip planes where `near < far`. The formula uses `1 / (near - far)` which produces negative values as expected.
