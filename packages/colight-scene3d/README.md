# @colight/scene3d

Standalone Scene3D WebGPU renderer from [Colight](https://github.com/colight-dev/colight). Declarative 3D scenes as plain component configs rendered into a React `<Scene>`.

## Install

```bash
npm install @colight/scene3d
# or
yarn add @colight/scene3d
```

## Usage

```tsx
import React from "react";
import {
  Scene,
  PointCloud,
  Ellipsoid,
  Cuboid,
  LineBeams,
} from "@colight/scene3d";

const components = [
  PointCloud({
    centers: [0, 0, 0, 1, 1, 1],
    colors: [1, 0, 0, 0, 1, 0],
  }),
  Ellipsoid({
    centers: [0, 0, 0],
    half_size: 0.2,
    colors: [0, 0.5, 1],
  }),
];

export function Example() {
  return (
    <Scene
      components={components}
      width={640}
      height={480}
      controls={["fps"]}
    />
  );
}
```

## Components

- `PointCloud` — instanced camera-facing points
- `Ellipsoid` — spheres/ellipsoids, solid or major-wireframe fill mode
- `Cuboid` — boxes with optional per-instance quaternions
- `LineBeams` — connected beam segments (polylines)
- `LineSegments` — independent segments from `starts`/`ends`
- `Mesh` — arbitrary triangle geometry with optional normals, vertex colors, UVs, and textures
- `ImagePlane` / `ImageProjection` — textured image quads, optionally placed from camera intrinsics/extrinsics
- `CameraFrustum` — wireframe frustum from intrinsics/extrinsics
- `BoundingBox` — wireframe boxes
- `GridHelper` — reference grid
- `Group` — hierarchical transforms (position/quaternion/scale) over child components, with event bubbling and per-group style props
- `deco` — per-instance decorations (color/alpha/scale overrides)

Custom primitives can be defined via the declarative `PrimitiveSpec` interface.

## Interaction

- Picking: `onHover` / `onClick` receive rich pick info (instance index, world position, component/group names).
- Hover styling: `hoverProps` applies color/alpha/scale or an outline overlay automatically, no state management required.
- Dragging: `onDrag` / `onDragStart` / `onDragEnd` with `dragConstraint` — build constraints with `dragAxis` / `dragPlane` or use the `DRAG_AXIS_*` / `DRAG_PLANE_*` constants.
- Gizmo: `createTranslateGizmo` builds a translation manipulator (prototype; API may change).

## Utilities

Also exported: camera defaults (`DEFAULT_CAMERA`) and types, `Vec3` math (`add`, `sub`, `cross`, `normalize`, ...), ray/plane intersection helpers, screen projection (`screenRay`, `projectToScreen`), pointer context helpers, group/quaternion utilities (`flattenGroups`, `quatFromAxisAngle`, ...), and canvas snapshot helpers for screenshots/PDF export. See `index.ts` type exports for the full surface.

## Notes

- Requires a browser with WebGPU enabled.
- `react` is a peer dependency.
- For camera control, pass `camera`, `defaultCamera`, and `onCameraChange`.
