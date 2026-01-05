# @colight/scene3d

Standalone Scene3D WebGPU renderer from Colight.

## Install

```bash
npm install @colight/scene3d
# or
yarn add @colight/scene3d
```

## Usage

```tsx
import React from "react";
import { Scene, PointCloud, Ellipsoid, Cuboid, LineBeams } from "@colight/scene3d";

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

## Notes

- Requires a browser with WebGPU enabled.
- `react` is a peer dependency.
- For camera control, pass `camera`, `defaultCamera`, and `onCameraChange`.

## API

Exports: `Scene`, `SceneWithLayers`, `PointCloud`, `Ellipsoid`, `Cuboid`, `LineBeams`, `deco`, `computeCanvasDimensions`, `SceneInner`, `DEFAULT_CAMERA`.
