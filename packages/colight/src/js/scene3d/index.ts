// Scene components
export {
  Scene,
  PointCloud,
  Ellipsoid,
  Cuboid,
  LineBeams,
  deco,
  computeCanvasDimensions,
} from "./scene3d";

// Camera
export { DEFAULT_CAMERA } from "./camera3d";
export type { CameraParams, CameraState } from "./camera3d";

// Vec3 utilities
export type { Vec3 } from "./vec3";
export {
  add,
  sub,
  scale,
  dot,
  cross,
  length,
  normalize,
  distance,
  readVec3,
} from "./vec3";

// Ray/Plane utilities
export type { Ray, Plane } from "./ray";
export {
  intersectPlane,
  intersectPlaneT,
  pointOnRay,
  planeFromHit,
  nearestPointOnAxis,
  distanceToPlane,
  projectOntoPlane,
} from "./ray";

// Projection utilities
export { screenRay, projectToScreen, isPointVisible } from "./project";
export type { ScreenPoint, Rect } from "./project";

// Pointer context
export type { PointerContext, CursorHint, CursorType } from "./pointer";
export { createPointerContext } from "./pointer";

// Component types
export type {
  ComponentConfig,
  PointCloudComponentConfig,
  EllipsoidComponentConfig,
  CuboidComponentConfig,
  LineBeamsComponentConfig,
} from "./components";

// Other types
export { NOOP_READY_STATE } from "./types";
export type {
  Decoration,
  PickInfo,
  ReadyState,
  PickHit,
  PickRay,
} from "./types";

// Canvas snapshot utilities (for PDF export, screenshots)
export {
  createCanvasOverlays,
  removeCanvasOverlays,
  getCanvasCount,
} from "./canvasSnapshot";
