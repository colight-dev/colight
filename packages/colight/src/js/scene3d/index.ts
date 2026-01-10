// Scene components
export {
  Scene,
  PointCloud,
  Ellipsoid,
  Cuboid,
  LineBeams,
  BoundingBox,
  Group,
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
  HoverProps,
  PickInfo,
  ReadyState,
  PickHit,
  PickRay,
  DragConstraint,
  DragInfo,
} from "./types";

// Drag utilities
export {
  resolveConstraint,
  computeConstrainedPosition,
  hasDragCallbacks,
  dragAxis,
  dragPlane,
  DRAG_AXIS_X,
  DRAG_AXIS_Y,
  DRAG_AXIS_Z,
  DRAG_PLANE_XY,
  DRAG_PLANE_XZ,
  DRAG_PLANE_YZ,
} from "./drag";
export type { ResolvedConstraint } from "./drag";

// Canvas snapshot utilities (for PDF export, screenshots)
export { createCanvasOverlays, removeCanvasOverlays } from "../canvasSnapshot";

// Gizmo utilities (prototype - API may change)
export {
  createTranslateGizmo,
  computeGizmoScale,
  identifyGizmoPart,
  GIZMO_COLORS,
} from "./gizmo";
export type {
  TranslateGizmoConfig,
  TranslateGizmoResult,
  GizmoAxis,
  GizmoPlane,
  GizmoPart,
  GizmoTranslateCallback,
} from "./gizmo";

// Primitive spec types (for defining custom primitives)
export type { PrimitiveSpec, BaseComponentConfig } from "./types";

// Group utilities (for hierarchical transforms)
export {
  flattenGroups,
  hasGroups,
  isGroup,
  composeTransforms,
  identityTransform,
  quatMultiply,
  quatRotate,
  quatFromAxisAngle,
  quatNormalize,
  IDENTITY_QUAT,
  IDENTITY_POS,
} from "./groups";
export type {
  GroupConfig,
  Transform,
  Quat,
  FlattenedComponent,
} from "./groups";
