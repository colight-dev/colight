/**
 * @module gizmo
 * @description Translate gizmo for Scene3D - allows interactive translation of objects.
 *
 * This is a PROTOTYPE implementation using composite primitives (LineBeams + Ellipsoids).
 * It validates that drag + layers work together and identifies gaps for the extensibility API.
 *
 * @internal Not yet stable - API may change
 */

import { Vec3, add, sub, scale, length } from "./vec3";
import { CameraState } from "./camera3d";
import { DragConstraint, DragInfo } from "./types";
import { LineBeamsComponentConfig, EllipsoidComponentConfig } from "./components";

// =============================================================================
// Types
// =============================================================================

/**
 * Gizmo axis identifier
 */
export type GizmoAxis = "x" | "y" | "z";

/**
 * Gizmo plane identifier
 */
export type GizmoPlane = "xy" | "xz" | "yz";

/**
 * Gizmo part identifier (axis, plane, or center)
 */
export type GizmoPart = GizmoAxis | GizmoPlane | "center";

/**
 * Callback fired when gizmo is dragged
 */
export type GizmoTranslateCallback = (
  part: GizmoPart,
  delta: Vec3,
  info: DragInfo,
) => void;

/**
 * Configuration for creating a translate gizmo
 */
export interface TranslateGizmoConfig {
  /** World position of the gizmo center */
  position: Vec3;

  /** Callback fired during drag with delta translation */
  onTranslate?: GizmoTranslateCallback;

  /** Which axes to show (default: all) */
  axes?: GizmoAxis[];

  /** Which planes to show (default: all) */
  planes?: GizmoPlane[];

  /** Show center sphere for free drag (default: true) */
  showCenter?: boolean;

  /** Target size in screen pixels (default: 100) */
  screenSize?: number;

  /** Currently hovered part (for highlighting) */
  hoveredPart?: GizmoPart | null;

  /** Currently dragging part (for highlighting) */
  draggingPart?: GizmoPart | null;
}

/**
 * Result of createTranslateGizmo - components and state management
 */
export interface TranslateGizmoResult {
  /** Array of components to render (LineBeams + Ellipsoids) */
  components: (LineBeamsComponentConfig | EllipsoidComponentConfig)[];

  /** Get the constraint for a given part */
  getConstraint: (part: GizmoPart) => DragConstraint;
}

// =============================================================================
// Constants
// =============================================================================

/** Gizmo colors */
export const GIZMO_COLORS = {
  x: [1.0, 0.2, 0.2] as Vec3, // Red
  y: [0.2, 1.0, 0.2] as Vec3, // Green
  z: [0.2, 0.4, 1.0] as Vec3, // Blue
  xy: [1.0, 1.0, 0.2] as Vec3, // Yellow
  xz: [0.2, 1.0, 1.0] as Vec3, // Cyan
  yz: [1.0, 0.2, 1.0] as Vec3, // Magenta
  center: [0.9, 0.9, 0.9] as Vec3, // White
};

/** Axis direction vectors */
const AXIS_DIRECTIONS: Record<GizmoAxis, Vec3> = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
};

/** Plane normal vectors */
const PLANE_NORMALS: Record<GizmoPlane, Vec3> = {
  xy: [0, 0, 1],
  xz: [0, 1, 0],
  yz: [1, 0, 0],
};

/** Brightness multiplier for hover/active states */
const HOVER_BRIGHTNESS = 1.5;

/** Alpha for dimmed parts during drag */
const DIMMED_ALPHA = 0.3;

// =============================================================================
// Screen-Space Sizing
// =============================================================================

/**
 * Compute scale factor to maintain constant screen pixel size.
 *
 * @param position - World position of the gizmo
 * @param camera - Current camera state
 * @param targetPixelSize - Desired size in screen pixels
 * @param viewportHeight - Viewport height in pixels
 * @returns Scale factor to apply to gizmo geometry
 */
export function computeGizmoScale(
  position: Vec3,
  camera: CameraState,
  targetPixelSize: number,
  viewportHeight: number,
): number {
  const camPos: Vec3 = [camera.position[0], camera.position[1], camera.position[2]];
  const dist = length(sub(position, camPos));

  // For perspective projection:
  // pixelSize = (worldSize / distance) * (viewportHeight / (2 * tan(fov/2)))
  // Solve for worldSize:
  const fovRadians = (camera.fov * Math.PI) / 180;
  const worldSize =
    (targetPixelSize * dist * 2 * Math.tan(fovRadians / 2)) / viewportHeight;

  return worldSize;
}

// =============================================================================
// Color Utilities
// =============================================================================

function brighten(color: Vec3, factor: number): Vec3 {
  return [
    Math.min(1, color[0] * factor),
    Math.min(1, color[1] * factor),
    Math.min(1, color[2] * factor),
  ];
}

function getPartColor(
  part: GizmoPart,
  hoveredPart: GizmoPart | null | undefined,
  draggingPart: GizmoPart | null | undefined,
): { color: Vec3; alpha: number } {
  const baseColor = GIZMO_COLORS[part];
  const isHovered = hoveredPart === part;
  const isDragging = draggingPart === part;
  const isOtherDragging = draggingPart && draggingPart !== part;

  if (isHovered || isDragging) {
    return { color: brighten(baseColor, HOVER_BRIGHTNESS), alpha: 1 };
  }
  if (isOtherDragging) {
    return { color: baseColor, alpha: DIMMED_ALPHA };
  }
  return { color: baseColor, alpha: 1 };
}

// =============================================================================
// Gizmo Creation
// =============================================================================

/**
 * Create a translate gizmo as a composite of LineBeams and Ellipsoids.
 *
 * The gizmo consists of:
 * - 3 axis arrows (LineBeams with cone tips)
 * - 3 plane handles (small cubes at axis intersections)
 * - 1 center sphere (for free drag)
 *
 * All components are rendered as overlays (always visible, no depth test).
 *
 * @param config - Gizmo configuration
 * @param camera - Current camera state (for screen-space sizing)
 * @param viewportHeight - Viewport height in pixels
 * @returns Components to render and helper functions
 */
export function createTranslateGizmo(
  config: TranslateGizmoConfig,
  camera: CameraState,
  viewportHeight: number,
): TranslateGizmoResult {
  const {
    position,
    onTranslate,
    axes = ["x", "y", "z"],
    planes = ["xy", "xz", "yz"],
    showCenter = true,
    screenSize = 100,
    hoveredPart,
    draggingPart,
  } = config;

  // Compute scale for screen-space sizing
  const gizmoScale = computeGizmoScale(position, camera, screenSize, viewportHeight);

  // Geometry dimensions (at scale=1, then multiplied by gizmoScale)
  const axisLength = 1.0 * gizmoScale;
  const shaftWidth = 0.02 * gizmoScale;
  const coneLength = 0.15 * gizmoScale;
  const coneRadius = 0.06 * gizmoScale;
  const planeSize = 0.25 * gizmoScale;
  const planeOffset = 0.35 * gizmoScale;
  const centerRadius = 0.08 * gizmoScale;

  const components: (LineBeamsComponentConfig | EllipsoidComponentConfig)[] = [];

  // Helper to create drag callbacks
  const createDragCallbacks = (part: GizmoPart) => {
    if (!onTranslate) return {};
    return {
      onDragStart: (info: DragInfo) => {
        onTranslate(part, [0, 0, 0], info);
      },
      onDrag: (info: DragInfo) => {
        onTranslate(part, info.delta.position, info);
      },
      onDragEnd: (info: DragInfo) => {
        onTranslate(part, info.delta.position, info);
      },
    };
  };

  // Create axis arrows
  for (const axis of axes) {
    const dir = AXIS_DIRECTIONS[axis];
    const { color, alpha } = getPartColor(axis, hoveredPart, draggingPart);

    // Axis shaft (LineBeams)
    const shaftEnd = add(position, scale(dir, axisLength - coneLength));
    const shaftPoints = new Float32Array([
      // Start point
      position[0], position[1], position[2], 0,
      // End point (before cone)
      shaftEnd[0], shaftEnd[1], shaftEnd[2], 0,
    ]);

    components.push({
      type: "LineBeams",
      points: shaftPoints,
      color: color as [number, number, number],
      alpha,
      size: shaftWidth,
      layer: "overlay",
      dragConstraint: { type: "axis", direction: dir as [number, number, number], point: position as [number, number, number] },
      ...createDragCallbacks(axis),
    } as LineBeamsComponentConfig);

    // Cone tip (Ellipsoid stretched along axis)
    const coneCenter = add(position, scale(dir, axisLength - coneLength / 2));

    // Determine cone half-sizes based on axis
    let coneHalfSize: Vec3;
    if (axis === "x") {
      coneHalfSize = [coneLength / 2, coneRadius, coneRadius];
    } else if (axis === "y") {
      coneHalfSize = [coneRadius, coneLength / 2, coneRadius];
    } else {
      coneHalfSize = [coneRadius, coneRadius, coneLength / 2];
    }

    components.push({
      type: "Ellipsoid",
      centers: new Float32Array(coneCenter),
      half_size: coneHalfSize as [number, number, number],
      color: color as [number, number, number],
      alpha,
      layer: "overlay",
      dragConstraint: { type: "axis", direction: dir as [number, number, number], point: position as [number, number, number] },
      ...createDragCallbacks(axis),
    } as EllipsoidComponentConfig);
  }

  // Create plane handles
  for (const plane of planes) {
    const { color, alpha } = getPartColor(plane, hoveredPart, draggingPart);
    const normal = PLANE_NORMALS[plane];

    // Position the plane handle at the intersection of the two axes
    // e.g., XY plane handle is offset along X and Y
    let handleCenter: Vec3;
    let handleHalfSize: Vec3;

    if (plane === "xy") {
      handleCenter = add(position, [planeOffset, planeOffset, 0]);
      handleHalfSize = [planeSize / 2, planeSize / 2, planeSize / 10];
    } else if (plane === "xz") {
      handleCenter = add(position, [planeOffset, 0, planeOffset]);
      handleHalfSize = [planeSize / 2, planeSize / 10, planeSize / 2];
    } else {
      // yz
      handleCenter = add(position, [0, planeOffset, planeOffset]);
      handleHalfSize = [planeSize / 10, planeSize / 2, planeSize / 2];
    }

    components.push({
      type: "Ellipsoid",
      centers: new Float32Array(handleCenter),
      half_size: handleHalfSize as [number, number, number],
      color: color as [number, number, number],
      alpha,
      layer: "overlay",
      dragConstraint: { type: "plane", normal: normal as [number, number, number], point: position as [number, number, number] },
      ...createDragCallbacks(plane),
    } as EllipsoidComponentConfig);
  }

  // Create center sphere
  if (showCenter) {
    const { color, alpha } = getPartColor("center", hoveredPart, draggingPart);

    components.push({
      type: "Ellipsoid",
      centers: new Float32Array(position),
      half_size: [centerRadius, centerRadius, centerRadius],
      color: color as [number, number, number],
      alpha,
      layer: "overlay",
      dragConstraint: { type: "free" },
      ...createDragCallbacks("center"),
    } as EllipsoidComponentConfig);
  }

  // Constraint getter
  const getConstraint = (part: GizmoPart): DragConstraint => {
    if (part === "x" || part === "y" || part === "z") {
      return {
        type: "axis",
        direction: AXIS_DIRECTIONS[part] as [number, number, number],
        point: position as [number, number, number],
      };
    }
    if (part === "xy" || part === "xz" || part === "yz") {
      return {
        type: "plane",
        normal: PLANE_NORMALS[part] as [number, number, number],
        point: position as [number, number, number],
      };
    }
    return { type: "free" };
  };

  return {
    components,
    getConstraint,
  };
}

/**
 * Identify which gizmo part was picked based on component index.
 *
 * Since the gizmo is composed of multiple components, this helper maps
 * component indices back to gizmo parts.
 *
 * @param componentIndex - Index of the picked component within the gizmo
 * @param config - Gizmo configuration (to know which parts are enabled)
 * @returns The gizmo part that was picked, or null if not a gizmo component
 */
export function identifyGizmoPart(
  componentIndex: number,
  config: TranslateGizmoConfig,
): GizmoPart | null {
  const axes = config.axes ?? ["x", "y", "z"];
  const planes = config.planes ?? ["xy", "xz", "yz"];
  const showCenter = config.showCenter ?? true;

  let idx = 0;

  // Each axis has 2 components (shaft + cone)
  for (const axis of axes) {
    if (componentIndex === idx || componentIndex === idx + 1) {
      return axis;
    }
    idx += 2;
  }

  // Each plane has 1 component
  for (const plane of planes) {
    if (componentIndex === idx) {
      return plane;
    }
    idx += 1;
  }

  // Center sphere
  if (showCenter && componentIndex === idx) {
    return "center";
  }

  return null;
}
