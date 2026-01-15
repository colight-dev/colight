/**
 * @module drag
 * @description Drag constraint resolution for Scene3D components.
 */

import { Vec3, sub, add, normalize, scale, dot } from "./vec3";
import {
  Ray,
  Plane,
  intersectPlane,
  nearestPointOnAxis,
  planeFromHit,
} from "./ray";
import { DragConstraint, PickInfo, PickHit, DragInfo } from "./types";
import { CameraState } from "./camera3d";

/**
 * Resolved constraint ready for position computation.
 * Internal representation after resolving user-facing DragConstraint.
 */
export type ResolvedConstraint =
  | { type: "plane"; plane: Plane }
  | { type: "axis"; origin: Vec3; direction: Vec3 };

/**
 * Resolve a user-facing DragConstraint into an internal ResolvedConstraint.
 * Uses the pick info to determine default values (e.g., instance center as constraint origin).
 *
 * @param constraint - User-provided constraint configuration
 * @param pickInfo - Pick info from drag start (contains hit position, instance info)
 * @param instanceCenter - Center of the dragged instance
 * @param camera - Current camera state (for screen/free constraints)
 * @returns Resolved constraint ready for position computation, or null if invalid
 */
export function resolveConstraint(
  constraint: DragConstraint | undefined,
  pickInfo: PickInfo,
  instanceCenter: Vec3,
  camera: CameraState,
): ResolvedConstraint | null {
  // Default to surface constraint if not specified
  if (!constraint) {
    constraint = { type: "surface" };
  }

  switch (constraint.type) {
    case "surface": {
      // Use the hit surface as the drag plane
      const plane = planeFromHit(pickInfo.hit);
      if (!plane) {
        // Fallback to camera-facing plane if no hit normal
        return createCameraFacingPlane(instanceCenter, camera);
      }
      return { type: "plane", plane };
    }

    case "plane": {
      const normal = normalize(constraint.normal);
      const point: Vec3 = constraint.point ?? instanceCenter;
      return {
        type: "plane",
        plane: { point, normal },
      };
    }

    case "axis": {
      const direction = normalize(constraint.direction);
      const origin: Vec3 = constraint.point ?? instanceCenter;
      return {
        type: "axis",
        origin,
        direction,
      };
    }

    case "screen": {
      // Screen-space drag: use a plane perpendicular to view direction at the hit depth
      const hitPos = pickInfo.hit?.position ?? instanceCenter;
      return createCameraFacingPlane(hitPos, camera);
    }

    case "free": {
      // Free drag on camera-facing plane through start point
      return createCameraFacingPlane(instanceCenter, camera);
    }

    default:
      return null;
  }
}

/**
 * Create a plane facing the camera through the given point.
 */
function createCameraFacingPlane(
  point: Vec3,
  camera: CameraState,
): ResolvedConstraint {
  // Camera forward direction (from camera to target)
  const forward = normalize(sub(camera.target, camera.position));
  return {
    type: "plane",
    plane: { point, normal: forward },
  };
}

/**
 * Compute the constrained world position for a drag operation.
 *
 * @param ray - Current mouse ray in world space
 * @param constraint - Resolved constraint
 * @returns Constrained world position, or null if computation fails (e.g., parallel ray)
 */
export function computeConstrainedPosition(
  ray: Ray,
  constraint: ResolvedConstraint,
): Vec3 | null {
  switch (constraint.type) {
    case "plane":
      return intersectPlane(ray, constraint.plane);

    case "axis":
      return nearestPointOnAxis(ray, constraint.origin, constraint.direction);

    default:
      return null;
  }
}

/**
 * Check if a component has any drag callbacks configured.
 */
export function hasDragCallbacks(component: {
  onDragStart?: unknown;
  onDrag?: unknown;
  onDragEnd?: unknown;
}): boolean {
  return !!(component.onDragStart || component.onDrag || component.onDragEnd);
}

/**
 * Convenience constants for common axis constraints.
 */
export const DRAG_AXIS_X: DragConstraint = {
  type: "axis",
  direction: [1, 0, 0],
};
export const DRAG_AXIS_Y: DragConstraint = {
  type: "axis",
  direction: [0, 1, 0],
};
export const DRAG_AXIS_Z: DragConstraint = {
  type: "axis",
  direction: [0, 0, 1],
};

/**
 * Convenience constants for common plane constraints.
 */
export const DRAG_PLANE_XY: DragConstraint = {
  type: "plane",
  normal: [0, 0, 1],
};
export const DRAG_PLANE_XZ: DragConstraint = {
  type: "plane",
  normal: [0, 1, 0],
};
export const DRAG_PLANE_YZ: DragConstraint = {
  type: "plane",
  normal: [1, 0, 0],
};

/**
 * Create an axis constraint.
 */
export function dragAxis(
  direction: [number, number, number],
  point?: [number, number, number],
): DragConstraint {
  const constraint: DragConstraint = { type: "axis", direction };
  if (point) {
    (
      constraint as {
        type: "axis";
        direction: [number, number, number];
        point: [number, number, number];
      }
    ).point = point;
  }
  return constraint;
}

/**
 * Create a plane constraint.
 */
export function dragPlane(
  normal: [number, number, number],
  point?: [number, number, number],
): DragConstraint {
  const constraint: DragConstraint = { type: "plane", normal };
  if (point) {
    (
      constraint as {
        type: "plane";
        normal: [number, number, number];
        point: [number, number, number];
      }
    ).point = point;
  }
  return constraint;
}

/**
 * Build a DragInfo object from pick info and drag state.
 *
 * @param pickInfo - Base pick info (component, instance, ray, etc.)
 * @param startInstanceCenter - Local-space center of the dragged instance
 * @param startWorldPosition - World-space position where drag started
 * @param startScreen - Screen coordinates at drag start
 * @param currentScreen - Current screen coordinates
 * @param currentPosition - Current world-space position (from constraint intersection)
 * @param memoizedValues - Map for storing initial values for applyDelta
 */
export function buildDragInfo(
  pickInfo: PickInfo,
  startInstanceCenter: [number, number, number],
  startWorldPosition: [number, number, number],
  startScreen: { x: number; y: number },
  currentScreen: { x: number; y: number },
  currentPosition?: [number, number, number],
  memoizedValues?: Map<string, [number, number, number]>,
): DragInfo {
  const curPos = currentPosition ?? startWorldPosition;

  const deltaPos: [number, number, number] = [
    curPos[0] - startWorldPosition[0],
    curPos[1] - startWorldPosition[1],
    curPos[2] - startWorldPosition[2],
  ];

  return {
    ...pickInfo,
    start: {
      position: startWorldPosition,
      instanceCenter: startInstanceCenter,
      screen: startScreen,
    },
    current: {
      position: curPos,
      instanceCenter: [
        startInstanceCenter[0] + deltaPos[0],
        startInstanceCenter[1] + deltaPos[1],
        startInstanceCenter[2] + deltaPos[2],
      ],
      screen: currentScreen,
    },
    delta: {
      position: deltaPos,
      screen: {
        x: currentScreen.x - startScreen.x,
        y: currentScreen.y - startScreen.y,
      },
    },
    applyDelta: (
      obj: Record<string, any>,
      key: string,
    ): [number, number, number] => {
      // Memoize the start value on first call
      if (memoizedValues && !memoizedValues.has(key)) {
        const startValue = obj[key];
        if (Array.isArray(startValue) && startValue.length >= 3) {
          memoizedValues.set(key, [
            startValue[0],
            startValue[1],
            startValue[2],
          ]);
        }
      }

      // Get memoized start value or fall back to current value
      const start = memoizedValues?.get(key) ?? obj[key];
      const result: [number, number, number] = [
        start[0] + deltaPos[0],
        start[1] + deltaPos[1],
        start[2] + deltaPos[2],
      ];

      // Set and return
      obj[key] = result;
      return result;
    },
  };
}
