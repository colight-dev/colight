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
import { DragConstraint, PickInfo, PickHit } from "./types";
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
