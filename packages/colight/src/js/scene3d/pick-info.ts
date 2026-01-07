/**
 * @module pick-info
 * @description Pure functions for building pick information from GPU picking results.
 * Extracted from impl3d.tsx for testability.
 */

import { normalize as normalizeQuat, rotateVector } from "./quaternion";
import { ComponentConfig } from "./components";
import { CameraParams, CameraState, createCameraParams } from "./camera3d";
import { screenRay as computeScreenRay } from "./impl3d";
import { getLineBeamsSegmentPointIndex } from "./components";
import { PickInfo } from "./types";

export type PickEventType = "hover" | "click";

export const FACE_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"] as const;
export type FaceName = (typeof FACE_NAMES)[number];

// ========== Vector utilities ==========

export function readVec3(
  arrayLike: ArrayLike<number>,
  index: number,
): [number, number, number] {
  const base = index * 3;
  return [arrayLike[base + 0], arrayLike[base + 1], arrayLike[base + 2]];
}

export function readQuat(
  arrayLike: ArrayLike<number>,
  index: number,
): [number, number, number, number] {
  const base = index * 4;
  return [
    arrayLike[base + 0],
    arrayLike[base + 1],
    arrayLike[base + 2],
    arrayLike[base + 3],
  ];
}

export function subVec3(
  a: [number, number, number],
  b: [number, number, number],
): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function dotVec3(
  a: [number, number, number],
  b: [number, number, number],
): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// ========== Quaternion utilities ==========

export function getQuaternion(
  component: ComponentConfig,
  index: number,
  fallback: [number, number, number, number],
): [number, number, number, number] {
  if ("quaternions" in component && component.quaternions) {
    return readQuat(component.quaternions, index);
  }
  if ("quaternion" in component && component.quaternion) {
    return component.quaternion as [number, number, number, number];
  }
  return fallback;
}

// ========== Normal decoding ==========

/**
 * Decodes a normal from GPU readback bytes [0-255] to normalized vector [-1, 1].
 * The GPU encodes normals as: encoded = normal * 0.5 + 0.5
 * We decode as: normal = (byte / 255) * 2 - 1
 */
export function decodeNormalFromBytes(
  bytes: Uint8Array,
): [number, number, number] {
  return [
    (bytes[0] / 255.0) * 2.0 - 1.0,
    (bytes[1] / 255.0) * 2.0 - 1.0,
    (bytes[2] / 255.0) * 2.0 - 1.0,
  ];
}

/**
 * Encodes a normal vector [-1, 1] to bytes [0-255] for GPU output.
 * This is the inverse of decodeNormalFromBytes.
 */
export function encodeNormalToBytes(
  normal: [number, number, number],
): [number, number, number] {
  return [
    Math.round((normal[0] * 0.5 + 0.5) * 255),
    Math.round((normal[1] * 0.5 + 0.5) * 255),
    Math.round((normal[2] * 0.5 + 0.5) * 255),
  ];
}

// ========== Face detection ==========

/**
 * Detects which face of a cuboid was hit based on the local normal.
 * Returns the face index [0-5] corresponding to FACE_NAMES.
 *
 * Face mapping:
 * - 0: +x (right)
 * - 1: -x (left)
 * - 2: +y (top)
 * - 3: -y (bottom)
 * - 4: +z (front)
 * - 5: -z (back)
 */
export function detectCuboidFace(
  localNormal: [number, number, number],
): { index: number; name: FaceName } {
  let hitAxis = 0;
  let hitSign = 1;

  // Find the axis with the largest absolute component
  for (let i = 0; i < 3; i++) {
    if (Math.abs(localNormal[i]) > 0.5) {
      hitAxis = i;
      hitSign = localNormal[i] > 0 ? 1 : -1;
    }
  }

  const faceIndex = hitAxis * 2 + (hitSign > 0 ? 0 : 1);
  return {
    index: faceIndex,
    name: FACE_NAMES[faceIndex],
  };
}

/**
 * Transforms a world-space normal to local space of a cuboid.
 */
export function worldNormalToLocal(
  worldNormal: [number, number, number],
  quaternion: [number, number, number, number],
): [number, number, number] {
  const q = normalizeQuat(quaternion);
  const qInv: [number, number, number, number] = [-q[0], -q[1], -q[2], q[3]];
  return rotateVector(worldNormal, qInv);
}

/**
 * Transforms a world-space position to local space of a component.
 */
export function worldPositionToLocal(
  worldPosition: [number, number, number],
  center: [number, number, number],
  quaternion: [number, number, number, number],
): [number, number, number] {
  const q = normalizeQuat(quaternion);
  const qInv: [number, number, number, number] = [-q[0], -q[1], -q[2], q[3]];
  return rotateVector(subVec3(worldPosition, center), qInv);
}

// ========== Build pick info ==========

export interface BuildPickInfoParams {
  mode: PickEventType;
  componentIndex: number;
  elementIndex: number;
  screenX: number;
  screenY: number;
  rect: { width: number; height: number };
  camera: CameraState;
  component: ComponentConfig;
  position?: [number, number, number];
  normal?: [number, number, number];
}

/**
 * Builds complete pick information from GPU picking results.
 * This is a pure function that can be tested without a browser.
 */
export function buildPickInfo(params: BuildPickInfoParams): PickInfo | null {
  const {
    mode,
    componentIndex,
    elementIndex,
    screenX,
    screenY,
    rect,
    camera,
    component,
    position,
    normal,
  } = params;

  const ray = computeScreenRay(screenX, screenY, rect, camera);
  if (!ray) return null;

  const info: PickInfo = {
    event: mode,
    component: { index: componentIndex, type: component.type },
    instanceIndex: elementIndex,
    screen: {
      x: screenX,
      y: screenY,
      dpr: typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1,
    },
    ray,
    camera: createCameraParams(camera),
  };

  if (position) {
    info.hit = { position, normal, t: 0 };

    // Calculate local position if applicable
    if (
      (component.type === "Cuboid" ||
        component.type === "Ellipsoid" ||
        component.type === "EllipsoidAxes") &&
      "centers" in component &&
      component.centers
    ) {
      const center = readVec3(component.centers, elementIndex);
      const quat = getQuaternion(component, elementIndex, [0, 0, 0, 1]);
      const localPos = worldPositionToLocal(position, center, quat);
      info.local = { position: localPos };

      if (component.type === "Cuboid" && normal) {
        const localNormal = worldNormalToLocal(normal, quat);
        info.face = detectCuboidFace(localNormal);
      }
    }

    if (component.type === "LineBeams" && "points" in component) {
      const segmentPointIndex = getLineBeamsSegmentPointIndex(
        component,
        elementIndex,
      );
      if (segmentPointIndex !== undefined) {
        const start: [number, number, number] = [
          component.points[segmentPointIndex * 4 + 0],
          component.points[segmentPointIndex * 4 + 1],
          component.points[segmentPointIndex * 4 + 2],
        ];
        const end: [number, number, number] = [
          component.points[(segmentPointIndex + 1) * 4 + 0],
          component.points[(segmentPointIndex + 1) * 4 + 1],
          component.points[(segmentPointIndex + 1) * 4 + 2],
        ];
        const lineIndex = Math.floor(
          component.points[segmentPointIndex * 4 + 3],
        );

        // Calculate t along the segment
        const v = subVec3(end, start);
        const w = subVec3(position, start);
        const u = dotVec3(w, v) / dotVec3(v, v);

        info.segment = {
          index: elementIndex,
          t: u,
          lineIndex,
        };
      }
    }

    return info;
  }

  // Fallback for cases where GPU position is not available
  if (
    component.type === "PointCloud" &&
    "centers" in component &&
    component.centers
  ) {
    const center = readVec3(component.centers, elementIndex);
    info.hit = { position: center };
    return info;
  }

  return info;
}
