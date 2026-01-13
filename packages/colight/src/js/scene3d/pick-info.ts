/**
 * @module pick-info
 * @description Pure functions for building pick information from GPU picking results.
 * Extracted from impl3d.tsx for testability.
 */

import { normalize as normalizeQuat, rotateVector } from "./quaternion";
import { ComponentConfig } from "./components";
import { CameraState, createCameraParams } from "./camera3d";
import { screenRay } from "./project";
import { getLineBeamsSegmentPointIndex } from "./components";
import { PickInfo } from "./types";
import { Vec3, sub, dot, readVec3 } from "./vec3";
import {
  Transform,
  applyInverseTransformToPoint,
  applyTransformToPoint,
  quatInvert,
} from "./groups";

export type PickEventType = "hover" | "click";

export const FACE_NAMES = ["+x", "-x", "+y", "-y", "+z", "-z"] as const;
export type FaceName = (typeof FACE_NAMES)[number];

// ========== Quaternion utilities ==========

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
export function decodeNormalFromBytes(bytes: Uint8Array): Vec3 {
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
export function encodeNormalToBytes(normal: Vec3): Vec3 {
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
export function detectCuboidFace(localNormal: Vec3): {
  index: number;
  name: FaceName;
} {
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
  worldNormal: Vec3,
  quaternion: [number, number, number, number],
): Vec3 {
  const q = normalizeQuat(quaternion);
  const qInv: [number, number, number, number] = [-q[0], -q[1], -q[2], q[3]];
  return rotateVector(worldNormal, qInv);
}

function getGroupTransform(component: ComponentConfig): Transform | undefined {
  return (component as any)._groupTransform as Transform | undefined;
}

function worldPositionToGroupLocal(
  worldPosition: Vec3,
  groupTransform?: Transform,
): Vec3 {
  if (!groupTransform) return worldPosition;
  return applyInverseTransformToPoint(groupTransform, worldPosition);
}

function worldNormalToGroupLocal(
  worldNormal: Vec3,
  groupTransform?: Transform,
): Vec3 {
  if (!groupTransform) return worldNormal;
  return rotateVector(worldNormal, quatInvert(groupTransform.quaternion));
}

/**
 * Transforms a world-space position to local space of a component.
 */
export function worldPositionToLocal(
  worldPosition: Vec3,
  center: Vec3,
  quaternion: [number, number, number, number],
): Vec3 {
  const q = normalizeQuat(quaternion);
  const qInv: [number, number, number, number] = [-q[0], -q[1], -q[2], q[3]];
  return rotateVector(sub(worldPosition, center), qInv);
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
  position?: Vec3;
  normal?: Vec3;
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

  const ray = screenRay(screenX, screenY, rect, camera);
  if (!ray) return null;

  const cameraParams = createCameraParams(camera);
  const info: PickInfo = {
    event: mode,
    component: { index: componentIndex, type: component.type },
    groupPath: (component as any)._groupPath,
    instanceIndex: elementIndex,
    screen: {
      x: screenX,
      y: screenY,
      dpr: typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1,
      rectWidth: rect.width,
      rectHeight: rect.height,
    },
    ray,
    camera: {
      position: cameraParams.position as [number, number, number],
      target: cameraParams.target as [number, number, number],
      up: cameraParams.up as [number, number, number],
      fov: cameraParams.fov,
      near: cameraParams.near,
      far: cameraParams.far,
    },
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
      const groupTransform = getGroupTransform(component);
      const center = readVec3(component.centers, elementIndex);
      const quat = getQuaternion(component, elementIndex, [0, 0, 0, 1]);
      const groupLocalPos = worldPositionToGroupLocal(position, groupTransform);
      const localPos = worldPositionToLocal(groupLocalPos, center, quat);
      info.local = { position: localPos };

      if (component.type === "Cuboid" && normal) {
        const groupLocalNormal = worldNormalToGroupLocal(normal, groupTransform);
        const localNormal = worldNormalToLocal(groupLocalNormal, quat);
        info.face = detectCuboidFace(localNormal);
      }
    }

    if (component.type === "LineBeams" && "points" in component) {
      const segmentPointIndex = getLineBeamsSegmentPointIndex(
        component,
        elementIndex,
      );
      if (segmentPointIndex !== undefined) {
        const groupTransform = getGroupTransform(component);
        const start: Vec3 = [
          component.points[segmentPointIndex * 4 + 0],
          component.points[segmentPointIndex * 4 + 1],
          component.points[segmentPointIndex * 4 + 2],
        ];
        const end: Vec3 = [
          component.points[(segmentPointIndex + 1) * 4 + 0],
          component.points[(segmentPointIndex + 1) * 4 + 1],
          component.points[(segmentPointIndex + 1) * 4 + 2],
        ];
        const worldStart = groupTransform
          ? applyTransformToPoint(groupTransform, start)
          : start;
        const worldEnd = groupTransform
          ? applyTransformToPoint(groupTransform, end)
          : end;
        const lineIndex = Math.floor(
          component.points[segmentPointIndex * 4 + 3],
        );

        // Calculate t along the segment
        const v = sub(worldEnd, worldStart);
        const w = sub(position, worldStart);
        const u = dot(w, v) / dot(v, v);

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
    const groupTransform = getGroupTransform(component);
    const worldCenter = groupTransform
      ? applyTransformToPoint(groupTransform, center)
      : center;
    info.hit = { position: worldCenter };
    return info;
  }

  return info;
}
