/**
 * @module project
 * @description Screen-space projection utilities.
 */

import * as glMatrix from "gl-matrix";
import { Vec3 } from "./vec3";
import { Ray } from "./ray";
import { CameraState, getViewMatrix, getProjectionMatrix } from "./camera3d";

export interface ScreenPoint {
  x: number;
  y: number;
}

export interface Rect {
  width: number;
  height: number;
}

/**
 * Convert screen pixel coordinates to a 3D ray.
 * The ray originates at the near plane and points into the scene.
 */
export function screenRay(
  screenX: number,
  screenY: number,
  rect: Rect,
  camera: CameraState,
): Ray | null {
  const ndcX = (screenX / rect.width) * 2 - 1;
  const ndcY = 1 - (screenY / rect.height) * 2;

  const view = getViewMatrix(camera);
  const proj = getProjectionMatrix(camera, rect.width / rect.height);
  const viewProj = glMatrix.mat4.multiply(glMatrix.mat4.create(), proj, view);
  const inv = glMatrix.mat4.invert(glMatrix.mat4.create(), viewProj);
  if (!inv) return null;

  const near = glMatrix.vec4.fromValues(ndcX, ndcY, 0, 1);
  const far = glMatrix.vec4.fromValues(ndcX, ndcY, 1, 1);
  glMatrix.vec4.transformMat4(near, near, inv);
  glMatrix.vec4.transformMat4(far, far, inv);

  const origin: Vec3 = [near[0] / near[3], near[1] / near[3], near[2] / near[3]];
  const farPoint: Vec3 = [far[0] / far[3], far[1] / far[3], far[2] / far[3]];

  const dx = farPoint[0] - origin[0];
  const dy = farPoint[1] - origin[1];
  const dz = farPoint[2] - origin[2];
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const direction: Vec3 = [dx / len, dy / len, dz / len];

  return { origin, direction };
}

/**
 * Project a 3D point to screen coordinates.
 * Returns null if the point is behind the camera.
 */
export function projectToScreen(
  point: Vec3,
  rect: Rect,
  camera: CameraState,
): ScreenPoint | null {
  const view = getViewMatrix(camera);
  const proj = getProjectionMatrix(camera, rect.width / rect.height);
  const viewProj = glMatrix.mat4.multiply(glMatrix.mat4.create(), proj, view);

  const clip = glMatrix.vec4.fromValues(point[0], point[1], point[2], 1);
  glMatrix.vec4.transformMat4(clip, clip, viewProj);

  if (clip[3] <= 0) return null; // behind camera

  const ndcX = clip[0] / clip[3];
  const ndcY = clip[1] / clip[3];

  return {
    x: ((ndcX + 1) / 2) * rect.width,
    y: ((1 - ndcY) / 2) * rect.height,
  };
}

/**
 * Check if a 3D point is visible (in front of camera and within frustum).
 */
export function isPointVisible(
  point: Vec3,
  rect: Rect,
  camera: CameraState,
): boolean {
  const view = getViewMatrix(camera);
  const proj = getProjectionMatrix(camera, rect.width / rect.height);
  const viewProj = glMatrix.mat4.multiply(glMatrix.mat4.create(), proj, view);

  const clip = glMatrix.vec4.fromValues(point[0], point[1], point[2], 1);
  glMatrix.vec4.transformMat4(clip, clip, viewProj);

  if (clip[3] <= 0) return false;

  const ndcX = clip[0] / clip[3];
  const ndcY = clip[1] / clip[3];
  const ndcZ = clip[2] / clip[3];

  return (
    ndcX >= -1 && ndcX <= 1 &&
    ndcY >= -1 && ndcY <= 1 &&
    ndcZ >= 0 && ndcZ <= 1
  );
}
