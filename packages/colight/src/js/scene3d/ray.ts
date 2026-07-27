/**
 * @module ray
 * @description Ray and Plane types with intersection utilities.
 */

import { Vec3, sub, dot, scale, add } from "./vec3";
import { PickHit } from "./types";

export interface Ray {
  origin: Vec3;
  direction: Vec3; // assumed normalized
}

export interface Plane {
  point: Vec3;
  normal: Vec3; // assumed normalized
}

/**
 * Find where ray intersects plane.
 * Returns null if parallel or behind ray origin.
 */
export function intersectPlane(ray: Ray, plane: Plane): Vec3 | null {
  const denom = dot(ray.direction, plane.normal);
  if (Math.abs(denom) < 1e-6) return null; // parallel

  const t = dot(sub(plane.point, ray.origin), plane.normal) / denom;
  if (t < 0) return null; // behind ray

  return add(ray.origin, scale(ray.direction, t));
}

/**
 * Find where ray intersects plane, returning the t parameter.
 * Returns null if parallel or behind ray origin.
 */
export function intersectPlaneT(ray: Ray, plane: Plane): number | null {
  const denom = dot(ray.direction, plane.normal);
  if (Math.abs(denom) < 1e-6) return null;

  const t = dot(sub(plane.point, ray.origin), plane.normal) / denom;
  if (t < 0) return null;

  return t;
}

/**
 * Find point on ray at parameter t.
 */
export function pointOnRay(ray: Ray, t: number): Vec3 {
  return add(ray.origin, scale(ray.direction, t));
}

/**
 * Create plane from PickInfo hit data.
 * Returns null if hit is missing position or normal.
 */
export function planeFromHit(hit: PickHit | undefined): Plane | null {
  if (!hit?.position || !hit?.normal) return null;
  return {
    point: [hit.position[0], hit.position[1], hit.position[2]],
    normal: [hit.normal[0], hit.normal[1], hit.normal[2]],
  };
}

/**
 * Find nearest point on axis to ray (for single-axis constraints).
 * Useful for gizmo-style axis-locked dragging.
 */
export function nearestPointOnAxis(
  ray: Ray,
  axisOrigin: Vec3,
  axisDirection: Vec3,
): Vec3 {
  // Standard closest-point-between-two-lines
  const w = sub(ray.origin, axisOrigin);
  const a = dot(axisDirection, axisDirection);
  const b = dot(axisDirection, ray.direction);
  const c = dot(ray.direction, ray.direction);
  const d = dot(axisDirection, w);
  const e = dot(ray.direction, w);
  const denom = a * c - b * b;

  if (Math.abs(denom) < 1e-10) {
    // Lines are parallel, return axis origin
    return axisOrigin;
  }

  const t = (b * e - c * d) / denom;
  return add(axisOrigin, scale(axisDirection, t));
}

/**
 * Distance from point to plane (signed).
 * Positive if point is on the side the normal points to.
 */
export function distanceToPlane(point: Vec3, plane: Plane): number {
  return dot(sub(point, plane.point), plane.normal);
}

/**
 * Project point onto plane.
 */
export function projectOntoPlane(point: Vec3, plane: Plane): Vec3 {
  const d = distanceToPlane(point, plane);
  return sub(point, scale(plane.normal, d));
}
