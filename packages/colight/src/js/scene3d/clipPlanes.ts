/**
 * @module scene3d/clipPlanes
 * @description Scene-level section / clipping planes for Scene3D.
 *
 * Planes are per-scene (not per-component). They pack into a small dedicated
 * uniform buffer at group(0) binding 3 (see `clipPlanesStruct` in shaders.ts)
 * and every fragment shader — render AND pick — calls `applyClipPlanes` to
 * discard fragments in the clipped half-space. Because they live in a uniform
 * (not per-instance data), a $state offset change is a cheap uniform write and
 * never re-uploads instance buffers.
 */

import { MAX_CLIP_PLANES } from "./shaders";

export { MAX_CLIP_PLANES };

/**
 * One clipping plane. A fragment at `worldPos` is KEPT where
 * `dot(worldPos, normal) <= offset`, discarded otherwise. Multiple planes
 * intersect (a fragment must pass every plane to survive).
 */
export interface ClipPlane {
  /** Plane normal (normalized Python-side). */
  normal: [number, number, number];
  /** Signed offset along the normal (world units, post-origin-shift). */
  offset: number;
}

/** Header floats (vec4) + MAX_CLIP_PLANES * vec4, all f32. */
export const CLIP_PLANES_FLOATS = 4 + MAX_CLIP_PLANES * 4;
/** Byte size of the clip-planes uniform buffer. */
export const CLIP_PLANES_BUFFER_SIZE = CLIP_PLANES_FLOATS * 4;

/**
 * Pack clip planes into the uniform-buffer float layout the shader expects:
 * `[count, 0, 0, 0, n0x, n0y, n0z, o0, n1x, ...]`. Unused plane slots are zero
 * (never read, since the shader loops only up to `count`).
 *
 * Throws if more than {@link MAX_CLIP_PLANES} planes are supplied — a loud
 * error rather than silently dropping planes (the design cap is a fixed-size
 * uniform array).
 */
export function packClipPlanes(planes: ClipPlane[] | undefined): Float32Array {
  const out = new Float32Array(CLIP_PLANES_FLOATS);
  if (!planes || planes.length === 0) {
    return out; // count = 0
  }
  if (planes.length > MAX_CLIP_PLANES) {
    throw new Error(
      `Scene supports at most ${MAX_CLIP_PLANES} clip_planes, got ${planes.length}. ` +
        `Reduce the number of section planes.`,
    );
  }
  out[0] = planes.length;
  for (let i = 0; i < planes.length; i++) {
    const p = planes[i];
    const base = 4 + i * 4;
    out[base] = p.normal[0];
    out[base + 1] = p.normal[1];
    out[base + 2] = p.normal[2];
    out[base + 3] = p.offset;
  }
  return out;
}

/**
 * True when the given planes clip away the ENTIRE axis-aligned bounds
 * `[min, max]` — i.e. every corner of the box is in the discarded half-space
 * of at least one plane, so nothing renders. Used to surface a "section
 * excludes entire scene" warning.
 *
 * Bounds are in the same post-origin-shift space as plane offsets.
 */
export function planesExcludeBounds(
  planes: ClipPlane[] | undefined,
  min: [number, number, number],
  max: [number, number, number],
): boolean {
  if (!planes || planes.length === 0) return false;
  const corners: [number, number, number][] = [];
  for (let xi = 0; xi < 2; xi++) {
    for (let yi = 0; yi < 2; yi++) {
      for (let zi = 0; zi < 2; zi++) {
        corners.push([
          xi ? max[0] : min[0],
          yi ? max[1] : min[1],
          zi ? max[2] : min[2],
        ]);
      }
    }
  }
  // The box is fully excluded if, for some plane, every corner is clipped
  // (dot(corner, normal) - offset > 0). That plane alone removes all geometry.
  for (const p of planes) {
    let allClipped = true;
    for (const c of corners) {
      const d =
        c[0] * p.normal[0] + c[1] * p.normal[1] + c[2] * p.normal[2] - p.offset;
      if (d <= 0) {
        allClipped = false;
        break;
      }
    }
    if (allClipped) return true;
  }
  return false;
}
