/**
 * @module primitives/lineSegments
 * @description LineSegments primitive using the declarative definition system.
 *
 * LineSegments renders independent line segments defined by start/end arrays.
 */

import { BaseComponentConfig } from "../types";
import { definePrimitive, attr } from "./define";

// =============================================================================
// Configuration Interface
// =============================================================================

export interface LineSegmentsComponentConfig extends BaseComponentConfig {
  type: "LineSegments";
  /** Segment start positions: [x, y, z, ...] */
  starts: Float32Array;
  /** Segment end positions: [x, y, z, ...] */
  ends: Float32Array;
  /** Per-segment sizes (thickness) */
  sizes?: Float32Array;
  /** Default size for all segments */
  size?: number;
}

// =============================================================================
// Helpers
// =============================================================================

function countSegments(elem: LineSegmentsComponentConfig): number {
  return Math.min(elem.starts.length, elem.ends.length) / 3;
}

function getSegmentCenters(elem: LineSegmentsComponentConfig): Float32Array {
  const segCount = countSegments(elem);
  const centers = new Float32Array(segCount * 3);

  for (let i = 0; i < segCount; i++) {
    const sx = elem.starts[i * 3 + 0];
    const sy = elem.starts[i * 3 + 1];
    const sz = elem.starts[i * 3 + 2];
    const ex = elem.ends[i * 3 + 0];
    const ey = elem.ends[i * 3 + 1];
    const ez = elem.ends[i * 3 + 2];

    centers[i * 3 + 0] = (sx + ex) * 0.5;
    centers[i * 3 + 1] = (sy + ey) * 0.5;
    centers[i * 3 + 2] = (sz + ez) * 0.5;
  }

  return centers;
}

// =============================================================================
// Primitive Definition
// =============================================================================

export const lineSegmentsSpec = definePrimitive<LineSegmentsComponentConfig>({
  name: "LineSegments",

  attributes: {
    start: attr.vec3("starts"),
    end: attr.vec3("ends"),
    size: attr.f32("sizes", 0.02),
    color: attr.vec3("colors", [0.5, 0.5, 0.5]),
    alpha: attr.f32("alphas", 1.0),
  },

  geometry: { type: "beam" },
  transform: "beam",
  shading: "unlit",
  cullMode: "none",

  getElementCount: countSegments,
  getCenters: getSegmentCenters,
});
