/**
 * @module primitives/lineBeams
 * @description LineBeams primitive using the declarative definition system.
 *
 * LineBeams renders connected line segments as 3D beams (rectangular tubes).
 * Points are specified as [x, y, z, lineIndex, ...] where segments are formed
 * between consecutive points with the same lineIndex.
 */

import { BaseComponentConfig } from "../types";
import {
  definePrimitive,
  attr,
  ProcessedSchema,
  packID,
  acopy,
} from "./define";

// =============================================================================
// Configuration Interface
// =============================================================================

export interface LineBeamsComponentConfig extends BaseComponentConfig {
  type: "LineBeams";
  /** Points array: [x, y, z, lineIndex, x, y, z, lineIndex, ...] */
  points: Float32Array;
  /** Per-line sizes (thickness) */
  sizes?: Float32Array;
  /** Default size for all lines */
  size?: number;
}

// =============================================================================
// Segment Preprocessing
// =============================================================================

/**
 * Cache for segment maps - avoids recomputing for the same config.
 * WeakMap allows GC when config is no longer referenced.
 */
const segmentMapCache = new WeakMap<
  LineBeamsComponentConfig,
  { segmentMap: number[] }
>();

/**
 * Computes the segment map for a LineBeams config.
 * A segment is formed between consecutive points with the same lineIndex.
 *
 * @returns Array of point indices that start valid segments
 */
function prepareLineSegments(elem: LineBeamsComponentConfig): number[] {
  const cached = segmentMapCache.get(elem);
  if (cached) return cached.segmentMap;

  const pointCount = elem.points.length / 4;
  const segmentIndices: number[] = [];

  for (let p = 0; p < pointCount - 1; p++) {
    const lineIndexCurrent = elem.points[p * 4 + 3];
    const lineIndexNext = elem.points[(p + 1) * 4 + 3];
    if (lineIndexCurrent === lineIndexNext) {
      segmentIndices.push(p);
    }
  }

  segmentMapCache.set(elem, { segmentMap: segmentIndices });
  return segmentIndices;
}

/**
 * Gets the number of segments (elements) in a LineBeams config.
 */
function countSegments(elem: LineBeamsComponentConfig): number {
  return prepareLineSegments(elem).length;
}

/**
 * Gets the point index for a given segment index.
 * Used for detailed pick info to map segment -> original point.
 */
export function getLineBeamsSegmentPointIndex(
  elem: LineBeamsComponentConfig,
  segmentIndex: number,
): number | undefined {
  const segmentMap = prepareLineSegments(elem);
  return segmentMap[segmentIndex];
}

/**
 * Computes the center (midpoint) of each segment for transparency sorting.
 */
function getSegmentCenters(elem: LineBeamsComponentConfig): Float32Array {
  const segMap = prepareLineSegments(elem);
  const segCount = segMap.length;
  const centers = new Float32Array(segCount * 3);

  for (let s = 0; s < segCount; s++) {
    const p = segMap[s];
    const x0 = elem.points[p * 4 + 0];
    const y0 = elem.points[p * 4 + 1];
    const z0 = elem.points[p * 4 + 2];
    const x1 = elem.points[(p + 1) * 4 + 0];
    const y1 = elem.points[(p + 1) * 4 + 1];
    const z1 = elem.points[(p + 1) * 4 + 2];

    centers[s * 3 + 0] = (x0 + x1) * 0.5;
    centers[s * 3 + 1] = (y0 + y1) * 0.5;
    centers[s * 3 + 2] = (z0 + z1) * 0.5;
  }

  return centers;
}

// =============================================================================
// Custom Fill Functions
// =============================================================================

/**
 * Custom render geometry fill for LineBeams.
 * Handles the segment-based data access pattern.
 */
function fillRenderGeometry(
  schema: ProcessedSchema,
  constants: Record<string, unknown>,
  elem: LineBeamsComponentConfig,
  segmentIndex: number,
  out: Float32Array,
  outIndex: number,
): void {
  const outOffset = outIndex * schema.floatsPerInstance;
  const segMap = prepareLineSegments(elem);
  const p = segMap[segmentIndex];

  // Start position (vec3)
  acopy(elem.points, p * 4, out, outOffset, 3);

  // End position (vec3)
  acopy(elem.points, (p + 1) * 4, out, outOffset + 3, 3);

  // Size (f32) - uses lineIndex since sizes are per-line
  const lineIndex = Math.floor(elem.points[p * 4 + 3]);
  out[outOffset + 6] =
    (constants.size as number) ?? elem.sizes?.[lineIndex] ?? 0.02;

  // Color (vec3) - uses segmentIndex for per-segment colors
  if (constants.color) {
    const c = constants.color as number[];
    out[outOffset + 7] = c[0];
    out[outOffset + 8] = c[1];
    out[outOffset + 9] = c[2];
  } else if (elem.colors) {
    acopy(elem.colors, segmentIndex * 3, out, outOffset + 7, 3);
  } else {
    // Default gray
    out[outOffset + 7] = 0.5;
    out[outOffset + 8] = 0.5;
    out[outOffset + 9] = 0.5;
  }

  // Alpha (f32) - uses segmentIndex for per-segment alphas
  out[outOffset + 10] =
    (constants.alpha as number) ?? elem.alphas?.[segmentIndex] ?? 1.0;
}

/**
 * Custom picking geometry fill for LineBeams.
 */
function fillPickingGeometry(
  schema: ProcessedSchema,
  constants: Record<string, unknown>,
  elem: LineBeamsComponentConfig,
  segmentIndex: number,
  out: Float32Array,
  outIndex: number,
  baseID: number,
): void {
  const outOffset = outIndex * schema.floatsPerPicking;
  const segMap = prepareLineSegments(elem);
  const p = segMap[segmentIndex];

  // Start position (vec3)
  acopy(elem.points, p * 4, out, outOffset, 3);

  // End position (vec3)
  acopy(elem.points, (p + 1) * 4, out, outOffset + 3, 3);

  // Size (f32)
  const lineIndex = Math.floor(elem.points[p * 4 + 3]);
  out[outOffset + 6] =
    (constants.size as number) ?? elem.sizes?.[lineIndex] ?? 0.02;

  // Pick ID (f32)
  out[outOffset + 7] = packID(baseID + segmentIndex);
}

// =============================================================================
// Primitive Definition
// =============================================================================

export const lineBeamsSpec = definePrimitive<LineBeamsComponentConfig>({
  name: "LineBeams",

  // Attribute schema defines buffer layout and shader inputs
  // Order: start, end, size, color, alpha
  attributes: {
    start: attr.vec3("points"), // Will be filled custom
    end: attr.vec3("points"), // Will be filled custom
    size: attr.f32("sizes", 0.02),
    color: attr.vec3("colors", [0.5, 0.5, 0.5]),
    alpha: attr.f32("alphas", 1.0),
  },

  geometry: { type: "beam" },
  transform: "beam",
  shading: "lit",
  cullMode: "none",

  // Custom element count - segments, not points
  getElementCount: countSegments,

  // Custom centers for transparency sorting
  getCenters: getSegmentCenters,

  // Custom fill functions for complex data access
  fillRenderGeometry,
  fillPickingGeometry,
});
