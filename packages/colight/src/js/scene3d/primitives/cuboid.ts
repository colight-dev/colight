/**
 * @module primitives/cuboid
 * @description Cuboid primitive using the declarative definition system.
 *
 * Cuboid renders axis-aligned or rotated boxes defined by center, half-sizes, and quaternion.
 */

import { BaseComponentConfig } from "../types";
import { definePrimitive, attr } from "./define";

// =============================================================================
// Configuration Interface
// =============================================================================

export interface CuboidComponentConfig extends BaseComponentConfig {
  type: "Cuboid";
  /** Cuboid centers: [x, y, z, ...] */
  centers: Float32Array;
  /** Per-cuboid half sizes: [hx, hy, hz, ...] */
  half_sizes?: Float32Array;
  /** Default half size for all cuboids */
  half_size?: number | [number, number, number];
  /** Per-cuboid rotations as quaternions: [w, x, y, z, ...] */
  quaternions?: Float32Array;
  /** Default quaternion for all cuboids [w, x, y, z] */
  quaternion?: [number, number, number, number];
}

// =============================================================================
// Primitive Definition (fill functions are auto-generated via code generation)
// =============================================================================

export const cuboidSpec = definePrimitive<CuboidComponentConfig>({
  name: "Cuboid",

  attributes: {
    position: attr.vec3("centers"),
    size: attr.vec3("half_sizes", [0.1, 0.1, 0.1]),
    rotation: attr.quat("quaternions"), // default: identity [1,0,0,0] in wxyz
    color: attr.vec3("colors", [0.5, 0.5, 0.5]),
    alpha: attr.f32("alphas", 1.0),
  },

  geometry: { type: "cube" },
  transform: "rigid",
  shading: "lit",
  cullMode: "none", // Allow seeing inside when camera is inside
});
