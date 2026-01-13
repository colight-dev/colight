/**
 * @module primitives/cuboid
 * @description Cuboid primitive using the declarative definition system.
 *
 * Cuboid renders axis-aligned or rotated boxes defined by center, half-sizes, and quaternion.
 */

import { BaseComponentConfig } from "../types";
import { definePrimitive, attr, resolveSingular, expandScalar } from "./define";

// =============================================================================
// Configuration Interface (internal format after coercion)
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
// Props Type (user-facing input)
// =============================================================================

export type CuboidProps = Omit<CuboidComponentConfig, "type" | "centers"> & {
  centers?: ArrayLike<number> | ArrayBufferView;
  center?: [number, number, number];
};

// =============================================================================
// Primitive Definition
// =============================================================================

export const cuboidSpec = definePrimitive<CuboidComponentConfig>({
  name: "Cuboid",

  coerce(props) {
    let coerced = resolveSingular(props, "center", "centers");
    coerced = expandScalar(coerced, "half_size");
    return { ...coerced, type: "Cuboid" };
  },

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
