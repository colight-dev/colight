/**
 * @module primitives/ellipsoid
 * @description Ellipsoid primitive using the declarative definition system.
 *
 * Ellipsoid renders spheres or ellipsoids defined by center, half-sizes (radii), and quaternion.
 */

import { BaseComponentConfig } from "../types";
import { definePrimitive, attr } from "./define";

// =============================================================================
// Configuration Interface
// =============================================================================

export interface EllipsoidComponentConfig extends BaseComponentConfig {
  type: "Ellipsoid" | "EllipsoidAxes";
  /** Ellipsoid centers: [x, y, z, ...] */
  centers: Float32Array | number[];
  /** Per-ellipsoid half sizes (radii): [rx, ry, rz, ...] */
  half_sizes?: Float32Array | number[];
  /** Default half size for all ellipsoids */
  half_size?: [number, number, number] | number;
  /** Per-ellipsoid rotations as quaternions: [w, x, y, z, ...] */
  quaternions?: Float32Array | number[];
  /** Default quaternion for all ellipsoids [w, x, y, z] */
  quaternion?: [number, number, number, number];
  /** Fill mode: Solid renders filled ellipsoids, MajorWireframe renders axis rings */
  fill_mode?: "Solid" | "MajorWireframe";
}

// =============================================================================
// Primitive Definition (fill functions are auto-generated via code generation)
// =============================================================================

export const ellipsoidSpec = definePrimitive<EllipsoidComponentConfig>({
  name: "Ellipsoid",

  attributes: {
    position: attr.vec3("centers"),
    size: attr.vec3("half_sizes", [0.5, 0.5, 0.5]),
    rotation: attr.quat("quaternions"), // default: identity [1,0,0,0] in wxyz
    color: attr.vec3("colors", [0.5, 0.5, 0.5]),
    alpha: attr.f32("alphas", 1.0),
  },

  geometry: { type: "sphere", stacks: 32, slices: 48 },
  transform: "rigid",
  shading: "lit",
  cullMode: "back",
});
