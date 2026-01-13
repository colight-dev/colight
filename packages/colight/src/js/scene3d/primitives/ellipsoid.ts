/**
 * @module primitives/ellipsoid
 * @description Ellipsoid primitive using the declarative definition system.
 *
 * Ellipsoid renders spheres or ellipsoids defined by center, half-sizes (radii), and quaternion.
 */

import { BaseComponentConfig } from "../types";
import {
  definePrimitive,
  attr,
  resolveSingular,
  expandScalar,
  coerceFloat32Fields,
} from "./define";

// =============================================================================
// Configuration Interface (internal format after coercion)
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
// Props Type (user-facing input)
// =============================================================================

export type EllipsoidProps = Omit<
  EllipsoidComponentConfig,
  "type" | "centers"
> & {
  centers?: ArrayLike<number> | ArrayBufferView;
  center?: [number, number, number];
};

// =============================================================================
// Coerce function (shared by Ellipsoid and EllipsoidAxes)
// =============================================================================

export function coerceEllipsoid(
  props: Record<string, any>,
): Record<string, any> {
  let coerced = resolveSingular(props, "center", "centers");
  coerced = expandScalar(coerced, "half_size");
  coerced = coerceFloat32Fields(coerced, [
    "centers",
    "half_sizes",
    "quaternions",
    "colors",
    "alphas",
  ]);
  const fillMode = coerced.fill_mode || "Solid";
  return {
    ...coerced,
    type: fillMode === "Solid" ? "Ellipsoid" : "EllipsoidAxes",
  };
}

// =============================================================================
// Primitive Definition
// =============================================================================

export const ellipsoidSpec = definePrimitive<EllipsoidComponentConfig>({
  name: "Ellipsoid",

  coerce: coerceEllipsoid,

  attributes: {
    position: attr.vec3("centers"),
    size: attr.vec3("half_sizes", [0.5, 0.5, 0.5]),
    rotation: attr.quat("quaternions"), // default: identity [1,0,0,0] in wxyz
    color: attr.vec3("colors", [0.5, 0.5, 0.5]),
    alpha: attr.f32("alphas", 1.0),
    groupId: attr.f32("_groupIds", 0),
  },

  geometry: { type: "sphere", stacks: 32, slices: 48 },
  transform: "rigid",
  shading: "lit",
  cullMode: "back",
});
