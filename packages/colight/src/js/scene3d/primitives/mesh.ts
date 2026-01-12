/**
 * @module primitives/mesh
 * @description Generic mesh primitive factory using the declarative definition system.
 *
 * Allows creating custom mesh primitives from geometry data (vertices + indices).
 */

import { BaseComponentConfig, GeometryData } from "../types";
import { definePrimitive, attr } from "./define";

// =============================================================================
// Configuration Interface
// =============================================================================

export interface MeshComponentConfig extends BaseComponentConfig {
  type: string; // The user-provided name
  /** Mesh centers: [x, y, z, ...] */
  centers: Float32Array | number[];
  /** Per-instance scales: [sx, sy, sz, ...] */
  scales?: Float32Array | number[];
  /** Default scale for all instances */
  scale?: [number, number, number] | number;
  /** Per-instance rotations as quaternions: [w, x, y, z, ...] */
  quaternions?: Float32Array | number[];
  /** Default quaternion for all instances [w, x, y, z] */
  quaternion?: [number, number, number, number];
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Defines a new mesh primitive type from geometry data.
 *
 * @param name - Unique name for this primitive type
 * @param geometry - Geometry data (vertices/optional indices) or a function that creates it
 * @param options - Optional rendering configuration
 */
export function defineMesh(
  name: string,
  geometry: GeometryData | (() => GeometryData),
  options: {
    /** Shading model: "lit" (default) or "unlit" */
    shading?: "lit" | "unlit";
    /** Face culling mode: "back" (default), "front", or "none" */
    cullMode?: GPUCullMode;
  } = {},
) {
  return definePrimitive<MeshComponentConfig>({
    name,

    attributes: {
      position: attr.vec3("centers"),
      size: attr.vec3("scales", [1, 1, 1]),
      rotation: attr.quat("quaternions"), // default: identity [1,0,0,0] in wxyz
      color: attr.vec3("colors", [0.5, 0.5, 0.5]),
      alpha: attr.f32("alphas", 1.0),
    },

    geometry: {
      type: "custom",
      create: typeof geometry === "function" ? geometry : () => geometry,
    },

    transform: "rigid",
    shading: options.shading ?? "lit",
    cullMode: options.cullMode ?? "back",
  });
}
