/**
 * @module primitives/pointCloud
 * @description PointCloud primitive using the declarative definition system.
 *
 * PointCloud renders points as camera-facing billboards (quads that always face the camera).
 */

import { BaseComponentConfig } from "../types";
import { definePrimitive, attr, resolveSingular } from "./define";

// =============================================================================
// Configuration Interface (internal format after coercion)
// =============================================================================

export interface PointCloudComponentConfig extends BaseComponentConfig {
  type: "PointCloud";
  /** Point positions: [x, y, z, ...] */
  centers: Float32Array;
  /** Per-point sizes */
  sizes?: Float32Array;
  /** Default size for all points */
  size?: number;
}

// =============================================================================
// Props Type (user-facing input)
// =============================================================================

export type PointCloudProps = Omit<
  PointCloudComponentConfig,
  "type" | "centers"
> & {
  centers?: ArrayLike<number> | ArrayBufferView;
  center?: [number, number, number];
};

// =============================================================================
// Primitive Definition
// =============================================================================

export const pointCloudSpec = definePrimitive<PointCloudComponentConfig>({
  name: "PointCloud",

  coerce(props) {
    return {
      ...resolveSingular(props, "center", "centers"),
      type: "PointCloud",
    };
  },

  attributes: {
    position: attr.vec3("centers"),
    size: attr.f32("sizes", 0.02),
    color: attr.vec3("colors", [0.5, 0.5, 0.5]),
    alpha: attr.f32("alphas", 1.0),
  },

  geometry: { type: "quad" },
  transform: "billboard",
  shading: "unlit",
  cullMode: "none",
});
