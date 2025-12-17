/**
 * @module primitives
 * @description Declarative primitive system for scene3d.
 *
 * This module provides:
 * - `definePrimitive()` - Factory for creating primitives from declarations
 * - `attr` - Attribute definition helpers (vec3, f32, quat, etc.)
 * - Ready-to-use primitive specs
 */

// Core framework
export {
  definePrimitive,
  attr,
  type PrimitiveDefinition,
  type ProcessedSchema,
  type AttributeDef,
  type AttributeType,
  type TransformType,
  type GeometrySource,
  // Re-exported utilities
  packID,
  acopy,
  createVertexBufferLayout,
  cameraStruct,
  lightingConstants,
  lightingCalc,
  pickingVSOut,
  pickingFragCode,
  quaternionShaderFunctions,
} from "./define";

// Primitive definitions
export { pointCloudSpec, type PointCloudComponentConfig } from "./pointCloud";
export { ellipsoidSpec, type EllipsoidComponentConfig } from "./ellipsoid";
export {
  ellipsoidAxesSpec,
  type EllipsoidAxesComponentConfig,
} from "./ellipsoidAxes";
export { cuboidSpec, type CuboidComponentConfig } from "./cuboid";
export { lineBeamsSpec, type LineBeamsComponentConfig } from "./lineBeams";
export {
  boundingBoxSpec,
  type BoundingBoxComponentConfig,
} from "./boundingBox";
