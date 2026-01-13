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
  // Coercion helpers
  resolveSingular,
  expandScalar,
} from "./define";

// Primitive definitions and Props types
export {
  pointCloudSpec,
  type PointCloudComponentConfig,
  type PointCloudProps,
} from "./pointCloud";
export {
  ellipsoidSpec,
  type EllipsoidComponentConfig,
  type EllipsoidProps,
  coerceEllipsoid,
} from "./ellipsoid";
export {
  ellipsoidAxesSpec,
  type EllipsoidAxesComponentConfig,
} from "./ellipsoidAxes";
export {
  cuboidSpec,
  type CuboidComponentConfig,
  type CuboidProps,
} from "./cuboid";
export { lineBeamsSpec, type LineBeamsComponentConfig } from "./lineBeams";
export {
  lineSegmentsSpec,
  type LineSegmentsComponentConfig,
} from "./lineSegments";
export {
  boundingBoxSpec,
  type BoundingBoxComponentConfig,
  type BoundingBoxProps,
} from "./boundingBox";
export {
  imagePlaneSpec,
  type ImagePlaneComponentConfig,
  type ImagePlaneProps,
  type ImageSource,
  getImageBindGroupLayout,
} from "./imagePlane";

// Generic mesh factory
export {
  defineMesh,
  defineMeshRaw,
  interleaveVertexData,
  getFormatKey,
  type MeshComponentConfig,
  type StructuredGeometry,
  type VertexFormat,
  type ProcessedMeshGeometry,
  type MeshOptions,
} from "./mesh";
