/**
 * @module primitives/ellipsoidAxes
 * @description EllipsoidAxes primitive using the declarative definition system.
 *
 * EllipsoidAxes renders three rings (one for each principal axis) to visualize
 * ellipsoid orientation. This is useful for showing rotation and scale.
 */

import { BaseComponentConfig } from "../types";
import {
  definePrimitive,
  attr,
  processSchema,
  instanceInputsWGSL,
  filterCollapseWGSL,
  rigidGroupTransformFn,
  type AttributeDef,
  createVertexBufferLayout,
  cameraStruct,
  clipPlanesStruct,
  groupTransformStruct,
  filterParamsStruct,
  applyGroupTransformFn,
  lightingConstants,
  lightingCalc,
  pickingVSOut,
  quaternionShaderFunctions,
} from "./define";
import { createEllipsoidAxes } from "../geometry";

// =============================================================================
// Configuration Interface (shared with Ellipsoid)
// =============================================================================

export interface EllipsoidAxesComponentConfig extends BaseComponentConfig {
  type: "EllipsoidAxes";
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
}

// =============================================================================
// Custom Shaders
// =============================================================================

/**
 * Ring geometry uses a different vertex format:
 * - centerline position (vec3) - center of the tube ring
 * - tube offset (vec3) - offset from centerline to vertex
 * - normal (vec3) - vertex normal
 */
/**
 * Per-instance attribute schema. The framework attributes (transformIndex /
 * filterValue / filterIndex) are auto-injected by processSchema.
 */
const AXES_ATTRIBUTES: Record<string, AttributeDef> = {
  position: attr.vec3("centers"),
  size: attr.vec3("half_sizes", [0.5, 0.5, 0.5]),
  rotation: attr.quat("quaternions"), // default: identity [1,0,0,0] in wxyz
  color: attr.vec3("colors", [0.5, 0.5, 0.5]),
  alpha: attr.f32("alphas", 1.0),
};

// The ring geometry layout is three locations wide (centerline, tube offset,
// normal), so instance attributes start at 3.
const INSTANCE_START_LOCATION = 3;

const schema = processSchema(AXES_ATTRIBUTES, INSTANCE_START_LOCATION);

/**
 * Shared vertex-shader body.
 *
 * EllipsoidAxes keeps its ring semantics — the centerline is scaled by the
 * ellipsoid's (possibly non-uniform) half extents, while the tube cross-section
 * offset takes a single uniform scale so the tube stays circular — but the
 * group transform now composes via the shared rigidGroupFrame, like every
 * other primitive.
 */
const ringTransformBody = /*wgsl*/ `
  let groupIdx = u32(transformIndex);
  let _frame = rigidGroupFrame(position, size, rotation, groupIdx);

  // Ring centerline: non-uniformly scaled by the effective half extents.
  let scaledCenter = quat_rotate(_frame.quat, center * _frame.size);
  // Tube cross-section: uniform scale (mean of the effective extents) so the
  // tube keeps a circular profile on a non-uniform ellipsoid.
  let uniformScale = (_frame.size.x + _frame.size.y + _frame.size.z) / 3.0;
  let scaledOffset = quat_rotate(_frame.quat, offset * uniformScale);

  let worldPos = _frame.origin + scaledCenter + scaledOffset;`;

const ringVertexShader = /*wgsl*/ `
${cameraStruct}
${groupTransformStruct}
${filterParamsStruct}
${quaternionShaderFunctions}
${applyGroupTransformFn}
${rigidGroupTransformFn}

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
};

@vertex
fn vs_main(
  @location(0) center: vec3<f32>,    // Centerline attribute
  @location(1) offset: vec3<f32>,    // Tube offset attribute
  @location(2) inNormal: vec3<f32>,  // Precomputed normal
  // Instance attributes (schema-derived)
  ${instanceInputsWGSL(schema, false).join(",\n  ")}
) -> VSOut {
${filterCollapseWGSL.test}
${ringTransformBody}

  // The tube's outward normal is its (normalized) cross-section offset,
  // rotated through the same composed quaternion the position uses.
  let worldNormal = quat_rotate(_frame.quat, normalize(offset));

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  out.normal = worldNormal;${filterCollapseWGSL.collapse}
  return out;
}`;

const ringPickingVertexShader = /*wgsl*/ `
${cameraStruct}
${groupTransformStruct}
${filterParamsStruct}
${quaternionShaderFunctions}
${pickingVSOut}
${applyGroupTransformFn}
${rigidGroupTransformFn}

@vertex
fn vs_main(
  @location(0) center: vec3<f32>,
  @location(1) offset: vec3<f32>,
  @location(2) inNormal: vec3<f32>,
  // Instance attributes (schema-derived, pickID last)
  ${instanceInputsWGSL(schema, true).join(",\n  ")}
) -> VSOut {
${filterCollapseWGSL.test}
${ringTransformBody}

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  out.worldPos = worldPos;${filterCollapseWGSL.collapse}
  return out;
}`;

const ringFragmentShader = /*wgsl*/ `
${cameraStruct}
${clipPlanesStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
) -> @location(0) vec4<f32> {
  applyClipPlanes(worldPos);
  let litColor = calculateLighting(color, normal, worldPos);
  return vec4<f32>(litColor, alpha);
}`;

// =============================================================================
// Custom Buffer Layouts
// =============================================================================

/** Ring geometry has 9 floats per vertex: center(3) + offset(3) + normal(3) */
const RING_GEOMETRY_LAYOUT = createVertexBufferLayout(
  [
    [0, "float32x3"], // centerline position
    [1, "float32x3"], // tube offset
    [2, "float32x3"], // normal
  ],
  "vertex",
);

// =============================================================================
// Primitive Definition
// =============================================================================

export const ellipsoidAxesSpec = definePrimitive<EllipsoidAxesComponentConfig>({
  name: "EllipsoidAxes",

  // Carried through from Ellipsoid coercion (fill_mode: "MajorWireframe")
  extraProps: ["fill_mode"],

  attributes: AXES_ATTRIBUTES,

  // 3 rings per ellipsoid
  instancesPerElement: 3,

  // Custom geometry - rings with tube offset
  geometry: {
    type: "custom",
    create: () => createEllipsoidAxes(1.0, 0.05, 32, 16),
  },

  // Custom geometry layout (3 locations); instance layout and fill are
  // schema-derived from AXES_ATTRIBUTES + INSTANCE_START_LOCATION.
  geometryLayout: RING_GEOMETRY_LAYOUT,
  instanceStartLocation: INSTANCE_START_LOCATION,

  // Custom shaders for the ring transform. Group transform and filter
  // collapse come from the shared helpers.
  vertexShader: ringVertexShader,
  pickingVertexShader: ringPickingVertexShader,
  fragmentShader: ringFragmentShader,

  transform: "rigid", // Not used (custom shaders), but required
  shading: "lit",
  cullMode: "back",
});
