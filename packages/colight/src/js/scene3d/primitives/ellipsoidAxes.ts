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
  ProcessedSchema,
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
const ringVertexShader = /*wgsl*/ `
${cameraStruct}
${quaternionShaderFunctions}

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
};

fn computeRingPosition(
  center: vec3<f32>,
  offset: vec3<f32>,
  position: vec3<f32>,
  size: vec3<f32>,
  quaternion: vec4<f32>
) -> vec3<f32> {
  // Apply non-uniform scaling to the centerline.
  let scaledCenter = quat_rotate(quaternion, center * size);

  // Compute a uniform scale for the tube offset (e.g. average of nonuniform scales).
  let uniformScale = (size.x + size.y + size.z) / 3.0;
  let scaledOffset = quat_rotate(quaternion, offset * uniformScale);

  // Final world position: instance position plus transformed center and offset.
  return position + scaledCenter + scaledOffset;
}

@vertex
fn vs_main(
  @location(0) center: vec3<f32>,    // Centerline attribute
  @location(1) offset: vec3<f32>,    // Tube offset attribute
  @location(2) inNormal: vec3<f32>,  // Precomputed normal
  @location(3) position: vec3<f32>,  // Instance center
  @location(4) size: vec3<f32>,      // Instance non-uniform scaling
  @location(5) quaternion: vec4<f32>,// Instance rotation
  @location(6) color: vec3<f32>,     // Color attribute
  @location(7) alpha: f32            // Alpha attribute
) -> VSOut {
  let worldPos = computeRingPosition(center, offset, position, size, quaternion);

  // For normals, we want the tube's offset direction unperturbed by nonuniform scaling.
  let worldNormal = quat_rotate(quaternion, normalize(offset));

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  out.normal = worldNormal;
  return out;
}`;

const ringPickingVertexShader = /*wgsl*/ `
${cameraStruct}
${quaternionShaderFunctions}
${pickingVSOut}

fn computeRingPosition(
  center: vec3<f32>,
  offset: vec3<f32>,
  position: vec3<f32>,
  size: vec3<f32>,
  quaternion: vec4<f32>
) -> vec3<f32> {
  let scaledCenter = quat_rotate(quaternion, center * size);
  let uniformScale = (size.x + size.y + size.z) / 3.0;
  let scaledOffset = quat_rotate(quaternion, offset * uniformScale);
  return position + scaledCenter + scaledOffset;
}

@vertex
fn vs_main(
  @location(0) center: vec3<f32>,
  @location(1) offset: vec3<f32>,
  @location(2) inNormal: vec3<f32>,
  @location(3) position: vec3<f32>,
  @location(4) size: vec3<f32>,
  @location(5) quaternion: vec4<f32>,
  @location(6) pickID: f32
) -> VSOut {
  let worldPos = computeRingPosition(center, offset, position, size, quaternion);

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

const ringFragmentShader = /*wgsl*/ `
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
) -> @location(0) vec4<f32> {
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

/** Ring instance layout: position(3) + size(3) + quat(4) + color(3) + alpha(1) = 14 */
const RING_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [3, "float32x3"], // instance center position
    [4, "float32x3"], // instance size
    [5, "float32x4"], // instance quaternion
    [6, "float32x3"], // instance color
    [7, "float32"], // instance alpha
  ],
  "instance",
);

/** Ring picking instance layout: position(3) + size(3) + quat(4) + pickID(1) = 11 */
const RING_PICKING_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [3, "float32x3"], // position
    [4, "float32x3"], // size
    [5, "float32x4"], // quaternion
    [6, "float32"], // pickID
  ],
  "instance",
);

// =============================================================================
// Custom Fill Functions
// =============================================================================

function fillRenderGeometry(
  schema: ProcessedSchema,
  constants: Record<string, unknown>,
  elem: EllipsoidAxesComponentConfig,
  elemIndex: number,
  out: Float32Array,
  outIndex: number,
): void {
  const floatsPerInstance = 14;
  const outOffset = outIndex * floatsPerInstance;

  // Position
  acopy(elem.centers, elemIndex * 3, out, outOffset, 3);

  // Half sizes
  if (constants.half_size) {
    acopy(constants.half_size as ArrayLike<number>, 0, out, outOffset + 3, 3);
  } else if (elem.half_sizes) {
    acopy(elem.half_sizes, elemIndex * 3, out, outOffset + 3, 3);
  } else {
    out[outOffset + 3] = 0.5;
    out[outOffset + 4] = 0.5;
    out[outOffset + 5] = 0.5;
  }

  // Quaternion (input is wxyz order, output to shader is xyzw)
  if (constants.quaternion) {
    const q = constants.quaternion as number[];
    out[outOffset + 6] = q[1]; // x from input[1]
    out[outOffset + 7] = q[2]; // y from input[2]
    out[outOffset + 8] = q[3]; // z from input[3]
    out[outOffset + 9] = q[0]; // w from input[0]
  } else if (elem.quaternions) {
    const base = elemIndex * 4;
    out[outOffset + 6] = elem.quaternions[base + 1]; // x
    out[outOffset + 7] = elem.quaternions[base + 2]; // y
    out[outOffset + 8] = elem.quaternions[base + 3]; // z
    out[outOffset + 9] = elem.quaternions[base + 0]; // w
  } else {
    // Identity quaternion in xyzw: [0, 0, 0, 1]
    out[outOffset + 6] = 0;
    out[outOffset + 7] = 0;
    out[outOffset + 8] = 0;
    out[outOffset + 9] = 1;
  }

  // Color
  if (constants.color) {
    const c = constants.color as number[];
    out[outOffset + 10] = c[0];
    out[outOffset + 11] = c[1];
    out[outOffset + 12] = c[2];
  } else if (elem.colors) {
    acopy(elem.colors, elemIndex * 3, out, outOffset + 10, 3);
  } else {
    out[outOffset + 10] = 0.5;
    out[outOffset + 11] = 0.5;
    out[outOffset + 12] = 0.5;
  }

  // Alpha
  out[outOffset + 13] =
    (constants.alpha as number) ?? elem.alphas?.[elemIndex] ?? 1.0;
}

function fillPickingGeometry(
  schema: ProcessedSchema,
  constants: Record<string, unknown>,
  elem: EllipsoidAxesComponentConfig,
  elemIndex: number,
  out: Float32Array,
  outIndex: number,
  baseID: number,
): void {
  const floatsPerPicking = 11;
  const outOffset = outIndex * floatsPerPicking;

  // Position
  acopy(elem.centers, elemIndex * 3, out, outOffset, 3);

  // Half sizes
  if (constants.half_size) {
    acopy(constants.half_size as ArrayLike<number>, 0, out, outOffset + 3, 3);
  } else if (elem.half_sizes) {
    acopy(elem.half_sizes, elemIndex * 3, out, outOffset + 3, 3);
  } else {
    out[outOffset + 3] = 0.5;
    out[outOffset + 4] = 0.5;
    out[outOffset + 5] = 0.5;
  }

  // Quaternion (input is wxyz order, output to shader is xyzw)
  if (constants.quaternion) {
    const q = constants.quaternion as number[];
    out[outOffset + 6] = q[1]; // x from input[1]
    out[outOffset + 7] = q[2]; // y from input[2]
    out[outOffset + 8] = q[3]; // z from input[3]
    out[outOffset + 9] = q[0]; // w from input[0]
  } else if (elem.quaternions) {
    const base = elemIndex * 4;
    out[outOffset + 6] = elem.quaternions[base + 1]; // x
    out[outOffset + 7] = elem.quaternions[base + 2]; // y
    out[outOffset + 8] = elem.quaternions[base + 3]; // z
    out[outOffset + 9] = elem.quaternions[base + 0]; // w
  } else {
    // Identity quaternion in xyzw: [0, 0, 0, 1]
    out[outOffset + 6] = 0;
    out[outOffset + 7] = 0;
    out[outOffset + 8] = 0;
    out[outOffset + 9] = 1;
  }

  // Pick ID - same for all 3 rings of this ellipsoid
  out[outOffset + 10] = packID(baseID + elemIndex);
}

// =============================================================================
// Primitive Definition
// =============================================================================

export const ellipsoidAxesSpec = definePrimitive<EllipsoidAxesComponentConfig>({
  name: "EllipsoidAxes",

  // Schema used for constants computation (even though fill is custom)
  attributes: {
    position: attr.vec3("centers"),
    size: attr.vec3("half_sizes", [0.5, 0.5, 0.5]),
    rotation: attr.quat("quaternions"), // default: identity [1,0,0,0] in wxyz
    color: attr.vec3("colors", [0.5, 0.5, 0.5]),
    alpha: attr.f32("alphas", 1.0),
  },

  // 3 rings per ellipsoid
  instancesPerElement: 3,

  // Custom geometry - rings with tube offset
  geometry: {
    type: "custom",
    create: () => createEllipsoidAxes(1.0, 0.05, 32, 16),
  },

  // Custom layouts for the ring vertex format
  geometryLayout: RING_GEOMETRY_LAYOUT,
  renderInstanceLayout: RING_INSTANCE_LAYOUT,
  pickingInstanceLayout: RING_PICKING_INSTANCE_LAYOUT,

  // Custom shaders for ring transform
  vertexShader: ringVertexShader,
  pickingVertexShader: ringPickingVertexShader,
  fragmentShader: ringFragmentShader,

  // Not used but matches schema
  transform: "rigid",
  shading: "lit",
  cullMode: "back",

  // Custom fill functions
  fillRenderGeometry,
  fillPickingGeometry,
});
