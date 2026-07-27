/**
 * @module primitives/define
 * @description Declarative primitive definition framework for scene3d.
 *
 * This module provides a way to define 3D primitives declaratively using schemas,
 * auto-generating shaders, buffer layouts, and data-filling functions.
 */

import {
  BaseComponentConfig,
  BASE_COMPONENT_PROPS,
  PrimitiveSpec,
  PipelineCacheEntry,
  GeometryResource,
  GeometryData,
  ElementConstants,
  VertexBufferLayout,
} from "../types";
import {
  createRenderPipeline,
  createTranslucentGeometryPipeline,
  createOverlayPipeline,
  createOverlayPickingPipeline,
  createBuffers,
  getOrCreatePipeline,
} from "../pipelineUtils";
import { packID } from "../picking";
import { acopy } from "../utils";

// Re-export utilities for use in custom fill functions and shader definitions
export { packID, acopy };

// Re-export shader utilities for custom shaders
export {
  createVertexBufferLayout,
  cameraStruct,
  groupTransformStruct,
  filterParamsStruct,
  clipPlanesStruct,
  applyGroupTransformFn,
  lightingConstants,
  lightingCalc,
  pickingVSOut,
  pickingFragCode,
} from "../shaders";

export { quaternionShaderFunctions } from "../quaternion";

// =============================================================================
// Input Coercion Helpers
// =============================================================================

import { coerceToFloat32 } from "../arrayUtils";

// Re-export for use in primitive coerce functions
export { coerceToFloat32 };

/**
 * Coerce specified fields to Float32Array if they exist.
 * Handles NdArrayView, regular arrays, and TypedArrays.
 */
export function coerceFloat32Fields<T extends object>(
  obj: T,
  fields: (keyof T)[],
): T {
  const result = { ...obj };
  for (const field of fields) {
    const value = obj[field];
    if (value !== undefined) {
      (result[field] as any) = coerceToFloat32(value);
    }
  }
  return result;
}

/**
 * Resolve a singular prop to its plural form by wrapping in an array.
 * e.g., `center: [0,0,0]` → `centers: [[0,0,0]]`
 */
export function resolveSingular<T extends Record<string, any>>(
  props: T,
  singular: string,
  plural: string,
): T {
  if (props[singular] !== undefined && props[plural] === undefined) {
    const { [singular]: value, ...rest } = props;
    return { ...rest, [plural]: [value] } as T;
  }
  // Remove singular if plural is already set
  if (props[singular] !== undefined && props[plural] !== undefined) {
    const { [singular]: _, ...rest } = props;
    return rest as T;
  }
  return props;
}

/**
 * Expand a scalar value to a vec3 array.
 * e.g., `half_size: 0.5` → `half_size: [0.5, 0.5, 0.5]`
 */
export function expandScalar<T extends Record<string, any>>(
  props: T,
  field: string,
): T {
  const val = props[field];
  if (typeof val === "number") {
    return { ...props, [field]: [val, val, val] } as T;
  }
  return props;
}

// =============================================================================
// Attribute Schema Types
// =============================================================================

export type AttributeType = "f32" | "vec2" | "vec3" | "vec4" | "quat";

/** Maps attribute types to their float counts */
const ATTRIBUTE_SIZES: Record<AttributeType, number> = {
  f32: 1,
  vec2: 2,
  vec3: 3,
  vec4: 4,
  quat: 4,
};

/** Maps attribute types to WebGPU vertex formats */
const ATTRIBUTE_FORMATS: Record<AttributeType, GPUVertexFormat> = {
  f32: "float32",
  vec2: "float32x2",
  vec3: "float32x3",
  vec4: "float32x4",
  quat: "float32x4",
};

/** Maps attribute types to WGSL types */
const ATTRIBUTE_WGSL_TYPES: Record<AttributeType, string> = {
  f32: "f32",
  vec2: "vec2<f32>",
  vec3: "vec3<f32>",
  vec4: "vec4<f32>",
  quat: "vec4<f32>",
};

/** Quaternion component order in input data */
export type QuaternionOrder = "wxyz" | "xyzw";

export interface AttributeDef<T = unknown> {
  type: AttributeType;
  /** Source field name in the component config (plural form, e.g., "centers") */
  source: string;
  /** Default value if not provided per-instance */
  default?: T;
  /** For picking shader: include this attribute? Defaults to true for geometry, false for color/alpha */
  picking?: boolean;
  /** For quaternions: input data order. Shader always uses xyzw internally. Default: "wxyz" */
  quaternionOrder?: QuaternionOrder;
}

/** Attribute definition helpers */
export const attr = {
  f32: (source: string, defaultValue?: number): AttributeDef<number> => ({
    type: "f32",
    source,
    default: defaultValue,
  }),

  vec2: (
    source: string,
    defaultValue?: [number, number],
  ): AttributeDef<[number, number]> => ({
    type: "vec2",
    source,
    default: defaultValue,
  }),

  vec3: (
    source: string,
    defaultValue?: [number, number, number],
  ): AttributeDef<[number, number, number]> => ({
    type: "vec3",
    source,
    default: defaultValue,
  }),

  vec4: (
    source: string,
    defaultValue?: [number, number, number, number],
  ): AttributeDef<[number, number, number, number]> => ({
    type: "vec4",
    source,
    default: defaultValue,
  }),

  /**
   * Quaternion attribute. Default order is "wxyz" (w first).
   * Shader internally uses xyzw, so codegen will swizzle as needed.
   * Default value is identity quaternion [1,0,0,0] in wxyz format.
   */
  quat: (
    source: string,
    defaultValue: [number, number, number, number] = [1, 0, 0, 0], // identity in wxyz
    order: QuaternionOrder = "wxyz",
  ): AttributeDef<[number, number, number, number]> => ({
    type: "quat",
    source,
    default: defaultValue,
    picking: true, // quaternions are usually needed for picking geometry
    quaternionOrder: order,
  }),
};

// =============================================================================
// Transform Types
// =============================================================================

export type TransformType =
  | "billboard"
  | "rigid"
  | "beam"
  | "screenspace_offset";

// =============================================================================
// Geometry Types
// =============================================================================

export type GeometrySource =
  | { type: "quad" }
  | { type: "sphere"; stacks?: number; slices?: number }
  | { type: "cube" }
  | { type: "beam" }
  | { type: "cylinderBeam"; segments?: number }
  | { type: "custom"; create: () => GeometryData };

// =============================================================================
// Primitive Definition
// =============================================================================

export interface PrimitiveDefinition<Config extends BaseComponentConfig> {
  /** Unique name for this primitive type */
  name: string;

  /**
   * Input coercion function. Transforms user props to internal config format.
   * Handles aliases (center → centers), scalar expansion (half_size: 0.5 → [0.5, 0.5, 0.5]), etc.
   * Should return props with `type` field set.
   */
  coerce?: (props: Record<string, any>) => Record<string, any>;

  /**
   * Additional accepted prop names beyond the attribute schema and the
   * shared BaseComponentConfig props (e.g. "fill_mode" for Ellipsoid).
   * Used to build the spec's `knownProps` set for unknown-key warnings.
   */
  extraProps?: string[];

  /**
   * Attribute schema defining instance data.
   * Order matters - this determines buffer layout.
   * Standard attributes (color, alpha) should come last.
   */
  attributes: Record<string, AttributeDef>;

  /** Base geometry for this primitive */
  geometry: GeometrySource;

  /** Transform type determining vertex shader behavior */
  transform: TransformType;

  /** Shading model: "lit" uses Blinn-Phong, "unlit" uses flat color */
  shading?: "lit" | "unlit";

  /** Face culling mode */
  cullMode?: GPUCullMode;

  /** Primitive topology */
  topology?: GPUPrimitiveTopology;

  /**
   * Custom element count function.
   * Default: `config[firstAttribute.source].length / attributeSize`
   */
  getElementCount?: (config: Config) => number;

  /**
   * Custom centers function for transparency sorting.
   * Default: uses first vec3 attribute as centers.
   */
  getCenters?: (config: Config) => Float32Array | number[];

  /**
   * Number of render instances per logical element.
   * Default: 1. Use >1 for things like EllipsoidAxes (3 rings per ellipsoid).
   */
  instancesPerElement?: number;

  /**
   * Custom color index function.
   * Use when color should come from a different index than element index.
   */
  getColorIndexForInstance?: (config: Config, elementIndex: number) => number;

  /**
   * Preprocessing hook called once per config.
   * Useful for computing derived data (e.g., segment maps for lines).
   */
  preprocess?: (config: Config) => void;

  /**
   * Custom fillRenderGeometry function.
   * Use when data access patterns are too complex for the schema.
   * If provided, bypasses auto-generated fill function.
   * Receives the processed schema for offset information.
   */
  fillRenderGeometry?: (
    schema: ProcessedSchema,
    constants: ElementConstants,
    elem: Config,
    elemIndex: number,
    out: Float32Array,
    outIndex: number,
  ) => void;

  /**
   * Custom fillPickingGeometry function.
   * Use when data access patterns are too complex for the schema.
   * If provided, bypasses auto-generated fill function.
   * Receives the processed schema for offset information.
   */
  fillPickingGeometry?: (
    schema: ProcessedSchema,
    constants: ElementConstants,
    elem: Config,
    elemIndex: number,
    out: Float32Array,
    outIndex: number,
    baseID: number,
  ) => void;

  /**
   * Custom vertex shader code.
   * If provided, bypasses auto-generated vertex shader.
   * Must export vs_main function.
   */
  vertexShader?: string;

  /**
   * Custom picking vertex shader code.
   * If provided, bypasses auto-generated picking vertex shader.
   * Must export vs_main function.
   */
  pickingVertexShader?: string;

  /**
   * Custom fragment shader code.
   * If provided, bypasses auto-generated fragment shader.
   */
  fragmentShader?: string;

  /**
   * Custom geometry buffer layout.
   * Default is position(vec3) + normal(vec3) = 24 bytes.
   */
  geometryLayout?: VertexBufferLayout;

  /**
   * Shader location of the first instance attribute. Default 2, matching the
   * default geometry layout (position at 0, normal at 1). Primitives with a
   * wider `geometryLayout` must pass the number of geometry locations so the
   * schema-derived instance layout starts after them.
   *
   * Prefer this over hand-writing `renderInstanceLayout` /
   * `pickingInstanceLayout`: the schema then remains the single source of
   * truth for offsets, strides and shader locations.
   */
  instanceStartLocation?: number;

  /**
   * Custom render instance buffer layout.
   * If provided, bypasses auto-generated layout from schema.
   *
   * Avoid: a hand-written layout must be kept in sync by hand with the
   * schema's `floatsPerInstance` (which sizes the CPU-side fill buffer) and
   * with the auto-injected framework attributes. Use `instanceStartLocation`
   * instead unless the instance record genuinely cannot be expressed as a
   * schema.
   */
  renderInstanceLayout?: VertexBufferLayout;

  /**
   * Custom picking instance buffer layout.
   * If provided, bypasses auto-generated layout from schema.
   * See the caveat on `renderInstanceLayout`.
   */
  pickingInstanceLayout?: VertexBufferLayout;

  /**
   * Custom vertex shader entry point name.
   * Default: "vs_main"
   */
  vertexEntryPoint?: string;

  /**
   * Custom picking vertex shader entry point name.
   * Default: "vs_main"
   */
  pickingVertexEntryPoint?: string;

  /**
   * Optional bind group layouts to use for pipeline creation.
   * If provided, replaces the default single bind group layout.
   */
  bindGroupLayouts?: (
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
  ) => GPUBindGroupLayout[];
}

// =============================================================================
// Schema Processing
// =============================================================================

/** Processed attribute with precomputed values */
interface ProcessedAttr {
  name: string;
  def: AttributeDef;
  offset: number;
  /** Precomputed size from ATTRIBUTE_SIZES */
  size: number;
  /** Precomputed constant key (singular form of source name for constants lookup) */
  constKey: string;
  /** For quaternions: input order (default wxyz). Shader uses xyzw internally. */
  quaternionOrder?: QuaternionOrder;
}

export interface ProcessedSchema {
  /** All attributes in order */
  attributes: ProcessedAttr[];

  /**
   * Shader location of the first instance attribute. Instance attributes
   * occupy `[instanceStartLocation, instanceStartLocation + n)`; pickID (in
   * the picking pass) sits immediately after the last picking attribute.
   */
  instanceStartLocation: number;

  /** Attributes for render pass (all) */
  renderAttributes: ProcessedAttr[];

  /** Attributes for picking pass (geometry only, plus pickID) */
  pickingAttributes: ProcessedAttr[];

  /** Total floats per render instance */
  floatsPerInstance: number;

  /** Total floats per picking instance */
  floatsPerPicking: number;

  /** Offset of color attribute in render buffer */
  colorOffset: number;

  /** Offset of alpha attribute in render buffer */
  alphaOffset: number;

  /** WebGPU buffer layout for render pass */
  renderLayout: VertexBufferLayout;

  /** WebGPU buffer layout for picking pass */
  pickingLayout: VertexBufferLayout;
}

/**
 * Turns an attribute schema into instance-buffer offsets, strides and WebGPU
 * vertex layouts, auto-injecting the framework attributes every primitive gets
 * (see the comment inside).
 *
 * Exported so primitives with hand-written shaders can derive their layouts,
 * strides and shader input declarations from the SAME schema the generated
 * path uses — there is exactly one place that decides what an instance record
 * looks like. Hand-written shaders should pair this with
 * `instanceInputsWGSL` / `filterCollapseWGSL` rather than hardcoding
 * locations, strides or offsets.
 *
 * @param attributes Primitive-specific attributes, in buffer order.
 * @param instanceStartLocation Shader location of the first instance
 *   attribute. Defaults to 2 (position + normal geometry attributes at 0/1);
 *   primitives with wider geometry layouts pass their own.
 */
export function processSchema(
  attributes: Record<string, AttributeDef>,
  instanceStartLocation: number = 2,
): ProcessedSchema {
  // Auto-inject transformIndex for GPU group transforms and filterIndex for
  // per-instance filtering. Both are added to every primitive for consistency.
  //
  // - transformIndex looks up the group world transform in the transforms buffer.
  // - filterValue is the per-instance scalar the threshold predicate tests. It is
  //   uploaded once as instance data; the min/max thresholds live in a small
  //   filterParams storage buffer indexed by filterIndex, so a threshold change
  //   (e.g. a $state slider) rewrites only that tiny buffer, never instance data.
  const augmentedAttributes: Record<string, AttributeDef> = {
    ...attributes,
    transformIndex: {
      type: "f32",
      source: "_transformIndex",
      default: 0,
      picking: true, // Need same transform for picking geometry
    },
    filterValue: {
      type: "f32",
      // Per-instance scalar array on the component (set by the compiler from
      // the user's filter_by.values). NaN default => "no filter value"
      // sentinel; ignored when the component's filterParams slot is inactive.
      source: "_filterValues",
      default: Number.NaN,
      picking: true, // Filtered-out instances must be unpickable too
    },
    filterIndex: {
      type: "f32",
      source: "_filterIndex",
      default: 0,
      picking: true, // Same filter thresholds for picking geometry
    },
  };

  const attrList: ProcessedAttr[] = [];

  // First pass: collect all attributes with render offsets and precomputed values
  let renderOffset = 0;
  for (const [name, def] of Object.entries(augmentedAttributes)) {
    const size = ATTRIBUTE_SIZES[def.type];
    // Precompute constant key: singular form of source name (e.g., "half_sizes" -> "half_size")
    const source = def.source;
    const constKey = source.endsWith("s") ? source.slice(0, -1) : source;
    // Carry quaternion order for codegen
    const quaternionOrder =
      def.type === "quat" ? (def.quaternionOrder ?? "wxyz") : undefined;
    attrList.push({
      name,
      def,
      offset: renderOffset,
      size,
      constKey,
      quaternionOrder,
    });
    renderOffset += size;
  }

  // Find color and alpha offsets
  const colorAttr = attrList.find((a) => a.name === "color");
  const alphaAttr = attrList.find((a) => a.name === "alpha");

  const colorOffset = colorAttr?.offset ?? -1;
  const alphaOffset = alphaAttr?.offset ?? -1;

  // Picking attributes: exclude color and alpha, but include geometry attributes
  const pickingAttrs: ProcessedAttr[] = [];
  let pickingOffset = 0;
  for (const attr of attrList) {
    // Skip color and alpha for picking
    if (attr.name === "color" || attr.name === "alpha") continue;

    // Include if explicitly marked for picking, or if not explicitly excluded
    if (attr.def.picking !== false) {
      pickingAttrs.push({ ...attr, offset: pickingOffset });
      pickingOffset += attr.size;
    }
  }

  // Add pickID at the end of picking buffer
  const floatsPerPicking = pickingOffset + 1; // +1 for pickID

  // Build WebGPU buffer layouts. The geometry buffer is slot 0 (handled
  // separately) and the instance buffer is slot 1; instance shader locations
  // start after the geometry attributes (2 by default: position + normal).
  const renderLayout = buildBufferLayout(
    attrList,
    instanceStartLocation,
    "instance",
  );
  const pickingLayout = buildPickingBufferLayout(
    pickingAttrs,
    instanceStartLocation,
    "instance",
  );

  return {
    attributes: attrList,
    instanceStartLocation,
    renderAttributes: attrList,
    pickingAttributes: pickingAttrs,
    floatsPerInstance: renderOffset,
    floatsPerPicking,
    colorOffset,
    alphaOffset,
    renderLayout,
    pickingLayout,
  };
}

function buildBufferLayout(
  attrs: ProcessedAttr[],
  startLocation: number,
  stepMode: GPUVertexStepMode,
): VertexBufferLayout {
  const attributes: VertexBufferLayout["attributes"] = [];
  let byteOffset = 0;

  for (let i = 0; i < attrs.length; i++) {
    const attr = attrs[i];
    attributes.push({
      shaderLocation: startLocation + i,
      offset: byteOffset,
      format: ATTRIBUTE_FORMATS[attr.def.type],
    });
    byteOffset += ATTRIBUTE_SIZES[attr.def.type] * 4; // 4 bytes per float
  }

  return {
    arrayStride: byteOffset,
    stepMode,
    attributes,
  };
}

function buildPickingBufferLayout(
  attrs: ProcessedAttr[],
  startLocation: number,
  stepMode: GPUVertexStepMode,
): VertexBufferLayout {
  const attributes: VertexBufferLayout["attributes"] = [];
  let byteOffset = 0;

  for (let i = 0; i < attrs.length; i++) {
    const attr = attrs[i];
    attributes.push({
      shaderLocation: startLocation + i,
      offset: byteOffset,
      format: ATTRIBUTE_FORMATS[attr.def.type],
    });
    byteOffset += ATTRIBUTE_SIZES[attr.def.type] * 4;
  }

  // Add pickID at the end
  attributes.push({
    shaderLocation: startLocation + attrs.length,
    offset: byteOffset,
    format: "float32",
  });
  byteOffset += 4;

  return {
    arrayStride: byteOffset,
    stepMode,
    attributes,
  };
}

// =============================================================================
// Shader Generation
// =============================================================================

// Import shader code templates for internal use in shader generation
import { quaternionShaderFunctions } from "../quaternion";
import {
  cameraStruct,
  groupTransformStruct,
  filterParamsStruct,
  clipPlanesStruct,
  applyGroupTransformFn,
  lightingConstants,
  lightingCalc,
  pickingFragCode,
} from "../shaders";

/**
 * WGSL `@location(...) name: type` declarations for a primitive's instance
 * attributes, derived from the processed schema — including the auto-injected
 * `transformIndex` / `filterValue` / `filterIndex` and, for the picking pass,
 * the trailing `pickID`.
 *
 * This is the single source of truth for instance shader locations. Both the
 * generated vertex shaders and the hand-written ones call it, so a schema
 * change (e.g. a new framework attribute) cannot silently desynchronise a
 * hand-written shader from the buffer layout that feeds it.
 *
 * Hand-written shaders must therefore name their instance inputs after the
 * schema attribute names.
 *
 * @param schema Processed schema (from `processSchema`).
 * @param forPicking Emit the picking attribute set (no color/alpha, + pickID).
 */
export function instanceInputsWGSL(
  schema: ProcessedSchema,
  forPicking: boolean,
): string[] {
  const attrs = forPicking ? schema.pickingAttributes : schema.renderAttributes;
  const start = schema.instanceStartLocation;
  const inputs = attrs.map(
    (attr, i) =>
      `@location(${start + i}) ${attr.name}: ${ATTRIBUTE_WGSL_TYPES[attr.def.type]}`,
  );
  if (forPicking) {
    inputs.push(`@location(${start + attrs.length}) pickID: f32`);
  }
  return inputs;
}

/**
 * Per-instance filter collapse, shared by the generated and hand-written
 * vertex shaders.
 *
 * `test` computes `_filterPass` from the instance's `filterValue` against its
 * component's thresholds (looked up in the `filterParams` storage buffer by
 * `filterIndex`). `collapse` then sends a failing instance to a degenerate
 * clip position, which the rasterizer discards.
 *
 * Instance data is untouched when only the thresholds change, so a `$state`
 * threshold sweep rewrites just the tiny filterParams buffer. Apply BOTH parts
 * in the render and picking passes so filtered-out instances are also
 * unpickable.
 *
 * Requires `filterParamsStruct` in the shader preamble, and the shader's
 * instance inputs to include `filterValue` / `filterIndex` (which
 * `instanceInputsWGSL` emits).
 */
export const filterCollapseWGSL = {
  /** Place at the top of the vertex function body; declares `_filterPass`. */
  test: /*wgsl*/ `
  // Per-instance filter collapse
  let _fp = filterParams[u32(filterIndex)];
  var _filterPass = true;
  if (_fp.isActive > 0.5) {
    let _fv = filterValue;
    _filterPass = (_fv == _fv) && (_fv >= _fp.minVal) && (_fv <= _fp.maxVal);
  }`,
  // A zero clip position (w=0, all components 0) is degenerate: the
  // perspective divide discards it and it produces no fragments. Using a
  // literal 0/0 division instead would be a WGSL const-eval error, so we
  // assign the zero vector directly.
  /** Place after `out.position` is assigned, before `return out`. */
  collapse: /*wgsl*/ `
  if (!_filterPass) {
    out.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }`,
} as const;

/**
 * Group-transform composition for the "rigid with per-instance rotation" case:
 * the instance's local frame is rotated by `groupQuat * instanceQuat`, scaled
 * by the group's scale, and its origin is carried through the group's full
 * TRS. Every primitive that places oriented geometry at an instance centre
 * composes the group this way; keeping it in one WGSL function means the
 * generated path and the hand-written shaders cannot disagree about it.
 *
 * Normals are rotated by the SAME composed quaternion (see
 * `rigidGroupNormal`), matching `applyGroupTransformNormal`.
 *
 * Requires `groupTransformStruct`, `quaternionShaderFunctions` and
 * `applyGroupTransformFn` in the shader preamble.
 */
export const rigidGroupTransformFn = /*wgsl*/ `
struct RigidGroupFrame {
  // groupQuat * instanceQuat — rotates local geometry AND normals.
  quat: vec4<f32>,
  // Instance size premultiplied by the group's (possibly non-uniform) scale.
  size: vec3<f32>,
  // Instance origin carried through the group's full TRS.
  origin: vec3<f32>,
};

fn rigidGroupFrame(
  center: vec3<f32>,
  size: vec3<f32>,
  instanceQuat: vec4<f32>,
  groupIdx: u32
) -> RigidGroupFrame {
  let groupT = transforms[groupIdx];
  var f: RigidGroupFrame;
  f.quat = quat_mul(groupT.quaternion, instanceQuat);
  f.size = size * groupT.scale;
  f.origin = applyGroupTransform(center, groupIdx);
  return f;
}

// World-space position of a local-space point in a rigid group frame.
fn rigidGroupPosition(f: RigidGroupFrame, localPos: vec3<f32>) -> vec3<f32> {
  return f.origin + quat_rotate(f.quat, localPos * f.size);
}

// World-space normal, inverse-scaled then rotated by the composed quaternion.
fn rigidGroupNormal(f: RigidGroupFrame, localNormal: vec3<f32>) -> vec3<f32> {
  return quat_rotate(f.quat, normalize(localNormal / f.size));
}`;

/**
 * Per-vertex WEIGHTED TRANSFORM REFERENCES (D2).
 *
 * The rigid case above places a whole component with ONE palette entry. This
 * is the same palette indexed per VERTEX with weights: a vertex is positioned
 * by a weighted combination of up to K palette entries rather than rigidly by
 * one. That is what makes continuous deformation driven by a coarse control
 * structure expressible — a bending tube, a fault-dragged surface, a linkage.
 *
 * SEMANTICS — read this before using it. When a mesh declares transform
 * references, the blend REPLACES the component's own group transform entirely.
 * The referenced palette entries are already composed through Group nesting at
 * flatten time, so a mesh may sit anywhere in the hierarchy but only the
 * entries it references move its vertices. There is no double application.
 *
 * Position is the standard linear blend of the referenced TRS transforms.
 * Normals are rotated by each referenced quaternion, blended with the same
 * weights, then renormalized — normalized-lerp rather than slerp, whose
 * rotation error is ~0.01° at the sample spacing control structures use.
 *
 * Requires `groupTransformStruct`, `quaternionShaderFunctions` and the
 * `vertexTransformRefs` storage buffer (see `blendBufferBindingWGSL`) in the
 * shader preamble.
 */
export const blendedGroupTransformFn = /*wgsl*/ `
struct BlendedFrame {
  // World-space position of the vertex under the weighted blend.
  position: vec3<f32>,
  // Blended, renormalized rotation for the vertex's normal.
  quat: vec4<f32>,
};

fn blendedFrame(localPos: vec3<f32>, vertexIndex: u32) -> BlendedFrame {
  var pos = vec3<f32>(0.0, 0.0, 0.0);
  var q = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  // The first reference's quaternion fixes the hemisphere: quaternions q and
  // -q are the same rotation, so blending them raw can cancel. Every later
  // term is flipped onto the same side before it is added.
  var ref0 = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  let base = vertexIndex * TRANSFORM_REF_COUNT;
  for (var k = 0u; k < TRANSFORM_REF_COUNT; k = k + 1u) {
    let e = vertexTransformRefs[base + k];
    let w = e.weight;
    let t = transforms[u32(e.index)];
    pos = pos + w * (quat_rotate(t.quaternion, localPos * t.scale) + t.position);
    var tq = t.quaternion;
    if (k == 0u) {
      ref0 = tq;
    } else if (dot(tq, ref0) < 0.0) {
      tq = -tq;
    }
    q = q + w * tq;
  }
  var out: BlendedFrame;
  out.position = pos;
  let qlen = length(q);
  if (qlen < 1e-8) {
    out.quat = vec4<f32>(0.0, 0.0, 0.0, 1.0);
  } else {
    out.quat = q / qlen;
  }
  return out;
}

fn blendedNormal(f: BlendedFrame, localNormal: vec3<f32>) -> vec3<f32> {
  return normalize(quat_rotate(f.quat, localNormal));
}`;

/**
 * WGSL declarations for the per-vertex transform-reference storage buffer.
 *
 * The references live in a storage buffer indexed by `vertex_index` (vertex
 * pulling), NOT in the interleaved vertex buffer. That keeps the vertex buffer
 * untouched when only weights change, avoids a vertex-format variant per K,
 * and leaves variable-K per vertex reachable later.
 *
 * `index` is an ABSOLUTE palette index, resolved from the mesh's
 * `transform_refs` names at compile time. `weight` is that reference's share
 * of the vertex; weights are normalized to sum to 1 before they ship.
 *
 * @param k Fixed number of references per vertex (1..8).
 * @param group Bind group the buffer occupies.
 * @param binding Binding number within that group.
 */
export function blendBufferBindingWGSL(
  k: number,
  group: number,
  binding: number,
): string {
  return /*wgsl*/ `
const TRANSFORM_REF_COUNT: u32 = ${k}u;

struct VertexTransformRef {
  index: f32,
  weight: f32,
};

@group(${group}) @binding(${binding}) var<storage, read> vertexTransformRefs: array<VertexTransformRef>;
`;
}

function generateVertexShader(
  def: PrimitiveDefinition<any>,
  schema: ProcessedSchema,
  forPicking: boolean,
): string {
  const attrs = forPicking ? schema.pickingAttributes : schema.renderAttributes;

  // Build input parameters
  const inputs: string[] = [
    "@location(0) localPos: vec3<f32>",
    "@location(1) normal: vec3<f32>",
    ...instanceInputsWGSL(schema, forPicking),
  ];

  // Build VSOut struct. The picking VSOut carries worldPos (location 1) so the
  // picking fragment shader can apply scene clip planes (discard behind a
  // section), keeping pick-at / pick-where truthful for sectioned views.
  const vsOut = forPicking
    ? `struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) pickID: f32,
  @location(1) worldPos: vec3<f32>
};`
    : `struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
};`;

  // Generate transform code based on transform type
  const transformCode = generateTransformCode(def.transform, attrs);

  const filterCollapseCode = filterCollapseWGSL.test;
  const filterCollapsePosition = filterCollapseWGSL.collapse;

  // Build return statement
  const returnStmt = forPicking
    ? `var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  out.worldPos = worldPos;${filterCollapsePosition}
  return out;`
    : `var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  out.normal = worldNormal;${filterCollapsePosition}
  return out;`;

  // Include quaternion functions - always needed for group transforms
  const quaternionCode = quaternionShaderFunctions;

  return `${cameraStruct}
${groupTransformStruct}
${filterParamsStruct}
${vsOut}
${quaternionCode}
${applyGroupTransformFn}
${rigidGroupTransformFn}

@vertex
fn vs_main(
  ${inputs.join(",\n  ")}
) -> VSOut {
${filterCollapseCode}
${transformCode}
  ${returnStmt}
}`;
}

function generateTransformCode(
  transform: TransformType,
  attrs: Array<{ name: string; def: AttributeDef }>,
): string {
  const hasRotation = attrs.some((a) => a.name === "rotation");

  // Group transform application - applied after instance transform
  // Uses transformIndex to look up the group's world transform
  const groupTransformCode = `
  // Apply group transform
  let groupIdx = u32(transformIndex);
  let groupT = transforms[groupIdx];
  let groupScale = (groupT.scale.x + groupT.scale.y + groupT.scale.z) / 3.0;`;

  const applyGroupTransform = `
  // Apply group transform to position and normal
  let worldPos = applyGroupTransform(instancePos, groupIdx);
  let worldNormal = applyGroupTransformNormal(instanceNormal, groupIdx);`;

  switch (transform) {
    case "billboard":
      return `  // Billboard transform - always face camera
  let right = camera.cameraRight;
  let up = camera.cameraUp;
  ${groupTransformCode}
  let effectiveSize = size * groupScale;
  let scaledRight = right * (localPos.x * effectiveSize);
  let scaledUp = up * (localPos.y * effectiveSize);
  let instancePos = applyGroupTransform(position, groupIdx) + scaledRight + scaledUp;
  let worldPos = instancePos;
  let worldNormal = normalize(camera.cameraPos - worldPos);`;

    case "rigid": {
      // Both variants go through the shared rigidGroupFrame helper — the
      // no-rotation case is the identity-quaternion instance of the same
      // composition. Hand-written shaders call the same helper.
      const instanceQuat = hasRotation
        ? "rotation"
        : "vec4<f32>(0.0, 0.0, 0.0, 1.0)";
      return `  // Rigid transform${hasRotation ? " with rotation" : ""}
  ${groupTransformCode}
  let _frame = rigidGroupFrame(position, size, ${instanceQuat}, groupIdx);
  let effectiveSize = _frame.size;
  let composedQuat = _frame.quat;
  let worldPos = rigidGroupPosition(_frame, localPos);
  let worldNormal = rigidGroupNormal(_frame, normal);`;
    }

    case "beam":
      return `  // Beam transform - orient along start->end
  ${groupTransformCode}
  let effectiveSize = size * groupScale;

  // Transform start/end by group transform
  let worldStart = applyGroupTransform(start, groupIdx);
  let worldEnd = applyGroupTransform(end, groupIdx);

  let segDir = worldEnd - worldStart;
  let segLength = max(length(segDir), 0.000001);
  let zDir = normalize(segDir);

  // Build orthonormal basis
  var tempUp = vec3<f32>(0.0, 0.0, 1.0);
  if (abs(dot(zDir, tempUp)) > 0.99) {
    tempUp = vec3<f32>(0.0, 1.0, 0.0);
  }
  let xDir = normalize(cross(zDir, tempUp));
  let yDir = cross(zDir, xDir);

  // Transform to world space
  let localX = localPos.x * effectiveSize;
  let localY = localPos.y * effectiveSize;
  let localZ = localPos.z * segLength;
  let worldPos = worldStart + xDir * localX + yDir * localY + zDir * localZ;

  // Transform normal
  let worldNormal = normalize(xDir * normal.x + yDir * normal.y + zDir * normal.z);`;

    case "screenspace_offset":
      return `  // Screenspace offset transform - fixed size in screenspace
  ${groupTransformCode}

  // Transform anchor by group transform
  let worldAnchor = applyGroupTransform(anchor, groupIdx);

  // Projects anchor to clip space to get w (distance) for scaling
  let clipAnchor = camera.mvp * vec4<f32>(worldAnchor, 1.0);
  // Scaling factor to maintain constant size. 0.001 is a baseline adjustment.
  let screenspaceScale = clipAnchor.w * 0.001 * size;

  let scaledOffset = offset * screenspaceScale;
  let scaledLocal = localPos * screenspaceScale;

  ${
    hasRotation
      ? `let composedQuat = quat_mul(groupT.quaternion, rotation);
  let rotatedPos = quat_rotate(composedQuat, scaledOffset + scaledLocal);
  let worldPos = worldAnchor + rotatedPos;
  let worldNormal = quat_rotate(composedQuat, normal);`
      : `let rotatedPos = quat_rotate(groupT.quaternion, scaledOffset + scaledLocal);
  let worldPos = worldAnchor + rotatedPos;
  let worldNormal = quat_rotate(groupT.quaternion, normal);`
  }`;

    default:
      throw new Error(`Unknown transform type: ${transform}`);
  }
}

function generateFragmentShader(
  def: PrimitiveDefinition<any>,
  forPicking: boolean,
): string {
  if (forPicking) {
    return pickingFragCode;
  }

  const isLit = def.shading !== "unlit";

  if (isLit) {
    return `${cameraStruct}
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
  } else {
    return `${clipPlanesStruct}

@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
) -> @location(0) vec4<f32> {
  applyClipPlanes(worldPos);
  return vec4<f32>(color, alpha);
}`;
  }
}

// =============================================================================
// Geometry Creation
// =============================================================================

import {
  createSphereGeometry,
  createCubeGeometry,
  createBeamGeometry,
  createCylinderBeamGeometry,
} from "../geometry";

function createGeometry(source: GeometrySource): GeometryData {
  switch (source.type) {
    case "quad":
      return {
        vertexData: new Float32Array([
          // Position (x,y,z) and Normal (nx,ny,nz) for each vertex
          -0.5,
          -0.5,
          0.0,
          0.0,
          0.0,
          1.0, // Bottom-left
          0.5,
          -0.5,
          0.0,
          0.0,
          0.0,
          1.0, // Bottom-right
          -0.5,
          0.5,
          0.0,
          0.0,
          0.0,
          1.0, // Top-left
          0.5,
          0.5,
          0.0,
          0.0,
          0.0,
          1.0, // Top-right
        ]),
        indexData: new Uint16Array([0, 1, 2, 2, 1, 3]),
      };

    case "sphere":
      return createSphereGeometry(source.stacks ?? 32, source.slices ?? 48);

    case "cube":
      return createCubeGeometry();

    case "beam":
      return createBeamGeometry();

    case "cylinderBeam":
      return createCylinderBeamGeometry(source.segments ?? 12);

    case "custom":
      return source.create();

    default:
      throw new Error(`Unknown geometry type: ${(source as any).type}`);
  }
}

// =============================================================================
// Data Filling
// =============================================================================

/** Geometry buffer layout - shared across all primitives */
const GEOMETRY_LAYOUT: VertexBufferLayout = {
  arrayStride: 24, // 6 floats * 4 bytes
  stepMode: "vertex",
  attributes: [
    { shaderLocation: 0, offset: 0, format: "float32x3" }, // position
    { shaderLocation: 1, offset: 12, format: "float32x3" }, // normal
  ],
};

/**
 * Returns source index mapping for quaternion swizzle.
 * Output is always xyzw (shader convention).
 * For wxyz input: output[0]=x comes from input[1], etc.
 * For xyzw input: direct copy.
 */
function getQuatSwizzle(order: QuaternionOrder | undefined): number[] {
  if (order === "wxyz") {
    // wxyz input [w,x,y,z] -> xyzw output [x,y,z,w]
    // output[0] = input[1], output[1] = input[2], output[2] = input[3], output[3] = input[0]
    return [1, 2, 3, 0];
  }
  // xyzw: direct copy
  return [0, 1, 2, 3];
}

/**
 * Generates optimized fill function code from schema.
 * The generated code has no loops - all attribute handling is inlined.
 */
function generateFillRenderCode(schema: ProcessedSchema): string {
  const { attributes, floatsPerInstance } = schema;
  const lines: string[] = [];

  lines.push(`var o = outIndex * ${floatsPerInstance};`);

  for (const attr of attributes) {
    const {
      name,
      offset,
      size,
      constKey,
      def: attrDef,
      quaternionOrder,
    } = attr;
    const source = attrDef.source;
    const defaultVal = attrDef.default;
    const isQuat = attrDef.type === "quat";
    const swizzle = isQuat ? getQuatSwizzle(quaternionOrder) : null;

    if (name === "color") {
      // Color uses elemIndex for lookup (colorIndex would need custom fill)
      lines.push(`if (constants.color) {`);
      for (let i = 0; i < size; i++) {
        lines.push(`  out[o + ${offset + i}] = constants.color[${i}];`);
      }
      lines.push(`} else if (elem.${source}) {`);
      for (let i = 0; i < size; i++) {
        lines.push(
          `  out[o + ${offset + i}] = elem.${source}[elemIndex * ${size} + ${i}];`,
        );
      }
      if (defaultVal !== undefined) {
        const defaults = defaultVal as number[];
        lines.push(`} else {`);
        for (let i = 0; i < size; i++) {
          lines.push(`  out[o + ${offset + i}] = ${defaults[i]};`);
        }
      }
      lines.push(`}`);
    } else if (name === "alpha") {
      // Alpha is scalar, uses elemIndex
      const def = defaultVal ?? 1.0;
      lines.push(
        `out[o + ${offset}] = constants.alpha !== undefined ? constants.alpha : (elem.${source} ? elem.${source}[elemIndex] : ${def});`,
      );
    } else if (size === 1) {
      // Scalar attribute - check if it's a per-element array or a single value
      const def = defaultVal ?? 0;
      lines.push(
        `out[o + ${offset}] = constants.${constKey} !== undefined ? constants.${constKey} : (typeof elem.${source} === 'number' ? elem.${source} : (elem.${source} ? elem.${source}[elemIndex] : ${def}));`,
      );
    } else if (isQuat && swizzle) {
      // Quaternion attribute with swizzle (wxyz input -> xyzw output)
      lines.push(`if (constants.${constKey}) {`);
      for (let i = 0; i < size; i++) {
        lines.push(
          `  out[o + ${offset + i}] = constants.${constKey}[${swizzle[i]}];`,
        );
      }
      lines.push(`} else if (elem.${source}) {`);
      for (let i = 0; i < size; i++) {
        lines.push(
          `  out[o + ${offset + i}] = elem.${source}[elemIndex * ${size} + ${swizzle[i]}];`,
        );
      }
      if (defaultVal !== undefined) {
        const defaults = defaultVal as number[];
        lines.push(`} else {`);
        for (let i = 0; i < size; i++) {
          // Default is also in input format, so swizzle it
          lines.push(`  out[o + ${offset + i}] = ${defaults[swizzle[i]]};`);
        }
      }
      lines.push(`}`);
    } else {
      // Vector attribute (vec3, vec4) - direct copy
      lines.push(`if (constants.${constKey}) {`);
      for (let i = 0; i < size; i++) {
        lines.push(`  out[o + ${offset + i}] = constants.${constKey}[${i}];`);
      }
      lines.push(`} else if (elem.${source}) {`);
      for (let i = 0; i < size; i++) {
        lines.push(
          `  out[o + ${offset + i}] = elem.${source}[elemIndex * ${size} + ${i}];`,
        );
      }
      if (defaultVal !== undefined) {
        const defaults = defaultVal as number[];
        lines.push(`} else {`);
        for (let i = 0; i < size; i++) {
          lines.push(`  out[o + ${offset + i}] = ${defaults[i]};`);
        }
      }
      lines.push(`}`);
    }
  }

  return lines.join("\n");
}

/** Creates the fillRenderGeometry function using runtime code generation */
function createFillRenderGeometry<Config extends BaseComponentConfig>(
  _def: PrimitiveDefinition<Config>,
  schema: ProcessedSchema,
): (
  constants: ElementConstants,
  elem: Config,
  elemIndex: number,
  out: Float32Array,
  outIndex: number,
) => void {
  const code = generateFillRenderCode(schema);
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  return new Function(
    "constants",
    "elem",
    "elemIndex",
    "out",
    "outIndex",
    code,
  ) as (
    constants: ElementConstants,
    elem: Config,
    elemIndex: number,
    out: Float32Array,
    outIndex: number,
  ) => void;
}

/**
 * Generates optimized picking fill function code from schema.
 * Similar to render but excludes color/alpha and adds pickID.
 */
function generateFillPickingCode(schema: ProcessedSchema): string {
  const { pickingAttributes, floatsPerPicking } = schema;
  const lines: string[] = [];

  lines.push(`var o = outIndex * ${floatsPerPicking};`);

  for (const attr of pickingAttributes) {
    const { offset, size, constKey, def: attrDef, quaternionOrder } = attr;
    const source = attrDef.source;
    const defaultVal = attrDef.default;
    const isQuat = attrDef.type === "quat";
    const swizzle = isQuat ? getQuatSwizzle(quaternionOrder) : null;

    if (size === 1) {
      // Scalar attribute - check if it's a per-element array or a single value
      const def = defaultVal ?? 0;
      lines.push(
        `out[o + ${offset}] = constants.${constKey} !== undefined ? constants.${constKey} : (typeof elem.${source} === 'number' ? elem.${source} : (elem.${source} ? elem.${source}[elemIndex] : ${def}));`,
      );
    } else if (isQuat && swizzle) {
      // Quaternion attribute with swizzle (wxyz input -> xyzw output)
      lines.push(`if (constants.${constKey}) {`);
      for (let i = 0; i < size; i++) {
        lines.push(
          `  out[o + ${offset + i}] = constants.${constKey}[${swizzle[i]}];`,
        );
      }
      lines.push(`} else if (elem.${source}) {`);
      for (let i = 0; i < size; i++) {
        lines.push(
          `  out[o + ${offset + i}] = elem.${source}[elemIndex * ${size} + ${swizzle[i]}];`,
        );
      }
      if (defaultVal !== undefined) {
        const defaults = defaultVal as number[];
        lines.push(`} else {`);
        for (let i = 0; i < size; i++) {
          lines.push(`  out[o + ${offset + i}] = ${defaults[swizzle[i]]};`);
        }
      }
      lines.push(`}`);
    } else {
      // Vector attribute (vec3, vec4) - direct copy
      lines.push(`if (constants.${constKey}) {`);
      for (let i = 0; i < size; i++) {
        lines.push(`  out[o + ${offset + i}] = constants.${constKey}[${i}];`);
      }
      lines.push(`} else if (elem.${source}) {`);
      for (let i = 0; i < size; i++) {
        lines.push(
          `  out[o + ${offset + i}] = elem.${source}[elemIndex * ${size} + ${i}];`,
        );
      }
      if (defaultVal !== undefined) {
        const defaults = defaultVal as number[];
        lines.push(`} else {`);
        for (let i = 0; i < size; i++) {
          lines.push(`  out[o + ${offset + i}] = ${defaults[i]};`);
        }
      }
      lines.push(`}`);
    }
  }

  // Add pickID at the end
  const pickIDOffset = floatsPerPicking - 1;
  lines.push(`out[o + ${pickIDOffset}] = packID(baseID + elemIndex);`);

  return lines.join("\n");
}

/** Creates the fillPickingGeometry function using runtime code generation */
function createFillPickingGeometry<Config extends BaseComponentConfig>(
  _def: PrimitiveDefinition<Config>,
  schema: ProcessedSchema,
): (
  constants: ElementConstants,
  elem: Config,
  elemIndex: number,
  out: Float32Array,
  outIndex: number,
  baseID: number,
) => void {
  const code = generateFillPickingCode(schema);
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  const generatedFn = new Function(
    "packID",
    "constants",
    "elem",
    "elemIndex",
    "out",
    "outIndex",
    "baseID",
    code,
  ) as (
    packID: (id: number) => number,
    constants: ElementConstants,
    elem: Config,
    elemIndex: number,
    out: Float32Array,
    outIndex: number,
    baseID: number,
  ) => void;

  // Return a wrapper that captures packID
  return (constants, elem, elemIndex, out, outIndex, baseID) =>
    generatedFn(packID, constants, elem, elemIndex, out, outIndex, baseID);
}

/** Creates applyDecorationScale based on which attributes affect scale */
function createApplyDecorationScale(
  schema: ProcessedSchema,
): (out: Float32Array, offset: number, scaleFactor: number) => void {
  // Find size attribute
  const sizeAttr = schema.attributes.find((a) => a.name === "size");

  if (!sizeAttr) {
    return () => {}; // No size attribute, nothing to scale
  }

  const sizeOffset = sizeAttr.offset;
  const sizeCount = ATTRIBUTE_SIZES[sizeAttr.def.type];

  return function applyDecorationScale(
    out: Float32Array,
    offset: number,
    scaleFactor: number,
  ): void {
    for (let i = 0; i < sizeCount; i++) {
      out[offset + sizeOffset + i] *= scaleFactor;
    }
  };
}

// =============================================================================
// Main Factory Function
// =============================================================================

/**
 * Creates a PrimitiveSpec from a declarative PrimitiveDefinition.
 * This is the main entry point for the new primitive system.
 */
export function definePrimitive<Config extends BaseComponentConfig>(
  def: PrimitiveDefinition<Config>,
): PrimitiveSpec<Config> {
  // Process the attribute schema. Primitives with hand-written shaders call
  // processSchema themselves with the same arguments to derive their shader
  // input declarations — same inputs, same schema, one seam.
  const schema = processSchema(def.attributes, def.instanceStartLocation ?? 2);

  // Generate or use custom shaders
  const renderVertexShader =
    def.vertexShader ?? generateVertexShader(def, schema, false);
  const renderFragmentShader =
    def.fragmentShader ?? generateFragmentShader(def, false);
  const pickingVertexShader =
    def.pickingVertexShader ?? generateVertexShader(def, schema, true);
  const pickingFragmentShader = generateFragmentShader(def, true);

  // Create data filling functions (use custom if provided)
  const autoFillRenderGeometry = createFillRenderGeometry(def, schema);
  const autoFillPickingGeometry = createFillPickingGeometry(def, schema);

  const fillRenderGeometry = def.fillRenderGeometry
    ? (
        constants: ElementConstants,
        elem: Config,
        elemIndex: number,
        out: Float32Array,
        outIndex: number,
      ) =>
        def.fillRenderGeometry!(
          schema,
          constants,
          elem,
          elemIndex,
          out,
          outIndex,
        )
    : autoFillRenderGeometry;

  const fillPickingGeometry = def.fillPickingGeometry
    ? (
        constants: ElementConstants,
        elem: Config,
        elemIndex: number,
        out: Float32Array,
        outIndex: number,
        baseID: number,
      ) =>
        def.fillPickingGeometry!(
          schema,
          constants,
          elem,
          elemIndex,
          out,
          outIndex,
          baseID,
        )
    : autoFillPickingGeometry;

  const applyDecorationScale = createApplyDecorationScale(schema);

  // Default getElementCount based on first attribute
  const firstAttr = schema.attributes[0];
  const firstAttrSize = ATTRIBUTE_SIZES[firstAttr.def.type];
  const defaultGetElementCount = (elem: Config): number => {
    const sourceArray = (elem as any)[firstAttr.def.source];
    return sourceArray ? sourceArray.length / firstAttrSize : 0;
  };

  // Default getCenters using first vec3 attribute
  const firstVec3Attr = schema.attributes.find((a) => a.def.type === "vec3");
  const defaultGetCenters = (elem: Config): Float32Array | number[] => {
    if (!firstVec3Attr) return new Float32Array(0);
    return (elem as any)[firstVec3Attr.def.source] || new Float32Array(0);
  };

  // Build defaults object from attribute schema for computeConstants compatibility
  // The old system uses singular key names (e.g., "size" not "sizes")
  const defaults: Record<string, unknown> = {};
  for (const attr of schema.attributes) {
    // Skip color and alpha - they're handled separately in computeConstants
    if (attr.name === "color" || attr.name === "alpha") continue;

    if (attr.def.default !== undefined) {
      // Map attribute name to the singular key expected by computeConstants
      // e.g., attr.name could be "size", "position", "rotation"
      // We need to figure out the singular form
      // The source is plural (e.g., "sizes", "half_sizes", "quaternions")
      // The default key should be singular (e.g., "size", "half_size", "quaternion")
      const source = attr.def.source;
      // Remove trailing 's' to get singular form (handles "sizes" -> "size", "quaternions" -> "quaternion")
      const singularKey = source.endsWith("s") ? source.slice(0, -1) : source;
      defaults[singularKey] = attr.def.default;
    }
  }

  // Complete set of accepted prop names: attribute sources (plural) and
  // their singular forms, shared base props, and primitive-specific extras.
  // The compiler warns about props outside this set instead of silently
  // dropping them.
  const knownProps = new Set<string>(BASE_COMPONENT_PROPS);
  for (const attr of schema.attributes) {
    knownProps.add(attr.def.source);
    knownProps.add(attr.constKey);
  }
  for (const extra of def.extraProps ?? []) {
    knownProps.add(extra);
  }

  const arrayFieldSet = new Set<string>();
  for (const attr of schema.attributes) {
    arrayFieldSet.add(attr.def.source);
  }

  const arrayFields =
    arrayFieldSet.size > 0
      ? {
          float32: Array.from(arrayFieldSet) as (keyof Config)[],
        }
      : undefined;

  const spec: PrimitiveSpec<Config> = {
    type: def.name,
    coerce: def.coerce,
    knownProps,
    instancesPerElement: def.instancesPerElement ?? 1,

    // Defaults for computeConstants compatibility
    defaults,
    arrayFields,

    getElementCount: def.getElementCount ?? defaultGetElementCount,
    getCenters: def.getCenters ?? defaultGetCenters,

    floatsPerInstance: schema.floatsPerInstance,
    floatsPerPicking: schema.floatsPerPicking,

    renderInstanceLayout: def.renderInstanceLayout ?? schema.renderLayout,
    pickingInstanceLayout: def.pickingInstanceLayout ?? schema.pickingLayout,

    colorOffset: schema.colorOffset,
    alphaOffset: schema.alphaOffset,

    fillRenderGeometry,
    fillPickingGeometry,
    applyDecorationScale,

    renderConfig: {
      cullMode: def.cullMode ?? "back",
      topology: def.topology ?? "triangle-list",
    },

    getRenderPipeline(
      device: GPUDevice,
      bindGroupLayout: GPUBindGroupLayout,
      cache: Map<string, PipelineCacheEntry>,
    ): GPURenderPipeline {
      const format = navigator.gpu.getPreferredCanvasFormat();
      const geometryLayout = def.geometryLayout ?? GEOMETRY_LAYOUT;
      const instanceLayout = spec.renderInstanceLayout;
      const vertexEntryPoint = def.vertexEntryPoint ?? "vs_main";
      const pipelineBindGroupLayout = def.bindGroupLayouts
        ? def.bindGroupLayouts(device, bindGroupLayout)
        : bindGroupLayout;
      return getOrCreatePipeline(
        device,
        `${def.name}Shading`,
        () =>
          createTranslucentGeometryPipeline(
            device,
            pipelineBindGroupLayout,
            {
              vertexShader: renderVertexShader,
              fragmentShader: renderFragmentShader,
              vertexEntryPoint,
              fragmentEntryPoint: "fs_main",
              bufferLayouts: [geometryLayout, instanceLayout],
            },
            format,
            spec,
          ),
        cache,
      );
    },

    getPickingPipeline(
      device: GPUDevice,
      bindGroupLayout: GPUBindGroupLayout,
      cache: Map<string, PipelineCacheEntry>,
    ): GPURenderPipeline {
      const geometryLayout = def.geometryLayout ?? GEOMETRY_LAYOUT;
      const instanceLayout = spec.pickingInstanceLayout;
      const vertexEntryPoint = def.pickingVertexEntryPoint ?? "vs_main";
      const pipelineBindGroupLayout = def.bindGroupLayouts
        ? def.bindGroupLayouts(device, bindGroupLayout)
        : bindGroupLayout;
      return getOrCreatePipeline(
        device,
        `${def.name}Picking`,
        () =>
          createRenderPipeline(
            device,
            pipelineBindGroupLayout,
            {
              vertexShader: pickingVertexShader,
              fragmentShader: pickingFragmentShader,
              vertexEntryPoint,
              fragmentEntryPoint: "fs_pick",
              bufferLayouts: [geometryLayout, instanceLayout],
              primitive: spec.renderConfig,
            },
            "rgba8unorm",
          ),
        cache,
      );
    },

    getOverlayPipeline(
      device: GPUDevice,
      bindGroupLayout: GPUBindGroupLayout,
      cache: Map<string, PipelineCacheEntry>,
    ): GPURenderPipeline {
      const format = navigator.gpu.getPreferredCanvasFormat();
      const geometryLayout = def.geometryLayout ?? GEOMETRY_LAYOUT;
      const instanceLayout = spec.renderInstanceLayout;
      const vertexEntryPoint = def.vertexEntryPoint ?? "vs_main";
      const pipelineBindGroupLayout = def.bindGroupLayouts
        ? def.bindGroupLayouts(device, bindGroupLayout)
        : bindGroupLayout;
      return getOrCreatePipeline(
        device,
        `${def.name}Overlay`,
        () =>
          createOverlayPipeline(
            device,
            pipelineBindGroupLayout,
            {
              vertexShader: renderVertexShader,
              fragmentShader: renderFragmentShader,
              vertexEntryPoint,
              fragmentEntryPoint: "fs_main",
              bufferLayouts: [geometryLayout, instanceLayout],
            },
            format,
            spec,
          ),
        cache,
      );
    },

    getOverlayPickingPipeline(
      device: GPUDevice,
      bindGroupLayout: GPUBindGroupLayout,
      cache: Map<string, PipelineCacheEntry>,
    ): GPURenderPipeline {
      const geometryLayout = def.geometryLayout ?? GEOMETRY_LAYOUT;
      const instanceLayout = spec.pickingInstanceLayout;
      const vertexEntryPoint = def.pickingVertexEntryPoint ?? "vs_main";
      const pipelineBindGroupLayout = def.bindGroupLayouts
        ? def.bindGroupLayouts(device, bindGroupLayout)
        : bindGroupLayout;
      return getOrCreatePipeline(
        device,
        `${def.name}OverlayPicking`,
        () =>
          createOverlayPickingPipeline(
            device,
            pipelineBindGroupLayout,
            {
              vertexShader: pickingVertexShader,
              fragmentShader: pickingFragmentShader,
              vertexEntryPoint,
              fragmentEntryPoint: "fs_pick",
              bufferLayouts: [geometryLayout, instanceLayout],
            },
            spec,
          ),
        cache,
      );
    },

    createGeometryResource(device: GPUDevice): GeometryResource {
      const layout = def.geometryLayout ?? GEOMETRY_LAYOUT;
      const vertexStrideFloats = layout.arrayStride / 4;
      return createBuffers(
        device,
        createGeometry(def.geometry),
        vertexStrideFloats,
      );
    },

    buildGeometryData(): GeometryData {
      return createGeometry(def.geometry);
    },

    geometryStrideFloats:
      (def.geometryLayout ?? GEOMETRY_LAYOUT).arrayStride / 4,
  };

  return spec;
}
