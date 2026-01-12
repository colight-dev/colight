/**
 * @module coercion
 * @description Type coercion utilities for scene3d data.
 *
 * Handles conversion of various array-like types (NdArray, regular arrays,
 * TypedArrays) to the formats expected by the rendering pipeline.
 */

import { isNdArray } from "@colight/serde";
import { PrimitiveSpec } from "./components";
import { defineMesh, StructuredGeometry } from "./primitives/mesh";

// =============================================================================
// Float32 Coercion
// =============================================================================

/**
 * Coerce a value to Float32Array if it's an array-like type.
 * Handles NdArrayView, regular arrays, and other TypedArrays.
 */
export function coerceToFloat32(value: unknown): Float32Array | unknown {
  if (isNdArray(value)) {
    const flat = value.flat;
    return flat instanceof Float32Array
      ? flat
      : new Float32Array(flat as ArrayLike<number>);
  }
  if (Array.isArray(value)) {
    const flattened = value.flat ? value.flat() : value;
    return new Float32Array(flattened as number[]);
  }
  if (ArrayBuffer.isView(value) && !(value instanceof Float32Array)) {
    return new Float32Array(value.buffer);
  }
  return value;
}

/**
 * Coerce array fields on a component based on its spec's arrayFields.
 * Mutates the component in place for efficiency.
 */
export function coerceComponentArrays<T extends { type: string }>(
  component: T,
  registry: Record<string, PrimitiveSpec<any>>,
): T {
  const spec = registry[component.type];
  const float32Fields = spec?.arrayFields?.float32;
  if (!float32Fields) return component;

  for (const field of float32Fields) {
    const value = (component as any)[field];
    if (value !== undefined) {
      (component as any)[field] = coerceToFloat32(value);
    }
  }

  return component;
}

// =============================================================================
// Vertex/Index Data Coercion
// =============================================================================

/**
 * Coerce vertex data to Float32Array.
 */
export function coerceVertexData(
  value: Float32Array | number[] | ArrayBufferView,
): Float32Array {
  if (isNdArray(value)) {
    const flat = value.flat;
    return flat instanceof Float32Array
      ? flat
      : new Float32Array(flat as ArrayLike<number>);
  }
  if (value instanceof Float32Array) return value;
  if (Array.isArray(value)) return new Float32Array(value);
  const asArray = Array.from(value as ArrayLike<number>);
  return new Float32Array(asArray);
}

/**
 * Coerce index data to Uint16Array or Uint32Array based on max index value.
 */
export function coerceIndexData(
  value?: Uint16Array | Uint32Array | number[] | ArrayBufferView,
): Uint16Array | Uint32Array | undefined {
  if (!value) return undefined;
  if (isNdArray(value)) {
    const flat = value.flat;
    value = Array.from(flat as ArrayLike<number>);
  }
  if (value instanceof Uint16Array || value instanceof Uint32Array) {
    return value;
  }
  if (Array.isArray(value)) {
    let max = 0;
    for (const idx of value) {
      if (idx > max) max = idx;
    }
    return max > 65535 ? new Uint32Array(value) : new Uint16Array(value);
  }
  const asArray = Array.from(value as ArrayLike<number>);
  let max = 0;
  for (const idx of asArray) {
    if (idx > max) max = idx;
  }
  return max > 65535 ? new Uint32Array(asArray) : new Uint16Array(asArray);
}

// =============================================================================
// Primitive Spec Normalization
// =============================================================================

/**
 * Mesh geometry input with structured attribute arrays.
 */
export interface MeshGeometry {
  /** Vertex positions (N, 3) - required */
  positions: Float32Array | number[] | ArrayBufferView;
  /** Vertex normals (N, 3) - optional, auto-computed if missing for lit shading */
  normals?: Float32Array | number[] | ArrayBufferView;
  /** Per-vertex colors (N, 3) RGB or (N, 4) RGBA - optional */
  colors?: Float32Array | number[] | ArrayBufferView;
  /** Texture coordinates (N, 2) - optional */
  uvs?: Float32Array | number[] | ArrayBufferView;
  /** Triangle indices - optional */
  indices?: Uint16Array | Uint32Array | number[] | ArrayBufferView;
}

export interface MeshDefinition extends MeshGeometry {
  shading?: "lit" | "unlit";
  cullMode?: GPUCullMode;
}

export type PrimitiveSpecInput = PrimitiveSpec<any> | MeshDefinition;
export type PrimitiveSpecMap = Record<string, PrimitiveSpecInput>;

export function isMeshDefinition(value: PrimitiveSpecInput): value is MeshDefinition {
  return (
    typeof value === "object" &&
    value !== null &&
    "positions" in value
  );
}

export function isPrimitiveSpec(value: unknown): value is PrimitiveSpec<any> {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as PrimitiveSpec<any>).createGeometryResource === "function"
  );
}

function coerceMeshGeometry(geometry: MeshGeometry): StructuredGeometry {
  return {
    positions: coerceToFloat32(geometry.positions) as Float32Array,
    normals: geometry.normals
      ? (coerceToFloat32(geometry.normals) as Float32Array)
      : undefined,
    colors: geometry.colors
      ? (coerceToFloat32(geometry.colors) as Float32Array)
      : undefined,
    uvs: geometry.uvs
      ? (coerceToFloat32(geometry.uvs) as Float32Array)
      : undefined,
    indices: coerceIndexData(geometry.indices),
  };
}

/**
 * Normalize primitive specs, converting MeshDefinitions to PrimitiveSpecs.
 */
export function normalizePrimitiveSpecs(
  specs?: PrimitiveSpecMap,
): Record<string, PrimitiveSpec<any>> | undefined {
  if (!specs) return undefined;
  const normalized: Record<string, PrimitiveSpec<any>> = {};
  for (const [name, spec] of Object.entries(specs)) {
    if (isMeshDefinition(spec)) {
      const geometry = coerceMeshGeometry(spec);
      normalized[name] = defineMesh(name, geometry, {
        shading: spec.shading,
        cullMode: spec.cullMode,
      });
    } else if (isPrimitiveSpec(spec)) {
      normalized[name] = spec;
    } else if (typeof spec === "object" && spec !== null) {
      const nestedEntries = Object.entries(spec as Record<string, PrimitiveSpecInput>);
      const hasNested = nestedEntries.some(
        ([, nested]) => isMeshDefinition(nested) || isPrimitiveSpec(nested),
      );
      if (hasNested) {
        console.warn("scene3d: flattening nested primitiveSpecs entry", name);
        for (const [nestedName, nestedSpec] of nestedEntries) {
          if (isMeshDefinition(nestedSpec)) {
            const geometry = coerceMeshGeometry(nestedSpec);
            normalized[nestedName] = defineMesh(nestedName, geometry, {
              shading: nestedSpec.shading,
              cullMode: nestedSpec.cullMode,
            });
          } else if (isPrimitiveSpec(nestedSpec)) {
            normalized[nestedName] = nestedSpec;
          }
        }
      } else {
        console.warn("scene3d: ignoring invalid primitiveSpecs entry", name, spec);
      }
    } else {
      console.warn("scene3d: ignoring invalid primitiveSpecs entry", name, spec);
    }
  }
  return normalized;
}
