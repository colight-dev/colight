/**
 * @module inlineMesh
 * @description Handles inline mesh geometry conversion to primitive specs.
 *
 * Allows using <Mesh geometry={...}/> syntax where geometry is specified inline
 * rather than pre-registered as a primitive spec.
 */

import { ComponentConfig, PrimitiveSpec } from "./components";
import {
  StructuredGeometry,
  MeshComponentConfig,
  defineMesh,
  getFormatKey,
  VertexFormat,
} from "./primitives/mesh";
import { coerceToFloat32 } from "./coercion";

// =============================================================================
// Types
// =============================================================================

/**
 * Geometry input for inline mesh - accepts various array types.
 * Coerced to Float32Array internally.
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

export type MeshProps = Omit<MeshComponentConfig, "type" | "centers"> & {
  geometry: MeshGeometry;
  geometryKey?: string | number;
  shading?: "lit" | "unlit";
  cullMode?: GPUCullMode;
  centers?: ArrayLike<number> | ArrayBufferView;
  center?: [number, number, number];
};

export type InlineMeshComponentConfig = MeshProps & { type: "Mesh" };

// =============================================================================
// Geometry Coercion
// =============================================================================

/** Get length of an array-like value */
function getLength(value: Float32Array | number[] | ArrayBufferView): number {
  if (Array.isArray(value)) return value.length;
  if ("length" in value) return (value as Float32Array).length;
  // For generic ArrayBufferView, calculate from byteLength
  return value.byteLength / 4; // Assume float32
}

function coerceIndexData(
  value?: Uint16Array | Uint32Array | number[] | ArrayBufferView,
): Uint16Array | Uint32Array | undefined {
  if (!value) return undefined;
  if (value instanceof Uint16Array || value instanceof Uint32Array) {
    return value;
  }
  let arr: number[];
  if (Array.isArray(value)) {
    arr = value;
  } else if (
    value instanceof Int8Array ||
    value instanceof Int16Array ||
    value instanceof Int32Array ||
    value instanceof Uint8Array ||
    value instanceof Uint8ClampedArray ||
    value instanceof Float32Array ||
    value instanceof Float64Array
  ) {
    arr = Array.from(value);
  } else {
    // Generic ArrayBufferView - try to interpret as Uint32
    arr = Array.from(
      new Uint32Array(value.buffer, value.byteOffset, value.byteLength / 4),
    );
  }
  const max = Math.max(...arr);
  return max > 65535 ? new Uint32Array(arr) : new Uint16Array(arr);
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
 * Detect vertex format from geometry for cache key generation.
 */
function detectFormat(
  geometry: MeshGeometry,
  shading: "lit" | "unlit",
): VertexFormat {
  const hasColors =
    geometry.colors !== undefined && getLength(geometry.colors) > 0;

  let colorComponents: 3 | 4 = 3;
  if (hasColors) {
    const posLen = getLength(geometry.positions);
    const colorLen = getLength(geometry.colors!);
    colorComponents = colorLen / (posLen / 3) === 4 ? 4 : 3;
  }

  const hasExplicitNormals =
    geometry.normals !== undefined && getLength(geometry.normals) > 0;

  return {
    // Lit shading will auto-compute normals, so hasNormals is true for lit
    hasNormals: shading === "lit" || hasExplicitNormals,
    hasColors,
    colorComponents,
    hasUVs: geometry.uvs !== undefined && getLength(geometry.uvs) > 0,
  };
}

// =============================================================================
// Inline Mesh Cache
// =============================================================================

interface InlineMeshCacheEntry {
  typeName: string;
  spec: PrimitiveSpec<any>;
  shading: "lit" | "unlit";
  cullMode: GPUCullMode;
  hasTexture: boolean;
}

// Use WeakMap keyed by geometry object - entries are garbage collected when geometry is no longer referenced
const inlineMeshCache = new WeakMap<
  MeshGeometry,
  Map<string, InlineMeshCacheEntry>
>();
let inlineMeshId = 0;

function getInlineMeshEntry(
  geometry: MeshGeometry,
  options: {
    geometryKey?: string | number;
    shading?: "lit" | "unlit";
    cullMode?: GPUCullMode;
    hasTexture?: boolean;
  },
): InlineMeshCacheEntry {
  const shading = options.shading ?? "lit";
  const cullMode = options.cullMode ?? "back";
  const hasTexture = options.hasTexture ?? false;

  // Include format in cache key to handle different vertex layouts
  const format = detectFormat(geometry, shading);
  const variantKey = `${getFormatKey(format, shading, hasTexture)}|${cullMode}`;

  let variants = inlineMeshCache.get(geometry);
  if (!variants) {
    variants = new Map();
    inlineMeshCache.set(geometry, variants);
  }

  let entry = variants.get(variantKey);
  if (entry) {
    return entry;
  }

  // Create new entry with structured geometry
  const normalizedGeometry = coerceMeshGeometry(geometry);
  const typeName = `__InlineMesh_${inlineMeshId++}`;
  const spec = defineMesh(typeName, normalizedGeometry, {
    shading,
    cullMode,
    hasTexture,
  });

  // Store the explicit key if provided, for impl3d to detect data changes
  if (options.geometryKey !== undefined) {
    (spec as any).geometryKey = options.geometryKey;
  }

  entry = {
    typeName,
    spec,
    shading,
    cullMode,
    hasTexture,
  };
  variants.set(variantKey, entry);
  return entry;
}

// =============================================================================
// Resolution
// =============================================================================

/**
 * Resolves inline mesh components to registered primitive specs.
 * Returns both the resolved components and any inline specs that need to be registered.
 */
export function resolveInlineMeshes(
  components: (ComponentConfig | InlineMeshComponentConfig)[],
): {
  components: ComponentConfig[];
  inlineSpecs?: Record<string, PrimitiveSpec<any>>;
} {
  let inlineSpecs: Record<string, PrimitiveSpec<any>> | undefined;
  const resolved = components.map((component) => {
    if (component.type !== "Mesh") return component;
    const meshComponent = component as InlineMeshComponentConfig;
    // Detect if mesh uses a texture (presence of texture prop)
    const hasTexture = meshComponent.texture !== undefined;
    const entry = getInlineMeshEntry(meshComponent.geometry, {
      geometryKey: meshComponent.geometryKey,
      shading: meshComponent.shading,
      cullMode: meshComponent.cullMode,
      hasTexture,
    });

    if (!inlineSpecs) inlineSpecs = {};
    inlineSpecs[entry.typeName] = entry.spec;

    const { geometry, geometryKey, shading, cullMode, ...rest } = meshComponent;
    return {
      ...rest,
      type: entry.typeName,
    } as ComponentConfig;
  });

  return { components: resolved, inlineSpecs };
}
