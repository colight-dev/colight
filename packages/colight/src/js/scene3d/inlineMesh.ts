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
  MAX_TRANSFORM_REFS_PER_VERTEX,
  FLOATS_PER_TRANSFORM_REF,
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
  /**
   * Names of the Groups whose composed world transforms this mesh's vertices
   * reference. Slot s of `transformIndices` means `transformRefs[s]`. All
   * three transform* fields are present together or not at all.
   */
  transformRefs?: string[];
  /** (N, K) local slots into `transformRefs`, row-major, flattened. */
  transformIndices?: Float32Array | number[] | ArrayBufferView;
  /** (N, K) weights matching `transformIndices`, row-major, flattened. */
  transformWeights?: Float32Array | number[] | ArrayBufferView;
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

// =============================================================================
// Per-vertex weighted transform references (D2)
// =============================================================================

/**
 * A mesh's per-vertex transform references, resolved for the GPU.
 *
 * `data` is the storage buffer's contents: K consecutive `[paletteIndex,
 * weight]` pairs per vertex, with paletteIndex ABSOLUTE (already resolved from
 * the mesh's `transformRefs` names through the scene's named-transform map).
 * Resolving at compile time is what keeps the shader a pure array read.
 */
export interface ResolvedTransformRefs {
  /** References per vertex (K), 1..8. */
  count: number;
  /** Flat [index, weight] pairs, length = vertexCount * K * 2. */
  data: Float32Array;
}

/**
 * Resolve a mesh's `transformRefs` / `transformIndices` / `transformWeights`
 * into the flat storage-buffer contents the blended shader reads.
 *
 * Name resolution is LOUD: a reference to a Group name the scene does not
 * define is a scene-authoring error that would otherwise silently read palette
 * entry 0 (identity) and render a rest pose. The compile pass is where the
 * flattened palette exists, so it is where the error can name both the missing
 * reference and the names that do exist.
 *
 * @param geometry Mesh geometry carrying the three transform_* fields.
 * @param namedTransforms Name → palette index, from `flattenGroups`.
 * @returns Resolved references, or undefined when the mesh declares none.
 */
export function resolveTransformRefs(
  geometry: MeshGeometry,
  namedTransforms: Map<string, number> | undefined,
): ResolvedTransformRefs | undefined {
  const { transformRefs, transformIndices, transformWeights } = geometry;
  const present = [
    transformRefs !== undefined,
    transformIndices !== undefined,
    transformWeights !== undefined,
  ];
  if (!present.some(Boolean)) return undefined;
  if (!present.every(Boolean)) {
    throw new Error(
      "scene3d: Mesh transform_refs, transform_indices and transform_weights " +
        "must be supplied together (per-vertex weighted transform references). " +
        `Got refs=${transformRefs !== undefined}, indices=${
          transformIndices !== undefined
        }, weights=${transformWeights !== undefined}.`,
    );
  }

  const refs = transformRefs!;
  if (refs.length === 0) {
    throw new Error("scene3d: Mesh transform_refs must not be empty");
  }

  const indices = coerceToFloat32(transformIndices!) as Float32Array;
  const weights = coerceToFloat32(transformWeights!) as Float32Array;
  if (indices.length !== weights.length) {
    throw new Error(
      `scene3d: Mesh transform_indices (${indices.length} values) and ` +
        `transform_weights (${weights.length} values) must have the same shape`,
    );
  }

  const vertexCount = getLength(geometry.positions) / 3;
  if (vertexCount === 0 || indices.length % vertexCount !== 0) {
    throw new Error(
      `scene3d: Mesh transform_indices holds ${indices.length} values, which ` +
        `is not a whole number of references per vertex for ${vertexCount} vertices`,
    );
  }
  const count = indices.length / vertexCount;
  if (count < 1 || count > MAX_TRANSFORM_REFS_PER_VERTEX) {
    throw new Error(
      `scene3d: Mesh allows 1..${MAX_TRANSFORM_REFS_PER_VERTEX} transform ` +
        `references per vertex, got ${count}`,
    );
  }

  // Resolve each declared name once, so a bad name is reported before the
  // per-vertex loop and reported by name rather than by slot.
  const resolvedRefs = refs.map((name) => {
    const palette = namedTransforms?.get(name);
    if (palette === undefined) {
      const known = namedTransforms ? [...namedTransforms.keys()] : [];
      throw new Error(
        `scene3d: Mesh transform_refs names Group "${name}", which this scene ` +
          `does not define. Named groups in this scene: ` +
          `${known.length ? known.map((k) => `"${k}"`).join(", ") : "(none)"}.`,
      );
    }
    return palette;
  });

  const data = new Float32Array(indices.length * FLOATS_PER_TRANSFORM_REF);
  for (let i = 0; i < indices.length; i++) {
    const slot = indices[i];
    if (!(slot >= 0 && slot < resolvedRefs.length)) {
      throw new Error(
        `scene3d: Mesh transform_indices[${i}] = ${slot} is outside the ` +
          `${resolvedRefs.length} slot(s) declared by transform_refs`,
      );
    }
    data[i * 2] = resolvedRefs[slot | 0];
    data[i * 2 + 1] = weights[i];
  }

  return { count, data };
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
//
// THE CONTENT-CHANGE CONTRACT (D1a)
// ---------------------------------
// Any live `$state` change re-evaluates the serialized AST, which rebuilds the
// whole props tree. A component's `geometry` is therefore a *fresh JS object*
// on every state change even when its typed arrays are the identical buffers.
// Keying this cache on geometry object identity (the old WeakMap) meant it
// never hit under live state: every frame minted a new `__InlineMesh_N` type
// name, hence a new spec, new shader modules, new pipelines, and new GPU
// buffers that nothing ever destroyed.
//
// So identity is keyed structurally, and consulted *before* a type name is
// minted. The contract has exactly two halves:
//
//   1. IDENTITY - what determines the spec, shaders and pipelines. That is the
//      vertex format variant (normals/colors/uvs/shading/texture/cullMode)
//      plus the topology shape (vertex count, index count, index width). Two
//      geometries with the same identity share one `__InlineMesh_N` type name,
//      one spec, one shader set and one pipeline set, forever.
//        - When the user supplies `geometry_key` it *names* the identity:
//          `geometry_key` says "this is the same geometry across frames". It is
//          combined with the variant key (a different vertex format genuinely
//          needs a different spec) but replaces the structural topology part,
//          so a user who bumps `geometry_key` gets a clean rebuild and a user
//          who keeps it stable keeps their pipelines even across a topology
//          change (the buffers still resize - see D1b).
//
//          COROLLARY, and the user's responsibility: DISTINCT GEOMETRIES MUST
//          BE GIVEN DISTINCT KEYS. Two components sharing one `geometry_key`
//          within a scene are declaring themselves the same geometry, so they
//          share a type name, a spec and a single GPU buffer pair - whichever
//          resolves last wins and both render it. That is the meaning of the
//          key, not a bug; if two meshes differ, key them differently (or omit
//          the key and let the structural path separate them).
//        - Without `geometry_key`, the structural key is derived. A topology
//          change (different vertex count, different index count/width) is a
//          different identity and correctly rebuilds.
//
//          Structure alone does NOT identify a component, though: two meshes in
//          one scene can share a vertex count, format and index shape while
//          holding entirely different data (two same-resolution spheres, two
//          quads, two copies of a template). Collapsing them onto one entry
//          would give both a single geometry holder, so both would render
//          whichever resolved last, and each compile would thrash that holder
//          for two spurious contents bumps. So the derived key is salted with
//          the component's OCCURRENCE INDEX among structurally identical
//          siblings in the same resolve pass (`|n:<i>`). Component order within
//          a scene is stable across compiles, so the Nth such mesh keeps
//          hitting its own entry frame after frame. The counter is per-call,
//          never module-global, so it cannot drift across compiles.
//
//   2. CONTENTS - the vertex bytes. With a reused identity, fresh arrays mean
//      "same geometry, possibly new contents". Contents changes are detected
//      *cheaply* - by array identity and byte length only, never by comparing
//      vertex bytes - and routed to the buffer-write path (D1b), never to
//      re-minting a type name. Each detected change bumps a monotonic
//      `contentsVersion` which is published on the spec as `geometryKey`, the
//      signal impl3d already consults to decide whether a geometry resource is
//      current.
//
// OUT OF CONTRACT: mutating the *same* typed array in place. Nothing observes
// it, so the GPU keeps the previous bytes. This is deliberate - detecting it
// would mean a per-frame deep compare of every vertex - and it costs nothing in
// practice because Python-driven updates always ship fresh arrays. A caller who
// really does mutate in place must bump `geometry_key`.

interface InlineMeshCacheEntry {
  typeName: string;
  spec: PrimitiveSpec<any>;
  shading: "lit" | "unlit";
  cullMode: GPUCullMode;
  hasTexture: boolean;
  /** Mutable holder the spec's geometry closure reads. Swapped on contents change. */
  source: { geometry: StructuredGeometry };
  /** Raw (uncoerced) arrays of the geometry currently in `source`, for cheap comparison. */
  signature: ContentsSignature;
  /** Bumped whenever contents change; published on the spec as `geometryKey`. */
  contentsVersion: number;
  /**
   * Per-vertex transform references, when this mesh blends (D2).
   *
   * Kept on its own version counter, separate from `contentsVersion`: weights
   * are NOT part of the interleaved vertex buffer, so a weight-only change
   * must write the reference buffer and leave the vertex buffer alone. That
   * separation is the whole reason the references live in a storage buffer.
   */
  transformRefs?: ResolvedTransformRefs;
  transformRefsVersion: number;
  refsSignature?: RefsSignature;
}

/**
 * Cheap contents fingerprint: the array *objects* plus their lengths. Compared
 * by identity, never by value - see the content-change contract above.
 */
interface ContentsSignature {
  positions: unknown;
  positionsLength: number;
  normals: unknown;
  colors: unknown;
  uvs: unknown;
  indices: unknown;
  indicesLength: number;
}

function contentsSignature(geometry: MeshGeometry): ContentsSignature {
  return {
    positions: geometry.positions,
    positionsLength: getLength(geometry.positions),
    normals: geometry.normals,
    colors: geometry.colors,
    uvs: geometry.uvs,
    indices: geometry.indices,
    indicesLength: geometry.indices ? getLength(geometry.indices) : 0,
  };
}

function signaturesEqual(a: ContentsSignature, b: ContentsSignature): boolean {
  return (
    a.positions === b.positions &&
    a.positionsLength === b.positionsLength &&
    a.normals === b.normals &&
    a.colors === b.colors &&
    a.uvs === b.uvs &&
    a.indices === b.indices &&
    a.indicesLength === b.indicesLength
  );
}

/**
 * Fingerprint of a mesh's transform references, by the same cheap
 * identity-not-value rule as the geometry signature. The resolved palette
 * indices are included because the same names can land on different palette
 * slots when the scene's group structure changes, and the shipped buffer holds
 * absolute indices.
 */
interface RefsSignature {
  indices: unknown;
  weights: unknown;
  names: string;
  resolved: string;
}

function refsSignature(
  geometry: MeshGeometry,
  resolved: ResolvedTransformRefs | undefined,
  namedTransforms: Map<string, number> | undefined,
): RefsSignature | undefined {
  if (!resolved) return undefined;
  const names = geometry.transformRefs ?? [];
  return {
    indices: geometry.transformIndices,
    weights: geometry.transformWeights,
    names: names.join("\u0000"),
    resolved: names.map((n) => namedTransforms?.get(n)).join(","),
  };
}

function refsSignaturesEqual(
  a: RefsSignature | undefined,
  b: RefsSignature | undefined,
): boolean {
  if (!a || !b) return a === b;
  return (
    a.indices === b.indices &&
    a.weights === b.weights &&
    a.names === b.names &&
    a.resolved === b.resolved
  );
}

/**
 * Index-buffer shape for the identity key: element count plus a coarse width
 * bucket. Presence-or-absence and element count are what matter here; nothing
 * about the spec, shaders or pipelines depends on index width.
 *
 * The bucket is deliberately approximate. `Uint16Array` and `Uint32Array` are
 * reported exactly; anything else (a plain JS array, some other view) is coerced
 * later by `coerceIndexData`, whose chosen width depends on the max index value
 * and so is not knowable without scanning - those all land in one "u" bucket.
 *
 * That approximation is safe because the DRAWN index format never comes from
 * this bucket. `createBuffers`/`updateBuffers` set `resource.indexFormat` from
 * the actual coerced array on every write, and the render path re-reads it from
 * the resource, so an index array whose values cross 65535 between frames
 * switches to `uint32` (and grows its buffer, since byteLength doubles) without
 * needing a new identity.
 */
function indexShape(geometry: MeshGeometry): string {
  if (!geometry.indices) return "0";
  const length = getLength(geometry.indices);
  const width =
    geometry.indices instanceof Uint32Array
      ? "w"
      : geometry.indices instanceof Uint16Array
        ? "n"
        : "u";
  return `${length}:${width}`;
}

/**
 * Entries hold only JS objects (a spec, closures, a geometry holder) - no GPU
 * resources; those live in impl3d's `resources` map, keyed by type name, and
 * are swept there. The cache is still capped so a pathological scene that mints
 * a new identity per frame (e.g. a growing vertex count) cannot grow without
 * bound. Insertion-order eviction = LRU, since a hit re-inserts.
 */
const MAX_INLINE_MESH_ENTRIES = 256;
const inlineMeshCache = new Map<string, InlineMeshCacheEntry>();
let inlineMeshId = 0;

/** Test/diagnostic hook: drop all cached inline-mesh identities. */
export function clearInlineMeshCache(): void {
  inlineMeshCache.clear();
}

/**
 * Per-resolve-pass tally of how many times each derived structural key has been
 * seen. Scoped to one `resolveInlineMeshes` call - never module-global, so
 * occurrence numbering restarts identically on every compile.
 */
type OccurrenceCounts = Map<string, number>;

/**
 * Publish an entry's resolved transform references on its spec, with a key the
 * renderer compares to decide whether to rewrite the reference buffer — the
 * same contents-signal shape `geometryKey` uses, on its own version so the two
 * buffers move independently.
 */
function publishTransformRefs(
  entry: InlineMeshCacheEntry,
  identityKey: string,
): void {
  const spec = entry.spec as any;
  spec.transformRefData = entry.transformRefs?.data;
  spec.transformRefsKey = entry.transformRefs
    ? `${identityKey}#r${entry.transformRefsVersion}`
    : undefined;
}

function getInlineMeshEntry(
  geometry: MeshGeometry,
  options: {
    geometryKey?: string | number;
    shading?: "lit" | "unlit";
    cullMode?: GPUCullMode;
    hasTexture?: boolean;
    namedTransforms?: Map<string, number>;
  },
  occurrences: OccurrenceCounts,
): InlineMeshCacheEntry {
  const shading = options.shading ?? "lit";
  const cullMode = options.cullMode ?? "back";
  const hasTexture = options.hasTexture ?? false;

  // Per-vertex weighted transform references (D2). Resolved here, at compile,
  // because this is where the flattened palette exists. K joins the variant
  // key below: blended and rigid meshes have different shaders and different
  // pipeline layouts, so they must never share a cache entry.
  const transformRefs = resolveTransformRefs(geometry, options.namedTransforms);

  // Variant: everything that changes the generated spec/shaders/pipelines.
  const format = detectFormat(geometry, shading);
  const variantKey = `${getFormatKey(
    format,
    shading,
    hasTexture,
    transformRefs?.count ?? 0,
  )}|${cullMode}`;

  // Identity: variant + a name for "the same geometry". A user-supplied
  // geometry_key names it directly (and two components sharing one are
  // declaring themselves the same geometry - see the contract above).
  // Otherwise derive it from topology shape, salted with this component's
  // occurrence index among structurally identical siblings so that distinct
  // meshes of identical structure get distinct identities.
  let identityKey: string;
  if (options.geometryKey !== undefined) {
    identityKey = `${variantKey}|k:${options.geometryKey}`;
  } else {
    const structuralKey = `${variantKey}|v:${getLength(geometry.positions)}|i:${indexShape(geometry)}`;
    const occurrence = occurrences.get(structuralKey) ?? 0;
    occurrences.set(structuralKey, occurrence + 1);
    identityKey = `${structuralKey}|n:${occurrence}`;
  }

  const signature = contentsSignature(geometry);
  const refsSig = refsSignature(
    geometry,
    transformRefs,
    options.namedTransforms,
  );
  const existing = inlineMeshCache.get(identityKey);
  if (existing) {
    // Same identity: reuse type name, spec, shaders, pipelines. If the arrays
    // are new objects, that is a *contents* change - swap the geometry the
    // spec reads and bump the version so impl3d rewrites the buffers.
    if (!signaturesEqual(existing.signature, signature)) {
      existing.source.geometry = coerceMeshGeometry(geometry);
      existing.signature = signature;
      existing.contentsVersion++;
      (existing.spec as any).geometryKey =
        `${identityKey}#${existing.contentsVersion}`;
    }
    // Transform references are versioned independently: a weights-only change
    // rewrites the reference storage buffer and MUST leave the interleaved
    // vertex buffer alone.
    if (!refsSignaturesEqual(existing.refsSignature, refsSig)) {
      existing.transformRefs = transformRefs;
      existing.refsSignature = refsSig;
      existing.transformRefsVersion++;
      publishTransformRefs(existing, identityKey);
    }
    // Re-insert to keep insertion order as recency for eviction.
    inlineMeshCache.delete(identityKey);
    inlineMeshCache.set(identityKey, existing);
    return existing;
  }

  // New identity: mint a type name and build the spec once.
  const source = { geometry: coerceMeshGeometry(geometry) };
  const typeName = `__InlineMesh_${inlineMeshId++}`;
  const spec = defineMesh(typeName, () => source.geometry, {
    shading,
    cullMode,
    hasTexture,
    transformRefCount: transformRefs?.count ?? 0,
  });

  // `geometryKey` on the spec is the contents signal impl3d consults. It is
  // always set (not only when the user supplied one) so that a contents change
  // under a stable identity is always observable downstream.
  (spec as any).geometryKey = `${identityKey}#0`;

  const entry: InlineMeshCacheEntry = {
    typeName,
    spec,
    shading,
    cullMode,
    hasTexture,
    source,
    signature,
    contentsVersion: 0,
    transformRefs,
    transformRefsVersion: 0,
    refsSignature: refsSig,
  };
  publishTransformRefs(entry, identityKey);
  inlineMeshCache.set(identityKey, entry);

  if (inlineMeshCache.size > MAX_INLINE_MESH_ENTRIES) {
    const oldest = inlineMeshCache.keys().next();
    if (!oldest.done) inlineMeshCache.delete(oldest.value);
  }
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
  namedTransforms?: Map<string, number>,
): {
  components: ComponentConfig[];
  inlineSpecs?: Record<string, PrimitiveSpec<any>>;
} {
  let inlineSpecs: Record<string, PrimitiveSpec<any>> | undefined;
  // Fresh per call: the Nth structurally-identical mesh in this components
  // array gets occurrence N, and component order is stable across compiles.
  const occurrences: OccurrenceCounts = new Map();
  const resolved = components.map((component) => {
    if (component.type !== "Mesh") return component;
    const meshComponent = component as InlineMeshComponentConfig;
    // Detect if mesh uses a texture (presence of texture prop)
    const hasTexture = meshComponent.texture !== undefined;
    const entry = getInlineMeshEntry(
      meshComponent.geometry,
      {
        geometryKey: meshComponent.geometryKey,
        shading: meshComponent.shading,
        cullMode: meshComponent.cullMode,
        hasTexture,
        namedTransforms,
      },
      occurrences,
    );

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
