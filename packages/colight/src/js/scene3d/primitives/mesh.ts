/**
 * @module primitives/mesh
 * @description Generic mesh primitive factory using the declarative definition system.
 *
 * Supports flexible vertex formats with optional normals, vertex colors, and UVs.
 */

import {
  BaseComponentConfig,
  GeometryData,
  VertexBufferLayout,
} from "../types";
import {
  definePrimitive,
  attr,
  createVertexBufferLayout,
  cameraStruct,
  groupTransformStruct,
  applyGroupTransformFn,
  lightingConstants,
  lightingCalc,
  quaternionShaderFunctions,
} from "./define";
import { ImageSource, getImageBindGroupLayout } from "./imagePlane";

// =============================================================================
// Structured Geometry Types
// =============================================================================

/**
 * Structured geometry with named attribute arrays.
 * More flexible and self-documenting than the opaque vertexData format.
 */
export interface StructuredGeometry {
  /** Vertex positions (N, 3) - required */
  positions: Float32Array | number[];
  /** Vertex normals (N, 3) - optional, auto-computed if missing for lit shading */
  normals?: Float32Array | number[];
  /** Per-vertex colors (N, 3) RGB or (N, 4) RGBA - optional */
  colors?: Float32Array | number[];
  /** Texture coordinates (N, 2) - optional */
  uvs?: Float32Array | number[];
  /** Triangle indices - optional */
  indices?: Uint16Array | Uint32Array | number[];
}

/**
 * Vertex format descriptor - determines shader variant and buffer layout.
 */
export interface VertexFormat {
  hasNormals: boolean;
  hasColors: boolean;
  colorComponents: 3 | 4;
  hasUVs: boolean;
}

/**
 * Result of processing structured geometry.
 */
export interface ProcessedMeshGeometry {
  vertexData: Float32Array;
  indexData?: Uint16Array | Uint32Array;
  format: VertexFormat;
  vertexCount: number;
  stride: number; // floats per vertex
}

// =============================================================================
// Geometry Processing
// =============================================================================

/**
 * Detect vertex format from structured geometry.
 */
function detectFormat(geometry: StructuredGeometry): VertexFormat {
  const hasColors = geometry.colors !== undefined && geometry.colors.length > 0;
  let colorComponents: 3 | 4 = 3;
  if (hasColors) {
    const positions = geometry.positions;
    const posCount = Array.isArray(positions)
      ? positions.length / 3
      : positions.length / 3;
    const colorLen = Array.isArray(geometry.colors)
      ? geometry.colors.length
      : geometry.colors!.length;
    colorComponents = colorLen / posCount === 4 ? 4 : 3;
  }

  return {
    hasNormals: geometry.normals !== undefined && geometry.normals.length > 0,
    hasColors,
    colorComponents,
    hasUVs: geometry.uvs !== undefined && geometry.uvs.length > 0,
  };
}

/**
 * Calculate stride (floats per vertex) from format.
 */
function calculateStride(format: VertexFormat): number {
  let stride = 3; // positions always present
  if (format.hasNormals) stride += 3;
  if (format.hasColors) stride += format.colorComponents;
  if (format.hasUVs) stride += 2;
  return stride;
}

/**
 * Compute flat-shaded normals from positions and indices.
 * Each face gets a uniform normal (no smooth shading).
 */
function computeFlatNormals(
  positions: Float32Array | number[],
  indices?: Uint16Array | Uint32Array | number[],
): Float32Array {
  const pos = Array.isArray(positions)
    ? new Float32Array(positions)
    : positions;
  const vertexCount = pos.length / 3;
  const normals = new Float32Array(vertexCount * 3);

  if (!indices) {
    // Non-indexed: every 3 vertices form a triangle
    for (let i = 0; i < vertexCount; i += 3) {
      const i0 = i * 3,
        i1 = (i + 1) * 3,
        i2 = (i + 2) * 3;
      const ax = pos[i1] - pos[i0],
        ay = pos[i1 + 1] - pos[i0 + 1],
        az = pos[i1 + 2] - pos[i0 + 2];
      const bx = pos[i2] - pos[i0],
        by = pos[i2 + 1] - pos[i0 + 1],
        bz = pos[i2 + 2] - pos[i0 + 2];
      let nx = ay * bz - az * by,
        ny = az * bx - ax * bz,
        nz = ax * by - ay * bx;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      nx /= len;
      ny /= len;
      nz /= len;
      for (let j = 0; j < 3; j++) {
        normals[(i + j) * 3] = nx;
        normals[(i + j) * 3 + 1] = ny;
        normals[(i + j) * 3 + 2] = nz;
      }
    }
  } else {
    // Indexed: accumulate face normals per vertex
    const idx = Array.isArray(indices) ? indices : Array.from(indices);
    for (let i = 0; i < idx.length; i += 3) {
      const v0 = idx[i],
        v1 = idx[i + 1],
        v2 = idx[i + 2];
      const i0 = v0 * 3,
        i1 = v1 * 3,
        i2 = v2 * 3;
      const ax = pos[i1] - pos[i0],
        ay = pos[i1 + 1] - pos[i0 + 1],
        az = pos[i1 + 2] - pos[i0 + 2];
      const bx = pos[i2] - pos[i0],
        by = pos[i2 + 1] - pos[i0 + 1],
        bz = pos[i2 + 2] - pos[i0 + 2];
      let nx = ay * bz - az * by,
        ny = az * bx - ax * bz,
        nz = ax * by - ay * bx;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      nx /= len;
      ny /= len;
      nz /= len;
      // Add to all 3 vertices of the face
      for (const v of [v0, v1, v2]) {
        normals[v * 3] += nx;
        normals[v * 3 + 1] += ny;
        normals[v * 3 + 2] += nz;
      }
    }
    // Normalize accumulated normals
    for (let i = 0; i < vertexCount; i++) {
      const i3 = i * 3;
      const nx = normals[i3],
        ny = normals[i3 + 1],
        nz = normals[i3 + 2];
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
      normals[i3] /= len;
      normals[i3 + 1] /= len;
      normals[i3 + 2] /= len;
    }
  }
  return normals;
}

/**
 * Interleave structured geometry into a packed vertex buffer.
 * Layout: [position, normal?, color?, uv?] per vertex
 */
export function interleaveVertexData(
  geometry: StructuredGeometry,
): ProcessedMeshGeometry {
  const format = detectFormat(geometry);
  const stride = calculateStride(format);

  const positions = Array.isArray(geometry.positions)
    ? new Float32Array(geometry.positions)
    : geometry.positions;
  const vertexCount = positions.length / 3;

  // Auto-compute normals if needed for lit shading and not provided
  let normals: Float32Array | undefined;
  if (format.hasNormals) {
    normals = Array.isArray(geometry.normals)
      ? new Float32Array(geometry.normals!)
      : geometry.normals;
  }

  const colors = geometry.colors
    ? Array.isArray(geometry.colors)
      ? new Float32Array(geometry.colors)
      : geometry.colors
    : undefined;

  const uvs = geometry.uvs
    ? Array.isArray(geometry.uvs)
      ? new Float32Array(geometry.uvs)
      : geometry.uvs
    : undefined;

  // Interleave into packed buffer
  const vertexData = new Float32Array(vertexCount * stride);
  for (let i = 0; i < vertexCount; i++) {
    let offset = i * stride;
    const i3 = i * 3;

    // Position (always)
    vertexData[offset++] = positions[i3];
    vertexData[offset++] = positions[i3 + 1];
    vertexData[offset++] = positions[i3 + 2];

    // Normal (if present)
    if (normals) {
      vertexData[offset++] = normals[i3];
      vertexData[offset++] = normals[i3 + 1];
      vertexData[offset++] = normals[i3 + 2];
    }

    // Color (if present)
    if (colors) {
      const ci = i * format.colorComponents;
      vertexData[offset++] = colors[ci];
      vertexData[offset++] = colors[ci + 1];
      vertexData[offset++] = colors[ci + 2];
      if (format.colorComponents === 4) {
        vertexData[offset++] = colors[ci + 3];
      }
    }

    // UV (if present)
    if (uvs) {
      const ui = i * 2;
      vertexData[offset++] = uvs[ui];
      vertexData[offset++] = uvs[ui + 1];
    }
  }

  // Process indices
  let indexData: Uint16Array | Uint32Array | undefined;
  if (geometry.indices) {
    if (
      geometry.indices instanceof Uint16Array ||
      geometry.indices instanceof Uint32Array
    ) {
      indexData = geometry.indices;
    } else {
      const maxIndex = Math.max(...geometry.indices);
      indexData =
        maxIndex > 65535
          ? new Uint32Array(geometry.indices)
          : new Uint16Array(geometry.indices);
    }
  }

  return { vertexData, indexData, format, vertexCount, stride };
}

// =============================================================================
// Configuration Interface
// =============================================================================

export interface MeshComponentConfig extends BaseComponentConfig {
  type: string;
  /** Mesh centers: [x, y, z, ...] */
  centers: Float32Array | number[];
  /** Per-instance scales: [sx, sy, sz, ...] - coerced to Float32Array */
  scales?: Float32Array;
  /** Default scale for all instances */
  scale?: [number, number, number] | number;
  /** Per-instance rotations as quaternions: [w, x, y, z, ...] */
  quaternions?: Float32Array;
  /** Default quaternion for all instances [w, x, y, z] */
  quaternion?: [number, number, number, number];
  /** Texture image source (requires UVs in geometry) */
  texture?: ImageSource;
  /** Optional key to force texture updates */
  textureKey?: string | number;
}

// =============================================================================
// Shader Generation
// =============================================================================

/**
 * Get a unique key for a vertex format (for caching).
 */
export function getFormatKey(
  format: VertexFormat,
  shading: "lit" | "unlit",
  hasTexture = false,
): string {
  return `${format.hasNormals ? "N" : ""}${format.hasColors ? `C${format.colorComponents}` : ""}${format.hasUVs ? "U" : ""}${hasTexture ? "T" : ""}_${shading}`;
}

/**
 * Count the number of geometry attribute locations used by a format.
 */
function countGeometryLocations(format: VertexFormat): number {
  let count = 1; // position always
  if (format.hasNormals) count++;
  if (format.hasColors) count++;
  if (format.hasUVs) count++;
  return count;
}

/**
 * Build geometry buffer layout for a vertex format.
 */
function buildGeometryLayout(format: VertexFormat): VertexBufferLayout {
  const attributes: {
    shaderLocation: number;
    offset: number;
    format: GPUVertexFormat;
  }[] = [];
  let offset = 0;
  let location = 0;

  // Position (always)
  attributes.push({ shaderLocation: location++, offset, format: "float32x3" });
  offset += 12;

  // Normal (optional)
  if (format.hasNormals) {
    attributes.push({
      shaderLocation: location++,
      offset,
      format: "float32x3",
    });
    offset += 12;
  }

  // Color (optional)
  if (format.hasColors) {
    const colorFormat: GPUVertexFormat =
      format.colorComponents === 4 ? "float32x4" : "float32x3";
    attributes.push({
      shaderLocation: location++,
      offset,
      format: colorFormat,
    });
    offset += format.colorComponents * 4;
  }

  // UV (optional)
  if (format.hasUVs) {
    attributes.push({
      shaderLocation: location++,
      offset,
      format: "float32x2",
    });
    offset += 8;
  }

  return {
    arrayStride: offset,
    stepMode: "vertex",
    attributes,
  };
}

/**
 * Build instance buffer layout for render pass.
 * Instance attributes: position(vec3), size(vec3), rotation(vec4), color(vec3), alpha(f32), transformIndex(f32)
 */
function buildRenderInstanceLayout(startLocation: number): VertexBufferLayout {
  let loc = startLocation;
  return {
    arrayStride: (3 + 3 + 4 + 3 + 1 + 1) * 4, // 15 floats * 4 bytes
    stepMode: "instance",
    attributes: [
      { shaderLocation: loc++, offset: 0, format: "float32x3" }, // position
      { shaderLocation: loc++, offset: 12, format: "float32x3" }, // size
      { shaderLocation: loc++, offset: 24, format: "float32x4" }, // rotation
      { shaderLocation: loc++, offset: 40, format: "float32x3" }, // color
      { shaderLocation: loc++, offset: 52, format: "float32" }, // alpha
      { shaderLocation: loc++, offset: 56, format: "float32" }, // transformIndex
    ],
  };
}

/**
 * Build instance buffer layout for picking pass.
 * Picking attributes: position(vec3), size(vec3), rotation(vec4), transformIndex(f32), pickID(f32)
 * Note: Order matches processSchema convention (transformIndex before pickID)
 */
function buildPickingInstanceLayout(startLocation: number): VertexBufferLayout {
  let loc = startLocation;
  return {
    arrayStride: (3 + 3 + 4 + 1 + 1) * 4, // 12 floats * 4 bytes
    stepMode: "instance",
    attributes: [
      { shaderLocation: loc++, offset: 0, format: "float32x3" }, // position
      { shaderLocation: loc++, offset: 12, format: "float32x3" }, // size
      { shaderLocation: loc++, offset: 24, format: "float32x4" }, // rotation
      { shaderLocation: loc++, offset: 40, format: "float32" }, // transformIndex
      { shaderLocation: loc++, offset: 44, format: "float32" }, // pickID
    ],
  };
}

/**
 * Generate vertex shader for a specific vertex format.
 */
function generateMeshVertexShader(
  format: VertexFormat,
  forPicking: boolean,
  hasTexture = false,
): string {
  // Build vertex inputs from geometry buffer
  const geoInputs: string[] = ["@location(0) localPos: vec3<f32>"];
  let geoLoc = 1;
  if (format.hasNormals)
    geoInputs.push(`@location(${geoLoc++}) normal: vec3<f32>`);
  if (format.hasColors) {
    const colorType = format.colorComponents === 4 ? "vec4<f32>" : "vec3<f32>";
    geoInputs.push(`@location(${geoLoc++}) vertexColor: ${colorType}`);
  }
  if (format.hasUVs) geoInputs.push(`@location(${geoLoc++}) uv: vec2<f32>`);

  // Instance inputs start after geometry inputs
  const instInputs: string[] = [
    `@location(${geoLoc}) position: vec3<f32>`,
    `@location(${geoLoc + 1}) size: vec3<f32>`,
    `@location(${geoLoc + 2}) rotation: vec4<f32>`,
  ];

  if (forPicking) {
    // Order matches processSchema convention: transformIndex before pickID
    instInputs.push(`@location(${geoLoc + 3}) transformIndex: f32`);
    instInputs.push(`@location(${geoLoc + 4}) pickID: f32`);
  } else {
    instInputs.push(`@location(${geoLoc + 3}) instanceColor: vec3<f32>`);
    instInputs.push(`@location(${geoLoc + 4}) alpha: f32`);
    instInputs.push(`@location(${geoLoc + 5}) transformIndex: f32`);
  }

  const allInputs = [...geoInputs, ...instInputs].join(",\n  ");

  // VSOut struct - include UV when textured (not for picking)
  let vsOut: string;
  if (forPicking) {
    vsOut = `struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) pickID: f32
};`;
  } else if (hasTexture) {
    vsOut = `struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>,
  @location(4) texCoord: vec2<f32>
};`;
  } else {
    vsOut = `struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
};`;
  }

  // Color computation
  let colorComputation = "";
  if (!forPicking) {
    if (format.hasColors) {
      // Multiply vertex color by instance color
      colorComputation =
        format.colorComponents === 4
          ? "let finalColor = vertexColor.rgb * instanceColor;\n  let finalAlpha = vertexColor.a * alpha;"
          : "let finalColor = vertexColor * instanceColor;\n  let finalAlpha = alpha;";
    } else {
      colorComputation =
        "let finalColor = instanceColor;\n  let finalAlpha = alpha;";
    }
  }

  // Normal handling (with group transform support)
  // Note: composedQuat is already declared in main body, just reference it here
  const normalComputation = format.hasNormals
    ? `let invScaledNorm = normalize(normal / effectiveSize);
  let worldNormal = quat_rotate(composedQuat, invScaledNorm);`
    : "let worldNormal = vec3<f32>(0.0, 1.0, 0.0);"; // Default up normal for unlit

  // Return statement
  let returnStmt: string;
  if (forPicking) {
    returnStmt = `var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;`;
  } else if (hasTexture) {
    returnStmt = `var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = finalColor;
  out.alpha = finalAlpha;
  out.worldPos = worldPos;
  out.normal = worldNormal;
  out.texCoord = uv;
  return out;`;
  } else {
    returnStmt = `var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = finalColor;
  out.alpha = finalAlpha;
  out.worldPos = worldPos;
  out.normal = worldNormal;
  return out;`;
  }

  return `${cameraStruct}
${groupTransformStruct}
${vsOut}
${quaternionShaderFunctions}
${applyGroupTransformFn}

@vertex
fn vs_main(
  ${allInputs}
) -> VSOut {
  // Get group transform
  let groupIdx = u32(transformIndex);
  let groupT = transforms[groupIdx];

  // Compose instance rotation with group rotation
  let composedQuat = quat_mul(groupT.quaternion, rotation);

  // Apply group scale to instance size
  let effectiveSize = size * groupT.scale;

  // Rigid transform with rotation and group transform
  let scaledLocal = localPos * effectiveSize;
  let rotatedPos = quat_rotate(composedQuat, scaledLocal);

  // Transform instance position by group, then add local offset
  let groupWorldPos = applyGroupTransform(position, groupIdx);
  let worldPos = groupWorldPos + rotatedPos;

  ${normalComputation}
  ${colorComputation}
  ${returnStmt}
}`;
}

/**
 * Generate fragment shader for a specific vertex format and shading mode.
 */
function generateMeshFragmentShader(
  shading: "lit" | "unlit",
  hasTexture = false,
): string {
  // Texture bindings (when textured)
  const textureBindings = hasTexture
    ? `
@group(1) @binding(0) var meshSampler: sampler;
@group(1) @binding(1) var meshTexture: texture_2d<f32>;
`
    : "";

  // Fragment inputs - include texCoord when textured
  const fragInputs = hasTexture
    ? `@location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>,
  @location(4) texCoord: vec2<f32>`
    : `@location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>`;

  if (shading === "lit") {
    const colorCalc = hasTexture
      ? `let tex = textureSample(meshTexture, meshSampler, texCoord);
  let baseColor = tex.rgb * color;
  let baseAlpha = tex.a * alpha;`
      : `let baseColor = color;
  let baseAlpha = alpha;`;

    return `${cameraStruct}
${lightingConstants}
${lightingCalc}
${textureBindings}
@fragment
fn fs_main(
  ${fragInputs}
) -> @location(0) vec4<f32> {
  ${colorCalc}
  let litColor = calculateLighting(baseColor, normal, worldPos);
  return vec4<f32>(litColor, baseAlpha);
}`;
  } else {
    const colorCalc = hasTexture
      ? `let tex = textureSample(meshTexture, meshSampler, texCoord);
  return vec4<f32>(tex.rgb * color, tex.a * alpha);`
      : `return vec4<f32>(color, alpha);`;

    return `${textureBindings}
@fragment
fn fs_main(
  ${fragInputs}
) -> @location(0) vec4<f32> {
  ${colorCalc}
}`;
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

export interface MeshOptions {
  /** Shading model: "lit" (default) or "unlit" */
  shading?: "lit" | "unlit";
  /** Face culling mode: "back" (default), "front", or "none" */
  cullMode?: GPUCullMode;
  /** Whether this mesh uses a texture (requires UVs in geometry) */
  hasTexture?: boolean;
}

/**
 * Defines a new mesh primitive from structured geometry.
 * Supports flexible vertex formats with optional normals, colors, and UVs.
 *
 * @param name - Unique name for this primitive type
 * @param geometry - Structured geometry with named attribute arrays
 * @param options - Optional rendering configuration
 */
export function defineMesh(
  name: string,
  geometry: StructuredGeometry | (() => StructuredGeometry),
  options: MeshOptions = {},
) {
  const shading = options.shading ?? "lit";
  const hasTexture = options.hasTexture ?? false;

  // Process geometry to determine format
  const getProcessed =
    typeof geometry === "function"
      ? () => {
          const geo = geometry();
          // Auto-compute normals for lit shading if not provided
          if (shading === "lit" && !geo.normals) {
            geo.normals = computeFlatNormals(geo.positions, geo.indices);
          }
          return interleaveVertexData(geo);
        }
      : () => {
          const geo = { ...geometry };
          if (shading === "lit" && !geo.normals) {
            geo.normals = computeFlatNormals(geo.positions, geo.indices);
          }
          return interleaveVertexData(geo);
        };

  // Get format from a sample processing (needed for layout/shaders)
  const sampleGeo = typeof geometry === "function" ? geometry() : geometry;
  const hasNormals =
    shading === "lit" ||
    (sampleGeo.normals !== undefined && sampleGeo.normals.length > 0);
  const format: VertexFormat = {
    hasNormals,
    hasColors: sampleGeo.colors !== undefined && sampleGeo.colors.length > 0,
    colorComponents: sampleGeo.colors
      ? (Array.isArray(sampleGeo.colors)
          ? sampleGeo.colors.length
          : sampleGeo.colors.length) /
          ((Array.isArray(sampleGeo.positions)
            ? sampleGeo.positions.length
            : sampleGeo.positions.length) /
            3) ===
        4
        ? 4
        : 3
      : 3,
    hasUVs: sampleGeo.uvs !== undefined && sampleGeo.uvs.length > 0,
  };

  const geometryLayout = buildGeometryLayout(format);
  const instanceStartLocation = countGeometryLocations(format);
  const renderInstanceLayout = buildRenderInstanceLayout(instanceStartLocation);
  const pickingInstanceLayout = buildPickingInstanceLayout(
    instanceStartLocation,
  );
  const vertexShader = generateMeshVertexShader(format, false, hasTexture);
  const pickingVertexShader = generateMeshVertexShader(format, true, false);
  const fragmentShader = generateMeshFragmentShader(shading, hasTexture);

  const spec = definePrimitive<MeshComponentConfig>({
    name,

    attributes: {
      position: attr.vec3("centers"),
      size: attr.vec3("scales", [1, 1, 1]),
      rotation: attr.quat("quaternions"),
      color: attr.vec3("colors", [0.5, 0.5, 0.5]),
      alpha: attr.f32("alphas", 1.0),
    },

    geometry: {
      type: "custom",
      create: () => {
        const processed = getProcessed();
        return {
          vertexData: processed.vertexData,
          indexData: processed.indexData,
        };
      },
    },

    geometryLayout,
    renderInstanceLayout,
    pickingInstanceLayout,
    vertexShader,
    pickingVertexShader,
    fragmentShader,

    transform: "rigid",
    shading,
    cullMode: options.cullMode ?? "back",
    // Textured meshes need an additional bind group for the texture
    bindGroupLayouts: hasTexture
      ? (device, baseLayout) => [baseLayout, getImageBindGroupLayout(device)]
      : undefined,
  });

  // Textured meshes batch by texture (like ImagePlane)
  if (hasTexture) {
    spec.getBatchKey = (elem) => {
      if (elem.textureKey !== undefined) return `key:${elem.textureKey}`;
      if (elem.texture && typeof elem.texture === "object") {
        return `tex:${getTextureObjectId(elem.texture as object)}`;
      }
      return "tex:unknown";
    };
  }

  return spec;
}

// Texture object identity tracking for batch keys
const textureIdMap = new WeakMap<object, string>();
let textureIdCounter = 0;

function getTextureObjectId(texture: object): string {
  let id = textureIdMap.get(texture);
  if (!id) {
    id = `texture_${textureIdCounter++}`;
    textureIdMap.set(texture, id);
  }
  return id;
}

/**
 * Legacy function for raw geometry data.
 * Prefer using defineMesh with StructuredGeometry for new code.
 */
export function defineMeshRaw(
  name: string,
  geometry: GeometryData | (() => GeometryData),
  options: MeshOptions = {},
) {
  return definePrimitive<MeshComponentConfig>({
    name,

    attributes: {
      position: attr.vec3("centers"),
      size: attr.vec3("scales", [1, 1, 1]),
      rotation: attr.quat("quaternions"),
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
