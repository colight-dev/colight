/**
 * @module primitives/boundingBox
 * @description BoundingBox primitive with custom wireframe geometry.
 *
 * Renders wireframe boxes using a single geometry containing all 12 edges.
 * Each box is one instance, all edges share the same quaternion transform.
 */

import { BaseComponentConfig } from "../types";
import {
  definePrimitive,
  attr,
  ProcessedSchema,
  packID,
  createVertexBufferLayout,
  cameraStruct,
  lightingConstants,
  lightingCalc,
  pickingVSOut,
  pickingFragCode,
  quaternionShaderFunctions,
  resolveSingular,
  expandScalar,
  coerceFloat32Fields,
} from "./define";
import { GeometryData } from "../types";

// =============================================================================
// Configuration Interface (internal format after coercion)
// =============================================================================

export interface BoundingBoxComponentConfig extends BaseComponentConfig {
  type: "BoundingBox";
  /** Box centers: [x, y, z, ...] */
  centers: Float32Array | number[];
  /** Per-box half sizes: [hx, hy, hz, ...] */
  half_sizes?: Float32Array | number[];
  /** Default half size for all boxes */
  half_size?: number | [number, number, number];
  /** Per-box rotations as quaternions: [w, x, y, z, ...] */
  quaternions?: Float32Array | number[];
  /** Default quaternion for all boxes [w, x, y, z] */
  quaternion?: [number, number, number, number];
  /** Edge thickness */
  sizes?: Float32Array | number[];
  /** Default edge thickness for all boxes */
  size?: number;
}

// =============================================================================
// Props Type (user-facing input)
// =============================================================================

export type BoundingBoxProps = Omit<
  BoundingBoxComponentConfig,
  "type" | "centers"
> & {
  centers?: ArrayLike<number> | ArrayBufferView;
  center?: [number, number, number];
};

// =============================================================================
// Coerce Function
// =============================================================================

export function coerceBoundingBox(
  props: Record<string, any>,
): Record<string, any> {
  let coerced = resolveSingular(props, "center", "centers");
  coerced = expandScalar(coerced, "half_size");
  coerced = coerceFloat32Fields(coerced, [
    "centers",
    "half_sizes",
    "quaternions",
    "colors",
    "sizes",
    "alphas",
  ]);
  return { ...coerced, type: "BoundingBox" };
}

// =============================================================================
// Wireframe Box Geometry
// =============================================================================

/**
 * 8 corners of a unit box centered at origin.
 */
const CORNERS: [number, number, number][] = [
  [-1, -1, -1], // 0
  [+1, -1, -1], // 1
  [+1, +1, -1], // 2
  [-1, +1, -1], // 3
  [-1, -1, +1], // 4
  [+1, -1, +1], // 5
  [+1, +1, +1], // 6
  [-1, +1, +1], // 7
];

/**
 * 12 edges as pairs of corner indices.
 */
const EDGES: [number, number][] = [
  // Bottom face (Z-)
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  // Top face (Z+)
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  // Vertical edges
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
];

/**
 * Creates wireframe box geometry with all 12 edges as rectangular beams.
 *
 * Vertex format: position(3) + normal(3) + beamOffset(3) + inset(3) = 12 floats
 * - position: corner position (±1 on each axis)
 * - normal: outward-facing normal for lighting
 * - beamOffset: perpendicular offset direction (will be scaled by edgeSize)
 * - inset: direction to pull endpoint inward (for horizontal edges to avoid overlap)
 *
 * Vertical (Z) edges extend fully; horizontal (X,Y) edges are inset by edgeSize.
 */
function createWireframeBoxGeometry(): GeometryData {
  const vertices: number[] = [];
  const indices: number[] = [];

  // Beam cross-section offsets (±0.5 will be scaled by edgeSize in shader)
  const offsets: [number, number][] = [
    [-0.5, -0.5],
    [+0.5, -0.5],
    [+0.5, +0.5],
    [-0.5, +0.5],
  ];

  for (const [c0, c1] of EDGES) {
    const p0 = CORNERS[c0];
    const p1 = CORNERS[c1];

    const dx = p1[0] - p0[0];
    const dy = p1[1] - p0[1];
    const dz = p1[2] - p0[2];

    // Determine edge type and perpendicular axes
    let ax: number, ay: number, az: number; // First perpendicular axis
    let bx: number, by: number, bz: number; // Second perpendicular axis
    let isVertical: boolean;

    if (Math.abs(dz) > 0.5) {
      // Z-aligned (vertical): perpendicular are X and Y, no inset
      ax = 1;
      ay = 0;
      az = 0;
      bx = 0;
      by = 1;
      bz = 0;
      isVertical = true;
    } else if (Math.abs(dx) > 0.5) {
      // X-aligned: perpendicular are Y and Z, inset along X
      ax = 0;
      ay = 1;
      az = 0;
      bx = 0;
      by = 0;
      bz = 1;
      isVertical = false;
    } else {
      // Y-aligned: perpendicular are X and Z, inset along Y
      ax = 1;
      ay = 0;
      az = 0;
      bx = 0;
      by = 0;
      bz = 1;
      isVertical = false;
    }

    // Edge direction (normalized)
    const edgeLen = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const edx = dx / edgeLen;
    const edy = dy / edgeLen;
    const edz = dz / edgeLen;

    const baseIdx = vertices.length / 12;

    // 8 vertices: 4 at start corner (end=0), 4 at end corner (end=1)
    for (let end = 0; end < 2; end++) {
      const p = end === 0 ? p0 : p1;
      // Inset direction: points inward along edge
      // For start (end=0): inset in positive edge direction
      // For end (end=1): inset in negative edge direction
      const insetSign = end === 0 ? 1 : -1;

      for (const [u, v] of offsets) {
        // Corner position
        vertices.push(p[0], p[1], p[2]);

        // Normal (perpendicular to edge, outward from beam center)
        const nx = u * ax + v * bx;
        const ny = u * ay + v * by;
        const nz = u * az + v * bz;
        const nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        vertices.push(nx / nl, ny / nl, nz / nl);

        // Beam offset direction (will be scaled by edgeSize)
        vertices.push(u * ax + v * bx, u * ay + v * by, u * az + v * bz);

        // Inset direction (scaled by edgeSize in shader)
        // Vertical edges: extend outward by 0.5 (half beam width) to own corners
        // Horizontal edges: inset by 0.5 to meet vertical edges cleanly
        if (isVertical) {
          // Negative inset = extend outward
          vertices.push(
            -0.5 * edx * insetSign,
            -0.5 * edy * insetSign,
            -0.5 * edz * insetSign,
          );
        } else {
          // Positive inset = pull inward by half beam width
          vertices.push(
            0.5 * edx * insetSign,
            0.5 * edy * insetSign,
            0.5 * edz * insetSign,
          );
        }
      }
    }

    // Indices for the beam faces
    // Start cap (vertices 0-3), winding for inward-facing normal
    indices.push(baseIdx + 0, baseIdx + 2, baseIdx + 1);
    indices.push(baseIdx + 0, baseIdx + 3, baseIdx + 2);
    // End cap (vertices 4-7), winding for outward-facing normal
    indices.push(baseIdx + 4, baseIdx + 5, baseIdx + 6);
    indices.push(baseIdx + 4, baseIdx + 6, baseIdx + 7);
    // Side faces
    indices.push(
      baseIdx + 0,
      baseIdx + 1,
      baseIdx + 5,
      baseIdx + 0,
      baseIdx + 5,
      baseIdx + 4,
    );
    indices.push(
      baseIdx + 1,
      baseIdx + 2,
      baseIdx + 6,
      baseIdx + 1,
      baseIdx + 6,
      baseIdx + 5,
    );
    indices.push(
      baseIdx + 2,
      baseIdx + 3,
      baseIdx + 7,
      baseIdx + 2,
      baseIdx + 7,
      baseIdx + 6,
    );
    indices.push(
      baseIdx + 3,
      baseIdx + 0,
      baseIdx + 4,
      baseIdx + 3,
      baseIdx + 4,
      baseIdx + 7,
    );
  }

  return {
    vertexData: new Float32Array(vertices),
    indexData: new Uint16Array(indices),
  };
}

// =============================================================================
// Custom Shaders
// =============================================================================

// Geometry layout: position(3) + normal(3) + beamOffset(3) + inset(3) = 12 floats
const GEOMETRY_LAYOUT = createVertexBufferLayout(
  [
    [0, "float32x3"], // corner position
    [1, "float32x3"], // normal
    [2, "float32x3"], // beam offset direction
    [3, "float32x3"], // inset direction
  ],
  "vertex",
);

const vertexShader = /*wgsl*/ `
${cameraStruct}
${quaternionShaderFunctions}

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
};

@vertex
fn vs_main(
  // Geometry attributes
  @location(0) cornerPos: vec3<f32>,    // Corner position (±1)
  @location(1) localNormal: vec3<f32>,  // Normal direction
  @location(2) beamOffset: vec3<f32>,   // Beam cross-section offset direction
  @location(3) inset: vec3<f32>,        // Inset direction for this vertex
  // Instance attributes
  @location(4) center: vec3<f32>,       // Box center
  @location(5) halfSize: vec3<f32>,     // Box half-extents
  @location(6) edgeSize: f32,           // Edge thickness
  @location(7) quaternion: vec4<f32>,   // Box rotation (xyzw)
  @location(8) color: vec3<f32>,
  @location(9) alpha: f32
) -> VSOut {
  // Scale corner by halfSize, offset by edgeSize, apply inset
  let scaledCorner = cornerPos * halfSize;
  let scaledOffset = beamOffset * edgeSize;
  let scaledInset = inset * edgeSize;
  let localPos = scaledCorner + scaledOffset + scaledInset;

  // Rotate by quaternion and translate to center
  let rotatedPos = quat_rotate(quaternion, localPos);
  let worldPos = center + rotatedPos;

  // Transform normal
  let worldNormal = quat_rotate(quaternion, localNormal);

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  out.normal = worldNormal;
  return out;
}`;

const fragmentShader = /*wgsl*/ `
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

const pickingVertexShader = /*wgsl*/ `
${cameraStruct}
${quaternionShaderFunctions}
${pickingVSOut}

@vertex
fn vs_main(
  // Geometry attributes
  @location(0) cornerPos: vec3<f32>,
  @location(1) localNormal: vec3<f32>,
  @location(2) beamOffset: vec3<f32>,
  @location(3) inset: vec3<f32>,
  // Instance attributes
  @location(4) center: vec3<f32>,
  @location(5) halfSize: vec3<f32>,
  @location(6) edgeSize: f32,
  @location(7) quaternion: vec4<f32>,
  @location(8) pickID: f32
) -> VSOut {
  let scaledCorner = cornerPos * halfSize;
  let scaledOffset = beamOffset * edgeSize;
  let scaledInset = inset * edgeSize;
  let localPos = scaledCorner + scaledOffset + scaledInset;

  let rotatedPos = quat_rotate(quaternion, localPos);
  let worldPos = center + rotatedPos;

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

// =============================================================================
// Custom Buffer Layouts
// =============================================================================

const RENDER_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [4, "float32x3"], // center
    [5, "float32x3"], // halfSize
    [6, "float32"], // edgeSize
    [7, "float32x4"], // quaternion
    [8, "float32x3"], // color
    [9, "float32"], // alpha
  ],
  "instance",
);

const PICKING_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [4, "float32x3"], // center
    [5, "float32x3"], // halfSize
    [6, "float32"], // edgeSize
    [7, "float32x4"], // quaternion
    [8, "float32"], // pickID
  ],
  "instance",
);

// =============================================================================
// Fill Functions
// =============================================================================

// Render: center(3) + halfSize(3) + edgeSize(1) + quat(4) + color(3) + alpha(1) = 15 floats
const FLOATS_PER_RENDER = 15;
// Picking: center(3) + halfSize(3) + edgeSize(1) + quat(4) + pickID(1) = 12 floats
const FLOATS_PER_PICKING = 12;

function fillRenderGeometry(
  schema: ProcessedSchema,
  constants: Record<string, unknown>,
  elem: BoundingBoxComponentConfig,
  elemIndex: number,
  out: Float32Array,
  outIndex: number,
): void {
  const o = outIndex * FLOATS_PER_RENDER;

  // Center
  out[o + 0] = elem.centers[elemIndex * 3 + 0];
  out[o + 1] = elem.centers[elemIndex * 3 + 1];
  out[o + 2] = elem.centers[elemIndex * 3 + 2];

  // Half size
  if (constants.half_size !== undefined) {
    const hs = constants.half_size;
    if (typeof hs === "number") {
      out[o + 3] = hs;
      out[o + 4] = hs;
      out[o + 5] = hs;
    } else {
      const arr = hs as number[];
      out[o + 3] = arr[0];
      out[o + 4] = arr[1];
      out[o + 5] = arr[2];
    }
  } else if (elem.half_sizes) {
    out[o + 3] = elem.half_sizes[elemIndex * 3 + 0];
    out[o + 4] = elem.half_sizes[elemIndex * 3 + 1];
    out[o + 5] = elem.half_sizes[elemIndex * 3 + 2];
  } else {
    out[o + 3] = 0.5;
    out[o + 4] = 0.5;
    out[o + 5] = 0.5;
  }

  // Edge size
  out[o + 6] = (constants.size as number) ?? elem.sizes?.[elemIndex] ?? 0.02;

  // Quaternion (input wxyz -> output xyzw for shader)
  if (constants.quaternion !== undefined) {
    const q = constants.quaternion as number[];
    out[o + 7] = q[1]; // x
    out[o + 8] = q[2]; // y
    out[o + 9] = q[3]; // z
    out[o + 10] = q[0]; // w
  } else if (elem.quaternions) {
    out[o + 7] = elem.quaternions[elemIndex * 4 + 1]; // x
    out[o + 8] = elem.quaternions[elemIndex * 4 + 2]; // y
    out[o + 9] = elem.quaternions[elemIndex * 4 + 3]; // z
    out[o + 10] = elem.quaternions[elemIndex * 4 + 0]; // w
  } else {
    out[o + 7] = 0; // x
    out[o + 8] = 0; // y
    out[o + 9] = 0; // z
    out[o + 10] = 1; // w (identity)
  }

  // Color
  if (constants.color) {
    const c = constants.color as number[];
    out[o + 11] = c[0];
    out[o + 12] = c[1];
    out[o + 13] = c[2];
  } else if (elem.colors) {
    out[o + 11] = elem.colors[elemIndex * 3 + 0];
    out[o + 12] = elem.colors[elemIndex * 3 + 1];
    out[o + 13] = elem.colors[elemIndex * 3 + 2];
  } else {
    out[o + 11] = 0.5;
    out[o + 12] = 0.5;
    out[o + 13] = 0.5;
  }

  // Alpha
  out[o + 14] = (constants.alpha as number) ?? elem.alphas?.[elemIndex] ?? 1.0;
}

function fillPickingGeometry(
  schema: ProcessedSchema,
  constants: Record<string, unknown>,
  elem: BoundingBoxComponentConfig,
  elemIndex: number,
  out: Float32Array,
  outIndex: number,
  baseID: number,
): void {
  const o = outIndex * FLOATS_PER_PICKING;

  // Center
  out[o + 0] = elem.centers[elemIndex * 3 + 0];
  out[o + 1] = elem.centers[elemIndex * 3 + 1];
  out[o + 2] = elem.centers[elemIndex * 3 + 2];

  // Half size
  if (constants.half_size !== undefined) {
    const hs = constants.half_size;
    if (typeof hs === "number") {
      out[o + 3] = hs;
      out[o + 4] = hs;
      out[o + 5] = hs;
    } else {
      const arr = hs as number[];
      out[o + 3] = arr[0];
      out[o + 4] = arr[1];
      out[o + 5] = arr[2];
    }
  } else if (elem.half_sizes) {
    out[o + 3] = elem.half_sizes[elemIndex * 3 + 0];
    out[o + 4] = elem.half_sizes[elemIndex * 3 + 1];
    out[o + 5] = elem.half_sizes[elemIndex * 3 + 2];
  } else {
    out[o + 3] = 0.5;
    out[o + 4] = 0.5;
    out[o + 5] = 0.5;
  }

  // Edge size
  out[o + 6] = (constants.size as number) ?? elem.sizes?.[elemIndex] ?? 0.02;

  // Quaternion (input wxyz -> output xyzw for shader)
  if (constants.quaternion !== undefined) {
    const q = constants.quaternion as number[];
    out[o + 7] = q[1];
    out[o + 8] = q[2];
    out[o + 9] = q[3];
    out[o + 10] = q[0];
  } else if (elem.quaternions) {
    out[o + 7] = elem.quaternions[elemIndex * 4 + 1];
    out[o + 8] = elem.quaternions[elemIndex * 4 + 2];
    out[o + 9] = elem.quaternions[elemIndex * 4 + 3];
    out[o + 10] = elem.quaternions[elemIndex * 4 + 0];
  } else {
    out[o + 7] = 0;
    out[o + 8] = 0;
    out[o + 9] = 0;
    out[o + 10] = 1;
  }

  // Pick ID
  out[o + 11] = packID(baseID + elemIndex);
}

// =============================================================================
// Helper Functions
// =============================================================================

function getElementCount(elem: BoundingBoxComponentConfig): number {
  const centers = elem.centers;
  return centers
    ? (Array.isArray(centers) ? centers.length : centers.length) / 3
    : 0;
}

function getCenters(elem: BoundingBoxComponentConfig): Float32Array {
  if (elem.centers instanceof Float32Array) {
    return elem.centers;
  }
  return new Float32Array(elem.centers);
}

// =============================================================================
// Primitive Definition
// =============================================================================

export const boundingBoxSpec = definePrimitive<BoundingBoxComponentConfig>({
  name: "BoundingBox",

  coerce: coerceBoundingBox,

  attributes: {
    center: attr.vec3("centers"),
    halfSize: attr.vec3("half_sizes", [0.5, 0.5, 0.5]),
    edgeSize: attr.f32("sizes", 0.02),
    rotation: attr.quat("quaternions"),
    color: attr.vec3("colors", [0.5, 0.5, 0.5]),
    alpha: attr.f32("alphas", 1.0),
  },

  geometry: { type: "custom", create: createWireframeBoxGeometry },

  // Custom layouts
  geometryLayout: GEOMETRY_LAYOUT,
  renderInstanceLayout: RENDER_INSTANCE_LAYOUT,
  pickingInstanceLayout: PICKING_INSTANCE_LAYOUT,

  // Custom shaders for wireframe box transform
  vertexShader,
  fragmentShader,
  pickingVertexShader,

  transform: "rigid", // Not used, but required
  shading: "lit",
  cullMode: "back",

  getElementCount,
  getCenters,
  fillRenderGeometry,
  fillPickingGeometry,
});
