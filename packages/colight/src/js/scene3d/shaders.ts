import { VertexBufferLayout } from "./types";
import { quaternionShaderFunctions } from "./quaternion";

/**
 * Global lighting configuration for the 3D scene.
 * Uses a simple Blinn-Phong lighting model with ambient, diffuse, and specular components.
 */
export const LIGHTING = {
  /** Ambient light intensity, affects overall scene brightness */
  AMBIENT_INTENSITY: 0.55,

  /** Diffuse light intensity, affects surface shading based on light direction */
  DIFFUSE_INTENSITY: 0.6,

  /** Specular highlight intensity */
  SPECULAR_INTENSITY: 0.2,

  /** Specular power/shininess, higher values create sharper highlights */
  SPECULAR_POWER: 20.0,

  /** Light direction components relative to camera */
  DIRECTION: {
    /** Right component of light direction */
    RIGHT: 0.2,
    /** Up component of light direction */
    UP: 0.5,
    /** Forward component of light direction */
    FORWARD: 0,
  },
} as const;

// Common shader code templates
export const cameraStruct = /*wgsl*/ `
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
  cameraPos: vec3<f32>,
  _pad4: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;`;

export const lightingConstants = /*wgsl*/ `
const AMBIENT_INTENSITY = ${LIGHTING.AMBIENT_INTENSITY}f;
const DIFFUSE_INTENSITY = ${LIGHTING.DIFFUSE_INTENSITY}f;
const SPECULAR_INTENSITY = ${LIGHTING.SPECULAR_INTENSITY}f;
const SPECULAR_POWER = ${LIGHTING.SPECULAR_POWER}f;`;

export const lightingCalc = /*wgsl*/ `
fn calculateLighting(baseColor: vec3<f32>, normal: vec3<f32>, worldPos: vec3<f32>) -> vec3<f32> {
  let N = normalize(normal);
  let L = normalize(camera.lightDir);
  let V = normalize(camera.cameraPos - worldPos);

  let lambert = max(dot(N, L), 0.0);
  let ambient = AMBIENT_INTENSITY;
  var color = baseColor * (ambient + lambert * DIFFUSE_INTENSITY);

  let H = normalize(L + V);
  let spec = pow(max(dot(N, H), 0.0), SPECULAR_POWER);
  color += vec3<f32>(1.0) * spec * SPECULAR_INTENSITY;

  return color;
}`;

// Standardize VSOut struct for regular rendering
const standardVSOut = /*wgsl*/ `
struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
};`;

// Standardize VSOut struct for picking
export const pickingVSOut = /*wgsl*/ `
struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) pickID: f32
};`;

export const billboardVertCode = /*wgsl*/ `
${cameraStruct}
${standardVSOut}

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) position: vec3<f32>,
  @location(3) size: f32,
  @location(4) color: vec3<f32>,
  @location(5) alpha: f32
)-> VSOut {
  // Create camera-facing orientation
  let right = camera.cameraRight;
  let up = camera.cameraUp;

  // Transform quad vertices to world space
  let scaledRight = right * (localPos.x * size);
  let scaledUp = up * (localPos.y * size);
  let worldPos = position + scaledRight + scaledUp;

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  out.normal = normal;
  return out;
}`;

export const billboardPickingVertCode = /*wgsl*/ `
${cameraStruct}
${pickingVSOut}

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) position: vec3<f32>,
  @location(3) size: f32,
  @location(4) pickID: f32
)-> VSOut {
  // Create camera-facing orientation
  let right = camera.cameraRight;
  let up = camera.cameraUp;

  // Transform quad vertices to world space
  let scaledRight = right * (localPos.x * size);
  let scaledUp = up * (localPos.y * size);
  let worldPos = position + scaledRight + scaledUp;

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

export const billboardFragCode = /*wgsl*/ `
@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
)-> @location(0) vec4<f32> {
  return vec4<f32>(color, alpha);
}`;

export const ellipsoidVertCode = /*wgsl*/ `
${cameraStruct}
${standardVSOut}
${quaternionShaderFunctions}

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) position: vec3<f32>,
  @location(3) size: vec3<f32>,
  @location(4) quaternion: vec4<f32>,
  @location(5) color: vec3<f32>,
  @location(6) alpha: f32
)-> VSOut {
  // Scale local position
  let scaledLocal = localPos * size;

  // Apply rotation using quaternion
  let rotatedPos = quat_rotate(quaternion, scaledLocal);

  // Apply translation
  let worldPos = position + rotatedPos;

  // Transform normal - first normalize by size, then rotate by quaternion
  let invScaledNorm = normalize(normal / size);
  let rotatedNorm = quat_rotate(quaternion, invScaledNorm);

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  out.normal = rotatedNorm;
  return out;
}`;

export const ellipsoidPickingVertCode = /*wgsl*/ `
${cameraStruct}
${pickingVSOut}
${quaternionShaderFunctions}

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) position: vec3<f32>,
  @location(3) size: vec3<f32>,
  @location(4) quaternion: vec4<f32>,
  @location(5) pickID: f32
)-> VSOut {
  // Scale local position
  let scaledLocal = localPos * size;

  // Apply rotation using quaternion
  let rotatedPos = quat_rotate(quaternion, scaledLocal);

  // Apply translation
  let worldPos = position + rotatedPos;

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

export const ellipsoidFragCode = /*wgsl*/ `
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
)-> @location(0) vec4<f32> {
  let litColor = calculateLighting(color, normal, worldPos);
  return vec4<f32>(litColor, alpha);
}`;

export const cuboidVertCode = /*wgsl*/ `
${cameraStruct}
${standardVSOut}
${quaternionShaderFunctions}

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) position: vec3<f32>,
  @location(3) size: vec3<f32>,
  @location(4) quaternion: vec4<f32>,
  @location(5) color: vec3<f32>,
  @location(6) alpha: f32
)-> VSOut {
  // Scale local position
  let scaledLocal = localPos * size;

  // Apply rotation using quaternion
  let rotatedPos = quat_rotate(quaternion, scaledLocal);

  // Apply translation
  let worldPos = position + rotatedPos;

  // Transform normal - first normalize by size, then rotate by quaternion
  let invScaledNorm = normalize(normal / size);
  let rotatedNorm = quat_rotate(quaternion, invScaledNorm);

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  out.normal = rotatedNorm;
  return out;
}`;

export const cuboidPickingVertCode = /*wgsl*/ `
${cameraStruct}
${pickingVSOut}
${quaternionShaderFunctions}

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) position: vec3<f32>,
  @location(3) size: vec3<f32>,
  @location(4) quaternion: vec4<f32>,
  @location(5) pickID: f32
)-> VSOut {
  // Scale local position
  let scaledLocal = localPos * size;

  // Apply rotation using quaternion
  let rotatedPos = quat_rotate(quaternion, scaledLocal);

  // Apply translation
  let worldPos = position + rotatedPos;

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

export const cuboidFragCode = /*wgsl*/ `
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
)-> @location(0) vec4<f32> {
  let litColor = calculateLighting(color, normal, worldPos);
  return vec4<f32>(litColor, alpha);
}`;

export const lineBeamVertCode = /*wgsl*/ `
${cameraStruct}
${standardVSOut}

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) startPos: vec3<f32>,
  @location(3) endPos: vec3<f32>,
  @location(4) size: f32,
  @location(5) color: vec3<f32>,
  @location(6) alpha: f32
)-> VSOut {
  let segDir = endPos - startPos;
  let length = max(length(segDir), 0.000001);
  let zDir = normalize(segDir);

  // Build basis vectors
  var tempUp = vec3<f32>(0,0,1);
  if (abs(dot(zDir, tempUp)) > 0.99) {
    tempUp = vec3<f32>(0,1,0);
  }
  let xDir = normalize(cross(zDir, tempUp));
  let yDir = cross(zDir, xDir);

  // Transform to world space
  let localX = localPos.x * size;
  let localY = localPos.y * size;
  let localZ = localPos.z * length;
  let worldPos = startPos
    + xDir * localX
    + yDir * localY
    + zDir * localZ;

  // Transform normal to world space
  let worldNorm = normalize(
    xDir * normal.x +
    yDir * normal.y +
    zDir * normal.z
  );

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  out.normal = worldNorm;
  return out;
}`;

export const lineBeamFragCode = /*wgsl*/ `
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>
)-> @location(0) vec4<f32> {
  let litColor = calculateLighting(color, normal, worldPos);
  return vec4<f32>(litColor, alpha);
}`;

export const lineBeamPickingVertCode = /*wgsl*/ `
${cameraStruct}
${pickingVSOut}

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) startPos: vec3<f32>,
  @location(3) endPos: vec3<f32>,
  @location(4) size: f32,
  @location(5) pickID: f32
)-> VSOut {
  let segDir = endPos - startPos;
  let length = max(length(segDir), 0.000001);
  let zDir = normalize(segDir);

  // Build basis vectors
  var tempUp = vec3<f32>(0,0,1);
  if (abs(dot(zDir, tempUp)) > 0.99) {
    tempUp = vec3<f32>(0,1,0);
  }
  let xDir = normalize(cross(zDir, tempUp));
  let yDir = cross(zDir, xDir);

  // Transform to world space
  let localX = localPos.x * size;
  let localY = localPos.y * size;
  let localZ = localPos.z * length;
  let worldPos = startPos
    + xDir * localX
    + yDir * localY
    + zDir * localZ;

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

export const pickingFragCode = /*wgsl*/ `
@fragment
fn fs_pick(@location(0) pickID: f32)-> @location(0) vec4<f32> {
  let iID = u32(pickID);
  let r = f32(iID & 255u)/255.0;
  let g = f32((iID>>8)&255u)/255.0;
  let b = f32((iID>>16)&255u)/255.0;
  return vec4<f32>(r,g,b,1.0);
}`;

// Helper function to create vertex buffer layouts
export function createVertexBufferLayout(
  attributes: Array<[number, GPUVertexFormat]>,
  stepMode: GPUVertexStepMode = "vertex",
): VertexBufferLayout {
  let offset = 0;
  const formattedAttrs = attributes.map(([location, format]) => {
    const attr = {
      shaderLocation: location,
      offset,
      format,
    };
    // Add to offset based on format size
    offset += format.includes("x4")
      ? 16
      : format.includes("x3")
        ? 12
        : format.includes("x2")
          ? 8
          : 4;
    return attr;
  });

  return {
    arrayStride: offset,
    stepMode,
    attributes: formattedAttrs,
  };
}

// Common vertex buffer layouts
export const POINT_CLOUD_GEOMETRY_LAYOUT = createVertexBufferLayout([
  [0, "float32x3"], // center
  [1, "float32x3"], // normal
]);

export const POINT_CLOUD_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [2, "float32x3"], // center
    [3, "float32"], // size
    [4, "float32x3"], // color
    [5, "float32"], // alpha
  ],
  "instance",
);

export const POINT_CLOUD_PICKING_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [2, "float32x3"], // center
    [3, "float32"], // size
    [4, "float32"], // pickID
  ],
  "instance",
);

export const MESH_GEOMETRY_LAYOUT = createVertexBufferLayout([
  [0, "float32x3"], // position
  [1, "float32x3"], // normal
]);

export const ELLIPSOID_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [2, "float32x3"], // position
    [3, "float32x3"], // size
    [4, "float32x4"], // quaternion (quaternion)
    [5, "float32x3"], // color
    [6, "float32"], // alpha
  ],
  "instance",
);

export const ELLIPSOID_PICKING_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [2, "float32x3"], // position
    [3, "float32x3"], // size
    [4, "float32x4"], // quaternion (quaternion)
    [5, "float32"], // pickID
  ],
  "instance",
);

export const LINE_BEAM_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [2, "float32x3"], // startPos (position1)
    [3, "float32x3"], // endPos (position2)
    [4, "float32"], // size
    [5, "float32x3"], // color
    [6, "float32"], // alpha
  ],
  "instance",
);

export const LINE_BEAM_PICKING_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [2, "float32x3"], // startPos (position1)
    [3, "float32x3"], // endPos (position2)
    [4, "float32"], // size
    [5, "float32"], // pickID
  ],
  "instance",
);

export const CUBOID_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [2, "float32x3"], // position
    [3, "float32x3"], // size
    [4, "float32x4"], // quaternion (quaternion)
    [5, "float32x3"], // color
    [6, "float32"], // alpha
  ],
  "instance",
);

export const CUBOID_PICKING_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [2, "float32x3"], // position
    [3, "float32x3"], // size
    [4, "float32x4"], // quaternion (quaternion)
    [5, "float32"], // pickID
  ],
  "instance",
);

// =============================================================================
// Hover Outline Shaders
// =============================================================================

/**
 * Post-process vertex shader for outline rendering.
 * Uses the full-screen triangle trick (no vertex buffer needed).
 */
export const outlineVertCode = /*wgsl*/ `
struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  // Full-screen triangle: 3 vertices that cover the entire screen
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var out: VSOut;
  out.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
  out.uv = (pos[vertexIndex] + 1.0) * 0.5;
  out.uv.y = 1.0 - out.uv.y; // Flip Y for texture coordinates
  return out;
}`;

/**
 * Post-process fragment shader for outline edge detection.
 * Samples the silhouette texture and draws an outline at edges.
 */
export const outlineFragCode = /*wgsl*/ `
@group(0) @binding(0) var outlineSampler: sampler;
@group(0) @binding(1) var outlineTexture: texture_2d<f32>;

struct OutlineParams {
  outlineColor: vec3<f32>,
  outlineWidth: f32,
  texelSize: vec2<f32>,
  _pad: vec2<f32>,
};
@group(0) @binding(2) var<uniform> params: OutlineParams;

@fragment
fn fs_outline(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  let center = textureSample(outlineTexture, outlineSampler, uv).a;

  // Sample neighbors for edge detection (4-connected)
  var edge = 0.0;
  let step = params.texelSize * params.outlineWidth;

  // Sample 4 neighbors
  let left = textureSample(outlineTexture, outlineSampler, uv + vec2<f32>(-step.x, 0.0)).a;
  let right = textureSample(outlineTexture, outlineSampler, uv + vec2<f32>(step.x, 0.0)).a;
  let up = textureSample(outlineTexture, outlineSampler, uv + vec2<f32>(0.0, -step.y)).a;
  let down = textureSample(outlineTexture, outlineSampler, uv + vec2<f32>(0.0, step.y)).a;

  edge = max(edge, abs(center - left));
  edge = max(edge, abs(center - right));
  edge = max(edge, abs(center - up));
  edge = max(edge, abs(center - down));

  // Draw outline where there's an edge and we're outside the object
  if (edge > 0.1 && center < 0.5) {
    return vec4<f32>(params.outlineColor, 1.0);
  }

  // Return fully transparent to not affect existing content (blend will handle this)
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}`;
