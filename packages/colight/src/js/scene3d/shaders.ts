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
  tanHalfFov: f32,
  cameraUp: vec3<f32>,
  aspect: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
  cameraPos: vec3<f32>,
  _pad4: f32,
  cameraForward: vec3<f32>,
  _pad5: f32,
  viewportSize: vec2<f32>,
  _pad6: vec2<f32>,
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

export type ShaderFieldType = "f32" | "vec2" | "vec3" | "vec4";

export interface ShaderFieldDefinition {
  name: string;
  type: ShaderFieldType;
}

export type PrimitiveShadingModel = "unlit" | "lit";

export type PrimitiveTransformSpec =
  | {
      kind: "billboard";
      position: string;
      size: string;
      geometryPosition?: string;
      geometryNormal?: string;
    }
  | {
      kind: "rigid";
      position: string;
      size: string;
      quaternion: string;
      geometryPosition?: string;
      geometryNormal?: string;
    }
  | {
      kind: "beam";
      start: string;
      end: string;
      size: string;
      geometryPosition?: string;
      geometryNormal?: string;
    };

export interface PrimitiveShaderDefinition {
  geometryLayout: VertexBufferLayout;
  renderFields: ShaderFieldDefinition[];
  transform: PrimitiveTransformSpec;
  shading: PrimitiveShadingModel;
  instanceLocationStart?: number;
}

export interface GeneratedPrimitiveShaderProgram {
  geometryLayout: VertexBufferLayout;
  renderLayout: VertexBufferLayout;
  pickIDLayout: VertexBufferLayout;
  renderVertex: string;
  fragment: string;
  renderFloatsPerInstance: number;
  pickIDFloatsPerInstance: number;
  colorOffset: number;
  alphaOffset: number;
}

const SHADER_FIELD_INFO: Record<
  ShaderFieldType,
  { format: GPUVertexFormat; wgslType: string; floats: number }
> = {
  f32: { format: "float32", wgslType: "f32", floats: 1 },
  vec2: { format: "float32x2", wgslType: "vec2<f32>", floats: 2 },
  vec3: { format: "float32x3", wgslType: "vec3<f32>", floats: 3 },
  vec4: { format: "float32x4", wgslType: "vec4<f32>", floats: 4 },
};

function createRenderVSOut(shading: PrimitiveShadingModel): string {
  if (shading === "lit") {
    return /*wgsl*/ `
struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>,
  @location(4) pickID: f32
};`;
  }

  return /*wgsl*/ `
struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) pickID: f32
};`;
}

export const pickIDEncoding = /*wgsl*/ `
fn encodePickID(pickID: f32) -> vec4<f32> {
  let iID = u32(pickID);
  let r = f32(iID & 255u) / 255.0;
  let g = f32((iID >> 8u) & 255u) / 255.0;
  let b = f32((iID >> 16u) & 255u) / 255.0;
  return vec4<f32>(r, g, b, 1.0);
}`;

export const pickingEncoding = /*wgsl*/ `
${pickIDEncoding}

struct FSOut {
  @location(0) color: vec4<f32>,
  @location(1) pick: vec4<f32>
};`;

function createFragmentShader(shading: PrimitiveShadingModel): string {
  if (shading === "lit") {
    return /*wgsl*/ `
${cameraStruct}
${lightingConstants}
${lightingCalc}
${pickingEncoding}

@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) worldPos: vec3<f32>,
  @location(3) normal: vec3<f32>,
  @location(4) pickID: f32
)-> FSOut {
  let litColor = calculateLighting(color, normal, worldPos);
  var out: FSOut;
  out.color = vec4<f32>(litColor, alpha);
  out.pick = encodePickID(pickID);
  return out;
}`;
  }

  return /*wgsl*/ `
${pickingEncoding}

@fragment
fn fs_main(
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32,
  @location(2) pickID: f32
)-> FSOut {
  var out: FSOut;
  out.color = vec4<f32>(color, alpha);
  out.pick = encodePickID(pickID);
  return out;
}`;
}

function createVertexBufferLayoutFromFields(
  fields: ShaderFieldDefinition[],
  startLocation: number,
  stepMode: GPUVertexStepMode = "instance",
): VertexBufferLayout {
  return createVertexBufferLayout(
    fields.map((field, index) => [
      startLocation + index,
      SHADER_FIELD_INFO[field.type].format,
    ]),
    stepMode,
  );
}

function countFloats(fields: ShaderFieldDefinition[]): number {
  return fields.reduce(
    (sum, field) => sum + SHADER_FIELD_INFO[field.type].floats,
    0,
  );
}

function findFieldOffset(
  fields: ShaderFieldDefinition[],
  name: string,
): number {
  let offset = 0;
  for (const field of fields) {
    if (field.name === name) {
      return offset;
    }
    offset += SHADER_FIELD_INFO[field.type].floats;
  }
  throw new Error(`scene3d: shader field "${name}" is required`);
}

function findField(
  fields: ShaderFieldDefinition[],
  name: string,
): ShaderFieldDefinition {
  const field = fields.find((candidate) => candidate.name === name);
  if (!field) {
    throw new Error(`scene3d: shader field "${name}" is required`);
  }
  return field;
}

function buildFieldParameters(
  fields: ShaderFieldDefinition[],
  startLocation: number,
): string[] {
  return fields.map(
    (field, index) =>
      `@location(${startLocation + index}) ${field.name}: ${SHADER_FIELD_INFO[field.type].wgslType}`,
  );
}

function buildTransformCode(
  transform: PrimitiveTransformSpec,
  includeNormal: boolean,
): { helpers: string; code: string } {
  const geometryPosition = transform.geometryPosition ?? "localPos";
  const geometryNormal = transform.geometryNormal ?? "normal";

  switch (transform.kind) {
    case "billboard":
      return {
        helpers: "",
        code: /*wgsl*/ `
  let right = camera.cameraRight;
  let up = camera.cameraUp;
  let scaledRight = right * (${geometryPosition}.x * ${transform.size});
  let scaledUp = up * (${geometryPosition}.y * ${transform.size});
  let worldPos = ${transform.position} + scaledRight + scaledUp;
${includeNormal ? `  let worldNormal = ${geometryNormal};` : ""}`,
      };

    case "rigid":
      return {
        helpers: quaternionShaderFunctions,
        code: /*wgsl*/ `
  let scaledLocal = ${geometryPosition} * ${transform.size};
  let rotatedPos = quat_rotate(${transform.quaternion}, scaledLocal);
  let worldPos = ${transform.position} + rotatedPos;
${
  includeNormal
    ? `  let invScaledNorm = normalize(${geometryNormal} / ${transform.size});
  let worldNormal = quat_rotate(${transform.quaternion}, invScaledNorm);`
    : ""
}`,
      };

    case "beam":
      return {
        helpers: "",
        code: /*wgsl*/ `
  let segDir = ${transform.end} - ${transform.start};
  let segmentLength = max(length(segDir), 0.000001);
  let zDir = normalize(segDir);

  var tempUp = vec3<f32>(0.0, 0.0, 1.0);
  if (abs(dot(zDir, tempUp)) > 0.99) {
    tempUp = vec3<f32>(0.0, 1.0, 0.0);
  }
  let xDir = normalize(cross(zDir, tempUp));
  let yDir = cross(zDir, xDir);

  let localX = ${geometryPosition}.x * ${transform.size};
  let localY = ${geometryPosition}.y * ${transform.size};
  let localZ = ${geometryPosition}.z * segmentLength;
  let worldPos = ${transform.start}
    + xDir * localX
    + yDir * localY
    + zDir * localZ;
${
  includeNormal
    ? `  let worldNormal = normalize(
    xDir * ${geometryNormal}.x +
    yDir * ${geometryNormal}.y +
    zDir * ${geometryNormal}.z
  );`
    : ""
}`,
      };
  }
}

function buildRenderVertexShader(
  definition: PrimitiveShaderDefinition,
): string {
  const geometryPosition = definition.transform.geometryPosition ?? "localPos";
  const geometryNormal = definition.transform.geometryNormal ?? "normal";
  const instanceLocationStart = definition.instanceLocationStart ?? 2;
  const pickIDLocation = instanceLocationStart + definition.renderFields.length;
  const parameters = [
    `@location(0) ${geometryPosition}: vec3<f32>`,
    `@location(1) ${geometryNormal}: vec3<f32>`,
    ...buildFieldParameters(definition.renderFields, instanceLocationStart),
    `@location(${pickIDLocation}) pickID: f32`,
  ].join(",\n  ");
  const transform = buildTransformCode(
    definition.transform,
    definition.shading === "lit",
  );
  const colorField = findField(definition.renderFields, "color");
  const alphaField = findField(definition.renderFields, "alpha");
  const lightingAssignments =
    definition.shading === "lit"
      ? `
  out.worldPos = worldPos;
  out.normal = worldNormal;`
      : "";

  return /*wgsl*/ `
${cameraStruct}
${createRenderVSOut(definition.shading)}
${transform.helpers}

@vertex
fn vs_main(
  ${parameters}
)-> VSOut {
${transform.code}

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = ${colorField.name};
  out.alpha = ${alphaField.name};
  out.pickID = pickID;${lightingAssignments}
  return out;
}`;
}

export function generatePrimitiveShaderProgram(
  definition: PrimitiveShaderDefinition,
): GeneratedPrimitiveShaderProgram {
  const instanceLocationStart = definition.instanceLocationStart ?? 2;
  const renderLayout = createVertexBufferLayoutFromFields(
    definition.renderFields,
    instanceLocationStart,
  );
  const pickIDLayout = createVertexBufferLayout(
    [[instanceLocationStart + definition.renderFields.length, "float32"]],
    "instance",
  );

  return {
    geometryLayout: definition.geometryLayout,
    renderLayout,
    pickIDLayout,
    renderVertex: buildRenderVertexShader(definition),
    fragment: createFragmentShader(definition.shading),
    renderFloatsPerInstance: countFloats(definition.renderFields),
    pickIDFloatsPerInstance: 1,
    colorOffset: findFieldOffset(definition.renderFields, "color"),
    alphaOffset: findFieldOffset(definition.renderFields, "alpha"),
  };
}

// Common vertex buffer layouts
export const POINT_CLOUD_GEOMETRY_LAYOUT = createVertexBufferLayout([
  [0, "float32x3"],
  [1, "float32x3"],
]);

export const MESH_GEOMETRY_LAYOUT = createVertexBufferLayout([
  [0, "float32x3"],
  [1, "float32x3"],
]);

export const billboardShaderProgram = generatePrimitiveShaderProgram({
  geometryLayout: POINT_CLOUD_GEOMETRY_LAYOUT,
  renderFields: [
    { name: "position", type: "vec3" },
    { name: "size", type: "f32" },
    { name: "color", type: "vec3" },
    { name: "alpha", type: "f32" },
  ],
  transform: {
    kind: "billboard",
    position: "position",
    size: "size",
  },
  shading: "unlit",
});

export const rigidLitShaderProgram = generatePrimitiveShaderProgram({
  geometryLayout: MESH_GEOMETRY_LAYOUT,
  renderFields: [
    { name: "position", type: "vec3" },
    { name: "size", type: "vec3" },
    { name: "quaternion", type: "vec4" },
    { name: "color", type: "vec3" },
    { name: "alpha", type: "f32" },
  ],
  transform: {
    kind: "rigid",
    position: "position",
    size: "size",
    quaternion: "quaternion",
  },
  shading: "lit",
});

export const lineBeamShaderProgram = generatePrimitiveShaderProgram({
  geometryLayout: MESH_GEOMETRY_LAYOUT,
  renderFields: [
    { name: "startPos", type: "vec3" },
    { name: "endPos", type: "vec3" },
    { name: "size", type: "f32" },
    { name: "color", type: "vec3" },
    { name: "alpha", type: "f32" },
  ],
  transform: {
    kind: "beam",
    start: "startPos",
    end: "endPos",
    size: "size",
  },
  shading: "lit",
});

const ellipsoidImpostorVertCode = /*wgsl*/ `
${cameraStruct}

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) color: vec3<f32>,
  @location(1) @interpolate(flat) alpha: f32,
  @location(2) @interpolate(flat) center: vec3<f32>,
  @location(3) @interpolate(flat) halfSize: vec3<f32>,
  @location(4) @interpolate(flat) quaternion: vec4<f32>,
  @location(5) @interpolate(flat) pickID: f32
};

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) _normal: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) halfSize: vec3<f32>,
  @location(4) quaternion: vec4<f32>,
  @location(5) color: vec3<f32>,
  @location(6) alpha: f32,
  @location(7) pickID: f32
) -> VSOut {
  let radius = max(max(halfSize.x, halfSize.y), halfSize.z);
  let worldPos = center
    + camera.cameraRight * (localPos.x * radius * 2.0)
    + camera.cameraUp * (localPos.y * radius * 2.0);

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = color;
  out.alpha = alpha;
  out.center = center;
  out.halfSize = halfSize;
  out.quaternion = quaternion;
  out.pickID = pickID;
  return out;
}`;

const ellipsoidImpostorFragCode = /*wgsl*/ `
${cameraStruct}
${lightingConstants}
${lightingCalc}
${pickIDEncoding}
${quaternionShaderFunctions}

struct ImpostorFSOut {
  @location(0) color: vec4<f32>,
  @location(1) pick: vec4<f32>,
  @builtin(frag_depth) depth: f32
};

fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn cameraRayDirection(fragCoord: vec4<f32>) -> vec3<f32> {
  let clipX =
    ((fragCoord.x + 0.5) / camera.viewportSize.x) * 2.0 - 1.0;
  let clipY =
    1.0 - ((fragCoord.y + 0.5) / camera.viewportSize.y) * 2.0;
  let viewX = clipX * camera.aspect * camera.tanHalfFov;
  let viewY = clipY * camera.tanHalfFov;
  return normalize(
    camera.cameraForward + camera.cameraRight * viewX + camera.cameraUp * viewY
  );
}

fn intersectEllipsoid(
  rayOrigin: vec3<f32>,
  rayDir: vec3<f32>,
  center: vec3<f32>,
  halfSize: vec3<f32>,
  quaternion: vec4<f32>
) -> vec4<f32> {
  let safeHalfSize = max(halfSize, vec3<f32>(0.000001));
  let inverseQuaternion = quat_conjugate(quaternion);
  let localOrigin = quat_rotate(inverseQuaternion, rayOrigin - center);
  let localDir = quat_rotate(inverseQuaternion, rayDir);
  let scaledOrigin = localOrigin / safeHalfSize;
  let scaledDir = localDir / safeHalfSize;

  let a = dot(scaledDir, scaledDir);
  if (a <= 0.0000001) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  let b = 2.0 * dot(scaledOrigin, scaledDir);
  let c = dot(scaledOrigin, scaledOrigin) - 1.0;
  let discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  let sqrtDiscriminant = sqrt(discriminant);
  var t = (-b - sqrtDiscriminant) / (2.0 * a);
  if (t <= 0.0) {
    t = (-b + sqrtDiscriminant) / (2.0 * a);
  }

  if (t <= 0.0) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }

  let hitLocal = localOrigin + localDir * t;
  return vec4<f32>(hitLocal, t);
}

@fragment
fn fs_main(
  @builtin(position) fragCoord: vec4<f32>,
  @location(0) @interpolate(flat) color: vec3<f32>,
  @location(1) @interpolate(flat) alpha: f32,
  @location(2) @interpolate(flat) center: vec3<f32>,
  @location(3) @interpolate(flat) halfSize: vec3<f32>,
  @location(4) @interpolate(flat) quaternion: vec4<f32>,
  @location(5) @interpolate(flat) pickID: f32
) -> ImpostorFSOut {
  let rayOrigin = camera.cameraPos;
  let rayDir = cameraRayDirection(fragCoord);
  let hit = intersectEllipsoid(rayOrigin, rayDir, center, halfSize, quaternion);
  if (hit.w <= 0.0) {
    discard;
  }

  let safeHalfSize = max(halfSize, vec3<f32>(0.000001));
  let hitLocal = hit.xyz;
  let hitWorld = center + quat_rotate(quaternion, hitLocal);
  let worldNormal = normalize(
    quat_rotate(quaternion, hitLocal / (safeHalfSize * safeHalfSize))
  );
  let clipPos = camera.mvp * vec4<f32>(hitWorld, 1.0);
  let depth = clamp((clipPos.z / clipPos.w) * 0.5 + 0.5, 0.0, 1.0);

  var out: ImpostorFSOut;
  out.color = vec4<f32>(calculateLighting(color, worldNormal, hitWorld), alpha);
  out.pick = encodePickID(pickID);
  out.depth = depth;
  return out;
}`;

export const ellipsoidImpostorShaderProgram: GeneratedPrimitiveShaderProgram = {
  geometryLayout: POINT_CLOUD_GEOMETRY_LAYOUT,
  renderLayout: rigidLitShaderProgram.renderLayout,
  pickIDLayout: rigidLitShaderProgram.pickIDLayout,
  renderVertex: ellipsoidImpostorVertCode,
  fragment: ellipsoidImpostorFragCode,
  renderFloatsPerInstance: rigidLitShaderProgram.renderFloatsPerInstance,
  pickIDFloatsPerInstance: rigidLitShaderProgram.pickIDFloatsPerInstance,
  colorOffset: rigidLitShaderProgram.colorOffset,
  alphaOffset: rigidLitShaderProgram.alphaOffset,
};

export const billboardVertCode = billboardShaderProgram.renderVertex;
export const billboardFragCode = billboardShaderProgram.fragment;

export const ellipsoidVertCode = rigidLitShaderProgram.renderVertex;
export const ellipsoidFragCode = rigidLitShaderProgram.fragment;

export const cuboidVertCode = rigidLitShaderProgram.renderVertex;
export const cuboidFragCode = rigidLitShaderProgram.fragment;

export const lineBeamVertCode = lineBeamShaderProgram.renderVertex;
export const lineBeamFragCode = lineBeamShaderProgram.fragment;

export const POINT_CLOUD_INSTANCE_LAYOUT = billboardShaderProgram.renderLayout;
export const POINT_CLOUD_PICKING_INSTANCE_LAYOUT =
  billboardShaderProgram.pickIDLayout;

export const ELLIPSOID_INSTANCE_LAYOUT = rigidLitShaderProgram.renderLayout;
export const ELLIPSOID_PICKING_INSTANCE_LAYOUT =
  rigidLitShaderProgram.pickIDLayout;

export const LINE_BEAM_INSTANCE_LAYOUT = lineBeamShaderProgram.renderLayout;
export const LINE_BEAM_PICKING_INSTANCE_LAYOUT =
  lineBeamShaderProgram.pickIDLayout;

export const CUBOID_INSTANCE_LAYOUT = rigidLitShaderProgram.renderLayout;
export const CUBOID_PICKING_INSTANCE_LAYOUT =
  rigidLitShaderProgram.pickIDLayout;
