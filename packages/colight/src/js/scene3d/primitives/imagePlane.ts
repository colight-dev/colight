/**
 * @module primitives/imagePlane
 * @description ImagePlane primitive using the declarative definition system.
 *
 * Renders a textured quad in 3D space with per-instance position/orientation/size.
 */

import { BaseComponentConfig } from "../types";
import {
  definePrimitive,
  attr,
  cameraStruct,
  pickingVSOut,
  pickingFragCode,
  quaternionShaderFunctions,
  coerceFloat32Fields,
} from "./define";

// =============================================================================
// Configuration Interface (internal format after coercion)
// =============================================================================

export type ImageSource =
  | ImageBitmap
  | HTMLImageElement
  | HTMLCanvasElement
  | ImageData
  | {
      data: Uint8Array | Uint8ClampedArray;
      width: number;
      height: number;
      channels?: number;
    };

export interface ImagePlaneComponentConfig extends BaseComponentConfig {
  type: "ImagePlane";
  /** Image source (ImageBitmap, ImageData, canvas, or raw pixels) */
  image: ImageSource;
  /** Optional key to force texture updates */
  imageKey?: string | number;
  /** Instance centers [x,y,z,...] */
  centers: Float32Array;
  /** Instance orientations [x,y,z,w,...] */
  quaternions?: Float32Array;
  /** Instance sizes [width,height,...] */
  sizes?: Float32Array;
  /** Default size (width,height) */
  size?: number | [number, number];
}

// =============================================================================
// Props Type (user-facing input)
// =============================================================================

export type ImagePlaneProps = Omit<
  ImagePlaneComponentConfig,
  "type" | "centers" | "quaternions" | "sizes"
> & {
  centers?: ArrayLike<number> | ArrayBufferView;
  center?: [number, number, number];
  position?: [number, number, number];
  quaternions?: ArrayLike<number> | ArrayBufferView;
  quaternion?: [number, number, number, number];
  sizes?: ArrayLike<number> | ArrayBufferView;
  size?: number | [number, number];
  width?: number;
  height?: number;
  opacity?: number;
};

// =============================================================================
// Coerce Function
// =============================================================================

export function coerceImagePlane(
  props: Record<string, any>,
): Record<string, any> {
  const {
    position,
    center,
    quaternion,
    width,
    height,
    opacity,
    centers,
    quaternions,
    sizes,
    size,
    alpha,
    ...rest
  } = props;

  // Resolve centers: centers > center > position > default
  const resolvedCenters =
    centers ?? (center ? [center] : position ? [position] : [[0, 0, 0]]);

  // Resolve quaternions: quaternions > quaternion
  const resolvedQuaternions =
    quaternions ?? (quaternion ? [quaternion] : undefined);

  // Resolve size: size > width/height combo
  const resolvedSize =
    size ??
    (width !== undefined || height !== undefined
      ? [width ?? 1, height ?? 1]
      : undefined);

  // Resolve alpha: alpha > opacity
  const resolvedAlpha = alpha !== undefined ? alpha : opacity;

  const result = {
    ...rest,
    centers: resolvedCenters,
    quaternions: resolvedQuaternions,
    sizes,
    size: resolvedSize,
    alpha: resolvedAlpha,
    type: "ImagePlane",
  };
  return coerceFloat32Fields(result, ["centers", "quaternions", "sizes"]);
}

// =============================================================================
// Shaders
// =============================================================================

const imagePlaneVertCode = /*wgsl*/ `
${cameraStruct}
${quaternionShaderFunctions}

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) alpha: f32,
};

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) rotation: vec4<f32>,
  @location(4) size: vec2<f32>,
  @location(5) color: vec3<f32>,
  @location(6) alpha: f32,
) -> VSOut {
  let scaledLocal = vec3<f32>(localPos.x * size.x, localPos.y * size.y, localPos.z);
  let worldPos = center + quat_rotate(rotation, scaledLocal);

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.uv = vec2<f32>(localPos.x + 0.5, 0.5 - localPos.y);
  out.color = color;
  out.alpha = alpha;
  return out;
}`;

const imagePlaneFragCode = /*wgsl*/ `
@group(1) @binding(0) var imageSampler: sampler;
@group(1) @binding(1) var imageTexture: texture_2d<f32>;

@fragment
fn fs_main(
  @location(0) uv: vec2<f32>,
  @location(1) color: vec3<f32>,
  @location(2) alpha: f32
) -> @location(0) vec4<f32> {
  let tex = textureSample(imageTexture, imageSampler, uv);
  return vec4<f32>(tex.rgb * color, tex.a * alpha);
}`;

const imagePlanePickingVertCode = /*wgsl*/ `
${cameraStruct}
${pickingVSOut}
${quaternionShaderFunctions}

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) rotation: vec4<f32>,
  @location(4) size: vec2<f32>,
  @location(5) pickID: f32
) -> VSOut {
  let scaledLocal = vec3<f32>(localPos.x * size.x, localPos.y * size.y, localPos.z);
  let worldPos = center + quat_rotate(rotation, scaledLocal);

  var out: VSOut;
  out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

// =============================================================================
// Bind group layout (sampler + texture)
// =============================================================================

const imageBindGroupLayoutCache = new WeakMap<GPUDevice, GPUBindGroupLayout>();

export function getImageBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
  const cached = imageBindGroupLayoutCache.get(device);
  if (cached) return cached;

  const layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      },
    ],
  });

  imageBindGroupLayoutCache.set(device, layout);
  return layout;
}

// =============================================================================
// Primitive Definition
// =============================================================================

const imageIdMap = new WeakMap<object, string>();
let imageIdCounter = 0;

function getImageObjectId(image: object): string {
  let id = imageIdMap.get(image);
  if (!id) {
    id = `image_${imageIdCounter++}`;
    imageIdMap.set(image, id);
  }
  return id;
}

export const imagePlaneSpec = definePrimitive<ImagePlaneComponentConfig>({
  name: "ImagePlane",

  coerce: coerceImagePlane,

  attributes: {
    position: attr.vec3("centers"),
    rotation: attr.quat("quaternions"),
    size: attr.vec2("sizes", [1, 1]),
    color: attr.vec3("colors", [1, 1, 1]),
    alpha: attr.f32("alphas", 1.0),
  },

  geometry: { type: "quad" },
  transform: "rigid",
  shading: "unlit",
  cullMode: "none",

  vertexShader: imagePlaneVertCode,
  fragmentShader: imagePlaneFragCode,
  pickingVertexShader: imagePlanePickingVertCode,
  bindGroupLayouts: (device, baseLayout) => [
    baseLayout,
    getImageBindGroupLayout(device),
  ],
});

imagePlaneSpec.getBatchKey = (elem) => {
  if (elem.imageKey !== undefined) return `key:${elem.imageKey}`;
  if (typeof elem.image === "string") return `url:${elem.image}`;
  if (elem.image && typeof elem.image === "object") {
    return `obj:${getImageObjectId(elem.image as object)}`;
  }
  return "image:unknown";
};
