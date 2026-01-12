/// <reference types="react" />

import * as glMatrix from "gl-matrix";
import React, {
  // DO NOT require MouseEvent
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { throttle, deepEqualModuloTypedArrays } from "./utils";
import { useCanvasSnapshot } from "./canvasSnapshot";
import {
  CameraParams,
  CameraState,
  createCameraParams,
  createCameraState,
  dolly,
  adjustFov,
  orbit,
  pan,
  roll,
  zoom,
  DraggingState,
  hasCameraMoved,
  getProjectionMatrix,
  getViewMatrix,
} from "./camera3d";

// @ts-ignore - lodash-es types are available via @types/lodash-es
import isEqual from "lodash-es/isEqual";

import {
  ComponentConfig,
  boundingBoxSpec,
  cuboidSpec,
  ellipsoidSpec,
  ellipsoidAxesSpec,
  lineBeamsSpec,
  lineSegmentsSpec,
  imagePlaneSpec,
  pointCloudSpec,
  buildPickingData,
  buildRenderData,
  createOverlayPipeline,
  getImageBindGroupLayout,
  ImagePlaneComponentConfig,
  ImageSource,
} from "./components";
import { unpackID } from "./picking";
import { LIGHTING, outlineVertCode, outlineFragCode } from "./shaders";
import {
  BufferInfo,
  GeometryResources,
  GeometryResource,
  PrimitiveSpec,
  RenderObject,
  PipelineCacheEntry,
  DynamicBuffers,
  RenderObjectCache,
  ComponentOffset,
  ReadyState,
  NOOP_READY_STATE,
  PickInfo,
  DragInfo,
} from "./types";
import {
  resolveConstraint,
  computeConstrainedPosition,
  hasDragCallbacks,
  ResolvedConstraint,
} from "./drag";
import { buildPickInfo } from "./pick-info";
import { screenRay } from "./project";

/**
 * Aligns a size or offset to 16 bytes, which is a common requirement for WebGPU buffers.
 * @param value The value to align
 * @returns The value aligned to the next 16-byte boundary
 */
function align16(value: number): number {
  return Math.ceil(value / 16) * 16;
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

interface ImageResource {
  texture: GPUTexture;
  view: GPUTextureView;
  sampler: GPUSampler;
  bindGroup: GPUBindGroup;
  width: number;
  height: number;
  imageKey?: string | number;
}

type ImageUpload =
  | {
      kind: "external";
      source: GPUCopyExternalImageSource;
      width: number;
      height: number;
    }
  | {
      kind: "raw";
      data: Uint8Array;
      width: number;
      height: number;
      bytesPerRow: number;
    };

const imageSamplerCache = new WeakMap<GPUDevice, GPUSampler>();

function getImageSampler(device: GPUDevice): GPUSampler {
  const cached = imageSamplerCache.get(device);
  if (cached) return cached;
  const sampler = device.createSampler({
    magFilter: "linear",
    minFilter: "linear",
  });
  imageSamplerCache.set(device, sampler);
  return sampler;
}

function asUint8Array(data: Uint8Array | Uint8ClampedArray): Uint8Array {
  return data instanceof Uint8Array
    ? data
    : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function expandRgbToRgba(
  data: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  const pixelCount = width * height;
  const rgba = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i++) {
    rgba[i * 4] = data[i * 3];
    rgba[i * 4 + 1] = data[i * 3 + 1];
    rgba[i * 4 + 2] = data[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

function padImageRows(
  data: Uint8Array,
  width: number,
  height: number,
): { data: Uint8Array; bytesPerRow: number } {
  const bytesPerRow = width * 4;
  const aligned = alignTo(bytesPerRow, 256);
  if (aligned === bytesPerRow) {
    return { data, bytesPerRow };
  }
  const padded = new Uint8Array(aligned * height);
  for (let row = 0; row < height; row++) {
    const srcStart = row * bytesPerRow;
    const srcEnd = srcStart + bytesPerRow;
    padded.set(data.subarray(srcStart, srcEnd), row * aligned);
  }
  return { data: padded, bytesPerRow: aligned };
}

function isExternalImageSource(
  image: unknown,
): image is GPUCopyExternalImageSource {
  if (!image || typeof image !== "object") return false;
  if (typeof ImageBitmap !== "undefined" && image instanceof ImageBitmap) {
    return true;
  }
  if (
    typeof HTMLImageElement !== "undefined" &&
    image instanceof HTMLImageElement
  ) {
    return true;
  }
  if (
    typeof HTMLCanvasElement !== "undefined" &&
    image instanceof HTMLCanvasElement
  ) {
    return true;
  }
  if (
    typeof OffscreenCanvas !== "undefined" &&
    image instanceof OffscreenCanvas
  ) {
    return true;
  }
  return false;
}

function getExternalImageSize(
  image: GPUCopyExternalImageSource,
): { width: number; height: number } {
  const anyImage = image as any;
  const width =
    typeof anyImage.naturalWidth === "number" && anyImage.naturalWidth > 0
      ? anyImage.naturalWidth
      : anyImage.width ?? 0;
  const height =
    typeof anyImage.naturalHeight === "number" && anyImage.naturalHeight > 0
      ? anyImage.naturalHeight
      : anyImage.height ?? 0;
  return { width, height };
}

function isImageDataLike(
  image: unknown,
): image is {
  data: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
  channels?: number;
} {
  return (
    !!image &&
    typeof image === "object" &&
    "data" in image &&
    "width" in image &&
    "height" in image
  );
}

function getImageUpload(image: ImageSource): ImageUpload | null {
  if (isImageDataLike(image)) {
    const width = image.width;
    const height = image.height;
    if (!width || !height) {
      console.warn("ImagePlane: invalid image dimensions.", image);
      return null;
    }
    const data = asUint8Array(image.data);
    const pixelCount = width * height;
    const channels =
      image.channels ?? Math.floor(data.length / Math.max(pixelCount, 1));
    if (channels !== 3 && channels !== 4) {
      console.warn("ImagePlane: unsupported channel count.", channels);
      return null;
    }
    const rgba = channels === 4 ? data : expandRgbToRgba(data, width, height);
    const padded = padImageRows(rgba, width, height);
    return {
      kind: "raw",
      data: padded.data,
      width,
      height,
      bytesPerRow: padded.bytesPerRow,
    };
  }

  if (isExternalImageSource(image)) {
    const { width, height } = getExternalImageSize(image);
    if (!width || !height) {
      console.warn("ImagePlane: external image has no size.", image);
      return null;
    }
    return {
      kind: "external",
      source: image,
      width,
      height,
    };
  }

  console.warn("ImagePlane: unsupported image type.", image);
  return null;
}

function uploadImageToTexture(
  device: GPUDevice,
  texture: GPUTexture,
  upload: ImageUpload,
): void {
  if (upload.kind === "external") {
    device.queue.copyExternalImageToTexture(
      { source: upload.source },
      { texture },
      { width: upload.width, height: upload.height },
    );
    return;
  }

  device.queue.writeTexture(
    { texture },
    upload.data,
    { bytesPerRow: upload.bytesPerRow, rowsPerImage: upload.height },
    { width: upload.width, height: upload.height },
  );
}

function ensureImageResource(
  device: GPUDevice,
  resources: Map<string, ImageResource>,
  key: string,
  image: ImageSource,
  imageKey?: string | number,
): ImageResource | null {
  const upload = getImageUpload(image);
  if (!upload) return null;

  const existing = resources.get(key);
  const needsRecreate =
    !existing ||
    existing.width !== upload.width ||
    existing.height !== upload.height;
  const needsUpdate =
    !existing || (imageKey !== undefined && existing.imageKey !== imageKey);

  if (needsRecreate) {
    existing?.texture.destroy();
    const texture = device.createTexture({
      size: [upload.width, upload.height, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    uploadImageToTexture(device, texture, upload);
    const view = texture.createView();
    const sampler = getImageSampler(device);
    const bindGroup = device.createBindGroup({
      layout: getImageBindGroupLayout(device),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: view },
      ],
    });
    const resource: ImageResource = {
      texture,
      view,
      sampler,
      bindGroup,
      width: upload.width,
      height: upload.height,
      imageKey,
    };
    resources.set(key, resource);
    return resource;
  }

  if (needsUpdate && existing) {
    uploadImageToTexture(device, existing.texture, upload);
    existing.imageKey = imageKey;
  }

  return existing ?? null;
}

export interface SceneImplProps {
  /** Array of 3D components to render in the scene */
  components: ComponentConfig[];

  /** Width of the container in pixels */
  containerWidth: number;

  /** Height of the container in pixels */
  containerHeight: number;

  /** Optional CSS styles to apply to the canvas */
  style?: React.CSSProperties;

  /** Optional controlled camera state. If provided, the component becomes controlled */
  camera?: CameraParams;

  /** Default camera configuration used when uncontrolled */
  defaultCamera?: CameraParams;

  /** Callback fired when camera parameters change */
  onCameraChange?: (camera: CameraParams) => void;

  /** Callback fired after each frame render with the render time in milliseconds */
  onFrameRendered?: (renderTime: number) => void;

  /** Callback to fire when scene is initially ready */
  onReady?: () => void;

  /** Scene-level hover callback. Called with PickInfo when hovering, null when not. */
  onHover?: (event: PickInfo | null) => void;

  /** Scene-level click callback. Called with PickInfo when an element is clicked. */
  onClick?: (event: PickInfo) => void;

  /** Optional ready state for coordinating updates. Defaults to NOOP_READY_STATE. */
  readyState?: ReadyState;

  /** Optional map of custom primitive specifications */
  primitiveSpecs?: Record<string, PrimitiveSpec<any>>;
}

function initGeometryResources(
  device: GPUDevice,
  resources: GeometryResources,
  registry: Record<string, PrimitiveSpec<any>>,
) {
  // Create geometry for each primitive type
  for (const [primitiveName, spec] of Object.entries(registry)) {
    // Only create if not already present or geometryKey changed
    // Cast to any since resources is typed with known keys but we might have custom ones
    const existing = (resources as any)[primitiveName] as
      | GeometryResource
      | null
      | undefined;
    const specKey = (spec as any).geometryKey;
    if (!existing || (specKey !== undefined && existing.geometryKey !== specKey)) {
      const resource = spec.createGeometryResource(device);
      (resource as any).geometryKey = specKey;
      (resources as any)[primitiveName] = resource;
    }
  }
}

const defaultPrimitiveRegistry: Record<
  ComponentConfig["type"],
  PrimitiveSpec<any>
> = {
  PointCloud: pointCloudSpec,
  Ellipsoid: ellipsoidSpec,
  EllipsoidAxes: ellipsoidAxesSpec,
  Cuboid: cuboidSpec,
  LineBeams: lineBeamsSpec,
  LineSegments: lineSegmentsSpec,
  ImagePlane: imagePlaneSpec,
  BoundingBox: boundingBoxSpec,
};

const DEFAULT_OUTLINE_COLOR: [number, number, number] = [1, 1, 1];
const DEFAULT_OUTLINE_WIDTH = 2;

type OutlineConfig = {
  outline?: boolean;
  outlineColor?: [number, number, number];
  outlineWidth?: number;
};

type OutlineTarget =
  | { kind: "component"; componentIdx: number }
  | { kind: "element"; componentIdx: number; elementIdx: number };

type OutlineStyle = {
  color: [number, number, number];
  width: number;
};

type OutlineGroup = {
  style: OutlineStyle;
  targets: OutlineTarget[];
};

function hasOutlineConfig(config?: OutlineConfig): boolean {
  return (
    !!config &&
    (config.outline !== undefined ||
      config.outlineColor !== undefined ||
      config.outlineWidth !== undefined)
  );
}

function isOutlineEnabled(config?: OutlineConfig): boolean {
  if (!config) return false;
  if (config.outline !== undefined) return config.outline;
  return config.outlineColor !== undefined || config.outlineWidth !== undefined;
}

function resolveOutlineStyle(
  config?: OutlineConfig,
  fallback?: OutlineConfig,
): {
  enabled: boolean;
  color: [number, number, number];
  width: number;
} {
  const enabled = isOutlineEnabled(config ?? fallback);
  if (!enabled) {
    return {
      enabled: false,
      color: DEFAULT_OUTLINE_COLOR,
      width: DEFAULT_OUTLINE_WIDTH,
    };
  }

  return {
    enabled: true,
    color:
      config?.outlineColor ?? fallback?.outlineColor ?? DEFAULT_OUTLINE_COLOR,
    width:
      config?.outlineWidth ?? fallback?.outlineWidth ?? DEFAULT_OUTLINE_WIDTH,
  };
}

function resolveComponentOutline(component?: ComponentConfig) {
  return resolveOutlineStyle(component);
}

function resolveHoverOutline(component?: ComponentConfig) {
  if (!component) {
    return {
      enabled: false,
      color: DEFAULT_OUTLINE_COLOR,
      width: DEFAULT_OUTLINE_WIDTH,
    };
  }
  if (!hasOutlineConfig(component.hoverProps)) {
    return {
      enabled: false,
      color: DEFAULT_OUTLINE_COLOR,
      width: DEFAULT_OUTLINE_WIDTH,
    };
  }
  return resolveOutlineStyle(component.hoverProps, component);
}

function outlineStyleKey(color: [number, number, number], width: number) {
  return `${color[0]}:${color[1]}:${color[2]}:${width}`;
}

function sameOutlineStyle(
  a: { color: [number, number, number]; width: number },
  b: { color: [number, number, number]; width: number },
) {
  return (
    a.width === b.width &&
    a.color[0] === b.color[0] &&
    a.color[1] === b.color[1] &&
    a.color[2] === b.color[2]
  );
}

function collectOutlineGroups(
  components: ComponentConfig[],
  hoverState: { componentIdx: number; elementIdx: number } | null,
): OutlineGroup[] {
  const groups = new Map<string, OutlineGroup>();

  components.forEach((component, componentIdx) => {
    const outline = resolveComponentOutline(component);
    if (!outline.enabled) return;

    const key = outlineStyleKey(outline.color, outline.width);
    const existing = groups.get(key);
    const target: OutlineTarget = { kind: "component", componentIdx };
    if (existing) {
      existing.targets.push(target);
    } else {
      groups.set(key, {
        style: { color: outline.color, width: outline.width },
        targets: [target],
      });
    }
  });

  if (hoverState) {
    const hoveredComponent = components[hoverState.componentIdx];
    const hoverOutline = resolveHoverOutline(hoveredComponent);
    if (hoverOutline.enabled) {
      const baseOutline = resolveComponentOutline(hoveredComponent);
      const shouldAdd =
        !baseOutline.enabled ||
        !sameOutlineStyle(baseOutline, hoverOutline);
      if (shouldAdd) {
        const key = outlineStyleKey(hoverOutline.color, hoverOutline.width);
        const existing = groups.get(key);
        const target: OutlineTarget = {
          kind: "element",
          componentIdx: hoverState.componentIdx,
          elementIdx: hoverState.elementIdx,
        };
        if (existing) {
          existing.targets.push(target);
        } else {
          groups.set(key, {
            style: {
              color: hoverOutline.color,
              width: hoverOutline.width,
            },
            targets: [target],
          });
        }
      }
    }
  }

  return Array.from(groups.values());
}

function ensurePickingData(
  device: GPUDevice,
  components: ComponentConfig[],
  ro: RenderObject,
) {
  if (!ro.pickingDataStale) return;

  const { pickingData, componentOffsets, spec, sortedPositions } = ro;

  let dataOffset = 0;
  for (let i = 0; i < componentOffsets.length; i++) {
    const offset = componentOffsets[i];
    const component = components[offset.componentIdx];
    const floatsPerInstance = spec.floatsPerPicking;
    const componentFloats =
      offset.elementCount * spec.instancesPerElement * floatsPerInstance;
    buildPickingData(
      component,
      spec,
      pickingData,
      offset.pickingStart,
      offset.elementStart,
      sortedPositions,
    );

    dataOffset += componentFloats;
  }

  // Write picking data to GPU

  device.queue.writeBuffer(
    ro.pickingInstanceBuffer.buffer,
    ro.pickingInstanceBuffer.offset,
    pickingData.buffer,
    pickingData.byteOffset,
    pickingData.byteLength,
  );

  ro.pickingDataStale = false;
}

function computeUniforms(
  containerWidth: number,
  containerHeight: number,
  camState: CameraState,
): {
  aspect: number;
  view: glMatrix.mat4;
  proj: glMatrix.mat4;
  mvp: glMatrix.mat4;
  forward: glMatrix.vec3;
  right: glMatrix.vec3;
  camUp: glMatrix.vec3;
  lightDir: glMatrix.vec3;
} {
  const aspect = containerWidth / containerHeight;
  const view = glMatrix.mat4.lookAt(
    glMatrix.mat4.create(),
    camState.position,
    camState.target,
    camState.up,
  );

  const proj = glMatrix.mat4.perspective(
    glMatrix.mat4.create(),
    glMatrix.glMatrix.toRadian(camState.fov),
    aspect,
    camState.near,
    camState.far,
  );

  const mvp = glMatrix.mat4.multiply(glMatrix.mat4.create(), proj, view);

  // Compute camera vectors for lighting
  const forward = glMatrix.vec3.sub(
    glMatrix.vec3.create(),
    camState.target,
    camState.position,
  );
  const right = glMatrix.vec3.cross(
    glMatrix.vec3.create(),
    forward,
    camState.up,
  );
  glMatrix.vec3.normalize(right, right);

  const camUp = glMatrix.vec3.cross(glMatrix.vec3.create(), right, forward);
  glMatrix.vec3.normalize(camUp, camUp);
  glMatrix.vec3.normalize(forward, forward);

  // Compute light direction in camera space
  const lightDir = glMatrix.vec3.create();
  glMatrix.vec3.scaleAndAdd(
    lightDir,
    lightDir,
    right,
    LIGHTING.DIRECTION.RIGHT,
  );
  glMatrix.vec3.scaleAndAdd(lightDir, lightDir, camUp, LIGHTING.DIRECTION.UP);
  glMatrix.vec3.scaleAndAdd(
    lightDir,
    lightDir,
    forward,
    LIGHTING.DIRECTION.FORWARD,
  );
  glMatrix.vec3.normalize(lightDir, lightDir);

  return { aspect, view, proj, mvp, forward, right, camUp, lightDir };
}

function encodeScenePass({
  commandEncoder,
  targetView,
  depthTexture,
  renderObjects,
  uniformBindGroup,
}: {
  commandEncoder: GPUCommandEncoder;
  targetView: GPUTextureView;
  depthTexture: GPUTexture | null;
  renderObjects: RenderObject[];
  uniformBindGroup: GPUBindGroup;
}) {
  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [
      {
        view: targetView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
    depthStencilAttachment: depthTexture
      ? {
          view: depthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        }
      : undefined,
  });

  // Draw each object
  for (const ro of renderObjects) {
    pass.setPipeline(ro.pipeline);
    pass.setBindGroup(0, uniformBindGroup);
    if (ro.textureBindGroup) {
      pass.setBindGroup(1, ro.textureBindGroup);
    }
    pass.setVertexBuffer(0, ro.geometryBuffer);
    pass.setVertexBuffer(1, ro.instanceBuffer.buffer, ro.instanceBuffer.offset);
    if (ro.indexBuffer && ro.indexCount > 0) {
      pass.setIndexBuffer(ro.indexBuffer, ro.indexFormat ?? "uint16");
      pass.drawIndexed(ro.indexCount, ro.instanceCount);
    } else if (ro.vertexCount > 0) {
      pass.draw(ro.vertexCount, ro.instanceCount);
    }
  }

  pass.end();
}

function computeUniformData(
  containerWidth: number,
  containerHeight: number,
  camState: CameraState,
): Float32Array {
  const { mvp, right, camUp, lightDir } = computeUniforms(
    containerWidth,
    containerHeight,
    camState,
  );
  return new Float32Array([
    ...Array.from(mvp),
    right[0],
    right[1],
    right[2],
    0, // pad to vec4
    camUp[0],
    camUp[1],
    camUp[2],
    0, // pad to vec4
    lightDir[0],
    lightDir[1],
    lightDir[2],
    0, // pad to vec4
    camState.position[0],
    camState.position[1],
    camState.position[2],
    0, // Add camera position
  ]);
}

function updateInstanceSorting(
  ro: RenderObject,
  components: ComponentConfig[],
  cameraPos: glMatrix.vec3,
): Uint32Array | undefined {
  if (!ro.hasAlphaComponents) return undefined;

  const [camX, camY, camZ] = cameraPos;
  let globalIdx = 0;

  // Fill distances and init sortedIndices
  for (let i = 0; i < ro.componentOffsets.length; i++) {
    const offset = ro.componentOffsets[i];
    const component = components[offset.componentIdx];
    // Access the centers
    const centers = ro.spec.getCenters(component);
    const { elementCount } = offset;

    for (let j = 0; j < elementCount; j++) {
      const baseIdx = j * 3;
      const dx = centers[baseIdx] - camX;
      const dy = centers[baseIdx + 1] - camY;
      const dz = centers[baseIdx + 2] - camZ;
      ro.distances![globalIdx] = dx * dx + dy * dy + dz * dz;

      ro.sortedIndices![globalIdx] = globalIdx;
      globalIdx++;
    }
  }

  ro.sortedIndices!.sort((iA, iB) => ro.distances![iB] - ro.distances![iA]);

  for (let sortedPos = 0; sortedPos < ro.totalElementCount; sortedPos++) {
    const originalIdx = ro.sortedIndices![sortedPos];
    ro.sortedPositions![originalIdx] = sortedPos;
  }

  return ro.sortedPositions;
}
export function getGeometryResource(
  resources: GeometryResources,
  type: string,
): GeometryResource {
  const resource = resources[type];
  if (!resource) {
    throw new Error(`No geometry resource found for type ${type}`);
  }
  return resource;
}

function alphaProperties(
  hasAlphaComponents: boolean,
  totalElementCount: number,
) {
  return {
    hasAlphaComponents,
    sortedIndices: hasAlphaComponents
      ? new Uint32Array(totalElementCount)
      : undefined,
    distances: hasAlphaComponents
      ? new Float32Array(totalElementCount)
      : undefined,
    sortedPositions: hasAlphaComponents
      ? new Uint32Array(totalElementCount)
      : undefined,
  };
}

const requestAdapterWithRetry = async (maxAttempts = 4, delayMs = 10) => {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });

    if (adapter) {
      return adapter;
    }

    if (attempt < maxAttempts - 1) {
      // console.log(`[Debug] Adapter request failed, retrying in ${delayMs}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  throw new Error(`Failed to get GPU adapter after ${maxAttempts} attempts`);
};

export function SceneImpl({
  components,
  containerWidth,
  containerHeight,
  style,
  camera: controlledCamera,
  defaultCamera,
  onCameraChange,
  onFrameRendered,
  onReady,
  onHover: onSceneHover,
  onClick: onSceneClick,
  readyState = NOOP_READY_STATE,
  primitiveSpecs,
}: SceneImplProps) {
  // Merge default registry with custom specs
  const primitiveRegistry = useMemo(() => {
    if (!primitiveSpecs) return defaultPrimitiveRegistry;
    return { ...defaultPrimitiveRegistry, ...primitiveSpecs };
  }, [primitiveSpecs]);

  // We'll store references to the GPU + other stuff in a ref object
  const gpuRef = useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;
    uniformBuffer: GPUBuffer;
    uniformBindGroup: GPUBindGroup;
    bindGroupLayout: GPUBindGroupLayout;
    depthTexture: GPUTexture | null;
    pickTexture: GPUTexture | null;
    pickDepthTexture: GPUTexture | null;
    readbackBuffer: GPUBuffer;

    // Outline rendering resources
    outlineTexture: GPUTexture | null;
    outlineDepthTexture: GPUTexture | null;
    outlinePipeline: GPURenderPipeline | null;
    outlineBindGroupLayout: GPUBindGroupLayout | null;
    outlineBindGroup: GPUBindGroup | null;
    outlineParamsBuffer: GPUBuffer | null;
    outlineSampler: GPUSampler | null;

    renderObjects: RenderObject[];
    pipelineCache: Map<string, PipelineCacheEntry>;
    dynamicBuffers: DynamicBuffers | null;
    resources: GeometryResources;
    imageResources: Map<string, ImageResource>;

    renderedCamera?: CameraState;
    renderedComponents?: ComponentConfig[];
  } | null>(null);

  const [internalCamera, setInternalCamera] = useState<CameraState>(() => {
    return createCameraState(defaultCamera);
  });

  // Use the appropriate camera state based on whether we're controlled or not
  const activeCameraRef = useRef<CameraState | null>(null);
  useMemo(() => {
    let nextCamera: CameraState;
    if (controlledCamera) {
      nextCamera = createCameraState(controlledCamera);
    } else {
      nextCamera = internalCamera;
    }
    activeCameraRef.current = nextCamera;
    return nextCamera;
  }, [controlledCamera, internalCamera]);

  const handleCameraUpdate = useCallback(
    (updateFn: (camera: CameraState) => CameraState) => {
      const newCameraState = updateFn(activeCameraRef.current!);

      if (controlledCamera) {
        onCameraChange?.(createCameraParams(newCameraState));
      } else {
        setInternalCamera(newCameraState);
        onCameraChange?.(createCameraParams(newCameraState));
      }
    },
    [controlledCamera, onCameraChange],
  );

    // Check if any component can render outlines
  const anyOutlineEnabled = useMemo(() => {
    return components.some(
      (component) =>
        isOutlineEnabled(component) || hasOutlineConfig(component.hoverProps),
    );
  }, [components]);

  const createOrUpdateOutlineTextures = useCallback(() => {
    if (!gpuRef.current || !canvasRef.current) return;
    const { device, outlineTexture, outlineDepthTexture } = gpuRef.current;

    const canvas = canvasRef.current;
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;

    if (outlineTexture) outlineTexture.destroy();
    if (outlineDepthTexture) outlineDepthTexture.destroy();

    // Silhouette mask texture - use same format as canvas for pipeline compatibility
    const format = navigator.gpu.getPreferredCanvasFormat();
    const silhouetteTex = device.createTexture({
      size: [displayWidth, displayHeight],
      format,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      label: "Outline silhouette texture",
    });

    // Depth texture for silhouette pass (object's own depth, not scene depth)
    const silhouetteDepthTex = device.createTexture({
      size: [displayWidth, displayHeight],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      label: "Outline depth texture",
    });

    gpuRef.current.outlineTexture = silhouetteTex;
    gpuRef.current.outlineDepthTexture = silhouetteDepthTex;

    // Create sampler if not exists
    if (!gpuRef.current.outlineSampler) {
      gpuRef.current.outlineSampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
      });
    }

    // Create or recreate bind group layout for outline post-process
    if (!gpuRef.current.outlineBindGroupLayout) {
      gpuRef.current.outlineBindGroupLayout = device.createBindGroupLayout({
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
          {
            binding: 2,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform" },
          },
        ],
        label: "Outline bind group layout",
      });
    }

    // Create params buffer if not exists (32 bytes: vec3 color + f32 width + vec2 texelSize + vec2 pad)
    if (!gpuRef.current.outlineParamsBuffer) {
      gpuRef.current.outlineParamsBuffer = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        label: "Outline params buffer",
      });
    }

    // Recreate bind group with new texture
    gpuRef.current.outlineBindGroup = device.createBindGroup({
      layout: gpuRef.current.outlineBindGroupLayout,
      entries: [
        { binding: 0, resource: gpuRef.current.outlineSampler },
        { binding: 1, resource: silhouetteTex.createView() },
        {
          binding: 2,
          resource: { buffer: gpuRef.current.outlineParamsBuffer },
        },
      ],
      label: "Outline bind group",
    });

    // Create outline post-process pipeline if not exists
    if (!gpuRef.current.outlinePipeline) {
      const bindGroupLayout = gpuRef.current.outlineBindGroupLayout;
      if (!bindGroupLayout) {
        console.error("Outline bind group layout is null!");
        return;
      }

      const format = navigator.gpu.getPreferredCanvasFormat();

      // Create shader modules
      const vertModule = device.createShaderModule({
        code: outlineVertCode,
        label: "Outline vertex shader",
      });
      const fragModule = device.createShaderModule({
        code: outlineFragCode,
        label: "Outline fragment shader",
      });

      try {
        gpuRef.current.outlinePipeline = device.createRenderPipeline({
          layout: device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
          }),
          vertex: {
            module: vertModule,
            entryPoint: "vs_main",
          },
          fragment: {
            module: fragModule,
            entryPoint: "fs_outline",
            targets: [
              {
                format,
                blend: {
                  color: {
                    srcFactor: "src-alpha",
                    dstFactor: "one-minus-src-alpha",
                    operation: "add",
                  },
                  alpha: {
                    srcFactor: "one",
                    dstFactor: "one-minus-src-alpha",
                    operation: "add",
                  },
                },
              },
            ],
          },
          primitive: {
            topology: "triangle-list",
          },
          label: "Outline post-process pipeline",
        });
      } catch (err) {
        console.error("Failed to create outline pipeline:", err);
      }
    }
  }, []);

  // Create a render callback for the canvas snapshot system
  // This function is called during PDF export to render the 3D scene to a texture
  // that can be captured as a static image
  const renderToTexture = useCallback(
    async (targetTexture: GPUTexture, depthTexture: GPUTexture | null) => {
      if (!gpuRef.current) return;
      const {
        device,
        uniformBindGroup,
        renderObjects,
        outlineTexture,
        outlineDepthTexture,
        outlinePipeline,
        outlineBindGroup,
        outlineParamsBuffer,
      } = gpuRef.current;

      const commandEncoder = device.createCommandEncoder({
        label: "Scene snapshot encoder",
      });
      const targetView = targetTexture.createView();

      const sortedRenderObjects = [...renderObjects].sort((a, b) => {
        const aOverlay = a.layer === "overlay" ? 1 : 0;
        const bOverlay = b.layer === "overlay" ? 1 : 0;
        return aOverlay - bOverlay;
      });

      encodeScenePass({
        commandEncoder,
        targetView,
        depthTexture: depthTexture || null,
        renderObjects: sortedRenderObjects,
        uniformBindGroup,
      });

      const outlineGroups = components
        ? collectOutlineGroups(components, lastHoverState.current)
        : [];

      let outlineReady = !!(
        outlineTexture &&
        outlineDepthTexture &&
        outlinePipeline &&
        outlineBindGroup &&
        outlineParamsBuffer
      );

      if (!outlineReady && outlineGroups.length > 0 && anyOutlineEnabled) {
        createOrUpdateOutlineTextures();
        const refreshed = gpuRef.current;
        outlineReady = !!(
          refreshed &&
          refreshed.outlineTexture &&
          refreshed.outlineDepthTexture &&
          refreshed.outlinePipeline &&
          refreshed.outlineBindGroup &&
          refreshed.outlineParamsBuffer
        );
      }

      const tempOutlineBuffers: GPUBuffer[] = [];
      if (outlineGroups.length > 0 && outlineReady) {
        for (const group of outlineGroups) {
          renderSilhouetteTargets(
            commandEncoder,
            group.targets,
            components,
            tempOutlineBuffers,
          );
          renderOutlinePostProcess(
            commandEncoder,
            targetView,
            group.style.color,
            group.style.width,
          );
        }
      }

      device.queue.submit([commandEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      for (const buffer of tempOutlineBuffers) {
        buffer.destroy();
      }
    },
    [
      components,
      anyOutlineEnabled,
      createOrUpdateOutlineTextures,
      containerWidth,
      containerHeight,
    ],
  );

  const { canvasRef } = useCanvasSnapshot(
    gpuRef.current?.device,
    gpuRef.current?.context,
    renderToTexture,
  );

  const [isReady, setIsReady] = useState(false);

  const pickingLockRef = useRef(false);

  const lastHoverState = useRef<{
    componentIdx: number;
    elementIdx: number;
  } | null>(null);

  // Track last pick screen coordinates for building PickInfo
  const lastPickScreenX = useRef<number>(0);
  const lastPickScreenY = useRef<number>(0);

  // Element drag state - tracks active drag operations on components
  const elementDragState = useRef<{
    componentIdx: number;
    elementIdx: number;
    constraint: ResolvedConstraint;
    startPickInfo: PickInfo;
    startInstanceCenter: [number, number, number];
    startScreenPos: { x: number; y: number };
  } | null>(null);

  // Track which element is hovered for components with hoverProps
  // Maps componentIdx -> elementIdx for components that need re-render on hover
  const hoverPropsState = useRef<Map<number, number>>(new Map());
  // Track if hoverProps-related data needs rebuild
  const hoverPropsDirty = useRef(false);

  const renderObjectCache = useRef<RenderObjectCache>({});

  /******************************************************
   * A) initWebGPU
   ******************************************************/
  const initWebGPU = useCallback(async () => {
    if (!canvasRef.current) return;
    if (!navigator.gpu) {
      console.error("[Debug] WebGPU not supported in this browser.");
      return;
    }
    try {
      const adapter = await requestAdapterWithRetry();

      const device = await adapter.requestDevice().catch((err) => {
        console.error("[Debug] Failed to create WebGPU device:", err);
        throw err;
      });

      // Add error handling for uncaptured errors
      device.addEventListener("uncapturederror", ((event: Event) => {
        if (event instanceof GPUUncapturedErrorEvent) {
          console.error("Uncaptured WebGPU error:", event.error);
          // Log additional context about where the error occurred
          console.error("Error source:", event.error.message);
        }
      }) as EventListener);

      const context = canvasRef.current.getContext(
        "webgpu",
      ) as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode: "premultiplied" });

      // Create all the WebGPU resources
      const bindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform" },
          },
        ],
      });

      const uniformBufferSize = 128;
      const uniformBuffer = device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      const uniformBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
      });

      const readbackBuffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: "Picking readback buffer",
      });

      gpuRef.current = {
        device,
        context,
        uniformBuffer,
        uniformBindGroup,
        bindGroupLayout,
        depthTexture: null,
        pickTexture: null,
        pickDepthTexture: null,
        readbackBuffer,
        // Outline resources (initialized lazily when needed)
        outlineTexture: null,
        outlineDepthTexture: null,
        outlinePipeline: null,
        outlineBindGroupLayout: null,
        outlineBindGroup: null,
        outlineParamsBuffer: null,
        outlineSampler: null,
        renderObjects: [],
        pipelineCache: new Map(),
        dynamicBuffers: null,
        resources: {
          PointCloud: null,
          Ellipsoid: null,
          EllipsoidAxes: null,
          Cuboid: null,
          LineBeams: null,
          LineSegments: null,
          BoundingBox: null,
          ImagePlane: null,
        },
        imageResources: new Map(),
      };

      // Now initialize geometry resources
      initGeometryResources(
        device,
        gpuRef.current.resources,
        primitiveRegistry,
      );

      setIsReady(true);
    } catch (err) {
      console.error("[Debug] Error during WebGPU initialization:", err);
    }
  }, [primitiveRegistry]);

  /******************************************************
   * B) Depth & Pick textures
   ******************************************************/
  const createOrUpdateDepthTexture = useCallback(() => {
    if (!gpuRef.current || !canvasRef.current) return;
    const { device, depthTexture } = gpuRef.current;

    // Get the actual canvas size
    const canvas = canvasRef.current;
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;

    if (depthTexture) depthTexture.destroy();
    const dt = device.createTexture({
      size: [displayWidth, displayHeight],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    gpuRef.current.depthTexture = dt;
  }, []);

  const createOrUpdatePickTextures = useCallback(() => {
    if (!gpuRef.current || !canvasRef.current) return;
    const { device, pickTexture, pickDepthTexture } = gpuRef.current;

    // Get the actual canvas size
    const canvas = canvasRef.current;
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;

    if (pickTexture) pickTexture.destroy();
    if (pickDepthTexture) pickDepthTexture.destroy();

    const colorTex = device.createTexture({
      size: [displayWidth, displayHeight],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const depthTex = device.createTexture({
      size: [displayWidth, displayHeight],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    gpuRef.current.pickTexture = colorTex;
    gpuRef.current.pickDepthTexture = depthTex;
  }, []);

  type ComponentType = ComponentConfig["type"];
  type RenderLayer = "scene" | "overlay";

  interface TypeInfo {
    offsets: number[];
    elementCounts: number[];
    indices: number[];
    totalRenderSize: number;
    totalPickingSize: number;
    totalElementCount: number;
    components: ComponentConfig[];
    elementOffsets: number[];
    layer: RenderLayer;
    type: ComponentType;
    batchKey?: string;
  }

  /**
   * Creates a key for type+layer combination.
   * This allows us to have separate RenderObjects for scene vs overlay components of the same type.
   */
  function typeLayerKey(
    type: ComponentType,
    layer: RenderLayer,
    batchKey?: string,
  ): string {
    return batchKey ? `${type}_${layer}_${batchKey}` : `${type}_${layer}`;
  }

  // Update the collectTypeData function signature
  function collectTypeData(
    components: ComponentConfig[],
  ): Map<string, TypeInfo> {
    const typeArrays = new Map<string, TypeInfo>();

    // Single pass through components
    components.forEach((comp, idx) => {
      const spec = primitiveRegistry[comp.type];
      if (!spec) return;

      // Get the element count and instance count
      const elementCount = spec.getElementCount(comp);

      if (elementCount === 0) return;
      const instanceCount = elementCount * spec.instancesPerElement;

      // Just allocate the array without building data, 4 bytes per float
      const renderSize =
        instanceCount * spec.floatsPerInstance * Float32Array.BYTES_PER_ELEMENT;
      const pickingSize =
        instanceCount * spec.floatsPerPicking * Float32Array.BYTES_PER_ELEMENT;

      // Determine layer for this component (default to "scene")
      const layer: RenderLayer = comp.layer || "scene";
      const rawBatchKey = spec.getBatchKey?.(comp);
      const batchKey = rawBatchKey !== undefined ? String(rawBatchKey) : undefined;
      const key = typeLayerKey(comp.type, layer, batchKey);

      let typeInfo = typeArrays.get(key);
      if (!typeInfo) {
        typeInfo = {
          totalElementCount: 0,
          totalRenderSize: 0,
          totalPickingSize: 0,
          components: [],
          indices: [],
          offsets: [],
          elementCounts: [],
          elementOffsets: [],
          layer,
          type: comp.type,
          batchKey,
        };
        typeArrays.set(key, typeInfo);
      }

      typeInfo.components.push(comp);
      typeInfo.indices.push(idx);
      typeInfo.offsets.push(typeInfo.totalRenderSize);
      typeInfo.elementCounts.push(elementCount);
      typeInfo.elementOffsets.push(typeInfo.totalElementCount);
      typeInfo.totalElementCount += elementCount;
      typeInfo.totalRenderSize += renderSize;
      typeInfo.totalPickingSize += pickingSize;
    });

    return typeArrays;
  }

  // Update buildRenderObjects to include caching
  function buildRenderObjects(components: ComponentConfig[]): RenderObject[] {
    if (!gpuRef.current) return [];
    const { device, bindGroupLayout, pipelineCache, resources } =
      gpuRef.current;

    // Collect render data using helper (now keyed by type_layer)
    const typeArrays = collectTypeData(components);

    // Clear out unused cache entries (now keyed by type_layer)
    Object.keys(renderObjectCache.current).forEach((key) => {
      if (!typeArrays.has(key)) {
        delete renderObjectCache.current[key];
      }
    });

    // Remove image resources for batches no longer used
    const activeImageKeys = new Set<string>();
    typeArrays.forEach((info: TypeInfo) => {
      if (info.type === "ImagePlane" && info.batchKey) {
        activeImageKeys.add(info.batchKey);
      }
    });
    gpuRef.current.imageResources.forEach((resource, key) => {
      if (!activeImageKeys.has(key)) {
        resource.texture.destroy();
        gpuRef.current.imageResources.delete(key);
      }
    });

    // Track global start index for all components
    let globalStartIndex = 0;

    // Calculate total buffer sizes needed
    let totalRenderSize = 0;
    let totalPickingSize = 0;
    typeArrays.forEach((info: TypeInfo, key: string) => {
      // Extract the component type from the group
      const type = info.type;
      if (!type) return;
      const spec = primitiveRegistry[type];
      if (!spec) return;

      // Calculate total instance count for this type
      const totalElementCount = info.elementCounts.reduce(
        (sum, count) => sum + count,
        0,
      );
      const totalInstanceCount = totalElementCount * spec.instancesPerElement;

      // Calculate total size needed for all instances of this type
      totalRenderSize += align16(
        totalInstanceCount *
          spec.floatsPerInstance *
          Float32Array.BYTES_PER_ELEMENT,
      );
      totalPickingSize += align16(
        totalInstanceCount *
          spec.floatsPerPicking *
          Float32Array.BYTES_PER_ELEMENT,
      );
    });

    // Create or recreate dynamic buffers if needed
    if (
      !gpuRef.current.dynamicBuffers ||
      gpuRef.current.dynamicBuffers.renderBuffer.size < totalRenderSize ||
      gpuRef.current.dynamicBuffers.pickingBuffer.size < totalPickingSize
    ) {
      gpuRef.current.dynamicBuffers?.renderBuffer.destroy();
      gpuRef.current.dynamicBuffers?.pickingBuffer.destroy();

      const renderBuffer = device.createBuffer({
        size: totalRenderSize,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
      });

      const pickingBuffer = device.createBuffer({
        size: totalPickingSize,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
      });

      gpuRef.current.dynamicBuffers = {
        renderBuffer,
        pickingBuffer,
        renderOffset: 0,
        pickingOffset: 0,
      };
    }
    const dynamicBuffers = gpuRef.current.dynamicBuffers!;

    // Reset buffer offsets
    dynamicBuffers.renderOffset = 0;
    dynamicBuffers.pickingOffset = 0;

    const validRenderObjects: RenderObject[] = [];

    // Create or update render objects and write buffer data
    typeArrays.forEach((info: TypeInfo, key: string) => {
      // Extract the component type from the group
      const type = info.type;
      if (!type) return;
      const spec = primitiveRegistry[type];
      if (!spec) return;

      // Get the layer for this group of components
      const layer = info.layer;
      const isOverlay = layer === "overlay";

      try {
        // Ensure 4-byte alignment for all offsets
        const renderOffset = align16(dynamicBuffers.renderOffset);
        const pickingOffset = align16(dynamicBuffers.pickingOffset);

        // Try to get existing render object (keyed by type_layer)
        let renderObject = renderObjectCache.current[key];
        const needNewRenderObject =
          !renderObject ||
          renderObject.totalElementCount !== info.totalElementCount;

        // Create or reuse render data arrays
        let renderData: Float32Array;
        let pickingData: Float32Array;

        if (needNewRenderObject) {
          renderData = new Float32Array(
            info.totalRenderSize / Float32Array.BYTES_PER_ELEMENT,
          );
          pickingData = new Float32Array(
            info.totalPickingSize / Float32Array.BYTES_PER_ELEMENT,
          );
        } else {
          renderData = renderObject.renderData;
          pickingData = renderObject.pickingData;
        }

        // Get or create pipeline based on layer
        const pipeline = isOverlay
          ? spec.getOverlayPipeline(device, bindGroupLayout, pipelineCache)
          : spec.getRenderPipeline(device, bindGroupLayout, pipelineCache);
        if (!pipeline) return;

        // Get picking pipeline based on layer
        const pickingPipeline = isOverlay
          ? spec.getOverlayPickingPipeline(
              device,
              bindGroupLayout,
              pipelineCache,
            )
          : spec.getPickingPipeline(device, bindGroupLayout, pipelineCache);
        if (!pickingPipeline) return;

        // Build component offsets for this type's components
        const typeComponentOffsets: ComponentOffset[] = [];
        let typeStartIndex = globalStartIndex;
        let elementStartIndex = 0;
        info.indices.forEach((componentIdx, i) => {
          const componentElementCount = info.elementCounts[i];
          typeComponentOffsets.push({
            componentIdx,
            pickingStart: typeStartIndex,
            elementStart: elementStartIndex,
            elementCount: componentElementCount,
          });
          typeStartIndex += componentElementCount;
          elementStartIndex += componentElementCount;
        });
        globalStartIndex = typeStartIndex;

        const totalInstanceCount =
          info.totalElementCount * spec.instancesPerElement;

        // Create or update buffer info
        const bufferInfo = {
          buffer: dynamicBuffers.renderBuffer,
          offset: renderOffset,
          stride: spec.floatsPerInstance * Float32Array.BYTES_PER_ELEMENT,
        };
        const pickingBufferInfo = {
          buffer: dynamicBuffers.pickingBuffer,
          offset: pickingOffset,
          stride: spec.floatsPerPicking * Float32Array.BYTES_PER_ELEMENT,
        };

        const hasAlphaComponents = components.some(componentHasAlpha);
        let textureBindGroup: GPUBindGroup | undefined;

        // Handle texture binding for ImagePlane and textured meshes
        const firstComponent = info.components[0] as any;
        const hasTexture =
          type === "ImagePlane" ||
          (firstComponent.texture !== undefined && spec.getBatchKey);

        if (hasTexture) {
          // ImagePlane uses 'image', textured meshes use 'texture'
          const imageSource =
            type === "ImagePlane"
              ? (firstComponent as ImagePlaneComponentConfig).image
              : firstComponent.texture;
          const imageKey =
            type === "ImagePlane"
              ? (firstComponent as ImagePlaneComponentConfig).imageKey
              : firstComponent.textureKey;
          const batchKey = info.batchKey ?? spec.getBatchKey?.(firstComponent);
          if (batchKey && imageSource) {
            const imageResource = ensureImageResource(
              device,
              gpuRef.current.imageResources,
              String(batchKey),
              imageSource,
              imageKey,
            );
            textureBindGroup = imageResource?.bindGroup;
          }
          if (!textureBindGroup) return;
        }

        const geometryResource = getGeometryResource(resources, type);
        if (needNewRenderObject) {
          // Create new render object with all the required resources
          renderObject = {
            pipeline,
            pickingPipeline,
            geometryBuffer: geometryResource.vb,
            instanceBuffer: bufferInfo,
            indexBuffer: geometryResource.ib,
            indexCount: geometryResource.indexCount,
            indexFormat: geometryResource.indexFormat,
            instanceCount: totalInstanceCount,
            vertexCount: geometryResource.vertexCount,
            textureBindGroup,
            pickingInstanceBuffer: pickingBufferInfo,
            pickingDataStale: true,
            componentIndex: info.indices[0],
            renderData: renderData,
            pickingData: pickingData,
            totalElementCount: info.totalElementCount,
            componentOffsets: typeComponentOffsets,
            spec: spec,
            layer: layer,
            ...alphaProperties(hasAlphaComponents, info.totalElementCount),
          };
          renderObjectCache.current[key] = renderObject;
        } else {
          // Update existing render object with new buffer info and state
          renderObject.instanceBuffer = bufferInfo;
          renderObject.pickingInstanceBuffer = pickingBufferInfo;
          renderObject.instanceCount = totalInstanceCount;
          renderObject.componentIndex = info.indices[0];
          renderObject.componentOffsets = typeComponentOffsets;
          renderObject.spec = spec;
          renderObject.layer = layer;
          renderObject.pickingDataStale = true;
          renderObject.indexFormat = geometryResource.indexFormat;
          renderObject.textureBindGroup = textureBindGroup;
          if (hasAlphaComponents && !renderObject.hasAlphaComponents) {
            Object.assign(
              renderObject,
              alphaProperties(hasAlphaComponents, info.totalElementCount),
            );
          }
        }

        validRenderObjects.push(renderObject);

        // Update buffer offsets ensuring alignment
        dynamicBuffers.renderOffset =
          renderOffset + align16(renderData.byteLength);
        dynamicBuffers.pickingOffset =
          pickingOffset +
          align16(
            totalInstanceCount *
              spec.floatsPerPicking *
              Float32Array.BYTES_PER_ELEMENT,
          );
      } catch (error) {
        console.error(`Error creating render object for type ${type}:`, error);
      }
    });

    return validRenderObjects;
  }

  /******************************************************
   * C) Render pass (single call, no loop)
   ******************************************************/

  /**
   * Renders silhouette targets into the outline texture.
   */
  function renderSilhouetteTargets(
    commandEncoder: GPUCommandEncoder,
    targets: OutlineTarget[],
    components: ComponentConfig[],
    tempBuffers: GPUBuffer[],
  ): void {
    if (!gpuRef.current || targets.length === 0) return;

    const {
      device,
      outlineTexture,
      outlineDepthTexture,
      uniformBindGroup,
      renderObjects,
    } = gpuRef.current;

    if (!outlineTexture || !outlineDepthTexture) return;

    const createTempInstanceBuffer = (
      data: Float32Array,
      label: string,
    ): GPUBuffer => {
      const buffer = device.createBuffer({
        size: align16(data.byteLength),
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        label,
      });
      device.queue.writeBuffer(
        buffer,
        0,
        data.buffer,
        data.byteOffset,
        data.byteLength,
      );
      tempBuffers.push(buffer);
      return buffer;
    };

    const buildComponentInstanceData = (
      ro: RenderObject,
      compOffset: ComponentOffset,
    ) => {
      const instancesPerElement = ro.spec.instancesPerElement;
      const floatsPerInstance = ro.spec.floatsPerInstance;
      const totalInstances = compOffset.elementCount * instancesPerElement;
      if (totalInstances === 0) return null;

      const data = new Float32Array(totalInstances * floatsPerInstance);
      for (let elemIndex = 0; elemIndex < compOffset.elementCount; elemIndex++) {
        const originalIndex = compOffset.elementStart + elemIndex;
        const sortedIndex = ro.sortedPositions
          ? ro.sortedPositions[originalIndex]
          : originalIndex;
        const sourceOffset =
          sortedIndex * instancesPerElement * floatsPerInstance;
        const destOffset =
          elemIndex * instancesPerElement * floatsPerInstance;
        data.set(
          ro.renderData.subarray(
            sourceOffset,
            sourceOffset + instancesPerElement * floatsPerInstance,
          ),
          destOffset,
        );
      }

      return data;
    };

    const pass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: outlineTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: outlineDepthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    for (const target of targets) {
      if (!components[target.componentIdx]) continue;

      const ro = renderObjects.find((r) =>
        r.componentOffsets.some((o) => o.componentIdx === target.componentIdx),
      );
      if (!ro) continue;

      const compOffset = ro.componentOffsets.find(
        (o) => o.componentIdx === target.componentIdx,
      );
      if (!compOffset) continue;

      const instancesPerElement = ro.spec.instancesPerElement;
      const floatsPerInstance = ro.spec.floatsPerInstance;
      let instanceBuffer = ro.instanceBuffer.buffer;
      let instanceOffset = ro.instanceBuffer.offset;
      let instanceCount = 0;

      if (target.kind === "component") {
        instanceCount = compOffset.elementCount * instancesPerElement;
        if (instanceCount === 0) continue;

        if (ro.sortedPositions) {
          const instanceData = buildComponentInstanceData(ro, compOffset);
          if (!instanceData) continue;
          const tempBuffer = createTempInstanceBuffer(
            instanceData,
            "Outline component instance buffer",
          );
          instanceBuffer = tempBuffer;
          instanceOffset = 0;
        } else {
          const startInstance = compOffset.elementStart * instancesPerElement;
          instanceOffset =
            ro.instanceBuffer.offset +
            startInstance *
              floatsPerInstance *
              Float32Array.BYTES_PER_ELEMENT;
        }
      } else {
        const originalIndex = compOffset.elementStart + target.elementIdx;
        if (
          target.elementIdx < 0 ||
          target.elementIdx >= compOffset.elementCount
        ) {
          continue;
        }
        const sortedIndex = ro.sortedPositions
          ? ro.sortedPositions[originalIndex]
          : originalIndex;
        const startOffset = sortedIndex * instancesPerElement * floatsPerInstance;
        const instanceData = ro.renderData.subarray(
          startOffset,
          startOffset + instancesPerElement * floatsPerInstance,
        );
        const tempBuffer = createTempInstanceBuffer(
          instanceData,
          "Outline element instance buffer",
        );
        instanceBuffer = tempBuffer;
        instanceOffset = 0;
        instanceCount = instancesPerElement;
      }

      if (instanceCount === 0) continue;

      pass.setPipeline(ro.pipeline);
      pass.setBindGroup(0, uniformBindGroup);
      if (ro.textureBindGroup) {
        pass.setBindGroup(1, ro.textureBindGroup);
      }
      pass.setVertexBuffer(0, ro.geometryBuffer);
      pass.setVertexBuffer(1, instanceBuffer, instanceOffset);
      if (ro.indexBuffer && ro.indexCount > 0) {
        pass.setIndexBuffer(ro.indexBuffer, ro.indexFormat ?? "uint16");
        pass.drawIndexed(ro.indexCount, instanceCount);
      } else if (ro.vertexCount > 0) {
        pass.draw(ro.vertexCount, instanceCount);
      }
    }

    pass.end();
  }

  /**
   * Renders the outline post-process pass, compositing the outline over the canvas.
   */
  function renderOutlinePostProcess(
    commandEncoder: GPUCommandEncoder,
    targetView: GPUTextureView,
    outlineColorVal: [number, number, number],
    outlineWidthVal: number,
  ): void {
    if (!gpuRef.current) return;

    const {
      device,
      outlinePipeline,
      outlineBindGroup,
      outlineParamsBuffer,
    } = gpuRef.current;

    if (!outlinePipeline || !outlineBindGroup || !outlineParamsBuffer) return;

    // Update outline params
    const dpr = window.devicePixelRatio || 1;
    const texelSizeX = 1.0 / (containerWidth * dpr);
    const texelSizeY = 1.0 / (containerHeight * dpr);

    const paramsData = new Float32Array([
      outlineColorVal[0],
      outlineColorVal[1],
      outlineColorVal[2],
      outlineWidthVal,
      texelSizeX,
      texelSizeY,
      0, // padding
      0, // padding
    ]);
    device.queue.writeBuffer(
      outlineParamsBuffer,
      0,
      paramsData.buffer,
      paramsData.byteOffset,
      paramsData.byteLength,
    );

    const pass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: targetView,
          loadOp: "load", // Keep existing content
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(outlinePipeline);
    pass.setBindGroup(0, outlineBindGroup);
    pass.draw(3); // Full-screen triangle

    pass.end();
  }

  const pendingAnimationFrameRef = useRef<number | null>(null);

  const renderFrame = useCallback(
    async function renderFrameInner(
      source: string,
      camState?: CameraState,
      components?: ComponentConfig[],
    ) {
      if (pendingAnimationFrameRef.current) {
        cancelAnimationFrame(pendingAnimationFrameRef.current);
        pendingAnimationFrameRef.current = null;
      }
      if (!gpuRef.current) return;

      camState = camState || activeCameraRef.current!;

      const onRenderComplete = readyState.beginUpdate("impl3d/renderFrame");

      components = components || gpuRef.current.renderedComponents;
      const componentsChanged =
        gpuRef.current.renderedComponents !== components;

      if (componentsChanged) {
        gpuRef.current.renderObjects = buildRenderObjects(components!);
        gpuRef.current.renderedComponents = components;
      }

      const {
        device,
        context,
        uniformBuffer,
        uniformBindGroup,
        renderObjects,
        depthTexture,
        outlineTexture,
        outlineDepthTexture,
        outlinePipeline,
        outlineBindGroup,
        outlineParamsBuffer,
      } = gpuRef.current;

      const cameraMoved = hasCameraMoved(
        camState.position,
        gpuRef.current.renderedCamera?.position,
        0.0001,
      );
      gpuRef.current.renderedCamera = camState;

      // Check if hoverProps state changed and needs rebuild
      const hoverPropsNeedsBuild = hoverPropsDirty.current;
      if (hoverPropsNeedsBuild) {
        hoverPropsDirty.current = false;
      }

      // Update data for objects that need it
      renderObjects.forEach(function updateRenderObject(ro) {
        const needsSorting = ro.hasAlphaComponents;
        const needsBuild =
          (needsSorting && cameraMoved) ||
          componentsChanged ||
          hoverPropsNeedsBuild;

        // Skip if no update needed
        if (!needsBuild) return;

        // Update sorting if needed
        if (needsSorting) {
          updateInstanceSorting(ro, components!, camState.position);
        }

        // We'll work directly with the cached render data to avoid an extra allocation and copy
        const renderData = ro.renderData;

        // Build render data for each component
        for (let i = 0; i < ro.componentOffsets.length; i++) {
          const offset = ro.componentOffsets[i];
          const component = components![offset.componentIdx];

          // Get hovered index if this component has hoverProps and is being hovered
          const hoveredIndex = component.hoverProps
            ? hoverPropsState.current.get(offset.componentIdx)
            : undefined;

          buildRenderData(
            component,
            ro.spec,
            renderData,
            offset.elementStart,
            ro.sortedPositions,
            hoveredIndex,
          );
        }

        ro.pickingDataStale = true;

        device.queue.writeBuffer(
          ro.instanceBuffer.buffer,
          ro.instanceBuffer.offset,
          renderData.buffer,
          renderData.byteOffset,
          renderData.byteLength,
        );
      });

      const uniformData = computeUniformData(
        containerWidth,
        containerHeight,
        camState,
      );
      device.queue.writeBuffer(
        uniformBuffer,
        0,
        uniformData.buffer,
        uniformData.byteOffset,
        uniformData.byteLength,
      );

      try {
        // Sort render objects: scene first, then overlay (overlay renders on top)
        const sortedRenderObjects = [...renderObjects].sort((a, b) => {
          const aOverlay = a.layer === "overlay" ? 1 : 0;
          const bOverlay = b.layer === "overlay" ? 1 : 0;
          return aOverlay - bOverlay;
        });

        const commandEncoder = device.createCommandEncoder({
          label: "Scene render encoder",
        });
        const targetTexture = context.getCurrentTexture();
        const targetView = targetTexture.createView();

        encodeScenePass({
          commandEncoder,
          targetView,
          depthTexture,
          renderObjects: sortedRenderObjects,
          uniformBindGroup,
        });

        const outlineReady = !!(
          outlineTexture &&
          outlineDepthTexture &&
          outlinePipeline &&
          outlineBindGroup &&
          outlineParamsBuffer
        );
        const tempOutlineBuffers: GPUBuffer[] = [];

        // Encode outline passes (hover + always-on) before submitting.
        if (components && outlineReady) {
          const outlineGroups = collectOutlineGroups(
            components,
            lastHoverState.current,
          );
          for (const group of outlineGroups) {
            renderSilhouetteTargets(
              commandEncoder,
              group.targets,
              components,
              tempOutlineBuffers,
            );
            renderOutlinePostProcess(
              commandEncoder,
              targetView,
              group.style.color,
              group.style.width,
            );
          }
        }

        device.queue.submit([commandEncoder.finish()]);

        // Now wait for all GPU work to complete
        await device.queue.onSubmittedWorkDone();

        for (const buffer of tempOutlineBuffers) {
          buffer.destroy();
        }
        onRenderComplete();
      } catch (err) {
        console.error(
          "[Debug] Error during renderFrame:",
          (err as Error).message,
        );
        onRenderComplete();
      }

      onFrameRendered?.(performance.now());
      onReady?.();
    },
    [
      containerWidth,
      containerHeight,
      onFrameRendered,
      components,
    ],
  );

  function requestRender(label: string) {
    if (!pendingAnimationFrameRef.current) {
      pendingAnimationFrameRef.current = requestAnimationFrame((t) =>
        renderFrame(label),
      );
    }
  }

  /******************************************************
   * D) Pick pass (on hover/click)
   ******************************************************/
  async function pickAtScreenXY(
    screenX: number,
    screenY: number,
    mode: "hover" | "click",
  ) {
    if (!gpuRef.current || !canvasRef.current || pickingLockRef.current) return;
    const pickingId = Date.now();
    const currentPickingId = pickingId;
    pickingLockRef.current = true;

    try {
      const {
        device,
        pickTexture,
        pickDepthTexture,
        readbackBuffer,
        uniformBindGroup,
        renderObjects,
      } = gpuRef.current;
      if (!pickTexture || !pickDepthTexture || !readbackBuffer) return;

      // Ensure picking data is ready for all objects
      for (let i = 0; i < renderObjects.length; i++) {
        ensurePickingData(
          gpuRef.current.device,
          gpuRef.current.renderedComponents!,
          renderObjects[i],
        );
      }

      // Store screen coordinates for building PickInfo
      lastPickScreenX.current = screenX;
      lastPickScreenY.current = screenY;

      // Convert screen coordinates to device pixels
      const dpr = window.devicePixelRatio || 1;
      const pickX = Math.floor(screenX * dpr);
      const pickY = Math.floor(screenY * dpr);
      const displayWidth = Math.floor(containerWidth * dpr);
      const displayHeight = Math.floor(containerHeight * dpr);

      if (
        pickX < 0 ||
        pickY < 0 ||
        pickX >= displayWidth ||
        pickY >= displayHeight
      ) {
        if (mode === "hover") handleHoverID(0);
        return;
      }

      // Sort render objects for picking: scene first, then overlay
      // Overlay objects use depthCompare: "always" so they overwrite scene objects,
      // giving them picking priority even when geometrically behind
      const sortedForPicking = [...renderObjects].sort((a, b) => {
        const aOverlay = a.layer === "overlay" ? 1 : 0;
        const bOverlay = b.layer === "overlay" ? 1 : 0;
        return aOverlay - bOverlay;
      });

      const cmd = device.createCommandEncoder({ label: "Picking encoder" });
      const passDesc: GPURenderPassDescriptor = {
        colorAttachments: [
          {
            view: pickTexture.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: pickDepthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      };
      const pass = cmd.beginRenderPass(passDesc);
      pass.setBindGroup(0, uniformBindGroup);

      for (const ro of sortedForPicking) {
        pass.setPipeline(ro.pickingPipeline);
        pass.setBindGroup(0, uniformBindGroup);
        if (ro.textureBindGroup) {
          pass.setBindGroup(1, ro.textureBindGroup);
        }
        pass.setVertexBuffer(0, ro.geometryBuffer);
        pass.setVertexBuffer(
          1,
          ro.pickingInstanceBuffer.buffer,
          ro.pickingInstanceBuffer.offset,
        );

        // Draw with indices if we have them, otherwise use vertex count
        if (ro.indexBuffer && ro.indexCount > 0) {
          pass.setIndexBuffer(ro.indexBuffer, ro.indexFormat ?? "uint16");
          pass.drawIndexed(ro.indexCount, ro.instanceCount);
        } else if (ro.vertexCount) {
          pass.draw(ro.vertexCount, ro.instanceCount);
        }
      }

      pass.end();

      cmd.copyTextureToBuffer(
        { texture: pickTexture, origin: { x: pickX, y: pickY } },
        { buffer: readbackBuffer, bytesPerRow: 256, rowsPerImage: 1 },
        [1, 1, 1],
      );
      device.queue.submit([cmd.finish()]);

      if (currentPickingId !== pickingId) return;
      await readbackBuffer.mapAsync(GPUMapMode.READ);
      if (currentPickingId !== pickingId) {
        readbackBuffer.unmap();
        return;
      }
      const arr = new Uint8Array(readbackBuffer.getMappedRange());
      const r = arr[0],
        g = arr[1],
        b = arr[2];
      readbackBuffer.unmap();
      const pickedID = (b << 16) | (g << 8) | r;

      if (mode === "hover") {
        handleHoverID(pickedID);
      } else {
        handleClickID(pickedID);
      }
    } finally {
      pickingLockRef.current = false;
    }
  }

  /**
   * Finds the component and element info for a given global picking index.
   * Returns null if no component is found.
   */
  function findPickedElement(globalIdx: number): {
    componentIdx: number;
    elementIdx: number;
    spec: PrimitiveSpec<any>;
  } | null {
    if (!gpuRef.current) return null;

    for (const ro of gpuRef.current.renderObjects) {
      if (!ro?.componentOffsets) continue;

      for (const offset of ro.componentOffsets) {
        if (
          globalIdx >= offset.pickingStart &&
          globalIdx < offset.pickingStart + offset.elementCount
        ) {
          return {
            componentIdx: offset.componentIdx,
            elementIdx: globalIdx - offset.pickingStart,
            spec: ro.spec,
          };
        }
      }
    }
    return null;
  }

  function handleHoverID(pickedID: number) {
    if (!gpuRef.current) return;

    // Get combined instance index
    const globalIdx = unpackID(pickedID);
    if (globalIdx === null) {
      // Clear previous hover if it exists
      if (lastHoverState.current) {
        const prevComponent = components[lastHoverState.current.componentIdx];
        prevComponent?.onHover?.(null);
        onSceneHover?.(null);

        // Clear hoverProps state if the previous component had hoverProps
        if (prevComponent?.hoverProps) {
          hoverPropsState.current.delete(lastHoverState.current.componentIdx);
          hoverPropsDirty.current = true;
        }

        const hadHover = lastHoverState.current !== null;
        const prevHadOutline = resolveHoverOutline(prevComponent).enabled;
        lastHoverState.current = null;

        // Re-render to clear outline or hoverProps styling
        if (prevHadOutline && hadHover) {
          requestRender("hover-clear");
        } else if (hoverPropsDirty.current) {
          requestRender("hoverProps-clear");
        }
      }
      return;
    }

    // Find which component this instance belongs to
    const found = findPickedElement(globalIdx);

    // If hover state hasn't changed, do nothing
    if (
      (!lastHoverState.current && !found) ||
      (lastHoverState.current &&
        found &&
        lastHoverState.current.componentIdx === found.componentIdx &&
        lastHoverState.current.elementIdx === found.elementIdx)
    ) {
      return;
    }

    // Clear previous hover if it exists
    if (lastHoverState.current) {
      const prevComponent = components[lastHoverState.current.componentIdx];
      prevComponent?.onHover?.(null);

      // Clear hoverProps state for previous component
      if (prevComponent?.hoverProps) {
        hoverPropsState.current.delete(lastHoverState.current.componentIdx);
        hoverPropsDirty.current = true;
      }
    }

    // Set new hover if it exists
    if (found) {
      const { componentIdx, elementIdx } = found;
      if (componentIdx >= 0 && componentIdx < components.length) {
        const component = components[componentIdx];

        // Build full pick info for scene callback
        if (onSceneHover) {
          const pickInfo = buildPickInfo({
            mode: "hover",
            componentIndex: componentIdx,
            elementIndex: elementIdx,
            screenX: lastPickScreenX.current,
            screenY: lastPickScreenY.current,
            rect: { width: containerWidth, height: containerHeight },
            camera: activeCameraRef.current!,
            component,
          });
          onSceneHover(pickInfo);
        }

        // Component-level callback with simple index
        component.onHover?.(elementIdx);

        // Update hoverProps state for new component
        if (component.hoverProps) {
          hoverPropsState.current.set(componentIdx, elementIdx);
          hoverPropsDirty.current = true;
        }
      }
    } else {
      onSceneHover?.(null);
    }

    // Update last hover state
    lastHoverState.current = found
      ? { componentIdx: found.componentIdx, elementIdx: found.elementIdx }
      : null;

    // Re-render to update outline or hoverProps styling
    // Check if new or previous component has outline enabled
    const newComponent = found ? components[found.componentIdx] : null;
    const newHasOutline = resolveHoverOutline(
      newComponent ?? undefined,
    ).enabled;
    if (newHasOutline) {
      requestRender("hover-change");
    } else if (hoverPropsDirty.current) {
      requestRender("hoverProps-change");
    }
  }

  function handleClickID(pickedID: number) {
    if (!gpuRef.current) return;

    // Get combined instance index
    const globalIdx = unpackID(pickedID);
    if (globalIdx === null) return;

    // Find which component this instance belongs to
    const found = findPickedElement(globalIdx);
    if (!found) return;

    const { componentIdx, elementIdx } = found;
    if (componentIdx >= 0 && componentIdx < components.length) {
      const component = components[componentIdx];

      // Build full pick info for scene callback
      if (onSceneClick) {
        const pickInfo = buildPickInfo({
          mode: "click",
          componentIndex: componentIdx,
          elementIndex: elementIdx,
          screenX: lastPickScreenX.current,
          screenY: lastPickScreenY.current,
          rect: { width: containerWidth, height: containerHeight },
          camera: activeCameraRef.current!,
          component,
        });
        if (pickInfo) {
          onSceneClick(pickInfo);
        }
      }

      // Component-level callback with simple index
      component.onClick?.(elementIdx);
    }
  }

  /******************************************************
   * E) Drag Info Builder
   ******************************************************/
  /**
   * Builds a DragInfo object from pick info and drag state.
   */
  function buildDragInfo(
    pickInfo: PickInfo,
    startInstanceCenter: [number, number, number],
    startScreen: { x: number; y: number },
    currentScreen: { x: number; y: number },
    currentPosition?: [number, number, number],
  ): DragInfo {
    const startPos = pickInfo.hit?.position ?? startInstanceCenter;
    const curPos = currentPosition ?? startPos;

    return {
      ...pickInfo,
      start: {
        position: startPos,
        instanceCenter: startInstanceCenter,
        screen: startScreen,
      },
      current: {
        position: curPos,
        screen: currentScreen,
      },
      delta: {
        position: [
          curPos[0] - startPos[0],
          curPos[1] - startPos[1],
          curPos[2] - startPos[2],
        ],
        screen: {
          x: currentScreen.x - startScreen.x,
          y: currentScreen.y - startScreen.y,
        },
      },
    };
  }

  /******************************************************
   * F) Mouse Handling
   ******************************************************/
  const draggingState = useRef<DraggingState | null>(null);

  // Helper functions to check event modifiers
  function hasModifiers(actual: string[], expected: string[]): boolean {
    if (actual.length !== expected.length) return false;

    const sortedActual = [...actual].sort();
    const sortedExpected = [...expected].sort();

    return isEqual(sortedActual, sortedExpected);
  }

  function eventHasModifiers(e: MouseEvent, expected: string[]): boolean {
    const modifiers: string[] = [];
    if (e.shiftKey) modifiers.push("shift");
    if (e.ctrlKey) modifiers.push("ctrl");
    if (e.altKey) modifiers.push("alt");
    if (e.metaKey) modifiers.push("meta");
    return hasModifiers(modifiers, expected);
  }

  // Add throttling for hover picking
  const throttledPickAtScreenXY = useCallback(
    throttle((x: number, y: number, mode: "hover" | "click") => {
      pickAtScreenXY(x, y, mode);
    }, 32), // ~30fps
    [pickAtScreenXY],
  );

  // Picking handler - always registered on canvas
  const handlePickingMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!canvasRef.current || draggingState.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      throttledPickAtScreenXY(x, y, "hover");
    },
    [throttledPickAtScreenXY],
  );

  // Drag handler - attached/detached directly during drag
  const handleDragMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Handle element drag if active
      if (elementDragState.current) {
        const {
          constraint,
          componentIdx,
          startPickInfo,
          startInstanceCenter,
          startScreenPos,
        } = elementDragState.current;
        const component = components[componentIdx];

        // Compute screen ray for current mouse position
        const ray = screenRay(
          x,
          y,
          { width: containerWidth, height: containerHeight },
          activeCameraRef.current!,
        );
        if (ray) {
          // Compute constrained position
          const constrainedPos = computeConstrainedPosition(ray, constraint);
          if (constrainedPos) {
            const dragInfo = buildDragInfo(
              { ...startPickInfo, event: "drag" },
              startInstanceCenter,
              startScreenPos,
              { x, y },
              constrainedPos as [number, number, number],
            );
            component.onDrag?.(dragInfo);
          }
        }
        return; // Don't do camera drag
      }

      // Handle camera drag
      if (!draggingState.current) return;
      const st = draggingState.current;
      st.x = x;
      st.y = y;
      if (e.button === 2 || hasModifiers(st.modifiers, ["shift"])) {
        handleCameraUpdate((cam) => pan(st));
      } else if (hasModifiers(st.modifiers, ["alt"])) {
        handleCameraUpdate((cam) => roll(st));
      } else if (st.button === 0) {
        handleCameraUpdate((cam) => orbit(st));
      }
    },
    [handleCameraUpdate, components, containerWidth, containerHeight],
  );

  const handleMouseUp = useCallback(
    (e: MouseEvent) => {
      // Handle element drag end
      if (elementDragState.current) {
        if (!canvasRef.current) return;
        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const {
          constraint,
          componentIdx,
          startPickInfo,
          startInstanceCenter,
          startScreenPos,
        } = elementDragState.current;
        const component = components[componentIdx];

        // Compute final constrained position
        const ray = screenRay(
          x,
          y,
          { width: containerWidth, height: containerHeight },
          activeCameraRef.current!,
        );
        let finalPosition: [number, number, number] | undefined;
        if (ray) {
          const constrainedPos = computeConstrainedPosition(ray, constraint);
          if (constrainedPos) {
            finalPosition = constrainedPos as [number, number, number];
          }
        }

        const dragInfo = buildDragInfo(
          { ...startPickInfo, event: "dragend" },
          startInstanceCenter,
          startScreenPos,
          { x, y },
          finalPosition,
        );
        component.onDragEnd?.(dragInfo);

        elementDragState.current = null;
        // Remove window listeners
        window.removeEventListener("mousemove", handleDragMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
        return;
      }

      // Handle camera drag end
      const st = draggingState.current;
      if (st) {
        if (!canvasRef.current) return;
        const dx = st.x! - st.startX;
        const dy = st.y! - st.startY;
        const dragDistance = Math.sqrt(dx * dx + dy * dy);
        if ((dragDistance || 0) < 4) {
          pickAtScreenXY(st.x!, st.y!, "click");
        }
        // Remove window listeners
        window.removeEventListener("mousemove", handleDragMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      }
      draggingState.current = null;
    },
    [
      pickAtScreenXY,
      handleDragMouseMove,
      components,
      containerWidth,
      containerHeight,
    ],
  );

  const handleScene3dMouseDown = useCallback(
    (e: MouseEvent) => {
      if (!canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();

      const modifiers: string[] = [];
      if (e.shiftKey) modifiers.push("shift");
      if (e.ctrlKey) modifiers.push("ctrl");
      if (e.altKey) modifiers.push("alt");

      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Check if we're clicking on a draggable element (use lastHoverState)
      if (lastHoverState.current && e.button === 0 && modifiers.length === 0) {
        const { componentIdx, elementIdx } = lastHoverState.current;
        const component = components[componentIdx];

        if (hasDragCallbacks(component)) {
          // This is an element drag, not a camera drag
          const spec = primitiveRegistry[component.type];

          // Build pick info for drag start
          const pickInfo = buildPickInfo({
            mode: "hover", // Will be overridden to "dragstart"
            componentIndex: componentIdx,
            elementIndex: elementIdx,
            screenX: x,
            screenY: y,
            rect: { width: containerWidth, height: containerHeight },
            camera: activeCameraRef.current!,
            component,
          });

          if (pickInfo) {
            // Get instance center
            const centers = spec.getCenters(component);
            const baseIdx = elementIdx * 3;
            const instanceCenter: [number, number, number] = [
              centers[baseIdx],
              centers[baseIdx + 1],
              centers[baseIdx + 2],
            ];

            // Resolve constraint
            const constraint = resolveConstraint(
              component.dragConstraint,
              pickInfo,
              instanceCenter,
              activeCameraRef.current!,
            );

            if (constraint) {
              elementDragState.current = {
                componentIdx,
                elementIdx,
                constraint,
                startPickInfo: { ...pickInfo, event: "dragstart" },
                startInstanceCenter: instanceCenter,
                startScreenPos: { x, y },
              };

              // Build DragInfo and call onDragStart
              const dragInfo = buildDragInfo(
                { ...pickInfo, event: "dragstart" },
                instanceCenter,
                { x, y },
                { x, y },
              );
              component.onDragStart?.(dragInfo);

              // Add window listeners for drag
              window.addEventListener("mousemove", handleDragMouseMove);
              window.addEventListener("mouseup", handleMouseUp);

              e.preventDefault();
              return; // Don't set up camera drag
            }
          }
        }
      }

      // Set up camera drag
      draggingState.current = {
        button: e.button,
        startX: x,
        startY: y,
        x: x,
        y: y,
        rect: rect,
        modifiers,
        startCam: activeCameraRef.current!,
      };

      // Add window listeners immediately when drag starts
      window.addEventListener("mousemove", handleDragMouseMove);
      window.addEventListener("mouseup", handleMouseUp);

      e.preventDefault();
    },
    [
      handleDragMouseMove,
      handleMouseUp,
      components,
      containerWidth,
      containerHeight,
    ],
  );

  // Update canvas event listener references - only for picking and mousedown
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener("mousemove", handlePickingMouseMove);
    canvas.addEventListener("mousedown", handleScene3dMouseDown);

    return () => {
      canvas.removeEventListener("mousemove", handlePickingMouseMove);
      canvas.removeEventListener("mousedown", handleScene3dMouseDown);
    };
  }, [handlePickingMouseMove, handleScene3dMouseDown]);

  /******************************************************
   * F) Lifecycle & Render-on-demand
   ******************************************************/
  // Init once
  useEffect(() => {
    initWebGPU();
    return () => {
      if (gpuRef.current) {
        const { device, resources, pipelineCache } = gpuRef.current;

        device.queue.onSubmittedWorkDone().then(() => {
          for (const resource of Object.values(resources)) {
            if (resource) {
              resource.vb.destroy();
              resource.ib.destroy();
            }
          }

          // Clear instance pipeline cache
          pipelineCache.clear();
        });
      }
    };
  }, [initWebGPU]);

  // Create/recreate depth + pick textures
  useEffect(() => {
    if (isReady && gpuRef.current) {
      createOrUpdateDepthTexture();
      createOrUpdatePickTextures();
      if (anyOutlineEnabled) {
        const hadOutline = !!gpuRef.current.outlineTexture;
        createOrUpdateOutlineTextures();
        if (!hadOutline) {
          requestRender("outline-init");
        }
      }
    }
  }, [
    isReady,
    containerWidth,
    containerHeight,
    createOrUpdateDepthTexture,
    createOrUpdatePickTextures,
    createOrUpdateOutlineTextures,
    anyOutlineEnabled,
  ]);

  // Update canvas size effect
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = Math.floor(containerWidth * dpr);
    const displayHeight = Math.floor(containerHeight * dpr);

    // Only update if size actually changed
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;

      // Update textures after canvas size change
      createOrUpdateDepthTexture();
      createOrUpdatePickTextures();
      if (anyOutlineEnabled) {
        createOrUpdateOutlineTextures();
      }
      requestRender("canvas");
    }
  }, [
    containerWidth,
    containerHeight,
    createOrUpdateDepthTexture,
    createOrUpdatePickTextures,
    createOrUpdateOutlineTextures,
    anyOutlineEnabled,
    renderFrame,
  ]);

  // Render when camera or components change
  useEffect(() => {
    if (isReady && gpuRef.current) {
      if (
        !deepEqualModuloTypedArrays(
          components,
          gpuRef.current.renderedComponents,
        )
      ) {
        renderFrame("components changed", activeCameraRef.current!, components);
      } else if (
        !deepEqualModuloTypedArrays(
          activeCameraRef.current,
          gpuRef.current.renderedCamera,
        )
      ) {
        requestRender("camera changed");
      }
    }
  }, [isReady, components, activeCameraRef.current]);

  // Wheel handling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (e: WheelEvent) => {
      if (!draggingState.current) {
        e.preventDefault();
        handleCameraUpdate((cam) => {
          if (eventHasModifiers(e, ["alt"])) {
            return adjustFov(cam, e.deltaY);
          } else if (eventHasModifiers(e, ["ctrl"])) {
            return dolly(cam, e.deltaY);
          } else {
            return zoom(cam, e.deltaY);
          }
        });
      }
    };

    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [handleCameraUpdate]);

  return (
    <div style={{ width: "100%", position: "relative" }}>
      <canvas ref={canvasRef} style={{ border: "none", ...style }} />
    </div>
  );
}

function componentHasAlpha(component: ComponentConfig) {
  return (
    (component.alphas && component.alphas?.length > 0) ||
    (component.alpha && component.alpha !== 1.0) ||
    component.decorations?.some(
      (d) => d.alpha !== undefined && d.alpha !== 1.0 && d.indexes?.length > 0,
    )
  );
}
