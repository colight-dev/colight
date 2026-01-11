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
  pointCloudSpec,
  buildPickingData,
  buildRenderData,
  createOverlayPipeline,
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
  PickEventType,
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

export interface SceneInnerProps {
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

  /** Scene-level hover callback. Called with PickEvent when hovering, null when not. */
  onHover?: (event: PickEventType | null) => void;

  /** Scene-level click callback. Called with PickEvent when an element is clicked. */
  onClick?: (event: PickEventType) => void;

  /** Default outline on hover for components that don't specify hoverOutline. Default: false */
  defaultHoverOutline?: boolean;

  /** Default outline color as RGB [0-1]. Default: [1, 1, 1] (white) */
  defaultOutlineColor?: [number, number, number];

  /** Default outline width in pixels. Default: 2 */
  defaultOutlineWidth?: number;

  /** Optional ready state for coordinating updates. Defaults to NOOP_READY_STATE. */
  readyState?: ReadyState;
}

function initGeometryResources(
  device: GPUDevice,
  resources: GeometryResources,
) {
  // Create geometry for each primitive type
  for (const [primitiveName, spec] of Object.entries(primitiveRegistry)) {
    const typedName = primitiveName as keyof GeometryResources;
    if (!resources[typedName]) {
      resources[typedName] = spec.createGeometryResource(device);
    }
  }
}

const primitiveRegistry: Record<ComponentConfig["type"], PrimitiveSpec<any>> = {
  PointCloud: pointCloudSpec,
  Ellipsoid: ellipsoidSpec,
  EllipsoidAxes: ellipsoidAxesSpec,
  Cuboid: cuboidSpec,
  LineBeams: lineBeamsSpec,
  BoundingBox: boundingBoxSpec,
};

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

async function renderPass({
  device,
  context,
  depthTexture,
  renderObjects,
  uniformBindGroup,
}: {
  device: GPUDevice;
  context: GPUCanvasContext;
  depthTexture: GPUTexture | null;
  renderObjects: RenderObject[];
  uniformBindGroup: GPUBindGroup;
}) {
  try {
    // Begin render pass
    const cmd = device.createCommandEncoder();
    const pass = cmd.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
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
      pass.setVertexBuffer(0, ro.geometryBuffer);
      pass.setVertexBuffer(
        1,
        ro.instanceBuffer.buffer,
        ro.instanceBuffer.offset,
      );
      pass.setIndexBuffer(ro.indexBuffer, "uint16");
      pass.drawIndexed(ro.indexCount, ro.instanceCount);
    }

    pass.end();
    device.queue.submit([cmd.finish()]);
    return device.queue.onSubmittedWorkDone();
  } catch (err) {
    console.error(err);
  }
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
  type: keyof GeometryResources,
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

export function SceneInner({
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
  defaultHoverOutline = false,
  defaultOutlineColor = [1, 1, 1],
  defaultOutlineWidth = 2,
  readyState = NOOP_READY_STATE,
}: SceneInnerProps) {
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

  // Create a render callback for the canvas snapshot system
  // This function is called during PDF export to render the 3D scene to a texture
  // that can be captured as a static image
  const renderToTexture = useCallback(
    async (targetTexture: GPUTexture, depthTexture: GPUTexture | null) => {
      if (!gpuRef.current) return;
      const { device, uniformBindGroup, renderObjects } = gpuRef.current;

      // Reuse the existing renderPass function with a temporary context
      // that redirects rendering to our snapshot texture
      const tempContext = {
        getCurrentTexture: () => targetTexture,
      } as GPUCanvasContext;

      return renderPass({
        device,
        context: tempContext,
        depthTexture: depthTexture || null,
        renderObjects,
        uniformBindGroup,
      });
    },
    [containerWidth, containerHeight, activeCameraRef.current!],
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
          BoundingBox: null,
        },
      };

      // Now initialize geometry resources
      initGeometryResources(device, gpuRef.current.resources);

      setIsReady(true);
    } catch (err) {
      console.error("[Debug] Error during WebGPU initialization:", err);
    }
  }, []);

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

      // Use async pipeline creation to catch errors properly
      device
        .createRenderPipelineAsync({
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
        })
        .then((pipeline) => {
          if (gpuRef.current) {
            gpuRef.current.outlinePipeline = pipeline;
          }
        })
        .catch((err) => {
          console.error("Failed to create outline pipeline:", err);
        });
    }
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
  }

  /**
   * Creates a key for type+layer combination.
   * This allows us to have separate RenderObjects for scene vs overlay components of the same type.
   */
  function typeLayerKey(type: ComponentType, layer: RenderLayer): string {
    return `${type}_${layer}`;
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
      const key = typeLayerKey(comp.type, layer);

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

    // Track global start index for all components
    let globalStartIndex = 0;

    // Calculate total buffer sizes needed
    let totalRenderSize = 0;
    let totalPickingSize = 0;
    typeArrays.forEach((info: TypeInfo, key: string) => {
      // Extract the component type from the key (format: "type_layer")
      const type = info.components[0]?.type;
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
      // Extract the component type from the first component
      const type = info.components[0]?.type;
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

        if (needNewRenderObject) {
          // Create new render object with all the required resources
          const geometryResource = getGeometryResource(resources, type);
          renderObject = {
            pipeline,
            pickingPipeline,
            geometryBuffer: geometryResource.vb,
            instanceBuffer: bufferInfo,
            indexBuffer: geometryResource.ib,
            indexCount: geometryResource.indexCount,
            instanceCount: totalInstanceCount,
            vertexCount: geometryResource.vertexCount,
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
   * Renders the silhouette of a single hovered element to the outline texture.
   */
  function renderSilhouettePass(
    hoveredState: { componentIdx: number; elementIdx: number },
    components: ComponentConfig[],
  ): void {
    if (!gpuRef.current) return;

    const {
      device,
      outlineTexture,
      outlineDepthTexture,
      uniformBindGroup,
      renderObjects,
    } = gpuRef.current;

    if (!outlineTexture || !outlineDepthTexture) return;

    const { componentIdx, elementIdx } = hoveredState;
    const component = components[componentIdx];
    if (!component) return;

    // Find the render object for this component type
    const ro = renderObjects.find((r) =>
      r.componentOffsets.some((o) => o.componentIdx === componentIdx),
    );
    if (!ro) return;

    // Find the component offset
    const compOffset = ro.componentOffsets.find(
      (o) => o.componentIdx === componentIdx,
    );
    if (!compOffset) return;

    // Calculate the instance index within the render object's buffer
    // Must account for transparency sorting - render data is in sorted order
    const baseOffset = compOffset.elementStart;
    const originalIndex = baseOffset + elementIdx;

    // If sorted, convert original index to sorted position
    const sortedIndex = ro.sortedPositions
      ? ro.sortedPositions[originalIndex]
      : originalIndex;

    const instancesPerElement = ro.spec.instancesPerElement;

    // Create a small buffer with just this one element's instance data
    const floatsPerInstance = ro.spec.floatsPerInstance;
    const instanceDataSize =
      floatsPerInstance * instancesPerElement * Float32Array.BYTES_PER_ELEMENT;
    const tempBuffer = device.createBuffer({
      size: align16(instanceDataSize),
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      label: "Silhouette temp instance buffer",
    });

    // Copy the instance data for this element (using sorted index for correct position)
    const startOffset = sortedIndex * instancesPerElement * floatsPerInstance;
    const instanceData = ro.renderData.subarray(
      startOffset,
      startOffset + floatsPerInstance * instancesPerElement,
    );
    device.queue.writeBuffer(
      tempBuffer,
      0,
      instanceData.buffer,
      instanceData.byteOffset,
      instanceData.byteLength,
    );

    // Get or create silhouette pipeline
    // For silhouette, we can reuse the regular render pipeline but with a simple white fragment shader
    // However, the fragment output needs to match - let's use the picking pipeline's vertex shader
    // since it has the same transforms but simpler attributes

    const cmd = device.createCommandEncoder({ label: "Silhouette encoder" });

    const pass = cmd.beginRenderPass({
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

    // Use the regular render pipeline - the fragment shader will still output lit color,
    // but we only care about coverage for the silhouette
    pass.setPipeline(ro.pipeline);
    pass.setBindGroup(0, uniformBindGroup);
    pass.setVertexBuffer(0, ro.geometryBuffer);
    pass.setVertexBuffer(1, tempBuffer, 0);
    pass.setIndexBuffer(ro.indexBuffer, "uint16");
    pass.drawIndexed(ro.indexCount, instancesPerElement);

    pass.end();
    device.queue.submit([cmd.finish()]);

    // Clean up temp buffer
    tempBuffer.destroy();
  }

  /**
   * Renders the outline post-process pass, compositing the outline over the canvas.
   */
  function renderOutlinePostProcess(
    outlineColorVal: [number, number, number],
    outlineWidthVal: number,
  ): void {
    if (!gpuRef.current) return;

    const {
      device,
      context,
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

    const cmd = device.createCommandEncoder({ label: "Outline post-process" });

    const pass = cmd.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "load", // Keep existing content
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(outlinePipeline);
    pass.setBindGroup(0, outlineBindGroup);
    pass.draw(3); // Full-screen triangle

    pass.end();
    device.queue.submit([cmd.finish()]);
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

        // Submit all render passes before waiting
        const renderPromise = renderPass({
          device,
          context,
          depthTexture,
          renderObjects: sortedRenderObjects,
          uniformBindGroup,
        });

        // Render hover outline if enabled and something is hovered
        // Must happen BEFORE awaiting to use the same swap chain texture
        if (lastHoverState.current && components) {
          const hoveredComponent =
            components[lastHoverState.current.componentIdx];
          // Resolve hoverOutline: component-level takes precedence over scene-level
          const shouldOutline =
            hoveredComponent?.hoverOutline ?? defaultHoverOutline;

          if (shouldOutline) {
            // Resolve outline styling: component-level takes precedence
            const effectiveOutlineColor =
              hoveredComponent?.outlineColor ?? defaultOutlineColor;
            const effectiveOutlineWidth =
              hoveredComponent?.outlineWidth ?? defaultOutlineWidth;

            renderSilhouettePass(lastHoverState.current, components);
            renderOutlinePostProcess(
              effectiveOutlineColor,
              effectiveOutlineWidth,
            );
          }
        }

        // Now wait for all GPU work to complete
        await renderPromise;
        onRenderComplete();
      } catch (err) {
        console.error(
          "[Debug] Error during renderPass:",
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
      defaultHoverOutline,
      defaultOutlineColor,
      defaultOutlineWidth,
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
        pass.setVertexBuffer(0, ro.geometryBuffer);
        pass.setVertexBuffer(
          1,
          ro.pickingInstanceBuffer.buffer,
          ro.pickingInstanceBuffer.offset,
        );

        // Draw with indices if we have them, otherwise use vertex count
        if (ro.indexBuffer) {
          pass.setIndexBuffer(ro.indexBuffer, "uint16");
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

  /**
   * Builds a PickEvent for the given component and element.
   */
  function buildPickEvent(
    componentIdx: number,
    elementIdx: number,
    spec: PrimitiveSpec<any>,
  ): PickEventType {
    const component = components[componentIdx];
    const centers = spec.getCenters(component);
    const baseIdx = elementIdx * 3;

    return {
      type: spec.type,
      id: component.id,
      index: elementIdx,
      position: [
        centers[baseIdx],
        centers[baseIdx + 1],
        centers[baseIdx + 2],
      ] as [number, number, number],
    };
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
        prevComponent?.onHoverDetail?.(null);
        onSceneHover?.(null);

        // Clear hoverProps state if the previous component had hoverProps
        if (prevComponent?.hoverProps) {
          hoverPropsState.current.delete(lastHoverState.current.componentIdx);
          hoverPropsDirty.current = true;
        }

        const hadHover = lastHoverState.current !== null;
        // Check if previous component had outline enabled
        const prevHadOutline =
          prevComponent?.hoverOutline ?? defaultHoverOutline;
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
      prevComponent?.onHoverDetail?.(null);

      // Clear hoverProps state for previous component
      if (prevComponent?.hoverProps) {
        hoverPropsState.current.delete(lastHoverState.current.componentIdx);
        hoverPropsDirty.current = true;
      }
    }

    // Set new hover if it exists
    if (found) {
      const { componentIdx, elementIdx, spec } = found;
      if (componentIdx >= 0 && componentIdx < components.length) {
        const event = buildPickEvent(componentIdx, elementIdx, spec);
        onSceneHover?.(event);
        components[componentIdx].onHover?.(elementIdx);

        // Call detailed hover callback if defined
        const component = components[componentIdx];
        if (component.onHoverDetail) {
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
          if (pickInfo) {
            component.onHoverDetail(pickInfo);
          }
        }

        // Update hoverProps state for new component
        const newComponent = components[componentIdx];
        if (newComponent?.hoverProps) {
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
    const newHasOutline = newComponent?.hoverOutline ?? defaultHoverOutline;
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

    const { componentIdx, elementIdx, spec } = found;
    if (componentIdx >= 0 && componentIdx < components.length) {
      const event = buildPickEvent(componentIdx, elementIdx, spec);
      onSceneClick?.(event);
      components[componentIdx].onClick?.(elementIdx);

      // Call detailed click callback if defined
      const component = components[componentIdx];
      if (component.onClickDetail) {
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
          component.onClickDetail(pickInfo);
        }
      }
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

  // Check if any component has hoverOutline enabled, or if scene default is enabled
  const anyOutlineEnabled = useMemo(() => {
    if (defaultHoverOutline) return true;
    return components.some((c) => c.hoverOutline === true);
  }, [components, defaultHoverOutline]);

  // Create/recreate depth + pick textures
  useEffect(() => {
    if (isReady) {
      createOrUpdateDepthTexture();
      createOrUpdatePickTextures();
      if (anyOutlineEnabled) {
        createOrUpdateOutlineTextures();
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

export { SceneInner as SceneImpl };

function componentHasAlpha(component: ComponentConfig) {
  return (
    (component.alphas && component.alphas?.length > 0) ||
    (component.alpha && component.alpha !== 1.0) ||
    component.decorations?.some(
      (d) => d.alpha !== undefined && d.alpha !== 1.0 && d.indexes?.length > 0,
    )
  );
}
