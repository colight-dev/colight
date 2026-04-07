/// <reference types="react" />

import * as glMatrix from "gl-matrix";
import React, {
  // DO NOT require MouseEvent
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useContext,
} from "react";
import { throttle, deepEqualModuloTypedArrays } from "../utils";
import { $StateContext } from "../context";
import { useCanvasSnapshot } from "../canvasSnapshot";
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
} from "./camera3d";

import isEqual from "lodash-es/isEqual";

import { ellipsoidAxesSpec } from "./components/ring";
import {
  ComponentConfig,
  cuboidSpec,
  ellipsoidSpec,
  lineBeamsSpec,
  pointCloudSpec,
  buildPickingData,
  buildRenderData,
  createPrimitiveResourceKey,
  resolvePrimitiveSpec,
} from "./components";
import { unpackID } from "./picking";
import { LIGHTING } from "./shaders";
import {
  BufferInfo,
  GeometryResources,
  GeometryResource,
  PrimitiveDefinition,
  PrimitiveSpec,
  RenderObject,
  PipelineCacheEntry,
  DynamicBuffers,
  RenderObjectCache,
  ComponentOffset,
  Scene3DGeometryOptions,
  resolveScene3DGeometryOptions,
} from "./types";
import {
  buildDrawSlices,
  createScene3DDebugProbe,
  formatInstanceLimit,
  parseScene3DDebugOptions,
  serializeGpuLimits,
  type Scene3DDebugProbe,
  withGpuErrorScopes,
} from "./debug";

/**
 * Aligns a size or offset to 16 bytes, which is a common requirement for WebGPU buffers.
 * @param value The value to align
 * @returns The value aligned to the next 16-byte boundary
 */
function align16(value: number): number {
  return Math.ceil(value / 16) * 16;
}

let scene3DDebugIdCounter = 0;

function nextScene3DDebugId() {
  scene3DDebugIdCounter += 1;
  return `scene3d-${scene3DDebugIdCounter}`;
}

export interface SceneInnerProps {
  /** Array of 3D components to render in the scene */
  components: ComponentConfig[];

  /** Optional geometry configuration applied when base meshes are created */
  geometryOptions?: Partial<Scene3DGeometryOptions>;

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
  onReady: () => void;
}

function initGeometryResources(
  device: GPUDevice,
  resources: GeometryResources,
  geometryOptions: Scene3DGeometryOptions,
) {
  for (const entry of Object.values(primitiveRegistry)) {
    if ("resolveSpec" in entry) {
      for (const implementation of Object.values(entry.implementations)) {
        if (!implementation) continue;
        const resourceKey = createPrimitiveResourceKey(
          entry.type,
          implementation.mode,
        );
        if (!resources[resourceKey]) {
          resources[resourceKey] = implementation.createGeometryResource(
            device,
            geometryOptions,
          );
        }
      }
    } else if (!resources[entry.resourceKey]) {
      resources[entry.resourceKey] = entry.createGeometryResource(
        device,
        geometryOptions,
      );
    }
  }
}

type PrimitiveRegistryEntry = PrimitiveDefinition<any> | PrimitiveSpec<any>;

const primitiveRegistry: Record<
  ComponentConfig["type"],
  PrimitiveRegistryEntry
> = {
  PointCloud: pointCloudSpec,
  Ellipsoid: ellipsoidSpec,
  EllipsoidAxes: ellipsoidAxesSpec,
  Cuboid: cuboidSpec,
  LineBeams: lineBeamsSpec,
};

function ensurePickingData(
  device: GPUDevice,
  components: ComponentConfig[],
  ro: RenderObject,
) {
  if (!ro.pickingDataStale) return;

  const { pickingData, componentOffsets, spec, sortedPositions } = ro;

  for (let i = 0; i < componentOffsets.length; i++) {
    const offset = componentOffsets[i];
    const component = components[offset.componentIdx];
    buildPickingData(
      component,
      spec,
      pickingData,
      offset.pickingStart,
      offset.elementStart,
      sortedPositions,
    );
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
  tanHalfFov: number;
  forward: glMatrix.vec3;
  right: glMatrix.vec3;
  camUp: glMatrix.vec3;
  lightDir: glMatrix.vec3;
} {
  const aspect = containerWidth / containerHeight;
  const tanHalfFov = Math.tan(glMatrix.glMatrix.toRadian(camState.fov) * 0.5);
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

  return {
    aspect,
    view,
    proj,
    mvp,
    tanHalfFov,
    forward,
    right,
    camUp,
    lightDir,
  };
}

async function renderPass({
  device,
  context,
  depthTexture,
  pickTexture,
  renderObjects,
  uniformBindGroup,
  debugProbe,
  maxInstancesPerDraw,
}: {
  device: GPUDevice;
  context: GPUCanvasContext;
  depthTexture: GPUTexture | null;
  pickTexture: GPUTexture | null;
  renderObjects: RenderObject[];
  uniformBindGroup: GPUBindGroup;
  debugProbe?: Scene3DDebugProbe | null;
  maxInstancesPerDraw: number;
}) {
  try {
    return await withGpuErrorScopes(
      device,
      debugProbe,
      "scene3d/render-pass",
      {
        renderObjectCount: renderObjects.length,
        maxInstancesPerDraw: Number.isFinite(maxInstancesPerDraw)
          ? maxInstancesPerDraw
          : "unlimited",
      },
      async () => {
        const cmd = device.createCommandEncoder({ label: "scene3d/render" });
        const pass = cmd.beginRenderPass({
          colorAttachments: [
            {
              view: context.getCurrentTexture().createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: "clear",
              storeOp: "store",
            },
            {
              view: pickTexture!.createView(),
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

        for (const ro of renderObjects) {
          const drawSlices = buildDrawSlices(
            ro.instanceCount,
            maxInstancesPerDraw,
          );
          if (debugProbe?.options.verbose) {
            debugProbe.record("render-draw", `${ro.spec.type} render`, {
              instanceCount: ro.instanceCount,
              drawCalls: drawSlices.length,
              indexCount: ro.indexCount,
              firstInstance:
                drawSlices.length > 1 ? drawSlices[0].firstInstance : 0,
            });
          }

          pass.setPipeline(ro.pipeline);
          pass.setBindGroup(0, uniformBindGroup);
          pass.setVertexBuffer(0, ro.geometryBuffer);
          pass.setVertexBuffer(
            1,
            ro.instanceBuffer.buffer,
            ro.instanceBuffer.offset,
          );
          pass.setVertexBuffer(
            2,
            ro.pickingInstanceBuffer.buffer,
            ro.pickingInstanceBuffer.offset,
          );

          if (ro.indexBuffer) {
            pass.setIndexBuffer(ro.indexBuffer, ro.indexFormat);
            drawSlices.forEach((slice) => {
              pass.drawIndexed(
                ro.indexCount,
                slice.instanceCount,
                0,
                0,
                slice.firstInstance,
              );
            });
          } else if (ro.vertexCount) {
            drawSlices.forEach((slice) => {
              pass.draw(
                ro.vertexCount!,
                slice.instanceCount,
                0,
                slice.firstInstance,
              );
            });
          }
        }

        pass.end();
        device.queue.submit([cmd.finish()]);
        return await device.queue.onSubmittedWorkDone();
      },
    );
  } catch (err) {
    console.error(err);
    debugProbe?.error("scene3d/render-pass", err);
    throw err;
  }
}

function computeUniformData(
  viewportWidth: number,
  viewportHeight: number,
  camState: CameraState,
): Float32Array {
  const { mvp, tanHalfFov, aspect, forward, right, camUp, lightDir } =
    computeUniforms(viewportWidth, viewportHeight, camState);
  return new Float32Array([
    ...Array.from(mvp),
    right[0],
    right[1],
    right[2],
    tanHalfFov,
    camUp[0],
    camUp[1],
    camUp[2],
    aspect,
    lightDir[0],
    lightDir[1],
    lightDir[2],
    0, // pad to vec4
    camState.position[0],
    camState.position[1],
    camState.position[2],
    0,
    forward[0],
    forward[1],
    forward[2],
    0,
    viewportWidth,
    viewportHeight,
    0,
    0,
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
  resourceKey: string,
): GeometryResource {
  const resource = resources[resourceKey];
  if (!resource) {
    throw new Error(
      `No geometry resource found for primitive resource key ${resourceKey}`,
    );
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
  geometryOptions,
  containerWidth,
  containerHeight,
  style,
  camera: controlledCamera,
  defaultCamera,
  onCameraChange,
  onFrameRendered,
  onReady,
}: SceneInnerProps) {
  const $state = useContext($StateContext);
  const sceneDebugOptionsRef =
    useRef<ReturnType<typeof parseScene3DDebugOptions>>();
  if (!sceneDebugOptionsRef.current) {
    sceneDebugOptionsRef.current = parseScene3DDebugOptions();
  }

  const sceneDebugRef = useRef<Scene3DDebugProbe>();
  if (!sceneDebugRef.current) {
    sceneDebugRef.current = createScene3DDebugProbe(
      nextScene3DDebugId(),
      sceneDebugOptionsRef.current,
    );
  }
  const maxInstancesPerDraw = sceneDebugOptionsRef.current.maxInstancesPerDraw;
  const resolvedGeometryOptions = useMemo(
    () => resolveScene3DGeometryOptions(geometryOptions),
    [
      geometryOptions?.ellipsoidStacks,
      geometryOptions?.ellipsoidSlices,
      geometryOptions?.ellipsoidAxesMajorSegments,
      geometryOptions?.ellipsoidAxesMinorSegments,
    ],
  );

  // We'll store references to the GPU + other stuff in a ref object
  const gpuRef = useRef<{
    device: GPUDevice;
    context: GPUCanvasContext;
    uniformBuffer: GPUBuffer;
    uniformBindGroup: GPUBindGroup;
    bindGroupLayout: GPUBindGroupLayout;
    depthTexture: GPUTexture | null;
    pickTexture: GPUTexture | null;
    readbackBuffer: GPUBuffer;

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
      const { device, uniformBindGroup, renderObjects, pickTexture } =
        gpuRef.current;
      if (!pickTexture) return;

      // Reuse the existing renderPass function with a temporary context
      // that redirects rendering to our snapshot texture
      const tempContext = {
        getCurrentTexture: () => targetTexture,
      } as GPUCanvasContext;

      return renderPass({
        device,
        context: tempContext,
        depthTexture: depthTexture || null,
        pickTexture,
        renderObjects,
        uniformBindGroup,
        debugProbe: sceneDebugRef.current,
        maxInstancesPerDraw,
      });
    },
    [
      containerWidth,
      containerHeight,
      activeCameraRef.current!,
      maxInstancesPerDraw,
    ],
  );

  const { canvasRef } = useCanvasSnapshot(
    gpuRef.current?.device,
    gpuRef.current?.context,
    renderToTexture,
  );

  const [isReady, setIsReady] = useState(false);

  const pickingLockRef = useRef(false);
  const lastHoverScreenPositionRef = useRef<{ x: number; y: number } | null>(
    null,
  );

  const lastHoverState = useRef<{
    componentIdx: number;
    elementIdx: number;
  } | null>(null);

  const renderObjectCache = useRef<RenderObjectCache>({});

  /******************************************************
   * A) initWebGPU
   ******************************************************/
  const initWebGPU = useCallback(async () => {
    if (!canvasRef.current) return;
    if (!navigator.gpu) {
      console.error("[Debug] WebGPU not supported in this browser.");
      sceneDebugRef.current.error(
        "scene3d/init/no-webgpu",
        new Error("navigator.gpu is not available"),
      );
      return;
    }
    try {
      const adapter = await requestAdapterWithRetry();
      if (isDisposedRef.current) return;
      sceneDebugRef.current.snapshot(
        "adapterLimits",
        serializeGpuLimits(adapter.limits),
      );

      const device = await adapter.requestDevice().catch((err) => {
        console.error("[Debug] Failed to create WebGPU device:", err);
        sceneDebugRef.current.error("scene3d/init/request-device", err);
        throw err;
      });
      if (isDisposedRef.current || !canvasRef.current) return;

      sceneDebugRef.current.snapshot("device", {
        maxInstancesPerDraw: Number.isFinite(maxInstancesPerDraw)
          ? maxInstancesPerDraw
          : null,
        maxInstancesPerDrawLabel: formatInstanceLimit(maxInstancesPerDraw),
        userAgent: sceneDebugOptionsRef.current.userAgent,
      });
      sceneDebugRef.current.snapshot(
        "deviceLimits",
        serializeGpuLimits(device.limits),
      );

      // Add error handling for uncaptured errors
      device.addEventListener("uncapturederror", ((event: Event) => {
        if (event instanceof GPUUncapturedErrorEvent) {
          console.error("Uncaptured WebGPU error:", event.error);
          // Log additional context about where the error occurred
          console.error("Error source:", event.error.message);
          sceneDebugRef.current.error("scene3d/uncapturederror", event.error, {
            source: event.error.message,
          });
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

      const uniformBufferSize = 160;
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
        readbackBuffer,
        renderObjects: [],
        pipelineCache: new Map(),
        dynamicBuffers: null,
        resources: {},
      };

      // Now initialize geometry resources
      sceneDebugRef.current.snapshot(
        "geometryOptions",
        resolvedGeometryOptions,
      );
      initGeometryResources(
        device,
        gpuRef.current.resources,
        resolvedGeometryOptions,
      );
      sceneDebugRef.current.record("lifecycle", "scene3d initialized", {
        canvasWidth: canvasRef.current.width,
        canvasHeight: canvasRef.current.height,
      });

      setIsReady(true);
    } catch (err) {
      console.error("[Debug] Error during WebGPU initialization:", err);
      sceneDebugRef.current.error("scene3d/init", err);
    }
  }, [maxInstancesPerDraw]);

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
    const { device, pickTexture } = gpuRef.current;

    // Get the actual canvas size
    const canvas = canvasRef.current;
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;

    if (pickTexture) pickTexture.destroy();

    const colorTex = device.createTexture({
      size: [displayWidth, displayHeight],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    gpuRef.current.pickTexture = colorTex;
  }, []);

  type RenderGroupKey = string;

  interface TypeInfo {
    spec: PrimitiveSpec<any>;
    offsets: number[];
    elementCounts: number[];
    indices: number[];
    totalRenderSize: number;
    totalPickingSize: number;
    totalElementCount: number;
    components: ComponentConfig[];
    elementOffsets: number[];
  }

  // Update the collectTypeData function signature
  function collectTypeData(
    components: ComponentConfig[],
  ): Map<RenderGroupKey, TypeInfo> {
    const typeArrays = new Map<RenderGroupKey, TypeInfo>();

    // Single pass through components
    components.forEach((comp, idx) => {
      const entry = primitiveRegistry[comp.type];
      if (!entry) return;
      const spec = resolvePrimitiveSpec(entry as any, comp as any);
      const groupKey = spec.resourceKey;

      // Get the element count and instance count
      const elementCount = spec.getElementCount(comp);

      if (elementCount === 0) return;
      const instanceCount = elementCount * spec.instancesPerElement;

      // Just allocate the array without building data, 4 bytes per float
      const renderSize =
        instanceCount * spec.floatsPerInstance * Float32Array.BYTES_PER_ELEMENT;
      const pickingSize =
        instanceCount * spec.floatsPerPicking * Float32Array.BYTES_PER_ELEMENT;

      let typeInfo = typeArrays.get(groupKey);
      if (!typeInfo) {
        typeInfo = {
          spec,
          totalElementCount: 0,
          totalRenderSize: 0,
          totalPickingSize: 0,
          components: [],
          indices: [],
          offsets: [],
          elementCounts: [],
          elementOffsets: [],
        };
        typeArrays.set(groupKey, typeInfo);
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

    const typeArrays = collectTypeData(components);
    const activeGroupKeys = new Set(typeArrays.keys());

    // Clear out unused cache entries
    Object.keys(renderObjectCache.current).forEach((type) => {
      if (!activeGroupKeys.has(type)) {
        delete renderObjectCache.current[type];
      }
    });

    // Track global start index for all components
    let globalStartIndex = 0;

    // Calculate total buffer sizes needed
    let totalRenderSize = 0;
    let totalPickingSize = 0;
    typeArrays.forEach((info: TypeInfo) => {
      const spec = info.spec;

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
    const maxBufferSize =
      device.limits?.maxBufferSize ?? Number.POSITIVE_INFINITY;

    sceneDebugRef.current.snapshot("bufferPlan", {
      componentCount: components.length,
      totalRenderSize,
      totalPickingSize,
      maxBufferSize: Number.isFinite(maxBufferSize) ? maxBufferSize : null,
    });

    if (totalRenderSize > maxBufferSize || totalPickingSize > maxBufferSize) {
      const error = new Error(
        `scene3d dynamic buffers exceed device maxBufferSize (render=${totalRenderSize}, picking=${totalPickingSize}, limit=${maxBufferSize})`,
      );
      sceneDebugRef.current.error("scene3d/buffer-limit", error, {
        totalRenderSize,
        totalPickingSize,
        maxBufferSize,
      });
      throw error;
    }

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
    typeArrays.forEach((info: TypeInfo, groupKey: RenderGroupKey) => {
      const spec = info.spec;

      try {
        // Ensure 4-byte alignment for all offsets
        const renderOffset = align16(dynamicBuffers.renderOffset);
        const pickingOffset = align16(dynamicBuffers.pickingOffset);

        // Try to get existing render object
        let renderObject = renderObjectCache.current[groupKey];
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

        // Get or create pipeline
        const pipeline = spec.getRenderPipeline(
          device,
          bindGroupLayout,
          pipelineCache,
        );
        if (!pipeline) return;
        const geometryResource = getGeometryResource(
          resources,
          spec.resourceKey,
        );

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

        const hasAlphaComponents = info.components.some(componentHasAlpha);

        if (needNewRenderObject) {
          // Create new render object with all the required resources
          renderObject = {
            pipeline,
            geometryBuffer: geometryResource.vb,
            instanceBuffer: bufferInfo,
            indexBuffer: geometryResource.ib,
            indexFormat: geometryResource.indexFormat,
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
            ...alphaProperties(hasAlphaComponents, info.totalElementCount),
          };
          renderObjectCache.current[groupKey] = renderObject;
        } else {
          // Update existing render object with new buffer info and state
          renderObject.geometryBuffer = geometryResource.vb;
          renderObject.indexBuffer = geometryResource.ib;
          renderObject.instanceBuffer = bufferInfo;
          renderObject.pickingInstanceBuffer = pickingBufferInfo;
          renderObject.indexFormat = geometryResource.indexFormat;
          renderObject.indexCount = geometryResource.indexCount;
          renderObject.vertexCount = geometryResource.vertexCount;
          renderObject.instanceCount = totalInstanceCount;
          renderObject.componentIndex = info.indices[0];
          renderObject.componentOffsets = typeComponentOffsets;
          renderObject.spec = spec;
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
        console.error(
          `Error creating render object for type ${groupKey}:`,
          error,
        );
        sceneDebugRef.current.error("scene3d/build-render-object", error, {
          type: groupKey,
        });
      }
    });

    sceneDebugRef.current.snapshot(
      "renderObjects",
      validRenderObjects.map((renderObject) => ({
        type: renderObject.spec.type,
        implementationMode: renderObject.spec.implementationMode,
        instanceCount: renderObject.instanceCount,
        indexCount: renderObject.indexCount,
        vertexCount: renderObject.vertexCount,
        renderOffset: renderObject.instanceBuffer.offset,
        renderBytes: renderObject.renderData.byteLength,
        pickingOffset: renderObject.pickingInstanceBuffer.offset,
        pickingBytes: renderObject.pickingData.byteLength,
        drawCalls: buildDrawSlices(
          renderObject.instanceCount,
          maxInstancesPerDraw,
        ).length,
      })),
    );

    return validRenderObjects;
  }

  /******************************************************
   * C) Render pass (single call, no loop)
   ******************************************************/

  const pendingAnimationFrameRef = useRef<number | null>(null);
  const renderInFlightRef = useRef(false);
  const isDisposedRef = useRef(false);
  const latestComponentsRef = useRef<ComponentConfig[]>(components);
  const pendingRenderRequestRef = useRef<{
    label: string;
    camState?: CameraState;
    components?: ComponentConfig[];
  } | null>(null);

  latestComponentsRef.current = components;

  const renderFrame = useCallback(
    async function renderFrameInner(
      source: string,
      camState?: CameraState,
      components?: ComponentConfig[],
    ) {
      if (!gpuRef.current || isDisposedRef.current) return;

      camState = camState || activeCameraRef.current!;

      const onRenderComplete = $state.beginUpdate("impl3d/renderFrame");

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
        pickTexture,
      } = gpuRef.current;
      if (!pickTexture) return;

      const cameraMoved = hasCameraMoved(
        camState.position,
        gpuRef.current.renderedCamera?.position,
        0.0001,
      );
      gpuRef.current.renderedCamera = camState;

      // Update data for objects that need it
      renderObjects.forEach(function updateRenderObject(ro) {
        const needsSorting = ro.hasAlphaComponents;
        const needsBuild = (needsSorting && cameraMoved) || componentsChanged;

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

          buildRenderData(
            component,
            ro.spec,
            renderData,
            offset.elementStart,
            ro.sortedPositions,
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

        ensurePickingData(device, components!, ro);
      });

      const viewportWidth = canvasRef.current?.width ?? containerWidth;
      const viewportHeight = canvasRef.current?.height ?? containerHeight;
      const uniformData = computeUniformData(
        viewportWidth,
        viewportHeight,
        camState,
      );
      device.queue.writeBuffer(uniformBuffer, 0, uniformData);
      sceneDebugRef.current.snapshot("lastRenderRequest", {
        source,
        componentCount: components?.length ?? 0,
        cameraPosition: Array.from(camState.position),
        drawLimit: formatInstanceLimit(maxInstancesPerDraw),
      });

      try {
        await renderPass({
          device,
          context,
          depthTexture,
          pickTexture,
          renderObjects,
          uniformBindGroup,
          debugProbe: sceneDebugRef.current,
          maxInstancesPerDraw,
        });
        const hoverPosition = lastHoverScreenPositionRef.current;
        if (!draggingState.current && hoverPosition) {
          await pickAtScreenXY(hoverPosition.x, hoverPosition.y, "hover");
        }
        onRenderComplete();
      } catch (err) {
        console.error("[Debug] Error during renderPass:", err.message);
        sceneDebugRef.current.error("scene3d/render-frame", err, { source });
        onRenderComplete();
      }

      if (isDisposedRef.current) return;
      onFrameRendered?.(performance.now());
      onReady();
    },
    [
      containerWidth,
      containerHeight,
      maxInstancesPerDraw,
      onFrameRendered,
      components,
    ],
  );

  const flushPendingRender = useCallback(() => {
    if (isDisposedRef.current) return;
    if (pendingAnimationFrameRef.current || renderInFlightRef.current) {
      return;
    }

    pendingAnimationFrameRef.current = requestAnimationFrame(() => {
      pendingAnimationFrameRef.current = null;
      if (isDisposedRef.current) return;

      if (renderInFlightRef.current) {
        flushPendingRender();
        return;
      }

      const request = pendingRenderRequestRef.current;
      if (!request) return;

      pendingRenderRequestRef.current = null;
      renderInFlightRef.current = true;

      void renderFrame(
        request.label,
        request.camState,
        request.components,
      ).finally(() => {
        renderInFlightRef.current = false;
        if (!isDisposedRef.current && pendingRenderRequestRef.current) {
          flushPendingRender();
        }
      });
    });
  }, [renderFrame]);

  const requestRender = useCallback(
    (
      label: string,
      camState: CameraState | undefined = activeCameraRef.current || undefined,
      nextComponents:
        | ComponentConfig[]
        | undefined = latestComponentsRef.current,
    ) => {
      if (isDisposedRef.current) return;
      pendingRenderRequestRef.current = {
        label,
        camState,
        components: nextComponents,
      };
      flushPendingRender();
    },
    [flushPendingRender],
  );

  /******************************************************
   * D) Pick pass (on hover/click)
   ******************************************************/
  async function pickAtScreenXY(
    screenX: number,
    screenY: number,
    mode: "hover" | "click",
  ) {
    if (!gpuRef.current || !canvasRef.current || pickingLockRef.current) return;
    pickingLockRef.current = true;

    try {
      const { device, pickTexture, readbackBuffer } = gpuRef.current;
      if (!pickTexture || !readbackBuffer) return;

      // Map from CSS pixels to the canvas backing texture.
      const canvasRect = canvasRef.current.getBoundingClientRect();
      const displayWidth = canvasRef.current.width;
      const displayHeight = canvasRef.current.height;
      const pickX = Math.floor((screenX / canvasRect.width) * displayWidth);
      const pickY = Math.floor((screenY / canvasRect.height) * displayHeight);

      if (
        pickX < 0 ||
        pickY < 0 ||
        pickX >= displayWidth ||
        pickY >= displayHeight
      ) {
        if (mode === "hover") handleHoverID(0);
        return;
      }

      sceneDebugRef.current.snapshot("lastPickRequest", {
        mode,
        pickX,
        pickY,
        canvasRectWidth: canvasRect.width,
        canvasRectHeight: canvasRect.height,
        textureWidth: displayWidth,
        textureHeight: displayHeight,
        maxInstancesPerDraw: formatInstanceLimit(maxInstancesPerDraw),
      });

      await withGpuErrorScopes(
        device,
        sceneDebugRef.current,
        "scene3d/pick-readback",
        {
          mode,
        },
        async () => {
          const cmd = device.createCommandEncoder({
            label: "Picking readback",
          });
          cmd.copyTextureToBuffer(
            { texture: pickTexture, origin: { x: pickX, y: pickY } },
            { buffer: readbackBuffer, bytesPerRow: 256, rowsPerImage: 1 },
            [1, 1, 1],
          );
          device.queue.submit([cmd.finish()]);
        },
      );

      await device.queue.onSubmittedWorkDone();
      await readbackBuffer.mapAsync(GPUMapMode.READ);
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
    } catch (error) {
      sceneDebugRef.current.error("scene3d/pick", error, { mode });
      throw error;
    } finally {
      pickingLockRef.current = false;
    }
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
        lastHoverState.current = null;
      }
      return;
    }

    // Find which component this instance belongs to by searching through all render objects
    let newHoverState = null;
    for (const ro of gpuRef.current.renderObjects) {
      for (const offset of ro.componentOffsets) {
        if (
          globalIdx >= offset.pickingStart &&
          globalIdx < offset.pickingStart + offset.elementCount
        ) {
          newHoverState = {
            componentIdx: offset.componentIdx,
            elementIdx: globalIdx - offset.pickingStart,
          };
          break;
        }
      }
      if (newHoverState) break; // Found the matching component
    }

    // If hover state hasn't changed, do nothing
    if (
      (!lastHoverState.current && !newHoverState) ||
      (lastHoverState.current &&
        newHoverState &&
        lastHoverState.current.componentIdx === newHoverState.componentIdx &&
        lastHoverState.current.elementIdx === newHoverState.elementIdx)
    ) {
      return;
    }

    // Clear previous hover if it exists
    if (lastHoverState.current) {
      const prevComponent = components[lastHoverState.current.componentIdx];
      prevComponent?.onHover?.(null);
    }

    // Set new hover if it exists
    if (newHoverState) {
      const { componentIdx, elementIdx } = newHoverState;
      if (componentIdx >= 0 && componentIdx < components.length) {
        components[componentIdx].onHover?.(elementIdx);
      }
    }

    // Update last hover state
    lastHoverState.current = newHoverState;
  }

  function handleClickID(pickedID: number) {
    if (!gpuRef.current) return;

    // Get combined instance index
    const globalIdx = unpackID(pickedID);
    if (globalIdx === null) return;

    // Find which component this instance belongs to by searching through all render objects
    for (const ro of gpuRef.current.renderObjects) {
      // Skip if no component offsets
      if (!ro?.componentOffsets) continue;

      // Check each component in this render object
      for (const offset of ro.componentOffsets) {
        if (
          globalIdx >= offset.pickingStart &&
          globalIdx < offset.pickingStart + offset.elementCount
        ) {
          const componentIdx = offset.componentIdx;
          const elementIdx = globalIdx - offset.pickingStart;
          if (componentIdx >= 0 && componentIdx < components.length) {
            components[componentIdx].onClick?.(elementIdx);
          }
          return; // Found and handled the click
        }
      }
    }
  }

  /******************************************************
   * E) Mouse Handling
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
      lastHoverScreenPositionRef.current = { x, y };
      throttledPickAtScreenXY(x, y, "hover");
    },
    [throttledPickAtScreenXY],
  );

  const handlePickingMouseLeave = useCallback(() => {
    lastHoverScreenPositionRef.current = null;
    handleHoverID(0);
  }, [components]);

  // Drag handler - attached/detached directly during drag
  const handleDragMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!canvasRef.current || !draggingState.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
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
    [handleCameraUpdate],
  );

  const handleMouseUp = useCallback(
    (e: MouseEvent) => {
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
    [pickAtScreenXY, handleDragMouseMove],
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
    [handleDragMouseMove, handleMouseUp],
  );

  // Update canvas event listener references - only for picking and mousedown
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener("mousemove", handlePickingMouseMove);
    canvas.addEventListener("mouseleave", handlePickingMouseLeave);
    canvas.addEventListener("mousedown", handleScene3dMouseDown);

    return () => {
      canvas.removeEventListener("mousemove", handlePickingMouseMove);
      canvas.removeEventListener("mouseleave", handlePickingMouseLeave);
      canvas.removeEventListener("mousedown", handleScene3dMouseDown);
    };
  }, [handlePickingMouseLeave, handlePickingMouseMove, handleScene3dMouseDown]);

  /******************************************************
   * F) Lifecycle & Render-on-demand
   ******************************************************/
  // Init once
  useEffect(() => {
    isDisposedRef.current = false;
    initWebGPU();
    return () => {
      isDisposedRef.current = true;
      if (pendingAnimationFrameRef.current !== null) {
        cancelAnimationFrame(pendingAnimationFrameRef.current);
        pendingAnimationFrameRef.current = null;
      }
      pendingRenderRequestRef.current = null;
      renderInFlightRef.current = false;
      pickingLockRef.current = false;
      sceneDebugRef.current.record("lifecycle", "scene3d dispose");
      if (gpuRef.current) {
        const {
          device,
          resources,
          pipelineCache,
          dynamicBuffers,
          depthTexture,
          pickTexture,
          readbackBuffer,
        } = gpuRef.current;

        device.queue.onSubmittedWorkDone().then(() => {
          depthTexture?.destroy();
          pickTexture?.destroy();
          readbackBuffer.destroy();
          dynamicBuffers?.renderBuffer.destroy();
          dynamicBuffers?.pickingBuffer.destroy();
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
      sceneDebugRef.current.dispose();
    };
  }, [initWebGPU]);

  // Create/recreate depth + pick textures
  useEffect(() => {
    if (isReady) {
      createOrUpdateDepthTexture();
      createOrUpdatePickTextures();
    }
  }, [
    isReady,
    containerWidth,
    containerHeight,
    createOrUpdateDepthTexture,
    createOrUpdatePickTextures,
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
      sceneDebugRef.current.snapshot("canvas", {
        cssWidth: containerWidth,
        cssHeight: containerHeight,
        pixelWidth: displayWidth,
        pixelHeight: displayHeight,
        devicePixelRatio: dpr,
      });

      // Update textures after canvas size change
      createOrUpdateDepthTexture();
      createOrUpdatePickTextures();
      requestRender("canvas");
    }
  }, [
    containerWidth,
    containerHeight,
    createOrUpdateDepthTexture,
    createOrUpdatePickTextures,
    requestRender,
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
        requestRender(
          "components changed",
          activeCameraRef.current || undefined,
          components,
        );
      } else if (
        !deepEqualModuloTypedArrays(
          activeCameraRef.current,
          gpuRef.current.renderedCamera,
        )
      ) {
        requestRender("camera changed");
      }
    }
  }, [isReady, components, requestRender, activeCameraRef.current]);

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
      <canvas
        ref={canvasRef}
        style={{
          border: "none",
          width: `${containerWidth}px`,
          height: `${containerHeight}px`,
          display: "block",
          ...style,
        }}
      />
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
