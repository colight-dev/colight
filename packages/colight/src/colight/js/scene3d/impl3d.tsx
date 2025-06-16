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
} from "./components";
import { unpackID } from "./picking";
import { LIGHTING } from "./shaders";
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
} from "./types";

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
  onReady: () => void;
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
}: SceneInnerProps) {
  const $state = useContext($StateContext);

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
        renderObjects: [],
        pipelineCache: new Map(),
        dynamicBuffers: null,
        resources: {
          PointCloud: null,
          Ellipsoid: null,
          EllipsoidAxes: null,
          Cuboid: null,
          LineBeams: null,
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

  type ComponentType = ComponentConfig["type"];

  interface TypeInfo {
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
  ): Map<ComponentType, TypeInfo> {
    const typeArrays = new Map<ComponentType, TypeInfo>();

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

      let typeInfo = typeArrays.get(comp.type);
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
        };
        typeArrays.set(comp.type, typeInfo);
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

    // Clear out unused cache entries
    Object.keys(renderObjectCache.current).forEach((type) => {
      if (!components.some((c) => c.type === type)) {
        delete renderObjectCache.current[type];
      }
    });

    // Track global start index for all components
    let globalStartIndex = 0;

    // Collect render data using helper
    const typeArrays = collectTypeData(components);

    // Calculate total buffer sizes needed
    let totalRenderSize = 0;
    let totalPickingSize = 0;
    typeArrays.forEach((info: TypeInfo, type: ComponentType) => {
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
    typeArrays.forEach((info: TypeInfo, type: ComponentType) => {
      const spec = primitiveRegistry[type];
      if (!spec) return;

      try {
        // Ensure 4-byte alignment for all offsets
        const renderOffset = align16(dynamicBuffers.renderOffset);
        const pickingOffset = align16(dynamicBuffers.pickingOffset);

        // Try to get existing render object
        let renderObject = renderObjectCache.current[type];
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

        // Get picking pipeline
        const pickingPipeline = spec.getPickingPipeline(
          device,
          bindGroupLayout,
          pipelineCache,
        );
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
            ...alphaProperties(hasAlphaComponents, info.totalElementCount),
          };
          renderObjectCache.current[type] = renderObject;
        } else {
          // Update existing render object with new buffer info and state
          renderObject.instanceBuffer = bufferInfo;
          renderObject.pickingInstanceBuffer = pickingBufferInfo;
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
        console.error(`Error creating render object for type ${type}:`, error);
      }
    });

    return validRenderObjects;
  }

  /******************************************************
   * C) Render pass (single call, no loop)
   ******************************************************/

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
      } = gpuRef.current;

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
      });

      const uniformData = computeUniformData(
        containerWidth,
        containerHeight,
        camState,
      );
      device.queue.writeBuffer(uniformBuffer, 0, uniformData);

      try {
        await renderPass({
          device,
          context,
          depthTexture,
          renderObjects,
          uniformBindGroup,
        });
        onRenderComplete();
      } catch (err) {
        console.error("[Debug] Error during renderPass:", err.message);
        onRenderComplete();
      }

      onFrameRendered?.(performance.now());
      onReady();
    },
    [containerWidth, containerHeight, onFrameRendered, components],
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

      for (const ro of renderObjects) {
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
      throttledPickAtScreenXY(x, y, "hover");
    },
    [throttledPickAtScreenXY],
  );

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
    <div
      style={{ width: "100%", border: "1px solid #ccc", position: "relative" }}
    >
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
