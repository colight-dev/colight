// pipelineUtils.ts
// GPU pipeline creation utilities - extracted to break circular dependency with primitives

import {
  PipelineCacheEntry,
  PipelineConfig,
  GeometryResource,
  GeometryData,
  PrimitiveSpec,
} from "./types";

/** ===================== GPU PIPELINE HELPERS ===================== **/

export function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  createFn: () => GPURenderPipeline,
  cache: Map<string, PipelineCacheEntry>, // This will be the instance cache
): GPURenderPipeline {
  const entry = cache.get(key);
  if (entry && entry.device === device) {
    return entry.pipeline;
  }

  // Create new pipeline and cache it with device reference
  const pipeline = createFn();
  cache.set(key, { pipeline, device });
  return pipeline;
}

export function createRenderPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout | GPUBindGroupLayout[],
  config: PipelineConfig,
  format: GPUTextureFormat,
): GPURenderPipeline {
  const bindGroupLayouts = Array.isArray(bindGroupLayout)
    ? bindGroupLayout
    : [bindGroupLayout];
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts,
  });

  // Include all values from config.primitive, including stripIndexFormat, if provided.
  const primitiveConfig = {
    topology: config.primitive?.topology || "triangle-list",
    cullMode: config.primitive?.cullMode || "back",
    stripIndexFormat: config.primitive?.stripIndexFormat,
  };

  return device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: device.createShaderModule({ code: config.vertexShader }),
      entryPoint: config.vertexEntryPoint,
      buffers: config.bufferLayouts,
    },
    fragment: {
      module: device.createShaderModule({ code: config.fragmentShader }),
      entryPoint: config.fragmentEntryPoint,
      targets: [
        {
          format,
          writeMask: config.colorWriteMask ?? GPUColorWrite.ALL,
          ...(config.blend && {
            blend: {
              color: config.blend.color || {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
              },
              alpha: config.blend.alpha || {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
              },
            },
          }),
        },
      ],
    },
    primitive: primitiveConfig,
    depthStencil: config.depthStencil || {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });
}

export function createTranslucentGeometryPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout | GPUBindGroupLayout[],
  config: PipelineConfig,
  format: GPUTextureFormat,
  primitiveSpec: PrimitiveSpec<any>, // Take the primitive spec instead of just type
): GPURenderPipeline {
  return createRenderPipeline(
    device,
    bindGroupLayout,
    {
      ...config,
      primitive: primitiveSpec.renderConfig,
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
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    },
    format,
  );
}

/**
 * Creates an overlay pipeline - renders in front of scene geometry.
 * Used for gizmos, helpers, and always-visible UI elements.
 */
export function createOverlayPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout | GPUBindGroupLayout[],
  config: PipelineConfig,
  format: GPUTextureFormat,
  primitiveSpec: PrimitiveSpec<any>,
): GPURenderPipeline {
  return createRenderPipeline(
    device,
    bindGroupLayout,
    {
      ...config,
      primitive: primitiveSpec.renderConfig,
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
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: false, // Don't write to depth buffer
        depthCompare: "always", // Always pass depth test (render in front)
      },
    },
    format,
  );
}

/**
 * Creates an overlay picking pipeline - picks in front of scene geometry.
 * Used for picking overlay elements with priority over scene elements.
 */
export function createOverlayPickingPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout | GPUBindGroupLayout[],
  config: PipelineConfig,
  primitiveSpec: PrimitiveSpec<any>,
): GPURenderPipeline {
  return createRenderPipeline(
    device,
    bindGroupLayout,
    {
      ...config,
      primitive: primitiveSpec.renderConfig,
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: false, // Don't write to depth buffer
        depthCompare: "always", // Always pass depth test (pick in front)
      },
    },
    "rgba8unorm",
  );
}

function align4(size: number): number {
  return Math.ceil(size / 4) * 4;
}

function writeBufferPadded(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: ArrayBufferView,
  size: number,
) {
  if (data.byteLength === size) {
    device.queue.writeBuffer(buffer, 0, data);
    return;
  }

  const padded = new Uint8Array(size);
  padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  device.queue.writeBuffer(buffer, 0, padded);
}

export const createBuffers = (
  device: GPUDevice,
  { vertexData, indexData }: GeometryData,
  vertexStrideFloats = 6,
): GeometryResource => {
  const vertexSize = align4(vertexData.byteLength);
  const vb = device.createBuffer({
    size: vertexSize,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  writeBufferPadded(device, vb, vertexData, vertexSize);

  let ib: GPUBuffer | null = null;
  let indexCount = 0;
  let indexFormat: GPUIndexFormat | undefined;

  if (indexData && indexData.length > 0) {
    const indexSize = align4(indexData.byteLength);
    ib = device.createBuffer({
      size: indexSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    writeBufferPadded(device, ib, indexData, indexSize);
    indexCount = indexData.length;
    indexFormat = indexData instanceof Uint32Array ? "uint32" : "uint16";
  }

  const vertexCount = vertexData.length / vertexStrideFloats;

  return {
    vb,
    ib,
    indexCount,
    vertexCount,
    indexFormat,
  };
};
