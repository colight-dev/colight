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

/**
 * Write new geometry contents into an existing GeometryResource (D1b).
 *
 * The grow-only reuse pattern already used for the instance and transform
 * buffers, applied to geometry: reuse `vb`/`ib` when the new bytes fit, and
 * when they do not, allocate a larger buffer and DESTROY the one it replaces.
 * Nothing here reallocates on a same-size write, so a deformation of stable
 * topology settles at zero net allocation.
 *
 * Note on layout: vertices are interleaved (`interleaveVertexData`), so a
 * positions-only change still rewrites the whole interleaved vertex buffer.
 * That is one `writeBuffer` of already-resident bytes and no pipeline churn -
 * acceptable. De-interleaving positions would only be worth it if measurement
 * ever showed the upload dominating; it does not today.
 *
 * The resource is mutated in place so every RenderObject already holding it
 * keeps pointing at the right buffers when they were reused. Callers must
 * refresh RenderObjects that cached `vb`/`ib`/counts if `grew` is true.
 *
 * @returns Whether a buffer had to be reallocated (and the old one destroyed).
 */
export const updateBuffers = (
  device: GPUDevice,
  resource: GeometryResource,
  { vertexData, indexData }: GeometryData,
  vertexStrideFloats = 6,
): { grew: boolean } => {
  let grew = false;

  const vertexSize = align4(vertexData.byteLength);
  if (resource.vb.size < vertexSize) {
    resource.vb.destroy();
    resource.vb = device.createBuffer({
      size: vertexSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    grew = true;
  }
  writeBufferPadded(device, resource.vb, vertexData, vertexSize);
  resource.vertexCount = vertexData.length / vertexStrideFloats;

  if (indexData && indexData.length > 0) {
    const indexSize = align4(indexData.byteLength);
    if (!resource.ib || resource.ib.size < indexSize) {
      if (resource.ib) resource.ib.destroy();
      resource.ib = device.createBuffer({
        size: indexSize,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      grew = true;
    }
    writeBufferPadded(device, resource.ib, indexData, indexSize);
    resource.indexCount = indexData.length;
    resource.indexFormat =
      indexData instanceof Uint32Array ? "uint32" : "uint16";
  } else if (resource.ib) {
    // Geometry lost its index buffer: release it rather than orphaning it.
    resource.ib.destroy();
    resource.ib = null;
    resource.indexCount = 0;
    resource.indexFormat = undefined;
  }

  return { grew };
};

/** Release both GPU buffers held by a geometry resource. Idempotent per buffer. */
export const destroyGeometryResource = (resource: GeometryResource) => {
  resource.vb.destroy();
  if (resource.ib) resource.ib.destroy();
};
