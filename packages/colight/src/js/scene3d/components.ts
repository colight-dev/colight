// components.ts

import {
  BaseComponentConfig,
  Decoration,
  PipelineCacheEntry,
  PrimitiveSpec,
  PipelineConfig,
  GeometryResource,
  GeometryData,
  ElementConstants,
  PickEvent,
} from "./types";

export type { PickEvent, PrimitiveSpec };

import { acopy } from "./utils";

/** ===================== DECORATIONS + COMMON UTILS ===================== **/

/** Helper function to apply decorations to an array of instances */
function applyDecorations(
  decorations: Decoration[] | undefined,
  setter: (i: number, dec: Decoration) => void,
  baseOffset: number,
  sortedPositions?: Uint32Array,
  instancesPerElement: number = 1,
) {
  if (!decorations) return;

  const isSorted = !!sortedPositions;

  for (const dec of decorations) {
    if (!dec.indexes) continue;

    for (let i of dec.indexes) {
      if (i < 0) continue;
      const sortedIndex = isSorted
        ? sortedPositions[baseOffset + i] - baseOffset
        : i;

      // Apply decoration to all instances of this element
      for (let instOffset = 0; instOffset < instancesPerElement; instOffset++) {
        const instanceIdx =
          (sortedIndex + baseOffset) * instancesPerElement + instOffset;
        setter(instanceIdx, dec);
      }
    }
  }
}

/** ===================== MINI-FRAMEWORK FOR RENDER/PICK DATA ===================== **/
/** Helper function to fill color from constants or element array */
export function fillColor(
  spec: PrimitiveSpec<any>,
  constants: ElementConstants,
  elem: BaseComponentConfig,
  elemIndex: number,
  out: Float32Array,
  outOffset: number,
) {
  if (constants.color) {
    acopy(constants.color, 0, out, outOffset + spec.colorOffset, 3);
  } else {
    acopy(elem.colors!, elemIndex * 3, out, outOffset + spec.colorOffset, 3);
  }
}

/** Helper function to fill alpha from constants or element array */
export function fillAlpha(
  spec: PrimitiveSpec<any>,
  constants: ElementConstants,
  elem: BaseComponentConfig,
  elemIndex: number,
  out: Float32Array,
  outOffset: number,
) {
  out[outOffset + spec.alphaOffset] =
    constants.alpha || elem.alphas![elemIndex];
}

export function applyDecoration(
  spec: PrimitiveSpec<any>,
  dec: Decoration,
  out: Float32Array,
  outOffset: number,
) {
  if (dec.color) {
    out[outOffset + spec.colorOffset + 0] = dec.color[0];
    out[outOffset + spec.colorOffset + 1] = dec.color[1];
    out[outOffset + spec.colorOffset + 2] = dec.color[2];
  }
  if (dec.alpha !== undefined) {
    out[outOffset + spec.alphaOffset] = dec.alpha;
  }
  if (dec.scale !== undefined) {
    spec.applyDecorationScale(out, outOffset, dec.scale);
  }
}

/**
 * Builds render data for any shape using the shape's fillRenderGeometry callback
 * plus the standard columnar/default color and alpha usage, sorted index handling,
 * and decoration loop.
 */
export function buildRenderData<ConfigType extends BaseComponentConfig>(
  elem: ConfigType,
  spec: PrimitiveSpec<ConfigType>,
  out: Float32Array,
  baseOffset: number,
  sortedPositions?: Uint32Array,
  hoveredIndex?: number,
): boolean {
  // Get element count and instance count
  const elementCount = spec.getElementCount(elem);
  const instancesPerElement = spec.instancesPerElement;

  if (elementCount === 0) return false;

  const constants = getElementConstants(spec, elem);

  for (let elemIndex = 0; elemIndex < elementCount; elemIndex++) {
    const sortedIndex = sortedPositions
      ? sortedPositions[baseOffset + elemIndex] - baseOffset
      : elemIndex;
    // For each instance of this element
    for (let instOffset = 0; instOffset < instancesPerElement; instOffset++) {
      const outIndex =
        (sortedIndex + baseOffset) * instancesPerElement + instOffset;
      spec.fillRenderGeometry(constants, elem, elemIndex, out, outIndex);
    }
  }

  applyDecorations(
    elem.decorations,
    (outIndex, dec) => {
      if (spec.applyDecoration) {
        // Use component-specific decoration handling
        spec.applyDecoration(dec, out, outIndex, spec.floatsPerInstance);
      } else {
        applyDecoration(spec, dec, out, outIndex * spec.floatsPerInstance);
      }
    },
    baseOffset,
    sortedPositions,
    instancesPerElement,
  );

  // Apply hoverProps to the hovered instance (after decorations, so it can override)
  if (hoveredIndex !== undefined && elem.hoverProps) {
    const hoverProps = elem.hoverProps;
    // Find the sorted index for the hovered element
    const sortedIndex = sortedPositions
      ? sortedPositions[baseOffset + hoveredIndex] - baseOffset
      : hoveredIndex;

    // Apply hoverProps to all instances of this element
    for (let instOffset = 0; instOffset < instancesPerElement; instOffset++) {
      const outIndex =
        (sortedIndex + baseOffset) * instancesPerElement + instOffset;
      const outOffset = outIndex * spec.floatsPerInstance;

      // Apply hoverProps color
      if (hoverProps.color) {
        out[outOffset + spec.colorOffset + 0] = hoverProps.color[0];
        out[outOffset + spec.colorOffset + 1] = hoverProps.color[1];
        out[outOffset + spec.colorOffset + 2] = hoverProps.color[2];
      }

      // Apply hoverProps alpha
      if (hoverProps.alpha !== undefined) {
        out[outOffset + spec.alphaOffset] = hoverProps.alpha;
      }

      // Apply hoverProps scale
      if (hoverProps.scale !== undefined) {
        spec.applyDecorationScale(out, outOffset, hoverProps.scale);
      }
    }
  }

  return true;
}

/**
 * Builds picking data for any shape using the shape's fillPickingGeometry callback,
 * plus handling sorted indices, decorations that affect scale, and base pick ID.
 */
export function buildPickingData<ConfigType extends BaseComponentConfig>(
  elem: ConfigType,
  spec: PrimitiveSpec<ConfigType>,
  out: Float32Array,
  pickingBase: number,
  baseOffset: number,
  sortedPositions?: Uint32Array,
): void {
  // Get element count and instance count
  const elementCount = spec.getElementCount(elem);
  const instancesPerElement = spec.instancesPerElement;

  if (elementCount === 0) return;

  const constants = getElementConstants(spec, elem);

  for (let i = 0; i < elementCount; i++) {
    const sortedIndex = sortedPositions
      ? sortedPositions[baseOffset + i] - baseOffset
      : i;
    // For each instance of this element
    for (let instOffset = 0; instOffset < instancesPerElement; instOffset++) {
      const outIndex =
        (sortedIndex + baseOffset) * instancesPerElement + instOffset;
      spec.fillPickingGeometry(constants, elem, i, out, outIndex, pickingBase);
    }
  }

  // Apply decorations that affect scale
  applyDecorations(
    elem.decorations,
    (outIndex, dec) => {
      if (dec.scale !== undefined && dec.scale !== 1.0) {
        if (spec.applyDecorationScale) {
          spec.applyDecorationScale(
            out,
            outIndex * spec.floatsPerPicking,
            dec.scale,
          );
        }
      }
    },
    baseOffset,
    sortedPositions,
    instancesPerElement,
  );

  // Apply pickingScale to all instances if set
  if (elem.pickingScale && elem.pickingScale !== 1.0) {
    const totalInstances = elementCount * instancesPerElement;
    for (let i = 0; i < totalInstances; i++) {
      spec.applyDecorationScale(
        out,
        i * spec.floatsPerPicking,
        elem.pickingScale,
      );
    }
  }
}

/** ===================== GPU PIPELINE HELPERS (unchanged) ===================== **/

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

const computeConstants = (spec: any, elem: any) => {
  const constants: ElementConstants = {};

  for (const [key, defaultValue] of Object.entries({
    alpha: 1.0,
    color: [0.5, 0.5, 0.5],
    ...spec.defaults,
  })) {
    const pluralKey = key + "s";
    const pluralValue = elem[pluralKey];
    const singularValue = elem[key];

    const targetTypeIsArray = Array.isArray(defaultValue);

    // Case 1: No plural form exists. Use element value or default.
    if (!pluralValue) {
      if (targetTypeIsArray && typeof singularValue === "number") {
        // Fill array with the single number value
        // @ts-ignore
        constants[key as keyof ElementConstants] = new Array(
          defaultValue.length,
        ).fill(singularValue);
      } else {
        constants[key as keyof ElementConstants] =
          singularValue || defaultValue;
      }
      continue;
    }
    // Case 2: Target value is an array, and the specified plural is of that length, so use it as a constant value.
    if (targetTypeIsArray && pluralValue.length === defaultValue.length) {
      constants[key as keyof ElementConstants] = pluralValue || defaultValue;
      continue;
    }

    // Case 3: Target value is an array, and the specified plural is of length 1, repeat it.
    if (targetTypeIsArray && pluralValue.length === 1) {
      // Fill array with the single value
      const filledArray = new Array((defaultValue as number[]).length).fill(
        pluralValue[0],
      );
      // @ts-ignore
      constants[key as keyof ElementConstants] = filledArray;
    }
  }

  return constants;
};

const constantsCache = new WeakMap<BaseComponentConfig, ElementConstants>();

const getElementConstants = (
  spec: PrimitiveSpec<BaseComponentConfig>,
  elem: BaseComponentConfig,
): ElementConstants => {
  let constants = constantsCache.get(elem);
  if (constants) return constants;
  constants = computeConstants(spec, elem);
  constantsCache.set(elem, constants);
  return constants;
};

/** ===================== PRIMITIVES ===================== **/

// All primitives are now defined using the declarative primitive system.
// Import from the new primitives module.
export {
  pointCloudSpec,
  type PointCloudComponentConfig,
  type PointCloudProps,
  ellipsoidSpec,
  type EllipsoidComponentConfig,
  type EllipsoidProps,
  ellipsoidAxesSpec,
  type EllipsoidAxesComponentConfig,
  cuboidSpec,
  type CuboidComponentConfig,
  type CuboidProps,
  lineBeamsSpec,
  type LineBeamsComponentConfig,
  lineSegmentsSpec,
  type LineSegmentsComponentConfig,
  boundingBoxSpec,
  type BoundingBoxComponentConfig,
  type BoundingBoxProps,
  imagePlaneSpec,
  type ImagePlaneComponentConfig,
  type ImagePlaneProps,
  type ImageSource,
  getImageBindGroupLayout,
  defineMesh,
  type MeshComponentConfig,
} from "./primitives";

// Re-export utility functions from primitives
export { getLineBeamsSegmentPointIndex } from "./primitives/lineBeams";

/** ===================== UNION TYPE FOR ALL COMPONENT CONFIGS ===================== **/

export type ComponentConfig =
  | PointCloudComponentConfig
  | EllipsoidComponentConfig
  | EllipsoidAxesComponentConfig
  | CuboidComponentConfig
  | LineBeamsComponentConfig
  | LineSegmentsComponentConfig
  | BoundingBoxComponentConfig
  | ImagePlaneComponentConfig;
