// components.ts

import {
  billboardShaderProgram,
  rigidLitShaderProgram,
  lineBeamShaderProgram,
  GeneratedPrimitiveShaderProgram,
} from "./shaders";

import {
  createCubeGeometry,
  createBeamGeometry,
  createSphereGeometry,
} from "./geometry";

import { packID } from "./picking";

import {
  BaseComponentConfig,
  Decoration,
  PipelineCacheEntry,
  PrimitiveSpec,
  PipelineConfig,
  GeometryResource,
  GeometryData,
  ElementConstants,
  Scene3DGeometryOptions,
} from "./types";

import { acopy } from "../utils";

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

  return true;
}

/**
 * Builds per-instance pick IDs in the same instance order as the render buffer.
 */
export function buildPickingData<ConfigType extends BaseComponentConfig>(
  elem: ConfigType,
  spec: PrimitiveSpec<ConfigType>,
  out: Float32Array,
  pickingBase: number,
  baseOffset: number,
  sortedPositions?: Uint32Array,
): void {
  const elementCount = spec.getElementCount(elem);
  const instancesPerElement = spec.instancesPerElement;

  if (elementCount === 0) return;

  for (let i = 0; i < elementCount; i++) {
    const sortedIndex = sortedPositions
      ? sortedPositions[baseOffset + i] - baseOffset
      : i;
    const pickID = packID(pickingBase + i);
    for (let instOffset = 0; instOffset < instancesPerElement; instOffset++) {
      const outIndex =
        (sortedIndex + baseOffset) * instancesPerElement + instOffset;
      out[outIndex * spec.floatsPerPicking] = pickID;
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
  bindGroupLayout: GPUBindGroupLayout,
  config: PipelineConfig,
  format: GPUTextureFormat,
): GPURenderPipeline {
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
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
        ...(config.pickFormat
          ? [
              {
                format: config.pickFormat,
                writeMask: GPUColorWrite.ALL,
              },
            ]
          : []),
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
  bindGroupLayout: GPUBindGroupLayout,
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

export const createBuffers = (
  device: GPUDevice,
  { vertexData, indexData }: GeometryData,
): GeometryResource => {
  const indexFormat: GPUIndexFormat =
    indexData instanceof Uint32Array ? "uint32" : "uint16";
  const vb = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(vb, 0, vertexData);

  const ib = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ib, 0, indexData);

  // Each vertex has 6 floats (position + normal)
  const vertexCount = vertexData.length / 6;

  return {
    vb,
    ib,
    indexFormat,
    indexCount: indexData.length,
    vertexCount,
  };
};

type PrimitiveDefinition<ConfigType extends BaseComponentConfig> = ThisType<
  PrimitiveSpec<ConfigType>
> & {
  type: string;
  defaults?: ElementConstants;
  instancesPerElement?: number;
  shaderProgram: GeneratedPrimitiveShaderProgram;
  pipelineKey: string;
  renderConfig: PrimitiveSpec<ConfigType>["renderConfig"];
  getElementCount: PrimitiveSpec<ConfigType>["getElementCount"];
  getCenters: PrimitiveSpec<ConfigType>["getCenters"];
  fillRenderGeometry: PrimitiveSpec<ConfigType>["fillRenderGeometry"];
  applyDecorationScale: PrimitiveSpec<ConfigType>["applyDecorationScale"];
  getColorIndexForInstance?: PrimitiveSpec<ConfigType>["getColorIndexForInstance"];
  applyDecoration?: PrimitiveSpec<ConfigType>["applyDecoration"];
  fillColor?: PrimitiveSpec<ConfigType>["fillColor"];
  fillAlpha?: PrimitiveSpec<ConfigType>["fillAlpha"];
  createGeometryResource: PrimitiveSpec<ConfigType>["createGeometryResource"];
};

function definePrimitive<ConfigType extends BaseComponentConfig>(
  definition: PrimitiveDefinition<ConfigType>,
): PrimitiveSpec<ConfigType> {
  return {
    type: definition.type,
    defaults: definition.defaults,
    instancesPerElement: definition.instancesPerElement ?? 1,
    floatsPerInstance: definition.shaderProgram.renderFloatsPerInstance,
    floatsPerPicking: definition.shaderProgram.pickIDFloatsPerInstance,
    getElementCount: definition.getElementCount,
    getCenters: definition.getCenters,
    colorOffset: definition.shaderProgram.colorOffset,
    alphaOffset: definition.shaderProgram.alphaOffset,
    fillRenderGeometry: definition.fillRenderGeometry,
    applyDecorationScale: definition.applyDecorationScale,
    getColorIndexForInstance: definition.getColorIndexForInstance,
    applyDecoration: definition.applyDecoration,
    fillColor: definition.fillColor,
    fillAlpha: definition.fillAlpha,
    renderConfig: definition.renderConfig,

    getRenderPipeline(device, bindGroupLayout, cache) {
      const format = navigator.gpu.getPreferredCanvasFormat();
      return getOrCreatePipeline(
        device,
        `${definition.pipelineKey}:render`,
        () =>
          createTranslucentGeometryPipeline(
            device,
            bindGroupLayout,
            {
              vertexShader: definition.shaderProgram.renderVertex,
              fragmentShader: definition.shaderProgram.fragment,
              vertexEntryPoint: "vs_main",
              fragmentEntryPoint: "fs_main",
              bufferLayouts: [
                definition.shaderProgram.geometryLayout,
                definition.shaderProgram.renderLayout,
                definition.shaderProgram.pickIDLayout,
              ],
              pickFormat: "rgba8unorm",
            },
            format,
            this,
          ),
        cache,
      );
    },

    createGeometryResource: definition.createGeometryResource,
  };
}

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

/** ===================== POINT CLOUD ===================== **/

export interface PointCloudComponentConfig extends BaseComponentConfig {
  type: "PointCloud";
  centers: Float32Array;
  sizes?: Float32Array; // Per-point sizes
  size?: number; // Default size, defaults to 0.02
}

export const pointCloudSpec = definePrimitive<PointCloudComponentConfig>({
  type: "PointCloud",
  shaderProgram: billboardShaderProgram,
  pipelineKey: "PointCloud",
  defaults: {
    size: 0.02,
  },

  getElementCount(elem) {
    return elem.centers.length / 3;
  },

  getCenters(elem) {
    return elem.centers;
  },

  // fillRenderGeometry: shape-specific code, ignoring color/alpha
  fillRenderGeometry(constants, elem, i, out, outIndex) {
    const outOffset = outIndex * this.floatsPerInstance;

    // Position
    acopy(elem.centers, i * 3, out, outOffset, 3);
    // Size - use constant or per-instance value
    out[outOffset + 3] = constants.size || elem.sizes![i];

    fillColor(this, constants, elem, i, out, outOffset);
    fillAlpha(this, constants, elem, i, out, outOffset);
  },

  applyDecorationScale(out, offset, scaleFactor) {
    out[offset + 3] *= scaleFactor;
  },
  renderConfig: {
    cullMode: "none",
    topology: "triangle-list",
  },

  createGeometryResource(device, _geometryOptions: Scene3DGeometryOptions) {
    return createBuffers(device, {
      vertexData: new Float32Array([
        // Position (x,y,z) and Normal (nx,ny,nz) for each vertex
        -0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
        1.0, // Bottom-left
        0.5,
        -0.5,
        0.0,
        0.0,
        0.0,
        1.0, // Bottom-right
        -0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0, // Top-left
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        1.0, // Top-right
      ]),
      indexData: new Uint16Array([0, 1, 2, 2, 1, 3]),
    });
  },
});

/** ===================== ELLIPSOID ===================== **/

export interface EllipsoidComponentConfig extends BaseComponentConfig {
  type: "Ellipsoid" | "EllipsoidAxes";
  centers: Float32Array | number[];
  half_sizes?: Float32Array | number[];
  half_size?: [number, number, number] | number;
  quaternions?: Float32Array | number[];
  quaternion?: [number, number, number, number];
  fill_mode?: "Solid" | "MajorWireframe";
}

export const ellipsoidSpec = definePrimitive<EllipsoidComponentConfig>({
  type: "Ellipsoid",
  shaderProgram: rigidLitShaderProgram,
  pipelineKey: "Ellipsoid",
  defaults: {
    half_size: [0.5, 0.5, 0.5],
    quaternion: [0, 0, 0, 1],
  },

  getElementCount(elem) {
    return elem.centers.length / 3;
  },

  getCenters(elem) {
    return elem.centers;
  },

  // Fill render geometry for a single element
  fillRenderGeometry(constants, elem, elemIndex, out, outIndex) {
    const outOffset = outIndex * this.floatsPerInstance;

    // Position - same for all 3 rings
    acopy(elem.centers, elemIndex * 3, out, outOffset, 3);

    // Half sizes - same for all 3 rings
    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, outOffset + 3, 3);
    } else {
      acopy(elem.half_sizes!, elemIndex * 3, out, outOffset + 3, 3);
    }

    // Quaternion - same for all 3 rings
    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, outOffset + 6, 4);
    } else {
      acopy(elem.quaternions!, elemIndex * 4, out, outOffset + 6, 4);
    }

    // The shader will handle the ring orientation based on instance_index
    // See ringVertCode in shaders.ts which uses the instance_index to determine
    // which axis (X, Y, or Z) the ring should be oriented along

    // Color and alpha - same for all 3 rings
    fillAlpha(this, constants, elem, elemIndex, out, outOffset);
    fillColor(this, constants, elem, elemIndex, out, outOffset);
  },

  applyDecorationScale(out, offset, scaleFactor) {
    // Multiply the sizes
    out[offset + 3] *= scaleFactor;
    out[offset + 4] *= scaleFactor;
    out[offset + 5] *= scaleFactor;
  },
  renderConfig: {
    cullMode: "back",
    topology: "triangle-list",
  },

  createGeometryResource(device, geometryOptions: Scene3DGeometryOptions) {
    return createBuffers(
      device,
      createSphereGeometry(
        geometryOptions.ellipsoidStacks,
        geometryOptions.ellipsoidSlices,
      ),
    );
  },
});

/** ===================== CUBOID ===================== **/

export interface CuboidComponentConfig extends BaseComponentConfig {
  type: "Cuboid";
  centers: Float32Array;
  half_sizes?: Float32Array;
  half_size?: number | [number, number, number];
  quaternions?: Float32Array;
  quaternion?: [number, number, number, number];
}

export const cuboidSpec = definePrimitive<CuboidComponentConfig>({
  type: "Cuboid",
  shaderProgram: rigidLitShaderProgram,
  pipelineKey: "Cuboid",
  defaults: {
    half_size: [0.1, 0.1, 0.1],
    quaternion: [0, 0, 0, 1],
  },

  getElementCount(elem) {
    return elem.centers.length / 3;
  },

  getCenters(elem) {
    return elem.centers;
  },

  fillRenderGeometry(constants, elem, i, out, outIndex) {
    const outOffset = outIndex * this.floatsPerInstance;

    // Position
    acopy(elem.centers, i * 3, out, outOffset, 3);

    // Half sizes
    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, outOffset + 3, 3);
    } else {
      acopy(elem.half_sizes as ArrayLike<number>, i * 3, out, outOffset + 3, 3);
    }

    // Quaternion
    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, outOffset + 6, 4);
    } else {
      acopy(elem.quaternions!, i * 4, out, outOffset + 6, 4);
    }

    fillAlpha(this, constants, elem, i, out, outOffset);
    fillColor(this, constants, elem, i, out, outOffset);
  },

  applyDecorationScale(out, offset, scaleFactor) {
    // multiply half_sizes
    out[offset + 3] *= scaleFactor;
    out[offset + 4] *= scaleFactor;
    out[offset + 5] *= scaleFactor;
  },
  renderConfig: {
    cullMode: "none",
    topology: "triangle-list",
  },

  createGeometryResource(device, _geometryOptions: Scene3DGeometryOptions) {
    return createBuffers(device, createCubeGeometry());
  },
});

/** ===================== LINE BEAMS ===================== **/

export interface LineBeamsComponentConfig extends BaseComponentConfig {
  type: "LineBeams";
  points: Float32Array; // [x,y,z,lineIndex, x,y,z,lineIndex, ...]
  sizes?: Float32Array; // Per-line sizes
  size?: number; // Default size
}

/** We store a small WeakMap to "cache" the segment map for each config. */
const lineBeamsSegmentMap = new WeakMap<
  LineBeamsComponentConfig,
  {
    segmentMap: number[];
  }
>();

function prepareLineSegments(elem: LineBeamsComponentConfig): number[] {
  // If we already did it, return cached
  const cached = lineBeamsSegmentMap.get(elem);
  if (cached) return cached.segmentMap;

  const pointCount = elem.points.length / 4;
  const segmentIndices: number[] = [];

  for (let p = 0; p < pointCount - 1; p++) {
    const iCurr = elem.points[p * 4 + 3];
    const iNext = elem.points[(p + 1) * 4 + 3];
    if (iCurr === iNext) {
      segmentIndices.push(p);
    }
  }
  lineBeamsSegmentMap.set(elem, { segmentMap: segmentIndices });
  return segmentIndices;
}

function countSegments(elem: LineBeamsComponentConfig): number {
  return prepareLineSegments(elem).length;
}

export const lineBeamsSpec = definePrimitive<LineBeamsComponentConfig>({
  type: "LineBeams",
  shaderProgram: lineBeamShaderProgram,
  pipelineKey: "LineBeams",
  defaults: {
    size: 0.02,
  },

  getElementCount(elem) {
    return countSegments(elem);
  },

  getCenters(elem) {
    // Build array of each segment's midpoint, for sorting or bounding
    const segMap = prepareLineSegments(elem);
    const segCount = segMap.length;
    const centers = new Float32Array(segCount * 3);
    for (let s = 0; s < segCount; s++) {
      const p = segMap[s];
      const x0 = elem.points[p * 4 + 0];
      const y0 = elem.points[p * 4 + 1];
      const z0 = elem.points[p * 4 + 2];
      const x1 = elem.points[(p + 1) * 4 + 0];
      const y1 = elem.points[(p + 1) * 4 + 1];
      const z1 = elem.points[(p + 1) * 4 + 2];
      centers[s * 3 + 0] = (x0 + x1) * 0.5;
      centers[s * 3 + 1] = (y0 + y1) * 0.5;
      centers[s * 3 + 2] = (z0 + z1) * 0.5;
    }
    return centers;
  },

  /**
   * We want color/alpha to come from the line index (points[..+3]),
   * not from the segment index. So we define getColorIndexForInstance:
   */
  getColorIndexForInstance(elem, segmentIndex) {
    const segMap = prepareLineSegments(elem);
    const p = segMap[segmentIndex];
    // The line index is floor(points[p*4+3])
    return Math.floor(elem.points[p * 4 + 3]);
  },

  fillRenderGeometry(constants, elem, segmentIndex, out, outIndex) {
    const outOffset = outIndex * this.floatsPerInstance;
    const segMap = prepareLineSegments(elem);
    const p = segMap[segmentIndex];

    // Start
    acopy(elem.points, p * 4, out, outOffset, 3);

    // End
    acopy(elem.points, (p + 1) * 4, out, outOffset + 3, 3);

    // Size
    const lineIndex = Math.floor(elem.points[p * 4 + 3]);
    out[outOffset + 6] = constants.size || elem.sizes![lineIndex];

    fillAlpha(this, constants, elem, segmentIndex, out, outOffset);
    fillColor(this, constants, elem, segmentIndex, out, outOffset);
  },

  applyDecorationScale(out, offset, scaleFactor) {
    // only the size is at offset+6
    out[offset + 6] *= scaleFactor;
  },
  renderConfig: {
    cullMode: "none",
    topology: "triangle-list",
  },

  createGeometryResource(device, _geometryOptions: Scene3DGeometryOptions) {
    return createBuffers(device, createBeamGeometry());
  },
});

/** ===================== UNION TYPE FOR ALL COMPONENT CONFIGS ===================== **/

export type ComponentConfig =
  | PointCloudComponentConfig
  | EllipsoidComponentConfig
  | CuboidComponentConfig
  | LineBeamsComponentConfig;
