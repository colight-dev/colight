/// <reference types="react" />

import * as glMatrix from 'gl-matrix';
import React, {
  // DO NOT require MouseEvent
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react';
import { throttle } from '../utils';
import { createCubeGeometry, createBeamGeometry, createSphereGeometry, createTorusGeometry } from './geometry';

import {
  CameraParams,
  CameraState,
  createCameraParams,
  createCameraState,
  orbit,
  pan,
  zoom
} from './camera3d';

import {
  LIGHTING,
  billboardVertCode,
  billboardFragCode,
  ellipsoidVertCode,
  ellipsoidFragCode,
  ringVertCode,
  ringFragCode,
  cuboidVertCode,
  cuboidFragCode,
  lineBeamVertCode,
  lineBeamFragCode,
  pickingVertCode
} from './shaders';


/******************************************************
 * 1) Types and Interfaces
 ******************************************************/

interface BaseComponentConfig {
  /**
   * Per-instance RGB color values as a Float32Array of RGB triplets.
   * Each instance requires 3 consecutive values in the range [0,1].
   */
  colors?: Float32Array;

  /**
   * Per-instance alpha (opacity) values.
   * Each value should be in the range [0,1].
   */
  alphas?: Float32Array;

  /**
   * Per-instance scale multipliers.
   * These multiply the base size/radius of each instance.
   */
  scales?: Float32Array;

  /**
   * Default RGB color applied to all instances without specific colors.
   * Values should be in range [0,1]. Defaults to [1,1,1] (white).
   */
  color?: [number, number, number];

  /**
   * Default alpha (opacity) for all instances without specific alpha.
   * Should be in range [0,1]. Defaults to 1.0.
   */
  alpha?: number;

  /**
   * Default scale multiplier for all instances without specific scale.
   * Defaults to 1.0.
   */
  scale?: number;

  /**
   * Callback fired when the mouse hovers over an instance.
   * The index parameter is the instance index, or null when hover ends.
   */
  onHover?: (index: number|null) => void;

  /**
   * Callback fired when an instance is clicked.
   * The index parameter is the clicked instance index.
   */
  onClick?: (index: number) => void;

  /**
   * Optional array of decorations to apply to specific instances.
   * Decorations can override colors, alpha, and scale for individual instances.
   */
  decorations?: Decoration[];
}

function getBaseDefaults(config: Partial<BaseComponentConfig>): Required<Omit<BaseComponentConfig, 'colors' | 'alphas' | 'scales' | 'decorations' | 'onHover' | 'onClick'>> {
  return {
    color: config.color ?? [1, 1, 1],
    alpha: config.alpha ?? 1.0,
    scale: config.scale ?? 1.0,
  };
}

function getColumnarParams(elem: BaseComponentConfig, count: number): {colors: Float32Array|null, alphas: Float32Array|null, scales: Float32Array|null} {

  // // Check for Float64Arrays and throw if found
  // if (elem.colors instanceof Float64Array) {
  //   throw new Error('Float64Array not supported for colors - please use Float32Array');
  // }
  // if (elem.alphas instanceof Float64Array) {
  //   throw new Error('Float64Array not supported for alphas - please use Float32Array');
  // }
  // if (elem.scales instanceof Float64Array) {
  //   throw new Error('Float64Array not supported for scales - please use Float32Array');
  // }

  const hasValidColors = elem.colors instanceof Float32Array && elem.colors.length >= count * 3;
  const hasValidAlphas = elem.alphas instanceof Float32Array && elem.alphas.length >= count;
  const hasValidScales = elem.scales instanceof Float32Array && elem.scales.length >= count;

  return {
    colors: hasValidColors ? (elem.colors as Float32Array) : null,
    alphas: hasValidAlphas ? (elem.alphas as Float32Array) : null,
    scales: hasValidScales ? (elem.scales as Float32Array) : null
  };
}

export interface BufferInfo {
  buffer: GPUBuffer;
  offset: number;
  stride: number;
}

export interface RenderObject {
  pipeline?: GPURenderPipeline;
  vertexBuffers: Partial<[GPUBuffer, BufferInfo]>;  // Allow empty or partial arrays
  indexBuffer?: GPUBuffer;
  vertexCount?: number;
  indexCount?: number;
  instanceCount?: number;

  pickingPipeline?: GPURenderPipeline;
  pickingVertexBuffers: Partial<[GPUBuffer, BufferInfo]>;  // Allow empty or partial arrays
  pickingIndexBuffer?: GPUBuffer;
  pickingVertexCount?: number;
  pickingIndexCount?: number;
  pickingInstanceCount?: number;

  componentIndex: number;
  pickingDataStale: boolean;

  // Add transparency info directly to RenderObject
  transparencyInfo?: {
    needsSort: boolean;
    centers: Float32Array;
    stride: number;
    offset: number;
    lastCameraPosition?: [number, number, number];
    sortedIndices?: number[];  // Store the most recent sorting
  };
}

export interface DynamicBuffers {
  renderBuffer: GPUBuffer;
  pickingBuffer: GPUBuffer;
  renderOffset: number;  // Current offset into render buffer
  pickingOffset: number; // Current offset into picking buffer
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
}

/******************************************************
 * 3) Data Structures & Primitive Specs
 ******************************************************/


interface PrimitiveSpec<E> {
  /**
   * Returns the number of instances in this component.
   * Used to allocate buffers and determine draw call parameters.
   */
  getCount(component: E): number;

  /**
   * Builds vertex buffer data for rendering.
   * Returns a Float32Array containing interleaved vertex attributes,
   * or null if the component has no renderable data.
   * @param component The component to build render data for
   * @param sortedIndices Optional array of indices for depth sorting
   */
  buildRenderData(component: E, sortedIndices?: number[]): Float32Array | null;

  /**
   * Builds vertex buffer data for GPU-based picking.
   * Returns a Float32Array containing picking IDs and instance data,
   * or null if the component doesn't support picking.
   * @param component The component to build picking data for
   * @param baseID Starting ID for this component's instances
   * @param sortedIndices Optional array of indices for depth sorting
   */
  buildPickingData(component: E, baseID: number, sortedIndices?: number[]): Float32Array | null;

  /**
   * Default WebGPU rendering configuration for this primitive type.
   * Specifies face culling and primitive topology.
   */
  renderConfig: RenderConfig;

  /**
   * Creates or retrieves a cached WebGPU render pipeline for this primitive.
   * @param device The WebGPU device
   * @param bindGroupLayout Layout for uniform bindings
   * @param cache Pipeline cache to prevent duplicate creation
   */
  getRenderPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>
  ): GPURenderPipeline;

  /**
   * Creates or retrieves a cached WebGPU pipeline for picking.
   * @param device The WebGPU device
   * @param bindGroupLayout Layout for uniform bindings
   * @param cache Pipeline cache to prevent duplicate creation
   */
  getPickingPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>
  ): GPURenderPipeline;

  /**
   * Creates the base geometry buffers needed for this primitive type.
   * These buffers are shared across all instances of the primitive.
   */
  createGeometryResource(device: GPUDevice): { vb: GPUBuffer; ib: GPUBuffer; indexCount: number };
}

interface Decoration {
  indexes: number[];
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
}

/** Helper function to apply decorations to an array of instances */
function applyDecorations(
  decorations: Decoration[] | undefined,
  instanceCount: number,
  setter: (i: number, dec: Decoration) => void
) {
  if (!decorations) return;
  for (const dec of decorations) {
    if (!dec.indexes) continue;
    for (const idx of dec.indexes) {
      if (idx < 0 || idx >= instanceCount) continue;
      setter(idx, dec);
    }
  }
}

/** Configuration for how a primitive type should be rendered */
interface RenderConfig {
  /** How faces should be culled */
  cullMode: GPUCullMode;
  /** How vertices should be interpreted */
  topology: GPUPrimitiveTopology;
}

/** ===================== POINT CLOUD ===================== **/


export interface PointCloudComponentConfig extends BaseComponentConfig {
  type: 'PointCloud';
  positions: Float32Array;
  sizes?: Float32Array;     // Per-point sizes
  size?: number;           // Default size, defaults to 0.02
}

/** Helper function to handle sorted indices and position mapping */
function getIndicesAndMapping(count: number, sortedIndices?: number[]): {
  useSequential: boolean,
  indices: number[] | null,  // Change to null instead of undefined
  indexToPosition: number[] | null
} {
  if (!sortedIndices) {
    return {
      useSequential: true,
      indices: null,
      indexToPosition: null
    };
  }

  // Only create mapping if we have sorted indices
  const indexToPosition = new Array(count);
  for(let j = 0; j < count; j++) {
    indexToPosition[sortedIndices[j]] = j;
  }

  return {
    useSequential: false,
    indices: sortedIndices,
    indexToPosition
  };
}

/** Helper to get index for accessing data */
function getDataIndex(j: number, info: ReturnType<typeof getIndicesAndMapping>): number {
  return info.useSequential ? j : info.indices![j];
}

/** Helper to get decoration target index based on sorting */
function getDecorationIndex(idx: number, indexToPosition: number[] | null): number {
  return indexToPosition ? indexToPosition[idx] : idx;
}

const pointCloudSpec: PrimitiveSpec<PointCloudComponentConfig> = {
  getCount(elem) {
    return elem.positions.length / 3;
  },

  buildRenderData(elem, sortedIndices?: number[]) {
    const count = elem.positions.length / 3;
    if(count === 0) return null;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const size = elem.size ?? 0.02;
    const sizes = elem.sizes instanceof Float32Array && elem.sizes.length >= count ? elem.sizes : null;

    const indexInfo = getIndicesAndMapping(count, sortedIndices);

    const arr = new Float32Array(count * 8);
    for(let j = 0; j < count; j++) {
      const i = getDataIndex(j, indexInfo);
      arr[j*8+0] = elem.positions[i*3+0];
      arr[j*8+1] = elem.positions[i*3+1];
      arr[j*8+2] = elem.positions[i*3+2];

      if(colors) {
        arr[j*8+3] = colors[i*3+0];
        arr[j*8+4] = colors[i*3+1];
        arr[j*8+5] = colors[i*3+2];
      } else {
        arr[j*8+3] = defaults.color[0];
        arr[j*8+4] = defaults.color[1];
        arr[j*8+5] = defaults.color[2];
      }

      arr[j*8+6] = alphas ? alphas[i] : defaults.alpha;
      const pointSize = sizes ? sizes[i] : size;
      const scale = scales ? scales[i] : defaults.scale;
      arr[j*8+7] = pointSize * scale;
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = getDecorationIndex(idx, indexInfo.indexToPosition);
      if(dec.color) {
        arr[j*8+3] = dec.color[0];
        arr[j*8+4] = dec.color[1];
        arr[j*8+5] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        arr[j*8+6] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        arr[j*8+7] *= dec.scale;
      }
    });

    return arr;
  },

  buildPickingData(elem, baseID, sortedIndices?: number[]) {
    const count = elem.positions.length / 3;
    if(count === 0) return null;

    const size = elem.size ?? 0.02;
    const arr = new Float32Array(count * 5);

    // Check array validities once before the loop
    const hasValidSizes = elem.sizes && elem.sizes.length >= count;
    const sizes = hasValidSizes ? elem.sizes : null;

    if (sortedIndices) {
      // Use sorted indices when available
      for(let j = 0; j < count; j++) {
        const i = sortedIndices[j];
        arr[j*5+0] = elem.positions[i*3+0];
        arr[j*5+1] = elem.positions[i*3+1];
        arr[j*5+2] = elem.positions[i*3+2];
        arr[j*5+3] = baseID + i;  // Keep original index for picking ID
        arr[j*5+4] = sizes?.[i] ?? size;
      }
    } else {
      // When no sorting, just use sequential access
      for(let i = 0; i < count; i++) {
        arr[i*5+0] = elem.positions[i*3+0];
        arr[i*5+1] = elem.positions[i*3+1];
        arr[i*5+2] = elem.positions[i*3+2];
        arr[i*5+3] = baseID + i;
        arr[i*5+4] = sizes?.[i] ?? size;
      }
    }
    return arr;
  },

  // Rendering configuration
  renderConfig: {
    cullMode: 'none',
    topology: 'triangle-list'
  },

  // Pipeline creation methods
  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "PointCloudShading",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: billboardVertCode,
        fragmentShader: billboardFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [POINT_CLOUD_GEOMETRY_LAYOUT, POINT_CLOUD_INSTANCE_LAYOUT],
        primitive: this.renderConfig,
        blend: {
          color: {
            srcFactor: 'src-alpha',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add'
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add'
          }
        },
        depthStencil: {
          format: 'depth24plus',
          depthWriteEnabled: true,
          depthCompare: 'less'
        }
      }, format),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "PointCloudPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode,
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_pointcloud',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [POINT_CLOUD_GEOMETRY_LAYOUT, POINT_CLOUD_PICKING_INSTANCE_LAYOUT]
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, {
      vertexData: new Float32Array([
        -0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
        -0.5,  0.5, 0.0,     0.0, 0.0, 1.0,
         0.5,  0.5, 0.0,     0.0, 0.0, 1.0
      ]),
      indexData: new Uint16Array([0,1,2, 2,1,3])
    });
  }
};

/** ===================== ELLIPSOID ===================== **/


export interface EllipsoidComponentConfig extends BaseComponentConfig {
  type: 'Ellipsoid';
  centers: Float32Array;
  radii?: Float32Array;     // Per-ellipsoid radii
  radius?: [number, number, number]; // Default radius, defaults to [1,1,1]
}

const ellipsoidSpec: PrimitiveSpec<EllipsoidComponentConfig> = {
  getCount(elem) {
    return elem.centers.length / 3;
  },

  buildRenderData(elem, sortedIndices?: number[]) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const defaultRadius = elem.radius ?? [1, 1, 1];
    const radii = elem.radii && elem.radii.length >= count * 3 ? elem.radii : null;

    const indexInfo = getIndicesAndMapping(count, sortedIndices);

    const arr = new Float32Array(count * 10);
    for(let j = 0; j < count; j++) {
      const i = getDataIndex(j, indexInfo);
      arr[j*10+0] = elem.centers[i*3+0];
      arr[j*10+1] = elem.centers[i*3+1];
      arr[j*10+2] = elem.centers[i*3+2];

      // Radii (with scale)
      const scale = scales ? scales[i] : defaults.scale;

      if(radii) {
        arr[j*10+3] = radii[i*3+0] * scale;
        arr[j*10+4] = radii[i*3+1] * scale;
        arr[j*10+5] = radii[i*3+2] * scale;
      } else {
        arr[j*10+3] = defaultRadius[0] * scale;
        arr[j*10+4] = defaultRadius[1] * scale;
        arr[j*10+5] = defaultRadius[2] * scale;
      }

      if(colors) {
        arr[j*10+6] = colors[i*3+0];
        arr[j*10+7] = colors[i*3+1];
        arr[j*10+8] = colors[i*3+2];
      } else {
        arr[j*10+6] = defaults.color[0];
        arr[j*10+7] = defaults.color[1];
        arr[j*10+8] = defaults.color[2];
      }

      arr[j*10+9] = alphas ? alphas[i] : defaults.alpha;
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = getDecorationIndex(idx, indexInfo.indexToPosition);
      if(dec.color) {
        arr[j*10+6] = dec.color[0];
        arr[j*10+7] = dec.color[1];
        arr[j*10+8] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        arr[j*10+9] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        arr[j*10+3] *= dec.scale;
        arr[j*10+4] *= dec.scale;
        arr[j*10+5] *= dec.scale;
      }
    });

    return arr;
  },

  buildPickingData(elem, baseID, sortedIndices?: number[]) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaultRadius = elem.radius ?? [1, 1, 1];
    const arr = new Float32Array(count * 7);

    // Check if we have valid radii array once before the loop
    const hasValidRadii = elem.radii && elem.radii.length >= count * 3;
    const radii = hasValidRadii ? elem.radii : null;

    if (sortedIndices) {
      // Use sorted indices when available
      for(let j = 0; j < count; j++) {
        const i = sortedIndices[j];
        arr[j*7+0] = elem.centers[i*3+0];
        arr[j*7+1] = elem.centers[i*3+1];
        arr[j*7+2] = elem.centers[i*3+2];

        if(radii) {
          arr[j*7+3] = radii[i*3+0];
          arr[j*7+4] = radii[i*3+1];
          arr[j*7+5] = radii[i*3+2];
        } else {
          arr[j*7+3] = defaultRadius[0];
          arr[j*7+4] = defaultRadius[1];
          arr[j*7+5] = defaultRadius[2];
        }
        arr[j*7+6] = baseID + i;  // Keep original index for picking ID
      }
    } else {
      // When no sorting, just use sequential access
      for(let i = 0; i < count; i++) {
        arr[i*7+0] = elem.centers[i*3+0];
        arr[i*7+1] = elem.centers[i*3+1];
        arr[i*7+2] = elem.centers[i*3+2];

        if(radii) {
          arr[i*7+3] = radii[i*3+0];
          arr[i*7+4] = radii[i*3+1];
          arr[i*7+5] = radii[i*3+2];
        } else {
          arr[i*7+3] = defaultRadius[0];
          arr[i*7+4] = defaultRadius[1];
          arr[i*7+5] = defaultRadius[2];
        }
        arr[i*7+6] = baseID + i;
      }
    }
    return arr;
  },

  renderConfig: {
    cullMode: 'back',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "EllipsoidShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: ellipsoidVertCode,
        fragmentShader: ellipsoidFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_INSTANCE_LAYOUT]
      }, format, ellipsoidSpec),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "EllipsoidPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode,
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_ellipsoid',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, MESH_PICKING_INSTANCE_LAYOUT]
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createSphereGeometry(32, 48));
  }
};

/** ===================== ELLIPSOID AXES ===================== **/


export interface EllipsoidAxesComponentConfig extends BaseComponentConfig {
  type: 'EllipsoidAxes';
  centers: Float32Array;
  radii?: Float32Array;
  radius?: [number, number, number];  // Make optional since we have BaseComponentConfig defaults
  colors?: Float32Array;
}

const ellipsoidAxesSpec: PrimitiveSpec<EllipsoidAxesComponentConfig> = {
  getCount(elem) {
    // Each ellipsoid has 3 rings
    return (elem.centers.length / 3) * 3;
  },

  buildRenderData(elem, sortedIndices?: number[]) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const defaultRadius = elem.radius ?? [1, 1, 1];
    const radii = elem.radii instanceof Float32Array && elem.radii.length >= count * 3 ? elem.radii : null;

    const indexInfo = getIndicesAndMapping(count, sortedIndices);

    const ringCount = count * 3;
    const arr = new Float32Array(ringCount * 10);

    for(let j = 0; j < count; j++) {
      const i = getDataIndex(j, indexInfo);
      const cx = elem.centers[i*3+0];
      const cy = elem.centers[i*3+1];
      const cz = elem.centers[i*3+2];
      // Get radii with scale
      const scale = scales ? scales[i] : defaults.scale;

      let rx: number, ry: number, rz: number;
      if (radii) {
        rx = radii[i*3+0];
        ry = radii[i*3+1];
        rz = radii[i*3+2];
      } else {
        rx = defaultRadius[0];
        ry = defaultRadius[1];
        rz = defaultRadius[2];
      }
      rx *= scale;
      ry *= scale;
      rz *= scale;

      // Get colors
      let cr: number, cg: number, cb: number;
      if (colors) {
        cr = colors[i*3+0];
        cg = colors[i*3+1];
        cb = colors[i*3+2];
      } else {
        cr = defaults.color[0];
        cg = defaults.color[1];
        cb = defaults.color[2];
      }
      let alpha = alphas ? alphas[i] : defaults.alpha;

      // Fill 3 rings
      for(let ring = 0; ring < 3; ring++) {
        const arrIdx = j*3 + ring;
        arr[arrIdx*10+0] = cx;
        arr[arrIdx*10+1] = cy;
        arr[arrIdx*10+2] = cz;
        arr[arrIdx*10+3] = rx;
        arr[arrIdx*10+4] = ry;
        arr[arrIdx*10+5] = rz;
        arr[arrIdx*10+6] = cr;
        arr[arrIdx*10+7] = cg;
        arr[arrIdx*10+8] = cb;
        arr[arrIdx*10+9] = alpha;
      }
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = getDecorationIndex(idx, indexInfo.indexToPosition);
      // For each decorated ellipsoid, update all 3 of its rings
      for(let ring = 0; ring < 3; ring++) {
        const arrIdx = j*3 + ring;
        if(dec.color) {
          arr[arrIdx*10+6] = dec.color[0];
          arr[arrIdx*10+7] = dec.color[1];
          arr[arrIdx*10+8] = dec.color[2];
        }
        if(dec.alpha !== undefined) {
          arr[arrIdx*10+9] = dec.alpha;
        }
        if(dec.scale !== undefined) {
          arr[arrIdx*10+3] *= dec.scale;
          arr[arrIdx*10+4] *= dec.scale;
          arr[arrIdx*10+5] *= dec.scale;
        }
      }
    });

    return arr;
  },

  buildPickingData(elem, baseID, sortedIndices?: number[]) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaultRadius = elem.radius ?? [1, 1, 1];
    const ringCount = count * 3;
    const arr = new Float32Array(ringCount * 7);

    if (sortedIndices) {
      // Use sorted indices when available
      for(let j = 0; j < count; j++) {
        const i = sortedIndices[j];
        const cx = elem.centers[i*3+0];
        const cy = elem.centers[i*3+1];
        const cz = elem.centers[i*3+2];
        const rx = elem.radii?.[i*3+0] ?? defaultRadius[0];
        const ry = elem.radii?.[i*3+1] ?? defaultRadius[1];
        const rz = elem.radii?.[i*3+2] ?? defaultRadius[2];
        const thisID = baseID + i;  // Keep original index for picking ID

        for(let ring = 0; ring < 3; ring++) {
          const idx = j*3 + ring;
          arr[idx*7+0] = cx;
          arr[idx*7+1] = cy;
          arr[idx*7+2] = cz;
          arr[idx*7+3] = rx;
          arr[idx*7+4] = ry;
          arr[idx*7+5] = rz;
          arr[idx*7+6] = thisID;
        }
      }
    } else {
      // When no sorting, just use sequential access
      for(let i = 0; i < count; i++) {
        const cx = elem.centers[i*3+0];
        const cy = elem.centers[i*3+1];
        const cz = elem.centers[i*3+2];
        const rx = elem.radii?.[i*3+0] ?? defaultRadius[0];
        const ry = elem.radii?.[i*3+1] ?? defaultRadius[1];
        const rz = elem.radii?.[i*3+2] ?? defaultRadius[2];
        const thisID = baseID + i;

        for(let ring = 0; ring < 3; ring++) {
          const idx = i*3 + ring;
          arr[idx*7+0] = cx;
          arr[idx*7+1] = cy;
          arr[idx*7+2] = cz;
          arr[idx*7+3] = rx;
          arr[idx*7+4] = ry;
          arr[idx*7+5] = rz;
          arr[idx*7+6] = thisID;
        }
      }
    }
    return arr;
  },

  renderConfig: {
    cullMode: 'back',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: ringVertCode,
        fragmentShader: ringFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_INSTANCE_LAYOUT],
        blend: {} // Use defaults
      }, format, ellipsoidAxesSpec),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode,
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_rings',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, MESH_PICKING_INSTANCE_LAYOUT]
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createTorusGeometry(1.0, 0.03, 40, 12));
  }
};

/** ===================== CUBOID ===================== **/


export interface CuboidComponentConfig extends BaseComponentConfig {
  type: 'Cuboid';
  centers: Float32Array;
  sizes: Float32Array;
  size?: [number, number, number];
}

const cuboidSpec: PrimitiveSpec<CuboidComponentConfig> = {
  getCount(elem){
    return elem.centers.length / 3;
  },
  buildRenderData(elem, sortedIndices?: number[]) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, count);

    const defaultSize = elem.size || [0.1, 0.1, 0.1];
    const sizes = elem.sizes && elem.sizes.length >= count * 3 ? elem.sizes : null;

    // Create mapping from original index to sorted position if needed
    const indexToPosition = sortedIndices ? new Array(count) : null;

    const arr = new Float32Array(count * 10);
    for(let j = 0; j < count; j++) {
      const i = sortedIndices ? sortedIndices[j] : j;
      if (indexToPosition) indexToPosition[i] = j;
      const cx = elem.centers[i*3+0];
      const cy = elem.centers[i*3+1];
      const cz = elem.centers[i*3+2];
      const scale = scales ? scales[i] : defaults.scale;

      // Get sizes with scale
      const sx = (sizes ? sizes[i*3+0] : defaultSize[0]) * scale;
      const sy = (sizes ? sizes[i*3+1] : defaultSize[1]) * scale;
      const sz = (sizes ? sizes[i*3+2] : defaultSize[2]) * scale;

      // Get colors
      let cr: number, cg: number, cb: number;
      if (colors) {
        cr = colors[i*3+0];
        cg = colors[i*3+1];
        cb = colors[i*3+2];
      } else {
        cr = defaults.color[0];
        cg = defaults.color[1];
        cb = defaults.color[2];
      }
      const alpha = alphas ? alphas[i] : defaults.alpha;

      // Fill array
      const idx = j * 10;
      arr[idx+0] = cx;
      arr[idx+1] = cy;
      arr[idx+2] = cz;
      arr[idx+3] = sx;
      arr[idx+4] = sy;
      arr[idx+5] = sz;
      arr[idx+6] = cr;
      arr[idx+7] = cg;
      arr[idx+8] = cb;
      arr[idx+9] = alpha;
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, count, (idx, dec) => {
      const j = indexToPosition ? indexToPosition[idx] : idx;  // Get the position where this index ended up
      if(dec.color) {
        arr[j*10+6] = dec.color[0];
        arr[j*10+7] = dec.color[1];
        arr[j*10+8] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        arr[j*10+9] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        arr[j*10+3] *= dec.scale;
        arr[j*10+4] *= dec.scale;
        arr[j*10+5] *= dec.scale;
      }
    });

    return arr;
  },
  buildPickingData(elem, baseID, sortedIndices?: number[]) {
    const count = elem.centers.length / 3;
    if(count === 0) return null;
    const defaultSize = elem.size || [0.1, 0.1, 0.1];
    const sizes = elem.sizes && elem.sizes.length >= count * 3 ? elem.sizes : null;
    const { scales } = getColumnarParams(elem, count);

    const arr = new Float32Array(count * 7);

    if (sortedIndices) {
      // Use sorted indices when available
      for(let j = 0; j < count; j++) {
        const i = sortedIndices[j];
        const scale = scales ? scales[i] : 1;
        // Position
        arr[j*7+0] = elem.centers[i*3+0];
        arr[j*7+1] = elem.centers[i*3+1];
        arr[j*7+2] = elem.centers[i*3+2];
        // Size
        arr[j*7+3] = (sizes ? sizes[i*3+0] : defaultSize[0]) * scale;
        arr[j*7+4] = (sizes ? sizes[i*3+1] : defaultSize[1]) * scale;
        arr[j*7+5] = (sizes ? sizes[i*3+2] : defaultSize[2]) * scale;
        // Picking ID
        arr[j*7+6] = baseID + i;  // Keep original index for picking ID
      }

      // Apply scale decorations for sorted case
      applyDecorations(elem.decorations, count, (idx, dec) => {
        // Find the position of this index in the sorted array
        const j = sortedIndices.indexOf(idx);
        if (j !== -1 && dec.scale !== undefined) {
          arr[j*7+3] *= dec.scale;
          arr[j*7+4] *= dec.scale;
          arr[j*7+5] *= dec.scale;
        }
      });
    } else {
      // When no sorting, just use sequential access
      for(let i = 0; i < count; i++) {
        const scale = scales ? scales[i] : 1;
        // Position
        arr[i*7+0] = elem.centers[i*3+0];
        arr[i*7+1] = elem.centers[i*3+1];
        arr[i*7+2] = elem.centers[i*3+2];
        // Size
        arr[i*7+3] = (sizes ? sizes[i*3+0] : defaultSize[0]) * scale;
        arr[i*7+4] = (sizes ? sizes[i*3+1] : defaultSize[1]) * scale;
        arr[i*7+5] = (sizes ? sizes[i*3+2] : defaultSize[2]) * scale;
        // Picking ID
        arr[i*7+6] = baseID + i;
      }

      // Apply scale decorations for unsorted case
      applyDecorations(elem.decorations, count, (idx, dec) => {
        if (dec.scale !== undefined) {
          arr[idx*7+3] *= dec.scale;
          arr[idx*7+4] *= dec.scale;
          arr[idx*7+5] *= dec.scale;
        }
      });
    }
    return arr;
  },
  renderConfig: {
    cullMode: 'none',  // Cuboids need to be visible from both sides
    topology: 'triangle-list'
  },
  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "CuboidShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: cuboidVertCode,
        fragmentShader: cuboidFragCode,
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, ELLIPSOID_INSTANCE_LAYOUT]
      }, format, cuboidSpec),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "CuboidPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode,
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_cuboid',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [MESH_GEOMETRY_LAYOUT, MESH_PICKING_INSTANCE_LAYOUT],
        primitive: this.renderConfig
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createCubeGeometry());
  }
};

/******************************************************
 *  LineBeams Type
 ******************************************************/

export interface LineBeamsComponentConfig extends BaseComponentConfig {
  type: 'LineBeams';
  positions: Float32Array;  // [x,y,z,i, x,y,z,i, ...]
  sizes?: Float32Array;     // Per-line sizes
  size?: number;         // Default size, defaults to 0.02
}

function countSegments(positions: Float32Array): number {
  const pointCount = positions.length / 4;
  if (pointCount < 2) return 0;

  let segCount = 0;
  for (let p = 0; p < pointCount - 1; p++) {
    const iCurr = positions[p * 4 + 3];
    const iNext = positions[(p+1) * 4 + 3];
    if (iCurr === iNext) {
      segCount++;
    }
  }
  return segCount;
}

const lineBeamsSpec: PrimitiveSpec<LineBeamsComponentConfig> = {
  getCount(elem) {
    return countSegments(elem.positions);
  },

  buildRenderData(elem, sortedIndices?: number[]) {
    const segCount = this.getCount(elem);
    if(segCount === 0) return null;

    const defaults = getBaseDefaults(elem);
    const { colors, alphas, scales } = getColumnarParams(elem, segCount);

    // First pass: build segment mapping
    const segmentMap = new Array(segCount);
    let segIndex = 0;

    const pointCount = elem.positions.length / 4;
    for(let p = 0; p < pointCount - 1; p++) {
      const iCurr = elem.positions[p * 4 + 3];
      const iNext = elem.positions[(p+1) * 4 + 3];
      if(iCurr !== iNext) continue;

      // Store mapping from segment index to point index
      segmentMap[segIndex] = p;
      segIndex++;
    }

    const defaultSize = elem.size ?? 0.02;
    const sizes = elem.sizes instanceof Float32Array && elem.sizes.length >= segCount ? elem.sizes : null;

    const indexInfo = getIndicesAndMapping(segCount, sortedIndices);

    const arr = new Float32Array(segCount * 11);
    for(let j = 0; j < segCount; j++) {
      const i = getDataIndex(j, indexInfo);
      const p = segmentMap[i];
      const lineIndex = Math.floor(elem.positions[p * 4 + 3]);

      // Start point
      arr[j*11+0] = elem.positions[p * 4 + 0];
      arr[j*11+1] = elem.positions[p * 4 + 1];
      arr[j*11+2] = elem.positions[p * 4 + 2];

      // End point
      arr[j*11+3] = elem.positions[(p+1) * 4 + 0];
      arr[j*11+4] = elem.positions[(p+1) * 4 + 1];
      arr[j*11+5] = elem.positions[(p+1) * 4 + 2];

      // Size with scale
      const scale = scales ? scales[lineIndex] : defaults.scale;
      arr[j*11+6] = (sizes ? sizes[lineIndex] : defaultSize) * scale;

      // Colors
      if(colors) {
        arr[j*11+7] = colors[lineIndex*3+0];
        arr[j*11+8] = colors[lineIndex*3+1];
        arr[j*11+9] = colors[lineIndex*3+2];
      } else {
        arr[j*11+7] = defaults.color[0];
        arr[j*11+8] = defaults.color[1];
        arr[j*11+9] = defaults.color[2];
      }

      arr[j*11+10] = alphas ? alphas[lineIndex] : defaults.alpha;
    }

    // Apply decorations using the mapping from original index to sorted position
    applyDecorations(elem.decorations, segCount, (idx, dec) => {
      const j = getDecorationIndex(idx, indexInfo.indexToPosition);
      if(dec.color) {
        arr[j*11+7] = dec.color[0];
        arr[j*11+8] = dec.color[1];
        arr[j*11+9] = dec.color[2];
      }
      if(dec.alpha !== undefined) {
        arr[j*11+10] = dec.alpha;
      }
      if(dec.scale !== undefined) {
        arr[j*11+6] *= dec.scale;
      }
    });

    return arr;
  },

  buildPickingData(elem, baseID, sortedIndices?: number[]) {
    const segCount = this.getCount(elem);
    if(segCount === 0) return null;

    const defaultSize = elem.size ?? 0.02;
    const floatsPerSeg = 8;
    const arr = new Float32Array(segCount * floatsPerSeg);

    // First pass: build segment mapping
    const segmentMap = new Array(segCount);
    let segIndex = 0;

    const pointCount = elem.positions.length / 4;
    for(let p = 0; p < pointCount - 1; p++) {
      const iCurr = elem.positions[p * 4 + 3];
      const iNext = elem.positions[(p+1) * 4 + 3];
      if(iCurr !== iNext) continue;

      // Store mapping from segment index to point index
      segmentMap[segIndex] = p;
      segIndex++;
    }

    if (sortedIndices) {
      // Use sorted indices when available
      for(let j = 0; j < segCount; j++) {
        const i = sortedIndices[j];
        const p = segmentMap[i];
        const lineIndex = Math.floor(elem.positions[p * 4 + 3]);
        let size = elem.sizes?.[lineIndex] ?? defaultSize;
        const scale = elem.scales?.[lineIndex] ?? 1.0;

        size *= scale;

        // Apply decorations that affect size
        applyDecorations(elem.decorations, lineIndex + 1, (idx, dec) => {
          if(idx === lineIndex && dec.scale !== undefined) {
            size *= dec.scale;
          }
        });

        const base = j * floatsPerSeg;
        arr[base + 0] = elem.positions[p * 4 + 0];     // start.x
        arr[base + 1] = elem.positions[p * 4 + 1];     // start.y
        arr[base + 2] = elem.positions[p * 4 + 2];     // start.z
        arr[base + 3] = elem.positions[(p+1) * 4 + 0]; // end.x
        arr[base + 4] = elem.positions[(p+1) * 4 + 1]; // end.y
        arr[base + 5] = elem.positions[(p+1) * 4 + 2]; // end.z
        arr[base + 6] = size;                        // size
        arr[base + 7] = baseID + i;                  // Keep original index for picking ID
      }
    } else {
      // When no sorting, just use sequential access
      for(let i = 0; i < segCount; i++) {
        const p = segmentMap[i];
        const lineIndex = Math.floor(elem.positions[p * 4 + 3]);
        let size = elem.sizes?.[lineIndex] ?? defaultSize;
        const scale = elem.scales?.[lineIndex] ?? 1.0;

        size *= scale;

        // Apply decorations that affect size
        applyDecorations(elem.decorations, lineIndex + 1, (idx, dec) => {
          if(idx === lineIndex && dec.scale !== undefined) {
            size *= dec.scale;
          }
        });

        const base = i * floatsPerSeg;
        arr[base + 0] = elem.positions[p * 4 + 0];     // start.x
        arr[base + 1] = elem.positions[p * 4 + 1];     // start.y
        arr[base + 2] = elem.positions[p * 4 + 2];     // start.z
        arr[base + 3] = elem.positions[(p+1) * 4 + 0]; // end.x
        arr[base + 4] = elem.positions[(p+1) * 4 + 1]; // end.y
        arr[base + 5] = elem.positions[(p+1) * 4 + 2]; // end.z
        arr[base + 6] = size;                        // size
        arr[base + 7] = baseID + i;                  // Keep original index for picking ID
      }
    }
    return arr;
  },

  // Standard triangle-list, cull as you like
  renderConfig: {
    cullMode: 'none',
    topology: 'triangle-list'
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "LineBeamsShading",
      () => createTranslucentGeometryPipeline(device, bindGroupLayout, {
        vertexShader: lineBeamVertCode,   // defined below
        fragmentShader: lineBeamFragCode, // defined below
        vertexEntryPoint: 'vs_main',
        fragmentEntryPoint: 'fs_main',
        bufferLayouts: [ MESH_GEOMETRY_LAYOUT, LINE_BEAM_INSTANCE_LAYOUT ],
      }, format, this),
      cache
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "LineBeamsPicking",
      () => createRenderPipeline(device, bindGroupLayout, {
        vertexShader: pickingVertCode, // We'll add a vs_lineCyl entry
        fragmentShader: pickingVertCode,
        vertexEntryPoint: 'vs_lineBeam',
        fragmentEntryPoint: 'fs_pick',
        bufferLayouts: [ MESH_GEOMETRY_LAYOUT, LINE_BEAM_PICKING_INSTANCE_LAYOUT ],
        primitive: this.renderConfig
      }, 'rgba8unorm'),
      cache
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createBeamGeometry());
  }
};


/******************************************************
 * 4) Pipeline Cache Helper
 ******************************************************/
// Update the pipeline cache to include device reference
export interface PipelineCacheEntry {
  pipeline: GPURenderPipeline;
  device: GPUDevice;
}

function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  createFn: () => GPURenderPipeline,
  cache: Map<string, PipelineCacheEntry>  // This will be the instance cache
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

/******************************************************
 * 5) Common Resources: Geometry, Layout, etc.
 ******************************************************/
export interface GeometryResource {
  vb: GPUBuffer;
  ib: GPUBuffer;
  indexCount?: number;
}

export type GeometryResources = {
  [K in keyof typeof primitiveRegistry]: GeometryResource | null;
}

function getGeometryResource(resources: GeometryResources, type: keyof GeometryResources): GeometryResource {
  const resource = resources[type];
  if (!resource) {
    throw new Error(`No geometry resource found for type ${type}`);
  }
  return resource;
}


const createBuffers = (device: GPUDevice, { vertexData, indexData }: { vertexData: Float32Array, indexData: Uint16Array | Uint32Array }) => {
  const vb = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(vb, 0, vertexData);

  const ib = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(ib, 0, indexData);

  return { vb, ib, indexCount: indexData.length };
};

function initGeometryResources(device: GPUDevice, resources: GeometryResources) {
  // Create geometry for each primitive type
  for (const [primitiveName, spec] of Object.entries(primitiveRegistry)) {
    const typedName = primitiveName as keyof GeometryResources;
    if (!resources[typedName]) {
      resources[typedName] = spec.createGeometryResource(device);
    }
  }
}

/******************************************************
 * 6) Pipeline Configuration Helpers
 ******************************************************/
interface VertexBufferLayout {
  arrayStride: number;
  stepMode?: GPUVertexStepMode;
  attributes: {
    shaderLocation: number;
    offset: number;
    format: GPUVertexFormat;
  }[];
}

interface PipelineConfig {
  vertexShader: string;
  fragmentShader: string;
  vertexEntryPoint: string;
  fragmentEntryPoint: string;
  bufferLayouts: VertexBufferLayout[];
  primitive?: {
    topology?: GPUPrimitiveTopology;
    cullMode?: GPUCullMode;
  };
  blend?: {
    color?: GPUBlendComponent;
    alpha?: GPUBlendComponent;
  };
  depthStencil?: {
    format: GPUTextureFormat;
    depthWriteEnabled: boolean;
    depthCompare: GPUCompareFunction;
  };
  colorWriteMask?: GPUColorWriteFlags;  // Add this to control color writes
}

function createRenderPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout,
  config: PipelineConfig,
  format: GPUTextureFormat
): GPURenderPipeline {
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout]
  });

  // Get primitive configuration with defaults
  const primitiveConfig = {
    topology: config.primitive?.topology || 'triangle-list',
    cullMode: config.primitive?.cullMode || 'back'
  };

  return device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: device.createShaderModule({ code: config.vertexShader }),
      entryPoint: config.vertexEntryPoint,
      buffers: config.bufferLayouts
    },
    fragment: {
      module: device.createShaderModule({ code: config.fragmentShader }),
      entryPoint: config.fragmentEntryPoint,
      targets: [{
        format,
        writeMask: config.colorWriteMask ?? GPUColorWrite.ALL,
        ...(config.blend && {
          blend: {
            color: config.blend.color || {
              srcFactor: 'src-alpha',
              dstFactor: 'one-minus-src-alpha'
            },
            alpha: config.blend.alpha || {
              srcFactor: 'one',
              dstFactor: 'one-minus-src-alpha'
            }
          }
        })
      }]
    },
    primitive: primitiveConfig,
    depthStencil: config.depthStencil || {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less'
    }
  });
}

function createTranslucentGeometryPipeline(
  device: GPUDevice,
  bindGroupLayout: GPUBindGroupLayout,
  config: PipelineConfig,
  format: GPUTextureFormat,
  primitiveSpec: PrimitiveSpec<any>  // Take the primitive spec instead of just type
): GPURenderPipeline {
  return createRenderPipeline(device, bindGroupLayout, {
    ...config,
    primitive: primitiveSpec.renderConfig,
    blend: {
      color: {
        srcFactor: 'src-alpha',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add'
      },
      alpha: {
        srcFactor: 'one',
        dstFactor: 'one-minus-src-alpha',
        operation: 'add'
      }
    },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less'
    }
  }, format);
}


// Common vertex buffer layouts
const POINT_CLOUD_GEOMETRY_LAYOUT: VertexBufferLayout = {
  arrayStride: 24,  // 6 floats * 4 bytes
  attributes: [
    {  // position xyz
      shaderLocation: 0,
      offset: 0,
      format: 'float32x3'
    },
    {  // normal xyz
      shaderLocation: 1,
      offset: 12,
      format: 'float32x3'
    }
  ]
};

const POINT_CLOUD_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 32,  // 8 floats * 4 bytes
  stepMode: 'instance',
  attributes: [
    {shaderLocation: 2, offset: 0,  format: 'float32x3'},  // instancePos
    {shaderLocation: 3, offset: 12, format: 'float32x3'},  // color
    {shaderLocation: 4, offset: 24, format: 'float32'},    // alpha
    {shaderLocation: 5, offset: 28, format: 'float32'}     // size
  ]
};

const POINT_CLOUD_PICKING_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 20,  // 5 floats * 4 bytes
  stepMode: 'instance',
  attributes: [
    {shaderLocation: 2, offset: 0,   format: 'float32x3'},  // instancePos
    {shaderLocation: 3, offset: 12,  format: 'float32'},    // pickID
    {shaderLocation: 4, offset: 16,  format: 'float32'}     // size
  ]
};

const MESH_GEOMETRY_LAYOUT: VertexBufferLayout = {
  arrayStride: 6*4,
  attributes: [
    {shaderLocation: 0, offset: 0,   format: 'float32x3'},
    {shaderLocation: 1, offset: 3*4, format: 'float32x3'}
  ]
};

const ELLIPSOID_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 10*4,
  stepMode: 'instance',
  attributes: [
    {shaderLocation: 2, offset: 0,     format: 'float32x3'},
    {shaderLocation: 3, offset: 3*4,   format: 'float32x3'},
    {shaderLocation: 4, offset: 6*4,   format: 'float32x3'},
    {shaderLocation: 5, offset: 9*4,   format: 'float32'}
  ]
};

const MESH_PICKING_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 7*4,  // 7 floats: position(3) + size(3) + pickID(1)
  stepMode: 'instance',
  attributes: [
    {shaderLocation: 2, offset: 0,   format: 'float32x3'},  // center
    {shaderLocation: 3, offset: 3*4, format: 'float32x3'},  // size
    {shaderLocation: 4, offset: 6*4, format: 'float32'}     // pickID
  ]
};

const CYL_GEOMETRY_LAYOUT: VertexBufferLayout = {
  arrayStride: 6 * 4, // (pos.x, pos.y, pos.z, norm.x, norm.y, norm.z)
  attributes: [
    { shaderLocation: 0, offset: 0,  format: 'float32x3' }, // position
    { shaderLocation: 1, offset: 12, format: 'float32x3' } // normal
  ]
};

// For rendering: 11 floats
// (start.xyz, end.xyz, size, color.rgb, alpha)
const LINE_BEAM_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 11 * 4,
  stepMode: 'instance',
  attributes: [
    { shaderLocation: 2, offset:  0,  format: 'float32x3' }, // startPos
    { shaderLocation: 3, offset: 12,  format: 'float32x3' }, // endPos
    { shaderLocation: 4, offset: 24,  format: 'float32'   }, // size
    { shaderLocation: 5, offset: 28,  format: 'float32x3' }, // color
    { shaderLocation: 6, offset: 40,  format: 'float32'   }, // alpha
  ]
};

// For picking: 8 floats
// (start.xyz, end.xyz, size, pickID)
const LINE_BEAM_PICKING_INSTANCE_LAYOUT: VertexBufferLayout = {
  arrayStride: 8 * 4,
  stepMode: 'instance',
  attributes: [
    { shaderLocation: 2, offset:  0,  format: 'float32x3' },
    { shaderLocation: 3, offset: 12,  format: 'float32x3' },
    { shaderLocation: 4, offset: 24,  format: 'float32'   }, // size
    { shaderLocation: 5, offset: 28,  format: 'float32'   },
  ]
};

/******************************************************
 * 7) Primitive Registry
 ******************************************************/
export type ComponentConfig =
  | PointCloudComponentConfig
  | EllipsoidComponentConfig
  | EllipsoidAxesComponentConfig
  | CuboidComponentConfig
  | LineBeamsComponentConfig;

const primitiveRegistry: Record<ComponentConfig['type'], PrimitiveSpec<any>> = {
  PointCloud: pointCloudSpec,  // Use consolidated spec
  Ellipsoid: ellipsoidSpec,
  EllipsoidAxes: ellipsoidAxesSpec,
  Cuboid: cuboidSpec,
  LineBeams: lineBeamsSpec
};


/******************************************************
 * 8) Scene
 ******************************************************/

export function SceneInner({
  components,
  containerWidth,
  containerHeight,
  style,
  camera: controlledCamera,
  defaultCamera,
  onCameraChange,
  onFrameRendered
}: SceneInnerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
    componentBaseId: number[];
    idToComponent: ({componentIdx: number, instanceIdx: number} | null)[];
    pipelineCache: Map<string, PipelineCacheEntry>;
    dynamicBuffers: DynamicBuffers | null;
    resources: GeometryResources;
  } | null>(null);

  const [isReady, setIsReady] = useState(false);

  // Update the camera initialization
  const [internalCamera, setInternalCamera] = useState<CameraState>(() => {
      return createCameraState(defaultCamera);
  });

  // Use the appropriate camera state based on whether we're controlled or not
  const activeCamera = useMemo(() => {
      if (controlledCamera) {
          return createCameraState(controlledCamera);
      }
      return internalCamera;
  }, [controlledCamera, internalCamera]);

  // Update handleCameraUpdate to use activeCamera
  const handleCameraUpdate = useCallback((updateFn: (camera: CameraState) => CameraState) => {
    const newCameraState = updateFn(activeCamera);

    if (controlledCamera) {
        onCameraChange?.(createCameraParams(newCameraState));
    } else {
        setInternalCamera(newCameraState);
        onCameraChange?.(createCameraParams(newCameraState));
    }
}, [activeCamera, controlledCamera, onCameraChange]);

  // We'll also track a picking lock
  const pickingLockRef = useRef(false);

  // Add hover state tracking
  const lastHoverState = useRef<{componentIdx: number, instanceIdx: number} | null>(null);

  /******************************************************
   * A) initWebGPU
   ******************************************************/
  const initWebGPU = useCallback(async()=>{
    if(!canvasRef.current) return;
    if(!navigator.gpu) {
      console.error("WebGPU not supported in this browser.");
      return;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if(!adapter) throw new Error("No GPU adapter found");
      const device = await adapter.requestDevice();

      const context = canvasRef.current.getContext('webgpu') as GPUCanvasContext;
      const format = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format, alphaMode:'premultiplied' });

      // Create bind group layout
      const bindGroupLayout = device.createBindGroupLayout({
        entries: [{
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {type:'uniform'}
        }]
      });

      // Create uniform buffer
      const uniformBufferSize=128;
      const uniformBuffer=device.createBuffer({
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });

      // Create bind group using the new layout
      const uniformBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [{ binding:0, resource:{ buffer:uniformBuffer } }]
      });

      // Readback buffer for picking
      const readbackBuffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: 'Picking readback buffer'
      });

      // First create gpuRef.current with empty resources
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
        componentBaseId: [],
        idToComponent: [null],  // First ID (0) is reserved
        pipelineCache: new Map(),
        dynamicBuffers: null,
        resources: {
          PointCloud: null,
          Ellipsoid: null,
          EllipsoidAxes: null,
          Cuboid: null,
          LineBeams: null
        },
      };

      // Now initialize geometry resources
      initGeometryResources(device, gpuRef.current.resources);

      setIsReady(true);
    } catch(err){
      console.error("initWebGPU error:", err);
    }
  },[]);

  /******************************************************
   * B) Depth & Pick textures
   ******************************************************/
  const createOrUpdateDepthTexture = useCallback(() => {
    if(!gpuRef.current || !canvasRef.current) return;
    const { device, depthTexture } = gpuRef.current;

    // Get the actual canvas size
    const canvas = canvasRef.current;
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;

    if(depthTexture) depthTexture.destroy();
    const dt = device.createTexture({
        size: [displayWidth, displayHeight],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    gpuRef.current.depthTexture = dt;
}, []);

  const createOrUpdatePickTextures = useCallback(() => {
    if(!gpuRef.current || !canvasRef.current) return;
    const { device, pickTexture, pickDepthTexture } = gpuRef.current;

    // Get the actual canvas size
    const canvas = canvasRef.current;
    const displayWidth = canvas.width;
    const displayHeight = canvas.height;

    if(pickTexture) pickTexture.destroy();
    if(pickDepthTexture) pickDepthTexture.destroy();

    const colorTex = device.createTexture({
        size: [displayWidth, displayHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });
    const depthTex = device.createTexture({
        size: [displayWidth, displayHeight],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    gpuRef.current.pickTexture = colorTex;
    gpuRef.current.pickDepthTexture = depthTex;
}, []);

  /******************************************************
   * C) Building the RenderObjects (no if/else)
   ******************************************************/
  // Move ID mapping logic to a separate function
  const buildComponentIdMapping = useCallback((components: ComponentConfig[]) => {
    if (!gpuRef.current) return;

    // Reset ID mapping
    gpuRef.current.idToComponent = [null];  // First ID (0) is reserved
    let currentID = 1;

    // Build new mapping
    components.forEach((elem, componentIdx) => {
      const spec = primitiveRegistry[elem.type];
      if (!spec) {
        gpuRef.current!.componentBaseId[componentIdx] = 0;
        return;
      }

      const count = spec.getCount(elem);
      gpuRef.current!.componentBaseId[componentIdx] = currentID;

      // Expand global ID table
      for (let j = 0; j < count; j++) {
        gpuRef.current!.idToComponent[currentID + j] = {
          componentIdx: componentIdx,
          instanceIdx: j
        };
      }
      currentID += count;
    });
  }, []);

  // Helper to collect and organize component data by type
  function collectTypeData<T>(
    components: ComponentConfig[],
    getData: (comp: ComponentConfig, spec: PrimitiveSpec<any>) => T | null,
    getSize: (data: T, count: number) => number
  ) {
    const typeArrays = new Map<ComponentConfig['type'], {
      totalCount: number,
      totalSize: number,
      components: ComponentConfig[],
      indices: number[],
      offsets: number[],
      counts: number[],
      datas: T[]
    }>();

    // Single pass through components
    components.forEach((comp, idx) => {
      const type = comp.type;
      const spec = primitiveRegistry[type];
      if (!spec) return;

      const count = spec.getCount(comp);
      if (count === 0) return;

      const data = getData(comp, spec);
      if (!data) return;

      const size = getSize(data, count);

      let typeInfo = typeArrays.get(type);
      if (!typeInfo) {
        typeInfo = {
          totalCount: 0,
          totalSize: 0,
          components: [],
          indices: [],
          offsets: [],
          counts: [],
          datas: []
        };
        typeArrays.set(type, typeInfo);
      }

      typeInfo.components.push(comp);
      typeInfo.indices.push(idx);
      typeInfo.offsets.push(typeInfo.totalSize);
      typeInfo.counts.push(count);
      typeInfo.datas.push(data);
      typeInfo.totalCount += count;
      typeInfo.totalSize += size;
    });

    return typeArrays;
  }

  // Create dynamic buffers helper
  function createDynamicBuffers(device: GPUDevice, renderSize: number, pickingSize: number) {
    const renderBuffer = device.createBuffer({
      size: renderSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false
    });

    const pickingBuffer = device.createBuffer({
      size: pickingSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: false
    });

    return {
      renderBuffer,
      pickingBuffer,
      renderOffset: 0,
      pickingOffset: 0
    };
  }

  // Replace getTransparencyInfo function
  function getTransparencyInfo(component: ComponentConfig, spec: PrimitiveSpec<any>): RenderObject['transparencyInfo'] | undefined {
    const count = spec.getCount(component);
    if (count === 0) return undefined;

    const defaults = getBaseDefaults(component);
    const { alphas } = getColumnarParams(component, count);
    const needsSort = hasTransparency(alphas, defaults.alpha, component.decorations);
    if (!needsSort) return undefined;

    // Extract centers based on component type
    switch (component.type) {
      case 'PointCloud':
        return {
          needsSort,
          centers: component.positions,
          stride: 3,
          offset: 0
        };
      case 'Ellipsoid':
      case 'Cuboid':
        return {
          needsSort,
          centers: component.centers,
          stride: 3,
          offset: 0
        };
      case 'LineBeams': {
        // Compute centers for line segments
        const segCount = spec.getCount(component);
        const centers = new Float32Array(segCount * 3);
        let segIndex = 0;
        const pointCount = component.positions.length / 4;

        for(let p = 0; p < pointCount - 1; p++) {
          const iCurr = component.positions[p * 4 + 3];
          const iNext = component.positions[(p+1) * 4 + 3];
          if(iCurr !== iNext) continue;

          centers[segIndex*3+0] = (component.positions[p*4+0] + component.positions[(p+1)*4+0]) * 0.5;
          centers[segIndex*3+1] = (component.positions[p*4+1] + component.positions[(p+1)*4+1]) * 0.5;
          centers[segIndex*3+2] = (component.positions[p*4+2] + component.positions[(p+1)*4+2]) * 0.5;
          segIndex++;
        }
        return {
          needsSort,
          centers,
          stride: 3,
          offset: 0
        };
      }
      default:
        return undefined;
    }
  }

  // Update buildRenderObjects to include transparency info
  function buildRenderObjects(components: ComponentConfig[]): RenderObject[] {
    if(!gpuRef.current) return [];
    const { device, bindGroupLayout, pipelineCache, resources } = gpuRef.current;

    // Collect render data using helper
    const typeArrays = collectTypeData(
      components,
      (comp, spec) => spec.buildRenderData(comp),  // No sorted indices yet - will be applied in render
      (data, count) => {
        const stride = Math.ceil(data.length / count) * 4;
        return stride * count;
      }
    );

    // Calculate total buffer size needed
    let totalRenderSize = 0;
    typeArrays.forEach(info => {
      totalRenderSize += info.totalSize;
    });

    // Create or recreate dynamic buffers if needed
    if (!gpuRef.current.dynamicBuffers ||
        gpuRef.current.dynamicBuffers.renderBuffer.size < totalRenderSize) {
      if (gpuRef.current.dynamicBuffers) {
        gpuRef.current.dynamicBuffers.renderBuffer.destroy();
        gpuRef.current.dynamicBuffers.pickingBuffer.destroy();
      }
      gpuRef.current.dynamicBuffers = createDynamicBuffers(
        device,
        totalRenderSize,
        totalRenderSize
      );
    }
    const dynamicBuffers = gpuRef.current.dynamicBuffers!;

    // Reset buffer offsets
    dynamicBuffers.renderOffset = 0;
    dynamicBuffers.pickingOffset = 0;

    // Initialize componentBaseId array and build ID mapping
    gpuRef.current.componentBaseId = new Array(components.length).fill(0);
    buildComponentIdMapping(components);

    const validRenderObjects: RenderObject[] = [];

    // Create render objects and write buffer data
    typeArrays.forEach((typeInfo, type) => {
      const spec = primitiveRegistry[type];
      if (!spec) return;

      try {
        const renderOffset = Math.ceil(dynamicBuffers.renderOffset / 4) * 4;

        // Write each component's data directly to final position
        typeInfo.datas.forEach((data, i) => {
          device.queue.writeBuffer(
            dynamicBuffers.renderBuffer,
            renderOffset + typeInfo.offsets[i],
            data.buffer,
            data.byteOffset,
            data.byteLength
          );
        });

        // Create pipeline once for this type
        const pipeline = spec.getRenderPipeline(device, bindGroupLayout, pipelineCache);
        if (!pipeline) return;

        // Create single render object for all instances of this type
        const geometryResource = getGeometryResource(resources, type);
        const stride = Math.ceil(
          typeInfo.datas[0].length / typeInfo.counts[0]
        ) * 4;

        const renderObject: RenderObject = {
          pipeline,
          pickingPipeline: undefined,
          vertexBuffers: [
            geometryResource.vb,
            {
              buffer: dynamicBuffers.renderBuffer,
              offset: renderOffset,
              stride: stride
            }
          ],
          indexBuffer: geometryResource.ib,
          indexCount: geometryResource.indexCount,
          instanceCount: typeInfo.totalCount,
          pickingVertexBuffers: [undefined, undefined] as [GPUBuffer | undefined, BufferInfo | undefined],
          pickingDataStale: true,
          componentIndex: typeInfo.indices[0]
        };

        // Add transparency info if needed
        const component = components[typeInfo.indices[0]];
        renderObject.transparencyInfo = getTransparencyInfo(component, spec);

        validRenderObjects.push(renderObject);
        dynamicBuffers.renderOffset = renderOffset + typeInfo.totalSize;

      } catch (error) {
        console.error(`Error creating render object for type ${type}:`, error);
      }
    });

    return validRenderObjects;
  }

  /******************************************************
   * D) Render pass (single call, no loop)
   ******************************************************/

  // Add validation helper
function isValidRenderObject(ro: RenderObject): ro is Required<Pick<RenderObject, 'pipeline' | 'vertexBuffers' | 'instanceCount'>> & {
  vertexBuffers: [GPUBuffer, BufferInfo];
} & RenderObject {
  return (
    ro.pipeline !== undefined &&
    Array.isArray(ro.vertexBuffers) &&
    ro.vertexBuffers.length === 2 &&
    ro.vertexBuffers[0] !== undefined &&
    ro.vertexBuffers[1] !== undefined &&
    'buffer' in ro.vertexBuffers[1] &&
    'offset' in ro.vertexBuffers[1] &&
    (ro.indexBuffer !== undefined || ro.vertexCount !== undefined) &&
    typeof ro.instanceCount === 'number' &&
    ro.instanceCount > 0
  );
}

  // Add this function to update sorted data in buffers
  function updateSortedBuffers(
    renderObjects: RenderObject[],
    components: ComponentConfig[],
    globalSortedIndices: Map<number, number[]>
  ) {
    if (!gpuRef.current?.device || !gpuRef.current.dynamicBuffers) return;
    const { device } = gpuRef.current;

    // For each render object that needs sorting
    renderObjects.forEach(ro => {
      const component = components[ro.componentIndex];
      const spec = primitiveRegistry[component.type];
      if (!spec) return;

      const sortedIndices = globalSortedIndices.get(ro.componentIndex);
      if (!sortedIndices) return;

      // Rebuild render data with new sorting
      const data = spec.buildRenderData(component, sortedIndices);
      if (!data) return;

      // Update buffer with new sorted data
      const vertexInfo = ro.vertexBuffers[1] as BufferInfo;
      device.queue.writeBuffer(
        vertexInfo.buffer,
        vertexInfo.offset,
        data.buffer,
        data.byteOffset,
        data.byteLength
      );
    });
  }

  // Update renderFrame to handle transparency sorting
  const renderFrame = useCallback((camState: CameraState) => {
    if(!gpuRef.current) return;
    const {
      device, context, uniformBuffer, uniformBindGroup,
      renderObjects, depthTexture
    } = gpuRef.current;

    // Update transparency sorting if needed
    const cameraPos: [number, number, number] = [
      camState.position[0],
      camState.position[1],
      camState.position[2]
    ];

    // Check each render object for transparency updates
    renderObjects.forEach(ro => {
      if (!ro.transparencyInfo?.needsSort) return;

      // Check if camera has moved enough to require resorting
      const lastPos = ro.transparencyInfo.lastCameraPosition;
      if (lastPos) {
        const dx = cameraPos[0] - lastPos[0];
        const dy = cameraPos[1] - lastPos[1];
        const dz = cameraPos[2] - lastPos[2];
        const moveDistSq = dx*dx + dy*dy + dz*dz;
        if (moveDistSq < 0.0001) return; // Skip if camera hasn't moved much
      }

      // Get sorted indices and store them
      const sortedIndices = getSortedIndices(ro.transparencyInfo.centers, cameraPos);
      ro.transparencyInfo.sortedIndices = sortedIndices;  // Save for picking

      // Update buffer with sorted data
      const component = components[ro.componentIndex];
      const spec = primitiveRegistry[component.type];
      const data = spec.buildRenderData(component, sortedIndices);

      if (data) {
        const vertexInfo = ro.vertexBuffers[1] as BufferInfo;
        device.queue.writeBuffer(
          vertexInfo.buffer,
          vertexInfo.offset,
          data.buffer,
          data.byteOffset,
          data.byteLength
        );
      }

      // Update last camera position
      ro.transparencyInfo.lastCameraPosition = cameraPos;
    });

    // Update camera uniforms
    const aspect = containerWidth / containerHeight;
    const view = glMatrix.mat4.lookAt(
      glMatrix.mat4.create(),
      camState.position,
      camState.target,
      camState.up
    );

    const proj = glMatrix.mat4.perspective(
      glMatrix.mat4.create(),
      glMatrix.glMatrix.toRadian(camState.fov),
      aspect,
      camState.near,
      camState.far
    );

    // Compute MVP matrix
    const mvp = glMatrix.mat4.multiply(
      glMatrix.mat4.create(),
      proj,
      view
    );

    // Compute camera vectors for lighting
    const forward = glMatrix.vec3.sub(glMatrix.vec3.create(), camState.target, camState.position);
    const right = glMatrix.vec3.cross(glMatrix.vec3.create(), forward, camState.up);
    glMatrix.vec3.normalize(right, right);

    const camUp = glMatrix.vec3.cross(glMatrix.vec3.create(), right, forward);
    glMatrix.vec3.normalize(camUp, camUp);
    glMatrix.vec3.normalize(forward, forward);

    // Compute light direction in camera space
    const lightDir = glMatrix.vec3.create();
    glMatrix.vec3.scaleAndAdd(lightDir, lightDir, right, LIGHTING.DIRECTION.RIGHT);
    glMatrix.vec3.scaleAndAdd(lightDir, lightDir, camUp, LIGHTING.DIRECTION.UP);
    glMatrix.vec3.scaleAndAdd(lightDir, lightDir, forward, LIGHTING.DIRECTION.FORWARD);
    glMatrix.vec3.normalize(lightDir, lightDir);

    // Write uniforms
    const uniformData = new Float32Array([
      ...Array.from(mvp),
      right[0], right[1], right[2], 0,  // pad to vec4
      camUp[0], camUp[1], camUp[2], 0,  // pad to vec4
      lightDir[0], lightDir[1], lightDir[2], 0,  // pad to vec4
      camState.position[0], camState.position[1], camState.position[2], 0  // Add camera position
    ]);
    device.queue.writeBuffer(uniformBuffer, 0, uniformData);

    // Begin render pass
    const cmd = device.createCommandEncoder();
    const pass = cmd.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store'
      }],
      depthStencilAttachment: depthTexture ? {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store'
      } : undefined
    });

    // Draw each object
    for(const ro of renderObjects) {
      if (!isValidRenderObject(ro)) {
        continue;
      }

      pass.setPipeline(ro.pipeline);
      pass.setBindGroup(0, uniformBindGroup);

      pass.setVertexBuffer(0, ro.vertexBuffers[0]);

      const instanceInfo = ro.vertexBuffers[1];
      pass.setVertexBuffer(1, instanceInfo.buffer, instanceInfo.offset);

      if(ro.indexBuffer) {
        pass.setIndexBuffer(ro.indexBuffer, 'uint16');
        pass.drawIndexed(ro.indexCount ?? 0, ro.instanceCount ?? 1);
      } else {
        pass.draw(ro.vertexCount ?? 0, ro.instanceCount ?? 1);
      }
    }

    pass.end();
    device.queue.submit([cmd.finish()]);
    onFrameRendered?.(performance.now());
  }, [containerWidth, containerHeight, onFrameRendered, components]);

  /******************************************************
   * E) Pick pass (on hover/click)
   ******************************************************/
  async function pickAtScreenXY(screenX: number, screenY: number, mode: 'hover'|'click') {
    if(!gpuRef.current || !canvasRef.current || pickingLockRef.current) return;
    const pickingId = Date.now();
    const currentPickingId = pickingId;
    pickingLockRef.current = true;

    try {
      const {
        device, pickTexture, pickDepthTexture, readbackBuffer,
        uniformBindGroup, renderObjects, idToComponent
      } = gpuRef.current;
      if(!pickTexture || !pickDepthTexture || !readbackBuffer) return;
      if (currentPickingId !== pickingId) return;

      // Ensure picking data is ready for all objects
      for (let i = 0; i < renderObjects.length; i++) {
        const ro = renderObjects[i];
        if (ro.pickingDataStale) {
          ensurePickingData(ro, components[i]);
        }
      }

      // Convert screen coordinates to device pixels
      const dpr = window.devicePixelRatio || 1;
      const pickX = Math.floor(screenX * dpr);
      const pickY = Math.floor(screenY * dpr);
      const displayWidth = Math.floor(containerWidth * dpr);
      const displayHeight = Math.floor(containerHeight * dpr);

      if(pickX < 0 || pickY < 0 || pickX >= displayWidth || pickY >= displayHeight) {
        if(mode === 'hover') handleHoverID(0);
        return;
      }

      const cmd = device.createCommandEncoder({label: 'Picking encoder'});
      const passDesc: GPURenderPassDescriptor = {
        colorAttachments:[{
          view: pickTexture.createView(),
          clearValue:{r:0,g:0,b:0,a:1},
          loadOp:'clear',
          storeOp:'store'
        }],
        depthStencilAttachment:{
          view: pickDepthTexture.createView(),
          depthClearValue:1.0,
          depthLoadOp:'clear',
          depthStoreOp:'store'
        }
      };
      const pass = cmd.beginRenderPass(passDesc);
      pass.setBindGroup(0, uniformBindGroup);

      for(const ro of renderObjects) {
        if (!ro.pickingPipeline || !ro.pickingVertexBuffers[0] || !ro.pickingVertexBuffers[1]) {
          continue;
        }

        pass.setPipeline(ro.pickingPipeline);

        // Set geometry buffer
        pass.setVertexBuffer(0, ro.pickingVertexBuffers[0]);

        // Set instance buffer
        const instanceInfo = ro.pickingVertexBuffers[1] as BufferInfo;
        pass.setVertexBuffer(1, instanceInfo.buffer, instanceInfo.offset);

        if(ro.pickingIndexBuffer) {
          pass.setIndexBuffer(ro.pickingIndexBuffer, 'uint16');
          pass.drawIndexed(ro.pickingIndexCount ?? 0, ro.pickingInstanceCount ?? 1);
        } else {
          pass.draw(ro.pickingVertexCount ?? 0, ro.pickingInstanceCount ?? 1);
        }
      }

      pass.end();

      cmd.copyTextureToBuffer(
        {texture: pickTexture, origin:{x:pickX,y:pickY}},
        {buffer: readbackBuffer, bytesPerRow:256, rowsPerImage:1},
        [1,1,1]
      );
      device.queue.submit([cmd.finish()]);

      if (currentPickingId !== pickingId) return;
      await readbackBuffer.mapAsync(GPUMapMode.READ);
      if (currentPickingId !== pickingId) {
        readbackBuffer.unmap();
        return;
      }
      const arr = new Uint8Array(readbackBuffer.getMappedRange());
      const r=arr[0], g=arr[1], b=arr[2];
      readbackBuffer.unmap();
      const pickedID = (b<<16)|(g<<8)|r;

      if(mode==='hover'){
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
    const { idToComponent } = gpuRef.current;

    // Get new hover state
    const newHoverState = idToComponent[pickedID] || null;

    // If hover state hasn't changed, do nothing
    if ((!lastHoverState.current && !newHoverState) ||
        (lastHoverState.current && newHoverState &&
         lastHoverState.current.componentIdx === newHoverState.componentIdx &&
         lastHoverState.current.instanceIdx === newHoverState.instanceIdx)) {
      return;
    }

    // Clear previous hover if it exists
    if (lastHoverState.current) {
      const prevComponent = components[lastHoverState.current.componentIdx];
      prevComponent?.onHover?.(null);
    }

    // Set new hover if it exists
    if (newHoverState) {
      const { componentIdx, instanceIdx } = newHoverState;
      if (componentIdx >= 0 && componentIdx < components.length) {
        components[componentIdx].onHover?.(instanceIdx);
      }
    }

    // Update last hover state
    lastHoverState.current = newHoverState;
  }

  function handleClickID(pickedID:number){
    if(!gpuRef.current) return;
    const {idToComponent} = gpuRef.current;
    const rec = idToComponent[pickedID];
    if(!rec) return;
    const {componentIdx, instanceIdx} = rec;
    if(componentIdx<0||componentIdx>=components.length) return;
    components[componentIdx].onClick?.(instanceIdx);
  }

  /******************************************************
   * F) Mouse Handling
   ******************************************************/
  /**
   * Tracks the current state of mouse interaction with the scene.
   * Used for camera control and picking operations.
   */
  interface MouseState {
    /** Current interaction mode */
    type: 'idle'|'dragging';

    /** Which mouse button initiated the drag (0=left, 1=middle, 2=right) */
    button?: number;

    /** Initial X coordinate when drag started */
    startX?: number;

    /** Initial Y coordinate when drag started */
    startY?: number;

    /** Most recent X coordinate during drag */
    lastX?: number;

    /** Most recent Y coordinate during drag */
    lastY?: number;

    /** Whether shift key was held when drag started */
    isShiftDown?: boolean;

    /** Accumulated drag distance in pixels */
    dragDistance?: number;
  }
  const mouseState=useRef<MouseState>({type:'idle'});

  // Add throttling for hover picking
  const throttledPickAtScreenXY = useCallback(
    throttle((x: number, y: number, mode: 'hover'|'click') => {
      pickAtScreenXY(x, y, mode);
    }, 32), // ~30fps
    [pickAtScreenXY]
  );

  // Rename to be more specific to scene3d
  const handleScene3dMouseMove = useCallback((e: MouseEvent) => {
    if(!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const st = mouseState.current;
    if(st.type === 'dragging' && st.lastX !== undefined && st.lastY !== undefined) {
        const dx = e.clientX - st.lastX;
        const dy = e.clientY - st.lastY;
        st.dragDistance = (st.dragDistance||0) + Math.sqrt(dx*dx + dy*dy);

        if(st.button === 2 || st.isShiftDown) {
            handleCameraUpdate(cam => pan(cam, dx, dy));
        } else if(st.button === 0) {
            handleCameraUpdate(cam => orbit(cam, dx, dy));
        }

        st.lastX = e.clientX;
        st.lastY = e.clientY;
    } else if(st.type === 'idle') {
        throttledPickAtScreenXY(x, y, 'hover');
    }
}, [handleCameraUpdate, throttledPickAtScreenXY]);

  const handleScene3dMouseDown = useCallback((e: MouseEvent) => {
    mouseState.current = {
      type: 'dragging',
      button: e.button,
      startX: e.clientX,
      startY: e.clientY,
      lastX: e.clientX,
      lastY: e.clientY,
      isShiftDown: e.shiftKey,
      dragDistance: 0
    };
    e.preventDefault();
  }, []);

  const handleScene3dMouseUp = useCallback((e: MouseEvent) => {
    const st = mouseState.current;
    if(st.type === 'dragging' && st.startX !== undefined && st.startY !== undefined) {
      if(!canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      if((st.dragDistance || 0) < 4) {
        pickAtScreenXY(x, y, 'click');
      }
    }
    mouseState.current = {type: 'idle'};
  }, [pickAtScreenXY]);

  const handleScene3dMouseLeave = useCallback(() => {
    mouseState.current = {type: 'idle'};
  }, []);

  // Update event listener references
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener('mousemove', handleScene3dMouseMove);
    canvas.addEventListener('mousedown', handleScene3dMouseDown);
    canvas.addEventListener('mouseup', handleScene3dMouseUp);
    canvas.addEventListener('mouseleave', handleScene3dMouseLeave);

    return () => {
      canvas.removeEventListener('mousemove', handleScene3dMouseMove);
      canvas.removeEventListener('mousedown', handleScene3dMouseDown);
      canvas.removeEventListener('mouseup', handleScene3dMouseUp);
      canvas.removeEventListener('mouseleave', handleScene3dMouseLeave);
    };
  }, [handleScene3dMouseMove, handleScene3dMouseDown, handleScene3dMouseUp, handleScene3dMouseLeave]);

  /******************************************************
   * G) Lifecycle & Render-on-demand
   ******************************************************/
  // Init once
  useEffect(()=>{
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
  },[initWebGPU]);

  // Create/recreate depth + pick textures
  useEffect(()=>{
    if(isReady){
      createOrUpdateDepthTexture();
      createOrUpdatePickTextures();
    }
  },[isReady, containerWidth, containerHeight, createOrUpdateDepthTexture, createOrUpdatePickTextures]);

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
        renderFrame(activeCamera);
    }
}, [containerWidth, containerHeight, createOrUpdateDepthTexture, createOrUpdatePickTextures, renderFrame]);

  // Update components effect
  useEffect(() => {
    if (isReady && gpuRef.current) {
      const ros = buildRenderObjects(components);
      gpuRef.current.renderObjects = ros;
      renderFrame(activeCamera);
    }
  }, [isReady, components]);

  // Add separate effect just for camera updates
  useEffect(() => {
    if (isReady && gpuRef.current) {
      renderFrame(activeCamera);
    }
  }, [isReady, activeCamera]);

  // Wheel handling
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (e: WheelEvent) => {
        if (mouseState.current.type === 'idle') {
            e.preventDefault();
            handleCameraUpdate(cam => zoom(cam, e.deltaY));
        }
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', handleWheel);
  }, [handleCameraUpdate]);

  // Move ensurePickingData inside component
  const ensurePickingData = useCallback((renderObject: RenderObject, component: ComponentConfig) => {
    if (!renderObject.pickingDataStale) return;
    if (!gpuRef.current) return;

    const { device, bindGroupLayout, pipelineCache } = gpuRef.current;

    // Just use the stored indices - no need to recalculate!
    const sortedIndices = renderObject.transparencyInfo?.sortedIndices;

    // Build picking data with same sorting as render
    const spec = primitiveRegistry[component.type];
    const pickingData = spec.buildPickingData(
      component,
      gpuRef.current.componentBaseId[renderObject.componentIndex],
      sortedIndices  // Use stored indices
    );

    if (pickingData) {
      const pickingOffset = Math.ceil(gpuRef.current.dynamicBuffers!.pickingOffset / 4) * 4;

      // Write picking data to buffer
      device.queue.writeBuffer(
        gpuRef.current.dynamicBuffers!.pickingBuffer,
        pickingOffset,
        pickingData.buffer,
        pickingData.byteOffset,
        pickingData.byteLength
      );

      // Set up picking pipeline and buffers
      renderObject.pickingPipeline = spec.getPickingPipeline(device, bindGroupLayout, pipelineCache);
      renderObject.pickingVertexBuffers = [
        renderObject.vertexBuffers[0],
        {
          buffer: gpuRef.current.dynamicBuffers!.pickingBuffer,
          offset: pickingOffset,
          stride: Math.ceil(pickingData.length / (renderObject.instanceCount || 1)) * 4
        }
      ];
      renderObject.pickingIndexBuffer = renderObject.indexBuffer;
      renderObject.pickingIndexCount = renderObject.indexCount;
      renderObject.pickingInstanceCount = renderObject.instanceCount;
      renderObject.pickingDataStale = false;

      // Update picking offset
      gpuRef.current.dynamicBuffers!.pickingOffset = pickingOffset + pickingData.byteLength;
    }
  }, [components]);

  return (
    <div style={{ width: '100%', border: '1px solid #ccc' }}>
        <canvas
            ref={canvasRef}
            style={style}
        />
    </div>
  );
}

// Add this helper function at the top of the file
function getSortedIndices(centers: Float32Array, cameraPosition: [number, number, number]): number[] {
  const count = centers.length / 3;
  const indices = Array.from({length: count}, (_, i) => i);

  // Calculate depths and sort back-to-front
  const depths = indices.map(i => {
    const x = centers[i*3 + 0];
    const y = centers[i*3 + 1];
    const z = centers[i*3 + 2];
    return (x - cameraPosition[0])**2 +
           (y - cameraPosition[1])**2 +
           (z - cameraPosition[2])**2;
  });

  return indices.sort((a, b) => depths[b] - depths[a]);
}

function hasTransparency(alphas: Float32Array | null, defaultAlpha: number, decorations?: Decoration[]): boolean {
    return alphas !== null ||
           defaultAlpha !== 1.0 ||
           (decorations?.some(d => d.alpha  !== undefined && d.alpha !== 1.0 && d.indexes?.length > 0) ?? false);
}
