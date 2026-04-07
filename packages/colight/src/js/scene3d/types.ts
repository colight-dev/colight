export interface PipelineCacheEntry {
  pipeline: GPURenderPipeline;
  device: GPUDevice;
}

export interface Scene3DGeometryOptions {
  ellipsoidStacks: number;
  ellipsoidSlices: number;
  ellipsoidAxesMajorSegments: number;
  ellipsoidAxesMinorSegments: number;
}

export const DEFAULT_SCENE3D_GEOMETRY_OPTIONS: Scene3DGeometryOptions = {
  ellipsoidStacks: 32,
  ellipsoidSlices: 48,
  ellipsoidAxesMajorSegments: 32,
  ellipsoidAxesMinorSegments: 16,
};

function normalizeSegmentCount(value: number | undefined, fallback: number) {
  if (!Number.isFinite(value)) return fallback;
  return Math.max(3, Math.floor(value!));
}

export function resolveScene3DGeometryOptions(
  options?: Partial<Scene3DGeometryOptions>,
): Scene3DGeometryOptions {
  return {
    ellipsoidStacks: normalizeSegmentCount(
      options?.ellipsoidStacks,
      DEFAULT_SCENE3D_GEOMETRY_OPTIONS.ellipsoidStacks,
    ),
    ellipsoidSlices: normalizeSegmentCount(
      options?.ellipsoidSlices,
      DEFAULT_SCENE3D_GEOMETRY_OPTIONS.ellipsoidSlices,
    ),
    ellipsoidAxesMajorSegments: normalizeSegmentCount(
      options?.ellipsoidAxesMajorSegments,
      DEFAULT_SCENE3D_GEOMETRY_OPTIONS.ellipsoidAxesMajorSegments,
    ),
    ellipsoidAxesMinorSegments: normalizeSegmentCount(
      options?.ellipsoidAxesMinorSegments,
      DEFAULT_SCENE3D_GEOMETRY_OPTIONS.ellipsoidAxesMinorSegments,
    ),
  };
}

export type PrimitiveImplementationMode = "mesh" | "impostor";

export interface PrimitiveSpec<ConfigType> {
  /**
   * The type/name of this primitive spec
   */
  type: string;

  /**
   * The concrete implementation mode selected for this spec.
   */
  implementationMode: PrimitiveImplementationMode;

  /**
   * Stable key used for caching render objects and geometry resources.
   */
  resourceKey: string;

  /**
   * Default values for the primitive's properties
   */
  defaults?: ElementConstants;

  /**
   * Returns the number of elements in this component.
   * For components with instancesPerElement > 1, this returns the element count.
   */
  getElementCount(elem: ConfigType): number;

  /**
   * Number of instances created per element. Defaults to 1 if not specified.
   * Used when a single logical element maps to multiple render instances.
   */
  instancesPerElement: number;

  /**
   * Number of floats needed per instance for render data.
   */
  floatsPerInstance: number;

  /**
   * Number of floats needed per instance for picking data.
   */
  floatsPerPicking: number;

  /**
   * Returns the centers of all instances in this component.
   * Used for transparency sorting and distance calculations.
   * @returns Float32Array or number[] containing center coordinates
   */
  getCenters(elem: ConfigType): Float32Array | number[];

  /**
   * Offset for color data in the vertex buffer
   */
  colorOffset: number;

  /**
   * Offset for alpha data in the vertex buffer
   */
  alphaOffset: number;

  /**
   * Fills geometry data for rendering a single instance.
   * @param component The component containing instance data
   * @param instanceIndex Index of the instance to fill data for
   * @param out Output Float32Array to write data to
   * @param offset Offset in the output array to start writing
   * @param scale Scale factor to apply to the instance
   */
  fillRenderGeometry(
    constants: ElementConstants,
    elem: ConfigType,
    i: number,
    out: Float32Array,
    offset: number,
  ): void;

  /**
   * Applies a scale decoration to an instance.
   * @param out Output Float32Array containing instance data
   * @param offset Offset in the output array where instance data starts
   * @param scaleFactor Scale factor to apply
   */
  applyDecorationScale(
    out: Float32Array,
    offset: number,
    scaleFactor: number,
  ): void;

  /**
   * Optional method to get the color index for an instance.
   * Used when the color index is different from the instance index.
   * @param component The component containing instance data
   * @param instanceIndex Index of the instance to get color for
   * @returns The index to use for color lookup
   */
  getColorIndexForInstance?(elem: ConfigType, i: number): number;

  /**
   * Optional method to apply decorations to an instance.
   * Used when decoration needs special handling beyond default color/alpha/scale.
   * @param out Output Float32Array containing instance data
   * @param instanceIndex Index of the instance being decorated
   * @param dec The decoration to apply
   * @param floatsPerInstance Number of floats per instance in the buffer
   */
  applyDecoration?(
    dec: Decoration,
    out: Float32Array,
    instanceIndex: number,
    floatsPerInstance: number,
  ): void;

  /**
   * Fills color data for a single instance.
   * @param constants The component constants
   * @param elem The component containing instance data
   * @param elemIndex Index of the instance
   * @param out Output Float32Array to write data to
   * @param outIndex Index in output array to write color
   */
  fillColor?(
    constants: ElementConstants,
    elem: BaseComponentConfig,
    elemIndex: number,
    out: Float32Array,
    outIndex: number,
  ): void;

  /**
   * Fills alpha data for a single instance.
   * @param constants The component constants
   * @param elem The component containing instance data
   * @param elemIndex Index of the instance
   * @param out Output Float32Array to write data to
   * @param outIndex Index in output array to write alpha
   */
  fillAlpha?(
    constants: ElementConstants,
    elem: BaseComponentConfig,
    elemIndex: number,
    out: Float32Array,
    outIndex: number,
  ): void;

  /**
   * Default WebGPU rendering configuration for this primitive type.
   * Specifies face culling and primitive topology.
   */
  renderConfig: {
    cullMode: GPUCullMode;
    topology: GPUPrimitiveTopology;
    stripIndexFormat?: GPUIndexFormat;
  };

  /**
   * Creates or retrieves a cached WebGPU render pipeline for this primitive.
   * @param device The WebGPU device
   * @param bindGroupLayout Layout for uniform bindings
   * @param cache Pipeline cache to prevent duplicate creation
   */
  getRenderPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>,
  ): GPURenderPipeline;

  /**
   * Creates the base geometry buffers needed for this primitive type.
   * These buffers are shared across all instances of the primitive.
   */
  createGeometryResource(
    device: GPUDevice,
    geometryOptions: Scene3DGeometryOptions,
  ): GeometryResource;
}

export interface PrimitiveImplementation<ConfigType> {
  mode: PrimitiveImplementationMode;
  floatsPerInstance: number;
  floatsPerPicking: number;
  colorOffset: number;
  alphaOffset: number;
  renderConfig: PrimitiveSpec<ConfigType>["renderConfig"];
  fillRenderGeometry?: PrimitiveSpec<ConfigType>["fillRenderGeometry"];
  applyDecorationScale?: PrimitiveSpec<ConfigType>["applyDecorationScale"];
  getColorIndexForInstance?: PrimitiveSpec<ConfigType>["getColorIndexForInstance"];
  applyDecoration?: PrimitiveSpec<ConfigType>["applyDecoration"];
  fillColor?: PrimitiveSpec<ConfigType>["fillColor"];
  fillAlpha?: PrimitiveSpec<ConfigType>["fillAlpha"];
  getRenderPipeline: PrimitiveSpec<ConfigType>["getRenderPipeline"];
  createGeometryResource: PrimitiveSpec<ConfigType>["createGeometryResource"];
}

export interface PrimitiveDefinition<ConfigType> {
  type: string;
  defaults?: ElementConstants;
  instancesPerElement: number;
  getElementCount: PrimitiveSpec<ConfigType>["getElementCount"];
  getCenters: PrimitiveSpec<ConfigType>["getCenters"];
  fillRenderGeometry?: PrimitiveSpec<ConfigType>["fillRenderGeometry"];
  applyDecorationScale?: PrimitiveSpec<ConfigType>["applyDecorationScale"];
  getColorIndexForInstance?: PrimitiveSpec<ConfigType>["getColorIndexForInstance"];
  applyDecoration?: PrimitiveSpec<ConfigType>["applyDecoration"];
  fillColor?: PrimitiveSpec<ConfigType>["fillColor"];
  fillAlpha?: PrimitiveSpec<ConfigType>["fillAlpha"];
  implementations: Partial<
    Record<PrimitiveImplementationMode, PrimitiveImplementation<ConfigType>>
  >;
  resolveImplementation(elem: ConfigType): PrimitiveImplementationMode;
  resolveSpec(elem: ConfigType): PrimitiveSpec<ConfigType>;
}

export interface Decoration {
  indexes: number[];
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
}

export interface ElementConstants {
  half_size?: number[] | Float32Array | number;
  quaternion?: number[] | Float32Array;
  size?: number;
  color?: [number, number, number] | Float32Array;
  alpha?: number;
  scale?: number;
}

export interface BaseComponentConfig {
  constants?: ElementConstants;
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
   * Callback fired when the mouse hovers over an instance.
   * The index parameter is the instance index, or null when hover ends.
   */
  onHover?: (index: number | null) => void;

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

export interface VertexBufferLayout {
  arrayStride: number;
  stepMode?: GPUVertexStepMode;
  attributes: {
    shaderLocation: number;
    offset: number;
    format: GPUVertexFormat;
  }[];
}

export interface PipelineConfig {
  vertexShader: string;
  fragmentShader: string;
  vertexEntryPoint: string;
  fragmentEntryPoint: string;
  bufferLayouts: VertexBufferLayout[];
  primitive?: {
    topology?: GPUPrimitiveTopology;
    cullMode?: GPUCullMode;
    stripIndexFormat?: GPUIndexFormat;
  };
  blend?: {
    color?: GPUBlendComponent;
    alpha?: GPUBlendComponent;
  };
  pickFormat?: GPUTextureFormat;
  depthStencil?: {
    format: GPUTextureFormat;
    depthWriteEnabled: boolean;
    depthCompare: GPUCompareFunction;
  };
  colorWriteMask?: number; // Use number instead of GPUColorWrite
}

export interface GeometryData {
  vertexData: Float32Array;
  indexData: Uint16Array | Uint32Array;
}

export interface GeometryResource {
  vb: GPUBuffer;
  ib: GPUBuffer;
  indexFormat: GPUIndexFormat;
  indexCount: number;
  vertexCount: number;
}

export interface GeometryResources {
  [key: string]: GeometryResource | null;
}

export interface BufferInfo {
  buffer: GPUBuffer;
  offset: number;
  stride: number;
}

export interface RenderObject {
  pipeline: GPURenderPipeline;
  geometryBuffer: GPUBuffer;
  instanceBuffer: BufferInfo;
  indexBuffer: GPUBuffer;
  indexFormat: GPUIndexFormat;
  indexCount: number;
  instanceCount: number;
  vertexCount: number;

  pickingInstanceBuffer: BufferInfo;

  componentIndex: number;
  pickingDataStale: boolean;

  // Arrays owned by this RenderObject, reallocated only when count changes
  renderData: Float32Array; // Make non-optional since all components must have render data
  pickingData: Float32Array; // Make non-optional since all components must have picking data

  totalElementCount: number;

  hasAlphaComponents: boolean;
  sortedIndices?: Uint32Array;
  distances?: Float32Array;
  sortedPositions?: Uint32Array;

  componentOffsets: ComponentOffset[];

  /** Reference to the primitive spec that created this render object */
  spec: PrimitiveSpec<any>;
}

export interface RenderObjectCache {
  [key: string]: RenderObject; // Key is componentType, value is the most recent render object
}

export interface DynamicBuffers {
  renderBuffer: GPUBuffer;
  pickingBuffer: GPUBuffer;
  renderOffset: number; // Current offset into render buffer
  pickingOffset: number; // Current offset into picking buffer
}

export interface ComponentOffset {
  componentIdx: number; // The index of the component in your overall component list.
  elementStart: number; // The first instance index in the combined buffer for this component.
  pickingStart: number;
  elementCount: number; // How many instances this component contributed.
}
