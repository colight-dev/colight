export interface ReadyState {
  beginUpdate: (label: string) => () => void;
}

export const NOOP_READY_STATE: ReadyState = {
  beginUpdate: () => () => {},
};

export interface PipelineCacheEntry {
  pipeline: GPURenderPipeline;
  device: GPUDevice;
}

export interface PrimitiveSpec<ConfigType> {
  /**
   * The type/name of this primitive spec
   */
  type: string;

  /**
   * Input coercion function. Transforms user props to internal config format.
   * Handles aliases (center → centers), scalar expansion, etc.
   */
  coerce?: (props: Record<string, any>) => Record<string, any>;

  /**
   * Fields that should be coerced to typed arrays.
   * Used by Scene to normalize NdArrayView and regular arrays.
   */
  arrayFields?: {
    float32?: (keyof ConfigType)[];
  };

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
   * Fills geometry data for picking a single instance.
   * @param component The component containing instance data
   * @param instanceIndex Index of the instance to fill data for
   * @param out Output Float32Array to write data to
   * @param offset Offset in the output array to start writing
   * @param baseID Base ID for picking
   * @param scale Scale factor to apply to the instance
   */
  fillPickingGeometry(
    constants: ElementConstants,
    elem: ConfigType,
    i: number,
    out: Float32Array,
    offset: number,
    baseID: number,
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
   * Creates or retrieves a cached WebGPU pipeline for picking.
   * @param device The WebGPU device
   * @param bindGroupLayout Layout for uniform bindings
   * @param cache Pipeline cache to prevent duplicate creation
   */
  getPickingPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>,
  ): GPURenderPipeline;

  /**
   * Optional batch key for grouping components into separate render objects.
   * Useful when instances must not share resources (e.g., textures).
   */
  getBatchKey?: (elem: ConfigType) => string | number | undefined;

  /**
   * Creates or retrieves a cached overlay render pipeline for this primitive.
   * Overlay pipelines render in front of scene geometry (depthCompare: "always", no depth write).
   * @param device The WebGPU device
   * @param bindGroupLayout Layout for uniform bindings
   * @param cache Pipeline cache to prevent duplicate creation
   */
  getOverlayPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>,
  ): GPURenderPipeline;

  /**
   * Creates or retrieves a cached overlay picking pipeline for this primitive.
   * Overlay picking pipelines render in front of scene geometry for picking priority.
   * @param device The WebGPU device
   * @param bindGroupLayout Layout for uniform bindings
   * @param cache Pipeline cache to prevent duplicate creation
   */
  getOverlayPickingPipeline(
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    cache: Map<string, PipelineCacheEntry>,
  ): GPURenderPipeline;

  /**
   * Creates the base geometry buffers needed for this primitive type.
   * These buffers are shared across all instances of the primitive.
   */
  createGeometryResource(device: GPUDevice): GeometryResource;
}

export interface Decoration {
  indexes: number[];
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
  /**
   * Enable outline effect for these specific instances.
   * If undefined, outlineColor/outlineWidth enable the outline.
   */
  outline?: boolean;
  /** Outline color, as RGB [0-1]. Defaults to [1, 1, 1]. */
  outlineColor?: [number, number, number];
  /** Outline width in pixels. Defaults to 2. */
  outlineWidth?: number;
}

/**
 * Properties to apply when an instance is hovered.
 * These are applied automatically without needing to track hover state externally.
 */
export interface HoverProps {
  /** RGB color to apply on hover */
  color?: [number, number, number];
  /** Alpha (opacity) to apply on hover */
  alpha?: number;
  /** Scale multiplier to apply on hover */
  scale?: number;
  /**
   * Enable outline effect when hovered.
   * If undefined, outlineColor/outlineWidth enable the outline.
   */
  outline?: boolean;
  /** Outline color when hovered, as RGB [0-1]. Defaults to [1, 1, 1]. */
  outlineColor?: [number, number, number];
  /** Outline width in pixels when hovered. Defaults to 2. */
  outlineWidth?: number;
}

export interface ElementConstants {
  half_size?: number[] | Float32Array | number;
  quaternion?: number[] | Float32Array;
  size?: number;
  color?: [number, number, number] | Float32Array;
  alpha?: number;
  scale?: number;
}

/**
 * Render layer for a component.
 * - "scene": Normal rendering with depth test/write (default)
 * - "overlay": Renders in front of scene geometry, always visible
 */
export type RenderLayer = "scene" | "overlay";

export interface BaseComponentConfig {
  constants?: ElementConstants;

  /**
   * Render layer for this component.
   * - "scene": Normal rendering with depth test/write (default)
   * - "overlay": Renders in front of scene geometry (no depth test),
   *              useful for gizmos, helpers, and always-visible UI elements.
   *              Overlay components are also picked with priority over scene components.
   */
  layer?: RenderLayer;
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
   * Callback fired when drag starts on this component.
   */
  onDragStart?: (info: DragInfo) => void;

  /**
   * Callback fired during drag with updated position info.
   */
  onDrag?: (info: DragInfo) => void;

  /**
   * Callback fired when drag ends.
   */
  onDragEnd?: (info: DragInfo) => void;

  /**
   * Constraint configuration for drag operations.
   * Determines how the dragged position is computed.
   * Defaults to "surface" (drag along the hit surface).
   */
  dragConstraint?: DragConstraint;

  /**
   * Optional array of decorations to apply to specific instances.
   * Decorations can override colors, alpha, and scale for individual instances.
   */
  decorations?: Decoration[];

  /**
   * Properties to apply automatically when an instance is hovered.
   * This provides declarative hover styling without external state management.
   *
   * @example
   * ```ts
   * Ellipsoid({
   *   centers: [[0, 0, 0]],
   *   color: [1, 0, 0],
   *   hoverProps: { color: [1, 0.5, 0.5], scale: 1.2 }
   * })
   * ```
   */
  hoverProps?: HoverProps;

  /**
   * Scale factor for picking geometry.
   * Values > 1 make the component easier to click by inflating the picking hit area.
   * Useful for thin geometry like gizmo axis handles.
   *
   * @example
   * ```ts
   * LineBeams({
   *   points: [...],
   *   size: 0.02,
   *   pickingScale: 3.0  // 3x larger hit area for easier clicking
   * })
   * ```
   */
  pickingScale?: number;

  /**
   * Enable outline effect for this component's instances.
   * If undefined, outlineColor/outlineWidth enable the outline.
   * @default false
   */
  outline?: boolean;

  /**
   * Outline color, as RGB [0-1].
   * @default [1, 1, 1]
   */
  outlineColor?: [number, number, number];

  /**
   * Outline width in pixels.
   * @default 2
   */
  outlineWidth?: number;
}

export type PickEventType =
  | "hover"
  | "click"
  | "dragstart"
  | "drag"
  | "dragend";

/**
 * Simple pick event for scene-level onHover/onClick callbacks.
 * Contains the essential pick information without the full detail of PickInfo.
 */
export interface PickEvent {
  /** Type of the picked primitive */
  type: string;
  /** Optional id of the picked component */
  id?: string;
  /** Index of the picked element within its component */
  index: number;
  /** World-space position of the picked element's center */
  position: [number, number, number];
  /** Path of group names from root to this component (if in a named group) */
  groupPath?: string[];
}

/**
 * Constraint configuration for drag operations.
 * Determines how the dragged position is computed from mouse movement.
 */
export type DragConstraint =
  | {
      type: "plane";
      normal: [number, number, number];
      point?: [number, number, number];
    }
  | {
      type: "axis";
      direction: [number, number, number];
      point?: [number, number, number];
    }
  | { type: "surface" } // Use hit surface as drag plane (default)
  | { type: "screen" } // Screen-space drag (fixed depth)
  | { type: "free" }; // Camera-facing plane through start point

/**
 * Information about a drag operation, extending PickInfo with drag-specific data.
 */
export interface DragInfo extends PickInfo {
  /** Original state at drag start */
  start: {
    /** World-space hit position at drag start */
    position: [number, number, number];
    /** Center of the dragged instance at drag start */
    instanceCenter: [number, number, number];
    /** Screen coordinates at drag start */
    screen: { x: number; y: number };
  };

  /** Current drag state */
  current: {
    /** Constrained world-space position (where the ray hits) */
    position: [number, number, number];
    /** Where the instance center should move to (start.instanceCenter + delta.position) */
    instanceCenter: [number, number, number];
    /** Screen coordinates */
    screen: { x: number; y: number };
  };

  /** Computed deltas from start to current */
  delta: {
    /** World-space position delta: current.position - start.position */
    position: [number, number, number];
    /** Screen-space delta */
    screen: { x: number; y: number };
  };

  /**
   * Apply drag delta to an object property with automatic memoization.
   * On first call, captures obj[key] as the start value.
   * On all calls, sets obj[key] = startValue + delta and returns the result.
   *
   * @param obj Object containing the position to update (e.g., $state)
   * @param key Property key (e.g., 'positions.red')
   * @returns The updated position [x, y, z]
   *
   * @example
   * on_drag=js("(info) => info.applyDelta($state, 'positions.red')")
   */
  applyDelta(obj: Record<string, any>, key: string): [number, number, number];
}

export interface PickRay {
  origin: [number, number, number];
  direction: [number, number, number];
}

export interface PickHit {
  position: [number, number, number];
  normal?: [number, number, number];
  t?: number;
  distance?: number;
}

export interface PickFace {
  index: number;
  name: string;
}

export interface PickSegment {
  index: number;
  t: number;
  lineIndex?: number;
}

export interface PickCamera {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fov: number;
  near: number;
  far: number;
}

export interface PickInfo {
  event: PickEventType;
  component: { index: number; type: string };
  instanceIndex: number;
  screen: {
    x: number;
    y: number;
    dpr: number;
    rectWidth: number;
    rectHeight: number;
  };
  ray: PickRay;
  camera: PickCamera;
  hit?: PickHit;
  face?: PickFace;
  segment?: PickSegment;
  local?: { position: [number, number, number] };
  /** Path of group names from root to this component (if component is in a group) */
  groupPath?: string[];
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
  depthStencil?: {
    format: GPUTextureFormat;
    depthWriteEnabled: boolean;
    depthCompare: GPUCompareFunction;
  };
  colorWriteMask?: number; // Use number instead of GPUColorWrite
  targets?: GPUColorTargetState[];
}

export interface GeometryData {
  vertexData: Float32Array;
  indexData?: Uint16Array | Uint32Array;
}

export interface GeometryResource {
  vb: GPUBuffer;
  ib: GPUBuffer | null;
  indexCount: number;
  vertexCount: number;
  indexFormat?: GPUIndexFormat;
  geometryKey?: unknown;
}

export interface GeometryResources {
  [key: string]: GeometryResource | null;
  PointCloud: GeometryResource | null;
  Ellipsoid: GeometryResource | null;
  EllipsoidAxes: GeometryResource | null;
  Cuboid: GeometryResource | null;
  LineBeams: GeometryResource | null;
  LineSegments: GeometryResource | null;
  ImagePlane: GeometryResource | null;
}

export interface BufferInfo {
  buffer: GPUBuffer;
  offset: number;
  stride: number;
}

export interface RenderObject {
  pipeline: GPURenderPipeline;
  /** Pipeline for overlay rendering (depthCompare: "always", no depth write) */
  overlayPipeline?: GPURenderPipeline;
  geometryBuffer: GPUBuffer;
  instanceBuffer: BufferInfo;
  indexBuffer: GPUBuffer | null;
  indexCount: number;
  indexFormat?: GPUIndexFormat;
  instanceCount: number;
  vertexCount: number;
  textureBindGroup?: GPUBindGroup;

  pickingPipeline: GPURenderPipeline;
  /** Pipeline for overlay picking (depthCompare: "always", no depth write) */
  overlayPickingPipeline?: GPURenderPipeline;
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

  /** Render layer for this object ("scene" or "overlay") */
  layer: RenderLayer;
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
