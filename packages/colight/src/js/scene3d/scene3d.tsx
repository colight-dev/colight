/**
 * @module scene3d
 * @description A high-level React component for rendering 3D scenes using WebGPU.
 * This module provides a declarative interface for 3D visualization, handling camera controls,
 * picking, and efficient rendering of various 3D primitives.
 *
 */

import React, {
  useMemo,
  useState,
  useCallback,
  useEffect,
  useRef,
} from "react";
import { SceneImpl } from "./impl3d";
import {
  ComponentConfig,
  PointCloudComponentConfig,
  CuboidComponentConfig,
  EllipsoidComponentConfig,
  LineBeamsComponentConfig,
  pointCloudSpec,
  ellipsoidSpec,
  cuboidSpec,
  lineBeamsSpec,
} from "./components";
import { NOOP_READY_STATE, ReadyState, PrimitiveSpec } from "./types";
import { CameraParams, DEFAULT_CAMERA } from "./camera3d";
import { PointerContext, CursorHint } from "./pointer";
import { FPSCounter, useFPSCounter } from "./fps";
import { isNdArray } from "@colight/serde";
import { useContainerWidth } from "./hooks";

/** Registry mapping component type names to their specs */
const componentSpecs: Record<string, PrimitiveSpec<any>> = {
  PointCloud: pointCloudSpec,
  Ellipsoid: ellipsoidSpec,
  EllipsoidAxes: ellipsoidSpec,
  Cuboid: cuboidSpec,
  LineBeams: lineBeamsSpec,
};

/**
 * Coerce a value to Float32Array if it's an array-like type.
 * Handles NdArrayView, regular arrays, and other TypedArrays.
 */
function coerceToFloat32(value: unknown): Float32Array | unknown {
  if (isNdArray(value)) {
    const flat = value.flat;
    return flat instanceof Float32Array
      ? flat
      : new Float32Array(flat as ArrayLike<number>);
  }
  if (Array.isArray(value)) {
    return new Float32Array(value);
  }
  if (ArrayBuffer.isView(value) && !(value instanceof Float32Array)) {
    return new Float32Array((value as Float32Array).buffer);
  }
  return value;
}

/**
 * Coerce array fields on a component based on its spec's arrayFields.
 * Mutates the component in place for efficiency.
 */
function coerceComponentArrays<T extends { type: string }>(component: T): T {
  const spec = componentSpecs[component.type];
  if (!spec?.arrayFields) return component;

  const { float32: float32Fields } = spec.arrayFields;
  if (float32Fields) {
    for (const field of float32Fields) {
      const value = (component as any)[field];
      if (value !== undefined) {
        (component as any)[field] = coerceToFloat32(value);
      }
    }
  }

  return component;
}

/**
 * @interface Decoration
 * @description Defines visual modifications that can be applied to specific instances of a primitive.
 */
interface Decoration {
  /** Array of instance indices to apply the decoration to */
  indexes: number[];
  /** Optional RGB color override */
  color?: [number, number, number];
  /** Optional alpha (opacity) override */
  alpha?: number;
  /** Optional scale multiplier override */
  scale?: number;
}

/**
 * Creates a decoration configuration for modifying the appearance of specific instances.
 * @param indexes - Single index or array of indices to apply decoration to
 * @param options - Optional visual modifications (color, alpha, scale)
 * @returns {Decoration} A decoration configuration object
 */
export function deco(
  indexes: number | number[],
  options: {
    color?: [number, number, number];
    alpha?: number;
    scale?: number;
  } = {},
): Decoration {
  const indexArray = typeof indexes === "number" ? [indexes] : indexes;
  return { indexes: indexArray, ...options };
}

/**
 * Creates a point cloud component configuration.
 * @param props - Point cloud configuration properties
 * @returns {PointCloudComponentConfig} Configuration for rendering points in 3D space
 */
export function PointCloud(
  props: Omit<PointCloudComponentConfig, "type">,
): PointCloudComponentConfig {
  return { ...props, type: "PointCloud" };
}

/**
 * Creates an ellipsoid component configuration.
 * @param props - Ellipsoid configuration properties
 * @returns {EllipsoidComponentConfig} Configuration for rendering ellipsoids in 3D space
 */
export function Ellipsoid(
  props: Omit<EllipsoidComponentConfig, "type">,
): EllipsoidComponentConfig {
  const half_size =
    typeof props.half_size === "number"
      ? ([props.half_size, props.half_size, props.half_size] as [
          number,
          number,
          number,
        ])
      : props.half_size;

  const fillMode = props.fill_mode || "Solid";

  return {
    ...props,
    half_size,
    type: fillMode === "Solid" ? "Ellipsoid" : "EllipsoidAxes",
  };
}

/**
 * Creates a cuboid component configuration.
 * @param props - Cuboid configuration properties
 * @returns {CuboidComponentConfig} Configuration for rendering cuboids in 3D space
 */
export function Cuboid(
  props: Omit<CuboidComponentConfig, "type">,
): CuboidComponentConfig {
  const half_size =
    typeof props.half_size === "number"
      ? ([props.half_size, props.half_size, props.half_size] as [
          number,
          number,
          number,
        ])
      : props.half_size;

  return {
    ...props,
    half_size,
    type: "Cuboid",
  };
}

/**
 * Creates a line beams component configuration.
 * @param props - Line beams configuration properties
 * @returns {LineBeamsComponentConfig} Configuration for rendering line beams in 3D space
 */
export function LineBeams(
  props: Omit<LineBeamsComponentConfig, "type">,
): LineBeamsComponentConfig {
  return { ...props, type: "LineBeams" };
}

/**
 * Computes canvas dimensions based on container width and desired aspect ratio.
 * @param containerWidth - Width of the container element
 * @param width - Optional explicit width override
 * @param height - Optional explicit height override
 * @param aspectRatio - Desired aspect ratio (width/height), defaults to 1
 * @returns Canvas dimensions and style configuration
 */
export function computeCanvasDimensions(
  containerWidth: number,
  width?: number,
  height?: number,
  aspectRatio = 1,
) {
  if (!containerWidth && !width) return;

  const finalWidth = width || containerWidth;
  const finalHeight = height || finalWidth / aspectRatio;

  return {
    width: finalWidth,
    height: finalHeight,
    style: {
      width: width ? `${width}px` : "100%",
      height: `${finalHeight}px`,
    },
  };
}

/**
 * @interface SceneProps
 * @description Props for the Scene component
 */
interface SceneProps {
  /** Array of 3D components to render */
  components: ComponentConfig[];
  /** Optional explicit width */
  width?: number;
  /** Optional explicit height */
  height?: number;
  /** Desired aspect ratio (width/height) */
  aspectRatio?: number;
  /** Current camera parameters (for controlled mode) */
  camera?: CameraParams;
  /** Default camera parameters (for uncontrolled mode) */
  defaultCamera?: CameraParams;
  /** Callback fired when camera parameters change */
  onCameraChange?: (camera: CameraParams) => void;
  /** Callback fired with canvas element when mounted/unmounted */
  onCanvasRef?: (canvas: HTMLCanvasElement | null) => void;
  /** Optional array of controls to show. Currently supports: ['fps'] */
  controls?: string[];
  /** Optional ready state manager for render lifecycle tracking */
  readyState?: ReadyState;
  /** Ref to receive pointer context updates (screen position, ray, pick info) */
  pointerRef?: React.MutableRefObject<PointerContext | null>;
  /** Cursor management mode: "auto" sets cursor based on hover state, "manual" lets you control it */
  cursor?: "auto" | "manual";
  /** Callback fired when cursor hint changes */
  onCursorHint?: (hint: CursorHint) => void;
  className?: string;
  style?: React.CSSProperties;
}

interface DevMenuProps {
  showFps: boolean;
  onToggleFps: () => void;
  onCopyCamera: () => void;
  position: { x: number; y: number } | null;
  onClose: () => void;
}

function DevMenu({
  showFps,
  onToggleFps,
  onCopyCamera,
  position,
  onClose,
}: DevMenuProps) {
  useEffect(() => {
    if (position) {
      document.addEventListener("click", onClose);
      return () => document.removeEventListener("click", onClose);
    }
  }, [position, onClose]);

  if (!position) return null;

  const menuStyle: React.CSSProperties = {
    position: "fixed",
    backgroundColor: "white",
    border: "1px solid #e5e7eb",
    boxShadow:
      "0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05)",
    borderRadius: "4px",
    padding: "4px",
    zIndex: 1000,
    top: position.y,
    left: position.x,
  };

  const menuItemStyle: React.CSSProperties = {
    padding: "8px 16px",
    cursor: "pointer",
    whiteSpace: "nowrap",
  };

  return (
    <div style={menuStyle}>
      <div onClick={onToggleFps} style={menuItemStyle}>
        {showFps ? "Hide" : "Show"} FPS Counter
      </div>
      <div
        onClick={onCopyCamera}
        style={{ ...menuItemStyle, borderTop: "1px solid #f3f4f6" }}
      >
        Copy Camera Position
      </div>
    </div>
  );
}

/** Props for Scene when using layers-based API */
interface SceneLayersProps {
  layers: any[];
}

/**
 * A React component for rendering 3D scenes.
 *
 * This component provides a high-level interface for 3D visualization, handling:
 * - WebGPU initialization and management
 * - Camera controls (orbit, pan, zoom)
 * - Mouse interaction and picking
 * - Efficient rendering of multiple primitive types
 *
 * Can be called with either:
 * - `layers` prop: For layer-based composition with automatic coercion
 * - `components` prop: For direct component array (SceneProps)
 *
 * @component
 * @example
 * ```tsx
 * // Using components directly
 * <Scene
 *   components={[
 *     PointCloud({ centers: points, color: [1,0,0] }),
 *     Ellipsoid({ centers: centers, half_size: 0.1 })
 *   ]}
 *   width={800}
 *   height={600}
 * />
 *
 * // Using layers (from Python/serde)
 * <Scene layers={[pointCloudLayer, ellipsoidLayer, { width: 800 }]} />
 * ```
 */
export function Scene(props: SceneLayersProps | SceneProps) {
  // Dispatch based on whether we have layers or components
  if ("layers" in props) {
    return <SceneFromLayers layers={props.layers} />;
  }
  return <SceneInner {...props} />;
}

/** Internal component for processing layers into components */
function SceneFromLayers({ layers }: SceneLayersProps) {
  const components: any[] = [];
  const props: any = {};

  for (const layer of layers) {
    if (!layer) continue;

    if (Array.isArray(layer) && layer[0] === Scene) {
      components.push(...layer[1].layers);
    } else if (layer.type) {
      components.push(layer);
    } else if (layer.constructor === Object) {
      Object.assign(props, layer);
    }
  }

  return <SceneInner components={components} {...props} />;
}

function SceneInner({
  components: rawComponents,
  width,
  height,
  aspectRatio = 1,
  camera,
  defaultCamera,
  onCameraChange,
  onCanvasRef,
  className,
  style,
  controls = [],
  readyState = NOOP_READY_STATE,
  pointerRef,
  cursor,
  onCursorHint,
}: SceneProps) {
  // Coerce array fields on all components based on their spec's arrayFields
  const components = useMemo(
    () => rawComponents.map(coerceComponentArrays),
    [rawComponents],
  );
  const [containerRef, measuredWidth] = useContainerWidth(1);
  const internalCameraRef = useRef({
    ...DEFAULT_CAMERA,
    ...defaultCamera,
    ...camera,
  });
  const onReady = useMemo(
    () => readyState.beginUpdate("scene3d/ready"),
    [readyState],
  );

  const cameraChangeCallback = useCallback(
    (camera: CameraParams) => {
      internalCameraRef.current = camera;
      onCameraChange?.(camera);
    },
    [onCameraChange],
  );

  const dimensions = useMemo(
    () => computeCanvasDimensions(measuredWidth, width, height, aspectRatio),
    [measuredWidth, width, height, aspectRatio],
  );

  const { fpsDisplayRef, updateDisplay } = useFPSCounter();
  const [showFps, setShowFps] = useState(controls.includes("fps"));
  const [menuPosition, setMenuPosition] = useState<{
    x: number;
    y: number;
  } | null>(null);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setMenuPosition({ x: e.clientX, y: e.clientY });
  }, []);

  const handleClickOutside = useCallback(() => {
    setMenuPosition(null);
  }, []);

  const toggleFps = useCallback(() => {
    setShowFps((prev) => !prev);
    setMenuPosition(null);
  }, []);

  const copyCamera = useCallback(() => {
    const currentCamera = internalCameraRef.current;

    // Format the camera position as Python-compatible string
    const formattedPosition = `[${Array.from(currentCamera.position)
      .map((n) => n.toFixed(6))
      .join(", ")}]`;
    const formattedTarget = `[${Array.from(currentCamera.target)
      .map((n) => n.toFixed(6))
      .join(", ")}]`;
    const formattedUp = `[${Array.from(currentCamera.up)
      .map((n) => n.toFixed(6))
      .join(", ")}]`;

    const pythonCode = `{
        "position": ${formattedPosition},
        "target": ${formattedTarget},
        "up": ${formattedUp},
        "fov": ${currentCamera.fov}
    }`;
    console.log(pythonCode);

    navigator.clipboard
      .writeText(pythonCode)
      .catch((err) => console.error("Failed to copy camera position", err));

    setMenuPosition(null);
  }, []);

  const containerStyle: React.CSSProperties = {
    fontSize: "16px",
    position: "relative",
    width: "100%",
    ...style,
  };

  return (
    <div
      ref={containerRef}
      className={className || undefined}
      style={containerStyle}
      onContextMenu={handleContextMenu}
    >
      {dimensions && (
        <>
          <SceneImpl
            components={components}
            containerWidth={dimensions.width}
            containerHeight={dimensions.height}
            style={dimensions.style}
            camera={camera}
            defaultCamera={defaultCamera}
            onCameraChange={cameraChangeCallback}
            onCanvasRef={onCanvasRef}
            onFrameRendered={updateDisplay}
            onReady={onReady}
            readyState={readyState}
            pointerRef={pointerRef}
            cursor={cursor}
            onCursorHint={onCursorHint}
          />
          {showFps && <FPSCounter fpsRef={fpsDisplayRef} />}
          <DevMenu
            showFps={showFps}
            onToggleFps={toggleFps}
            onCopyCamera={copyCamera}
            position={menuPosition}
            onClose={handleClickOutside}
          />
        </>
      )}
    </div>
  );
}
