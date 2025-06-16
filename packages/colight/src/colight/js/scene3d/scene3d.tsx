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
  useContext,
} from "react";
import { SceneInner } from "./impl3d";
import {
  ComponentConfig,
  PointCloudComponentConfig,
  CuboidComponentConfig,
  EllipsoidComponentConfig,
  LineBeamsComponentConfig,
} from "./components";
import { CameraParams, DEFAULT_CAMERA } from "./camera3d";
import { useContainerWidth } from "../utils";
import { FPSCounter, useFPSCounter } from "./fps";
import { tw } from "../utils";
import { $StateContext } from "../context";

/**
 * Helper function to coerce specified fields to Float32Array if they exist and are arrays
 */
function coerceFloat32Fields<T extends object>(obj: T, fields: (keyof T)[]): T {
  const result = obj;
  for (const field of fields) {
    const value = obj[field];
    if (Array.isArray(value)) {
      (result[field] as any) = new Float32Array(value);
    } else if (ArrayBuffer.isView(value) && !(value instanceof Float32Array)) {
      (result[field] as any) = new Float32Array(value.buffer);
    }
  }
  return result;
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
  props: PointCloudComponentConfig,
): PointCloudComponentConfig {
  return {
    ...coerceFloat32Fields(props, ["centers", "colors", "sizes"]),
    type: "PointCloud",
  };
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
    ...coerceFloat32Fields(props, [
      "centers",
      "half_sizes",
      "quaternions",
      "colors",
      "alphas",
    ]),
    half_size,
    type: fillMode === "Solid" ? "Ellipsoid" : "EllipsoidAxes",
  };
}

/**
 * Creates a cuboid component configuration.
 * @param props - Cuboid configuration properties
 * @returns {CuboidComponentConfig} Configuration for rendering cuboids in 3D space
 */
export function Cuboid(props: CuboidComponentConfig): CuboidComponentConfig {
  const half_size =
    typeof props.half_size === "number"
      ? ([props.half_size, props.half_size, props.half_size] as [
          number,
          number,
          number,
        ])
      : props.half_size;

  return {
    ...coerceFloat32Fields(props, [
      "centers",
      "half_sizes",
      "quaternions",
      "colors",
      "alphas",
    ]),
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
  props: LineBeamsComponentConfig,
): LineBeamsComponentConfig {
  return {
    ...coerceFloat32Fields(props, ["points", "colors"]),
    type: "LineBeams",
  };
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
  /** Optional array of controls to show. Currently supports: ['fps'] */
  controls?: string[];
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

  return (
    <div
      className={tw(
        "fixed bg-white border border-gray-200 shadow-lg rounded p-1 z-[1000]",
      )}
      style={{
        top: position.y,
        left: position.x,
      }}
    >
      <div
        onClick={onToggleFps}
        className={tw(
          "px-4 py-2 cursor-pointer whitespace-nowrap hover:bg-gray-100",
        )}
      >
        {showFps ? "Hide" : "Show"} FPS Counter
      </div>
      <div
        onClick={onCopyCamera}
        className={tw(
          "px-4 py-2 cursor-pointer whitespace-nowrap border-t border-gray-100 hover:bg-gray-100",
        )}
      >
        Copy Camera Position
      </div>
    </div>
  );
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
 * @component
 * @example
 * ```tsx
 * <Scene
 *   components={[
 *     PointCloud({ centers: points, color: [1,0,0] }),
 *     Ellipsoid({ centers: centers, half_size: 0.1 })
 *   ]}
 *   width={800}
 *   height={600}
 *   onCameraChange={handleCameraChange}
 *   controls={['fps']}  // Show FPS counter
 * />
 * ```
 */
export function SceneWithLayers({ layers }: { layers: any[] }) {
  const components: any[] = [];
  const props: any = {};

  for (const layer of layers) {
    if (!layer) continue;

    if (Array.isArray(layer) && layer[0] === SceneWithLayers) {
      components.push(...layer[1].layers);
    } else if (layer.type) {
      components.push(layer);
    } else if (layer.constructor === Object) {
      Object.assign(props, layer);
    }
  }

  return <Scene components={components} {...props} />;
}

export function Scene({
  components,
  width,
  height,
  aspectRatio = 1,
  camera,
  defaultCamera,
  onCameraChange,
  className,
  style,
  controls = [],
}: SceneProps) {
  const [containerRef, measuredWidth] = useContainerWidth(1);
  const internalCameraRef = useRef({
    ...DEFAULT_CAMERA,
    ...defaultCamera,
    ...camera,
  });
  const $state: any = useContext($StateContext);
  const onReady = useMemo(() => $state.beginUpdate("scene3d/ready"), []);

  const cameraChangeCallback = useCallback(
    (camera) => {
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
    const formattedPosition = `[${currentCamera.position.map((n) => n.toFixed(6)).join(", ")}]`;
    const formattedTarget = `[${currentCamera.target.map((n) => n.toFixed(6)).join(", ")}]`;
    const formattedUp = `[${currentCamera.up.map((n) => n.toFixed(6)).join(", ")}]`;

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

  return (
    <div
      ref={containerRef as React.RefObject<HTMLDivElement | null>}
      className={`${className || ""} ${tw("font-base relative w-full")}`}
      style={{ ...style }}
      onContextMenu={handleContextMenu}
    >
      {dimensions && (
        <>
          <SceneInner
            components={components}
            containerWidth={dimensions.width}
            containerHeight={dimensions.height}
            style={dimensions.style}
            camera={camera}
            defaultCamera={defaultCamera}
            onCameraChange={cameraChangeCallback}
            onFrameRendered={updateDisplay}
            onReady={onReady}
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
