/**
 * @module scene3d
 * @description A high-level React component for rendering 3D scenes using WebGPU.
 * This module provides a declarative interface for 3D visualization, handling camera controls,
 * picking, and efficient rendering of various 3D primitives.
 *
 * Supports two composition styles:
 * 1. JSX children: <Scene><PointCloud ... /><Ellipsoid ... /></Scene>
 * 2. Components array: <Scene components={[...]} /> (used by Python interop)
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
  BoundingBoxComponentConfig,
  PickEvent,
} from "./components";
import { GroupConfig, flattenGroups, hasGroups } from "./groups";
import { CameraParams, DEFAULT_CAMERA } from "./camera3d";
import { useContainerWidth } from "../utils";
import { FPSCounter, useFPSCounter } from "./fps";
import { tw } from "../utils";
import { $StateContext } from "../context";

// =============================================================================
// Primitive Components (JSX API)
// =============================================================================

/**
 * Symbol used to identify scene3d primitive components.
 * Components with this symbol are collected by Scene for rendering.
 */
const SCENE3D_TYPE = Symbol.for("scene3d.type");

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
 * Processes props into a component config, handling type coercion and defaults.
 */
function processConfig(
  typeName: string,
  props: Record<string, any>,
): ComponentConfig {
  switch (typeName) {
    case "PointCloud":
      return {
        ...coerceFloat32Fields(props, ["centers", "colors", "sizes"]),
        type: "PointCloud",
      } as PointCloudComponentConfig;

    case "Ellipsoid": {
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
      } as EllipsoidComponentConfig;
    }

    case "Cuboid": {
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
      } as CuboidComponentConfig;
    }

    case "LineBeams":
      return {
        ...coerceFloat32Fields(props, ["points", "colors"]),
        type: "LineBeams",
      } as LineBeamsComponentConfig;

    case "BoundingBox": {
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
          "sizes",
          "alphas",
        ]),
        half_size,
        type: "BoundingBox",
      } as BoundingBoxComponentConfig;
    }

    default:
      // For unknown types, pass through with type field
      return { ...props, type: typeName } as ComponentConfig;
  }
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

// =============================================================================
// Props types for JSX usage (omit 'type' which is added automatically)
// =============================================================================

export type PointCloudProps = Omit<PointCloudComponentConfig, "type">;
export type EllipsoidProps = Omit<EllipsoidComponentConfig, "type">;
export type CuboidProps = Omit<CuboidComponentConfig, "type">;
export type LineBeamsProps = Omit<LineBeamsComponentConfig, "type">;
export type BoundingBoxProps = Omit<BoundingBoxComponentConfig, "type">;
export type GroupProps = Omit<GroupConfig, "type" | "children"> & {
  children?: React.ReactNode;
};

export type { PickEvent };

// =============================================================================
// Primitive Components
//
// Each primitive can be used in two ways:
// 1. As a function: `PointCloud({centers, color})` returns a config object
// 2. As a JSX component: `<PointCloud centers={...} />` (collected by Scene)
// =============================================================================

/** PointCloud - renders points as camera-facing billboards. */
export function PointCloud(props: PointCloudProps): PointCloudComponentConfig {
  return processConfig("PointCloud", props) as PointCloudComponentConfig;
}
(PointCloud as any)[SCENE3D_TYPE] = "PointCloud";

/** Ellipsoid - renders spheres or ellipsoids. */
export function Ellipsoid(props: EllipsoidProps): EllipsoidComponentConfig {
  return processConfig("Ellipsoid", props) as EllipsoidComponentConfig;
}
(Ellipsoid as any)[SCENE3D_TYPE] = "Ellipsoid";

/** Cuboid - renders axis-aligned or rotated boxes. */
export function Cuboid(props: CuboidProps): CuboidComponentConfig {
  return processConfig("Cuboid", props) as CuboidComponentConfig;
}
(Cuboid as any)[SCENE3D_TYPE] = "Cuboid";

/** LineBeams - renders connected line segments as 3D beams. */
export function LineBeams(props: LineBeamsProps): LineBeamsComponentConfig {
  return processConfig("LineBeams", props) as LineBeamsComponentConfig;
}
(LineBeams as any)[SCENE3D_TYPE] = "LineBeams";

/** BoundingBox - renders wireframe boxes. */
export function BoundingBox(
  props: BoundingBoxProps,
): BoundingBoxComponentConfig {
  return processConfig("BoundingBox", props) as BoundingBoxComponentConfig;
}
(BoundingBox as any)[SCENE3D_TYPE] = "BoundingBox";

/** Group - applies a transform to children. */
export function Group(_props: GroupProps): GroupConfig {
  // Note: The actual children processing happens in collectComponentsFromChildren.
  // When called as a function (not JSX), this returns a placeholder config.
  // The real Group config is built during JSX collection.
  return {
    type: "Group",
    children: [],
  } as GroupConfig;
}
(Group as any)[SCENE3D_TYPE] = "Group";

/**
 * Set of valid primitive type names.
 * Used by SceneWithLayers to identify component configs.
 */
const PRIMITIVE_TYPES = new Set([
  "PointCloud",
  "Ellipsoid",
  "EllipsoidAxes",
  "Cuboid",
  "LineBeams",
  "BoundingBox",
  "Group",
]);

// =============================================================================
// Scene Components
// =============================================================================

/**
 * Computes canvas dimensions based on container width and desired aspect ratio.
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
  /** Primitive components as JSX children */
  children?: React.ReactNode;
  /** Array of 3D component configs (alternative to children, used by Python interop) */
  components?: (ComponentConfig | GroupConfig)[];
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
  /** Scene-level hover callback. Called with PickEvent when hovering, null when not. */
  onHover?: (event: PickEvent | null) => void;
  /** Scene-level click callback. Called with PickEvent when an element is clicked. */
  onClick?: (event: PickEvent) => void;
  /** Default outline on hover for components that don't specify hoverOutline. Default: false */
  defaultHoverOutline?: boolean;
  /** Default outline color as RGB [0-1]. Default: [1, 1, 1] (white) */
  defaultOutlineColor?: [number, number, number];
  /** Default outline width in pixels. Default: 2 */
  defaultOutlineWidth?: number;
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
 * Collects component configs from React children.
 * Recursively processes children to handle fragments, arrays, and Groups.
 */
function collectComponentsFromChildren(
  children: React.ReactNode,
): (ComponentConfig | GroupConfig)[] {
  const configs: (ComponentConfig | GroupConfig)[] = [];

  React.Children.forEach(children, (child) => {
    if (!React.isValidElement(child)) return;

    const typeName = (child.type as any)?.[SCENE3D_TYPE];
    if (typeName === "Group") {
      // Process group children recursively
      const props = child.props as Record<string, any>;
      const groupChildren = props.children
        ? collectComponentsFromChildren(props.children)
        : [];
      configs.push({
        type: "Group",
        children: groupChildren,
        position: props.position,
        quaternion: props.quaternion,
        scale: props.scale,
        name: props.name,
      } as GroupConfig);
    } else if (typeName) {
      configs.push(processConfig(typeName, child.props as Record<string, any>));
    }
  });

  return configs;
}

/**
 * Python interop entry point - converts layers array to Scene with children.
 *
 * This is called when Python composition like `PointCloud(...) + Ellipsoid(...) + {camera}`
 * is serialized and evaluated on the JS side.
 */
export function SceneWithLayers({ layers }: { layers: any[] }) {
  const components: (ComponentConfig | GroupConfig)[] = [];
  const sceneProps: Record<string, any> = {};
  for (const layer of layers) {
    if (!layer) continue;

    // Handle nested SceneWithLayers (from Python Scene + Scene)
    if (Array.isArray(layer) && layer[0] === SceneWithLayers) {
      const nestedLayers = layer[1].layers;
      for (const nestedLayer of nestedLayers) {
        if (nestedLayer?.type && PRIMITIVE_TYPES.has(nestedLayer.type)) {
          components.push(nestedLayer);
        } else if (nestedLayer?.constructor === Object) {
          Object.assign(sceneProps, nestedLayer);
        }
      }
    } else if (layer.type && PRIMITIVE_TYPES.has(layer.type)) {
      components.push(layer);
    } else if (layer.constructor === Object) {
      Object.assign(sceneProps, layer);
    }
  }

  return <Scene components={components} {...sceneProps} />;
}

/**
 * A React component for rendering 3D scenes.
 *
 * Supports two composition styles:
 *
 * **JSX Children (preferred for TSX):**
 * ```tsx
 * <Scene defaultCamera={{...}}>
 *   <PointCloud centers={points} color={[1,0,0]} />
 *   <Ellipsoid centers={centers} half_size={0.1} />
 * </Scene>
 * ```
 *
 * **Components Array (used by Python interop):**
 * ```tsx
 * <Scene components={[...configs]} />
 * ```
 */
export function Scene({
  children,
  components: componentsProp,
  width,
  height,
  aspectRatio = 1,
  camera,
  defaultCamera,
  onCameraChange,
  onHover,
  onClick,
  defaultHoverOutline,
  defaultOutlineColor,
  defaultOutlineWidth,
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
  const onReady = useMemo(
    () => $state?.beginUpdate?.("scene3d/ready"),
    [$state],
  );

  // Collect components from children or use components prop
  const rawComponents = useMemo(() => {
    if (componentsProp) return componentsProp;
    return collectComponentsFromChildren(children);
  }, [children, componentsProp]);

  // Flatten groups into transformed primitives
  const components = useMemo(() => {
    if (hasGroups(rawComponents)) {
      return flattenGroups(rawComponents);
    }
    return rawComponents as ComponentConfig[];
  }, [rawComponents]);

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
            onHover={onHover}
            onClick={onClick}
            defaultHoverOutline={defaultHoverOutline}
            defaultOutlineColor={defaultOutlineColor}
            defaultOutlineWidth={defaultOutlineWidth}
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
