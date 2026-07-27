/**
 * @module scene3d
 * @description A high-level React component for rendering 3D scenes using WebGPU.
 * This module provides a declarative interface for 3D visualization, handling camera controls,
 * picking, and efficient rendering of various 3D primitives.
 *
 * Supports three composition styles:
 * 1. JSX children: <Scene><PointCloud ... /><Ellipsoid ... /></Scene>
 * 2. Components array: <Scene components={[...]} /> (used by Python interop)
 * 3. Layers array: <Scene layers={[...]} /> (serialized from Python)
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
  LineSegmentsComponentConfig,
  ImagePlaneComponentConfig,
  BoundingBoxComponentConfig,
  PickEvent,
  PrimitiveSpec,
  ImageSource,
  pointCloudSpec,
  ellipsoidSpec,
  cuboidSpec,
  imagePlaneSpec,
  boundingBoxSpec,
  lineBeamsSpec,
  lineSegmentsSpec,
  // Props types (user-facing input types)
  PointCloudProps,
  EllipsoidProps,
  CuboidProps,
  BoundingBoxProps,
  ImagePlaneProps,
} from "./components";
import { GroupConfig, GroupRegistry } from "./groups";
import { GPUTransform } from "./gpu-transforms";
import { compileScene, RawComponent } from "./compiler";
import { CameraParams, DEFAULT_CAMERA } from "./camera3d";
import { useContainerWidth } from "../utils";
import { FPSCounter, useFPSCounter } from "./fps";
import { tw } from "../utils";
import { ReadyState, NOOP_READY_STATE, PickInfo } from "./types";
import { PrimitiveSpecMap, MeshGeometry, MeshDefinition } from "./coercion";
import { InlineMeshComponentConfig, MeshProps } from "./inlineMesh";
import {
  GridHelper,
  GridHelperProps,
  CameraFrustum,
  CameraFrustumProps,
  ImageProjection,
  ImageProjectionProps,
  ImageProjectionResult,
  CameraIntrinsics,
  CameraExtrinsics,
} from "./helpers";

// Re-export helpers for external use
export { GridHelper, CameraFrustum, ImageProjection };
export type {
  GridHelperProps,
  CameraFrustumProps,
  ImageProjectionProps,
  ImageProjectionResult,
  CameraIntrinsics,
  CameraExtrinsics,
};

// Re-export coercion types
export type { MeshGeometry, MeshDefinition, MeshProps };

// =============================================================================
// Raw Component Coercion
// =============================================================================

// =============================================================================
// Primitive Components (JSX API)
// =============================================================================

/**
 * Symbol used to identify scene3d primitive components.
 * Components with this symbol are collected by Scene for rendering.
 */
const SCENE3D_TYPE = Symbol.for("scene3d.type");

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
 * @param options - Optional visual modifications (color, alpha, scale, outline)
 * @returns {Decoration} A decoration configuration object
 */
export function deco(
  indexes: number | number[],
  options: {
    color?: [number, number, number];
    alpha?: number;
    scale?: number;
    outline?: boolean;
    outlineColor?: [number, number, number];
    outlineWidth?: number;
  } = {},
): Decoration {
  const indexArray = typeof indexes === "number" ? [indexes] : indexes;
  return { indexes: indexArray, ...options };
}

// =============================================================================
// Props types for JSX usage
// Most Props types are imported from ./components (defined in primitive files)
// =============================================================================

export type {
  PointCloudProps,
  EllipsoidProps,
  CuboidProps,
  BoundingBoxProps,
  ImagePlaneProps,
};
export type LineBeamsProps = Omit<LineBeamsComponentConfig, "type">;
export type LineSegmentsProps = Omit<LineSegmentsComponentConfig, "type">;
export type GroupProps = Omit<GroupConfig, "type" | "children"> & {
  children?: React.ReactNode;
};

export type { PickEvent, ImageSource };

// =============================================================================
// Primitive Components
//
// Each primitive can be used in two ways:
// 1. As a function: `PointCloud({centers, color})` returns a config object
// 2. As a JSX component: `<PointCloud centers={...} />` (collected by Scene)
// =============================================================================

/** PointCloud - renders points as camera-facing billboards. */
export function PointCloud(props: PointCloudProps): PointCloudComponentConfig {
  return pointCloudSpec.coerce!(props) as PointCloudComponentConfig;
}
(PointCloud as any)[SCENE3D_TYPE] = true;

/** Ellipsoid - renders spheres or ellipsoids. */
export function Ellipsoid(props: EllipsoidProps): EllipsoidComponentConfig {
  return ellipsoidSpec.coerce!(props) as EllipsoidComponentConfig;
}
(Ellipsoid as any)[SCENE3D_TYPE] = true;

/** Cuboid - renders axis-aligned or rotated boxes. */
export function Cuboid(props: CuboidProps): CuboidComponentConfig {
  return cuboidSpec.coerce!(props) as CuboidComponentConfig;
}
(Cuboid as any)[SCENE3D_TYPE] = true;

/** LineBeams - renders connected line segments as 3D beams. */
export function LineBeams(props: LineBeamsProps): LineBeamsComponentConfig {
  return { ...props, type: "LineBeams" } as LineBeamsComponentConfig;
}
(LineBeams as any)[SCENE3D_TYPE] = true;

/** LineSegments - renders independent line segments as 3D beams. */
export function LineSegments(
  props: LineSegmentsProps,
): LineSegmentsComponentConfig {
  return { ...props, type: "LineSegments" } as LineSegmentsComponentConfig;
}
(LineSegments as any)[SCENE3D_TYPE] = true;

/** Mesh - renders custom geometry using inline vertex/index data. */
export function Mesh(props: MeshProps): InlineMeshComponentConfig {
  const { center, centers, ...rest } = props;
  return {
    ...rest,
    centers: centers ?? (center ? [center] : undefined),
    type: "Mesh",
  } as InlineMeshComponentConfig;
}
(Mesh as any)[SCENE3D_TYPE] = true;

/** ImagePlane - renders a textured quad in 3D. */
export function ImagePlane(props: ImagePlaneProps): ImagePlaneComponentConfig {
  return imagePlaneSpec.coerce!(props) as ImagePlaneComponentConfig;
}
(ImagePlane as any)[SCENE3D_TYPE] = true;

// Mark helper components with SCENE3D_TYPE for JSX collection
(GridHelper as any)[SCENE3D_TYPE] = true;
(CameraFrustum as any)[SCENE3D_TYPE] = true;
(ImageProjection as any)[SCENE3D_TYPE] = true;

/** BoundingBox - renders wireframe boxes. */
export function BoundingBox(
  props: BoundingBoxProps,
): BoundingBoxComponentConfig {
  return boundingBoxSpec.coerce!(props) as BoundingBoxComponentConfig;
}
(BoundingBox as any)[SCENE3D_TYPE] = true;

/** Group - applies a transform to children. */
export function Group(props: GroupProps): GroupConfig {
  // When called from Python (via JSCall), props contains the full config.
  // When used as JSX, collectComponentsFromChildren builds the config from React children.
  // Note: children here is React.ReactNode, but GroupConfig expects an array.
  // The actual children array is built by collectComponentsFromChildren.
  return {
    type: "Group",
    children: (props.children || []) as unknown as GroupConfig["children"],
    position: props.position,
    quaternion: props.quaternion,
    scale: props.scale,
    name: props.name,
    childDefaults: props.childDefaults,
    childOverrides: props.childOverrides,
    hoverProps: props.hoverProps,
    onHover: props.onHover,
    onClick: props.onClick,
    onDragStart: props.onDragStart,
    onDrag: props.onDrag,
    onDragEnd: props.onDragEnd,
    dragConstraint: props.dragConstraint,
  };
}
(Group as any)[SCENE3D_TYPE] = true;

/**
 * Custom primitive factory - allows passing arbitrary props with a custom type.
 * Used for rendering custom primitives defined via `primitiveSpecs`.
 */
export function CustomPrimitive(props: any): ComponentConfig {
  // Pass through props. The 'type' field in props determines the primitive type.
  return props as ComponentConfig;
}
(CustomPrimitive as any)[SCENE3D_TYPE] = true;

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
  components?: (ComponentConfig | GroupConfig | InlineMeshComponentConfig)[];
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
  /** Scene-level hover callback. Called with PickInfo when hovering, null when not. */
  onHover?: (event: PickInfo | null) => void;
  /** Scene-level click callback. Called with PickInfo when an element is clicked. */
  onClick?: (event: PickInfo) => void;
  /** Optional array of controls to show. Currently supports: ['fps'] */
  controls?: string[];
  className?: string;
  style?: React.CSSProperties;
  /** Optional ready state for coordinating updates. Defaults to NOOP_READY_STATE. */
  readyState?: ReadyState;
  /** Optional map of custom primitive specifications or mesh definitions */
  primitiveSpecs?: PrimitiveSpecMap;
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

interface SceneLayersProps {
  layers: any[];
  primitiveSpecs?: PrimitiveSpecMap;
  readyState?: ReadyState;
}

/**
 * Collects component configs from React children.
 * Recursively processes children to handle fragments, arrays, and Groups.
 */
function collectComponentsFromChildren(
  children: React.ReactNode,
): (ComponentConfig | GroupConfig | InlineMeshComponentConfig)[] {
  const configs: (ComponentConfig | GroupConfig | InlineMeshComponentConfig)[] =
    [];

  React.Children.forEach(children, (child) => {
    if (!React.isValidElement(child)) return;

    const isScene3dComponent = (child.type as any)?.[SCENE3D_TYPE];
    if (!isScene3dComponent) return;

    // Group needs special handling to recursively collect children
    if ((child.type as unknown) === Group) {
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
        childDefaults: props.childDefaults,
        childOverrides: props.childOverrides,
        hoverProps: props.hoverProps,
        onHover: props.onHover,
        onClick: props.onClick,
        onDragStart: props.onDragStart,
        onDrag: props.onDrag,
        onDragEnd: props.onDragEnd,
        dragConstraint: props.dragConstraint,
      } as GroupConfig);
    } else {
      // Call the component function directly - it returns config(s)
      const componentFn = child.type as (props: any) => any;
      const result = componentFn(child.props);
      if (Array.isArray(result)) {
        configs.push(...result);
      } else {
        configs.push(result);
      }
    }
  });

  return configs;
}

function collectLayers(layers: any[]): {
  components: (ComponentConfig | GroupConfig | InlineMeshComponentConfig)[];
  sceneProps: Record<string, any>;
  primitiveSpecs?: PrimitiveSpecMap;
} {
  const components: (
    | ComponentConfig
    | GroupConfig
    | InlineMeshComponentConfig
  )[] = [];
  const sceneProps: Record<string, any> = {};
  let mergedPrimitiveSpecs: PrimitiveSpecMap | undefined;

  const mergePrimitiveSpecs = (specs?: PrimitiveSpecMap) => {
    if (!specs) return;
    if (!mergedPrimitiveSpecs) {
      mergedPrimitiveSpecs = { ...specs };
      return;
    }
    Object.assign(mergedPrimitiveSpecs, specs);
  };

  const addLayer = (layer: any) => {
    if (!layer) return;

    if (Array.isArray(layer)) {
      if (layer[1]?.layers) {
        const nestedLayers = layer[1].layers;
        for (const nestedLayer of nestedLayers) {
          addLayer(nestedLayer);
        }
      } else {
        for (const nestedLayer of layer) {
          addLayer(nestedLayer);
        }
      }

      return;
    }

    if (layer.type) {
      components.push(layer);
      return;
    }

    if (layer.constructor === Object) {
      if ("primitiveSpecs" in layer) {
        mergePrimitiveSpecs(layer.primitiveSpecs as PrimitiveSpecMap);
      }
      const { primitiveSpecs: _primitiveSpecs, ...rest } = layer;
      Object.assign(sceneProps, rest);
    }
  };

  for (const layer of layers) {
    addLayer(layer);
  }

  return { components, sceneProps, primitiveSpecs: mergedPrimitiveSpecs };
}

/**
 * Dispatches between layers-based and component-based scene composition.
 */
export function Scene(props: SceneLayersProps | SceneProps) {
  if ("layers" in props) {
    return (
      <SceneFromLayers
        layers={props.layers}
        primitiveSpecs={props.primitiveSpecs}
        readyState={props.readyState}
      />
    );
  }

  return <SceneInner {...props} />;
}

/**
 * Python interop entry point - converts layers array to Scene with components.
 *
 * This is called when Python composition like `PointCloud(...) + Ellipsoid(...) + {camera}`
 * is serialized and evaluated on the JS side.
 */
function SceneFromLayers({
  layers,
  primitiveSpecs,
  readyState,
}: SceneLayersProps) {
  // Collect layers and extract scene props
  // Note: No filtering here - the compiler in SceneInner handles helper expansion
  // and filtering, ensuring helpers like ImageProjection are processed correctly.
  const { components, mergedPrimitiveSpecs, sceneProps } = useMemo(() => {
    const {
      components,
      sceneProps,
      primitiveSpecs: layerSpecs,
    } = collectLayers(layers);

    // Merge primitive specs from layers and props
    let mergedPrimitiveSpecs: PrimitiveSpecMap | undefined;
    if (layerSpecs || primitiveSpecs) {
      mergedPrimitiveSpecs = { ...primitiveSpecs, ...layerSpecs };
    }

    return { components, mergedPrimitiveSpecs, sceneProps };
  }, [layers, primitiveSpecs]);

  return (
    <SceneInner
      components={components}
      primitiveSpecs={mergedPrimitiveSpecs}
      {...sceneProps}
      readyState={readyState}
    />
  );
}

export function SceneWithLayers(props: SceneLayersProps) {
  return <SceneFromLayers {...props} />;
}

/**
 * A React component for rendering 3D scenes.
 *
 * Supports three composition styles:
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
 *
 * **Layers Array (serialized from Python):**
 * ```tsx
 * <Scene layers={[...layers]} />
 * ```
 */
function SceneInner({
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
  className,
  style,
  controls = [],
  readyState = NOOP_READY_STATE,
  primitiveSpecs,
}: SceneProps) {
  const [containerRef, measuredWidth] = useContainerWidth(1);
  const internalCameraRef = useRef({
    ...DEFAULT_CAMERA,
    ...defaultCamera,
    ...camera,
  });

  // Memoize ready callback
  const onReady = useMemo(
    () => readyState.beginUpdate("scene3d/ready"),
    [readyState],
  );

  // Compile scene: expand helpers, coerce, flatten groups, resolve meshes
  // Uses unified compiler for consistent processing across all entry paths.
  const {
    components,
    primitiveSpecs: mergedSpecs,
    groupRegistry,
    transforms,
  } = useMemo(() => {
    // Collect raw components from children or prop
    const rawComponents = componentsProp
      ? (componentsProp as RawComponent[])
      : (collectComponentsFromChildren(children) as RawComponent[]);

    // Run through unified compilation pipeline
    return compileScene(rawComponents, primitiveSpecs);
  }, [children, componentsProp, primitiveSpecs]);

  const cameraChangeCallback = useCallback(
    (cam: CameraParams) => {
      internalCameraRef.current = cam;
      onCameraChange?.(cam);
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
          <SceneImpl
            components={components}
            transforms={transforms}
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
            readyState={readyState}
            primitiveSpecs={mergedSpecs}
            groupRegistry={groupRegistry}
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
