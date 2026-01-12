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
  defineMesh,
  MeshComponentConfig,
  pointCloudSpec,
  ellipsoidSpec,
  ellipsoidAxesSpec,
  cuboidSpec,
  lineBeamsSpec,
  lineSegmentsSpec,
  imagePlaneSpec,
  boundingBoxSpec,
  ImageSource,
} from "./components";
import { GroupConfig, flattenGroups, hasGroups } from "./groups";
import { CameraParams, DEFAULT_CAMERA } from "./camera3d";
import { useContainerWidth } from "../utils";
import { FPSCounter, useFPSCounter } from "./fps";
import { tw } from "../utils";
import { ReadyState, NOOP_READY_STATE } from "./types";
import { isNdArray } from "@colight/serde";

// =============================================================================
// Primitive Components (JSX API)
// =============================================================================

/**
 * Symbol used to identify scene3d primitive components.
 * Components with this symbol are collected by Scene for rendering.
 */
const SCENE3D_TYPE = Symbol.for("scene3d.type");

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
    const flattened = value.flat ? value.flat() : value;
    return new Float32Array(flattened as number[]);
  }
  if (ArrayBuffer.isView(value) && !(value instanceof Float32Array)) {
    return new Float32Array(value.buffer);
  }
  return value;
}

/**
 * Coerce array fields on a component based on its spec's arrayFields.
 * Mutates the component in place for efficiency.
 */
function coerceComponentArrays<T extends { type: string }>(
  component: T,
  registry: Record<string, PrimitiveSpec<any>>,
): T {
  const spec = registry[component.type];
  const float32Fields = spec?.arrayFields?.float32;
  if (!float32Fields) return component;

  for (const field of float32Fields) {
    const value = (component as any)[field];
    if (value !== undefined) {
      (component as any)[field] = coerceToFloat32(value);
    }
  }

  return component;
}

/**
 * Processes props into a component config, handling type coercion and defaults.
 */
function processConfig(
  typeName: string,
  props: Record<string, any>,
): ComponentConfig | GroupConfig | InlineMeshComponentConfig {
  switch (typeName) {
    case "PointCloud":
      return {
        ...props,
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
        ...props,
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
        ...props,
        half_size,
        type: "Cuboid",
      } as CuboidComponentConfig;
    }

    case "LineBeams":
      return {
        ...props,
        type: "LineBeams",
      } as LineBeamsComponentConfig;

    case "LineSegments":
      return {
        ...props,
        type: "LineSegments",
      } as LineSegmentsComponentConfig;

    case "ImagePlane": {
      const {
        position,
        quaternion,
        width,
        height,
        opacity,
        centers,
        quaternions,
        sizes,
        size,
        ...rest
      } = props;
      const resolvedCenters =
        centers ??
        (position ? [position] : [[0, 0, 0]]);
      const resolvedQuaternions =
        quaternions ?? (quaternion ? [quaternion] : undefined);
      const resolvedSize =
        size ??
        (width !== undefined || height !== undefined
          ? [width ?? 1, height ?? 1]
          : undefined);
      const resolvedAlpha =
        rest.alpha !== undefined ? rest.alpha : opacity;

      return {
        ...rest,
        centers: resolvedCenters,
        quaternions: resolvedQuaternions,
        sizes,
        size: resolvedSize,
        alpha: resolvedAlpha,
        type: "ImagePlane",
      } as ImagePlaneComponentConfig;
    }

    case "Mesh":
      return {
        ...props,
        type: "Mesh",
      } as InlineMeshComponentConfig;

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
        ...props,
        half_size,
        type: "BoundingBox",
      } as BoundingBoxComponentConfig;
    }

    case "ImageProjection":
      return createImageProjectionGroup(props);

    case "CustomPrimitive":
      return props as ComponentConfig;

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
export type LineSegmentsProps = Omit<LineSegmentsComponentConfig, "type">;
export type ImagePlaneProps = Omit<
  ImagePlaneComponentConfig,
  "type" | "centers" | "quaternions" | "sizes"
> & {
  centers?: ArrayLike<number> | ArrayBufferView;
  position?: [number, number, number];
  quaternions?: ArrayLike<number> | ArrayBufferView;
  quaternion?: [number, number, number, number];
  sizes?: ArrayLike<number> | ArrayBufferView;
  size?: number | [number, number];
  width?: number;
  height?: number;
  opacity?: number;
};
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

/** LineSegments - renders independent line segments as 3D beams. */
export function LineSegments(
  props: LineSegmentsProps,
): LineSegmentsComponentConfig {
  return processConfig("LineSegments", props) as LineSegmentsComponentConfig;
}
(LineSegments as any)[SCENE3D_TYPE] = "LineSegments";

/** Mesh - renders custom geometry using inline vertex/index data. */
export function Mesh(props: MeshProps): InlineMeshComponentConfig {
  return processConfig("Mesh", props) as InlineMeshComponentConfig;
}
(Mesh as any)[SCENE3D_TYPE] = "Mesh";

/** ImagePlane - renders a textured quad in 3D. */
export function ImagePlane(props: ImagePlaneProps): ImagePlaneComponentConfig {
  return processConfig("ImagePlane", props) as ImagePlaneComponentConfig;
}
(ImagePlane as any)[SCENE3D_TYPE] = "ImagePlane";

export interface GridHelperProps {
  size?: number;
  divisions?: number;
  color?: [number, number, number];
  centerColor?: [number, number, number];
  lineWidth?: number;
  layer?: "scene" | "overlay";
}

/** GridHelper - renders a simple XZ grid using LineSegments. */
export function GridHelper({
  size = 10,
  divisions = 10,
  color = [0.5, 0.5, 0.5],
  centerColor = [0.7, 0.7, 0.7],
  lineWidth = 0.02,
  layer,
}: GridHelperProps = {}): LineSegmentsComponentConfig {
  const half = size / 2;
  const step = size / divisions;
  const lineCount = divisions + 1;
  const segmentCount = lineCount * 2;

  const starts = new Float32Array(segmentCount * 3);
  const ends = new Float32Array(segmentCount * 3);
  const colors = new Float32Array(segmentCount * 3);

  let segIndex = 0;
  for (let i = 0; i < lineCount; i++) {
    const pos = -half + i * step;
    const isCenter = Math.abs(pos) < 1e-6;
    const lineColor = isCenter ? centerColor : color;

    // Lines parallel to X (vary Z)
    starts[segIndex * 3 + 0] = -half;
    starts[segIndex * 3 + 1] = 0;
    starts[segIndex * 3 + 2] = pos;
    ends[segIndex * 3 + 0] = half;
    ends[segIndex * 3 + 1] = 0;
    ends[segIndex * 3 + 2] = pos;
    colors.set(lineColor, segIndex * 3);
    segIndex++;

    // Lines parallel to Z (vary X)
    starts[segIndex * 3 + 0] = pos;
    starts[segIndex * 3 + 1] = 0;
    starts[segIndex * 3 + 2] = -half;
    ends[segIndex * 3 + 0] = pos;
    ends[segIndex * 3 + 1] = 0;
    ends[segIndex * 3 + 2] = half;
    colors.set(lineColor, segIndex * 3);
    segIndex++;
  }

  const config: LineSegmentsComponentConfig = {
    type: "LineSegments",
    starts,
    ends,
    colors,
    size: lineWidth,
  };
  if (layer) config.layer = layer;

  return config;
}

export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

export interface CameraExtrinsics {
  position: [number, number, number];
  quaternion: [number, number, number, number];
}

type Vec3 = [number, number, number];

function rotatePoint(q: [number, number, number, number], v: Vec3): Vec3 {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;
  const ix = qw * vx + qy * vz - qz * vy;
  const iy = qw * vy + qz * vx - qx * vz;
  const iz = qw * vz + qx * vy - qy * vx;
  const iw = -qx * vx - qy * vy - qz * vz;
  return [
    ix * qw + iw * -qx + iy * -qz - iz * -qy,
    iy * qw + iw * -qy + iz * -qx - ix * -qz,
    iz * qw + iw * -qz + ix * -qy - iy * -qx,
  ];
}

function projectCorner(
  intrinsics: CameraIntrinsics,
  u: number,
  v: number,
  depth: number,
): Vec3 {
  const { fx, fy, cx, cy } = intrinsics;
  const x = ((u - cx) / fx) * depth;
  const y = ((v - cy) / fy) * depth;
  return [x, y, depth];
}

export interface CameraFrustumProps {
  intrinsics: CameraIntrinsics;
  extrinsics: CameraExtrinsics;
  near?: number;
  far?: number;
  color?: [number, number, number];
  lineWidth?: number;
  layer?: "scene" | "overlay";
}

/** CameraFrustum - renders frustum edges based on intrinsics/extrinsics. */
export function CameraFrustum({
  intrinsics,
  extrinsics,
  near = 0.1,
  far = 1.0,
  color = [1, 0.8, 0.2],
  lineWidth = 0.02,
  layer,
}: CameraFrustumProps): LineSegmentsComponentConfig {
  const { width, height } = intrinsics;
  const { position, quaternion } = extrinsics;

  const cornersAtDepth = (depth: number) => [
    projectCorner(intrinsics, 0, 0, depth),
    projectCorner(intrinsics, width, 0, depth),
    projectCorner(intrinsics, width, height, depth),
    projectCorner(intrinsics, 0, height, depth),
  ];

  const transform = (point: [number, number, number]) => {
    const rotated = rotatePoint(quaternion, point);
    return [
      rotated[0] + position[0],
      rotated[1] + position[1],
      rotated[2] + position[2],
    ] as [number, number, number];
  };

  const nearCorners = cornersAtDepth(near).map(transform);
  const farCorners = cornersAtDepth(far).map(transform);

  const segments: Array<[number, number, number, number, number, number]> = [];
  const addEdge = (a: [number, number, number], b: [number, number, number]) => {
    segments.push([a[0], a[1], a[2], b[0], b[1], b[2]]);
  };

  for (let i = 0; i < 4; i++) {
    addEdge(nearCorners[i], nearCorners[(i + 1) % 4]);
    addEdge(farCorners[i], farCorners[(i + 1) % 4]);
    addEdge(nearCorners[i], farCorners[i]);
  }

  const starts = new Float32Array(segments.length * 3);
  const ends = new Float32Array(segments.length * 3);
  const colors = new Float32Array(segments.length * 3);

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    starts.set(seg.slice(0, 3), i * 3);
    ends.set(seg.slice(3, 6), i * 3);
    colors.set(color, i * 3);
  }

  const config: LineSegmentsComponentConfig = {
    type: "LineSegments",
    starts,
    ends,
    colors,
    size: lineWidth,
  };
  if (layer) config.layer = layer;
  return config;
}

export interface ImageProjectionProps {
  image: ImageSource;
  imageKey?: string | number;
  intrinsics: CameraIntrinsics;
  extrinsics: CameraExtrinsics;
  depth?: number;
  opacity?: number;
  color?: [number, number, number];
  showFrustum?: boolean;
  frustumColor?: [number, number, number];
  lineWidth?: number;
  layer?: "scene" | "overlay";
}

function createImageProjectionGroup(props: ImageProjectionProps): GroupConfig {
  const {
    image,
    imageKey,
    intrinsics,
    extrinsics,
    depth = 1.0,
    opacity,
    color = [1, 1, 1],
    showFrustum = false,
    frustumColor = [1, 0.8, 0.2],
    lineWidth = 0.02,
    layer,
  } = props;

  const { width, height } = intrinsics;
  const { position, quaternion } = extrinsics;

  const cornersCam: Vec3[] = [
    projectCorner(intrinsics, 0, 0, depth),
    projectCorner(intrinsics, width, 0, depth),
    projectCorner(intrinsics, width, height, depth),
    projectCorner(intrinsics, 0, height, depth),
  ];

  const centerCam: Vec3 = [
    (cornersCam[0][0] + cornersCam[1][0] + cornersCam[2][0] + cornersCam[3][0]) /
      4,
    (cornersCam[0][1] + cornersCam[1][1] + cornersCam[2][1] + cornersCam[3][1]) /
      4,
    (cornersCam[0][2] + cornersCam[1][2] + cornersCam[2][2] + cornersCam[3][2]) /
      4,
  ];

  const widthWorld = Math.hypot(
    cornersCam[1][0] - cornersCam[0][0],
    cornersCam[1][1] - cornersCam[0][1],
    cornersCam[1][2] - cornersCam[0][2],
  );
  const heightWorld = Math.hypot(
    cornersCam[3][0] - cornersCam[0][0],
    cornersCam[3][1] - cornersCam[0][1],
    cornersCam[3][2] - cornersCam[0][2],
  );

  const centerWorld = rotatePoint(quaternion, centerCam);
  centerWorld[0] += position[0];
  centerWorld[1] += position[1];
  centerWorld[2] += position[2];

  const plane: ImagePlaneComponentConfig = {
    type: "ImagePlane",
    image,
    imageKey,
    centers: new Float32Array(centerWorld),
    quaternions: new Float32Array(quaternion),
    size: [widthWorld, heightWorld],
    color,
    alpha: opacity,
  };
  if (layer) plane.layer = layer;

  const children: (ComponentConfig | GroupConfig)[] = [plane];

  if (showFrustum) {
    const cornersWorld = cornersCam.map((corner) => {
      const rotated = rotatePoint(quaternion, corner);
      return [
        rotated[0] + position[0],
        rotated[1] + position[1],
        rotated[2] + position[2],
      ] as Vec3;
    });

    const segmentCount = 8;
    const starts = new Float32Array(segmentCount * 3);
    const ends = new Float32Array(segmentCount * 3);
    const colors = new Float32Array(segmentCount * 3);

    let segIndex = 0;
    for (let i = 0; i < 4; i++) {
      const a = cornersWorld[i];
      const b = cornersWorld[(i + 1) % 4];
      starts.set(a, segIndex * 3);
      ends.set(b, segIndex * 3);
      colors.set(frustumColor, segIndex * 3);
      segIndex++;
    }

    for (let i = 0; i < 4; i++) {
      starts.set(position, segIndex * 3);
      ends.set(cornersWorld[i], segIndex * 3);
      colors.set(frustumColor, segIndex * 3);
      segIndex++;
    }

    const frustum: LineSegmentsComponentConfig = {
      type: "LineSegments",
      starts,
      ends,
      colors,
      size: lineWidth,
    };
    if (layer) frustum.layer = layer;
    children.push(frustum);
  }

  return {
    type: "Group",
    children,
  };
}

/** ImageProjection - composite of ImagePlane and optional frustum edges. */
export function ImageProjection(props: ImageProjectionProps): GroupConfig {
  return createImageProjectionGroup(props);
}
(ImageProjection as any)[SCENE3D_TYPE] = "ImageProjection";

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
 * Used by layers handling to identify component configs.
 */
const PRIMITIVE_TYPES = new Set([
  "PointCloud",
  "Ellipsoid",
  "EllipsoidAxes",
  "Cuboid",
  "LineBeams",
  "LineSegments",
  "ImagePlane",
  "Mesh",
  "BoundingBox",
  "Group",
]);

const defaultPrimitiveRegistry: Record<string, PrimitiveSpec<any>> = {
  PointCloud: pointCloudSpec,
  Ellipsoid: ellipsoidSpec,
  EllipsoidAxes: ellipsoidAxesSpec,
  Cuboid: cuboidSpec,
  LineBeams: lineBeamsSpec,
  LineSegments: lineSegmentsSpec,
  ImagePlane: imagePlaneSpec,
  BoundingBox: boundingBoxSpec,
};

/**
 * Custom primitive factory - allows passing arbitrary props with a custom type.
 * Used for rendering custom primitives defined via `primitiveSpecs`.
 */
export function CustomPrimitive(props: any): ComponentConfig {
  // Pass through props. The 'type' field in props determines the primitive type.
  return props as ComponentConfig;
}
(CustomPrimitive as any)[SCENE3D_TYPE] = "CustomPrimitive";

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

export interface MeshGeometry {
  vertexData: Float32Array | number[] | ArrayBufferView;
  indexData?: Uint16Array | Uint32Array | number[] | ArrayBufferView;
}

export interface MeshDefinition extends MeshGeometry {
  shading?: "lit" | "unlit";
  cullMode?: GPUCullMode;
}

export type MeshProps = Omit<MeshComponentConfig, "type"> & {
  geometry: MeshGeometry;
  geometryKey?: string | number;
  shading?: "lit" | "unlit";
  cullMode?: GPUCullMode;
};

type InlineMeshComponentConfig = MeshProps & { type: "Mesh" };

type PrimitiveSpecInput = PrimitiveSpec<any> | MeshDefinition;
type PrimitiveSpecMap = Record<string, PrimitiveSpecInput>;

function isMeshDefinition(value: PrimitiveSpecInput): value is MeshDefinition {
  return (
    typeof value === "object" &&
    value !== null &&
    "vertexData" in value
  );
}

function isPrimitiveSpec(value: unknown): value is PrimitiveSpec<any> {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as PrimitiveSpec<any>).createGeometryResource === "function"
  );
}

function coerceVertexData(
  value: Float32Array | number[] | ArrayBufferView,
): Float32Array {
  if (isNdArray(value)) {
    const flat = value.flat;
    return flat instanceof Float32Array
      ? flat
      : new Float32Array(flat as ArrayLike<number>);
  }
  if (value instanceof Float32Array) return value;
  if (Array.isArray(value)) return new Float32Array(value);
  const asArray = Array.from(value as ArrayLike<number>);
  return new Float32Array(asArray);
}

function coerceIndexData(
  value?: Uint16Array | Uint32Array | number[] | ArrayBufferView,
): Uint16Array | Uint32Array | undefined {
  if (!value) return undefined;
  if (isNdArray(value)) {
    const flat = value.flat;
    value = Array.from(flat as ArrayLike<number>);
  }
  if (value instanceof Uint16Array || value instanceof Uint32Array) {
    return value;
  }
  if (Array.isArray(value)) {
    let max = 0;
    for (const idx of value) {
      if (idx > max) max = idx;
    }
    return max > 65535 ? new Uint32Array(value) : new Uint16Array(value);
  }
  const asArray = Array.from(value as ArrayLike<number>);
  let max = 0;
  for (const idx of asArray) {
    if (idx > max) max = idx;
  }
  return max > 65535 ? new Uint32Array(asArray) : new Uint16Array(asArray);
}

function coerceMeshGeometry(geometry: MeshGeometry): MeshGeometry {
  return {
    vertexData: coerceVertexData(geometry.vertexData),
    indexData: coerceIndexData(geometry.indexData),
  };
}

function normalizePrimitiveSpecs(
  specs?: PrimitiveSpecMap,
): Record<string, PrimitiveSpec<any>> | undefined {
  if (!specs) return undefined;
  const normalized: Record<string, PrimitiveSpec<any>> = {};
  for (const [name, spec] of Object.entries(specs)) {
    if (isMeshDefinition(spec)) {
      const geometry = coerceMeshGeometry(spec);
      normalized[name] = defineMesh(
        name,
        {
          vertexData: geometry.vertexData as Float32Array,
          indexData: geometry.indexData as Uint16Array | Uint32Array | undefined,
        },
        {
          shading: spec.shading,
          cullMode: spec.cullMode,
        },
      );
    } else if (isPrimitiveSpec(spec)) {
      normalized[name] = spec;
    } else if (typeof spec === "object" && spec !== null) {
      const nestedEntries = Object.entries(
        spec as Record<string, PrimitiveSpecInput>,
      );
      const hasNested = nestedEntries.some(
        ([, nested]) => isMeshDefinition(nested) || isPrimitiveSpec(nested),
      );
      if (hasNested) {
        console.warn(
          "scene3d: flattening nested primitiveSpecs entry",
          name,
        );
        for (const [nestedName, nestedSpec] of nestedEntries) {
          if (isMeshDefinition(nestedSpec)) {
            const geometry = coerceMeshGeometry(nestedSpec);
            normalized[nestedName] = defineMesh(
              nestedName,
              {
                vertexData: geometry.vertexData as Float32Array,
                indexData:
                  geometry.indexData as Uint16Array | Uint32Array | undefined,
              },
              {
                shading: nestedSpec.shading,
                cullMode: nestedSpec.cullMode,
              },
            );
          } else if (isPrimitiveSpec(nestedSpec)) {
            normalized[nestedName] = nestedSpec;
          }
        }
      } else {
        console.warn(
          "scene3d: ignoring invalid primitiveSpecs entry",
          name,
          spec,
        );
      }
    } else {
      console.warn(
        "scene3d: ignoring invalid primitiveSpecs entry",
        name,
        spec,
      );
    }
  }
  return normalized;
}

interface InlineMeshCacheEntry {
  typeName: string;
  spec: PrimitiveSpec<any>;
  geometryRef: MeshGeometry;
  geometryKey: string | number | MeshGeometry;
  shading: "lit" | "unlit";
  cullMode: GPUCullMode;
}

const inlineMeshCache = new WeakMap<
  MeshGeometry,
  Map<string, InlineMeshCacheEntry>
>();
const inlineMeshKeyCache = new Map<
  string | number,
  Map<string, InlineMeshCacheEntry>
>();
let inlineMeshId = 0;

function getInlineMeshEntry(
  geometry: MeshGeometry,
  options: {
    geometryKey?: string | number;
    shading?: "lit" | "unlit";
    cullMode?: GPUCullMode;
  },
): InlineMeshCacheEntry {
  const shading = options.shading ?? "lit";
  const cullMode = options.cullMode ?? "back";
  const variantKey = `${shading}|${cullMode}`;
  const explicitKey = options.geometryKey;
  const cache =
    explicitKey !== undefined ? inlineMeshKeyCache : inlineMeshCache;
  const cacheKey = explicitKey !== undefined ? explicitKey : geometry;

  let variants = cache.get(cacheKey as any);
  if (!variants) {
    variants = new Map();
    cache.set(cacheKey as any, variants);
  }

  let entry = variants.get(variantKey);
  const normalizedGeometry = coerceMeshGeometry(geometry);
  const geometryKey = explicitKey !== undefined ? explicitKey : geometry;

  if (!entry) {
    const typeName = `__InlineMesh_${inlineMeshId++}`;
    const spec = defineMesh(
      typeName,
      {
        vertexData: normalizedGeometry.vertexData as Float32Array,
        indexData:
          normalizedGeometry.indexData as Uint16Array | Uint32Array | undefined,
      },
      { shading, cullMode },
    );
    (spec as any).geometryKey = geometryKey;
    entry = {
      typeName,
      spec,
      geometryRef: geometry,
      geometryKey,
      shading,
      cullMode,
    };
    variants.set(variantKey, entry);
    return entry;
  }

  if (entry.geometryRef !== geometry || entry.geometryKey !== geometryKey) {
    entry.spec = defineMesh(
      entry.typeName,
      {
        vertexData: normalizedGeometry.vertexData as Float32Array,
        indexData:
          normalizedGeometry.indexData as Uint16Array | Uint32Array | undefined,
      },
      { shading, cullMode },
    );
    (entry.spec as any).geometryKey = geometryKey;
    entry.geometryRef = geometry;
    entry.geometryKey = geometryKey;
    entry.shading = shading;
    entry.cullMode = cullMode;
  }

  return entry;
}

function resolveInlineMeshes(
  components: (ComponentConfig | InlineMeshComponentConfig)[],
): {
  components: ComponentConfig[];
  inlineSpecs?: Record<string, PrimitiveSpec<any>>;
} {
  let inlineSpecs: Record<string, PrimitiveSpec<any>> | undefined;
  const resolved = components.map((component) => {
    if (component.type !== "Mesh") return component;
    const meshComponent = component as InlineMeshComponentConfig;
    const entry = getInlineMeshEntry(meshComponent.geometry, {
      geometryKey: meshComponent.geometryKey,
      shading: meshComponent.shading,
      cullMode: meshComponent.cullMode,
    });

    if (!inlineSpecs) inlineSpecs = {};
    inlineSpecs[entry.typeName] = entry.spec;

    const { geometry, geometryKey, shading, cullMode, ...rest } = meshComponent;
    return {
      ...rest,
      type: entry.typeName,
    } as ComponentConfig;
  });

  return { components: resolved, inlineSpecs };
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
  /** Scene-level hover callback. Called with PickEvent when hovering, null when not. */
  onHover?: (event: PickEvent | null) => void;
  /** Scene-level click callback. Called with PickEvent when an element is clicked. */
  onClick?: (event: PickEvent) => void;
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

function collectLayers(layers: any[]): {
  components: (ComponentConfig | GroupConfig | InlineMeshComponentConfig)[];
  sceneProps: Record<string, any>;
  primitiveSpecs?: PrimitiveSpecMap;
} {
  const components: (ComponentConfig | GroupConfig | InlineMeshComponentConfig)[] =
    [];
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

    if (Array.isArray(layer) && layer[1]?.layers) {
      const nestedLayers = layer[1].layers;
      for (const nestedLayer of nestedLayers) {
        addLayer(nestedLayer);
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
  const { components: rawComponents, sceneProps, primitiveSpecs: layerSpecs } =
    useMemo(() => collectLayers(layers), [layers]);
  const mergedPrimitiveSpecs = useMemo(() => {
    if (!layerSpecs) return primitiveSpecs;
    if (!primitiveSpecs) return layerSpecs;
    return { ...primitiveSpecs, ...layerSpecs };
  }, [layerSpecs, primitiveSpecs]);
  const components = useMemo(() => {
    if (!mergedPrimitiveSpecs) {
      return rawComponents.filter((component) =>
        PRIMITIVE_TYPES.has(component.type),
      );
    }

    return rawComponents.filter(
      (component) =>
        PRIMITIVE_TYPES.has(component.type) ||
        component.type in mergedPrimitiveSpecs,
    );
  }, [rawComponents, mergedPrimitiveSpecs]);

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
  const onReady = useMemo(
    () => readyState.beginUpdate("scene3d/ready"),
    [readyState],
  );

  const normalizedSpecs = useMemo(
    () => normalizePrimitiveSpecs(primitiveSpecs),
    [primitiveSpecs],
  );

  // Collect components from children or use components prop
  const rawComponents = useMemo(() => {
    if (componentsProp) return componentsProp;
    return collectComponentsFromChildren(children);
  }, [children, componentsProp]);

  const flattenedComponents = useMemo(() => {
    const flattened = hasGroups(rawComponents)
      ? flattenGroups(rawComponents as (ComponentConfig | GroupConfig)[])
      : (rawComponents as (ComponentConfig | InlineMeshComponentConfig)[]);
    return flattened;
  }, [rawComponents]);

  const { components: resolvedComponents, inlineSpecs } = useMemo(
    () => resolveInlineMeshes(flattenedComponents),
    [flattenedComponents],
  );

  const mergedSpecs = useMemo(() => {
    if (!inlineSpecs) return normalizedSpecs;
    if (!normalizedSpecs) return inlineSpecs;
    return { ...normalizedSpecs, ...inlineSpecs };
  }, [normalizedSpecs, inlineSpecs]);

  const primitiveRegistry = useMemo(() => {
    if (!mergedSpecs) return defaultPrimitiveRegistry;
    return { ...defaultPrimitiveRegistry, ...mergedSpecs };
  }, [mergedSpecs]);

  // Coerce arrays by spec after inline meshes are resolved
  const components = useMemo(
    () =>
      resolvedComponents.map((component) =>
        coerceComponentArrays(component, primitiveRegistry),
      ),
    [resolvedComponents, primitiveRegistry],
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
