/**
 * @module helpers
 * @description Composite helper components built from primitives.
 *
 * These are convenience functions that generate common 3D visualization aids
 * like grids, camera frustums, and image projections.
 */

import { LineSegmentsComponentConfig, ImagePlaneComponentConfig, ImageSource } from "./components";

// =============================================================================
// Types
// =============================================================================

type Vec3 = [number, number, number];
type Quat = [number, number, number, number];

export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

export interface CameraExtrinsics {
  position: Vec3;
  quaternion: Quat;
}

// =============================================================================
// Math Utilities
// =============================================================================

function rotatePoint(q: Quat, v: Vec3): Vec3 {
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

// =============================================================================
// GridHelper
// =============================================================================

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
  lineWidth = 0.005,
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

// =============================================================================
// CameraFrustum
// =============================================================================

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
  lineWidth = 0.005,
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

  const transform = (point: Vec3): Vec3 => {
    const rotated = rotatePoint(quaternion, point);
    return [
      rotated[0] + position[0],
      rotated[1] + position[1],
      rotated[2] + position[2],
    ];
  };

  const nearCorners = cornersAtDepth(near).map(transform);
  const farCorners = cornersAtDepth(far).map(transform);

  const segments: Array<[number, number, number, number, number, number]> = [];
  const addEdge = (a: Vec3, b: Vec3) => {
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

// =============================================================================
// ImageProjection
// =============================================================================

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

/** ImageProjection result type - array of components */
export type ImageProjectionResult = (ImagePlaneComponentConfig | LineSegmentsComponentConfig)[];

/** Creates an ImageProjection as an array of components (ImagePlane + optional frustum). */
export function createImageProjectionGroup(props: ImageProjectionProps): ImageProjectionResult {
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
    lineWidth = 0.005,
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
    (cornersCam[0][0] + cornersCam[1][0] + cornersCam[2][0] + cornersCam[3][0]) / 4,
    (cornersCam[0][1] + cornersCam[1][1] + cornersCam[2][1] + cornersCam[3][1]) / 4,
    (cornersCam[0][2] + cornersCam[1][2] + cornersCam[2][2] + cornersCam[3][2]) / 4,
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
  

  const result: ImageProjectionResult = [plane];

  if (showFrustum) {
    // Reuse CameraFrustum for the frustum visualization
    const frustum = CameraFrustum({
      intrinsics,
      extrinsics,
      near: 0,
      far: depth,
      color: frustumColor,
      lineWidth,
      layer,
    });
    result.push(frustum);
  }

  return result;
}

/** ImageProjection - composite of ImagePlane and optional frustum edges. */
export function ImageProjection(props: ImageProjectionProps): ImageProjectionResult {
  return createImageProjectionGroup(props);
}
