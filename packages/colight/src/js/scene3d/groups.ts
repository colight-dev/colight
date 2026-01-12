/**
 * @module groups
 * @description Hierarchical transform groups for Scene3D.
 *
 * Groups allow composing multiple components with a shared transform.
 * At render time, groups are flattened into transformed primitives.
 */

import { Vec3, add, scale as scaleVec3 } from "./vec3";
import { ComponentConfig, BaseComponentConfig } from "./components";

// =============================================================================
// Types
// =============================================================================

/** Quaternion type: [x, y, z, w] */
export type Quat = [number, number, number, number];

/** Identity quaternion */
export const IDENTITY_QUAT: Quat = [0, 0, 0, 1];

/** Identity position */
export const IDENTITY_POS: Vec3 = [0, 0, 0];

/**
 * Configuration for a Group component.
 * Groups apply a transform to all their children.
 */
export interface GroupConfig {
  type: "Group";
  /** Child components (can include nested groups) */
  children: (ComponentConfig | GroupConfig)[];
  /** Position offset in parent space */
  position?: Vec3;
  /** Rotation as quaternion [x, y, z, w] */
  quaternion?: Quat;
  /** Scale factor (uniform or per-axis) */
  scale?: number | Vec3;
  /** Optional name for identifying this group in pick info */
  name?: string;
}

/**
 * A transform combining position, rotation, and scale.
 */
export interface Transform {
  position: Vec3;
  quaternion: Quat;
  scale: Vec3;
}

/**
 * A flattened component with group ancestry information.
 * This extends the original component config with optional metadata.
 */
export type FlattenedComponent = ComponentConfig & {
  /** Path of group names from root to this component */
  _groupPath?: string[];
  /** Original component index before flattening (for debugging) */
  _originalIndex?: number;
};

// =============================================================================
// Quaternion Math
// =============================================================================

/**
 * Multiply two quaternions: result = a * b
 * This composes rotations (b applied first, then a).
 */
export function quatMultiply(a: Quat, b: Quat): Quat {
  const [ax, ay, az, aw] = a;
  const [bx, by, bz, bw] = b;
  return [
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz,
  ];
}

/**
 * Rotate a vector by a quaternion.
 */
export function quatRotate(q: Quat, v: Vec3): Vec3 {
  const [qx, qy, qz, qw] = q;
  const [vx, vy, vz] = v;

  // q * v * q^-1 optimized
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

/**
 * Create a quaternion from axis-angle representation.
 */
export function quatFromAxisAngle(axis: Vec3, angle: number): Quat {
  const halfAngle = angle / 2;
  const s = Math.sin(halfAngle);
  const len = Math.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2);
  if (len < 1e-10) return IDENTITY_QUAT;
  return [
    (axis[0] / len) * s,
    (axis[1] / len) * s,
    (axis[2] / len) * s,
    Math.cos(halfAngle),
  ];
}

/**
 * Normalize a quaternion.
 */
export function quatNormalize(q: Quat): Quat {
  const len = Math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2);
  if (len < 1e-10) return IDENTITY_QUAT;
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

// =============================================================================
// Transform Operations
// =============================================================================

/**
 * Create an identity transform.
 */
export function identityTransform(): Transform {
  return {
    position: [0, 0, 0],
    quaternion: [0, 0, 0, 1],
    scale: [1, 1, 1],
  };
}

/**
 * Compose two transforms: result = parent * child
 * The child transform is applied first, then the parent.
 */
export function composeTransforms(parent: Transform, child: Transform): Transform {
  // Scale the child position by parent scale
  const scaledChildPos: Vec3 = [
    child.position[0] * parent.scale[0],
    child.position[1] * parent.scale[1],
    child.position[2] * parent.scale[2],
  ];

  // Rotate the scaled child position by parent rotation
  const rotatedChildPos = quatRotate(parent.quaternion, scaledChildPos);

  // Add parent position
  const position = add(parent.position, rotatedChildPos);

  // Compose rotations
  const quaternion = quatMultiply(parent.quaternion, child.quaternion);

  // Multiply scales
  const scale: Vec3 = [
    parent.scale[0] * child.scale[0],
    parent.scale[1] * child.scale[1],
    parent.scale[2] * child.scale[2],
  ];

  return { position, quaternion, scale };
}

/**
 * Extract transform from a GroupConfig.
 */
function getGroupTransform(group: GroupConfig): Transform {
  const position = group.position ?? [0, 0, 0];
  const quaternion = group.quaternion ?? [0, 0, 0, 1];
  const rawScale = group.scale ?? 1;
  const scale: Vec3 =
    typeof rawScale === "number"
      ? [rawScale, rawScale, rawScale]
      : rawScale;

  return { position, quaternion, scale };
}

// =============================================================================
// Component Transform Application
// =============================================================================

/**
 * Check if a config is a Group.
 */
export function isGroup(config: ComponentConfig | GroupConfig): config is GroupConfig {
  return config.type === "Group";
}

/**
 * Apply a transform to a single component's data.
 * Returns a new component config with transformed positions and composed orientations.
 */
function applyTransformToComponent(
  component: ComponentConfig,
  transform: Transform,
  groupPath: string[],
): FlattenedComponent {
  const isImagePlane = component.type === "ImagePlane";
  const result: FlattenedComponent = {
    ...component,
    _groupPath: groupPath.length > 0 ? groupPath : undefined,
  };

  // Transform centers if present
  if ("centers" in component && component.centers) {
    const centers = component.centers;
    const newCenters = new Float32Array(centers.length);

    for (let i = 0; i < centers.length / 3; i++) {
      const local: Vec3 = [centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2]];

      // Scale
      const scaled: Vec3 = [
        local[0] * transform.scale[0],
        local[1] * transform.scale[1],
        local[2] * transform.scale[2],
      ];

      // Rotate
      const rotated = quatRotate(transform.quaternion, scaled);

      // Translate
      const transformed = add(transform.position, rotated);

      newCenters[i * 3] = transformed[0];
      newCenters[i * 3 + 1] = transformed[1];
      newCenters[i * 3 + 2] = transformed[2];
    }

    (result as any).centers = newCenters;
  }

  // Transform segment endpoints for LineSegments (starts/ends)
  if ("starts" in component && component.starts && "ends" in component && component.ends) {
    const starts = component.starts as Float32Array;
    const ends = component.ends as Float32Array;
    const segCount = Math.min(starts.length, ends.length) / 3;
    const newStarts = new Float32Array(segCount * 3);
    const newEnds = new Float32Array(segCount * 3);

    for (let i = 0; i < segCount; i++) {
      const startLocal: Vec3 = [
        starts[i * 3],
        starts[i * 3 + 1],
        starts[i * 3 + 2],
      ];
      const endLocal: Vec3 = [
        ends[i * 3],
        ends[i * 3 + 1],
        ends[i * 3 + 2],
      ];

      const startScaled: Vec3 = [
        startLocal[0] * transform.scale[0],
        startLocal[1] * transform.scale[1],
        startLocal[2] * transform.scale[2],
      ];
      const endScaled: Vec3 = [
        endLocal[0] * transform.scale[0],
        endLocal[1] * transform.scale[1],
        endLocal[2] * transform.scale[2],
      ];

      const startRotated = quatRotate(transform.quaternion, startScaled);
      const endRotated = quatRotate(transform.quaternion, endScaled);

      const startTransformed = add(transform.position, startRotated);
      const endTransformed = add(transform.position, endRotated);

      newStarts[i * 3] = startTransformed[0];
      newStarts[i * 3 + 1] = startTransformed[1];
      newStarts[i * 3 + 2] = startTransformed[2];

      newEnds[i * 3] = endTransformed[0];
      newEnds[i * 3 + 1] = endTransformed[1];
      newEnds[i * 3 + 2] = endTransformed[2];
    }

    (result as any).starts = newStarts;
    (result as any).ends = newEnds;
  }

  // Transform points for LineBeams (format: [x,y,z,lineIndex, ...])
  if ("points" in component && component.points) {
    const points = component.points;
    const newPoints = new Float32Array(points.length);

    for (let i = 0; i < points.length / 4; i++) {
      const local: Vec3 = [points[i * 4], points[i * 4 + 1], points[i * 4 + 2]];
      const lineIndex = points[i * 4 + 3];

      // Scale
      const scaled: Vec3 = [
        local[0] * transform.scale[0],
        local[1] * transform.scale[1],
        local[2] * transform.scale[2],
      ];

      // Rotate
      const rotated = quatRotate(transform.quaternion, scaled);

      // Translate
      const transformed = add(transform.position, rotated);

      newPoints[i * 4] = transformed[0];
      newPoints[i * 4 + 1] = transformed[1];
      newPoints[i * 4 + 2] = transformed[2];
      newPoints[i * 4 + 3] = lineIndex;
    }

    (result as any).points = newPoints;
  }

  // Compose quaternions if component has orientation
  if ("quaternion" in component && component.quaternion) {
    const composed = quatMultiply(transform.quaternion, component.quaternion as Quat);
    (result as any).quaternion = composed;
  }
  if ("quaternions" in component && component.quaternions) {
    const quats = component.quaternions as Float32Array;
    const newQuats = new Float32Array(quats.length);

    for (let i = 0; i < quats.length / 4; i++) {
      const local: Quat = [quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]];
      const composed = quatMultiply(transform.quaternion, local);
      newQuats[i * 4] = composed[0];
      newQuats[i * 4 + 1] = composed[1];
      newQuats[i * 4 + 2] = composed[2];
      newQuats[i * 4 + 3] = composed[3];
    }

    (result as any).quaternions = newQuats;
  }

  // Scale half_sizes if component has them
  if ("half_size" in component && component.half_size) {
    const hs = component.half_size;
    if (Array.isArray(hs)) {
      (result as any).half_size = [
        hs[0] * transform.scale[0],
        hs[1] * transform.scale[1],
        hs[2] * transform.scale[2],
      ];
    } else {
      // Uniform scale - use average
      const avgScale = (transform.scale[0] + transform.scale[1] + transform.scale[2]) / 3;
      (result as any).half_size = (hs as number) * avgScale;
    }
  }
  if ("half_sizes" in component && component.half_sizes) {
    const sizes = component.half_sizes as Float32Array;
    const newSizes = new Float32Array(sizes.length);

    for (let i = 0; i < sizes.length / 3; i++) {
      newSizes[i * 3] = sizes[i * 3] * transform.scale[0];
      newSizes[i * 3 + 1] = sizes[i * 3 + 1] * transform.scale[1];
      newSizes[i * 3 + 2] = sizes[i * 3 + 2] * transform.scale[2];
    }

    (result as any).half_sizes = newSizes;
  }

  if (isImagePlane) {
    if ("size" in component && component.size !== undefined) {
      const sizeVal = component.size as number | Vec3;
      if (Array.isArray(sizeVal)) {
        (result as any).size = [
          sizeVal[0] * transform.scale[0],
          sizeVal[1] * transform.scale[1],
        ];
      } else {
        const avgScale =
          (transform.scale[0] + transform.scale[1] + transform.scale[2]) / 3;
        (result as any).size = sizeVal * avgScale;
      }
    }
    if ("sizes" in component && component.sizes) {
      const sizes = component.sizes as Float32Array;
      const newSizes = new Float32Array(sizes.length);
      if (sizes.length % 2 === 0) {
        for (let i = 0; i < sizes.length / 2; i++) {
          newSizes[i * 2] = sizes[i * 2] * transform.scale[0];
          newSizes[i * 2 + 1] = sizes[i * 2 + 1] * transform.scale[1];
        }
      } else {
        const avgScale =
          (transform.scale[0] + transform.scale[1] + transform.scale[2]) / 3;
        for (let i = 0; i < sizes.length; i++) {
          newSizes[i] = sizes[i] * avgScale;
        }
      }
      (result as any).sizes = newSizes;
    }
  } else {
    // Scale size for PointCloud/LineBeams (uniform scale)
    if ("size" in component && component.size !== undefined) {
      const avgScale =
        (transform.scale[0] + transform.scale[1] + transform.scale[2]) / 3;
      (result as any).size = (component.size as number) * avgScale;
    }
    if ("sizes" in component && component.sizes) {
      const sizes = component.sizes as Float32Array;
      const avgScale =
        (transform.scale[0] + transform.scale[1] + transform.scale[2]) / 3;
      const newSizes = new Float32Array(sizes.length);

      for (let i = 0; i < sizes.length; i++) {
        newSizes[i] = sizes[i] * avgScale;
      }

      (result as any).sizes = newSizes;
    }
  }

  // Scale mesh-style scale/scales (vec3)
  if ("scale" in component && component.scale !== undefined) {
    const scaleVal = component.scale as number | Vec3;
    if (Array.isArray(scaleVal)) {
      (result as any).scale = [
        scaleVal[0] * transform.scale[0],
        scaleVal[1] * transform.scale[1],
        scaleVal[2] * transform.scale[2],
      ];
    } else {
      const avgScale = (transform.scale[0] + transform.scale[1] + transform.scale[2]) / 3;
      (result as any).scale = scaleVal * avgScale;
    }
  }
  if ("scales" in component && component.scales) {
    const scales = component.scales as Float32Array;
    const newScales = new Float32Array(scales.length);
    if (scales.length % 3 === 0) {
      for (let i = 0; i < scales.length / 3; i++) {
        newScales[i * 3] = scales[i * 3] * transform.scale[0];
        newScales[i * 3 + 1] = scales[i * 3 + 1] * transform.scale[1];
        newScales[i * 3 + 2] = scales[i * 3 + 2] * transform.scale[2];
      }
    } else {
      const avgScale =
        (transform.scale[0] + transform.scale[1] + transform.scale[2]) / 3;
      for (let i = 0; i < scales.length; i++) {
        newScales[i] = scales[i] * avgScale;
      }
    }
    (result as any).scales = newScales;
  }

  return result;
}

// =============================================================================
// Flattening
// =============================================================================

/**
 * Flatten a component tree, resolving all groups into transformed primitives.
 *
 * @param components - Array of components (may include Groups)
 * @param parentTransform - Transform to apply from parent (default: identity)
 * @param groupPath - Path of group names from root (default: [])
 * @returns Flattened array of primitive components with transforms applied
 */
export function flattenGroups(
  components: (ComponentConfig | GroupConfig)[],
  parentTransform: Transform = identityTransform(),
  groupPath: string[] = [],
): FlattenedComponent[] {
  const result: FlattenedComponent[] = [];

  for (const component of components) {
    if (isGroup(component)) {
      // Compose transforms
      const groupTransform = getGroupTransform(component);
      const worldTransform = composeTransforms(parentTransform, groupTransform);

      // Update group path
      const newPath = component.name
        ? [...groupPath, component.name]
        : groupPath;

      // Recursively flatten children
      const flattened = flattenGroups(component.children, worldTransform, newPath);
      result.push(...flattened);
    } else {
      // Apply parent transform to this primitive
      const transformed = applyTransformToComponent(
        component,
        parentTransform,
        groupPath,
      );
      result.push(transformed);
    }
  }

  return result;
}

/**
 * Check if any components in the array are Groups.
 */
export function hasGroups(components: (ComponentConfig | GroupConfig)[]): boolean {
  return components.some(isGroup);
}
