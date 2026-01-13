/**
 * @module groups
 * @description Hierarchical transform groups for Scene3D.
 *
 * Groups allow composing multiple components with a shared transform.
 * At render time, groups are flattened into primitives with GPU-applied transforms.
 */

import { Vec3, add } from "./vec3";
import { ComponentConfig } from "./components";

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
  /** Group transform index for GPU scene graph */
  _groupId?: number;
  /** World-space group transform for CPU-side utilities */
  _groupTransform?: Transform;
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
 * Invert a quaternion (assumes unit length).
 */
export function quatInvert(q: Quat): Quat {
  return [-q[0], -q[1], -q[2], q[3]];
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
 * Check if a transform is identity (no translation, rotation, or scale).
 * Used to short-circuit expensive transform operations.
 */
export function isIdentityTransform(t: Transform): boolean {
  return (
    t.position[0] === 0 &&
    t.position[1] === 0 &&
    t.position[2] === 0 &&
    t.quaternion[0] === 0 &&
    t.quaternion[1] === 0 &&
    t.quaternion[2] === 0 &&
    t.quaternion[3] === 1 &&
    t.scale[0] === 1 &&
    t.scale[1] === 1 &&
    t.scale[2] === 1
  );
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
 * Apply a transform to a point (scale -> rotate -> translate).
 */
export function applyTransformToPoint(transform: Transform, point: Vec3): Vec3 {
  const scaled: Vec3 = [
    point[0] * transform.scale[0],
    point[1] * transform.scale[1],
    point[2] * transform.scale[2],
  ];
  const rotated = quatRotate(transform.quaternion, scaled);
  return add(transform.position, rotated);
}

/**
 * Apply the inverse of a transform to a point.
 */
export function applyInverseTransformToPoint(transform: Transform, point: Vec3): Vec3 {
  const translated: Vec3 = [
    point[0] - transform.position[0],
    point[1] - transform.position[1],
    point[2] - transform.position[2],
  ];
  const rotated = quatRotate(quatInvert(transform.quaternion), translated);
  return [
    rotated[0] / transform.scale[0],
    rotated[1] / transform.scale[1],
    rotated[2] / transform.scale[2],
  ];
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
 * Check if a Group has any transform properties defined.
 * Groups without transforms are used purely for composition.
 */
function groupHasTransform(group: GroupConfig): boolean {
  return (
    group.position !== undefined ||
    group.quaternion !== undefined ||
    group.scale !== undefined
  );
}

/**
 * Apply a transform to a single component's data.
 * Returns a new component config with transformed positions and composed orientations.
 */
export interface FlattenGroupsResult {
  components: FlattenedComponent[];
  groupTransforms: Transform[];
}

// =============================================================================
// Flattening
// =============================================================================

/**
 * Flatten a component tree, resolving all groups into primitives with group transforms.
 *
 * @param components - Array of components (may include Groups)
 * @param parentTransform - Transform to apply from parent (default: identity)
 * @param groupPath - Path of group names from root (default: [])
 * @returns Flattened primitives and group transforms for GPU application
 */
export function flattenGroups(
  components: (ComponentConfig | GroupConfig)[],
): FlattenGroupsResult {
  const groupTransforms: Transform[] = [identityTransform()];
  const result = flattenGroupsRecursive(
    components,
    identityTransform(),
    [],
    0,
    groupTransforms,
  );
  return { components: result, groupTransforms };
}

function flattenGroupsRecursive(
  components: (ComponentConfig | GroupConfig)[],
  parentTransform: Transform,
  groupPath: string[],
  parentGroupId: number,
  groupTransforms: Transform[],
): FlattenedComponent[] {
  const result: FlattenedComponent[] = [];

  for (const component of components) {
    if (isGroup(component)) {
      const newPath = component.name
        ? [...groupPath, component.name]
        : groupPath;

      let nextGroupId = parentGroupId;
      let worldTransform = parentTransform;

      if (groupHasTransform(component)) {
        worldTransform = composeTransforms(parentTransform, getGroupTransform(component));
        nextGroupId = groupTransforms.length;
        groupTransforms.push(worldTransform);
      }

      const flattened = flattenGroupsRecursive(
        component.children,
        worldTransform,
        newPath,
        nextGroupId,
        groupTransforms,
      );
      result.push(...flattened);
    } else {
      if (groupPath.length === 0 && parentGroupId === 0) {
        result.push(component as FlattenedComponent);
        continue;
      }

      const transformed: FlattenedComponent = {
        ...component,
        _groupPath: groupPath.length > 0 ? groupPath : undefined,
        _groupId: parentGroupId > 0 ? parentGroupId : undefined,
        _groupTransform:
          parentGroupId > 0 ? groupTransforms[parentGroupId] : undefined,
      };
      result.push(transformed);
    }
  }

  return result;
}

/**
 * Check if any components in the tree are Groups with transforms.
 * Returns false for composition-only groups (no position/quaternion/scale).
 */
export function hasGroups(components: (ComponentConfig | GroupConfig)[]): boolean {
  for (const component of components) {
    if (isGroup(component)) {
      if (groupHasTransform(component)) {
        return true;
      }
      // Recurse into children of composition-only groups
      if (hasGroups(component.children)) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Unwrap composition-only groups, extracting children into a flat array.
 * Use this when hasGroups returns false but there are still Group configs.
 */
export function unwrapGroups(
  components: (ComponentConfig | GroupConfig)[],
): ComponentConfig[] {
  const result: ComponentConfig[] = [];
  for (const component of components) {
    if (isGroup(component)) {
      result.push(...unwrapGroups(component.children));
    } else {
      result.push(component);
    }
  }
  return result;
}

/**
 * Check if any components are Group configs (regardless of transforms).
 */
export function hasAnyGroups(components: (ComponentConfig | GroupConfig)[]): boolean {
  return components.some(isGroup);
}
