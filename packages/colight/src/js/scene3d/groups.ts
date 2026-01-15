/**
 * @module groups
 * @description Hierarchical transform groups for Scene3D.
 *
 * Groups allow composing multiple components with a shared transform.
 * At render time, groups are flattened into transformed primitives.
 */

import { Vec3, add } from "./vec3";
import { ComponentConfig } from "./components";
import {
  BaseComponentConfig,
  HoverProps,
  PickInfo,
  DragInfo,
  DragConstraint,
} from "./types";
import {
  GPUTransform,
  toGPUTransform,
  IDENTITY_GPU_TRANSFORM,
} from "./gpu-transforms";

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
  /** Default props for children (child values take precedence) */
  childDefaults?: GroupStyleProps;
  /** Override props for children (group values take precedence) */
  childOverrides?: GroupStyleProps;
  /** Position offset in parent space */
  position?: Vec3;
  /** Rotation as quaternion [x, y, z, w] */
  quaternion?: Quat;
  /** Scale factor (uniform or per-axis) */
  scale?: number | Vec3;
  /** Optional name for identifying this group in pick info */
  name?: string;
  /** Props to apply to ALL children when ANY child is hovered */
  hoverProps?: HoverProps;
  /** Handler called when any child is hovered (bubbles up) */
  onHover?: (info: PickInfo) => void;
  /** Handler called when any child is clicked (bubbles up) */
  onClick?: (info: PickInfo) => void;
  /** Handler called when drag starts on any child */
  onDragStart?: (info: DragInfo) => void;
  /** Handler called when any child is dragged (bubbles up) */
  onDrag?: (info: DragInfo) => void;
  /** Handler called when drag ends on any child */
  onDragEnd?: (info: DragInfo) => void;
  /** Constraint for drag operations (applies to all children) */
  dragConstraint?: DragConstraint;
}

/**
 * Group handlers and hover props, keyed by group path.
 * Used to implement event bubbling and group-level hover styling.
 */
export interface GroupHandlers {
  hoverProps?: HoverProps;
  onHover?: (info: PickInfo) => void;
  onClick?: (info: PickInfo) => void;
  onDragStart?: (info: DragInfo) => void;
  onDrag?: (info: DragInfo) => void;
  onDragEnd?: (info: DragInfo) => void;
  dragConstraint?: DragConstraint;
}

/**
 * Registry mapping group paths to their handlers.
 * Key is the group path joined with "/" (e.g., "outer/inner").
 */
export type GroupRegistry = Map<string, GroupHandlers>;

/**
 * A transform combining position, rotation, and scale.
 */
export interface Transform {
  position: Vec3;
  quaternion: Quat;
  scale: Vec3;
}

/**
 * Props that can be inherited from Groups onto child components.
 * Uses only scalar/object props to avoid copying large arrays.
 */
export type GroupStyleProps = Pick<
  BaseComponentConfig,
  | "color"
  | "alpha"
  | "outline"
  | "outlineColor"
  | "outlineWidth"
  | "layer"
  | "pickingScale"
  | "hoverProps"
>;

/**
 * A flattened component with group ancestry information.
 * This extends the original component config with optional metadata.
 */
export type FlattenedComponent = ComponentConfig & {
  /** Path of group names from root to this component */
  _groupPath?: string[];
  /** Original component index before flattening (for debugging) */
  _originalIndex?: number;
  /** Index into GPU transforms buffer (0 = identity, no transform) */
  _transformIndex?: number;
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
export function composeTransforms(
  parent: Transform,
  child: Transform,
): Transform {
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
    typeof rawScale === "number" ? [rawScale, rawScale, rawScale] : rawScale;

  return { position, quaternion, scale };
}

// =============================================================================
// Component Transform Application
// =============================================================================

/**
 * Check if a config is a Group.
 */
export function isGroup(
  config: ComponentConfig | GroupConfig,
): config is GroupConfig {
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

function groupHasChildStyling(group: GroupConfig): boolean {
  const hasDefaults =
    !!group.childDefaults && Object.keys(group.childDefaults).length > 0;
  const hasOverrides =
    !!group.childOverrides && Object.keys(group.childOverrides).length > 0;
  return hasDefaults || hasOverrides;
}

/**
 * Accumulated style props from parent groups.
 */
interface AccumulatedProps {
  defaults?: GroupStyleProps;
  overrides?: GroupStyleProps;
}

function mergeAccumulatedProps(
  parent: AccumulatedProps | undefined,
  group: GroupConfig,
): AccumulatedProps | undefined {
  const defaults =
    parent?.defaults || group.childDefaults
      ? { ...parent?.defaults, ...group.childDefaults }
      : undefined;
  const overrides =
    parent?.overrides || group.childOverrides
      ? { ...parent?.overrides, ...group.childOverrides }
      : undefined;
  if (!defaults && !overrides) return undefined;
  return { defaults, overrides };
}

function applyAccumulatedProps(
  component: ComponentConfig,
  accumulated?: AccumulatedProps,
): ComponentConfig {
  if (!accumulated) return component;
  // Apply defaults first (child wins), then overrides (group wins)
  let result = applyDefaultsToComponent(component, accumulated.defaults);
  result = applyOverridesToComponent(result, accumulated.overrides);
  return result;
}

const STYLE_KEYS: (keyof GroupStyleProps)[] = [
  "color",
  "alpha",
  "outline",
  "outlineColor",
  "outlineWidth",
  "layer",
  "pickingScale",
  "hoverProps",
];

/**
 * Apply default props to a component (component values take precedence).
 */
function applyDefaultsToComponent(
  component: ComponentConfig,
  defaults?: GroupStyleProps,
): ComponentConfig {
  if (!defaults) return component;

  let updated: ComponentConfig | null = null;
  for (const key of STYLE_KEYS) {
    const groupValue = defaults[key];
    if (groupValue === undefined) continue;
    // Skip if component already has this prop (child wins)
    if ((component as GroupStyleProps)[key] !== undefined) continue;
    if (!updated) updated = { ...component };
    (updated as any)[key] = groupValue;
  }

  return updated ?? component;
}

/**
 * Apply override props to a component (group values take precedence).
 */
function applyOverridesToComponent(
  component: ComponentConfig,
  overrides?: GroupStyleProps,
): ComponentConfig {
  if (!overrides) return component;

  let updated: ComponentConfig | null = null;
  for (const key of STYLE_KEYS) {
    const groupValue = overrides[key];
    if (groupValue === undefined) continue;
    // Always apply (group wins)
    if (!updated) updated = { ...component };
    (updated as any)[key] = groupValue;
  }

  return updated ?? component;
}

/**
 * Attach group path metadata to a component.
 * Transform application is now done on GPU via the transforms buffer.
 * Always clones the component to avoid mutating the original object
 * (prevents stale values in constants cache when components are reused).
 */
function applyTransformToComponent(
  component: ComponentConfig,
  _transform: Transform,
  groupPath: string[],
): FlattenedComponent {
  // GPU transforms: just add metadata, no CPU transformation needed
  // Always spread to avoid mutating original component (constants cache uses WeakMap keyed by object)
  if (groupPath.length === 0) {
    return { ...component } as FlattenedComponent;
  }
  return {
    ...component,
    _groupPath: groupPath,
  };
}

// =============================================================================
// Flattening
// =============================================================================

/**
 * Check if a Group has any handlers or hoverProps that require registry tracking.
 */
function groupHasHandlers(group: GroupConfig): boolean {
  return !!(
    group.hoverProps ||
    group.onHover ||
    group.onClick ||
    group.onDragStart ||
    group.onDrag ||
    group.onDragEnd ||
    group.dragConstraint
  );
}

/**
 * Result of flattening groups, including both flattened components and group registry.
 */
export interface FlattenGroupsResult {
  /** Flattened array of primitive components with transforms applied */
  components: FlattenedComponent[];
  /** Registry mapping group paths to their handlers */
  groupRegistry: GroupRegistry;
  /** Array of GPU transforms (index 0 is always identity) */
  transforms: GPUTransform[];
}

// Counter for generating unique anonymous group names
let anonymousGroupCounter = 0;

/**
 * Internal recursive implementation of flattenGroups.
 */
function flattenGroupsInternal(
  components: (ComponentConfig | GroupConfig)[],
  parentTransform: Transform,
  groupPath: string[],
  parentProps: AccumulatedProps | undefined,
  registry: GroupRegistry,
  transforms: GPUTransform[],
): FlattenedComponent[] {
  const result: FlattenedComponent[] = [];

  for (const component of components) {
    if (isGroup(component)) {
      // Get group name - auto-generate one if group has handlers but no explicit name
      const groupName =
        component.name ??
        (groupHasHandlers(component)
          ? `_group_${anonymousGroupCounter++}`
          : undefined);

      // Update group path if named (explicitly or auto-generated)
      const newPath = groupName ? [...groupPath, groupName] : groupPath;

      // Compute world transform first (needed for registry position)
      const worldTransform = groupHasTransform(component)
        ? composeTransforms(parentTransform, getGroupTransform(component))
        : parentTransform;

      // Register group handlers if present
      if (groupName && groupHasHandlers(component)) {
        const pathKey = newPath.join("/");
        registry.set(pathKey, {
          hoverProps: component.hoverProps,
          onHover: component.onHover,
          onClick: component.onClick,
          onDragStart: component.onDragStart,
          onDrag: component.onDrag,
          onDragEnd: component.onDragEnd,
          dragConstraint: component.dragConstraint,
        });
      }

      const mergedProps = mergeAccumulatedProps(parentProps, component);

      // Recursively flatten children
      const flattened = flattenGroupsInternal(
        component.children,
        worldTransform,
        newPath,
        mergedProps,
        registry,
        transforms,
      );
      result.push(...flattened);
    } else {
      const styledComponent = applyAccumulatedProps(component, parentProps);

      // Determine transform index for GPU transforms
      // Index 0 = identity (no transform needed)
      let transformIndex = 0;
      if (!isIdentityTransform(parentTransform)) {
        // Add this transform to the array and use its index
        transformIndex = transforms.length;
        transforms.push(toGPUTransform(parentTransform));
      }

      // Apply parent transform to this primitive (CPU path - will be removed later)
      const transformed = applyTransformToComponent(
        styledComponent,
        parentTransform,
        groupPath,
      );

      // Attach transform index for GPU path
      transformed._transformIndex = transformIndex;

      result.push(transformed);
    }
  }

  return result;
}

/**
 * Flatten a component tree, resolving all groups into transformed primitives.
 *
 * @param components - Array of components (may include Groups)
 * @param parentTransform - Transform to apply from parent (default: identity)
 * @param groupPath - Path of group names from root (default: [])
 * @returns Object containing flattened components and group registry
 */
export function flattenGroups(
  components: (ComponentConfig | GroupConfig)[],
  parentTransform: Transform = identityTransform(),
  groupPath: string[] = [],
): FlattenGroupsResult {
  // Reset anonymous group counter for deterministic names within each flatten call
  anonymousGroupCounter = 0;
  const registry: GroupRegistry = new Map();

  // Initialize transforms array with identity at index 0
  const transforms: GPUTransform[] = [IDENTITY_GPU_TRANSFORM];

  const flattenedComponents = flattenGroupsInternal(
    components,
    parentTransform,
    groupPath,
    undefined,
    registry,
    transforms,
  );
  return {
    components: flattenedComponents,
    groupRegistry: registry,
    transforms,
  };
}

/**
 * Check if any components in the tree are Groups with transforms, child styling, or handlers.
 * Returns false for composition-only groups (no position/quaternion/scale/childDefaults/childOverrides/handlers).
 */
export function hasGroups(
  components: (ComponentConfig | GroupConfig)[],
): boolean {
  for (const component of components) {
    if (isGroup(component)) {
      if (
        groupHasTransform(component) ||
        groupHasChildStyling(component) ||
        groupHasHandlers(component)
      ) {
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
export function hasAnyGroups(
  components: (ComponentConfig | GroupConfig)[],
): boolean {
  return components.some(isGroup);
}
