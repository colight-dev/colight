/**
 * @module compiler
 * @description Unified scene compilation pipeline for Scene3D.
 *
 * All entry paths (JSX, components prop, layers) are normalized through this module
 * to ensure consistent helper expansion, coercion, group flattening, and mesh resolution.
 */

import { ComponentConfig, PrimitiveSpec } from "./components";
import {
  pointCloudSpec,
  ellipsoidSpec,
  cuboidSpec,
  imagePlaneSpec,
  boundingBoxSpec,
  lineBeamsSpec,
  lineSegmentsSpec,
} from "./components";
import {
  GroupConfig,
  GroupRegistry,
  flattenGroups,
  hasAnyGroups,
} from "./groups";
import { GPUTransform, IDENTITY_GPU_TRANSFORM } from "./gpu-transforms";
import {
  GridHelper,
  GridHelperProps,
  CameraFrustum,
  CameraFrustumProps,
  ImageProjection,
  ImageProjectionProps,
} from "./helpers";
import { resolveInlineMeshes, InlineMeshComponentConfig } from "./inlineMesh";
import { normalizePrimitiveSpecs, PrimitiveSpecMap } from "./coercion";

// =============================================================================
// Types
// =============================================================================

/** Input types that can be passed to the compiler */
export type RawComponent =
  | ComponentConfig
  | GroupConfig
  | InlineMeshComponentConfig
  | HelperConfig;

/** Helper component configs (before expansion) */
export type HelperConfig =
  | ({ type: "ImageProjection" } & ImageProjectionProps)
  | ({ type: "CameraFrustum" } & CameraFrustumProps)
  | ({ type: "GridHelper" } & GridHelperProps);

/** Result of compiling a scene */
export interface CompiledScene {
  /** Flattened, coerced components ready for rendering */
  components: ComponentConfig[];
  /** GPU transforms array (index 0 = identity) */
  transforms: GPUTransform[];
  /** Registry of group handlers for event bubbling */
  groupRegistry: GroupRegistry | undefined;
  /** Merged primitive specs (user-provided + inline meshes) */
  primitiveSpecs: Record<string, PrimitiveSpec<any>> | undefined;
}

// =============================================================================
// Primitive Spec Registry
// =============================================================================

/**
 * Registry mapping primitive type names to their specs.
 * Used to apply coercion to raw component data.
 */
const PRIMITIVE_SPECS: Record<string, PrimitiveSpec<any>> = {
  PointCloud: pointCloudSpec,
  Ellipsoid: ellipsoidSpec,
  Cuboid: cuboidSpec,
  ImagePlane: imagePlaneSpec,
  BoundingBox: boundingBoxSpec,
  LineBeams: lineBeamsSpec,
  LineSegments: lineSegmentsSpec,
};

/**
 * Set of valid primitive type names (after helper expansion).
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

/**
 * Helper type names that need expansion.
 */
const HELPER_TYPES = new Set([
  "ImageProjection",
  "CameraFrustum",
  "GridHelper",
]);

// =============================================================================
// Step 1: Helper Expansion
// =============================================================================

/**
 * Expands a helper component into its primitive components.
 */
function expandHelper(component: HelperConfig): RawComponent[] {
  switch (component.type) {
    case "ImageProjection":
      return ImageProjection(component as ImageProjectionProps);
    case "CameraFrustum":
      return [CameraFrustum(component as CameraFrustumProps)];
    case "GridHelper":
      return [GridHelper(component as GridHelperProps)];
    default:
      return [component];
  }
}

/**
 * Recursively expands all helper components in a component tree.
 * This is the first step in the compilation pipeline.
 */
function expandHelpers(components: RawComponent[]): RawComponent[] {
  const result: RawComponent[] = [];

  for (const component of components) {
    if (!component) continue;

    // Handle arrays (from nested compositions)
    if (Array.isArray(component)) {
      result.push(...expandHelpers(component as RawComponent[]));
      continue;
    }

    // Expand helper types
    if (HELPER_TYPES.has(component.type)) {
      const expanded = expandHelper(component as HelperConfig);
      // Recursively expand in case helpers return other helpers
      result.push(...expandHelpers(expanded));
      continue;
    }

    // Recurse into Group children
    if (component.type === "Group") {
      const group = component as GroupConfig;
      result.push({
        ...group,
        children: expandHelpers(group.children as RawComponent[]) as (
          | ComponentConfig
          | GroupConfig
        )[],
      });
      continue;
    }

    // Pass through primitive types
    result.push(component);
  }

  return result;
}

// =============================================================================
// Step 2: Coercion
// =============================================================================

/**
 * Applies coercion to a raw component config via spec.coerce.
 * Each primitive's coerce function handles:
 * - Input coercion (singular → plural, scalar expansion)
 * - Array coercion (NdArray/arrays → Float32Array)
 */
function coerceComponent<T extends { type: string }>(component: T): T {
  const spec = PRIMITIVE_SPECS[component.type];
  if (spec?.coerce) {
    return spec.coerce(component) as T;
  }
  return component;
}

/**
 * Recursively applies coercion to components, including nested Group children.
 * This is the second step in the compilation pipeline (after helper expansion).
 */
function coerceComponents(components: RawComponent[]): RawComponent[] {
  const result: RawComponent[] = [];

  for (const component of components) {
    if (!component) continue;

    // Handle arrays
    if (Array.isArray(component)) {
      result.push(...coerceComponents(component as RawComponent[]));
      continue;
    }

    // Recurse into Group children
    if (component.type === "Group") {
      const group = component as GroupConfig;
      result.push({
        ...coerceComponent(group),
        children: coerceComponents(group.children as RawComponent[]) as (
          | ComponentConfig
          | GroupConfig
        )[],
      });
      continue;
    }

    // Apply coercion to primitive
    result.push(coerceComponent(component));
  }

  return result;
}

// =============================================================================
// Step 3: Validation & Filtering
// =============================================================================

/**
 * Check if a component type is valid (known primitive or custom spec).
 */
function isValidType(
  type: string,
  customSpecs?: Record<string, PrimitiveSpec<any>>,
): boolean {
  if (PRIMITIVE_TYPES.has(type)) return true;
  if (customSpecs && type in customSpecs) return true;
  return false;
}

/**
 * Recursively filters components to only include valid primitive types.
 * Unknown types are logged and skipped. Recurses into Group children.
 */
function filterValidComponents(
  components: RawComponent[],
  customSpecs?: Record<string, PrimitiveSpec<any>>,
): RawComponent[] {
  const result: RawComponent[] = [];

  for (const component of components) {
    if (!component || typeof component !== "object") continue;

    const type = component.type;

    // Handle Group: recurse into children
    if (type === "Group") {
      const group = component as GroupConfig;
      const filteredChildren = filterValidComponents(
        group.children as RawComponent[],
        customSpecs,
      );
      // Only include group if it has valid children
      if (filteredChildren.length > 0) {
        result.push({
          ...group,
          children: filteredChildren as (ComponentConfig | GroupConfig)[],
        });
      }
      continue;
    }

    // Check if type is valid
    if (isValidType(type, customSpecs)) {
      result.push(component);
      continue;
    }

    // Log unknown types in development
    if (process.env.NODE_ENV !== "production") {
      console.warn(`Scene3D: Unknown component type "${type}", skipping`);
    }
  }

  return result;
}

// =============================================================================
// Main Compiler
// =============================================================================

/**
 * Compiles a scene from raw components into a normalized, flattened representation.
 *
 * Pipeline steps:
 * 1. Expand helpers (ImageProjection → [ImagePlane, LineSegments], etc.)
 * 2. Coerce raw data (apply spec.coerce for type normalization)
 * 3. Filter to valid primitive types
 * 4. Flatten groups (resolve hierarchy, record _transformIndex per component)
 * 5. Resolve inline meshes (convert Mesh components to generated specs)
 *
 * @param rawComponents - Input components (may include helpers and groups)
 * @param userSpecs - User-provided primitive specs (for custom primitives)
 * @returns Compiled scene ready for rendering
 */
export function compileScene(
  rawComponents: RawComponent[],
  userSpecs?: PrimitiveSpecMap,
): CompiledScene {
  // 1. Expand helpers
  const expanded = expandHelpers(rawComponents);

  // 2. Apply coercion
  const coerced = coerceComponents(expanded);

  // Normalize user specs for filtering
  const normalizedUserSpecs = normalizePrimitiveSpecs(userSpecs);

  // 3. Filter to valid types
  const valid = filterValidComponents(coerced, normalizedUserSpecs);

  // 4. Flatten groups
  let components: (ComponentConfig | InlineMeshComponentConfig)[];
  let groupRegistry: GroupRegistry | undefined;
  let transforms: GPUTransform[];

  if (hasAnyGroups(valid as (ComponentConfig | GroupConfig)[])) {
    const result = flattenGroups(valid as (ComponentConfig | GroupConfig)[]);
    components = result.components as (
      | ComponentConfig
      | InlineMeshComponentConfig
    )[];
    groupRegistry =
      result.groupRegistry.size > 0 ? result.groupRegistry : undefined;
    transforms = result.transforms;
  } else {
    components = valid as (ComponentConfig | InlineMeshComponentConfig)[];
    transforms = [IDENTITY_GPU_TRANSFORM];
  }

  // 5. Resolve inline meshes
  const { components: resolvedComponents, inlineSpecs } =
    resolveInlineMeshes(components);

  // 6. Merge specs (user-provided + inline meshes)
  let primitiveSpecs: Record<string, PrimitiveSpec<any>> | undefined;
  if (inlineSpecs || normalizedUserSpecs) {
    primitiveSpecs = { ...normalizedUserSpecs, ...inlineSpecs };
  }

  return {
    components: resolvedComponents,
    transforms,
    groupRegistry,
    primitiveSpecs,
  };
}

// =============================================================================
// Helpers for Entry Points
// =============================================================================

/**
 * Checks if a component type is a valid primitive or helper.
 * Used by entry points to pre-filter before compilation.
 */
export function isValidComponentType(
  type: string,
  customSpecs?: Record<string, PrimitiveSpec<any>>,
): boolean {
  if (PRIMITIVE_TYPES.has(type)) return true;
  if (HELPER_TYPES.has(type)) return true;
  if (customSpecs && type in customSpecs) return true;
  return false;
}

/**
 * Re-export types for external use.
 */
export { PRIMITIVE_TYPES, HELPER_TYPES };
