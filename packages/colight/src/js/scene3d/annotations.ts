/**
 * @module annotations
 * @description Named text callouts for Scene3D — the annotation analogue of
 * named selections. An annotation is a NAMED object resident in
 * `$state.annotations`, so it syncs Python<->JS, persists into `.colight`
 * artifacts, and either party can mutate it; the NAME is the shared referent.
 *
 * Each annotation anchors in data space (a world-space position, or the center
 * of a component instance) and renders as a marker dot + thin leader + text
 * label — a DOM overlay projected by the CURRENT camera (see scene3d.tsx), so
 * it is captured by the same full-page screenshot path as the legend overlay.
 *
 * This module is GPU-free and unit-testable: it resolves anchors to world
 * positions and (given a camera + viewport) projects them to screen positions
 * with a visibility flag. The projection reuses the renderer's own camera
 * transforms (project.ts / camera3d.ts) so the overlay can never disagree with
 * the pixels.
 */

import { ComponentConfig, PrimitiveSpec } from "./components";
import { GPUTransform } from "./gpu-transforms";
import { rotateVector } from "./quaternion";
import { Vec3 } from "./vec3";
import { CameraState } from "./camera3d";
import { projectToScreen, Rect } from "./project";

/** An annotation's data-space anchor: a world position or an instance center. */
export type AnnotationAnchor =
  | { position: [number, number, number] }
  | { component: number; instance: number };

/** Small styling knobs for a callout (marker / leader / label accent). */
export interface AnnotationStyle {
  /** Accent color [r, g, b] in 0-1 (marker fill, leader, label border). */
  color?: [number, number, number];
}

/** A named annotation as it lives in `$state.annotations`. */
export interface Annotation {
  /** The callout's free text. */
  text: string;
  /** Where the callout points (world position or instance center). */
  anchor: AnnotationAnchor;
  /** Optional accent styling. */
  style?: AnnotationStyle;
}

/** `$state.annotations` shape: name -> Annotation. */
export type Annotations = Record<string, Annotation>;

/** True when an anchor is the instance-center form. */
export function isInstanceAnchor(
  anchor: AnnotationAnchor,
): anchor is { component: number; instance: number } {
  return (
    (anchor as any).component !== undefined &&
    (anchor as any).instance !== undefined
  );
}

/**
 * Applies a group transform (scale, rotate, translate) to a local point,
 * mirroring the renderer's per-component transform (see pick-snapshot's
 * applyTransform). No-op when the transform is absent/identity.
 */
function applyTransform(point: Vec3, transform?: GPUTransform): Vec3 {
  if (!transform) return point;
  const scaled: Vec3 = [
    point[0] * transform.scale[0],
    point[1] * transform.scale[1],
    point[2] * transform.scale[2],
  ];
  const rotated = rotateVector(scaled, transform.quaternion);
  return [
    rotated[0] + transform.position[0],
    rotated[1] + transform.position[1],
    rotated[2] + transform.position[2],
  ];
}

/**
 * Resolves an annotation anchor to a world position in the RENDERED (origin-
 * shifted) coordinate space, i.e. the space the camera projects.
 *
 * - Position anchors are given in world (pre-origin) coordinates; `origin` is
 *   subtracted so they line up with the geometry, whose positions were already
 *   shifted by -origin in Python (see Scene(origin=...)).
 * - Instance anchors resolve to the instance's center from the primitive
 *   registry (the same centers picking uses), transformed by the component's
 *   group transform. These centers are already in the shifted space (they come
 *   from the pre-shifted component data), so `origin` is NOT applied to them.
 *
 * Returns null when an instance anchor cannot be resolved (unknown component,
 * out-of-range instance, or missing spec).
 */
export function resolveAnchorWorld(
  anchor: AnnotationAnchor,
  components: ComponentConfig[],
  specFor: (component: ComponentConfig) => PrimitiveSpec<any> | undefined,
  transforms: GPUTransform[],
  origin?: [number, number, number] | null,
): Vec3 | null {
  if (isInstanceAnchor(anchor)) {
    const component = components[anchor.component];
    if (!component) return null;
    const spec = specFor(component);
    if (!spec) return null;
    const count = spec.getElementCount(component);
    if (anchor.instance < 0 || anchor.instance >= count) return null;
    const centers = spec.getCenters(component);
    const i = anchor.instance;
    if (centers.length < (i + 1) * 3) return null;
    const local: Vec3 = [
      centers[i * 3],
      centers[i * 3 + 1],
      centers[i * 3 + 2],
    ];
    const transformIndex = (component as any)._transformIndex ?? 0;
    return applyTransform(local, transforms[transformIndex]);
  }
  const p = anchor.position;
  if (!p || p.length < 3) return null;
  if (origin) {
    return [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
  }
  return [p[0], p[1], p[2]];
}

/** A resolved annotation ready for overlay rendering and machine reporting. */
export interface ResolvedAnnotation {
  /** The annotation name (shared referent). */
  name: string;
  /** The callout text. */
  text: string;
  /** The anchor as given (for reporting; instance anchors keep their ids). */
  anchor: AnnotationAnchor;
  /** Accent color [r, g, b] in 0-1. */
  color: [number, number, number];
  /** World position in the rendered (origin-shifted) space, or null when the
   * instance anchor could not be resolved. */
  world: Vec3 | null;
  /** Screen position in CANVAS-LOCAL CSS pixels (origin = canvas top-left), or
   * null when behind the camera / anchor unresolved. */
  screen: { x: number; y: number } | null;
  /** True when the anchor projects in front of the camera AND within the
   * viewport rectangle (label is drawn only when visible). */
  visible: boolean;
}

/** Default callout accent (matches the selection highlight amber). */
export const DEFAULT_ANNOTATION_COLOR: [number, number, number] = [
  1.0, 0.85, 0.2,
];

/**
 * Resolves every annotation to a world position and projects it with the
 * current camera + viewport. Sort is by name for stable overlay/report order.
 *
 * @param annotations `$state.annotations` (name -> Annotation).
 * @param components   Compiled components (for instance-anchor centers).
 * @param specFor      Resolves a component's primitive spec.
 * @param transforms   Group transforms (index by `_transformIndex`).
 * @param camera       The current camera state (renderer's live camera).
 * @param rect         Canvas rect ({width, height} in CSS pixels).
 * @param origin       Scene origin ([x,y,z]) or null; applied to position
 *                     anchors so they line up with the shifted geometry.
 */
export function resolveAnnotations(
  annotations: Annotations | undefined,
  components: ComponentConfig[],
  specFor: (component: ComponentConfig) => PrimitiveSpec<any> | undefined,
  transforms: GPUTransform[],
  camera: CameraState | null,
  rect: Rect | null,
  origin?: [number, number, number] | null,
): ResolvedAnnotation[] {
  if (!annotations) return [];
  const out: ResolvedAnnotation[] = [];
  const names = Object.keys(annotations).sort();
  for (const name of names) {
    const ann = annotations[name];
    if (!ann || !ann.anchor) continue;
    const world = resolveAnchorWorld(
      ann.anchor,
      components,
      specFor,
      transforms,
      origin,
    );
    let screen: { x: number; y: number } | null = null;
    let visible = false;
    if (world && camera && rect && rect.width > 0 && rect.height > 0) {
      screen = projectToScreen(world, rect, camera);
      if (screen) {
        visible =
          screen.x >= 0 &&
          screen.x <= rect.width &&
          screen.y >= 0 &&
          screen.y <= rect.height;
      }
    }
    out.push({
      name,
      text: ann.text ?? "",
      anchor: ann.anchor,
      color: ann.style?.color ?? DEFAULT_ANNOTATION_COLOR,
      world,
      screen,
      visible,
    });
  }
  return out;
}

/**
 * Names of the annotations anchored to a given (component, instance) — used to
 * add `annotations: [...]` to pick-at results. Only instance anchors belong to
 * an instance; position anchors never do.
 */
export function annotationsForInstance(
  annotations: Annotations | undefined,
  componentIndex: number,
  instanceIndex: number,
): string[] {
  if (!annotations) return [];
  const names: string[] = [];
  for (const name of Object.keys(annotations).sort()) {
    const anchor = annotations[name]?.anchor;
    if (
      anchor &&
      isInstanceAnchor(anchor) &&
      anchor.component === componentIndex &&
      anchor.instance === instanceIndex
    ) {
      names.push(name);
    }
  }
  return names;
}
