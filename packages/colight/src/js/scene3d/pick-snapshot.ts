/**
 * @module pick-snapshot
 * @description Pure helpers for agent-facing full-frame pick readback.
 *
 * The GPU pick pass writes packed instance ids (see ./picking.ts) into a
 * full-frame buffer. This module serializes the *legend* that maps those ids
 * back to components — built from the same `ComponentOffset` registry that
 * interactive picking walks (`findPickedElement` in impl3d.tsx) — plus
 * selection-bounds and instance-dereference helpers used by the snapshot API.
 * Everything here is GPU-free and unit-testable.
 */

import { ComponentConfig, PrimitiveSpec } from "./components";
import { GPUTransform } from "./gpu-transforms";
import { unpackID } from "./picking";
import { rotateVector } from "./quaternion";
import { ComponentOffset } from "./types";
import { Vec3 } from "./vec3";

/** One contiguous pick-id range, serialized for out-of-process decoding. */
export interface PickLegendEntry {
  /** Component index in the compiled scene (matches interactive PickInfo). */
  component: number;
  /** Primitive type, e.g. "Cuboid". */
  type: string;
  /** Number of pickable elements this component contributed. */
  count: number;
  /** First global element index (raw pick id = packID(idBase + i)). */
  idBase: number;
  /** Path of group names from root, when the component is inside groups. */
  groupPath?: string[];
}

/**
 * Serializes the pick-id legend from render objects' component offsets.
 *
 * This consumes the exact registry interactive picking decodes with
 * (`ComponentOffset.pickingStart`/`elementCount`), so the CLI and the
 * browser can never disagree about what an id means.
 */
export function buildPickLegend(
  components: ComponentConfig[],
  offsetsPerRenderObject: ComponentOffset[][],
): PickLegendEntry[] {
  const entries: PickLegendEntry[] = [];
  for (const offsets of offsetsPerRenderObject) {
    for (const offset of offsets) {
      const component = components[offset.componentIdx];
      const entry: PickLegendEntry = {
        component: offset.componentIdx,
        type: component?.type ?? "unknown",
        count: offset.elementCount,
        idBase: offset.pickingStart,
      };
      const groupPath = (component as any)?._groupPath as string[] | undefined;
      if (groupPath?.length) entry.groupPath = groupPath;
      entries.push(entry);
    }
  }
  entries.sort((a, b) => a.idBase - b.idBase);
  return entries;
}

/**
 * Decodes one raw pick-buffer id against a legend.
 * Mirrors `unpackID` + `findPickedElement`: returns null for background.
 */
export function decodePickId(
  rawId: number,
  legend: PickLegendEntry[],
): { component: number; instance: number } | null {
  const globalIdx = unpackID(rawId);
  if (globalIdx === null) return null;
  for (const entry of legend) {
    if (globalIdx >= entry.idBase && globalIdx < entry.idBase + entry.count) {
      return { component: entry.component, instance: globalIdx - entry.idBase };
    }
  }
  return null;
}

const BASE64_ALPHABET =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/** Encodes a Uint32Array as base64 (little-endian byte order). */
export function u32ToBase64(ids: Uint32Array): string {
  const bytes = new Uint8Array(ids.buffer, ids.byteOffset, ids.byteLength);
  const parts: string[] = [];
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : 0;
    parts.push(
      BASE64_ALPHABET[b0 >> 2],
      BASE64_ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)],
      i + 1 < bytes.length
        ? BASE64_ALPHABET[((b1 & 0x0f) << 2) | (b2 >> 6)]
        : "=",
      i + 2 < bytes.length ? BASE64_ALPHABET[b2 & 0x3f] : "=",
    );
  }
  return parts.join("");
}

/** Decodes base64 (as produced by {@link u32ToBase64}) to a Uint32Array. */
export function base64ToU32(encoded: string): Uint32Array {
  const clean = encoded.replace(/=+$/, "");
  const bytes = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let out = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const n0 = BASE64_ALPHABET.indexOf(clean[i]);
    const n1 = BASE64_ALPHABET.indexOf(clean[i + 1] ?? "A");
    const n2 = BASE64_ALPHABET.indexOf(clean[i + 2] ?? "A");
    const n3 = BASE64_ALPHABET.indexOf(clean[i + 3] ?? "A");
    if (out < bytes.length) bytes[out++] = (n0 << 2) | (n1 >> 4);
    if (out < bytes.length) bytes[out++] = ((n1 & 0x0f) << 4) | (n2 >> 2);
    if (out < bytes.length) bytes[out++] = ((n2 & 0x03) << 6) | n3;
  }
  return new Uint32Array(bytes.buffer, 0, Math.floor(bytes.length / 4));
}

/** Instance index ranges, inclusive: [[start, end], ...]. */
export type InstanceRanges = Array<[number, number]>;

/** Returns whether `index` falls in any of the (inclusive) ranges. */
export function inRanges(index: number, ranges?: InstanceRanges): boolean {
  if (!ranges || ranges.length === 0) return true;
  for (const [start, end] of ranges) {
    if (index >= start && index <= end) return true;
  }
  return false;
}

// ========== Selection bounds ==========

export interface Bounds3 {
  min: Vec3;
  max: Vec3;
}

function readVec3At(arr: ArrayLike<number>, i: number): Vec3 {
  return [arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]];
}

/**
 * Conservative per-element radius from a component's size-like fields.
 * Falls back to the spec defaults when the component holds no explicit size.
 */
export function elementRadius(
  component: ComponentConfig,
  spec: PrimitiveSpec<any>,
  index: number,
): number {
  const comp = component as any;
  const defaults = (spec.defaults ?? {}) as Record<string, any>;
  const norm = (v: ArrayLike<number>, offset: number, n: number) => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += v[offset + i] * v[offset + i];
    return Math.sqrt(sum);
  };
  let radius = 0;
  if (comp.half_sizes && comp.half_sizes.length >= (index + 1) * 3) {
    radius = norm(comp.half_sizes, index * 3, 3);
  } else if (Array.isArray(comp.half_size) || comp.half_size?.length === 3) {
    radius = norm(comp.half_size, 0, 3);
  } else if (typeof comp.half_size === "number") {
    radius = comp.half_size * Math.sqrt(3);
  } else if (comp.sizes && comp.sizes.length > index) {
    radius = comp.sizes[index];
  } else if (typeof comp.size === "number") {
    radius = comp.size;
  } else if (defaults.half_size) {
    const d = defaults.half_size;
    radius = Array.isArray(d) ? norm(d, 0, 3) : Number(d) * Math.sqrt(3);
  } else if (typeof defaults.size === "number") {
    radius = defaults.size;
  }
  const scale =
    comp.scales && comp.scales.length > index
      ? comp.scales[index]
      : typeof comp.scale === "number"
        ? comp.scale
        : 1;
  return radius * scale;
}

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
 * World-space bounds of an instance selection: element centers (transformed
 * by the component's group transform) expanded by a conservative
 * per-element radius. Returns null for an empty selection.
 */
export function computeSelectionBounds(
  component: ComponentConfig,
  spec: PrimitiveSpec<any>,
  transforms: GPUTransform[],
  ranges?: InstanceRanges,
): Bounds3 | null {
  const centers = spec.getCenters(component);
  const elementCount = spec.getElementCount(component);
  if (!elementCount) return null;
  const transformIndex = (component as any)._transformIndex ?? 0;
  const transform = transforms[transformIndex];
  const maxScale = transform
    ? Math.max(...transform.scale.map((s: number) => Math.abs(s)))
    : 1;

  // Meshes carry a local geometry AABB: a single instance at its center can
  // span the whole scene, so the per-instance radius heuristic misses it.
  // Expand each instance center by the geometry's local half-extent instead.
  const localBounds = (spec as any).localBounds as Bounds3 | undefined;

  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  let any = false;
  for (let i = 0; i < elementCount; i++) {
    if (!inRanges(i, ranges)) continue;
    any = true;
    const rawCenter = readVec3At(centers, i);
    if (localBounds) {
      // Per-instance uniform scale (mesh ``scale``/``scales``), then group.
      const comp = component as any;
      const instScale =
        comp.scales && comp.scales.length > i
          ? comp.scales[i]
          : typeof comp.scale === "number"
            ? comp.scale
            : 1;
      // Add the local bbox (offset by the raw instance center), transformed
      // by the group transform. Using both bbox extremes captures the extent
      // even when the instance center is at the origin.
      for (const bx of [localBounds.min[0], localBounds.max[0]]) {
        for (const by of [localBounds.min[1], localBounds.max[1]]) {
          for (const bz of [localBounds.min[2], localBounds.max[2]]) {
            const local: Vec3 = [
              rawCenter[0] + bx * instScale,
              rawCenter[1] + by * instScale,
              rawCenter[2] + bz * instScale,
            ];
            const world = applyTransform(local, transform);
            for (let axis = 0; axis < 3; axis++) {
              min[axis] = Math.min(min[axis], world[axis]);
              max[axis] = Math.max(max[axis], world[axis]);
            }
          }
        }
      }
      continue;
    }
    const center = applyTransform(rawCenter, transform);
    const radius = elementRadius(component, spec, i) * maxScale;
    for (let axis = 0; axis < 3; axis++) {
      min[axis] = Math.min(min[axis], center[axis] - radius);
      max[axis] = Math.max(max[axis], center[axis] + radius);
    }
  }
  return any ? { min, max } : null;
}

/** Union of bounds; either side may be null. */
export function unionBounds(
  a: Bounds3 | null,
  b: Bounds3 | null,
): Bounds3 | null {
  if (!a) return b;
  if (!b) return a;
  return {
    min: [
      Math.min(a.min[0], b.min[0]),
      Math.min(a.min[1], b.min[1]),
      Math.min(a.min[2], b.min[2]),
    ],
    max: [
      Math.max(a.max[0], b.max[0]),
      Math.max(a.max[1], b.max[1]),
      Math.max(a.max[2], b.max[2]),
    ],
  };
}

// ========== Instance dereference ==========

/**
 * Dereferenced attribute values for a single instance, read from the same
 * compiled component config the renderer consumes (post-coercion, so
 * aliases like `center` → `centers` are already resolved).
 */
export function describeInstance(
  component: ComponentConfig,
  spec: PrimitiveSpec<any>,
  index: number,
): Record<string, unknown> {
  const comp = component as any;
  const defaults = (spec.defaults ?? {}) as Record<string, any>;
  const values: Record<string, unknown> = {};

  const centers = spec.getCenters(component);
  if (centers.length >= (index + 1) * 3) {
    values.center = Array.from(readVec3At(centers, index));
  }

  const vec3Field = (plural: string, singular: string) => {
    if (comp[plural] && comp[plural].length >= (index + 1) * 3) {
      return Array.from(readVec3At(comp[plural], index));
    }
    const single = comp[singular];
    if (single !== undefined) {
      return typeof single === "number"
        ? [single, single, single]
        : Array.from(single as ArrayLike<number>);
    }
    return undefined;
  };
  const scalarField = (plural: string, singular: string) => {
    if (comp[plural] && comp[plural].length > index) return comp[plural][index];
    if (typeof comp[singular] === "number") return comp[singular];
    return undefined;
  };

  // Decorations override per-instance color/alpha/scale for matching indexes
  // (last matching decoration wins, mirroring buildRenderData). Read them so
  // the reported values match what is actually drawn — a mesh decorated to
  // 25% alpha must report alpha 0.25, not 1.
  let decoColor: number[] | undefined;
  let decoAlpha: number | undefined;
  let decoScale: number | undefined;
  const decorations = comp.decorations as
    | Array<{
        indexes?: number[];
        color?: number[];
        alpha?: number;
        scale?: number;
      }>
    | undefined;
  if (Array.isArray(decorations)) {
    for (const deco of decorations) {
      if (!deco?.indexes || !deco.indexes.includes(index)) continue;
      if (deco.color !== undefined) decoColor = Array.from(deco.color);
      if (deco.alpha !== undefined) decoAlpha = deco.alpha;
      if (deco.scale !== undefined) decoScale = deco.scale;
    }
  }

  const color = vec3Field("colors", "color");
  values.color = decoColor ?? color ?? defaults.color ?? [0.5, 0.5, 0.5];
  const alpha = scalarField("alphas", "alpha");
  values.alpha = decoAlpha ?? alpha ?? defaults.alpha ?? 1.0;

  const halfSize = vec3Field("half_sizes", "half_size");
  if (halfSize !== undefined) values.half_size = halfSize;
  else if (defaults.half_size !== undefined) {
    const d = defaults.half_size;
    values.half_size = Array.isArray(d) ? d : [d, d, d];
  }

  const size = scalarField("sizes", "size");
  if (size !== undefined) values.size = size;
  else if (typeof defaults.size === "number") values.size = defaults.size;

  const scale = scalarField("scales", "scale");
  if (decoScale !== undefined) values.scale = decoScale;
  else if (scale !== undefined) values.scale = scale;

  if (comp.quaternions && comp.quaternions.length >= (index + 1) * 4) {
    values.quaternion = Array.from(
      comp.quaternions.slice(index * 4, index * 4 + 4) as ArrayLike<number>,
    );
  } else if (comp.quaternion) {
    values.quaternion = Array.from(comp.quaternion as ArrayLike<number>);
  }

  const groupPath = comp._groupPath as string[] | undefined;
  if (groupPath?.length) values.groupPath = groupPath;

  return values;
}
