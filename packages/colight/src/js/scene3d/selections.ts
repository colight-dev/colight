/**
 * @module selections
 * @description Named selections for Scene3D — the second consumer of the
 * shared per-instance mask abstraction (filtering is the first).
 *
 * A selection is a NAMED per-instance mask resolved from data (either an
 * explicit instance list or a threshold predicate over a per-instance scalar),
 * consumed two ways:
 *  - as decoration: the selected instances get a highlight style, compiled to
 *    the existing per-instance decoration system (no new render machinery);
 *  - as addressability: `pick-at` reports selection membership, and the CLI
 *    resolves selection names to component + instances for framing / queries.
 *
 * Selections live in `$state.selections`, so they sync Python<->JS, persist
 * into `.colight` artifacts, and either side can mutate them. This module is
 * GPU-free and unit-testable.
 */

import { ComponentConfig, Decoration } from "./components";

/** A selection's mask source: explicit instances or a threshold predicate. */
export type SelectionSource =
  | { instances: number[] }
  | { values_ref?: string; values?: number[]; min?: number; max?: number };

/** A named selection as it lives in `$state.selections`. */
export interface Selection {
  /** Compiled-component index the selection targets. */
  component: number;
  /** How to resolve the masked instances. */
  source: SelectionSource;
  /** Decoration style, or "default" for the built-in highlight. */
  style?: SelectionStyle | "default";
}

/** Styling applied to selected instances (a subset of Decoration). */
export interface SelectionStyle {
  color?: [number, number, number];
  alpha?: number;
  scale?: number;
  outline?: boolean;
  outlineColor?: [number, number, number];
  outlineWidth?: number;
}

/** The built-in highlight used when a selection's style is "default". */
export const DEFAULT_SELECTION_STYLE: SelectionStyle = {
  color: [1.0, 0.85, 0.2],
  outline: true,
  outlineColor: [1.0, 0.85, 0.2],
  outlineWidth: 3,
};

/** `$state.selections` shape: name -> Selection. */
export type Selections = Record<string, Selection>;

/**
 * Resolves a selection source to the sorted, de-duplicated list of instance
 * indices it selects, given the target component. This is the SAME mask logic
 * filtering uses: an explicit index list, or a threshold predicate over a
 * per-instance scalar (NaN never matches).
 *
 * @param source  The selection's mask source.
 * @param component The target compiled component (for values_ref lookup).
 * @param elementCount Number of elements in the component (bounds the mask).
 */
export function resolveSelectionInstances(
  source: SelectionSource,
  component: ComponentConfig | undefined,
  elementCount: number,
): number[] {
  if ("instances" in source && source.instances) {
    // Explicit list: keep in-range, unique, sorted.
    const seen = new Set<number>();
    for (const i of source.instances) {
      if (Number.isInteger(i) && i >= 0 && i < elementCount) seen.add(i);
    }
    return Array.from(seen).sort((a, b) => a - b);
  }

  // Threshold predicate over a per-instance scalar. The values come either
  // inline (source.values) or by name from the component (source.values_ref).
  const pred = source as {
    values_ref?: string;
    values?: number[];
    min?: number;
    max?: number;
  };
  let values: ArrayLike<number> | undefined = pred.values;
  if (!values && pred.values_ref && component) {
    values = (component as any)[pred.values_ref] as ArrayLike<number>;
  }
  if (!values) return [];

  const min = pred.min ?? -Infinity;
  const max = pred.max ?? Infinity;
  const out: number[] = [];
  const n = Math.min(values.length, elementCount);
  for (let i = 0; i < n; i++) {
    const v = values[i];
    // NaN never matches (v === v is false for NaN).
    if (v === v && v >= min && v <= max) out.push(i);
  }
  return out;
}

/**
 * Turns a selection's style into a Decoration for the given instance indexes.
 * "default" (or an absent style) uses {@link DEFAULT_SELECTION_STYLE}.
 */
export function selectionToDecoration(
  selection: Selection,
  indexes: number[],
): Decoration {
  const style =
    selection.style === "default" || selection.style === undefined
      ? DEFAULT_SELECTION_STYLE
      : selection.style;
  return { indexes, ...style };
}

/**
 * Compiles `$state.selections` into per-component decorations, appending them
 * to each targeted component's `decorations` (so they render through the
 * existing decoration pipeline — no new render machinery). Returns metadata for
 * agent-facing reporting.
 *
 * Later selections win where they overlap (decorations apply in order).
 */
export function applySelections(
  components: ComponentConfig[],
  selections: Selections | undefined,
  getElementCount: (component: ComponentConfig) => number,
): SelectionReport[] {
  if (!selections) return [];
  const report: SelectionReport[] = [];

  for (const [name, selection] of Object.entries(selections)) {
    const component = components[selection.component];
    if (!component) continue;
    const elementCount = getElementCount(component);
    const indexes = resolveSelectionInstances(
      selection.source,
      component,
      elementCount,
    );
    report.push({
      name,
      component: selection.component,
      type: component.type,
      count: indexes.length,
      predicate: !("instances" in selection.source),
      indexes,
    });
    if (indexes.length === 0) continue;
    const decoration = selectionToDecoration(selection, indexes);
    const existing = (component as any).decorations as Decoration[] | undefined;
    (component as any).decorations = existing
      ? [...existing, decoration]
      : [decoration];
  }

  return report;
}

/** Agent-facing summary of a resolved selection (inspect / pick-at). */
export interface SelectionReport {
  name: string;
  component: number;
  type: string;
  count: number;
  /** True when resolved from a threshold predicate (vs an explicit list). */
  predicate: boolean;
  /** The resolved instance indexes (used for membership + framing). */
  indexes: number[];
}

/**
 * Names of the selections that contain a given (component, instance) hit.
 * Used to add `selections: [...]` to pick-at results.
 */
export function selectionsForInstance(
  reports: SelectionReport[],
  componentIndex: number,
  instanceIndex: number,
): string[] {
  const names: string[] = [];
  for (const r of reports) {
    if (r.component !== componentIndex) continue;
    if (r.indexes.includes(instanceIndex)) names.push(r.name);
  }
  return names;
}
