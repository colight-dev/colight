/**
 * @module legend
 * @description Colormap legends for scene3d.
 *
 * Components carrying a `color_by` spec (attached by the Python side, see
 * colight/colormaps.py) get a legend rendered as a DOM overlay docked in a
 * corner of the scene container — NOT painted into the WebGPU canvas. The
 * overlay is part of the captured DOM, so it appears in screenshots, and
 * each legend card exposes its spec via a `data-colight-legend` attribute
 * so `colight screenshot --json` can report what the colors encode.
 */

import React from "react";
import { tw } from "../utils";

// =============================================================================
// Types
// =============================================================================

export type LegendPosition =
  | "top-left"
  | "top-right"
  | "bottom-left"
  | "bottom-right";

/**
 * Colormap spec attached to a component as `color_by` (JSON produced by
 * colight/colormaps.py — snake_case-free single-word keys).
 */
export interface ColorByMeta {
  /** Colormap name (e.g. "viridis", "tab10"). */
  cmap: string;
  /** True for categorical palettes, false for continuous ramps. */
  categorical?: boolean;
  /** What the colors encode (legend title). */
  label?: string;
  /** Continuous only: [min, max] the ramp spans. */
  domain?: [number, number];
  /** Continuous only: RGB anchor colors of the ramp, in [0, 1]. */
  stops?: number[][];
  /** Categorical only: one RGB swatch per category, in [0, 1]. */
  colors?: number[][];
  /** Categorical only: display names for codes 0..K-1. */
  categories?: string[];
  /** False suppresses the in-scene legend. */
  legend?: boolean;
  /** Dock corner (default "top-right"). */
  position?: LegendPosition;
}

// =============================================================================
// Formatting helpers
// =============================================================================

/** Compact tick label: trims float noise, keeps ~4 significant digits. */
export function formatTick(value: number): string {
  if (!isFinite(value)) return String(value);
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 10000 || abs < 0.001) return value.toExponential(1);
  const text = Number(value.toPrecision(4)).toString();
  return text;
}

function cssColor(rgb: number[]): string {
  const [r, g, b] = rgb;
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

/** CSS linear-gradient through a ramp's anchor colors. */
export function gradientCss(stops: number[][]): string {
  const parts = stops.map((rgb, i) => {
    const pct = (i / Math.max(1, stops.length - 1)) * 100;
    return `${cssColor(rgb)} ${pct.toFixed(1)}%`;
  });
  return `linear-gradient(to right, ${parts.join(", ")})`;
}

/**
 * Machine-readable payload for one legend, serialized into the
 * `data-colight-legend` attribute (`colight screenshot --json` collects
 * these from the DOM). Lean on purpose: no color tables.
 */
export function legendReport(
  spec: ColorByMeta,
  component?: { index: number; type: string },
): Record<string, unknown> {
  const report: Record<string, unknown> = {
    cmap: spec.cmap,
    categorical: !!spec.categorical,
  };
  if (spec.label !== undefined) report.label = spec.label;
  if (spec.domain !== undefined) report.domain = spec.domain;
  if (spec.categories !== undefined) report.categories = spec.categories;
  if (component) {
    report.component = component.index;
    report.type = component.type;
  }
  return report;
}

// =============================================================================
// Legend card
// =============================================================================

interface LegendCardProps {
  spec: ColorByMeta;
  /** Compiled component this legend belongs to (in-scene legends only). */
  component?: { index: number; type: string };
}

/** One legend card: gradient bar + ticks (continuous) or swatch list. */
function LegendCard({ spec, component }: LegendCardProps) {
  const report = JSON.stringify(legendReport(spec, component));

  const body = spec.categorical ? (
    <div className={tw("flex flex-col gap-[3px]")}>
      {(spec.colors ?? []).map((rgb, i) => (
        <div key={i} className={tw("flex items-center gap-[6px]")}>
          <span
            className={tw("inline-block w-[12px] h-[12px] rounded-[2px]")}
            style={{
              background: cssColor(rgb),
              border: "1px solid rgba(0,0,0,0.25)",
            }}
          />
          <span>{spec.categories?.[i] ?? String(i)}</span>
        </div>
      ))}
    </div>
  ) : (
    <div>
      <div
        className={tw("h-[10px] w-[160px] rounded-[2px]")}
        style={{
          background: spec.stops ? gradientCss(spec.stops) : "#888",
          border: "1px solid rgba(0,0,0,0.25)",
        }}
      />
      {spec.domain && (
        <div
          className={tw("flex justify-between w-[160px] mt-[1px] tabular-nums")}
        >
          <span>{formatTick(spec.domain[0])}</span>
          <span>{formatTick((spec.domain[0] + spec.domain[1]) / 2)}</span>
          <span>{formatTick(spec.domain[1])}</span>
        </div>
      )}
    </div>
  );

  return (
    <div
      data-colight-legend={report}
      className={tw(
        "rounded-md px-[8px] py-[6px] text-[11px] leading-[1.35] text-gray-900",
      )}
      style={{
        background: "rgba(255,255,255,0.88)",
        border: "1px solid rgba(0,0,0,0.15)",
        boxShadow: "0 1px 3px rgba(0,0,0,0.12)",
        pointerEvents: "none",
      }}
    >
      {spec.label && (
        <div className={tw("font-medium mb-[3px]")}>{spec.label}</div>
      )}
      {body}
    </div>
  );
}

/**
 * Standalone legend component, renderable anywhere in a layout
 * (Python: `scene3d.Legend(...)` -> JSRef "scene3d.Legend").
 */
export function Legend({ spec }: { spec: ColorByMeta }) {
  return (
    <div className={tw("inline-block my-[4px]")}>
      <LegendCard spec={spec} />
    </div>
  );
}

// =============================================================================
// Scene overlay
// =============================================================================

const POSITION_CLASSES: Record<LegendPosition, string> = {
  "top-left": "top-2 left-2 items-start",
  "top-right": "top-2 right-2 items-end",
  "bottom-left": "bottom-2 left-2 items-start",
  "bottom-right": "bottom-2 right-2 items-end",
};

export interface SceneLegendEntry {
  spec: ColorByMeta;
  component: { index: number; type: string };
}

/**
 * Collects `color_by` specs from compiled component configs.
 * Index positions match the compiled components array (the same index
 * space the pick/coverage tooling reports). Components sharing an
 * identical spec (e.g. several layers of one attribute) produce a single
 * legend, attributed to the first such component.
 */
export function collectLegendEntries(
  components: ({ type: string; color_by?: ColorByMeta } | null | undefined)[],
): SceneLegendEntry[] {
  const entries: SceneLegendEntry[] = [];
  const seen = new Set<string>();
  components.forEach((component, index) => {
    const spec = component?.color_by;
    if (!spec || spec.legend === false) return;
    const key = JSON.stringify([
      spec.cmap,
      spec.domain,
      spec.label,
      spec.categorical,
      spec.categories,
      spec.position,
    ]);
    if (seen.has(key)) return;
    seen.add(key);
    entries.push({ spec, component: { index, type: component!.type } });
  });
  return entries;
}

/**
 * Docked legend overlay for a scene. Renders one stack of legend cards per
 * occupied corner (default "top-right"), absolutely positioned over the
 * canvas inside the scene container.
 */
export function SceneLegends({ entries }: { entries: SceneLegendEntry[] }) {
  if (entries.length === 0) return null;

  const byPosition = new Map<LegendPosition, SceneLegendEntry[]>();
  for (const entry of entries) {
    const position = entry.spec.position ?? "top-right";
    const group = byPosition.get(position) ?? [];
    group.push(entry);
    byPosition.set(position, group);
  }

  return (
    <>
      {[...byPosition.entries()].map(([position, group]) => (
        <div
          key={position}
          className={tw(
            `absolute z-10 flex flex-col gap-2 ${POSITION_CLASSES[position]}`,
          )}
          style={{ pointerEvents: "none" }}
        >
          {group.map((entry) => (
            <LegendCard
              key={entry.component.index}
              spec={entry.spec}
              component={entry.component}
            />
          ))}
        </div>
      ))}
    </>
  );
}
