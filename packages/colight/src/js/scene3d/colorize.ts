/**
 * @module colorize
 * @description Client-side recoloring for switchable color channels.
 *
 * A component may carry `color_channels` (each channel ships its raw `values`
 * once, plus a compact `colorizer` produced by colight/colormaps.py) and an
 * `active_channel` name. When the active channel changes (a `$state`-driven
 * dropdown), we recolor the ACTIVE channel's values into a fresh per-instance
 * `colors` buffer here — no Python round-trip, no colormap reimplementation:
 * continuous channels sample a 256-entry RGB LUT, categorical channels read a
 * resolved category table. Geometry/instance data is untouched, so a switch
 * re-uploads only the colors buffer (see impl3d render path).
 */

import { ColorByMeta, CategoryEntry, FallbackEntry } from "./legend";

/** Continuous colorizer: sample `lut` across `domain`. */
export interface ContinuousColorizer {
  kind: "continuous";
  /** (K, 3) RGB lookup table in [0, 1]. */
  lut: number[][];
  /** [min, max] the LUT spans; null derives per-channel (already resolved). */
  domain?: [number, number] | null;
  /** RGB for NaN/non-finite values. */
  nan_color?: number[];
}

/** Categorical colorizer: declared table or ordinal palette. */
export interface CategoricalColorizer {
  kind: "categorical";
  /** First-class declared table ({value, label, color}). */
  categories?: CategoryEntry[];
  /** Declared-table fallback for unmatched/NaN. */
  fallback?: FallbackEntry | null;
  /** Ordinal palette swatches (categories declared as strings). */
  colors?: number[][];
  /** Ordinal-mode NaN/invalid color. */
  nan_color?: number[];
}

export type Colorizer = ContinuousColorizer | CategoricalColorizer;

/** One named channel as it reaches JS. */
export interface ColorChannel {
  label: string;
  legend: ColorByMeta;
  colorizer: Colorizer;
  /** Per-instance raw values (Float32Array or plain array). */
  values: ArrayLike<number>;
  count: number;
}

/** A component's channel bundle. */
export interface ColorChannels {
  [name: string]: ColorChannel;
}

const DEFAULT_NAN: [number, number, number] = [0.5, 0.5, 0.5];

/**
 * Resolve a channel's raw `values` into a flat Float32 RGB buffer using its
 * colorizer. Mirrors colight.colormaps.apply_colormap exactly (same clamp,
 * same LUT sampling, same fallback rules) so JS-recolored colors match what
 * Python would have baked.
 */
export function colorizeChannel(channel: ColorChannel): Float32Array {
  const { values, colorizer } = channel;
  const n = values.length;
  const out = new Float32Array(n * 3);

  if (colorizer.kind === "continuous") {
    const lut = colorizer.lut;
    const steps = lut.length - 1;
    const nan = colorizer.nan_color ?? DEFAULT_NAN;
    let lo = 0;
    let hi = 1;
    if (colorizer.domain) {
      [lo, hi] = colorizer.domain;
    } else {
      // Derive from finite values (matches resolve_domain's fallback).
      lo = Infinity;
      hi = -Infinity;
      for (let i = 0; i < n; i++) {
        const v = values[i];
        if (Number.isFinite(v)) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
      if (!(hi > lo)) {
        lo = 0;
        hi = 1;
      }
    }
    const span = hi - lo || 1;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) {
        out[i * 3] = nan[0];
        out[i * 3 + 1] = nan[1];
        out[i * 3 + 2] = nan[2];
        continue;
      }
      const t = Math.min(1, Math.max(0, (v - lo) / span));
      // Nearest-neighbor LUT read; the LUT is dense (256 entries) so this
      // matches the Python piecewise-linear ramp within display tolerance.
      const idx = Math.round(t * steps);
      const rgb = lut[idx];
      out[i * 3] = rgb[0];
      out[i * 3 + 1] = rgb[1];
      out[i * 3 + 2] = rgb[2];
    }
    return out;
  }

  // Categorical.
  const nan = colorizer.nan_color ?? DEFAULT_NAN;
  if (colorizer.categories) {
    // Declared table: map value code -> color; unmatched/NaN -> fallback.
    const lookup = new Map<number, number[]>();
    for (const cat of colorizer.categories) lookup.set(cat.value, cat.color);
    const fb = colorizer.fallback?.color ?? nan;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      const color =
        Number.isFinite(v) && lookup.has(Math.trunc(v))
          ? lookup.get(Math.trunc(v))!
          : fb;
      out[i * 3] = color[0];
      out[i * 3 + 1] = color[1];
      out[i * 3 + 2] = color[2];
    }
    return out;
  }

  // Ordinal palette: code i -> colors[i % len]; NaN/negative -> nan_color.
  const palette = colorizer.colors ?? [];
  const len = palette.length || 1;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (!Number.isFinite(v) || v < 0) {
      out[i * 3] = nan[0];
      out[i * 3 + 1] = nan[1];
      out[i * 3 + 2] = nan[2];
      continue;
    }
    const rgb = palette[Math.trunc(v) % len] ?? nan;
    out[i * 3] = rgb[0];
    out[i * 3 + 1] = rgb[1];
    out[i * 3 + 2] = rgb[2];
  }
  return out;
}

/**
 * The dereferenced value of an instance's channel, as it should be reported by
 * pick-at: categorical channels return the category/fallback LABEL (string);
 * continuous channels return the raw numeric value.
 */
export function channelValueAt(
  channel: ColorChannel,
  instance: number,
): number | string | null {
  const v = channel.values[instance];
  if (v === undefined) return null;
  const c = channel.colorizer;
  if (c.kind === "categorical") {
    if (c.categories) {
      if (Number.isFinite(v)) {
        const match = c.categories.find((cat) => cat.value === Math.trunc(v));
        if (match) return match.label;
      }
      return c.fallback?.label ?? null;
    }
    // Ordinal palette: no declared labels beyond the legend's category strings.
    const labels = channel.legend.categories as string[] | undefined;
    if (Number.isFinite(v) && labels && v >= 0) {
      return labels[Math.trunc(v)] ?? Math.trunc(v);
    }
    return Number.isFinite(v) ? v : null;
  }
  return Number.isFinite(v) ? v : null;
}

/**
 * Pick the active channel name for a component: the resolved `active_channel`
 * (a string post-$state-eval) when it names a real channel, else the first
 * declared channel (stable insertion order).
 */
export function activeChannelName(
  channels: ColorChannels,
  active: unknown,
): string | null {
  const names = Object.keys(channels);
  if (names.length === 0) return null;
  if (typeof active === "string" && active in channels) return active;
  return names[0];
}

/**
 * Recolor a component in place for its active channel: rewrites `colors` (flat
 * Float32 RGB) and `color_by` (the active channel's legend). No-op when the
 * component has no `color_channels`. Returns the active channel name (or null).
 *
 * Called during compilation so a channel switch (new `active_channel`) yields a
 * component that differs only in `colors`/`color_by`/`active_channel` — the
 * render path re-uploads the colors buffer without rebuilding geometry.
 */
export function applyActiveChannel(component: any): string | null {
  const channels = component.color_channels as ColorChannels | undefined;
  if (!channels) return null;
  const name = activeChannelName(channels, component.active_channel);
  if (!name) return null;
  const channel = channels[name];
  component.colors = colorizeChannel(channel);
  component.color_by = channel.legend;
  component._activeChannel = name;
  return name;
}
