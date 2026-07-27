/**
 * @module channels
 * @description Client-side resampling of declared value channels.
 *
 * A channel ships its sample rows once and names the `$state` key that indexes
 * them. `resampleChannel` turns (parameter value, samples, rule) into the value
 * a prop takes right now. It is pure: no `$state` access, no DOM, no caching —
 * the AST evaluator has already resolved the parameter to a number by the time
 * it is called.
 */

/** One channel's declaration, as it arrives from the AST evaluator. */
export interface ChannelConfig {
  /** The `$state` key this channel is indexed by (declarative, for inspect). */
  parameter: string;
  /** The parameter's current value, resolved during AST evaluation. */
  value: number;
  /** (N,) strictly increasing sample coordinates. */
  at: ArrayLike<number>;
  /**
   * Sample rows. A 1-D declaration arrives as a flat typed array (one scalar
   * per sample); a 2-D one arrives as an array of per-row typed arrays.
   */
  values: ArrayLike<number> | ArrayLike<ArrayLike<number>>;
  /** How values between samples are produced. */
  rule: "nearest" | "step" | "linear" | "qlerp";
}

/** The resampled value: a scalar channel yields a number, a wide one a row. */
export type ChannelValue = number | number[] | Float32Array;

/**
 * Index of the last sample at or before `x`, clamped into `[0, n - 2]`.
 *
 * Returns the lower end of the bracketing interval, so callers can always read
 * both `i` and `i + 1`. `n === 1` is handled by the caller.
 */
function bracket(at: ArrayLike<number>, x: number, n: number): number {
  let lo = 0;
  let hi = n - 1;
  // Invariant: at[lo] <= x < at[hi] once the loop settles (x pre-clamped).
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (at[mid] <= x) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/** A row of a channel's values, whatever nesting the decode produced. */
function rowAt(
  values: ChannelConfig["values"],
  i: number,
): number | ArrayLike<number> {
  return (values as ArrayLike<number | ArrayLike<number>>)[i] as
    | number
    | ArrayLike<number>;
}

/** A fresh copy of a row: identity change is the contents-write signal. */
function copyRow(row: number | ArrayLike<number>): ChannelValue {
  if (typeof row === "number") return row;
  const width = row.length;
  if (width <= 4) {
    const out = new Array<number>(width);
    for (let k = 0; k < width; k++) out[k] = row[k];
    return out;
  }
  const out = new Float32Array(width);
  for (let k = 0; k < width; k++) out[k] = row[k];
  return out;
}

/** Elementwise `a + (b - a) * t`, into a fresh row. */
function lerpRow(
  a: number | ArrayLike<number>,
  b: number | ArrayLike<number>,
  t: number,
): ChannelValue {
  if (typeof a === "number" || typeof b === "number") {
    const av = a as number;
    const bv = b as number;
    return av + (bv - av) * t;
  }
  const width = a.length;
  if (width <= 4) {
    const out = new Array<number>(width);
    for (let k = 0; k < width; k++) out[k] = a[k] + (b[k] - a[k]) * t;
    return out;
  }
  const out = new Float32Array(width);
  for (let k = 0; k < width; k++) out[k] = a[k] + (b[k] - a[k]) * t;
  return out;
}

/**
 * Normalized quaternion lerp between two xyzw rows, shortest path.
 *
 * Negating the far quaternion when the pair is antipodal keeps the
 * interpolation on the short arc; normalizing afterwards puts the result back
 * on the unit sphere. Cheaper than slerp and, for samples spaced a few degrees
 * apart, indistinguishable from it.
 */
function qlerpRow(
  a: ArrayLike<number>,
  b: ArrayLike<number>,
  t: number,
): number[] {
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const sign = dot < 0 ? -1 : 1;
  const out = new Array<number>(4);
  for (let k = 0; k < 4; k++) {
    out[k] = a[k] + (sign * b[k] - a[k]) * t;
  }
  const len = Math.hypot(out[0], out[1], out[2], out[3]);
  if (len > 0) {
    for (let k = 0; k < 4; k++) out[k] /= len;
  }
  return out;
}

/**
 * Resample a declared channel at its parameter's current value.
 *
 * The parameter is clamped to the sample domain, so a slider that runs past
 * either end holds the end sample rather than extrapolating. Multi-element
 * results are always a freshly allocated row: downstream geometry keys its
 * contents-changed check on array identity.
 *
 * Scalar channels return a plain number, which the AST evaluator passes
 * through unchanged: `evaluate` constructs a resolved reference only when it is
 * a genuine ES class (see `isClassConstructor` in eval.js), so a registered
 * function's return value is never discarded. The arrow form here is style, not
 * a workaround.
 *
 * @param config - The channel declaration plus the resolved parameter value.
 * @returns The value at the parameter: a number for scalar channels, a fresh
 *   array for vector, quaternion and wide rows.
 */
export const resampleChannel = (config: ChannelConfig): ChannelValue => {
  const { at, values, rule } = config;
  const n = at.length;
  if (n === 0) {
    throw new Error(
      `channel "${config.parameter}" has no samples to resample from`,
    );
  }
  if (n === 1) {
    return copyRow(rowAt(values, 0));
  }

  const raw = config.value;
  const x = Math.min(
    Math.max(typeof raw === "number" ? raw : at[0], at[0]),
    at[n - 1],
  );

  const i = bracket(at, x, n);
  const x0 = at[i];
  const x1 = at[i + 1];
  const span = x1 - x0;
  // Coincident samples would divide by zero; hold the lower one.
  const t = span > 0 ? (x - x0) / span : 0;

  switch (rule) {
    case "step":
      // At (or past) the top of the domain the last sample is the one in
      // force, not the interval it opens.
      return copyRow(rowAt(values, x >= at[n - 1] ? n - 1 : i));
    case "nearest":
      return copyRow(rowAt(values, t < 0.5 ? i : i + 1));
    case "qlerp":
      return qlerpRow(
        rowAt(values, i) as ArrayLike<number>,
        rowAt(values, i + 1) as ArrayLike<number>,
        t,
      );
    case "linear":
    default:
      return lerpRow(rowAt(values, i), rowAt(values, i + 1), t);
  }
};
