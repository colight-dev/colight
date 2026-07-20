/**
 * Utility functions for scene3d.
 * These are self-contained to avoid external dependencies.
 */

/**
 * Throttle function execution to at most once per `limit` milliseconds.
 */
export function throttle<T extends (...args: any[]) => void>(
  func: T,
  limit: number,
): (...args: Parameters<T>) => void {
  let inThrottle = false;
  return function (this: any, ...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
}

/**
 * Deep equality check that treats TypedArrays as equal if they have the same contents.
 * Regular arrays are compared recursively, objects by their enumerable properties.
 */
export function deepEqualModuloTypedArrays(a: any, b: any): boolean {
  // Identity check handles primitives and references
  if (a === b) return true;

  // If either is null/undefined or not an object, we already know they're not equal
  if (!a || !b || typeof a !== "object" || typeof b !== "object") {
    return false;
  }

  // Handle arrays
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      return false;
    }
    for (let i = 0; i < a.length; i++) {
      if (!deepEqualModuloTypedArrays(a[i], b[i])) {
        return false;
      }
    }
    return true;
  }

  // Handle TypedArrays - treat as equal if same type and contents
  if (ArrayBuffer.isView(a) && ArrayBuffer.isView(b)) {
    // Different types of typed arrays are not equal
    if (a.constructor !== b.constructor) return false;
    // Compare as regular arrays for typed arrays
    const aArr = a as unknown as ArrayLike<number>;
    const bArr = b as unknown as ArrayLike<number>;
    if (aArr.length !== bArr.length) return false;
    for (let i = 0; i < aArr.length; i++) {
      if (aArr[i] !== bArr[i]) return false;
    }
    return true;
  }

  // Handle plain objects
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) {
    return false;
  }

  for (const key of keysA) {
    if (!keysB.includes(key)) {
      return false;
    }
    if (!deepEqualModuloTypedArrays(a[key], b[key])) {
      return false;
    }
  }

  return true;
}

/**
 * Keys that carry filter *thresholds* (not per-instance data). Two compiled
 * scenes that differ only in these are handled by the light filterParams-only
 * render path, so the large instance buffers are never re-uploaded.
 */
const FILTER_THRESHOLD_KEYS = new Set(["filter_by", "_filterIndex"]);

/**
 * Deep-equal two component arrays while ignoring per-component filter
 * *thresholds* (`filter_by`, `_filterIndex`). The per-instance filter values
 * (`_filterValues`) are NOT ignored: if those differ, the instance data really
 * changed and the heavy rebuild path is required.
 *
 * Returns true when the only difference between the two component lists is the
 * filter thresholds — the signal to take the cheap uniform-only render path.
 */
export function componentsEqualIgnoringFilter(a: any, b: any): boolean {
  if (a === b) return true;
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    const ca = a[i];
    const cb = b[i];
    if (ca === cb) continue;
    if (!ca || !cb || typeof ca !== "object" || typeof cb !== "object") {
      if (ca !== cb) return false;
      continue;
    }
    const keys = new Set([...Object.keys(ca), ...Object.keys(cb)]);
    for (const key of keys) {
      if (FILTER_THRESHOLD_KEYS.has(key)) continue;
      if (!deepEqualModuloTypedArrays(ca[key], cb[key])) return false;
    }
  }
  return true;
}

/**
 * Copy n elements from source array starting at sourceI to out array starting at outI.
 */
export function acopy(
  source: ArrayLike<number>,
  sourceI: number,
  out: ArrayLike<number> & { [n: number]: number },
  outI: number,
  n: number,
): void {
  for (let i = 0; i < n; i++) {
    out[outI + i] = source[sourceI + i];
  }
}
