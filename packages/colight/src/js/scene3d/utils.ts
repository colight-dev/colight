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
