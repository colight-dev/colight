/**
 * @module arrayUtils
 * @description Array coercion utilities for scene3d.
 *
 * This module contains low-level array conversion functions with no
 * dependencies on other scene3d modules to avoid circular imports.
 */

/**
 * Coerce a value to Float32Array if it's an array-like type.
 * Handles regular arrays (including nested) and other TypedArrays.
 *
 * Nested rows may themselves be typed arrays: a multi-dimensional `$state`
 * ndarray arrives as an Array of TypedArray rows (see `reshapeArray` in
 * `binary.ts`). `Array.prototype.flat` does not flatten those — a TypedArray is
 * not an Array — and each row would then coerce to NaN, so rows are flattened
 * explicitly here.
 */
export function coerceToFloat32(value: unknown): Float32Array | unknown {
  if (Array.isArray(value)) {
    return new Float32Array(flattenRows(value));
  }
  if (ArrayBuffer.isView(value)) {
    if (value instanceof Float32Array) {
      return value;
    }
    if (value instanceof DataView) {
      if (value.byteOffset % 4 !== 0 || value.byteLength % 4 !== 0) {
        console.warn(
          "[scene3d] DataView is not 4-byte aligned; leaving it as-is.",
          value,
        );
        return value;
      }
      console.warn(
        "[scene3d] Interpreting DataView bytes as Float32Array values.",
        value,
      );
      return new Float32Array(
        value.buffer,
        value.byteOffset,
        value.byteLength / 4,
      );
    }
    // Treat typed arrays as element values, not raw byte buffers.
    return new Float32Array(value as ArrayLike<number>);
  }
  return value;
}

/**
 * Flattens one level of an array whose rows may be plain arrays or typed
 * arrays, into a flat array of numbers.
 *
 * Deeper nesting (a 3-D ndarray) is handled recursively; scalar elements pass
 * through, so a flat array of numbers is returned unchanged.
 */
function flattenRows(value: readonly unknown[]): number[] {
  const out: number[] = [];
  for (const row of value) {
    if (ArrayBuffer.isView(row) && !(row instanceof DataView)) {
      const typed = row as unknown as ArrayLike<number>;
      for (let i = 0; i < typed.length; i++) out.push(typed[i]);
    } else if (Array.isArray(row)) {
      out.push(...flattenRows(row));
    } else {
      out.push(row as number);
    }
  }
  return out;
}
