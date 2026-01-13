/**
 * @module arrayUtils
 * @description Array coercion utilities for scene3d.
 *
 * This module contains low-level array conversion functions with no
 * dependencies on other scene3d modules to avoid circular imports.
 */

import { isNdArray } from "@colight/serde";

/**
 * Coerce a value to Float32Array if it's an array-like type.
 * Handles NdArrayView, regular arrays, and other TypedArrays.
 */
export function coerceToFloat32(value: unknown): Float32Array | unknown {
  if (isNdArray(value)) {
    const flat = value.flat;
    return flat instanceof Float32Array
      ? flat
      : new Float32Array(flat as ArrayLike<number>);
  }
  if (Array.isArray(value)) {
    const flattened = value.flat ? value.flat() : value;
    return new Float32Array(flattened as number[]);
  }
  if (ArrayBuffer.isView(value) && !(value instanceof Float32Array)) {
    return new Float32Array(value.buffer);
  }
  return value;
}
