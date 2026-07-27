/**
 * Canonical `.colight` dtype names and their mapping to JS typed arrays.
 *
 * Per spec §3.3 the buffer holds elements little-endian, densely packed, in
 * C (row-major) order, and the `dtype` string is always one of the canonical
 * names below — never a byte-order-qualified spelling like `">f4"`.
 */

/** The canonical dtype names a `.colight` file may carry. */
export type Dtype =
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "float32"
  | "float64"
  | "bool";

/** Element size in bytes for each canonical dtype. */
export const DTYPE_BYTES: Readonly<Record<Dtype, number>> = Object.freeze({
  int8: 1,
  int16: 2,
  int32: 4,
  int64: 8,
  uint8: 1,
  uint16: 2,
  uint32: 4,
  uint64: 8,
  float32: 4,
  float64: 8,
  bool: 1,
});

const DTYPE_NAMES = Object.keys(DTYPE_BYTES) as Dtype[];

/** True if `name` is a canonical `.colight` dtype. */
export function isDtype(name: string): name is Dtype {
  return Object.prototype.hasOwnProperty.call(DTYPE_BYTES, name);
}

/**
 * Throws unless `name` is a canonical dtype. Readers and writers must fail
 * loudly on unknown dtypes rather than guessing an element type (spec §3.3).
 */
export function assertDtype(name: string): Dtype {
  if (!isDtype(name)) {
    throw new Error(
      `Unknown .colight dtype ${JSON.stringify(name)}. ` +
        `Supported dtypes: ${DTYPE_NAMES.join(", ")}.`,
    );
  }
  return name;
}

/** Byte length of an array with the given `dtype` and `shape`. */
export function byteLengthFor(dtype: Dtype, shape: readonly number[]): number {
  let elements = 1;
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0) {
      throw new Error(
        `ndarray shape must be non-negative integers, got ${JSON.stringify(shape)}.`,
      );
    }
    elements *= dim;
  }
  return elements * DTYPE_BYTES[dtype];
}

/**
 * Infers the canonical dtype of a JS typed array.
 *
 * `BigInt64Array`/`BigUint64Array` map to `int64`/`uint64`. There is no JS
 * typed array whose canonical dtype is `bool`; bool arrays must be declared
 * explicitly (see {@link boolArray}).
 */
export function dtypeOfTypedArray(array: ArrayBufferView): Dtype {
  if (array instanceof Int8Array) return "int8";
  if (array instanceof Int16Array) return "int16";
  if (array instanceof Int32Array) return "int32";
  if (array instanceof Uint8Array || array instanceof Uint8ClampedArray)
    return "uint8";
  if (array instanceof Uint16Array) return "uint16";
  if (array instanceof Uint32Array) return "uint32";
  if (array instanceof Float32Array) return "float32";
  if (array instanceof Float64Array) return "float64";
  if (typeof BigInt64Array !== "undefined" && array instanceof BigInt64Array)
    return "int64";
  if (typeof BigUint64Array !== "undefined" && array instanceof BigUint64Array)
    return "uint64";
  throw new Error(
    `Cannot infer a .colight dtype from ${array.constructor?.name ?? "value"}. ` +
      `Pass an explicit dtype via ndarray({ dtype, shape, data }).`,
  );
}
