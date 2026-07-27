/**
 * Buffer-carrying values that may appear anywhere inside a `.colight` JSON
 * payload (spec §3.3).
 *
 * These are placeholders: the writer walks the payload, hoists each one's bytes
 * into the entry's binary section, and replaces the placeholder with the JSON
 * envelope that references the assigned buffer index.
 */

import {
  DTYPE_BYTES,
  type Dtype,
  assertDtype,
  byteLengthFor,
  dtypeOfTypedArray,
} from "./dtypes.js";

/** Raw bytes, serialized as `{"__buffer_index__": i}`. */
export class RawBuffer {
  readonly bytes: Uint8Array;

  constructor(bytes: Uint8Array | ArrayBuffer | ArrayBufferView) {
    this.bytes = toBytes(bytes);
  }
}

/** Wraps bytes so the writer stores them as a raw buffer reference. */
export function rawBuffer(
  bytes: Uint8Array | ArrayBuffer | ArrayBufferView,
): RawBuffer {
  return new RawBuffer(bytes);
}

/**
 * An n-dimensional array, serialized as the `{"__type__":"ndarray", ...}`
 * envelope of spec §3.3 plus a little-endian, densely packed, C-order buffer.
 */
export class NDArray {
  readonly dtype: Dtype;
  readonly shape: readonly number[];
  readonly bytes: Uint8Array;

  constructor(dtype: Dtype, shape: readonly number[], bytes: Uint8Array) {
    this.dtype = dtype;
    this.shape = shape;
    this.bytes = bytes;
    const expected = byteLengthFor(dtype, shape);
    if (bytes.byteLength !== expected) {
      throw new Error(
        `ndarray buffer length mismatch: dtype ${dtype} with shape ` +
          `[${shape.join(", ")}] needs ${expected} bytes, got ${bytes.byteLength}.`,
      );
    }
  }
}

/** Options accepted by {@link ndarray} when the data is not a typed array. */
export interface NDArraySpec {
  dtype: Dtype | string;
  /** Defaults to `[n]` where `n` is the element count implied by `data`. */
  shape?: readonly number[];
  data: ArrayBufferView | ArrayBuffer | readonly number[] | readonly bigint[];
}

/**
 * Builds an {@link NDArray} from a typed array, an `ArrayBuffer`, or a plain
 * numeric array.
 *
 * A typed array's dtype and length are inferred; its bytes are copied and
 * normalized to little-endian. Pass `shape` for anything but a 1-D array. The
 * spec forbids 0-d ndarrays — serialize scalars as plain JSON numbers instead.
 */
export function ndarray(
  input: ArrayBufferView | NDArraySpec,
  shape?: readonly number[],
): NDArray {
  if (ArrayBuffer.isView(input)) {
    const dtype = dtypeOfTypedArray(input);
    const elements = input.byteLength / DTYPE_BYTES[dtype];
    return new NDArray(
      dtype,
      checkShape(shape ?? [elements]),
      littleEndianBytes(input, dtype),
    );
  }

  const dtype = assertDtype(input.dtype);
  const data = input.data;
  if (ArrayBuffer.isView(data) || data instanceof ArrayBuffer) {
    const bytes = ArrayBuffer.isView(data)
      ? littleEndianBytes(data, dtype)
      : new Uint8Array(data.slice(0));
    const elements = bytes.byteLength / DTYPE_BYTES[dtype];
    return new NDArray(dtype, checkShape(input.shape ?? [elements]), bytes);
  }

  const values = data as readonly (number | bigint)[];
  return new NDArray(
    dtype,
    checkShape(input.shape ?? [values.length]),
    packValues(dtype, values),
  );
}

/**
 * Builds a `bool` ndarray. There is no JS typed array that maps to `bool`, so
 * boolean data must go through this helper (or `ndarray({dtype: "bool", ...})`).
 * Each element occupies one byte, `0` or `1`, matching NumPy's `np.bool_`.
 */
export function boolArray(
  values: readonly boolean[],
  shape?: readonly number[],
): NDArray {
  const bytes = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++) bytes[i] = values[i] ? 1 : 0;
  return new NDArray("bool", checkShape(shape ?? [values.length]), bytes);
}

function checkShape(shape: readonly number[]): readonly number[] {
  if (shape.length === 0) {
    throw new Error(
      "Zero-dimensional ndarrays are never written to .colight files " +
        "(spec §3.3); serialize the scalar as a plain JSON number instead.",
    );
  }
  return shape.slice();
}

function toBytes(
  value: Uint8Array | ArrayBuffer | ArrayBufferView,
): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
}

/**
 * Returns the array's bytes in little-endian order, copying only when the host
 * is big-endian or the element type is multi-byte on a big-endian host.
 */
function littleEndianBytes(view: ArrayBufferView, dtype: Dtype): Uint8Array {
  const raw = new Uint8Array(
    view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength),
  );
  const size = DTYPE_BYTES[dtype];
  if (size === 1 || !HOST_IS_BIG_ENDIAN) return raw;
  for (let start = 0; start < raw.length; start += size) {
    for (let i = 0, j = size - 1; i < j; i++, j--) {
      const tmp = raw[start + i];
      raw[start + i] = raw[start + j];
      raw[start + j] = tmp;
    }
  }
  return raw;
}

const HOST_IS_BIG_ENDIAN = (() => {
  const probe = new Uint16Array([1]);
  return new Uint8Array(probe.buffer)[0] === 0;
})();

/** Largest and smallest integers a JS number can hold without loss. */
const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);
const MIN_SAFE = BigInt(Number.MIN_SAFE_INTEGER);

function packValues(
  dtype: Dtype,
  values: readonly (number | bigint)[],
): Uint8Array {
  const size = DTYPE_BYTES[dtype];
  const bytes = new Uint8Array(values.length * size);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    const at = i * size;
    switch (dtype) {
      case "int8":
        view.setInt8(at, Number(value));
        break;
      case "uint8":
        view.setUint8(at, Number(value));
        break;
      case "bool":
        view.setUint8(at, value ? 1 : 0);
        break;
      case "int16":
        view.setInt16(at, Number(value), true);
        break;
      case "uint16":
        view.setUint16(at, Number(value), true);
        break;
      case "int32":
        view.setInt32(at, Number(value), true);
        break;
      case "uint32":
        view.setUint32(at, Number(value), true);
        break;
      case "float32":
        view.setFloat32(at, Number(value), true);
        break;
      case "float64":
        view.setFloat64(at, Number(value), true);
        break;
      case "int64":
        view.setBigInt64(at, toSafeBigInt(value, dtype, i), true);
        break;
      case "uint64":
        view.setBigUint64(at, toSafeBigInt(value, dtype, i), true);
        break;
    }
  }
  return bytes;
}

/**
 * Converts a value destined for an `int64`/`uint64` element to a BigInt,
 * refusing plain numbers outside the exactly-representable range.
 *
 * Spec §3.3 / §8.2 note that JS readers convert 64-bit integers to doubles and
 * lose precision above 2^53. This writer will not *create* such a value from an
 * imprecise source: pass a BigInt when you need the full 64-bit range.
 */
function toSafeBigInt(
  value: number | bigint,
  dtype: Dtype,
  index: number,
): bigint {
  if (typeof value === "bigint") return value;
  if (!Number.isInteger(value)) {
    throw new Error(
      `${dtype} element at index ${index} must be an integer, got ${value}.`,
    );
  }
  const big = BigInt(value);
  if (big > MAX_SAFE || big < MIN_SAFE) {
    throw new Error(
      `${dtype} element at index ${index} has magnitude ${value}, which is ` +
        `outside the exactly-representable range of a JavaScript number ` +
        `(±2^53-1). Pass a BigInt to write this value without losing precision.`,
    );
  }
  return big;
}
