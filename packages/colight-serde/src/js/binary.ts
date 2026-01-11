import {
  ndarray,
  toNestedArray as toNestedArrayFromView,
  type NdArrayView,
} from "./ndarray";

export type TypedArray =
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Uint8ClampedArray
  | Float32Array
  | Float64Array;

/** Type representing arrays that contain BigInt values */
export type BigIntArray = BigInt64Array | BigUint64Array;

/** Options for evaluateNdarray */
export interface EvaluateOptions {
  /**
   * How to return multidimensional arrays:
   * - "view": NdArrayView with arr[i][j] access + .flat/.shape/.strides (default)
   * - "nested": Copy to nested JS arrays
   */
  multidimensional?: "view" | "nested";
}

/**
 * Type guard to check if a value is a BigInt array type
 */
export function isBigIntArray(value: unknown): value is BigIntArray {
  return value instanceof BigInt64Array || value instanceof BigUint64Array;
}

/** Maximum safe integer in JavaScript */
const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER;
const MIN_SAFE_INTEGER = Number.MIN_SAFE_INTEGER;

/**
 * Converts a BigInt array to a regular number array.
 * Warns if any values exceed safe integer range.
 */
function convertBigIntArray(value: unknown): number[] | unknown {
  if (value instanceof BigInt64Array || value instanceof BigUint64Array) {
    let hasOverflow = false;
    const result = Array.from(value, (v) => {
      const n = Number(v);
      if (!hasOverflow && (n > MAX_SAFE_INTEGER || n < MIN_SAFE_INTEGER)) {
        hasOverflow = true;
        console.warn(
          `[colight-serde] BigInt value ${v} exceeds Number.MAX_SAFE_INTEGER. ` +
            `Precision may be lost. Consider using smaller integer types.`,
        );
      }
      return n;
    });
    return result;
  }
  return value;
}

/**
 * Compute element strides from shape and order.
 */
function computeElementStrides(dims: number[], order: "C" | "F"): number[] {
  const strides = new Array(dims.length);
  if (order === "F") {
    let acc = 1;
    for (let i = 0; i < dims.length; i += 1) {
      strides[i] = acc;
      acc *= dims[i];
    }
    return strides;
  }
  let acc = 1;
  for (let i = dims.length - 1; i >= 0; i -= 1) {
    strides[i] = acc;
    acc *= dims[i];
  }
  return strides;
}

function sliceWithStride(
  flat: TypedArray | number[],
  offset: number,
  length: number,
  stride: number,
) {
  if (stride === 1 && typeof flat.slice === "function") {
    return flat.slice(offset, offset + length);
  }
  const out = ArrayBuffer.isView(flat)
    ? new (flat.constructor as any)(length)
    : new Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = flat[offset + i * stride];
  }
  return out;
}

/**
 * Reshape a flat array into nested arrays.
 * Note: This copies data. For zero-copy access, use ndarray view.
 */
export function reshapeArray(
  flat: TypedArray | number[],
  dims: number[],
  offset = 0,
  strides: number[] | null = null,
): any {
  const [dim, ...restDims] = dims;
  const stride = strides ? strides[0] : null;

  if (restDims.length === 0) {
    if (stride === null) {
      return flat.slice(offset, offset + dim);
    }
    return sliceWithStride(flat, offset, dim, stride);
  }

  const nextStride = stride ?? restDims.reduce((a, b) => a * b, 1);
  const nextStrides = strides ? strides.slice(1) : null;
  return Array.from({ length: dim }, (_, i) =>
    reshapeArray(flat, restDims, offset + i * nextStride, nextStrides),
  );
}

const dtypeMap: Record<string, any> = {
  float32: Float32Array,
  float64: Float64Array,
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  int64: BigInt64Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  uint64: BigUint64Array,
  uint8clamped: Uint8ClampedArray,
  bigint64: BigInt64Array,
  biguint64: BigUint64Array,
};

/**
 * Infers the numpy dtype for a TypedArray by examining its constructor name.
 */
export function inferDtype(value: ArrayBuffer | ArrayBufferView): string {
  if (!(value instanceof ArrayBuffer || ArrayBuffer.isView(value))) {
    throw new Error("Value must be a TypedArray or ArrayBuffer");
  }

  const name = value.constructor.name.toLowerCase().replace("array", "");
  // Handle ArrayBuffer specially - it's raw bytes, not a typed array
  if (name === "buffer") {
    return "uint8"; // Default interpretation for raw bytes
  }
  return name;
}

/**
 * Convert an NdArrayView to nested JavaScript arrays.
 */
export function asNestedArray(view: NdArrayView): any {
  return toNestedArrayFromView(view);
}

/**
 * Evaluates an ndarray node by converting binary data to typed array.
 *
 * @param node - The ndarray metadata and data
 * @param options - How to handle multidimensional arrays
 * @returns TypedArray for 1D, or based on options for >1D
 */
export function evaluateNdarray(
  node: {
    data: DataView | ArrayBuffer | ArrayBufferView;
    dtype: string;
    shape: number[];
    order?: "C" | "F";
    strides?: number[];
  },
  options: EvaluateOptions = {},
): TypedArray | number[] | NdArrayView {
  const { data, dtype, shape } = node;
  const order = node.order || "C";
  const wireStrides = node.strides || null;
  const multidimensional = options.multidimensional ?? "view";

  const ArrayConstructor = dtypeMap[dtype] || Float64Array;
  const bytesPerElement = ArrayConstructor.BYTES_PER_ELEMENT;

  const view =
    data instanceof DataView
      ? data
      : ArrayBuffer.isView(data)
        ? new DataView(data.buffer, data.byteOffset, data.byteLength)
        : new DataView(data);

  // Create typed array directly from the DataView's buffer (zero-copy)
  const flatArray = new ArrayConstructor(
    view.buffer,
    view.byteOffset,
    view.byteLength / bytesPerElement,
  );

  // Convert BigInt arrays to regular number arrays (with overflow warning)
  const convertedArray = convertBigIntArray(flatArray) as TypedArray | number[];

  const elementStrides = wireStrides
    ? wireStrides.map((stride) => stride / bytesPerElement)
    : computeElementStrides(shape, order);

  // 1D arrays are always returned directly
  if (shape.length <= 1) {
    return convertedArray;
  }

  // Handle multidimensional based on options
  if (multidimensional === "nested") {
    return reshapeArray(convertedArray, shape, 0, elementStrides);
  }

  // Default: return NdArrayView with .flat, .shape, .strides and [i][j] access
  return ndarray(convertedArray, shape, elementStrides);
}

/**
 * Estimates the size of a JSON string in bytes.
 */
export function estimateJSONSize(
  jsonString: string | null | undefined,
): string {
  if (!jsonString) return "0 B";

  const encoder = new TextEncoder();
  const bytes = encoder.encode(jsonString).length;

  if (bytes < 1024) {
    return `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  } else {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }
}
