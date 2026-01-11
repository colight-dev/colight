/**
 * Minimal ndarray view for zero-copy multidimensional array access.
 *
 * Provides method-based access to arrays without proxy overhead:
 *   .get(i, j, ...)     - Direct element access
 *   .set(v, i, j, ...)  - Direct element mutation
 *   .slice(i)           - Returns sub-view (no data copy)
 *   .reduce(fn, init)   - Fast iteration with accumulator
 *   .forEach(fn)        - Fast iteration
 *   .map(fn)            - Map over first dimension
 *   .mapRows(fn)        - Fast 2D row iteration with lightweight row accessor
 *
 * Example:
 *   const view = ndarray(new Float32Array([1,2,3,4,5,6]), [2, 3]);
 *   view.get(0, 1)  // => 2
 *   view.get(1, 2)  // => 6
 *   view.flat       // => Float32Array([1,2,3,4,5,6])
 *   view.shape      // => [2, 3]
 *
 * Sub-views:
 *   view.slice(0)           // => NdArrayView for first row
 *   view.slice(0).get(1)    // => 2
 *
 * Iteration (fast callback-based):
 *   view.reduce((acc, v) => acc + v, 0)  // sum all elements
 *   view.mapRows(row => row.reduce((a, v) => a + v, 0))  // row sums
 */

export type TypedArrayLike =
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | Uint8ClampedArray
  | Float32Array
  | Float64Array
  | number[];

/**
 * Shape tuple type - represents array dimensions.
 * Can be concrete ([3, 4]) or use `number` for dynamic dimensions ([number, 3]).
 */
export type Shape = readonly number[];

/**
 * Drop the first element of a tuple type.
 * [A, B, C] -> [B, C]
 * [A] -> []
 */
type Tail<S extends Shape> = S extends readonly [number, ...infer Rest]
  ? Rest extends Shape
    ? Rest
    : []
  : number[]; // fallback for generic Shape

/**
 * Compute the return type when slicing an NdArrayView.
 * - Shape [N, ...rest] sliced -> NdArrayView with shape [...rest]
 * - Shape [N] sliced -> number
 */
type SliceResult<
  T extends TypedArrayLike,
  S extends Shape,
> = S extends readonly [number]
  ? number
  : S extends readonly []
    ? number
    : NdArrayView<T, Tail<S>>;

/**
 * Lightweight row accessor for fast 2D iteration.
 * Passed to mapRows callback - not a full NdArrayView.
 */
export interface RowAccessor {
  /** Get element at index */
  get(j: number): number;
  /** Number of elements */
  readonly length: number;
  /** Iterate over elements */
  forEach(callback: (value: number, index: number) => void): void;
  /** Reduce over elements */
  reduce<U>(
    callback: (acc: U, value: number, index: number) => U,
    initial: U,
  ): U;
}

/**
 * NdArrayView with optional shape type parameter.
 *
 * @typeParam T - The underlying TypedArray type (e.g., Float32Array)
 * @typeParam S - The shape as a tuple type (e.g., [3, 4] or [number, 7])
 *
 * Examples:
 *   NdArrayView                           // untyped shape
 *   NdArrayView<Float32Array>             // typed data, untyped shape
 *   NdArrayView<Float32Array, [3]>        // vector of 3 floats
 *   NdArrayView<Float32Array, [4, 4]>     // 4x4 matrix
 *   NdArrayView<Float32Array, [number, 7]> // dynamic batch of 7-vectors
 */
export interface NdArrayView<
  T extends TypedArrayLike = TypedArrayLike,
  S extends Shape = Shape,
> {
  /** The underlying flat typed array */
  readonly flat: T;
  /** Array shape */
  readonly shape: S;
  /** Element strides for each dimension */
  readonly strides: readonly number[];
  /** Number of dimensions */
  readonly ndim: S["length"];
  /** Total number of elements */
  readonly length: number;
  /** Byte offset into underlying buffer (for sub-views) */
  readonly offset: number;

  /** Get element at indices */
  get(...indices: number[]): number;
  /** Set element at indices */
  set(value: number, ...indices: number[]): void;

  /** Get sub-view along first dimension (or scalar for 1D) */
  slice(i: number): SliceResult<T, S>;

  /** Iterate over all elements with callback */
  forEach(callback: (value: number, ...indices: number[]) => void): void;

  /** Reduce over all elements */
  reduce<U>(
    callback: (acc: U, value: number, ...indices: number[]) => U,
    initial: U,
  ): U;

  /** Map over first dimension, passing sub-views (or scalars for 1D) to callback */
  map<U>(callback: (slice: SliceResult<T, S>, index: number) => U): U[];

  /** Fast row iteration for 2D arrays - callback receives lightweight RowAccessor */
  mapRows<U>(callback: (row: RowAccessor, index: number) => U): U[];
}

/**
 * Compute default C-order strides from shape.
 */
function computeStrides(shape: readonly number[]): number[] {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Create a minimal ndarray view (no Proxy).
 *
 * @param data - Flat typed array or number array
 * @param shape - Array dimensions
 * @param strides - Optional element strides (defaults to C-order)
 * @param offset - Offset into flat array (for sub-views)
 */
export function ndarray<T extends TypedArrayLike>(
  data: T,
  shape: readonly number[],
  strides?: readonly number[],
  offset: number = 0,
): NdArrayView<T> {
  const actualStrides = strides ?? computeStrides(shape);
  const ndim = shape.length;
  const length = shape.reduce((a, b) => a * b, 1);

  const view: NdArrayView<T> = {
    flat: data,
    shape: shape as NdArrayView<T>["shape"],
    strides: actualStrides,
    ndim: ndim as NdArrayView<T>["ndim"],
    length,
    offset,

    get(...indices: number[]): number {
      let idx = offset;
      for (let i = 0; i < indices.length; i++) {
        idx += indices[i] * actualStrides[i];
      }
      return data[idx] as number;
    },

    set(value: number, ...indices: number[]): void {
      let idx = offset;
      for (let i = 0; i < indices.length; i++) {
        idx += indices[i] * actualStrides[i];
      }
      (data as number[])[idx] = value;
    },

    slice(i: number): any {
      if (ndim < 1) throw new Error("slice() requires 1D+ array");
      if (ndim === 1) {
        // For 1D, return scalar
        return data[offset + i * actualStrides[0]] as number;
      }
      return ndarray(
        data,
        shape.slice(1),
        actualStrides.slice(1),
        offset + i * actualStrides[0],
      );
    },

    forEach(callback: (value: number, ...indices: number[]) => void): void {
      if (ndim === 1) {
        const s0 = actualStrides[0];
        for (let i = 0; i < shape[0]; i++) {
          callback(data[offset + i * s0] as number, i);
        }
      } else if (ndim === 2) {
        const s0 = actualStrides[0],
          s1 = actualStrides[1];
        for (let i = 0; i < shape[0]; i++) {
          const base = offset + i * s0;
          for (let j = 0; j < shape[1]; j++) {
            callback(data[base + j * s1] as number, i, j);
          }
        }
      } else if (ndim === 3) {
        const s0 = actualStrides[0],
          s1 = actualStrides[1],
          s2 = actualStrides[2];
        for (let i = 0; i < shape[0]; i++) {
          const base0 = offset + i * s0;
          for (let j = 0; j < shape[1]; j++) {
            const base1 = base0 + j * s1;
            for (let k = 0; k < shape[2]; k++) {
              callback(data[base1 + k * s2] as number, i, j, k);
            }
          }
        }
      } else {
        // General case - recursive
        const iterate = (dim: number, off: number, indices: number[]): void => {
          if (dim === ndim) {
            callback(data[off] as number, ...indices);
          } else {
            for (let i = 0; i < shape[dim]; i++) {
              iterate(dim + 1, off + i * actualStrides[dim], [...indices, i]);
            }
          }
        };
        iterate(0, offset, []);
      }
    },

    reduce<U>(
      callback: (acc: U, value: number, ...indices: number[]) => U,
      initial: U,
    ): U {
      let acc = initial;
      if (ndim === 1) {
        const s0 = actualStrides[0];
        for (let i = 0; i < shape[0]; i++) {
          acc = callback(acc, data[offset + i * s0] as number, i);
        }
      } else if (ndim === 2) {
        const s0 = actualStrides[0],
          s1 = actualStrides[1];
        for (let i = 0; i < shape[0]; i++) {
          const base = offset + i * s0;
          for (let j = 0; j < shape[1]; j++) {
            acc = callback(acc, data[base + j * s1] as number, i, j);
          }
        }
      } else if (ndim === 3) {
        const s0 = actualStrides[0],
          s1 = actualStrides[1],
          s2 = actualStrides[2];
        for (let i = 0; i < shape[0]; i++) {
          const base0 = offset + i * s0;
          for (let j = 0; j < shape[1]; j++) {
            const base1 = base0 + j * s1;
            for (let k = 0; k < shape[2]; k++) {
              acc = callback(acc, data[base1 + k * s2] as number, i, j, k);
            }
          }
        }
      } else {
        view.forEach((val, ...indices) => {
          acc = callback(acc, val, ...indices);
        });
      }
      return acc;
    },

    map<U>(callback: (slice: any, index: number) => U): U[] {
      const results = new Array(shape[0]);
      for (let i = 0; i < shape[0]; i++) {
        results[i] = callback(view.slice(i), i);
      }
      return results;
    },

    mapRows<U>(callback: (row: RowAccessor, index: number) => U): U[] {
      if (ndim !== 2) throw new Error("mapRows() requires 2D array");
      const results = new Array(shape[0]);
      const rowStride = actualStrides[0];
      const colStride = actualStrides[1];
      const cols = shape[1];
      for (let i = 0; i < shape[0]; i++) {
        const rowOffset = offset + i * rowStride;
        // Lightweight row accessor (not a full NdArrayView)
        results[i] = callback(
          {
            get(j: number): number {
              return data[rowOffset + j * colStride] as number;
            },
            length: cols,
            forEach(cb: (value: number, index: number) => void): void {
              for (let j = 0; j < cols; j++) {
                cb(data[rowOffset + j * colStride] as number, j);
              }
            },
            reduce<V>(
              cb: (acc: V, value: number, index: number) => V,
              init: V,
            ): V {
              let acc = init;
              for (let j = 0; j < cols; j++) {
                acc = cb(acc, data[rowOffset + j * colStride] as number, j);
              }
              return acc;
            },
          },
          i,
        );
      }
      return results;
    },
  };

  return view;
}

/**
 * Check if a value is an NdArrayView.
 */
export function isNdArray(value: unknown): value is NdArrayView {
  return (
    value !== null &&
    typeof value === "object" &&
    "flat" in value &&
    "shape" in value &&
    "strides" in value &&
    "ndim" in value &&
    "get" in value
  );
}

/**
 * Convert ndarray view to nested JavaScript arrays.
 * Useful when you need actual nested arrays (e.g., for JSON serialization).
 */
export function toNestedArray(view: NdArrayView): number[] | number[][] | any {
  if (view.ndim === 1) {
    return Array.from({ length: view.shape[0] }, (_, i) => view.get(i));
  }

  return Array.from({ length: view.shape[0] }, (_, i) => {
    const subView = view.slice(i);
    if (isNdArray(subView)) {
      return toNestedArray(subView);
    }
    return subView;
  });
}
