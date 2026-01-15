/**
 * Type-level tests for NdArrayView shape types.
 * These tests verify that TypeScript correctly infers types.
 */

import { describe, it, expect, expectTypeOf } from "vitest";
import {
  NdArrayView,
  TypedArrayLike,
  ndarray,
  RowAccessor,
} from "../../src/js/ndarray";

describe("NdArrayView shape types", () => {
  it("infers number for 1D array .get()", () => {
    const vec: NdArrayView<Float32Array, [3]> = ndarray(
      new Float32Array([1, 2, 3]),
      [3] as const,
    );

    // .get() on 1D array returns number
    const x = vec.get(0);
    expectTypeOf(x).toEqualTypeOf<number>();

    // Runtime check
    expect(x).toBe(1);
  });

  it("infers sub-view for 2D array .slice()", () => {
    const mat: NdArrayView<Float32Array, [2, 3]> = ndarray(
      new Float32Array([1, 2, 3, 4, 5, 6]),
      [2, 3] as const,
    );

    // .slice() on 2D array should return a 1D view
    const row = mat.slice(0);
    expectTypeOf(row).toEqualTypeOf<NdArrayView<Float32Array, [3]>>();

    // .get() on the slice returns number
    const val = mat.slice(0).get(1);
    expectTypeOf(val).toEqualTypeOf<number>();

    // .get() with multiple indices returns number
    const val2 = mat.get(0, 1);
    expectTypeOf(val2).toEqualTypeOf<number>();

    // Runtime check
    expect(val).toBe(2);
    expect(val2).toBe(2);
  });

  it("infers correct types for 3D arrays", () => {
    const tensor: NdArrayView<Float32Array, [2, 3, 4]> = ndarray(
      new Float32Array(24),
      [2, 3, 4] as const,
    );

    // First slice -> 2D view
    const slice = tensor.slice(0);
    expectTypeOf(slice).toEqualTypeOf<NdArrayView<Float32Array, [3, 4]>>();

    // Second slice -> 1D view
    const row = tensor.slice(0).slice(1);
    expectTypeOf(row).toEqualTypeOf<NdArrayView<Float32Array, [4]>>();

    // Third slice -> number (scalar)
    const val = tensor.slice(0).slice(1).slice(2);
    expectTypeOf(val).toEqualTypeOf<number>();
  });

  it("handles dynamic batch dimension", () => {
    // Shape [number, 7] - dynamic batch of 7-vectors
    const batched: NdArrayView<Float32Array, [number, 7]> = ndarray(
      new Float32Array(21),
      [3, 7],
    );

    // Slicing should give a 7-vector
    const vec = batched.slice(0);
    expectTypeOf(vec).toEqualTypeOf<NdArrayView<Float32Array, [7]>>();

    // .get() on 2D returns number
    const val = batched.get(0, 0);
    expectTypeOf(val).toEqualTypeOf<number>();
  });

  it("shape property has correct type", () => {
    const mat: NdArrayView<Float32Array, [2, 3]> = ndarray(
      new Float32Array(6),
      [2, 3] as const,
    );

    // Shape should be the tuple type
    expectTypeOf(mat.shape).toEqualTypeOf<[2, 3]>();
    expect(mat.shape).toEqual([2, 3]);
  });

  it("ndim property reflects shape length", () => {
    const mat: NdArrayView<Float32Array, [2, 3]> = ndarray(
      new Float32Array(6),
      [2, 3] as const,
    );

    // ndim should be 2
    expect(mat.ndim).toBe(2);
  });

  it("untyped NdArrayView is backward compatible", () => {
    // Should still work without shape type parameter
    const view: NdArrayView = ndarray(new Float32Array([1, 2, 3]), [3]);

    // .get() returns number
    const x = view.get(0);
    expectTypeOf(x).toEqualTypeOf<number>();

    expect(x).toBe(1);
  });

  it("provides forEach for iteration", () => {
    const mat: NdArrayView<Float32Array, [2, 3]> = ndarray(
      new Float32Array([1, 2, 3, 4, 5, 6]),
      [2, 3] as const,
    );

    let sum = 0;
    mat.forEach((value) => {
      sum += value;
    });

    expect(sum).toBe(21);
  });

  it("provides reduce for accumulation", () => {
    const mat: NdArrayView<Float32Array, [2, 3]> = ndarray(
      new Float32Array([1, 2, 3, 4, 5, 6]),
      [2, 3] as const,
    );

    const sum = mat.reduce((acc, value) => acc + value, 0);
    expect(sum).toBe(21);
  });

  it("provides mapRows for fast 2D row iteration", () => {
    const mat: NdArrayView<Float32Array, [2, 3]> = ndarray(
      new Float32Array([1, 2, 3, 4, 5, 6]),
      [2, 3] as const,
    );

    // mapRows callback receives RowAccessor
    const rowSums = mat.mapRows((row: RowAccessor, _index: number) => {
      return row.reduce((acc, val) => acc + val, 0);
    });

    expect(rowSums).toEqual([6, 15]);
  });

  it("provides map for first dimension iteration", () => {
    const mat: NdArrayView<Float32Array, [2, 3]> = ndarray(
      new Float32Array([1, 2, 3, 4, 5, 6]),
      [2, 3] as const,
    );

    // map receives sub-view for each row
    const firstElements = mat.map((row) => {
      if (typeof row === "number") return row;
      return row.get(0);
    });

    expect(firstElements).toEqual([1, 4]);
  });
});
