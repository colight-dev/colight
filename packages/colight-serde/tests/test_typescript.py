"""Tests for TypeScript interface generation."""

from dataclasses import dataclass
from typing import Annotated, Optional

import numpy as np
from numpy.typing import NDArray

from colight_serde import Shape, generate_typescript


def test_basic_types():
    """Test generation of basic Python types."""

    @dataclass
    class BasicTypes:
        i: int
        f: float
        s: str
        b: bool
        opt: Optional[str]

    ts = generate_typescript(BasicTypes)

    assert "i: number;" in ts
    assert "f: number;" in ts
    assert "s: string;" in ts
    assert "b: boolean;" in ts
    assert "opt: string | null;" in ts


def test_ndarray_without_shape():
    """Test NDArray without shape annotation."""

    @dataclass
    class ArrayTypes:
        floats: NDArray[np.float32]
        ints: NDArray[np.int32]
        doubles: NDArray[np.float64]

    ts = generate_typescript(ArrayTypes)

    assert "floats: NdArrayView<Float32Array>;" in ts
    assert "ints: NdArrayView<Int32Array>;" in ts
    assert "doubles: NdArrayView<Float64Array>;" in ts


def test_ndarray_with_shape():
    """Test NDArray with Shape annotation."""

    @dataclass
    class ShapedArrays:
        vec3: Annotated[NDArray[np.float32], Shape(3)]
        mat4x4: Annotated[NDArray[np.float32], Shape(4, 4)]
        batch: Annotated[NDArray[np.float32], Shape(None, 7)]
        dynamic: Annotated[NDArray[np.float64], Shape(None)]

    ts = generate_typescript(ShapedArrays)

    assert "vec3: NdArrayView<Float32Array, [3]>;" in ts
    assert "mat4x4: NdArrayView<Float32Array, [4, 4]>;" in ts
    assert "batch: NdArrayView<Float32Array, [number, 7]>;" in ts
    assert "dynamic: NdArrayView<Float64Array, [number]>;" in ts


def test_nested_dataclasses():
    """Test nested dataclass references."""

    @dataclass
    class Inner:
        value: float

    @dataclass
    class Outer:
        inner: Inner
        inners: list[Inner]

    ts = generate_typescript(Inner, Outer)

    assert "export interface Inner {" in ts
    assert "export interface Outer {" in ts
    assert "inner: Inner;" in ts
    assert "inners: Inner[];" in ts


def test_shape_class():
    """Test Shape class methods."""
    s1 = Shape(7)
    assert s1.to_typescript() == "[7]"

    s2 = Shape(3, 4)
    assert s2.to_typescript() == "[3, 4]"

    s3 = Shape(None, 7)
    assert s3.to_typescript() == "[number, 7]"

    s4 = Shape(None, None, 3)
    assert s4.to_typescript() == "[number, number, 3]"


def test_includes_import():
    """Test that import statement is included."""

    @dataclass
    class WithArray:
        arr: NDArray[np.float32]

    ts = generate_typescript(WithArray)

    assert 'import type { NdArrayView } from "@colight/serde";' in ts


def test_all_dtypes():
    """Test all supported numpy dtypes."""

    @dataclass
    class AllDtypes:
        f32: Annotated[NDArray[np.float32], Shape(1)]
        f64: Annotated[NDArray[np.float64], Shape(1)]
        i8: Annotated[NDArray[np.int8], Shape(1)]
        i16: Annotated[NDArray[np.int16], Shape(1)]
        i32: Annotated[NDArray[np.int32], Shape(1)]
        u8: Annotated[NDArray[np.uint8], Shape(1)]
        u16: Annotated[NDArray[np.uint16], Shape(1)]
        u32: Annotated[NDArray[np.uint32], Shape(1)]

    ts = generate_typescript(AllDtypes)

    assert "f32: NdArrayView<Float32Array, [1]>;" in ts
    assert "f64: NdArrayView<Float64Array, [1]>;" in ts
    assert "i8: NdArrayView<Int8Array, [1]>;" in ts
    assert "i16: NdArrayView<Int16Array, [1]>;" in ts
    assert "i32: NdArrayView<Int32Array, [1]>;" in ts
    assert "u8: NdArrayView<Uint8Array, [1]>;" in ts
    assert "u16: NdArrayView<Uint16Array, [1]>;" in ts
    assert "u32: NdArrayView<Uint32Array, [1]>;" in ts


def test_no_classes():
    """Test behavior when no classes are passed."""
    ts = generate_typescript()
    assert "// No types specified" in ts


def test_int64_uint64_downcast_to_number():
    """Test that int64/uint64 arrays generate number[] (not BigInt arrays).

    BigInt64Array/BigUint64Array values are downcast to number[] at runtime
    because JS numbers lose precision beyond Number.MAX_SAFE_INTEGER.
    """

    @dataclass
    class BigIntTypes:
        i64: NDArray[np.int64]
        u64: NDArray[np.uint64]

    ts = generate_typescript(BigIntTypes)

    # Should generate number[] not NdArrayView<BigInt64Array>
    assert "i64: number[];" in ts
    assert "u64: number[];" in ts
    assert "BigInt64Array" not in ts
    assert "BigUint64Array" not in ts


def test_tuple_ellipsis_generates_array():
    """Test that tuple[T, ...] generates T[] not [T, any]."""

    @dataclass
    class VarTuple:
        items: tuple[int, ...]
        fixed: tuple[int, str, float]

    ts = generate_typescript(VarTuple)

    # Variable-length tuple -> array
    assert "items: number[];" in ts
    # Fixed-length tuple -> tuple type
    assert "fixed: [number, string, number];" in ts


# --- Constructor generation tests ---


def test_constructors_basic():
    """Test constructor generation for basic types."""

    @dataclass
    class Point:
        x: float
        y: float
        z: float

    ts = generate_typescript(Point)

    # Should include interface with __serde__ tag
    assert "export interface Point {" in ts
    assert '__serde__: "Point";' in ts

    # Should include constructor function
    assert "export function Point(x: number, y: number, z: number): Point {" in ts
    assert 'return { __serde__: "Point", x, y, z };' in ts


def test_constructors_with_arrays():
    """Test constructor generation for array types."""

    @dataclass
    class Trajectory:
        name: str
        points: Annotated[NDArray[np.float32], Shape(None, 3)]

    ts = generate_typescript(Trajectory)

    # Should include interface with __serde__ and array field
    assert "export interface Trajectory {" in ts
    assert '__serde__: "Trajectory";' in ts
    assert "points: NdArrayView<Float32Array, [number, 3]>;" in ts

    # Should include simple constructor (no buffer extraction)
    assert "export function Trajectory(name: string, points: NdArrayView<Float32Array, [number, 3]>): Trajectory {" in ts
    assert 'return { __serde__: "Trajectory", name, points };' in ts


def test_constructors_nested():
    """Test constructor generation for nested dataclasses."""

    @dataclass
    class Inner:
        value: int

    @dataclass
    class Outer:
        name: str
        inner: Inner

    ts = generate_typescript(Inner, Outer)

    # Should have constructors for both
    assert "export function Inner(value: number): Inner {" in ts
    assert "export function Outer(name: string, inner: Inner): Outer {" in ts
    # Both should just pass through fields (no special handling)
    assert 'return { __serde__: "Inner", value };' in ts
    assert 'return { __serde__: "Outer", name, inner };' in ts


def test_constructors_import():
    """Test that constructors generate proper import (type-only)."""

    @dataclass
    class Simple:
        x: float

    ts = generate_typescript(Simple)

    # Should only import type (no isNdArray needed for simple constructors)
    assert 'import type { NdArrayView } from "@colight/serde";' in ts
