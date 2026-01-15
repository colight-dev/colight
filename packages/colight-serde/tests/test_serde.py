from dataclasses import dataclass

import numpy as np
import pytest

from typing import Annotated

from numpy.typing import NDArray

from colight_serde import (
    BufferCollector,
    Shape,
    TypeRegistry,
    pack_message,
    register_type,
    register_types,
    serialize,
    deserialize,
    unpack_message,
)


def test_pack_unpack_roundtrip() -> None:
    payload = {
        "points": np.arange(6, dtype=np.float32).reshape(2, 3),
        "raw": b"hello",
        "meta": {"count": 2},
    }
    envelope, buffers = pack_message(payload)
    assert envelope["buffer_count"] == len(buffers)

    restored = unpack_message(envelope, buffers)
    assert np.array_equal(restored["points"], payload["points"])
    assert restored["raw"] == payload["raw"]
    assert restored["meta"] == payload["meta"]


def test_fortran_order_roundtrip() -> None:
    array = np.asfortranarray(np.arange(6, dtype=np.float64).reshape(2, 3, order="F"))
    collector = BufferCollector()
    serialized = serialize(array, collector)

    assert serialized["order"] == "F"
    assert serialized["strides"] == list(array.strides)

    restored = deserialize(serialized, collector.buffers)
    assert np.array_equal(restored, array)
    assert restored.flags["F_CONTIGUOUS"]


def test_non_contiguous_input_is_materialized() -> None:
    base = np.arange(10, dtype=np.int32)
    view = base[::2]
    collector = BufferCollector()
    serialized = serialize(view, collector)

    restored = deserialize(serialized, collector.buffers)
    assert np.array_equal(restored, np.ascontiguousarray(view))
    assert restored.flags["C_CONTIGUOUS"]


def test_big_endian_array_is_normalized() -> None:
    """Big-endian arrays should be converted to little-endian for JS compatibility."""
    # Create explicit big-endian array
    big_endian = np.array([1.0, 2.0, 3.0], dtype=">f4")  # big-endian float32
    assert big_endian.dtype.byteorder == ">"

    collector = BufferCollector()
    serialized = serialize(big_endian, collector)

    # Should still report float32 dtype (without byte order prefix)
    assert serialized["dtype"] == "float32"

    # Restored array should have correct values
    restored = deserialize(serialized, collector.buffers)
    assert np.array_equal(restored, [1.0, 2.0, 3.0])

    # Buffer should be little-endian (native for JS)
    buffer_bytes = collector.buffers[0]
    restored_from_bytes = np.frombuffer(buffer_bytes, dtype="<f4")
    assert np.array_equal(restored_from_bytes, [1.0, 2.0, 3.0])


def test_unpack_message_buffer_count_mismatch() -> None:
    envelope, buffers = pack_message({"x": np.arange(3, dtype=np.float32)})
    envelope = {**envelope, "buffer_count": envelope["buffer_count"] + 1}

    with pytest.raises(ValueError):
        unpack_message(envelope, buffers)


# --- Dataclass serialization tests ---


def test_simple_dataclass() -> None:
    """Test serializing a simple dataclass with scalar fields."""

    @dataclass
    class Point:
        x: float
        y: float
        z: float

    p = Point(1.0, 2.0, 3.0)
    collector = BufferCollector()
    result = serialize(p, collector)

    assert result == {"__serde__": "Point", "x": 1.0, "y": 2.0, "z": 3.0}
    assert len(collector.buffers) == 0


def test_dataclass_with_array() -> None:
    """Test serializing a dataclass containing numpy arrays."""

    @dataclass
    class Trajectory:
        positions: np.ndarray
        timestamps: np.ndarray

    positions = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    timestamps = np.array([0.0, 1.0], dtype=np.float64)

    traj = Trajectory(positions=positions, timestamps=timestamps)
    collector = BufferCollector()
    result = serialize(traj, collector)

    assert "__type__" in result["positions"]
    assert result["positions"]["__type__"] == "ndarray"
    assert result["positions"]["shape"] == (2, 3)
    assert result["positions"]["dtype"] == "float32"

    assert result["timestamps"]["__type__"] == "ndarray"
    assert result["timestamps"]["shape"] == (2,)
    assert result["timestamps"]["dtype"] == "float64"

    assert len(collector.buffers) == 2


def test_nested_dataclass() -> None:
    """Test serializing nested dataclasses."""

    @dataclass
    class Position:
        x: float
        y: float
        z: float

    @dataclass
    class Pose:
        position: Position
        rotation: np.ndarray  # quaternion

    pose = Pose(
        position=Position(1.0, 2.0, 3.0),
        rotation=np.array([0, 0, 0, 1], dtype=np.float32),
    )
    collector = BufferCollector()
    result = serialize(pose, collector)

    assert result["position"] == {"__serde__": "Position", "x": 1.0, "y": 2.0, "z": 3.0}
    assert result["rotation"]["__type__"] == "ndarray"
    assert result["rotation"]["shape"] == (4,)
    assert len(collector.buffers) == 1


def test_dataclass_roundtrip() -> None:
    """Test full roundtrip: serialize dataclass -> pack -> unpack -> deserialize."""

    @dataclass
    class Frame:
        id: int
        image: np.ndarray
        metadata: dict

    frame = Frame(
        id=42,
        image=np.arange(12, dtype=np.uint8).reshape(3, 4),
        metadata={"camera": "front", "exposure": 0.01},
    )

    envelope, buffers = pack_message(frame)
    restored = unpack_message(envelope, buffers)

    assert restored["id"] == 42
    assert np.array_equal(restored["image"], frame.image)
    assert restored["metadata"] == frame.metadata


def test_list_of_dataclasses() -> None:
    """Test serializing a list of dataclasses."""

    @dataclass
    class Point:
        x: float
        y: float

    points = [Point(0, 0), Point(1, 1), Point(2, 2)]
    collector = BufferCollector()
    result = serialize(points, collector)

    assert len(result) == 3
    assert result[0] == {"__serde__": "Point", "x": 0, "y": 0}
    assert result[1] == {"__serde__": "Point", "x": 1, "y": 1}
    assert result[2] == {"__serde__": "Point", "x": 2, "y": 2}


# --- Bidirectional deserialization tests ---


def test_deserialize_with_type_hint() -> None:
    """Test deserializing wire data to a typed dataclass using explicit hint."""

    @dataclass
    class Point3D:
        x: float
        y: float
        z: float

    # Simulate wire data (as if received from JS)
    wire_data = {"__serde__": "Point3D", "x": 1.0, "y": 2.0, "z": 3.0}
    buffers: list[bytes] = []

    result = deserialize(wire_data, buffers, Point3D)

    assert isinstance(result, Point3D)
    assert result.x == 1.0
    assert result.y == 2.0
    assert result.z == 3.0


def test_deserialize_with_arrays() -> None:
    """Test deserializing dataclass with array fields."""

    @dataclass
    class Trajectory:
        name: str
        points: Annotated[NDArray[np.float32], Shape(None, 3)]

    # Create wire data with array reference
    wire_data = {
        "__serde__": "Trajectory",
        "name": "path1",
        "points": {
            "__type__": "ndarray",
            "__buffer_index__": 0,
            "dtype": "float32",
            "shape": [2, 3],
            "order": "C",
            "strides": [12, 4],
        },
    }
    buffers = [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32).tobytes()]

    result = deserialize(wire_data, buffers, Trajectory)

    assert isinstance(result, Trajectory)
    assert result.name == "path1"
    assert result.points.shape == (2, 3)
    assert result.points.dtype == np.float32
    np.testing.assert_array_equal(result.points, [[1, 2, 3], [4, 5, 6]])


def test_deserialize_with_registry() -> None:
    """Test deserializing using type registry lookup from __type__ tag."""

    @register_type
    @dataclass
    class RegisteredPoint:
        x: float
        y: float

    wire_data = {"__serde__": "RegisteredPoint", "x": 10.0, "y": 20.0}
    buffers: list[bytes] = []

    # No explicit hint - should look up from registry
    result = deserialize(wire_data, buffers)

    assert isinstance(result, RegisteredPoint)
    assert result.x == 10.0
    assert result.y == 20.0


def test_deserialize_nested_dataclass() -> None:
    """Test deserializing nested dataclasses."""

    @dataclass
    class Inner:
        value: int

    @dataclass
    class Outer:
        name: str
        inner: Inner

    register_types(Inner, Outer)

    wire_data = {
        "__serde__": "Outer",
        "name": "test",
        "inner": {"__serde__": "Inner", "value": 42},
    }
    buffers: list[bytes] = []

    result = deserialize(wire_data, buffers, Outer)

    assert isinstance(result, Outer)
    assert result.name == "test"
    assert isinstance(result.inner, Inner)
    assert result.inner.value == 42


def test_full_roundtrip() -> None:
    """Test full serialize -> deserialize roundtrip."""

    @dataclass
    class Frame:
        id: int
        data: Annotated[
            NDArray[np.float32],
            Shape(
                None,
            ),
        ]
        metadata: dict[str, str]

    original = Frame(
        id=123,
        data=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        metadata={"key": "value"},
    )

    # Serialize
    collector = BufferCollector()
    wire_data = serialize(original, collector)

    # Deserialize back
    result = deserialize(wire_data, collector.buffers, Frame)

    assert isinstance(result, Frame)
    assert result.id == original.id
    np.testing.assert_array_equal(result.data, original.data)
    assert result.metadata == original.metadata


def test_deserialize_list_of_dataclasses() -> None:
    """Test deserializing a list of typed dataclasses."""

    @dataclass
    class Item:
        name: str
        value: int

    wire_data = [
        {"__serde__": "Item", "name": "a", "value": 1},
        {"__serde__": "Item", "name": "b", "value": 2},
    ]
    buffers: list[bytes] = []

    result = deserialize(wire_data, buffers, list[Item])

    assert len(result) == 2
    # Note: without registry, these come back as dicts
    # With registry they would be Item instances


def test_deserialize_optional_field() -> None:
    """Test deserializing dataclass with optional field using default."""

    @dataclass
    class WithDefault:
        required: int
        optional: str = "default"

    # Wire data missing the optional field
    wire_data = {"__serde__": "WithDefault", "required": 42}
    buffers: list[bytes] = []

    result = deserialize(wire_data, buffers, WithDefault)

    assert isinstance(result, WithDefault)
    assert result.required == 42
    assert result.optional == "default"


# --- TypeRegistry tests ---


def test_type_registry_basic() -> None:
    """Test TypeRegistry creation and lookup."""

    @dataclass
    class Point:
        x: float
        y: float

    registry = TypeRegistry(Point)
    assert registry.get("Point") is Point
    assert registry.get("Unknown") is None


def test_type_registry_deserialize() -> None:
    """Test TypeRegistry.deserialize with local registry."""

    @dataclass
    class LocalPoint:
        x: float
        y: float

    registry = TypeRegistry(LocalPoint)

    wire_data = {"__serde__": "LocalPoint", "x": 1.0, "y": 2.0}
    result = registry.deserialize(wire_data, [])

    assert isinstance(result, LocalPoint)
    assert result.x == 1.0
    assert result.y == 2.0


def test_type_registry_with_arrays() -> None:
    """Test TypeRegistry.deserialize with array fields."""

    @dataclass
    class ArrayData:
        name: str
        values: Annotated[
            NDArray[np.float32],
            Shape(
                None,
            ),
        ]

    registry = TypeRegistry(ArrayData)

    wire_data = {
        "__serde__": "ArrayData",
        "name": "test",
        "values": {
            "__type__": "ndarray",
            "__buffer_index__": 0,
            "dtype": "float32",
            "shape": [3],
            "order": "C",
            "strides": [4],
        },
    }
    buffers = [np.array([1, 2, 3], dtype=np.float32).tobytes()]

    result = registry.deserialize(wire_data, buffers)

    assert isinstance(result, ArrayData)
    assert result.name == "test"
    np.testing.assert_array_equal(result.values, [1, 2, 3])


def test_type_registry_generate_typescript() -> None:
    """Test TypeRegistry.generate_typescript."""

    @dataclass
    class Simple:
        x: float

    registry = TypeRegistry(Simple)
    ts = registry.generate_typescript()

    # Should include interface with __serde__ tag and constructor by default
    assert "export interface Simple {" in ts
    assert '__serde__: "Simple";' in ts
    assert "export function Simple(x: number): Simple {" in ts
    assert 'return { __serde__: "Simple", x };' in ts


def test_type_registry_rejects_non_dataclass() -> None:
    """Test TypeRegistry rejects non-dataclass types."""

    class NotADataclass:
        pass

    with pytest.raises(TypeError):
        TypeRegistry(NotADataclass)
