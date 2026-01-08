from dataclasses import dataclass

import numpy as np
import pytest

from colight_serde import (
    BufferCollector,
    pack_message,
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

    assert result == {"x": 1.0, "y": 2.0, "z": 3.0}
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

    assert result["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
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
    assert result[0] == {"x": 0, "y": 0}
    assert result[1] == {"x": 1, "y": 1}
    assert result[2] == {"x": 2, "y": 2}
