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


def test_unpack_message_buffer_count_mismatch() -> None:
    envelope, buffers = pack_message({"x": np.arange(3, dtype=np.float32)})
    envelope = {**envelope, "buffer_count": envelope["buffer_count"] + 1}

    with pytest.raises(ValueError):
        unpack_message(envelope, buffers)
