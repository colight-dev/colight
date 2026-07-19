"""Spec-conformance tests for the .colight file format.

Every claim these tests assert is documented in docs/src/colight_docs/format.md
(the authoritative format specification). If a test here needs to change, the
spec (and CURRENT_VERSION) almost certainly needs to change too.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

import colight.plot as Plot
from colight.format import (
    CURRENT_VERSION,
    HEADER_SIZE,
    MAGIC_BYTES,
    append_update,
    create_file,
    parse_file,
    parse_file_with_updates,
    save_updates,
)
from colight.widget import to_json_with_state

# One array per supported dtype, values chosen to have unambiguous bytes.
ARRAYS = {
    "int8": np.array([-1, 2, 3], dtype=np.int8),
    "int16": np.array([-4, 5], dtype=np.int16),
    "int32": np.array([[6, -7], [8, 9]], dtype=np.int32),
    "int64": np.array([10, -11], dtype=np.int64),
    "uint8": np.array([12, 13, 14], dtype=np.uint8),
    "uint16": np.array([1], dtype=np.uint16),
    "uint32": np.array([15, 16], dtype=np.uint32),
    "uint64": np.array([17], dtype=np.uint64),
    "float32": np.arange(10, dtype=np.float32).reshape(2, 5),
    "float64": np.array([1.5, -2.5, 3.5], dtype=np.float64),
}
RAW_BYTES = b"raw-bytes-payload"  # 17 bytes: exercises inter-buffer padding


def _parse_header(content: bytes, offset: int) -> dict:
    magic = content[offset : offset + 8]
    fields = struct.unpack_from("<6Q", content, offset + 8)
    return {
        "magic": magic,
        "version": fields[0],
        "json_offset": fields[1],
        "json_length": fields[2],
        "binary_offset": fields[3],
        "binary_length": fields[4],
        "num_buffers": fields[5],
        "reserved": content[offset + 56 : offset + HEADER_SIZE],
    }


@pytest.fixture()
def spec_file(tmp_path: Path) -> Path:
    """A representative file: initial entry with all dtypes + two updates."""
    path = tmp_path / "spec.colight"
    payload = {"arrays": ARRAYS, "blob": RAW_BYTES}
    json_data, buffers = to_json_with_state(payload)
    create_file(json_data, buffers, path)
    # State-only update, then an ast-bearing update.
    append_update(path, Plot.State({"zoom": 2.0, "tiny": ARRAYS["uint8"]}))
    append_update(path, Plot.raster(np.eye(2, dtype=np.float32)))
    return path


def test_header_byte_layout(spec_file: Path):
    content = spec_file.read_bytes()
    h = _parse_header(content, 0)

    # Magic: "COLIGHT\x00"
    assert h["magic"] == b"COLIGHT\x00" == MAGIC_BYTES
    # Version: uint64le == 2
    assert h["version"] == 2 == CURRENT_VERSION
    # JSON section immediately follows the 96-byte header
    assert h["json_offset"] == HEADER_SIZE == 96
    # Reserved bytes 56-95 are zero
    assert h["reserved"] == b"\x00" * 40

    # binary_offset == align8(json_offset + json_length)
    unaligned = h["json_offset"] + h["json_length"]
    assert h["binary_offset"] == (unaligned + 7) & ~7
    # Padding between JSON and binary section is zeroed
    assert content[unaligned : h["binary_offset"]] == b"\x00" * (
        h["binary_offset"] - unaligned
    )
    # Binary section of the first entry is 8-byte aligned in the file
    assert h["binary_offset"] % 8 == 0


def test_json_section_and_buffer_layout(spec_file: Path):
    content = spec_file.read_bytes()
    h = _parse_header(content, 0)

    json_bytes = content[h["json_offset"] : h["json_offset"] + h["json_length"]]
    entry_json = json.loads(json_bytes.decode("utf-8"))

    layout = entry_json["bufferLayout"]
    assert len(layout["offsets"]) == h["num_buffers"]
    assert len(layout["lengths"]) == h["num_buffers"]
    assert layout["count"] == h["num_buffers"]
    assert (
        layout["totalSize"]
        == layout["offsets"][-1] + layout["lengths"][-1]
        == h["binary_length"]
    )

    # Every buffer starts 8-byte aligned relative to the binary section,
    # padding between buffers is zeroed, and no buffer exceeds the section.
    prev_end = 0
    binary_start = h["binary_offset"]
    for off, length in zip(layout["offsets"], layout["lengths"]):
        assert off % 8 == 0
        assert off >= prev_end
        assert content[binary_start + prev_end : binary_start + off] == b"\x00" * (
            off - prev_end
        )
        assert off + length <= h["binary_length"]
        prev_end = off + length


def test_ndarray_envelopes_and_buffer_bytes(spec_file: Path):
    content = spec_file.read_bytes()
    h = _parse_header(content, 0)
    entry_json = json.loads(
        content[h["json_offset"] : h["json_offset"] + h["json_length"]]
    )
    layout = entry_json["bufferLayout"]

    def buffer_bytes(index: int) -> bytes:
        start = h["binary_offset"] + layout["offsets"][index]
        return content[start : start + layout["lengths"][index]]

    arrays_json = entry_json["ast"]["arrays"]
    for dtype, array in ARRAYS.items():
        envelope = arrays_json[dtype]
        assert envelope["__type__"] == "ndarray"
        assert envelope["data"] is None
        assert envelope["dtype"] == dtype
        assert envelope["shape"] == list(array.shape)
        idx = envelope["__buffer_index__"]
        # Buffer holds exactly tobytes(): C-order, densely packed
        assert buffer_bytes(idx) == array.tobytes()

    # Little-endianness spot check: uint16 value 1 is b"\x01\x00"
    assert buffer_bytes(arrays_json["uint16"]["__buffer_index__"]) == b"\x01\x00"

    # Raw bytes reference: {"__buffer_index__": n} with no __type__
    blob = entry_json["ast"]["blob"]
    assert set(blob.keys()) == {"__buffer_index__"}
    assert buffer_bytes(blob["__buffer_index__"]) == RAW_BYTES


def test_entry_framing_and_update_entries(spec_file: Path):
    content = spec_file.read_bytes()

    # Walk entries exactly as the spec prescribes: next entry starts at
    # align8(binary_offset + binary_length).
    entries = []
    offset = 0
    while offset < len(content):
        # Every entry starts at an 8-byte aligned absolute offset, so every
        # buffer's absolute offset is 8-aligned too (zero-copy guarantee).
        assert offset % 8 == 0
        h = _parse_header(content, offset)
        assert h["magic"] == MAGIC_BYTES
        assert h["version"] == CURRENT_VERSION
        entry_json = json.loads(
            content[
                offset + h["json_offset"] : offset + h["json_offset"] + h["json_length"]
            ]
        )
        if "bufferLayout" in entry_json:
            for buf_offset in entry_json["bufferLayout"]["offsets"]:
                assert (offset + h["binary_offset"] + buf_offset) % 8 == 0
        entries.append((h, entry_json))
        unpadded_end = offset + h["binary_offset"] + h["binary_length"]
        offset += (h["binary_offset"] + h["binary_length"] + 7) & ~7
        # The trailing entry padding is zeroed.
        assert content[unpadded_end:offset] == b"\x00" * (offset - unpadded_end)
    # The walk must consume the file exactly, with no trailing bytes.
    assert offset == len(content)
    assert len(entries) == 3

    # The alignment padding is real: the first entry ends with a 17-byte raw
    # buffer, so its unpadded size is not a multiple of 8.
    first = entries[0][0]
    assert (first["binary_offset"] + first["binary_length"]) % 8 != 0

    # Entry roles: initial entry has no "updates" key; update entries do.
    assert "updates" not in entries[0][1]
    for _, entry_json in entries[1:]:
        assert "updates" in entry_json

    # State-only update: ast is null, state carries the values,
    # buffer indices are local to the update entry (start at 0).
    state_update = entries[1][1]["updates"]
    assert state_update["ast"] is None
    assert state_update["state"]["zoom"] == 2.0
    tiny = state_update["state"]["tiny"]
    assert tiny["__type__"] == "ndarray"
    assert tiny["__buffer_index__"] == 0
    assert entries[1][1]["bufferLayout"]["count"] == entries[1][0]["num_buffers"] == 1

    # AST-bearing update: ast is not null.
    assert entries[2][1]["updates"]["ast"] is not None

    # The reference parser agrees with the manual walk.
    initial, buffers, update_entries = parse_file_with_updates(spec_file)
    assert initial is not None
    assert len(buffers) == entries[0][0]["num_buffers"]
    assert len(update_entries) == 2
    assert update_entries[0]["data"]["state"]["zoom"] == 2.0
    assert bytes(update_entries[0]["buffers"][0]) == ARRAYS["uint8"].tobytes()


def test_writer_normalizes_big_endian_arrays(tmp_path: Path):
    """The wire format is little-endian: big-endian arrays are converted on write."""
    big_endian = np.array([1.5, -2.0, 3.25], dtype=">f4")
    path = tmp_path / "endian.colight"
    json_data, buffers = to_json_with_state({"arr": big_endian})
    create_file(json_data, buffers, path)

    initial, parsed_buffers, _updates = parse_file(path)
    assert initial is not None
    envelope = initial["ast"]["arr"]
    # dtype is the canonical name, never a byte-order-qualified string
    assert envelope["dtype"] == "float32"
    # bytes are little-endian
    assert (
        bytes(parsed_buffers[envelope["__buffer_index__"]])
        == big_endian.astype("<f4").tobytes()
    )


def test_version_mismatch_errors_loudly(spec_file: Path):
    content = bytearray(spec_file.read_bytes())
    struct.pack_into("<Q", content, 8, CURRENT_VERSION + 1)
    bumped = spec_file.parent / "bumped.colight"
    bumped.write_bytes(content)

    with pytest.raises(ValueError) as excinfo:
        parse_file(bumped)
    message = str(excinfo.value)
    assert f"found {CURRENT_VERSION + 1}" in message
    assert f"supports version {CURRENT_VERSION}" in message


def test_wrong_magic_errors(tmp_path: Path):
    bad = tmp_path / "bad.colight"
    bad.write_bytes(b"NOTMAGIC" + b"\x00" * 88)
    with pytest.raises(ValueError, match="magic"):
        parse_file(bad)


def test_updates_only_file(tmp_path: Path):
    path = tmp_path / "updates-only.colight"
    save_updates(path, [Plot.State({"a": 1}), Plot.State({"a": 2})])

    content = path.read_bytes()
    h = _parse_header(content, 0)
    assert h["magic"] == MAGIC_BYTES
    entry_json = json.loads(
        content[h["json_offset"] : h["json_offset"] + h["json_length"]]
    )
    assert "updates" in entry_json

    initial, buffers, updates = parse_file(path)
    assert initial is None
    assert buffers == []
    assert [u["state"]["a"] for u in updates] == [1, 2]


def test_zero_buffer_entry_has_no_buffer_layout(tmp_path: Path):
    path = tmp_path / "no-buffers.colight"
    json_data, buffers = to_json_with_state({"message": "hello"})
    create_file(json_data, buffers, path)

    content = path.read_bytes()
    h = _parse_header(content, 0)
    assert h["num_buffers"] == 0
    assert h["binary_length"] == 0
    entry_json = json.loads(
        content[h["json_offset"] : h["json_offset"] + h["json_length"]]
    )
    # bufferLayout is present iff the entry has buffers
    assert "bufferLayout" not in entry_json
    # Entry is still padded out to the aligned binary offset
    assert len(content) == h["binary_offset"]
    assert h["binary_offset"] % 8 == 0
