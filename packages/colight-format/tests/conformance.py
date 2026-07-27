"""Python side of the `.colight` two-way conformance suite.

This script is driven over a subprocess by the vitest suite in
`packages/colight-format/tests/`. It speaks a tiny JSON protocol on stdin/stdout
so the fixture set stays defined in exactly one place (`fixtures.ts`).

Commands (one JSON object on stdin, one JSON object on stdout):

``{"command": "write", "spec": <fixture>, "path": <out>}``
    Build the fixture with Python's ``colight.format`` writer and save it, so
    the JS side can byte-compare its own output against it.

``{"command": "read", "path": <in>}``
    Parse a file (written by the JS writer) with Python's public parse API and
    return its decoded values in the portable shape ``fixtures.ts`` expects.

Only Colight's public writer/reader entry points are used: this side is a black
box that the JS implementation is measured against.
"""

from __future__ import annotations

import base64
import json
import sys
from typing import Any

import numpy as np

from colight.format import create_bytes, parse_file_with_updates

# Dtypes the fixture vocabulary may name, mapped to NumPy dtypes. `bool` has no
# byte-order-qualified spelling; the rest are pinned little-endian explicitly so
# the fixture bytes do not depend on the host's endianness.
DTYPES = {
    "int8": "<i1",
    "int16": "<i2",
    "int32": "<i4",
    "int64": "<i8",
    "uint8": "<u1",
    "uint16": "<u2",
    "uint32": "<u4",
    "uint64": "<u8",
    "float32": "<f4",
    "float64": "<f8",
    "bool": "?",
}


def build_value(spec: dict[str, Any], buffers: list[bytes]) -> Any:
    """Turns a fixture ValueSpec into JSON + buffers, mirroring `buildPayload`."""
    kind = spec["kind"]
    if kind == "ndarray":
        array = np.array(spec["values"], dtype=DTYPES[spec["dtype"]]).reshape(
            spec["shape"]
        )
        index = len(buffers)
        buffers.append(array.tobytes(order="C"))
        return {
            "__type__": "ndarray",
            "data": None,
            "dtype": spec["dtype"],
            "shape": list(spec["shape"]),
            "__buffer_index__": index,
        }
    if kind == "bool":
        array = np.array(spec["values"], dtype="?").reshape(spec["shape"])
        index = len(buffers)
        buffers.append(array.tobytes(order="C"))
        return {
            "__type__": "ndarray",
            "data": None,
            "dtype": "bool",
            "shape": list(spec["shape"]),
            "__buffer_index__": index,
        }
    if kind == "raw":
        index = len(buffers)
        buffers.append(bytes(spec["bytes"]))
        return {"__buffer_index__": index}
    if kind == "int":
        return int(spec["value"])
    if kind == "float":
        return float(spec["value"])
    if kind in ("str", "bool_scalar"):
        return spec["value"]
    if kind == "null":
        return None
    if kind == "list":
        return [build_value(item, buffers) for item in spec["items"]]
    if kind == "object":
        return {key: build_value(value, buffers) for key, value in spec["entries"]}
    raise ValueError(f"Unknown fixture value kind: {kind!r}")


def build_entry(spec: dict[str, Any], wrap_as_update: bool) -> bytes:
    """Serializes one entry with Python's writer."""
    buffers: list[bytes] = []
    payload = build_value(spec, buffers)
    json_data = {"updates": payload} if wrap_as_update else payload
    return create_bytes(json_data, buffers)


def write_fixture(spec: dict[str, Any], path: str) -> dict[str, Any]:
    chunks: list[bytes] = []
    if spec["initial"] is not None:
        chunks.append(build_entry(spec["initial"], wrap_as_update=False))
    for update in spec["updates"]:
        chunks.append(build_entry(update, wrap_as_update=True))
    data = b"".join(chunks)
    with open(path, "wb") as handle:
        handle.write(data)
    return {"ok": True, "size": len(data)}


def decode_value(value: Any, buffers: list[bytes]) -> Any:
    """Resolves buffer references into the portable shape `fixtures.ts` expects."""
    if isinstance(value, dict):
        if value.get("__type__") == "ndarray":
            dtype = value["dtype"]
            if dtype not in DTYPES:
                raise ValueError(f"Unknown dtype in file: {dtype!r}")
            raw = buffers[value["__buffer_index__"]]
            array = np.frombuffer(raw, dtype=DTYPES[dtype]).reshape(value["shape"])
            flat = array.reshape(-1)
            return {
                "__array__": {
                    "dtype": dtype,
                    "shape": list(value["shape"]),
                    "values": [scalar_to_json(v) for v in flat.tolist()],
                }
            }
        if set(value.keys()) == {"__buffer_index__"}:
            return {"__raw__": list(buffers[value["__buffer_index__"]])}
        return {key: decode_value(item, buffers) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_value(item, buffers) for item in value]
    return value


def scalar_to_json(value: Any) -> Any:
    """NumPy `tolist()` yields Python bools for `?`; normalize to 0/1 ints."""
    if isinstance(value, bool):
        return 1 if value else 0
    return value


def strip_layout(value: Any) -> Any:
    """Drops container metadata the writer derives, so only payload is compared.

    `bufferLayout` is part of the entry's JSON but is computed from the buffers
    rather than supplied by the caller; the conformance fixtures describe
    payloads, and `writer.test.ts` checks the layout table structurally.
    """
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if key != "bufferLayout"}
    return value


def read_fixture(path: str) -> dict[str, Any]:
    initial, initial_buffers, updates = parse_file_with_updates(path)
    return {
        "ok": True,
        "initial": None
        if initial is None
        else strip_layout(decode_value(initial, initial_buffers)),
        "updates": [
            strip_layout(decode_value(entry["data"], entry["buffers"]))
            for entry in updates
        ],
    }


def read_raw(path: str) -> dict[str, Any]:
    """Reads a file's bytes back to the JS side, base64-encoded."""
    with open(path, "rb") as handle:
        return {"ok": True, "base64": base64.b64encode(handle.read()).decode("ascii")}


def main() -> None:
    request = json.load(sys.stdin)
    command = request["command"]
    try:
        if command == "write":
            result = write_fixture(request["spec"], request["path"])
        elif command == "read":
            result = read_fixture(request["path"])
        elif command == "read_raw":
            result = read_raw(request["path"])
        else:
            raise ValueError(f"Unknown command: {command!r}")
    except Exception as error:  # surfaced as a test failure on the JS side
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
