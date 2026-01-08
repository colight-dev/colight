"""Binary wire protocol utilities for array transfer."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .handlers import find_handler_for_value

Buffer = bytes | bytearray | memoryview


class BufferCollector:
    """Collect buffers for out-of-band binary transfer.

    Implements BufferCollectorProtocol for use with handlers.
    """

    def __init__(self, buffers: Optional[Sequence[Buffer]] = None) -> None:
        self._buffers: List[Buffer] = list(buffers) if buffers else []

    def append(self, data: Buffer) -> None:
        """Add a buffer (protocol method)."""
        self._buffers.append(data)

    def add(self, data: Buffer) -> int:
        """Add a buffer and return its index."""
        self._buffers.append(data)
        return len(self._buffers) - 1

    def extend(self, buffers: Iterable[Buffer]) -> None:
        self._buffers.extend(buffers)

    def get_buffers(self) -> List[Buffer]:
        return self._buffers

    def __len__(self) -> int:
        return len(self._buffers)

    @property
    def buffers(self) -> List[Buffer]:
        return self._buffers


def _normalize_buffers(
    buffers: Optional[List[Buffer] | BufferCollector],
) -> Optional[List[Buffer]]:
    if buffers is None:
        return None
    if isinstance(buffers, BufferCollector):
        return buffers.buffers
    return buffers


class _ListWrapper:
    """Wrapper that makes a list implement BufferCollectorProtocol."""

    def __init__(self, lst: List[Buffer]) -> None:
        self._list = lst

    def append(self, data: Buffer) -> None:
        self._list.append(data)

    def __len__(self) -> int:
        return len(self._list)


def _ensure_collector(
    collector: Optional[List[Buffer] | BufferCollector],
) -> Optional[BufferCollector | _ListWrapper]:
    """Convert list to protocol-compatible wrapper if needed."""
    if collector is None:
        return None
    if isinstance(collector, BufferCollector):
        return collector
    # Wrap list so appends go to the original list
    return _ListWrapper(collector)


def serialize_binary_data(
    buffers: Optional[List[Buffer] | BufferCollector],
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Add binary data to buffers list and return reference.

    Args:
        buffers: List or collector to append binary data to
        entry: Dictionary containing binary data under 'data' key

    Returns:
        Modified entry with buffer index reference
    """
    if buffers is None:
        return entry

    normalized = _normalize_buffers(buffers)
    if normalized is None:
        return entry

    normalized.append(entry["data"])
    index = len(normalized) - 1
    return {
        **entry,
        "__buffer_index__": index,
        "data": None,
    }


def deserialize_buffer_entry(
    data: Dict[str, Any],
    buffers: List[Buffer] | BufferCollector,
) -> Any:
    """Parse a buffer entry, converting to numpy array if needed.

    Args:
        data: Dictionary with buffer reference and optional type info
        buffers: List of binary buffers or a collector

    Returns:
        Raw buffer or numpy array depending on type
    """
    normalized = _normalize_buffers(buffers)
    if normalized is None:
        raise ValueError("buffers must be provided to deserialize buffer entry")

    buffer_idx = data["__buffer_index__"]
    if "__type__" in data and data["__type__"] == "ndarray":
        buffer = normalized[buffer_idx]
        dtype = np.dtype(data.get("dtype", "float64"))
        shape = tuple(data.get("shape", [len(buffer)]))
        order = data.get("order", "C")
        strides = data.get("strides")
        if strides is not None:
            return np.ndarray(
                shape=shape,
                dtype=dtype,
                buffer=buffer,
                strides=tuple(int(s) for s in strides),
            )
        return np.frombuffer(buffer, dtype=dtype).reshape(shape, order=order)
    return normalized[buffer_idx]


def replace_buffers(
    data: Any,
    buffers: List[Buffer] | BufferCollector,
) -> Any:
    """Replace buffer indices with actual buffer data in a nested structure.

    Args:
        data: Nested data structure potentially containing buffer references
        buffers: List of binary buffers or a collector

    Returns:
        Data structure with buffer references replaced by actual data
    """
    normalized = _normalize_buffers(buffers)
    if not normalized:
        return data

    if isinstance(data, dict):
        if "__buffer_index__" in data:
            return deserialize_buffer_entry(data, normalized)

        for k, v in data.items():
            if isinstance(v, dict) and "__buffer_index__" in v:
                data[k] = deserialize_buffer_entry(v, normalized)
            elif isinstance(v, (dict, list, tuple)):
                data[k] = replace_buffers(v, normalized)
        return data

    if not isinstance(data, (dict, list, tuple)):
        return data

    if isinstance(data, list):
        for i, x in enumerate(data):
            if isinstance(x, dict) and "__buffer_index__" in x:
                data[i] = deserialize_buffer_entry(x, normalized)
            elif isinstance(x, (dict, list, tuple)):
                data[i] = replace_buffers(x, normalized)
        return data

    result = list(data)
    modified = False
    for i, x in enumerate(data):
        if isinstance(x, dict) and "__buffer_index__" in x:
            result[i] = deserialize_buffer_entry(x, normalized)
            modified = True
        elif isinstance(x, (dict, list, tuple)):
            new_val = replace_buffers(x, normalized)
            if new_val is not x:
                result[i] = new_val
                modified = True

    if modified:
        return tuple(result)
    return data


def serialize(
    data: Any,
    collector: Optional[List[Buffer] | BufferCollector] = None,
) -> Any:
    """Serialize arrays and binary blobs into binary JSON structures.

    Uses the unified handler registry from handlers.py to ensure
    serialization stays in sync with TypeScript type generation.
    """
    # Ensure we have a proper collector (or None)
    coll = _ensure_collector(collector)

    handler = find_handler_for_value(data)
    if handler:
        return handler.serialize(data, coll, recurse=serialize)

    # Fallback: return as-is (primitives handled by PrimitiveHandler)
    return data


def deserialize(
    data: Any,
    buffers: List[Buffer] | BufferCollector,
) -> Any:
    """Deserialize binary JSON structures using provided buffers."""
    return replace_buffers(data, buffers)


def pack_message(
    payload: Any,
    buffers: Optional[List[Buffer] | BufferCollector] = None,
    message_id: Optional[str] = None,
) -> tuple[Dict[str, Any], List[Buffer]]:
    """Serialize payload and wrap it in a message envelope.

    Args:
        payload: Application-level payload to serialize.
        buffers: Optional buffer list/collector to append to.
        message_id: Optional message id (defaults to random UUID hex).

    Returns:
        (envelope, buffers): Envelope JSON and the buffer list.
    """
    message_id = message_id or uuid.uuid4().hex
    buffer_target: List[Buffer] | BufferCollector
    if buffers is None:
        buffer_target = []
    else:
        buffer_target = buffers
    serialized = serialize(payload, buffer_target)
    normalized = _normalize_buffers(buffer_target) or []
    envelope = {
        "message_id": message_id,
        "buffer_count": len(normalized),
        "payload": serialized,
    }
    return envelope, normalized


def unpack_message(
    envelope: Dict[str, Any],
    buffers: List[Buffer] | BufferCollector,
) -> Any:
    """Validate and deserialize a message envelope with associated buffers."""
    expected = envelope.get("buffer_count")
    normalized = _normalize_buffers(buffers)
    if normalized is None:
        raise ValueError("buffers must be provided to unpack message")
    if expected is not None and expected != len(normalized):
        raise ValueError(
            f"buffer_count mismatch: expected {expected}, got {len(normalized)}"
        )
    return deserialize(envelope.get("payload"), normalized)
