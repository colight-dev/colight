"""Binary data serialization utilities for colight widgets."""

from typing import Any, Dict, List, Optional
import numpy as np


def serialize_binary_data(
    buffers: Optional[List[bytes | bytearray | memoryview]], entry: Dict[str, Any]
) -> Dict[str, Any]:
    """Add binary data to buffers list and return reference.

    Args:
        buffers: List to append binary data to
        entry: Dictionary containing binary data under 'data' key

    Returns:
        Modified entry with buffer index reference
    """
    if buffers is None:
        return entry

    buffers.append(entry["data"])
    index = len(buffers) - 1
    return {
        **entry,
        "__buffer_index__": index,
        "data": None,
    }


def deserialize_buffer_entry(data: Dict[str, Any], buffers: List[bytes]) -> Any:
    """Parse a buffer entry, converting to numpy array if needed.

    Args:
        data: Dictionary with buffer reference and optional type info
        buffers: List of binary buffers

    Returns:
        Raw buffer or numpy array depending on type
    """
    buffer_idx = data["__buffer_index__"]
    if "__type__" in data and data["__type__"] == "ndarray":
        # Convert buffer to numpy array
        buffer = buffers[buffer_idx]
        dtype = data.get("dtype", "float64")
        shape = data.get("shape", [len(buffer)])
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)
    return buffers[buffer_idx]


def replace_buffers(data: Any, buffers: List[bytes]) -> Any:
    """Replace buffer indices with actual buffer data in a nested data structure.

    Args:
        data: Nested data structure potentially containing buffer references
        buffers: List of binary buffers

    Returns:
        Data structure with buffer references replaced by actual data
    """
    if not buffers:
        return data

    # Fast path for direct buffer reference
    if isinstance(data, dict):
        if "__buffer_index__" in data:
            return deserialize_buffer_entry(data, buffers)

        # Process dictionary values in-place
        for k, v in data.items():
            if isinstance(v, dict) and "__buffer_index__" in v:
                data[k] = deserialize_buffer_entry(v, buffers)
            elif isinstance(v, (dict, list, tuple)):
                data[k] = replace_buffers(v, buffers)
        return data

    # Fast path for non-container types
    if not isinstance(data, (dict, list, tuple)):
        return data

    if isinstance(data, list):
        # Mutate list in-place
        for i, x in enumerate(data):
            if isinstance(x, dict) and "__buffer_index__" in x:
                data[i] = deserialize_buffer_entry(x, buffers)
            elif isinstance(x, (dict, list, tuple)):
                data[i] = replace_buffers(x, buffers)
        return data

    # Handle tuples
    result = list(data)
    modified = False
    for i, x in enumerate(data):
        if isinstance(x, dict) and "__buffer_index__" in x:
            result[i] = deserialize_buffer_entry(x, buffers)
            modified = True
        elif isinstance(x, (dict, list, tuple)):
            new_val = replace_buffers(x, buffers)
            if new_val is not x:
                result[i] = new_val
                modified = True

    if modified:
        return tuple(result)
    return data
