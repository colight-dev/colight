"""Wire protocol utilities for binary data transfer."""

from .serialization import (
    BufferCollector,
    deserialize,
    deserialize_buffer_entry,
    pack_message,
    replace_buffers,
    serialize,
    serialize_binary_data,
    unpack_message,
)

__all__ = [
    "BufferCollector",
    "deserialize",
    "deserialize_buffer_entry",
    "pack_message",
    "replace_buffers",
    "serialize",
    "serialize_binary_data",
    "unpack_message",
]
