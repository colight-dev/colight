"""Wire protocol utilities for binary data transfer."""

from .arrays import (
    auto_register,
    is_array_like,
    register_array_converter,
    register_jax,
    register_tensorflow,
    register_torch,
    register_warp,
    to_numpy,
)
from .handlers import Shape
from .serialization import (
    BufferCollector,
    TypeRegistry,
    deserialize,
    deserialize_buffer_entry,
    get_registered_type,
    pack_message,
    register_type,
    register_types,
    replace_buffers,
    serialize,
    serialize_binary_data,
    unpack_message,
)
from .typescript import (
    generate_typescript,
    write_typescript,
)

__all__ = [
    # Array conversion
    "auto_register",
    "is_array_like",
    "register_array_converter",
    "register_jax",
    "register_tensorflow",
    "register_torch",
    "register_warp",
    "to_numpy",
    # Serialization
    "BufferCollector",
    "TypeRegistry",
    "deserialize",
    "deserialize_buffer_entry",
    "get_registered_type",
    "pack_message",
    "register_type",
    "register_types",
    "replace_buffers",
    "serialize",
    "serialize_binary_data",
    "unpack_message",
    # TypeScript generation
    "Shape",
    "generate_typescript",
    "write_typescript",
]
