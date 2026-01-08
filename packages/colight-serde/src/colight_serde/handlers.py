"""Unified type handlers for serialization and TypeScript generation.

Each handler knows how to:
1. Match a type (by hint or value)
2. Serialize a value to wire format
3. Generate TypeScript type string from a hint

This ensures the serializer and type generator stay in sync.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import (
    Annotated,
    Any,
    Dict,
    ForwardRef,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

import numpy as np

from .arrays import to_numpy

Buffer = bytes | bytearray | memoryview


class BufferCollectorProtocol(Protocol):
    """Protocol for buffer collection."""

    def append(self, data: Buffer) -> None: ...
    def __len__(self) -> int: ...


# Numpy dtype -> (wire dtype string, TypeScript TypedArray)
# Note: int64/uint64 are downcast to number[] at runtime (with overflow warning)
# since JS numbers lose precision beyond Number.MAX_SAFE_INTEGER.
DTYPE_MAP: Dict[Any, Tuple[str, str]] = {
    np.float32: ("float32", "Float32Array"),
    np.float64: ("float64", "Float64Array"),
    np.int8: ("int8", "Int8Array"),
    np.int16: ("int16", "Int16Array"),
    np.int32: ("int32", "Int32Array"),
    np.int64: ("int64", "number[]"),  # Downcast from BigInt64Array
    np.uint8: ("uint8", "Uint8Array"),
    np.uint16: ("uint16", "Uint16Array"),
    np.uint32: ("uint32", "Uint32Array"),
    np.uint64: ("uint64", "number[]"),  # Downcast from BigUint64Array
}


def _dtype_to_ts(dtype: Any) -> str:
    """Map numpy dtype to TypeScript TypedArray type."""
    if dtype in DTYPE_MAP:
        return DTYPE_MAP[dtype][1]
    # Handle dtype instances
    if hasattr(dtype, "type") and dtype.type in DTYPE_MAP:
        return DTYPE_MAP[dtype.type][1]
    return "TypedArrayLike"


def _get_ndarray_dtype_from_hint(hint: Any) -> Optional[Any]:
    """Extract dtype from NDArray[dtype] hint."""
    args = get_args(hint)
    # Handle numpy.ndarray[Any, numpy.dtype[numpy.float32]] pattern
    if args and len(args) >= 2:
        dtype_arg = args[1]
        dtype_args = get_args(dtype_arg)
        if dtype_args:
            return dtype_args[0]
    # Handle NDArray[np.float32] shorthand
    if args and len(args) == 1:
        return args[0]
    return None


class TypeHandler(ABC):
    """Base class for type handlers."""

    @abstractmethod
    def matches_value(self, value: Any) -> bool:
        """Check if this handler can serialize the given value."""
        ...

    @abstractmethod
    def matches_hint(self, hint: Any) -> bool:
        """Check if this handler can generate TypeScript for the given hint."""
        ...

    @abstractmethod
    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        """Serialize value to wire format.

        Args:
            value: The value to serialize
            collector: Buffer collector for binary data
            recurse: Function to call for nested serialization
        """
        ...

    @abstractmethod
    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        """Generate TypeScript type string.

        Args:
            hint: The type hint
            recurse: Function to call for nested types
            seen: Set of referenced type names
            known_names: Set of known dataclass names
        """
        ...


class ArrayHandler(TypeHandler):
    """Handles numpy arrays and array-like types (JAX, PyTorch, etc)."""

    def matches_value(self, value: Any) -> bool:
        return to_numpy(value) is not None

    def matches_hint(self, hint: Any) -> bool:
        if hint is np.ndarray:
            return True
        origin = get_origin(hint)
        if origin is np.ndarray:
            return True
        # Check for string type aliases
        if isinstance(hint, str) and hint in ("FloatArray", "IntArray", "Array"):
            return True
        return False

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        array = to_numpy(value)
        if array is None:
            raise ValueError(f"Cannot convert {type(value)} to numpy array")

        if array.ndim == 0:
            return array.item()

        contiguous, order = self._to_contiguous(array)
        bytes_data = contiguous.tobytes(order=order)

        entry = {
            "__type__": "ndarray",
            "dtype": str(contiguous.dtype),
            "shape": contiguous.shape,
            "order": order,
            "strides": list(contiguous.strides),
        }

        if collector is not None:
            collector.append(bytes_data)
            entry["__buffer_index__"] = len(collector) - 1
        else:
            entry["data"] = bytes_data

        return entry

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        dtype = _get_ndarray_dtype_from_hint(hint)
        if dtype is not None:
            ts_type = _dtype_to_ts(dtype)
            if ts_type != "TypedArrayLike":
                # int64/uint64 are downcast to number[] (not wrapped in NdArrayView)
                if ts_type == "number[]":
                    return ts_type
                return f"NdArrayView<{ts_type}>"
        return "NdArrayView"

    def _to_contiguous(
        self, array: np.ndarray
    ) -> Tuple[np.ndarray, Literal["C", "F"]]:
        """Ensure array is contiguous and normalize byte order to little-endian."""
        # Normalize to little-endian for JS compatibility
        # byteorder: '<' = little, '>' = big, '=' = native, '|' = not applicable
        if array.dtype.byteorder not in ("<", "|"):
            array = array.astype(array.dtype.newbyteorder("<"))

        if array.flags["F_CONTIGUOUS"] and not array.flags["C_CONTIGUOUS"]:
            return np.asfortranarray(array), "F"
        return np.ascontiguousarray(array), "C"


class DataclassHandler(TypeHandler):
    """Handles dataclass instances and types."""

    def matches_value(self, value: Any) -> bool:
        return dataclasses.is_dataclass(value) and not isinstance(value, type)

    def matches_hint(self, hint: Any) -> bool:
        return isinstance(hint, type) and dataclasses.is_dataclass(hint)

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        return {
            f.name: recurse(getattr(value, f.name), collector)
            for f in dataclasses.fields(value)
        }

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        name = hint.__name__
        seen.add(name)
        return name


class PrimitiveHandler(TypeHandler):
    """Handles primitive types: int, float, str, bool, None."""

    _PRIMITIVES = (str, int, float, bool, type(None))
    _TS_MAP = {
        int: "number",
        float: "number",
        str: "string",
        bool: "boolean",
        type(None): "null",
    }

    def matches_value(self, value: Any) -> bool:
        return isinstance(value, self._PRIMITIVES)

    def matches_hint(self, hint: Any) -> bool:
        return hint in self._TS_MAP

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        return value

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        return self._TS_MAP.get(hint, "any")


class NumpyScalarHandler(TypeHandler):
    """Handles numpy scalar types (np.float32, etc)."""

    def matches_value(self, value: Any) -> bool:
        return isinstance(value, np.generic)

    def matches_hint(self, hint: Any) -> bool:
        return False  # Numpy scalars aren't used as type hints

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        return value.item()

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        return "number"


class BytesHandler(TypeHandler):
    """Handles bytes, bytearray, memoryview."""

    def matches_value(self, value: Any) -> bool:
        return isinstance(value, (bytes, bytearray, memoryview))

    def matches_hint(self, hint: Any) -> bool:
        return hint in (bytes, bytearray)

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        if collector is not None:
            collector.append(value)
            return {"__buffer_index__": len(collector) - 1}
        return value

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        return "ArrayBuffer"


class DictHandler(TypeHandler):
    """Handles dict types."""

    def matches_value(self, value: Any) -> bool:
        return isinstance(value, dict)

    def matches_hint(self, hint: Any) -> bool:
        origin = get_origin(hint)
        return origin is dict or hint is dict

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        return {k: recurse(v, collector) for k, v in value.items()}

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        args = get_args(hint)
        if len(args) == 2:
            key_type = recurse(args[0], seen, known_names)
            val_type = recurse(args[1], seen, known_names)
            if key_type in ("string", "number"):
                return f"{{ [key: {key_type}]: {val_type} }}"
            return f"Record<{key_type}, {val_type}>"
        return "Record<string, any>"


class ListHandler(TypeHandler):
    """Handles list and tuple types."""

    def matches_value(self, value: Any) -> bool:
        return isinstance(value, (list, tuple))

    def matches_hint(self, hint: Any) -> bool:
        origin = get_origin(hint)
        return origin in (list, tuple) or hint in (list, tuple)

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        return [recurse(x, collector) for x in value]

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        origin = get_origin(hint)
        args = get_args(hint)

        if origin is tuple or hint is tuple:
            if args:
                # Handle tuple[T, ...] (variable-length homogeneous tuple) -> T[]
                if len(args) == 2 and args[1] is ...:
                    inner = recurse(args[0], seen, known_names)
                    return f"{inner}[]"
                # Fixed-length tuple -> [A, B, C]
                ts_types = [recurse(a, seen, known_names) for a in args]
                return f"[{', '.join(ts_types)}]"
            return "any[]"

        # list
        if args:
            inner = recurse(args[0], seen, known_names)
            return f"{inner}[]"
        return "any[]"


class UnionHandler(TypeHandler):
    """Handles Union and Optional types."""

    def matches_value(self, value: Any) -> bool:
        return False  # Unions are type-level only

    def matches_hint(self, hint: Any) -> bool:
        return get_origin(hint) is Union

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        raise NotImplementedError("Union values are handled by their concrete type")

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        args = get_args(hint)
        non_none = [a for a in args if a is not type(None)]

        # Optional[T] -> T | null
        if len(non_none) == 1 and len(args) == 2:
            inner = recurse(non_none[0], seen, known_names)
            return f"{inner} | null"

        # General union
        ts_types = [recurse(a, seen, known_names) for a in args]
        return " | ".join(ts_types)


class ForwardRefHandler(TypeHandler):
    """Handles forward references (string type names)."""

    def matches_value(self, value: Any) -> bool:
        return False

    def matches_hint(self, hint: Any) -> bool:
        return isinstance(hint, (str, ForwardRef))

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        raise NotImplementedError("Forward refs are type-level only")

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        if isinstance(hint, ForwardRef):
            name = hint.__forward_arg__
        else:
            name = hint

        if name in known_names:
            seen.add(name)
            return name
        return name


class Shape:
    """Marker for array shape in Annotated types.

    Use with Annotated to specify the shape of an ndarray field:

        from typing import Annotated
        from numpy.typing import NDArray

        @dataclass
        class Pose:
            # Fixed shape [7]
            posquat: Annotated[NDArray[np.float32], Shape(7)]

            # Dynamic first dimension, fixed second: [number, 3]
            points: Annotated[NDArray[np.float32], Shape(None, 3)]

            # Fully dynamic: number[]
            values: Annotated[NDArray[np.float32], Shape(None)]

    Args:
        *dims: Dimension sizes. Use None for dynamic dimensions (becomes `number` in TS).
    """

    def __init__(self, *dims: int | None):
        self.dims = dims

    def __repr__(self) -> str:
        return f"Shape{self.dims}"

    def to_typescript(self) -> str:
        """Convert shape to TypeScript tuple type."""
        if not self.dims:
            return "number[]"
        ts_dims = ["number" if d is None else str(d) for d in self.dims]
        return f"[{', '.join(ts_dims)}]"


class AnnotatedHandler(TypeHandler):
    """Handles Annotated types, extracting Shape metadata for arrays."""

    def matches_value(self, value: Any) -> bool:
        return False  # Annotated is type-level only

    def matches_hint(self, hint: Any) -> bool:
        return get_origin(hint) is Annotated

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        raise NotImplementedError("Annotated is type-level only")

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        args = get_args(hint)
        if not args:
            return "any"

        base_type = args[0]
        shape: Optional[Shape] = None

        # Look for Shape in metadata
        for meta in args[1:]:
            if isinstance(meta, Shape):
                shape = meta
                break

        if shape is not None:
            # Generate typed NdArrayView with shape
            dtype_ts = _dtype_to_ts(_get_ndarray_dtype_from_hint(base_type))
            shape_ts = shape.to_typescript()
            return f"NdArrayView<{dtype_ts}, {shape_ts}>"

        # No Shape - recurse on base type
        return recurse(base_type, seen, known_names)


class LiteralHandler(TypeHandler):
    """Handles Literal types."""

    def matches_value(self, value: Any) -> bool:
        return False  # Literal is type-level only

    def matches_hint(self, hint: Any) -> bool:
        return get_origin(hint) is Literal

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        raise NotImplementedError("Literal is type-level only")

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        args = get_args(hint)
        literals = [repr(a) if isinstance(a, str) else str(a) for a in args]
        return " | ".join(literals)


class AnyHandler(TypeHandler):
    """Handles Any type."""

    def matches_value(self, value: Any) -> bool:
        return False  # Any matches everything at type level, not value level

    def matches_hint(self, hint: Any) -> bool:
        return hint is Any

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        raise NotImplementedError("Any is type-level only")

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        return "any"


# Handler registry - order matters (more specific handlers first)
HANDLERS: List[TypeHandler] = [
    NumpyScalarHandler(),
    ArrayHandler(),
    BytesHandler(),
    DataclassHandler(),
    PrimitiveHandler(),
    DictHandler(),
    ListHandler(),
    UnionHandler(),
    AnnotatedHandler(),
    LiteralHandler(),
    ForwardRefHandler(),
    AnyHandler(),  # Should be last - catches Any type
]


def find_handler_for_value(value: Any) -> Optional[TypeHandler]:
    """Find the handler that can serialize this value."""
    for handler in HANDLERS:
        if handler.matches_value(value):
            return handler
    return None


def find_handler_for_hint(hint: Any) -> Optional[TypeHandler]:
    """Find the handler that can generate TypeScript for this hint."""
    for handler in HANDLERS:
        if handler.matches_hint(hint):
            return handler
    return None
