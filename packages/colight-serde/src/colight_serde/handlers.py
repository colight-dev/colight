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

    def deserialize(
        self,
        value: Any,
        hint: Any,
        buffers: List[Buffer],
        recurse: Any,
    ) -> Any:
        """Deserialize wire format back to Python value.

        Args:
            value: The wire format value (JSON-like)
            hint: The expected Python type hint
            buffers: List of binary buffers
            recurse: Function to call for nested deserialization

        Returns:
            The reconstructed Python value

        Note:
            Not all handlers implement deserialization. Those that don't
            will raise NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support deserialization"
        )


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

    def deserialize(
        self,
        value: Any,
        hint: Any,
        buffers: List[Buffer],
        recurse: Any,
    ) -> np.ndarray:
        """Reconstruct numpy array from wire format."""
        if not isinstance(value, dict) or value.get("__type__") != "ndarray":
            raise ValueError(f"Expected ndarray wire format, got {type(value)}")

        buffer_idx = value["__buffer_index__"]
        buffer = buffers[buffer_idx]
        dtype = np.dtype(value.get("dtype", "float64"))
        shape = tuple(value.get("shape", [len(buffer) // dtype.itemsize]))
        order = value.get("order", "C")
        strides = value.get("strides")

        if strides is not None:
            return np.ndarray(
                shape=shape,
                dtype=dtype,
                buffer=buffer,
                strides=tuple(int(s) for s in strides),
            )
        return np.frombuffer(buffer, dtype=dtype).reshape(shape, order=order)


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
            "__serde__": type(value).__name__,
            **{
                f.name: recurse(getattr(value, f.name), collector)
                for f in dataclasses.fields(value)
            },
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

    def deserialize(
        self,
        value: Any,
        hint: Any,
        buffers: List[Buffer],
        recurse: Any,
    ) -> Any:
        """Reconstruct dataclass from wire format."""
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict for dataclass, got {type(value)}")

        # Get field type hints
        try:
            from typing import get_type_hints

            field_hints = get_type_hints(hint, include_extras=True)
        except Exception:
            field_hints = getattr(hint, "__annotations__", {})

        # Reconstruct each field
        kwargs = {}
        for field in dataclasses.fields(hint):
            if field.name not in value:
                # Check if field has a default
                if (
                    field.default is not dataclasses.MISSING
                    or field.default_factory is not dataclasses.MISSING
                ):
                    continue
                raise ValueError(f"Missing required field: {field.name}")

            field_hint = field_hints.get(field.name, Any)
            kwargs[field.name] = recurse(value[field.name], field_hint, buffers)

        return hint(**kwargs)


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

    def deserialize(
        self,
        value: Any,
        hint: Any,
        buffers: List[Buffer],
        recurse: Any,
    ) -> Any:
        """Primitives pass through unchanged."""
        return value


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

    def deserialize(
        self,
        value: Any,
        hint: Any,
        buffers: List[Buffer],
        recurse: Any,
    ) -> bytes:
        """Reconstruct bytes from wire format."""
        if isinstance(value, dict) and "__buffer_index__" in value:
            buffer = buffers[value["__buffer_index__"]]
            return bytes(buffer) if not isinstance(buffer, bytes) else buffer
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value)
        raise ValueError(f"Expected buffer reference or bytes, got {type(value)}")


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

    def deserialize(
        self,
        value: Any,
        hint: Any,
        buffers: List[Buffer],
        recurse: Any,
    ) -> Dict[Any, Any]:
        """Reconstruct dict from wire format."""
        if not isinstance(value, dict):
            raise ValueError(f"Expected dict, got {type(value)}")

        args = get_args(hint)
        value_hint = args[1] if len(args) == 2 else Any
        return {k: recurse(v, value_hint, buffers) for k, v in value.items()}


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

    def deserialize(
        self,
        value: Any,
        hint: Any,
        buffers: List[Buffer],
        recurse: Any,
    ) -> List[Any] | Tuple[Any, ...]:
        """Reconstruct list/tuple from wire format."""
        if not isinstance(value, list):
            raise ValueError(f"Expected list, got {type(value)}")

        origin = get_origin(hint)
        args = get_args(hint)

        if origin is tuple:
            if args and len(args) >= 2 and args[-1] is not ...:
                # Fixed-length tuple: tuple[A, B, C]
                if len(value) != len(args):
                    raise ValueError(
                        f"Tuple length mismatch: expected {len(args)}, got {len(value)}"
                    )
                return tuple(
                    recurse(v, arg_hint, buffers)
                    for v, arg_hint in zip(value, args)
                )
            # Variable-length tuple or untyped
            item_hint = args[0] if args else Any
            return tuple(recurse(v, item_hint, buffers) for v in value)

        # list
        item_hint = args[0] if args else Any
        return [recurse(v, item_hint, buffers) for v in value]


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

    def deserialize(
        self,
        value: Any,
        hint: Any,
        buffers: List[Buffer],
        recurse: Any,
    ) -> Any:
        """Unwrap Annotated and deserialize the base type."""
        args = get_args(hint)
        if not args:
            return value
        base_type = args[0]
        return recurse(value, base_type, buffers)


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


class GenericDataclassHandler(TypeHandler):
    """Handles generic dataclass types like Mask[T] where T is resolved.

    This handles parameterized dataclasses by
    generating a TypeScript inline object type for the base class with the
    type parameter resolved.
    """

    def _is_generic_dataclass(self, hint: Any) -> bool:
        """Check if hint is a parameterized dataclass."""
        origin = get_origin(hint)
        if origin is None:
            return False
        # Check if the origin is a dataclass
        return dataclasses.is_dataclass(origin)

    def matches_value(self, value: Any) -> bool:
        return False  # Type-level only

    def matches_hint(self, hint: Any) -> bool:
        return self._is_generic_dataclass(hint)

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        raise NotImplementedError("Generic dataclass hints are type-level only")

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        origin = get_origin(hint)
        args = get_args(hint)

        # Get the base class name
        name = origin.__name__

        # Generate inline interface based on dataclass fields
        if dataclasses.is_dataclass(origin):
            # Get field type hints
            try:
                from typing import get_type_hints

                field_hints = get_type_hints(origin, include_extras=True)
            except Exception:
                field_hints = getattr(origin, "__annotations__", {})

            # Build object type parts
            parts = [f'__serde__: "{name}"']
            for field in dataclasses.fields(origin):
                field_hint = field_hints.get(field.name, Any)

                # If the field hint is a TypeVar, resolve it from args
                # TypeVar has __name__ attribute and is in typing module
                if (
                    hasattr(field_hint, "__name__")
                    and hasattr(field_hint, "__bound__")
                    and args
                ):
                    # It's a TypeVar - use the first type argument
                    field_hint = args[0]

                ts_type = recurse(field_hint, seen, known_names)
                parts.append(f"{field.name}: {ts_type}")

            return "{ " + "; ".join(parts) + " }"

        # Fallback
        return "any"


class JaxtypingHandler(TypeHandler):
    """Handles jaxtyping array types like Float[Array, '...'], Int[Array, '...'].

    These are typed array hints from the jaxtyping library. We map them to
    appropriate TypeScript NdArrayView types based on their dtype category.
    """

    # Map jaxtyping dtype categories to TypeScript types
    _DTYPE_MAP = {
        # Float types -> Float32Array (most common in practice)
        "float": "Float32Array",
        "float32": "Float32Array",
        "float64": "Float64Array",
        "float16": "Float32Array",  # Upcast for TS compatibility
        "bfloat16": "Float32Array",
        # Int types -> Int32Array (most common)
        "int": "Int32Array",
        "int8": "Int8Array",
        "int16": "Int16Array",
        "int32": "Int32Array",
        "int64": "number[]",  # BigInt64Array not widely used
        # Unsigned int types
        "uint": "Uint32Array",
        "uint8": "Uint8Array",
        "uint16": "Uint16Array",
        "uint32": "Uint32Array",
        "uint64": "number[]",
        # Bool
        "bool": "Uint8Array",  # Boolean arrays as uint8
    }

    def _is_jaxtyping_hint(self, hint: Any) -> bool:
        """Check if hint is a jaxtyping array type."""
        # jaxtyping types have a special metaclass from jaxtyping._array_types
        hint_module = getattr(type(hint), "__module__", "")
        return hint_module.startswith("jaxtyping")

    def _get_dtype_category(self, hint: Any) -> Optional[str]:
        """Extract dtype category from jaxtyping hint."""
        # jaxtyping hints have a 'dtypes' attribute with allowed dtype names
        dtypes = getattr(hint, "dtypes", None)
        if not dtypes:
            return None

        # Check for common dtype patterns
        dtypes_set = set(dtypes)

        # Float types (float8 variants, bfloat16, float16, float32, float64)
        float_dtypes = {
            "float8_e4m3b11fnuz",
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
            "bfloat16",
            "float16",
            "float32",
            "float64",
        }
        if dtypes_set & float_dtypes:
            # Prefer float32 if available, otherwise use generic float
            if "float32" in dtypes_set:
                return "float32"
            if "float64" in dtypes_set:
                return "float64"
            return "float"

        # Int types
        int_dtypes = {"int8", "int16", "int32", "int64"}
        if dtypes_set & int_dtypes:
            if "int32" in dtypes_set:
                return "int32"
            if "int64" in dtypes_set:
                return "int64"
            return "int"

        # Unsigned int types
        uint_dtypes = {"uint8", "uint16", "uint32", "uint64"}
        if dtypes_set & uint_dtypes:
            if "uint8" in dtypes_set:
                return "uint8"
            if "uint32" in dtypes_set:
                return "uint32"
            return "uint"

        # Bool
        if "bool" in dtypes_set:
            return "bool"

        return None

    def matches_value(self, value: Any) -> bool:
        return False  # jaxtyping hints are type-level only

    def matches_hint(self, hint: Any) -> bool:
        return self._is_jaxtyping_hint(hint)

    def serialize(
        self,
        value: Any,
        collector: Optional[BufferCollectorProtocol],
        recurse: Any,
    ) -> Any:
        raise NotImplementedError("jaxtyping hints are type-level only")

    def to_typescript(
        self,
        hint: Any,
        recurse: Any,
        seen: Set[str],
        known_names: Set[str],
    ) -> str:
        dtype_category = self._get_dtype_category(hint)
        if dtype_category:
            ts_type = self._DTYPE_MAP.get(dtype_category, "Float32Array")
            if ts_type == "number[]":
                return ts_type
            return f"NdArrayView<{ts_type}>"
        return "NdArrayView"


# Handler registry - order matters (more specific handlers first)
HANDLERS: List[TypeHandler] = [
    NumpyScalarHandler(),
    ArrayHandler(),
    JaxtypingHandler(),  # Before DataclassHandler - matches jaxtyping array hints
    GenericDataclassHandler(),  # Handles Mask[T] and other parameterized dataclasses
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
