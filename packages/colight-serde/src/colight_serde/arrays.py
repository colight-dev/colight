"""Array type detection and conversion registry.

Provides extensible support for converting array-like objects (JAX, PyTorch,
TensorFlow, etc.) to numpy arrays for serialization.

By default, converters are auto-registered for libraries that are already
imported when this module loads. For dynamically loaded libraries, use the
explicit registration functions:

    from colight_serde.arrays import register_jax, register_torch
    register_jax()  # Enable JAX array support

For custom array types:

    from colight_serde.arrays import register_array_converter

    def is_my_array(value):
        return type(value).__module__.startswith('mylib')

    def convert_my_array(value):
        return value.to_numpy()

    register_array_converter(is_my_array, convert_my_array)
"""

from __future__ import annotations

import sys
from typing import Any, Callable

import numpy as np

# Type aliases
ArrayChecker = Callable[[Any], bool]
ArrayConverter = Callable[[Any], np.ndarray]

# Registry: list of (checker, converter, priority, name) tuples, checked in order
_CONVERTERS: list[tuple[ArrayChecker, ArrayConverter, int, str]] = []

# Track which built-in converters are registered to avoid duplicates
_REGISTERED: set[str] = set()


def register_array_converter(
    check: ArrayChecker,
    convert: ArrayConverter,
    *,
    priority: int = 0,
    name: str | None = None,
) -> None:
    """Register a converter for a custom array type.

    Args:
        check: Function that returns True if value is this array type.
        convert: Function that converts value to numpy array.
        priority: Higher priority converters are checked first. Default is 0.
            Built-in converters use priority -10.
        name: Optional name for this converter (used to prevent duplicates).
    """
    _CONVERTERS.append((check, convert, priority, name or ""))
    # Sort by priority descending
    _CONVERTERS.sort(key=lambda x: -x[2])


def to_numpy(value: Any) -> np.ndarray | None:
    """Convert an array-like value to numpy, or return None if not array-like.

    Checks registered converters in priority order, then falls back to
    the __array__ protocol.

    Args:
        value: Value to convert.

    Returns:
        numpy array if value is array-like, None otherwise.
    """
    # Fast path for numpy arrays
    if isinstance(value, np.ndarray):
        return value

    # Check registered converters
    for check, convert, _, _ in _CONVERTERS:
        try:
            if check(value):
                return convert(value)
        except Exception:
            # Checker or converter failed, try next
            continue

    # Fallback: duck-type check for __array__ protocol
    if hasattr(value, "__array__"):
        try:
            return np.asarray(value)
        except Exception:
            pass

    return None


def is_array_like(value: Any) -> bool:
    """Check if a value can be converted to a numpy array."""
    if isinstance(value, np.ndarray):
        return True
    return to_numpy(value) is not None


# --- Built-in converter definitions ---


def _is_jax_array(value: Any) -> bool:
    """Check if value is a JAX array."""
    module = type(value).__module__
    return module.startswith("jax") or module.startswith("jaxlib")


def _convert_jax_array(value: Any) -> np.ndarray:
    """Convert JAX array to numpy."""
    # JAX arrays support __array__ protocol, but we may need to block
    # until computation is complete on device
    return np.asarray(value)


def _is_torch_tensor(value: Any) -> bool:
    """Check if value is a PyTorch tensor."""
    return type(value).__module__.startswith("torch")


def _convert_torch_tensor(value: Any) -> np.ndarray:
    """Convert PyTorch tensor to numpy."""
    # Must detach from computation graph and move to CPU
    return value.detach().cpu().numpy()


def _is_tf_tensor(value: Any) -> bool:
    """Check if value is a TensorFlow tensor."""
    module = type(value).__module__
    return module.startswith("tensorflow") or module.startswith("tf.")


def _convert_tf_tensor(value: Any) -> np.ndarray:
    """Convert TensorFlow tensor to numpy."""
    return value.numpy()


def _is_warp_array(value: Any) -> bool:
    """Check if value is a NVIDIA Warp array."""
    return type(value).__module__.startswith("warp")


def _convert_warp_array(value: Any) -> np.ndarray:
    """Convert Warp array to numpy."""
    return value.numpy()


# --- Explicit registration functions ---


def register_jax() -> None:
    """Register converter for JAX arrays.

    Called automatically if JAX is already imported, but can be called
    manually for dynamically loaded modules.
    """
    if "jax" in _REGISTERED:
        return
    _REGISTERED.add("jax")
    register_array_converter(_is_jax_array, _convert_jax_array, priority=-10, name="jax")


def register_torch() -> None:
    """Register converter for PyTorch tensors.

    Called automatically if PyTorch is already imported, but can be called
    manually for dynamically loaded modules.
    """
    if "torch" in _REGISTERED:
        return
    _REGISTERED.add("torch")
    register_array_converter(
        _is_torch_tensor, _convert_torch_tensor, priority=-10, name="torch"
    )


def register_tensorflow() -> None:
    """Register converter for TensorFlow tensors.

    Called automatically if TensorFlow is already imported, but can be called
    manually for dynamically loaded modules.
    """
    if "tensorflow" in _REGISTERED:
        return
    _REGISTERED.add("tensorflow")
    register_array_converter(
        _is_tf_tensor, _convert_tf_tensor, priority=-10, name="tensorflow"
    )


def register_warp() -> None:
    """Register converter for NVIDIA Warp arrays.

    Called automatically if Warp is already imported, but can be called
    manually for dynamically loaded modules.
    """
    if "warp" in _REGISTERED:
        return
    _REGISTERED.add("warp")
    register_array_converter(
        _is_warp_array, _convert_warp_array, priority=-10, name="warp"
    )


def auto_register() -> None:
    """Register converters for libraries that are already imported.

    This is called automatically at module import time. Call it again
    if you dynamically import array libraries after this module loads.
    """
    if "jax" in sys.modules or "jaxlib" in sys.modules:
        register_jax()
    if "torch" in sys.modules:
        register_torch()
    if "tensorflow" in sys.modules:
        register_tensorflow()
    if "warp" in sys.modules:
        register_warp()


# Auto-register on module import
auto_register()
