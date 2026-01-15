#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "jax",
#     "torch",
#     "warp-lang",
# ]
# ///
"""Self-contained test for colight_serde.arrays module.

Run with: uv run packages/colight-serde/tests/test_arrays_standalone.py

This script uses inline script metadata (PEP 723) to declare dependencies,
so it can run without modifying the project's pyproject.toml.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Add the src directory to the path so we can import colight_serde
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np

# Import array libraries BEFORE colight_serde.arrays so auto-registration works
import jax  # noqa: F401
import torch  # noqa: F401

# Now import arrays module - it will auto-register jax and torch
from colight_serde.arrays import (
    is_array_like,
    register_array_converter,
    register_jax,
    register_torch,
    register_warp,
    to_numpy,
)


def test_numpy_array():
    """Test that numpy arrays pass through unchanged."""
    arr = np.array([1, 2, 3])
    result = to_numpy(arr)
    assert result is arr  # Should be the same object
    assert is_array_like(arr)


def test_numpy_array_2d():
    """Test 2D numpy arrays."""
    arr = np.array([[1, 2], [3, 4]])
    result = to_numpy(arr)
    assert result is arr
    assert is_array_like(arr)


def test_non_array():
    """Test that non-arrays return None."""
    assert to_numpy("hello") is None
    assert to_numpy(123) is None
    assert to_numpy([1, 2, 3]) is None  # Regular Python list
    assert not is_array_like("hello")
    assert not is_array_like(123)


def test_jax_array():
    """Test JAX array conversion."""
    import jax.numpy as jnp

    arr = jnp.array([1.0, 2.0, 3.0])
    result = to_numpy(arr)

    assert result is not None
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    assert is_array_like(arr)


def test_jax_array_2d():
    """Test 2D JAX array conversion."""
    import jax.numpy as jnp

    arr = jnp.array([[1, 2], [3, 4]])
    result = to_numpy(arr)

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


def test_torch_tensor():
    """Test PyTorch tensor conversion."""
    import torch

    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = to_numpy(tensor)

    assert result is not None
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    assert is_array_like(tensor)


def test_torch_tensor_requires_grad():
    """Test PyTorch tensor with requires_grad=True."""
    import torch

    tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = to_numpy(tensor)

    assert result is not None
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_torch_cuda_tensor():
    """Test PyTorch CUDA tensor (if available)."""

    if not torch.cuda.is_available():
        print("  (CUDA not available, skipping)")
        return

    tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    result = to_numpy(tensor)

    assert result is not None
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_custom_converter_priority():
    """Test that custom converters with higher priority are checked first."""
    call_order = []

    class CustomArray:
        def __init__(self, data):
            self.data = np.array(data)

    def check_custom(value):
        call_order.append("check_custom")
        return isinstance(value, CustomArray)

    def convert_custom(value):
        call_order.append("convert_custom")
        return value.data * 2  # Double values to verify this converter was used

    # Register with default priority (0, higher than built-ins at -10)
    register_array_converter(check_custom, convert_custom)

    arr = CustomArray([1, 2, 3])
    result = to_numpy(arr)

    assert result is not None
    np.testing.assert_array_equal(result, [2, 4, 6])  # Doubled
    assert "check_custom" in call_order
    assert "convert_custom" in call_order


def test_array_with_dunder_array():
    """Test objects implementing __array__ protocol."""

    class ArrayLike:
        def __init__(self, data):
            self._data = data

        def __array__(self):
            return np.array(self._data)

    arr = ArrayLike([1, 2, 3])
    result = to_numpy(arr)

    assert result is not None
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])
    assert is_array_like(arr)


def test_warp_array():
    """Test NVIDIA Warp array conversion."""
    try:
        import warp as wp
    except ImportError:
        print("  (warp not available, skipping)")
        return

    # Register warp since it was imported after arrays module
    register_warp()
    wp.init()

    arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
    result = to_numpy(arr)

    assert result is not None
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    assert is_array_like(arr)


def test_warp_array_2d():
    """Test 2D NVIDIA Warp array conversion."""
    try:
        import warp as wp
    except ImportError:
        print("  (warp not available, skipping)")
        return

    wp.init()

    arr = wp.array([[1, 2], [3, 4]], dtype=wp.int32)
    result = to_numpy(arr)

    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


def test_auto_registration():
    """Test that auto-registration works for pre-imported libraries."""
    from colight_serde.arrays import _REGISTERED

    # JAX and torch were imported before arrays module, so should be registered
    assert "jax" in _REGISTERED, "JAX should be auto-registered"
    assert "torch" in _REGISTERED, "torch should be auto-registered"


def test_manual_registration_idempotent():
    """Test that calling register_* multiple times is safe."""
    from colight_serde.arrays import _CONVERTERS

    initial_count = len(_CONVERTERS)

    # Call register functions again
    register_jax()
    register_torch()
    register_warp()

    # Should not add duplicate converters
    assert len(_CONVERTERS) == initial_count, "Duplicate converters were added"


def test_mixed_dtypes():
    """Test arrays with different dtypes."""
    import jax.numpy as jnp

    # Float32
    jax_f32 = jnp.array([1.0, 2.0], dtype=jnp.float32)
    torch_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)

    result_jax = to_numpy(jax_f32)
    result_torch = to_numpy(torch_f32)

    assert result_jax.dtype == np.float32
    assert result_torch.dtype == np.float32

    # Int32 (JAX defaults to 32-bit on some platforms without x64 enabled)
    jax_i32 = jnp.array([1, 2], dtype=jnp.int32)
    torch_i32 = torch.tensor([1, 2], dtype=torch.int32)

    result_jax = to_numpy(jax_i32)
    result_torch = to_numpy(torch_i32)

    assert result_jax.dtype == np.int32
    assert result_torch.dtype == np.int32


def run_tests():
    """Run all tests and report results."""
    tests = [
        test_numpy_array,
        test_numpy_array_2d,
        test_non_array,
        test_jax_array,
        test_jax_array_2d,
        test_torch_tensor,
        test_torch_tensor_requires_grad,
        test_torch_cuda_tensor,
        test_warp_array,
        test_warp_array_2d,
        test_custom_converter_priority,
        test_array_with_dunder_array,
        test_auto_registration,
        test_manual_registration_idempotent,
        test_mixed_dtypes,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
