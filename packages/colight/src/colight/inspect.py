import colight.plot as Plot
from datetime import datetime, date, time
from typing import Any, Dict


def _get_type_info(value: Any) -> Dict[str, Any]:
    """Extract type information and metadata from a Python value."""
    value_type = type(value).__name__
    module = getattr(type(value), "__module__", None)

    # Handle built-in types
    if module == "builtins":
        type_category = "builtin"
    # Handle NumPy scalars (treat as builtin primitives)
    elif module == "numpy" and hasattr(value, "shape") and value.shape == ():
        type_category = "builtin"  # NumPy scalars behave like primitives
    # Handle NumPy arrays
    elif module == "numpy" and hasattr(value, "shape"):
        type_category = "numpy"
    # Handle JAX scalars (treat as builtin primitives)
    elif module and "jax" in module and hasattr(value, "shape") and value.shape == ():
        type_category = "builtin"  # JAX scalars behave like primitives
    # Handle JAX arrays
    elif module and "jax" in module and hasattr(value, "shape"):
        type_category = "jax"
    # Handle pandas
    elif module == "pandas.core.frame" or module == "pandas.core.series":
        type_category = "pandas"
    else:
        type_category = "custom"

    return {"type": value_type, "category": type_category, "module": module}


def _serialize_value(
    value: Any, max_items: int = 100, max_depth: int = 3, current_depth: int = 0
) -> Dict[str, Any] | bool | int | float | str | None:
    """Serialize a Python value for JavaScript consumption with size limits."""
    # Handle primitives - return them directly
    if isinstance(value, (bool, int, float, str, type(None))):
        return value

    if current_depth >= max_depth:
        return {
            "type_info": _get_type_info(value),
            "truncated": True,
            "value": f"<max depth {max_depth} reached>",
        }

    type_info = _get_type_info(value)

    # Handle bytes
    if isinstance(value, bytes):
        preview = value[:50]
        return {
            "type_info": type_info,
            "value": preview.hex(),
            "length": len(value),
            "truncated": len(value) > 50,
        }

    # Handle datetime objects
    if isinstance(value, (datetime, date, time)):
        return {
            "type_info": type_info,
            "value": str(value),
            "iso": value.isoformat() if hasattr(value, "isoformat") else None,
        }

    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        length = len(value)
        items = []
        for i, item in enumerate(value):
            if i >= max_items:
                break
            items.append(
                _serialize_value(item, max_items, max_depth, current_depth + 1)
            )

        return {
            "type_info": type_info,
            "value": items,
            "length": length,
            "truncated": length > max_items,
        }

    # Handle sets
    if isinstance(value, set):
        length = len(value)
        items = []
        for i, item in enumerate(value):
            if i >= max_items:
                break
            items.append(
                _serialize_value(item, max_items, max_depth, current_depth + 1)
            )

        return {
            "type_info": type_info,
            "value": items,
            "length": length,
            "truncated": length > max_items,
        }

    # Handle dictionaries
    if isinstance(value, dict):
        length = len(value)
        items = []
        for i, (k, v) in enumerate(value.items()):
            if i >= max_items:
                break
            items.append(
                {
                    "key": _serialize_value(k, max_items, max_depth, current_depth + 1),
                    "value": _serialize_value(
                        v, max_items, max_depth, current_depth + 1
                    ),
                }
            )

        return {
            "type_info": type_info,
            "value": items,
            "length": length,
            "truncated": length > max_items,
        }

    # Handle NumPy arrays
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                # Get basic info
                shape = value.shape
                dtype = str(value.dtype)
                size = value.size

                # Convert to list for serialization, with truncation if needed
                if size <= max_items:
                    data_value = value.tolist()
                else:
                    # For large arrays, only convert a sample
                    flat = value.flatten()
                    data_value = flat[:max_items].tolist()

                return {
                    "type_info": type_info,
                    "value": data_value,
                    "shape": shape,
                    "dtype": dtype,
                    "size": size,
                    "truncated": size > max_items,
                }
        except ImportError:
            pass

    # Handle JAX arrays similarly
    if (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and "jax" in str(type(value))
    ):
        try:
            shape = tuple(value.shape)
            dtype = str(value.dtype)
            size = value.size

            # Convert to numpy for easier handling
            if size <= max_items:
                data_value = value.tolist() if hasattr(value, "tolist") else str(value)
            else:
                data_value = "<JAX array too large for preview>"

            return {
                "type_info": type_info,
                "value": data_value,
                "shape": shape,
                "dtype": dtype,
                "size": size,
                "truncated": size > max_items,
            }
        except Exception:
            pass

    # Handle pandas DataFrames and Series
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            shape = value.shape
            columns = list(value.columns)
            dtypes = {col: str(dtype) for col, dtype in value.dtypes.items()}

            # Sample data for preview
            sample_size = min(10, len(value))
            sample_data = value.head(sample_size).to_dict("records")

            return {
                "type_info": type_info,
                "value": sample_data,
                "shape": shape,
                "columns": columns,
                "dtypes": dtypes,
                "truncated": len(value) > sample_size,
            }

        elif isinstance(value, pd.Series):
            shape = (len(value),)
            dtype = str(value.dtype)

            sample_size = min(20, len(value))
            sample_data = value.head(sample_size).tolist()

            return {
                "type_info": type_info,
                "value": sample_data,
                "shape": shape,
                "dtype": dtype,
                "name": value.name,
                "truncated": len(value) > sample_size,
            }
    except ImportError:
        pass

    # Handle other objects - try to extract some useful information
    try:
        # Try to get string representation
        str_repr = str(value)
        if len(str_repr) > 200:
            str_repr = str_repr[:200] + "..."

        # Try to get attributes
        attrs = []
        if hasattr(value, "__dict__"):
            for key, val in list(value.__dict__.items())[:max_items]:
                attrs.append(
                    {
                        "key": key,
                        "value": _serialize_value(
                            val, max_items, max_depth, current_depth + 1
                        ),
                    }
                )

        return {
            "type_info": type_info,
            "value": str_repr,
            "attributes": attrs,
            "truncated": len(attrs) >= max_items or len(str(value)) > 200,
        }
    except Exception as e:
        return {
            "type_info": type_info,
            "value": f"<unable to serialize: {str(e)}>",
            "error": str(e),
        }


def inspect(value: Any, max_items: int = 100, max_depth: int = 5) -> Any:
    """
    Create an interactive visualization for inspecting Python values.

    Args:
        value: The Python value to inspect
        max_items: Maximum number of items to show for collections (default: 100)
        max_depth: Maximum nesting depth to traverse (default: 5)

    Returns:
        A colight visualization component for complex types, or the value itself for primitives
    """
    # Return primitives as-is (like int, float, str, bool, None)
    if isinstance(value, (Plot.LayoutItem)):
        return value

    # Create visualization for complex types
    serialized = _serialize_value(value, max_items, max_depth)
    return Plot.html([Plot.js("colight.api.inspect"), {"data": serialized}])
