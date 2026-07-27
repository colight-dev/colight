"""Tests for the inspect module."""

import pytest
from datetime import datetime, date, time
from colight.inspect import inspect, _serialize_value, _get_type_info


class TestTypeInfo:
    """Test type information extraction."""

    def test_builtin_types(self):
        """Test type info for built-in types."""
        assert _get_type_info(42) == {
            "type": "int",
            "category": "builtin",
            "module": "builtins",
        }
        assert _get_type_info("hello") == {
            "type": "str",
            "category": "builtin",
            "module": "builtins",
        }
        assert _get_type_info([1, 2, 3]) == {
            "type": "list",
            "category": "builtin",
            "module": "builtins",
        }
        assert _get_type_info({1: 2}) == {
            "type": "dict",
            "category": "builtin",
            "module": "builtins",
        }

    def test_numpy_types(self):
        """Test type info for numpy types."""
        try:
            import numpy as np

            arr = np.array([1, 2, 3])
            info = _get_type_info(arr)
            assert info["type"] == "ndarray"
            assert info["category"] == "numpy"
            assert info["module"] == "numpy"
        except ImportError:
            pytest.skip("NumPy not available")

    def test_custom_types(self):
        """Test type info for custom types."""

        class CustomClass:
            pass

        obj = CustomClass()
        info = _get_type_info(obj)
        assert info["type"] == "CustomClass"
        assert info["category"] == "custom"
        assert "test_inspect" in info["module"]


class TestSerialization:
    """Test value serialization."""

    def test_primitives(self):
        """Test serialization of primitive types."""
        # None
        result = _serialize_value(None)
        assert result is None

        # Boolean
        result = _serialize_value(True)
        assert result is True

        # Integer
        result = _serialize_value(42)
        assert result == 42

        # Float
        result = _serialize_value(3.14)
        assert result == 3.14

        # String
        result = _serialize_value("hello")
        assert result == "hello"

    def test_bytes(self):
        """Test serialization of bytes."""
        data = b"hello world"
        result = _serialize_value(data)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "bytes"
        assert result["value"] == data.hex()
        assert result["length"] == len(data)
        assert result["truncated"] is False

        # Test truncation
        long_data = b"x" * 100
        result = _serialize_value(long_data)
        assert isinstance(result, dict)
        assert result["truncated"] is True
        assert len(result["value"]) == 100  # 50 bytes * 2 chars per byte in hex

    def test_datetime_types(self):
        """Test serialization of datetime types."""
        now = datetime.now()
        result = _serialize_value(now)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "datetime"
        assert result["value"] == str(now)
        assert result["iso"] == now.isoformat()

        today = date.today()
        result = _serialize_value(today)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "date"
        assert result["value"] == str(today)

        current_time = time(14, 30, 0)
        assert isinstance(result, dict)
        result = _serialize_value(current_time)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "time"
        assert result["value"] == str(current_time)

    def test_collections(self):
        """Test serialization of collections."""
        # List
        lst = [1, 2, 3, 4, 5]
        result = _serialize_value(lst)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "list"
        assert result["length"] == 5
        assert len(result["value"]) == 5
        assert result["truncated"] is False

        # Tuple
        tpl = (1, 2, 3)
        result = _serialize_value(tpl)
        assert isinstance(result, dict)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "tuple"
        assert result["length"] == 3

        # Set
        s = {1, 2, 3}
        result = _serialize_value(s)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "set"
        assert result["length"] == 3

        # Dict
        d = {"a": 1, "b": 2}
        result = _serialize_value(d)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "dict"
        assert result["length"] == 2
        assert len(result["value"]) == 2
        assert result["value"][0]["key"] == "a"
        assert result["value"][0]["value"] == 1

    def test_truncation(self):
        """Test truncation of large collections."""
        # Large list
        large_list = list(range(200))
        result = _serialize_value(large_list, max_items=50)
        assert isinstance(result, dict)
        assert result["length"] == 200
        assert len(result["value"]) == 50
        assert result["truncated"] is True

        # Large dict
        large_dict = {str(i): i for i in range(200)}
        result = _serialize_value(large_dict, max_items=50)
        assert isinstance(result, dict)
        assert result["length"] == 200
        assert len(result["value"]) == 50
        assert result["truncated"] is True

    def test_nested_structures(self):
        """Test serialization of nested structures."""
        nested = {
            "list": [1, 2, {"inner": "value"}],
            "dict": {"a": 1, "b": [2, 3]},
            "tuple": (1, 2, 3),
        }
        result = _serialize_value(nested)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "dict"
        assert result["length"] == 3

        # Check nested list
        list_item = next(item for item in result["value"] if item["key"] == "list")
        assert list_item["value"]["type_info"]["type"] == "list"
        assert list_item["value"]["value"][2]["type_info"]["type"] == "dict"

    def test_max_depth(self):
        """Test max depth limiting."""
        deeply_nested = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        result = _serialize_value(deeply_nested, max_depth=3)
        assert isinstance(result, dict)

        # Navigate to the truncated part
        current = result
        for _ in range(3):
            current = current["value"][0]["value"]

        assert current["truncated"] is True
        assert "max depth" in current["value"]

    def test_numpy_arrays(self):
        """Test serialization of numpy arrays."""
        try:
            import numpy as np

            # 1D array
            arr1d = np.array([1, 2, 3, 4, 5])
            result = _serialize_value(arr1d)
            assert isinstance(result, dict)
            assert result["type_info"]["type"] == "ndarray"
            assert result["type_info"]["category"] == "numpy"
            assert result["shape"] == (5,)
            assert result["dtype"] == "int64"
            assert result["value"] == [1, 2, 3, 4, 5]

            # 2D array
            arr2d = np.array([[1, 2], [3, 4]])
            result = _serialize_value(arr2d)
            assert isinstance(result, dict)
            assert result["shape"] == (2, 2)
            assert result["value"] == [[1, 2], [3, 4]]

            # Large array (should be truncated)
            large_arr = np.zeros(1000)
            result = _serialize_value(large_arr, max_items=100)
            assert isinstance(result, dict)
            assert result["truncated"] is True
            assert result["size"] == 1000

        except ImportError:
            pytest.skip("NumPy not available")

    def test_custom_objects(self):
        """Test serialization of custom objects."""

        class TestObject:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42

            def __str__(self):
                return "TestObject instance"

        obj = TestObject()
        result = _serialize_value(obj)
        assert isinstance(result, dict)
        assert result["type_info"]["type"] == "TestObject"
        assert result["type_info"]["category"] == "custom"
        assert result["value"] == "TestObject instance"
        assert len(result["attributes"]) == 2

        # Check attributes
        attrs = {attr["key"]: attr["value"] for attr in result["attributes"]}
        assert attrs["attr1"] == "value1"
        assert attrs["attr2"] == 42


class TestInspectFunction:
    """Test the main inspect function."""

    def test_inspect_returns_hiccup_for_complex_types(self):
        """Test that inspect returns a Hiccup object for complex types."""
        from colight.layout import Hiccup, JSCode

        # Lists should get the inspect visualization
        result = inspect([1, 2, 3])
        assert isinstance(result, Hiccup)

        # Check the internal structure
        assert hasattr(result, "hiccup_element")
        assert isinstance(result.hiccup_element, list)
        assert len(result.hiccup_element) == 2

        # First element is the Plot.js reference (a JSCode object)
        js_ref = result.hiccup_element[0]
        assert isinstance(js_ref, JSCode)
        assert js_ref.code == "colight.api.inspect"
        assert js_ref.expression is True

        # Second element contains the data
        assert isinstance(result.hiccup_element[1], dict)
        assert "data" in result.hiccup_element[1]

    def test_inspect_with_options(self):
        """Test inspect with custom options."""
        from colight.layout import Hiccup

        large_list = list(range(1000))
        result = inspect(large_list, max_items=10, max_depth=2)

        assert isinstance(result, Hiccup)

        # Extract the serialized data
        data = result.hiccup_element[1]["data"]
        assert data["length"] == 1000
        assert len(data["value"]) == 10
        assert data["truncated"] is True
