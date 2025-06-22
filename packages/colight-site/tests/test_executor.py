"""Test the executor module - actual Python code evaluation."""

import pathlib
import tempfile
from colight_site.executor import FormExecutor, SafeFormExecutor
from colight_site.parser import parse_colight_file


def test_basic_execution():
    """Test basic Python code execution."""
    executor = FormExecutor()

    # Test simple statement
    content = """x = 42"""
    forms = _parse_content(content)

    result = executor.execute_form(forms[0])
    assert result is None  # Statements return None
    assert executor.env["x"] == 42


def test_expression_evaluation():
    """Test expression evaluation and result capture."""
    executor = FormExecutor()

    # Setup variable
    executor.env["x"] = 10

    # Test expression
    content = """x * 2"""
    forms = _parse_content(content)

    result = executor.execute_form(forms[0])
    assert result == 20


def test_persistent_namespace():
    """Test that execution maintains persistent state across forms."""
    executor = FormExecutor()

    content = """
import math

# Define a variable
radius = 5

# Calculate area (this should be an expression)
math.pi * radius ** 2
"""

    forms = _parse_content(content)

    # Execute import (form 0)
    result1 = executor.execute_form(forms[0])
    assert result1 is None
    assert "math" in executor.env

    # Execute variable assignment (form 1)
    result2 = executor.execute_form(forms[1])
    assert result2 is None
    assert executor.env["radius"] == 5

    # Execute expression using previously defined variables (form 2)
    result3 = executor.execute_form(forms[2])
    assert result3 is not None
    assert abs(result3 - (3.14159 * 25)) < 0.1  # Approximately pi * 25


def test_numpy_arrays():
    """Test handling of numpy arrays (common colight use case)."""
    executor = FormExecutor()

    content = """
import numpy as np

# Create array
x = np.linspace(0, 10, 5)

# Return array data
x, np.sin(x)
"""

    forms = _parse_content(content)

    # Execute import (form 0)
    executor.execute_form(forms[0])

    # Execute array creation (form 1)
    executor.execute_form(forms[1])

    # Execute expression that returns tuple of arrays (form 2)
    result = executor.execute_form(forms[2])

    assert result is not None
    assert len(result) == 2  # tuple of (x, sin(x))

    x_array, sin_array = result
    assert len(x_array) == 5
    assert len(sin_array) == 5


def test_colight_visualization_bytes():
    """Test getting visualizations as bytes."""
    executor = FormExecutor()

    # Create some data that should be visualizable
    executor.env["data"] = [1, 2, 3, 4, 5]

    content = """data"""
    forms = _parse_content(content)

    result = executor.execute_form(forms[0])
    assert result == [1, 2, 3, 4, 5]

    # Test getting visualization as bytes
    colight_bytes = executor.get_colight_bytes(result)

    assert colight_bytes is not None
    assert isinstance(colight_bytes, bytes)
    assert colight_bytes.startswith(b"COLIGHT\x00")  # Check magic bytes

    # Save to temporary file to test parsing
    with tempfile.NamedTemporaryFile(suffix=".colight", delete=False) as f:
        f.write(colight_bytes)
        temp_path = pathlib.Path(f.name)

    try:
        # The file should be parseable by the colight format module
        from colight.format import parse_file

        json_data, buffers = parse_file(temp_path)
        assert "ast" in json_data  # Should have AST structure
        assert "display_as" in json_data  # Should have display preferences
    finally:
        temp_path.unlink()


def test_error_handling():
    """Test error handling in code execution."""
    executor = SafeFormExecutor()

    # Test runtime error (can't test syntax errors since LibCST would fail to parse)
    content = """undefined_variable"""
    forms = _parse_content(content)

    try:
        executor.execute_form(forms[0])
        assert False, "Should have raised an exception"
    except NameError:
        pass  # Expected

    # Test division by zero
    content = """1 / 0"""
    forms = _parse_content(content)

    try:
        executor.execute_form(forms[0])
        assert False, "Should have raised an exception"
    except ZeroDivisionError:
        pass  # Expected


def test_environment_isolation():
    """Test that each executor has its own environment."""
    executor1 = FormExecutor()
    executor2 = FormExecutor()

    # Set variable in first executor
    content = '''test_var = "executor1"'''
    forms = _parse_content(content)
    executor1.execute_form(forms[0])

    # Check it doesn't exist in second executor
    content = """test_var"""
    forms = _parse_content(content)

    try:
        executor2.execute_form(forms[0])
        assert False, "Should have raised NameError"
    except NameError:
        pass  # Expected - variable not defined in executor2


def _parse_content(content: str):
    """Helper to parse content into forms."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))
        pathlib.Path(f.name).unlink()
        return forms
