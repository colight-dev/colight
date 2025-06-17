"""Test the executor module - actual Python code evaluation."""

import pathlib
import tempfile
import json
from colight_site.executor import FormExecutor, SafeFormExecutor
from colight_site.parser import parse_colight_file


def test_basic_execution():
    """Test basic Python code execution."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        executor = FormExecutor(output_dir)

        # Test simple statement
        content = """x = 42"""
        forms = _parse_content(content)

        result = executor.execute_form(forms[0])
        assert result is None  # Statements return None
        assert executor.env["x"] == 42


def test_expression_evaluation():
    """Test expression evaluation and result capture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        executor = FormExecutor(output_dir)

        # Setup variable
        executor.env["x"] = 10

        # Test expression
        content = """x * 2"""
        forms = _parse_content(content)

        result = executor.execute_form(forms[0])
        assert result == 20


def test_persistent_namespace():
    """Test that execution maintains persistent state across forms."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        executor = FormExecutor(output_dir)

        content = """
import math

# Define a variable
radius = 5

# Calculate area (this should be an expression)
math.pi * radius ** 2
"""

        forms = _parse_content(content)

        # Execute import
        result1 = executor.execute_form(forms[0])
        assert result1 is None
        assert "math" in executor.env

        # Execute variable assignment
        result2 = executor.execute_form(forms[1])
        assert result2 is None
        assert executor.env["radius"] == 5

        # Execute expression using previously defined variables
        result3 = executor.execute_form(forms[2])
        assert result3 is not None
        assert abs(result3 - (3.14159 * 25)) < 0.1  # Approximately pi * 25


def test_numpy_arrays():
    """Test handling of numpy arrays (common colight use case)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        executor = FormExecutor(output_dir)

        content = """
import numpy as np

# Create array
x = np.linspace(0, 10, 5)

# Return array data
x, np.sin(x)
"""

        forms = _parse_content(content)

        # Execute import
        executor.execute_form(forms[0])

        # Execute array creation
        executor.execute_form(forms[1])

        # Execute expression that returns tuple of arrays
        result = executor.execute_form(forms[2])

        assert result is not None
        assert len(result) == 2  # tuple of (x, sin(x))

        x_array, sin_array = result
        assert len(x_array) == 5
        assert len(sin_array) == 5


def test_colight_visualization_saving():
    """Test saving of visualizations to .colight files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        executor = FormExecutor(output_dir)

        # Create some data that should be visualizable
        executor.env["data"] = [1, 2, 3, 4, 5]

        content = """data"""
        forms = _parse_content(content)

        result = executor.execute_form(forms[0])
        assert result == [1, 2, 3, 4, 5]

        # Test saving visualization
        colight_file = executor.save_colight_visualization(result, 0)

        assert colight_file is not None
        assert colight_file.exists()
        assert colight_file.name == "form-000.colight"

        # Verify file content
        content = json.loads(colight_file.read_text())
        assert (
            content["data"] == [1, 2, 3, 4, 5] or content["repr"] == "[1, 2, 3, 4, 5]"
        )


def test_error_handling():
    """Test error handling in code execution."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        executor = SafeFormExecutor(output_dir)

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


def test_non_visualizable_returns():
    """Test handling of return values that aren't visualizable."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        executor = FormExecutor(output_dir)

        # Test basic types that shouldn't be visualized
        test_cases = [("42", 42), ('"hello"', "hello"), ("True", True), ("None", None)]

        for code, expected in test_cases:
            forms = _parse_content(code)
            result = executor.execute_form(forms[0])
            assert result == expected

            # These shouldn't create colight files
            colight_file = executor.save_colight_visualization(result, 0)
            assert colight_file is None


def test_environment_isolation():
    """Test that each executor has its own environment."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)

        executor1 = FormExecutor(output_dir)
        executor2 = FormExecutor(output_dir)

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

        forms = parse_colight_file(pathlib.Path(f.name))
        pathlib.Path(f.name).unlink()
        return forms
