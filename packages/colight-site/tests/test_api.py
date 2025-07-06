"""Test the public API."""

import pathlib
import tempfile
from colight_site import api


def test_evaluate_python():
    """Test the main evaluate_python API."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create a test file
        test_file = temp_path / "test.colight.py"
        test_content = """# Test API

import numpy as np

# Create data
x = np.array([1, 2, 3, 4, 5])
x  # Visualize
"""
        test_file.write_text(test_content)

        # Process the file
        result = api.evaluate_python(
            test_file,
            output_dir=temp_path / "output",
            inline_threshold=100,  # Force file saving (lower than 520 bytes)
            verbose=True,
        )

        # Check result structure
        assert isinstance(result, api.EvaluatedPython)
        assert len(result.forms) == 2  # Import and combined assignment+expression
        assert result.markdown_content is not None
        assert "Test API" in result.markdown_content

        # Check that visualization was processed
        viz_form = result.forms[1]  # The combined form
        assert viz_form.visual_data is not None

        # Since we set a low threshold, it should be saved as a file
        assert isinstance(viz_form.visual_data, pathlib.Path)
        assert viz_form.visual_data.exists()
        assert viz_form.visual_data.name == "form-001.colight"


def test_evaluate_python_with_inline():
    """Test processing with inline visualizations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create a test file
        test_file = temp_path / "test.colight.py"
        test_content = """import numpy as np
x = np.array([1, 2, 3])
x
"""
        test_file.write_text(test_content)

        # Process with high threshold to force inlining
        result = api.evaluate_python(
            test_file,
            inline_threshold=100000,  # Force inlining
        )

        # Check that visualization was inlined
        viz_form = result.forms[0]  # The single combined form
        assert viz_form.visual_data is not None
        assert isinstance(viz_form.visual_data, bytes)
        assert viz_form.visual_data.startswith(b"COLIGHT\x00")

        # Check markdown has inline script
        assert '<script type="application/x-colight">' in result.markdown_content


def test_build_file_api():
    """Test the convenience build_file API."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create test file
        test_file = temp_path / "test.colight.py"
        test_file.write_text("# Test\nimport numpy as np")

        output_file = temp_path / "test.md"

        # Build the file
        api.build_file(test_file, output_file, verbose=False)

        # Check output exists
        assert output_file.exists()
        content = output_file.read_text()
        assert "Test" in content


def test_api_imports():
    """Test that all public API functions are importable."""
    # These should all be available from the main module
    from colight_site import (
        evaluate_python,
        is_colight_file,
        build_file,
        build_directory,
        init_project,
        get_output_path,
    )

    # Verify they are the right types
    assert callable(evaluate_python)
    assert callable(is_colight_file)
    assert callable(build_file)
    assert callable(build_directory)
    assert callable(init_project)
    assert callable(get_output_path)
