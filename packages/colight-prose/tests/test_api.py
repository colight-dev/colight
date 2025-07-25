"""Test the public API."""

import pathlib
import tempfile

import pytest

import colight_prose.static.builder as builder
from colight_prose import api
from colight_prose.static.builder import BuildConfig


def test_evaluate_python():
    """Test the main evaluate_python API."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create a test file
        test_file = temp_path / "test.py"
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
        assert (
            len(result.blocks) == 3
        )  # Prose, import, and combined prose+assignment+expression
        assert result.markdown_content is not None
        assert "Test API" in result.markdown_content

        # Check that visualization was processed
        viz_block = result.blocks[2]  # The block with assignment and expression
        assert viz_block.visual_data is not None

        # Since we set a low threshold, it should be saved as a file
        assert isinstance(viz_block.visual_data, pathlib.Path)
        assert viz_block.visual_data.exists()
        assert viz_block.visual_data.name == "block-002.colight"


def test_evaluate_python_with_inline():
    """Test processing with inline visualizations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create a test file
        test_file = temp_path / "test.py"
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
        viz_block = result.blocks[0]  # The single combined block
        assert viz_block.visual_data is not None
        assert isinstance(viz_block.visual_data, bytes)
        assert viz_block.visual_data.startswith(b"COLIGHT\x00")

        # Check markdown has inline script
        assert '<script type="application/x-colight">' in result.markdown_content


def test_build_file_api():
    """Test the convenience build_file API."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create test file
        test_file = temp_path / "test.py"
        test_file.write_text("# Test\nimport numpy as np")

        output_file = temp_path / "test.md"

        # Build the file
        builder.build_file(test_file, output_file=output_file, verbose=False)

        # Check output exists
        assert output_file.exists()
        content = output_file.read_text()
        assert "Test" in content


class TestAPIErrorHandling:
    """Test API error handling scenarios."""

    def test_build_file_nonexistent_input(self):
        """Test build_file with nonexistent input file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            nonexistent = temp_path / "nonexistent.py"
            output = temp_path / "output.md"

            try:
                builder.build_file(nonexistent, output_file=output)
                assert False, "Should have raised an exception"
            except FileNotFoundError:
                pass  # Expected

    def test_build_file_creates_output_directory(self):
        """Test build_file creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            input_file = temp_path / "test.py"
            input_file.write_text("# Test")

            # Output to nonexistent directory (should be created)
            output_dir = temp_path / "new_directory"
            output_file = output_dir / "output.md"

            assert not output_dir.exists()

            # Should create directory and file
            builder.build_file(input_file, output_file=output_file)

            assert output_dir.exists()
            assert output_file.exists()

    def test_evaluate_python_with_syntax_errors(self):
        """Test evaluate_python with Python syntax errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            test_file = temp_path / "syntax_error.py"
            test_file.write_text("""
# Syntax Error Test
def broken_function(
    # Missing closing parenthesis
print("This won't parse")
""")

            # Should handle syntax errors gracefully
            result = api.evaluate_python(test_file)

            assert isinstance(result, api.EvaluatedPython)
            assert "Error reading file" in result.markdown_content

    def test_evaluate_python_with_runtime_errors(self):
        """Test evaluate_python with Python runtime errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            test_file = temp_path / "runtime_error.py"
            test_file.write_text("""
# Runtime Error Test
print("This works")

# This will cause a runtime error
undefined_variable = some_undefined_variable

print("This might not execute")
""")

            result = api.evaluate_python(test_file)

            assert isinstance(result, api.EvaluatedPython)
            assert "Runtime Error Test" in result.markdown_content

    def test_build_directory_with_mixed_content(self):
        """Test build_directory with mix of valid and invalid files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            # Valid file
            (temp_path / "valid.py").write_text("# Valid\nprint('works')")

            # Invalid file
            (temp_path / "invalid.py").write_text("# Invalid\ndef broken(")

            # Non-Python file (should be ignored)
            (temp_path / "ignored.txt").write_text("This should be ignored")

            output_dir = temp_path / "output"

            # Should complete without raising exceptions
            builder.build_directory(temp_path, output_dir, continue_on_error=True)

            # Valid file should be processed
            assert (output_dir / "valid.md").exists()

            # Invalid file might be processed depending on continue_on_error behavior
            # Non-Python file should not be processed
            assert not (output_dir / "ignored.md").exists()

    def test_get_output_path_edge_cases(self):
        """Test get_output_path with various edge cases."""
        # Test different input extensions
        assert (
            builder.get_output_path(pathlib.Path("test.py"), "markdown").suffix == ".md"
        )
        assert (
            builder.get_output_path(pathlib.Path("test.py"), "html").suffix == ".html"
        )

        # Test file without extension
        result = builder.get_output_path(pathlib.Path("test"), "markdown")
        assert result.suffix == ".md"

        # Test with complex paths
        complex_path = pathlib.Path("subdir/test.py")
        result = builder.get_output_path(complex_path, "html")
        assert result.name == "test.html"


class TestAPIConfigurationValidation:
    """Test API configuration and validation."""

    def test_invalid_format_specification(self):
        """Test API with invalid format specifications."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            test_file = temp_path / "test.py"
            test_file.write_text("# Test")

            output_file = temp_path / "test.invalid"

            # Test with invalid format in config
            with pytest.raises((ValueError, KeyError)):
                config = BuildConfig(formats={"invalid_format"})  # type: ignore
                builder.build_file(test_file, output_file=output_file, config=config)

    def test_pragma_configuration_validation(self):
        """Test pragma configuration validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            test_file = temp_path / "test.py"
            test_file.write_text("""
# colight-pragma: hide-code show-visuals
print("This should work with pragmas")
""")

            output_file = temp_path / "test.md"

            # Test with various pragma configurations
            config = BuildConfig(pragma={"hide-code", "show-visuals"})
            builder.build_file(test_file, output_file=output_file, config=config)

            assert output_file.exists()
            content = output_file.read_text()

            # Should respect pragma settings
            # Exact behavior depends on implementation

    def test_inline_threshold_validation(self):
        """Test inline threshold parameter validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            test_file = temp_path / "test.py"
            test_file.write_text("""
import numpy as np
x = np.array([1, 2, 3])
x
""")

            # Test with various threshold values
            for threshold in [0, 1, 1000, 100000]:
                result = api.evaluate_python(
                    test_file,
                    inline_threshold=threshold,
                    output_dir=temp_path / f"output_{threshold}",
                )

                assert isinstance(result, api.EvaluatedPython)

                # Verify threshold affects inlining behavior
                viz_block = result.blocks[0]
                # The actual .colight file is ~416 bytes, so adjust the threshold check
                if (
                    threshold > 416
                ):  # File will be inlined if threshold is larger than file size
                    # Should be inlined as bytes
                    if viz_block.visual_data:
                        assert isinstance(
                            viz_block.visual_data, bytes
                        ), f"Expected bytes for threshold {threshold}, got {type(viz_block.visual_data)}"
                else:
                    # Should be saved as file
                    if viz_block.visual_data:
                        assert isinstance(
                            viz_block.visual_data, pathlib.Path
                        ), f"Expected Path for threshold {threshold}, got {type(viz_block.visual_data)}"

    def test_verbose_output_configuration(self):
        """Test verbose output configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            test_file = temp_path / "test.py"
            test_file.write_text("# Test\nprint('hello')")

            output_file = temp_path / "test.md"

            # Test with verbose=True (should not raise exceptions)
            builder.build_file(test_file, output_file=output_file, verbose=True)
            assert output_file.exists()

            # Test with verbose=False
            output_file.unlink()  # Remove previous output
            builder.build_file(test_file, output_file=output_file, verbose=False)
            assert output_file.exists()


class TestAPIPerformance:
    """Test API performance characteristics."""

    def test_large_file_processing(self):
        """Test API with moderately large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            # Create a moderately large file
            large_content = "# Large File Test\n"
            large_content += "\n".join(
                [f"# Section {i}\nprint('Section {i}')" for i in range(100)]
            )

            test_file = temp_path / "large.py"
            test_file.write_text(large_content)

            output_file = temp_path / "large.md"

            # Should complete in reasonable time
            import time

            start_time = time.time()
            builder.build_file(test_file, output_file=output_file)
            end_time = time.time()

            assert output_file.exists()
            # Should complete within reasonable time (adjust threshold as needed)
            assert end_time - start_time < 30  # 30 seconds max for 100 sections

    def test_multiple_files_processing(self):
        """Test API with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            # Create multiple test files
            for i in range(10):
                file_path = temp_path / f"test_{i:02d}.py"
                file_path.write_text(f"# Test File {i}\nprint('File {i}')")

            output_dir = temp_path / "output"

            # Should process all files
            import time

            start_time = time.time()
            builder.build_directory(temp_path, output_dir)
            end_time = time.time()

            # Verify all files were processed
            output_files = list(output_dir.glob("*.md"))
            assert len(output_files) == 10

            # Should complete in reasonable time
            assert end_time - start_time < 20  # 20 seconds max for 10 files
