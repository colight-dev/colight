"""End-to-end integration tests for colight-site."""

import pathlib
import tempfile
from colight_site.builder import build_file, init_project
from colight_site.parser import is_colight_file


def test_end_to_end_build():
    """Test complete workflow from .colight.py to markdown + visualizations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create a test .colight.py file
        input_file = temp_path / "test.colight.py"
        input_content = """# Data Visualization Example
# This demonstrates creating a simple plot with numpy.

import numpy as np

# Generate sample data
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)

# Create visualization data
x, y  # This should generate a .colight file

# Additional narrative
# We can also do calculations:

# Calculate some statistics
np.mean(y)

print("Processing complete!")
"""

        input_file.write_text(input_content)

        # Build the file
        output_file = temp_path / "test.md"
        build_file(input_file, output_file, verbose=True)

        # Verify markdown output exists
        assert output_file.exists()
        markdown_content = output_file.read_text()

        # Check markdown structure
        assert "# Test" in markdown_content
        assert "Data Visualization Example" in markdown_content
        assert "```python" in markdown_content
        assert "import numpy as np" in markdown_content
        assert "np.linspace" in markdown_content
        assert "Additional narrative" in markdown_content

        # Check for colight embed - can be either inline script or external embed
        has_inline_script = '<script type="application/x-colight">' in markdown_content
        has_external_embed = (
            'class="colight-embed"' in markdown_content
            and "data-src=" in markdown_content
        )
        assert (
            has_inline_script or has_external_embed
        ), "Should have either inline script or external embed"

        # Verify colight handling
        colight_dir = temp_path / "test_colight"

        # With the new optimization, small files are inlined and not saved to disk
        if has_inline_script:
            # Files were inlined - directory might not exist or be empty
            pass  # This is expected behavior for small files
        else:
            # Files were saved to disk
            assert colight_dir.exists()
            colight_files = list(colight_dir.glob("*.colight"))
            assert len(colight_files) >= 1  # At least one visualization

            # Check first colight file content - it should be a binary .colight file
            first_colight = colight_files[0]
            file_content = first_colight.read_bytes()
            assert file_content.startswith(b"COLIGHT\x00")  # Check magic bytes

            # The file should be parseable by the colight format module
            from colight.format import parse_file

            json_data, buffers = parse_file(first_colight)
            assert "ast" in json_data  # Should have AST structure


def test_project_initialization():
    """Test project initialization creates correct structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = pathlib.Path(temp_dir) / "test_project"

        init_project(project_path)

        # Check directory structure
        assert project_path.exists()
        assert (project_path / "src").exists()
        assert (project_path / "build").exists()
        assert (project_path / "README.md").exists()

        # Check example file
        example_file = project_path / "src" / "example.colight.py"
        assert example_file.exists()
        assert is_colight_file(example_file)

        # Check README content
        readme_content = (project_path / "README.md").read_text()
        assert "colight-site" in readme_content
        assert "build src" in readme_content


def test_error_recovery():
    """Test that build process handles errors gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        # Create a file with syntax error
        input_file = temp_path / "broken.colight.py"
        input_content = """# This file has errors
# But some parts should still work

import numpy as np

# This works
x = np.array([1, 2, 3])

# This has a syntax error - unclosed parenthesis
z = (

# This should work if reached
x.sum()
"""

        input_file.write_text(input_content)
        output_file = temp_path / "broken.md"

        # Build should handle errors gracefully
        try:
            build_file(input_file, output_file, verbose=True)
        except Exception:
            pass  # Errors are expected, but shouldn't crash the whole process

        # Should still create some output with error message
        assert output_file.exists()
        content = output_file.read_text()
        assert "Parse Error" in content  # Error message from builder
        assert "broken.colight.py" in content  # Filename in error message


def test_multiple_visualizations():
    """Test file with multiple visualization-generating expressions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        input_file = temp_path / "multi_viz.colight.py"
        input_content = """# Multiple Visualizations
# This file creates several visualizations.

import numpy as np

# First dataset
x1 = np.linspace(0, 10, 20)
y1 = np.sin(x1)

# First visualization
x1, y1

# Second dataset
x2 = np.linspace(0, 5, 15) 
y2 = np.cos(x2)

# Second visualization  
x2, y2

# Combined data
combined = np.column_stack([x1[:15], y1[:15], x2, y2])

# Third visualization
combined
"""

        input_file.write_text(input_content)
        output_file = temp_path / "multi_viz.md"

        build_file(input_file, output_file, verbose=True)

        # Check output
        markdown_content = output_file.read_text()

        # Should have multiple colight embeds (either inline scripts or external embeds)
        inline_count = markdown_content.count('<script type="application/x-colight">')
        external_count = markdown_content.count('class="colight-embed"')
        total_embeds = inline_count + external_count
        assert total_embeds >= 3  # At least 3 visualizations

        # Check colight handling - with optimization, small files might be inlined
        colight_dir = temp_path / "multi_viz_colight"
        if colight_dir.exists():
            # Small files are inlined, so we might have fewer files on disk
            # The important thing is that we have the visualizations embedded
            assert total_embeds >= 3  # Already checked above
        else:
            # All files were inlined - this is fine
            assert inline_count >= 3


def test_empty_and_comment_only_forms():
    """Test handling of forms with only comments or empty code."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        input_file = temp_path / "comments.colight.py"
        input_content = """# Title Comment
# This is just narrative text.

# More comments
# With multiple lines.

# Final section
# Just comments, no code.
"""

        input_file.write_text(input_content)
        output_file = temp_path / "comments.md"

        build_file(input_file, output_file, verbose=True)

        # Should create markdown even with no executable code
        assert output_file.exists()
        content = output_file.read_text()

        assert "Title Comment" in content
        assert "More comments" in content
        assert "Final section" in content
