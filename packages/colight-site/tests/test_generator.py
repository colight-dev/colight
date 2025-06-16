"""Test the generator module."""

import pathlib
import tempfile
from colight_site.generator import MarkdownGenerator
from colight_site.parser import Form
import libcst as cst


def test_markdown_generation():
    """Test basic markdown generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        generator = MarkdownGenerator(output_dir)

        # Create mock forms
        import_stmt = cst.parse_statement("import numpy as np")
        expr_stmt = cst.parse_statement("np.sin(x)")

        forms = [
            Form(markdown=["This is a test"], node=import_stmt, start_line=1),
            Form(markdown=["Create visualization"], node=expr_stmt, start_line=3),
        ]

        colight_files = [None, pathlib.Path("test.colight")]

        markdown = generator.generate_markdown(forms, colight_files, "Test Document")

        assert "# Test Document" in markdown
        assert "This is a test" in markdown
        assert "Create visualization" in markdown
        assert "```python" in markdown
        assert "import numpy as np" in markdown
        assert "np.sin(x)" in markdown
        assert 'data-src="test.colight"' in markdown


def test_html_template():
    """Test HTML template generation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        generator = MarkdownGenerator(output_dir)

        markdown = "# Test\n\nSome content"
        html = generator.generate_html_template(markdown, "Test Title")

        assert "<!DOCTYPE html>" in html
        assert "<title>Test Title</title>" in html
        assert "colight-embed" in html
        assert "marked.parse" in html
