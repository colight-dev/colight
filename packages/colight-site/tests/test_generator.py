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


def test_html_generation():
    """Test HTML generation."""
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

        html = generator.generate_html(forms, colight_files, "Test Document")

        assert "<!DOCTYPE html>" in html
        assert "<title>Test Document</title>" in html
        assert "colight-embed" in html
        assert "@colight/core/embed.js" in html
        assert "<h1>Test Document</h1>" in html
        assert "<p>This is a test</p>" in html
