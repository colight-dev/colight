"""Tests for the new generator implementation."""

import textwrap

from colight_prose.executor import DocumentExecutor, ExecutionResult
from colight_prose.parser import parse_document
from colight_prose.static.generator import (
    EMBED_URL,
    HTMLGenerator,
    MarkdownGenerator,
    write_colight_files,
)


class TestMarkdownGenerator:
    """Test the MarkdownGenerator class."""

    def test_simple_generation(self, tmp_path):
        """Test generating simple markdown."""
        source = textwrap.dedent("""
            # Hello World
            
            print("Hello")
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path)
        md = generator.generate(doc, results)

        assert "Hello World" in md  # Prose content is there
        assert "```python" in md
        assert 'print("Hello")' in md

    def test_pragma_hiding(self, tmp_path):
        """Test that pragmas control visibility."""
        # Use file-level pragma with hide-all-
        source = textwrap.dedent("""
            #| hide-all-code
            
            # Title
            x = 42
            x
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path)
        md = generator.generate(doc, results)

        assert "Title" in md
        assert "x = 42" not in md  # Code hidden by file pragma
        assert "```python" not in md

    def test_show_override(self, tmp_path):
        """Test that show pragmas override hide pragmas."""
        source = textwrap.dedent("""
            #| hide-all-code
            
            # Global hide
            x = 1
            
            #| show-code
            # Local show
            y = 2
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path)
        md = generator.generate(doc, results)

        assert "x = 1" not in md  # Hidden by file pragma
        assert "y = 2" in md  # Shown by block pragma

    def test_visualization_inline(self, tmp_path):
        """Test inline visualization embedding."""
        source = "[1, 2, 3]"

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path, inline_threshold=10000)
        md = generator.generate(doc, results)

        assert '<script type="application/x-colight">' in md
        assert "</script>" in md

    def test_visualization_external(self, tmp_path):
        """Test external visualization reference."""
        source = "[1, 2, 3]"

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path, inline_threshold=1)  # Force external
        md = generator.generate(doc, results, {"basename": "test"})

        assert '<div class="colight-embed"' in md
        assert 'data-src="test_colight/block-000.colight"' in md

    def test_error_display(self, tmp_path):
        """Test that errors are displayed."""
        source = "1/0"

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path)
        md = generator.generate(doc, results)

        assert "ZeroDivisionError" in md
        assert "```" in md  # Error in code block

    def test_hide_statements(self, tmp_path):
        """Test hiding statements but showing expressions."""
        # Put pragma with code to ensure it's in the same block
        source = textwrap.dedent("""
            # Code block
            #| hide-statements
            x = 10  # statement
            y = 20  # statement
            x + y   # expression
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path)
        md = generator.generate(doc, results)

        assert "x = 10" not in md
        assert "y = 20" not in md
        assert "x + y" in md

    def test_pragma_with_blank_line(self, tmp_path):
        """Test that pragmas work with blank lines before code."""
        source = textwrap.dedent("""
            #| hide-code
            
            x = 42
            y = x * 2
            y
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path)
        md = generator.generate(doc, results)

        # Code should be hidden by pragma
        assert "x = 42" not in md
        assert "```python" not in md  # No code blocks
        # But visualization should still show if enabled
        assert "application/x-colight" in md

    def test_literal_hiding(self, tmp_path):
        """Test that literal expressions are hidden."""
        source = textwrap.dedent("""
            x = 10
            42  # literal - should be hidden
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path)
        md = generator.generate(doc, results)

        assert "x = 10" in md
        assert "42" not in md

    def test_empty_lines_preserved(self, tmp_path):
        """Test that empty lines between statements are preserved."""
        source = textwrap.dedent("""
            x = 1
            #
            y = 2
            #
            z = 3
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = MarkdownGenerator(tmp_path)
        md = generator.generate(doc, results)

        # Check that the code block has empty lines preserved
        assert "```python" in md
        assert "x = 1\n\ny = 2\n\nz = 3" in md


class TestHTMLGenerator:
    """Test the HTMLGenerator class."""

    def test_html_generation(self, tmp_path):
        """Test generating HTML."""
        source = textwrap.dedent("""
            # Test Document
            
            print("Hello, HTML!")
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = HTMLGenerator(tmp_path)
        html = generator.generate(doc, results, title="Test Page")

        assert "<!DOCTYPE html>" in html
        assert "<title>Test Page</title>" in html
        assert "Test Document" in html  # Content is there as plain text (not h1)
        assert "Hello, HTML!" in html
        assert EMBED_URL in html

    def test_html_with_visualization(self, tmp_path):
        """Test HTML with embedded visualization."""
        source = "[1, 2, 3, 4]"

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        generator = HTMLGenerator(tmp_path)
        html = generator.generate(doc, results)

        assert '<script type="application/x-colight">' in html
        assert 'colight.api.tw("prose")' in html


class TestHelpers:
    """Test helper functions."""

    def test_write_colight_files(self, tmp_path):
        """Test writing colight files."""
        # Create some mock results
        results = [
            ExecutionResult(colight_bytes=b"data1"),
            ExecutionResult(colight_bytes=None),  # No visualization
            ExecutionResult(colight_bytes=b"data3"),
        ]

        output_dir = tmp_path / "output"
        paths = write_colight_files(output_dir, results, basename="test")

        assert len(paths) == 3
        assert paths[0] == output_dir / "test-000.colight"
        assert paths[1] is None
        assert paths[2] == output_dir / "test-002.colight"

        # Check files were written
        assert paths[0] is not None
        assert paths[0].read_bytes() == b"data1"
        assert paths[2] is not None
        assert paths[2].read_bytes() == b"data3"
