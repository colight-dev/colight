"""Test embed threshold functionality."""

import pathlib
import tempfile
from typing import List, Optional
from pathlib import Path
from colight_site.generator import MarkdownGenerator
from colight_site.parser import Form, FormMetadata
import libcst as cst
import re


def _create_test_form(code: str, start_line: int) -> Form:
    """Helper to create a test form from code string."""
    # Parse the code into CST
    module = cst.parse_module(code)

    # Get the first statement
    if module.body:
        node = module.body[0]
    else:
        node = cst.SimpleStatementLine([])

    return Form(markdown=[], node=node, start_line=start_line, metadata=FormMetadata())


def test_inline_threshold():
    """Test that small files are embedded as script tags and large files as external references."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)

        # Create a generator with a low threshold for testing
        generator = MarkdownGenerator(tmpdir_path, inline_threshold=100)

        # Create test forms
        forms = [_create_test_form("x, y", 1), _create_test_form("data", 2)]

        # Create test colight files
        small_file = tmpdir_path / "small.colight"
        large_file = tmpdir_path / "large.colight"

        # Write small file (< 100 bytes)
        small_file.write_bytes(b"COLIGHT\x00" + b"\x00" * 50)  # 58 bytes

        # Write large file (> 100 bytes)
        large_file.write_bytes(b"COLIGHT\x00" + b"\x00" * 200)  # 208 bytes

        colight_files: List[Optional[Path]] = [small_file, large_file]

        # Generate markdown
        markdown_content = generator.generate_markdown(
            forms, colight_files, title="Test", path_context={"basename": "test"}
        )

        # Check that small file is embedded as script tag
        assert '<script type="application/x-colight">' in markdown_content
        assert "</script>" in markdown_content

        # Check that large file uses external reference
        assert (
            '<div class="colight-embed" data-src="test_colight/form-001.colight"></div>'
            in markdown_content
        )

        # Count occurrences
        script_tags = len(
            re.findall(r'<script[^>]*type="application/x-colight"', markdown_content)
        )
        div_tags = len(
            re.findall(
                r'<div[^>]*class="colight-embed"[^>]*data-src=', markdown_content
            )
        )

        assert script_tags == 1  # One small file
        assert div_tags == 1  # One large file


def test_default_inline_threshold():
    """Test default embed threshold of 50KB."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)

        # Create a generator with default threshold
        generator = MarkdownGenerator(tmpdir_path)  # Should use 50000 bytes default

        assert generator.inline_threshold == 50000


def test_inline_threshold_html_generation():
    """Test that embed threshold works for HTML generation too."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)

        # Create a generator with a low threshold
        generator = MarkdownGenerator(tmpdir_path, inline_threshold=100)

        # Create test form
        forms = [_create_test_form("x", 1)]

        # Create small colight file
        small_file = tmpdir_path / "small.colight"
        small_file.write_bytes(b"COLIGHT\x00" + b"\x00" * 50)

        # Generate HTML
        html_content = generator.generate_html(
            forms, [small_file], title="Test HTML", path_context={"basename": "test"}
        )

        # Check that the script tag is in the HTML
        assert '<script type="application/x-colight">' in html_content


def test_base64_encoding():
    """Test that embedded files are properly base64 encoded."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = pathlib.Path(tmpdir)

        # Create a generator with a high threshold
        generator = MarkdownGenerator(tmpdir_path, inline_threshold=1000)

        # Create test form
        forms = [_create_test_form("x", 1)]

        # Create a colight file with known content
        test_file = tmpdir_path / "test.colight"
        test_content = b"COLIGHT\x00TEST"
        test_file.write_bytes(test_content)

        # Generate markdown
        markdown_content = generator.generate_markdown(
            forms, [test_file], title="Test", path_context={"basename": "test"}
        )

        # Extract the base64 content
        import base64

        match = re.search(r"<script[^>]*>([^<]+)</script>", markdown_content, re.DOTALL)
        assert match is not None

        base64_content = match.group(1).strip()
        decoded = base64.b64decode(base64_content)

        # Verify the decoded content matches
        assert decoded == test_content
