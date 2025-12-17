"""Smoke tests for the colight publish CLI."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from colight_cli import publish


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def docs_dir(tmp_path):
    """Create a temporary docs directory with sample files."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.md").write_text("# Welcome\n\nHello world.")
    (docs / "example.py").write_text("# Example\nx = 1 + 1\nx")
    return docs


def test_publish_help(runner):
    """Test that --help works."""
    result = runner.invoke(publish, ["--help"])
    assert result.exit_code == 0
    assert "--format" in result.output
    assert "md" in result.output
    assert "html" in result.output
    assert "site" in result.output


def test_publish_requires_format(runner, docs_dir):
    """Test that --format is required."""
    result = runner.invoke(publish, [str(docs_dir)])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "required" in result.output.lower()


def test_publish_md_format_with_python(runner, docs_dir, tmp_path):
    """Test building markdown output from Python files."""
    output_dir = tmp_path / "out"

    result = runner.invoke(
        publish,
        [
            str(docs_dir),
            "--format",
            "md",
            "--output",
            str(output_dir),
            "--include",
            "*.py",
        ],
    )

    assert result.exit_code == 0
    assert output_dir.exists()


def test_publish_site_format(runner, docs_dir, tmp_path):
    """Test building static site output."""
    output_dir = tmp_path / "site-out"

    with patch(
        "colight_publish.static.site_builder._prepare_bundle",
        return_value="/dist/live.js",
    ):
        result = runner.invoke(
            publish,
            [
                str(docs_dir),
                "--format",
                "site",
                "--output",
                str(output_dir),
                "--include",
                "*.md",
            ],
        )

    assert result.exit_code == 0
    assert (output_dir / "index.html").exists()
    assert (output_dir / "api" / "index.json").exists()


def test_publish_html_format_with_python(runner, docs_dir, tmp_path):
    """Test building HTML output from Python files."""
    output_dir = tmp_path / "html-out"

    result = runner.invoke(
        publish,
        [
            str(docs_dir),
            "--format",
            "html",
            "--output",
            str(output_dir),
            "--include",
            "*.py",
        ],
    )

    assert result.exit_code == 0
    assert output_dir.exists()


def test_publish_single_python_file(runner, docs_dir, tmp_path):
    """Test publishing a single Python file."""
    single_file = docs_dir / "example.py"
    output_dir = tmp_path / "single-out"

    result = runner.invoke(
        publish,
        [str(single_file), "--format", "md", "--output", str(output_dir)],
    )

    assert result.exit_code == 0


def test_publish_verbose_flag(runner, docs_dir, tmp_path):
    """Test that --verbose flag is accepted."""
    output_dir = tmp_path / "verbose-out"

    result = runner.invoke(
        publish,
        [
            str(docs_dir),
            "--format",
            "md",
            "--output",
            str(output_dir),
            "--verbose",
            "--include",
            "*.py",
        ],
    )

    assert result.exit_code == 0


def test_publish_site_with_include_pattern(runner, docs_dir, tmp_path):
    """Test --include pattern with site format."""
    output_dir = tmp_path / "pattern-out"

    with patch(
        "colight_publish.static.site_builder._prepare_bundle",
        return_value="/dist/live.js",
    ):
        result = runner.invoke(
            publish,
            [
                str(docs_dir),
                "--format",
                "site",
                "--output",
                str(output_dir),
                "--include",
                "*.md",
            ],
        )

    assert result.exit_code == 0
    assert (output_dir / "api" / "index.json").exists()
