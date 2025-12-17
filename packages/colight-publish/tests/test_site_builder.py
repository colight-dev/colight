import json
from unittest.mock import patch

from colight_publish.static.site_builder import build_static_site


def test_static_site_builder_handles_markdown(tmp_path):
    """Test that markdown files are converted to JSON documents."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    markdown = docs_dir / "guide.md"
    markdown.write_text("# Guide\n\nWelcome to the docs.")

    output_dir = tmp_path / "site"

    build_static_site(docs_dir, output_dir, include=["*.md"])

    index_file = output_dir / "api" / "index.json"
    assert index_file.exists()

    doc_json = output_dir / "api" / "document" / "guide.md.json"
    assert doc_json.exists()

    data = json.loads(doc_json.read_text(encoding="utf-8"))
    assert data["file"] == "guide.md"
    assert (
        data["blocks"][0]["elements"][0]["value"] == "# Guide\n\nWelcome to the docs."
    )


def test_static_site_builder_creates_index_html(tmp_path):
    """Test that index.html is created with static mode flag."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "readme.md").write_text("# Hello")

    output_dir = tmp_path / "site"

    # Mock the bundle preparation to avoid needing live.js
    with patch(
        "colight_publish.static.site_builder._prepare_bundle",
        return_value="/dist/live.js",
    ):
        build_static_site(docs_dir, output_dir, include=["*.md"])

    index_html = output_dir / "index.html"
    assert index_html.exists()

    content = index_html.read_text(encoding="utf-8")
    assert "window.__COLIGHT_STATIC_MODE__ = true" in content
    assert "/dist/live.js" in content


def test_static_site_builder_handles_python_files(tmp_path):
    """Test that Python files are processed via JsonDocumentGenerator."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    py_file = docs_dir / "example.py"
    py_file.write_text("# A simple example\nx = 1 + 1\nx")

    output_dir = tmp_path / "site"

    with patch(
        "colight_publish.static.site_builder._prepare_bundle",
        return_value="/dist/live.js",
    ):
        build_static_site(docs_dir, output_dir, include=["*.py"])

    doc_json = output_dir / "api" / "document" / "example.py.json"
    assert doc_json.exists()

    data = json.loads(doc_json.read_text(encoding="utf-8"))
    assert data["file"] == "example.py"
    assert len(data["blocks"]) > 0


def test_static_site_builder_single_file_in_directory(tmp_path):
    """Test building a site with only one file in the directory."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    single_file = docs_dir / "single.md"
    single_file.write_text("# Single File")

    output_dir = tmp_path / "site"

    with patch(
        "colight_publish.static.site_builder._prepare_bundle",
        return_value="/dist/live.js",
    ):
        build_static_site(docs_dir, output_dir, include=["*.md"])

    doc_json = output_dir / "api" / "document" / "single.md.json"
    assert doc_json.exists()


def test_static_site_builder_subdirectories(tmp_path):
    """Test that files in subdirectories are handled correctly."""
    docs_dir = tmp_path / "docs"
    sub_dir = docs_dir / "guide" / "advanced"
    sub_dir.mkdir(parents=True)

    (docs_dir / "index.md").write_text("# Home")
    (sub_dir / "tips.md").write_text("# Tips")

    output_dir = tmp_path / "site"

    with patch(
        "colight_publish.static.site_builder._prepare_bundle",
        return_value="/dist/live.js",
    ):
        build_static_site(docs_dir, output_dir, include=["*.md"])

    # Check both files exist in correct locations
    assert (output_dir / "api" / "document" / "index.md.json").exists()
    assert (
        output_dir / "api" / "document" / "guide" / "advanced" / "tips.md.json"
    ).exists()

    # Check index.json has correct tree structure
    index_data = json.loads((output_dir / "api" / "index.json").read_text())
    assert index_data["name"] == "docs"


def test_static_site_builder_api_index_json_structure(tmp_path):
    """Test that api/index.json has the expected tree structure."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "a.md").write_text("# A")
    (docs_dir / "b.md").write_text("# B")

    output_dir = tmp_path / "site"

    with patch(
        "colight_publish.static.site_builder._prepare_bundle",
        return_value="/dist/live.js",
    ):
        build_static_site(docs_dir, output_dir, include=["*.md"])

    index_data = json.loads((output_dir / "api" / "index.json").read_text())

    assert index_data["type"] == "directory"
    assert "children" in index_data
    child_names = [c["name"] for c in index_data["children"]]
    assert "a.md" in child_names
    assert "b.md" in child_names
