import os
import tempfile

import numpy as np

import colight.plot as Plot
from colight.html import (
    html_snippet,
    html_page,
)

from colight.format import parse_file, MAGIC_BYTES, HEADER_SIZE
from notebooks.save_and_embed_file import create_embed_example


def test_html_snippet():
    """Test that html_snippet generates valid HTML with new format"""
    p = Plot.barY(
        {"x": ["A", "B", "C"], "y": [1, 2, 3]},
    )

    html = html_snippet(p)

    # Basic checks
    assert "<div" in html
    assert '<script type="application/x-colight"' in html  # Updated for new format
    assert '<script type="module">' in html
    assert "render" in html
    assert "parseColightScript" in html  # Should use new parser


def test_html_page():
    """Test that html_page generates a full HTML page"""
    p = Plot.barY(
        {"x": ["A", "B", "C"], "y": [1, 2, 3]},
    )
    id = "colight-test"

    html = html_page(p, id)

    # Basic checks
    assert "<!DOCTYPE html>" in html
    assert "<html>" in html
    assert "<head>" in html
    assert "<body>" in html
    assert html_snippet(p, id) in html


def test_export_colight():
    """Test that export_colight creates a valid .colight file with new binary format"""
    # Create a visual with binary data
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    p = Plot.raster(data)

    # Test without example file - also create in test-artifacts for JS tests
    test_artifacts_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "test-artifacts"
    )
    os.makedirs(test_artifacts_dir, exist_ok=True)

    # Create test file in artifacts directory for JS tests to use
    artifact_path = os.path.join(test_artifacts_dir, "test-raster.colight")
    result_path = p.save_file(artifact_path)

    # Check that the file exists
    assert os.path.exists(result_path)

    # Test the new binary format
    with open(result_path, "rb") as f:
        content = f.read()

    # Check header
    assert len(content) >= HEADER_SIZE
    magic = content[:8]
    assert magic == MAGIC_BYTES

    # Parse using our parser
    json_data, buffers = parse_file(result_path)

    # Verify buffer layout
    assert "bufferLayout" in json_data
    assert "offsets" in json_data["bufferLayout"]
    assert "lengths" in json_data["bufferLayout"]
    assert "count" in json_data["bufferLayout"]
    assert "totalSize" in json_data["bufferLayout"]

    # Verify we have buffers
    assert len(buffers) > 0
    buffer_layout = json_data["bufferLayout"]
    assert len(buffers) == buffer_layout["count"]

    # Test with example file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test2.colight")
        colight_path = p.save_file(output_path)
        example_path = create_embed_example(colight_path, False)

        # Check that both files exist
        assert os.path.exists(colight_path)
        assert os.path.exists(example_path)

        # Verify the example file has expected content
        with open(example_path, "r") as f:
            html_content = f.read()

        assert "<!DOCTYPE html>" in html_content
        assert "colight-embed" in html_content
        assert "data-src" in html_content


def test_create_embed_example():
    """Test that create_embed_example creates a valid HTML example"""
    # Create a visual
    p = Plot.barY(
        {"x": ["A", "B", "C"], "y": [1, 2, 3]},
    )

    # Export to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        colight_path = os.path.join(tmpdir, "test.colight")
        p.save_file(colight_path)
        example_path = create_embed_example(colight_path)

        # Check that the file exists
        assert os.path.exists(example_path)

        # Read the file
        with open(example_path, "r") as f:
            html_content = f.read()

        # Check for key elements
        assert "<!DOCTYPE html>" in html_content
        assert "colight-embed" in html_content
        assert "data-src" in html_content
        assert "loadVisual" in html_content
        assert "test.colight" in html_content

    # Test with local embed option
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export the .colight file
        colight_path = os.path.join(tmpdir, "test.colight")
        p.save_file(colight_path)
        example_path = create_embed_example(colight_path, False)
        assert os.path.exists(example_path)
