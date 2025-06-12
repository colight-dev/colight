import os
import json
import tempfile

import numpy as np

import colight.plot as Plot
from colight.html import (
    html_snippet,
    html_page,
    export_colight,
    BINARY_DELIMITER,
)
from notebooks.embed_examples import create_embed_example


def test_html_snippet():
    """Test that html_snippet generates valid HTML"""
    p = Plot.barY(
        {"x": ["A", "B", "C"], "y": [1, 2, 3]},
    )

    html = html_snippet(p)

    # Basic checks
    assert "<style>" in html
    assert "<div" in html
    assert '<script type="application/json">' in html
    assert '<script type="module">' in html
    assert "renderData" in html


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
    """Test that export_colight creates a valid .colight file"""
    # Create a visualization with binary data
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    p = Plot.raster(data)

    # Test without example file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test.colight")
        result_path = str(export_colight(p, output_path, create_example=False))

        # Check that the file exists
        assert os.path.exists(result_path)

        # Read the file
        with open(result_path, "rb") as f:
            content = f.read()

        # Check that it contains the delimiter
        assert BINARY_DELIMITER in content

        # Split the content
        parts = content.split(BINARY_DELIMITER)
        assert len(parts) == 2

        # Check the JSON header
        json_header = parts[0].decode("utf-8")
        header_data = json.loads(json_header)

        # Verify buffer layout
        assert "bufferLayout" in header_data
        assert "offsets" in header_data["bufferLayout"]
        assert "count" in header_data["bufferLayout"]
        assert "totalSize" in header_data["bufferLayout"]

        # Verify binary data
        binary_data = parts[1]
        assert len(binary_data) > 0

    # Test with example file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test2.colight")
        colight_path, example_path = export_colight(p, output_path, create_example=True)

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
    # Create a visualization
    p = Plot.barY(
        {"x": ["A", "B", "C"], "y": [1, 2, 3]},
    )

    # Export to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export the .colight file
        colight_path = os.path.join(tmpdir, "test.colight")
        # Use create_example=False to ensure we're just exporting the file
        export_colight(p, colight_path, create_example=False)

        # Create the example HTML separately
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
        export_colight(p, colight_path, create_example=False)

        # Create the example HTML with local embed
        example_path = create_embed_example(colight_path, use_local_embed=True)

        # Check that the file exists
        assert os.path.exists(example_path)
