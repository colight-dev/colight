import numpy as np

import colight.plot as Plot
from colight.html import (
    html_snippet,
    html_page,
)

data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
p = Plot.dot(data)


def test_html_snippet():
    """Test that html_snippet generates valid HTML with new format"""

    html = html_snippet(p)

    # Basic checks
    assert "<div" in html
    assert '<script type="application/x-colight"' in html  # Updated for new format
    assert '<script type="module">' in html
    assert "render" in html
    assert "parseColightScript" in html  # Should use new parser


def test_html_page():
    """Test that html_page generates a full HTML page"""

    id = "colight-test"

    html = html_page(p, id)

    # Basic checks
    assert "<!DOCTYPE html>" in html
    assert "<html>" in html
    assert "<head>" in html
    assert "<body>" in html
    assert html_snippet(p, id) in html


def test_html_file():
    html_path = p.save_html("test-artifacts/html_test.html", dist_url="/dist")

    # Read the generated HTML file and check it contains <html>
    with open(html_path, "r") as f:
        content = f.read()

    assert "<html>" in content
