"""Test the parser module."""

import pathlib
import tempfile
from colight_site.parser import parse_colight_file, is_colight_file


def test_is_colight_file():
    """Test colight file detection."""
    assert is_colight_file(pathlib.Path("example.colight.py"))
    assert is_colight_file(pathlib.Path("test.colight.py"))
    assert not is_colight_file(pathlib.Path("regular.py"))
    assert not is_colight_file(pathlib.Path("test.txt"))


def test_parse_simple_colight_file():
    """Test parsing a simple colight file."""
    content = """# This is a title
# Some description

import numpy as np

# Create data
x = np.linspace(0, 10, 100)

# This creates a visualization
np.sin(x)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms = parse_colight_file(pathlib.Path(f.name))

        # Should have 3 forms
        assert len(forms) == 3

        # First form: import (should have the title comments)
        assert "import numpy as np" in forms[0].code
        assert len(forms[0].markdown) > 0
        assert "This is a title" in forms[0].markdown[0]

        # Second form: assignment
        assert "x = np.linspace" in forms[1].code
        assert "Create data" in forms[1].markdown[0]

        # Third form: expression
        assert "np.sin(x)" in forms[2].code
        assert forms[2].is_expression

        # Clean up
        pathlib.Path(f.name).unlink()


def test_parse_empty_file():
    """Test parsing an empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write("")
        f.flush()

        forms = parse_colight_file(pathlib.Path(f.name))
        assert len(forms) == 0

        pathlib.Path(f.name).unlink()
