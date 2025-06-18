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

        # Should have 5 forms (improved separation of markdown and code)
        assert len(forms) == 5

        # First form: import (should have the title comments)
        assert "import numpy as np" in forms[0].code
        assert len(forms[0].markdown) > 0
        assert "This is a title" in forms[0].markdown[0]

        # Second form: dummy markdown form for "Create data"
        assert "Create data" in forms[1].markdown[0]
        assert forms[1].code.strip() == "pass"

        # Third form: assignment
        assert "x = np.linspace" in forms[2].code
        assert len(forms[2].markdown) == 0

        # Fourth form: dummy markdown form for "This creates a visualization"
        assert "This creates a visualization" in forms[3].markdown[0]
        assert forms[3].code.strip() == "pass"

        # Fifth form: expression
        assert "np.sin(x)" in forms[4].code
        assert forms[4].is_expression

        # Clean up
        pathlib.Path(f.name).unlink()


def test_parse_empty_file():
    """Test parsing an empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write("")
        f.flush()

        forms = parse_colight_file(pathlib.Path(f.name))
        assert len(forms) == 0


def test_consecutive_code_grouping():
    """Test that consecutive code statements are grouped into single forms."""
    content = """# Title
# Description

import numpy as np

# Some comment
x = np.linspace(0, 10, 100)
y = np.sin(x)
z = np.cos(x)

# Another comment
result = x, y, z
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms = parse_colight_file(pathlib.Path(f.name))

        # Should have 6 forms total (improved separation of markdown and code)
        assert len(forms) == 6

        # Form 0: import with title markdown
        assert "Title" in forms[0].markdown[0]
        assert "import numpy as np" in forms[0].code

        # Form 1: dummy markdown form for "Some comment"
        assert "Some comment" in forms[1].markdown[0]
        assert forms[1].code.strip() == "pass"

        # Form 2: first assignment
        assert "x = np.linspace" in forms[2].code
        assert len(forms[2].markdown) == 0

        # Form 3: grouped assignments (consecutive code)
        assert "y = np.sin" in forms[3].code
        assert "z = np.cos" in forms[3].code
        assert len(forms[3].markdown) == 0

        # Form 4: dummy markdown form for "Another comment"
        assert "Another comment" in forms[4].markdown[0]
        assert forms[4].code.strip() == "pass"

        # Form 5: final assignment
        assert "result = x, y, z" in forms[5].code
        assert len(forms[5].markdown) == 0

        pathlib.Path(f.name).unlink()
