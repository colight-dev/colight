"""Test the new element-based parser structure."""

import pathlib
import tempfile
from colight_site.parser import (
    parse_colight_file,
    FormElement,
    Form,
    should_show_statements,
    should_show_visuals,
    should_show_code,
    should_show_prose,
)


def test_form_elements_ordering():
    """Test that forms maintain ordered list of elements."""
    content = """# This is prose
# More prose

import numpy as np

# Another comment
x = 1
y = 2

# Final comment
x + y
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Should have 3 forms with new parser
        assert len(forms) == 3

        # First form: prose + import statement
        form1 = forms[0]
        assert len(form1.elements) == 2
        assert form1.elements[0].type == "prose"
        assert "This is prose" in form1.elements[0].content
        assert form1.elements[1].type == "statement"
        assert "import numpy as np" in form1.elements[1].get_code()

        # Second form: prose + two statements
        form2 = forms[1]
        assert len(form2.elements) == 2
        assert form2.elements[0].type == "prose"
        assert "Another comment" in form2.elements[0].content
        assert form2.elements[1].type == "statement"
        assert "x = 1" in form2.elements[1].get_code()
        assert "y = 2" in form2.elements[1].get_code()

        # Third form: prose + expression
        form3 = forms[2]
        assert len(form3.elements) == 2
        assert form3.elements[0].type == "prose"
        assert "Final comment" in form3.elements[0].content
        assert form3.elements[1].type == "expression"
        assert "x + y" in form3.elements[1].get_code()

        pathlib.Path(f.name).unlink()


def test_visual_display_for_any_expression():
    """Test that any expression can show visual (not just last element)."""
    # Case 1: Expression in the middle - should still show visual
    content1 = """# Setup
x = 1
42  # This expression can show a visual
y = 2
"""

    # Case 2: Multiple expressions - all can show visuals
    content2 = """# Multiple expressions
x + 1
y + 2
"""

    # Case 3: Only statement - visuals allowed but no expressions to show
    content3 = """# Statement only
x = 1
"""

    # Case 4: With hide-visuals pragma - no visuals
    content4 = """#| hide-visuals
# Expression that won't show visual
42
"""

    test_cases = [
        (content1, True, 1, "expression in middle"),
        (content2, True, 2, "multiple expressions"),
        (content3, True, 0, "only statement"),  # Visuals allowed, just no expressions
        (content4, False, 1, "hide-visuals pragma"),  # Has expression but hides visual
    ]

    for content, should_show, expr_count, desc in test_cases:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".colight.py", delete=False
        ) as f:
            f.write(content)
            f.flush()

            forms, metadata = parse_colight_file(pathlib.Path(f.name))
            assert len(forms) >= 1, f"{desc}: Expected at least one form"

            form = forms[0]
            assert form.should_show_visual() == should_show, (
                f"{desc}: Expected should_show_visual={should_show}, "
                f"got {form.should_show_visual()}"
            )
            
            # Count expression elements
            expressions = [e for e in form.elements if e.type == "expression"]
            assert len(expressions) == expr_count, (
                f"{desc}: Expected {expr_count} expressions, got {len(expressions)}"
            )

            pathlib.Path(f.name).unlink()


def test_form_level_pragma_visibility():
    """Test that form-level pragmas control element visibility."""
    content = """#| hide-prose
# This prose should be hidden
import something

#| show-prose hide-code
# This prose should be visible
x = 1  # This code should be hidden

#| hide-visuals
# This comment is visible
42  # This expression won't show visual
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Should have 3 separate forms
        assert len(forms) == 3

        # First form: prose hidden, code visible
        form1 = forms[0]
        assert "hide-prose" in form1.metadata.pragma
        prose_elem = next((e for e in form1.elements if e.type == "prose"), None)
        code_elem = next((e for e in form1.elements if e.type == "statement"), None)
        assert prose_elem is not None
        assert code_elem is not None
        assert not form1.should_show_element(prose_elem)  # prose hidden
        assert form1.should_show_element(code_elem)  # code visible

        # Second form: prose visible, code hidden
        form2 = forms[1]
        assert "show-prose" in form2.metadata.pragma
        assert "hide-code" in form2.metadata.pragma
        prose_elem2 = next((e for e in form2.elements if e.type == "prose"), None)
        code_elem2 = next((e for e in form2.elements if e.type == "statement"), None)
        assert prose_elem2 is not None
        assert code_elem2 is not None
        assert form2.should_show_element(prose_elem2)  # prose visible (show-prose)
        assert not form2.should_show_element(code_elem2)  # code hidden

        # Third form: visuals hidden
        form3 = forms[2]
        assert "hide-visuals" in form3.metadata.pragma
        assert not form3.should_show_visual()  # hide-visuals pragma
        expr_elem = next((e for e in form3.elements if e.type == "expression"), None)
        assert expr_elem is not None
        assert form3.should_show_element(expr_elem)  # expression code still visible

        pathlib.Path(f.name).unlink()


def test_blank_lines_create_form_boundaries():
    """Test that blank lines create form boundaries."""
    content = """# First form
x = 1

# Second form starts after blank line
y = 2

# Third form
42
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Should have 3 forms due to blank lines
        assert len(forms) == 3

        # First form: prose + statement
        form1 = forms[0]
        assert len(form1.elements) == 2
        assert form1.elements[0].type == "prose"
        assert form1.elements[1].type == "statement"
        assert "x = 1" in form1.elements[1].get_code()

        # Second form: prose + statement
        form2 = forms[1]
        assert len(form2.elements) == 2
        assert form2.elements[0].type == "prose"
        assert form2.elements[1].type == "statement"
        assert "y = 2" in form2.elements[1].get_code()

        # Third form: prose + expression
        form3 = forms[2]
        assert len(form3.elements) == 2
        assert form3.elements[0].type == "prose"
        assert form3.elements[1].type == "expression"
        assert "42" in form3.elements[1].get_code()

        pathlib.Path(f.name).unlink()


def test_mixed_statements_and_expressions():
    """Test forms with mixed statements and expressions."""
    content = """# Setup with mixed code
a = 1
b = 2
a + b  # Expression in the middle
c = 3  # Statement at the end
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        form = forms[0]
        # Should have prose + multiple code elements
        assert form.elements[0].type == "prose"
        
        # Find all code elements
        code_elements = [e for e in form.elements if e.type in ("statement", "expression")]
        assert len(code_elements) >= 2  # At least statement and expression
        
        # Should have at least one expression
        expressions = [e for e in form.elements if e.type == "expression"]
        assert len(expressions) >= 1
        assert any("a + b" in e.get_code() for e in expressions)
        
        # Form should allow visuals (not hidden by pragma)
        assert form.should_show_visual()

        pathlib.Path(f.name).unlink()


def test_pragma_accumulation():
    """Test that multiple pragma lines accumulate in a form."""
    content = """#| hide-code
#| show-visuals
# Multiple pragmas should combine
x = 1
42
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        form = forms[0]
        # Both pragma tags should be present
        assert "hide-code" in form.metadata.pragma
        assert "show-visuals" in form.metadata.pragma
        
        # Code should be hidden
        for elem in form.elements:
            if elem.type in ("statement", "expression"):
                assert not form.should_show_element(elem)
        
        # But visuals should be shown
        assert form.should_show_visual()

        pathlib.Path(f.name).unlink()