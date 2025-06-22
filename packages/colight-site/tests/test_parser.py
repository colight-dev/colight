"""Test the parser module."""

import pathlib
import tempfile
from colight_site.parser import (
    parse_colight_file,
    is_colight_file,
    parse_file_metadata,
    FileMetadata,
    should_hide_statements,
    should_hide_visuals,
    should_hide_code,
    _get_formats_from_tags,
)


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

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Should have 3 forms (better grouping with new parser)
        assert len(forms) == 3

        # First form: import with title comments
        assert "import numpy as np" in forms[0].code
        assert len(forms[0].markdown) > 0
        assert "This is a title" in forms[0].markdown[0]

        # Second form: assignment with "Create data" comment
        assert "x = np.linspace" in forms[1].code
        assert "Create data" in forms[1].markdown[0]

        # Third form: expression with "This creates a visualization" comment
        assert "np.sin(x)" in forms[2].code
        assert "This creates a visualization" in forms[2].markdown[0]
        assert forms[2].is_expression

        # Clean up
        pathlib.Path(f.name).unlink()


def test_parse_empty_file():
    """Test parsing an empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write("")
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))
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

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Should have 3 forms total (better grouping with new parser)
        assert len(forms) == 3

        # Form 0: import with title markdown
        assert "Title" in forms[0].markdown[0]
        assert "import numpy as np" in forms[0].code

        # Form 1: all three consecutive assignments grouped together with comment
        assert "Some comment" in forms[1].markdown[0]
        assert "x = np.linspace" in forms[1].code
        assert "y = np.sin" in forms[1].code
        assert "z = np.cos" in forms[1].code

        # Form 2: final assignment with comment
        assert "Another comment" in forms[2].markdown[0]
        assert "result = x, y, z" in forms[2].code

        pathlib.Path(f.name).unlink()


def test_parse_file_metadata_basic():
    """Test parsing basic file metadata."""
    source = """#| colight: hide-statements
#| colight: format-html

import numpy as np
"""
    metadata = parse_file_metadata(source)
    assert "hide-statements" in metadata.pragma_tags
    assert "format-html" in metadata.pragma_tags
    assert "hide-visuals" not in metadata.pragma_tags
    assert "hide-code" not in metadata.pragma_tags


def test_parse_file_metadata_multiple_options():
    """Test parsing multiple options in one line."""
    source = """#| colight: hide-statements, hide-visuals, format-markdown

import numpy as np
"""
    metadata = parse_file_metadata(source)
    assert should_hide_statements(metadata.pragma_tags) is True
    assert should_hide_visuals(metadata.pragma_tags) is True
    assert should_hide_code(metadata.pragma_tags) is False
    formats = _get_formats_from_tags(metadata.pragma_tags)
    assert "markdown" in formats


def test_parse_file_metadata_combined_flags():
    """Test combining multiple flags."""
    source = """#| hide-statements hide-code

import numpy as np
"""
    metadata = parse_file_metadata(source)
    assert should_hide_statements(metadata.pragma_tags) is True
    assert should_hide_visuals(metadata.pragma_tags) is False
    assert should_hide_code(metadata.pragma_tags) is True
    formats = _get_formats_from_tags(metadata.pragma_tags)
    assert len(formats) == 0


def test_parse_file_metadata_no_pragmas():
    """Test parsing a file without any pragmas."""
    source = """# This is just a regular comment
import numpy as np
"""
    metadata = parse_file_metadata(source)
    assert should_hide_statements(metadata.pragma_tags) is False
    assert should_hide_visuals(metadata.pragma_tags) is False
    assert should_hide_code(metadata.pragma_tags) is False
    formats = _get_formats_from_tags(metadata.pragma_tags)
    assert len(formats) == 0


def test_parse_file_metadata_unknown_options():
    """Test that unknown options are silently ignored."""
    source = """#| colight: hide-statements, unknown-option, format-html

import numpy as np
"""
    metadata = parse_file_metadata(source)
    assert should_hide_statements(metadata.pragma_tags) is True
    formats = _get_formats_from_tags(metadata.pragma_tags)
    assert "html" in formats


def test_file_metadata_merge_with_cli():
    """Test merging file metadata with CLI options."""
    metadata = FileMetadata(pragma_tags={"hide-statements", "format-html"})

    # CLI options should override file metadata
    result_tags, result_formats = metadata.merge_with_cli_options(
        hide_visuals=True, format="markdown"
    )

    assert should_hide_statements(result_tags) is True  # from file
    assert should_hide_visuals(result_tags) is True  # from CLI
    assert should_hide_code(result_tags) is False  # default
    assert "markdown" in result_formats  # CLI override


def test_file_metadata_cli_overrides():
    """Test that CLI flags work correctly."""
    metadata = FileMetadata(pragma_tags={"hide-visuals"})

    result_tags, result_formats = metadata.merge_with_cli_options(
        hide_statements=True, hide_code=True
    )

    assert should_hide_statements(result_tags) is True  # from CLI
    assert should_hide_code(result_tags) is True  # from CLI
    assert should_hide_visuals(result_tags) is True  # preserved from file


def test_parse_colight_file_with_metadata():
    """Test parsing a colight file with metadata."""
    content = """#| colight: hide-statements, format-html

# This is a title
# Some description

import numpy as np

# Create data
x = np.linspace(0, 10, 100)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Check that metadata was parsed correctly
        assert should_hide_statements(metadata.pragma_tags) is True
        formats = _get_formats_from_tags(metadata.pragma_tags)
        assert "html" in formats

        # Check that forms were still parsed correctly
        assert len(forms) >= 1
        assert "import numpy as np" in forms[0].code

        pathlib.Path(f.name).unlink()


def test_pragma_comments_filtered():
    """Test that pragma comments are not included in markdown output."""
    content = """#| colight: hide-statements
#| colight: format-html

# This is a regular comment
# This should appear in output

import numpy as np

# Another regular comment
x = np.linspace(0, 10, 100)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Check that metadata was parsed correctly
        assert should_hide_statements(metadata.pragma_tags) is True
        formats = _get_formats_from_tags(metadata.pragma_tags)
        assert "html" in formats

        # Check that pragma comments are not in any form's markdown
        all_markdown = []
        for form in forms:
            all_markdown.extend(form.markdown)

        # Regular comments should be present
        assert any("This is a regular comment" in line for line in all_markdown)
        assert any("Another regular comment" in line for line in all_markdown)

        # Pragma comments should NOT be present
        assert not any("colight:" in line for line in all_markdown)
        assert not any("hide-statements" in line for line in all_markdown)
        assert not any("format-html" in line for line in all_markdown)

        pathlib.Path(f.name).unlink()


def test_show_options_override_hide():
    """Test that show- options override hide- options."""
    source = """#| colight: hide-statements, show-statements
    
import numpy as np
"""
    metadata = parse_file_metadata(source)
    # show-statements should override hide-statements
    assert should_hide_statements(metadata.pragma_tags) is False


def test_per_form_pragma_parsing():
    """Test parsing per-form pragma annotations."""
    content = """# Regular comment
    
#| colight: hide-code
# This form should hide its code
import numpy as np

# Regular comment without pragma
x = np.linspace(0, 10, 100)

#| colight: show-visuals
# This form should show visuals
y = np.sin(x)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Check that forms have the right metadata
        assert len(forms) >= 3

        # First form should have hide_code=True
        form_with_hide_code = None
        for form in forms:
            if "import numpy as np" in form.code:
                form_with_hide_code = form
                break

        assert form_with_hide_code is not None
        assert should_hide_code(form_with_hide_code.metadata.pragma_tags) is True

        # Find form with show-visuals
        form_with_show_visuals = None
        for form in forms:
            if "y = np.sin(x)" in form.code:
                form_with_show_visuals = form
                break

        assert form_with_show_visuals is not None
        assert should_hide_visuals(form_with_show_visuals.metadata.pragma_tags) is False

        pathlib.Path(f.name).unlink()


def test_form_metadata_resolve_with_defaults():
    """Test FormMetadata.resolve_with_defaults method."""
    from colight_site.parser import FormMetadata

    # Test with some overrides
    metadata = FormMetadata(pragma_tags={"hide-code", "show-visuals"})
    default_tags = {"hide-statements", "hide-visuals"}
    resolved = metadata.resolve_with_defaults(default_tags)

    # Form metadata should override defaults
    assert resolved == {"hide-code", "show-visuals"}


def test_file_vs_per_form_pragma_distinction():
    """Test that file-level pragmas only apply at the top, per-form pragmas are local."""
    content = """#| hide-statements hide-code

# First form - affected by file-level pragmas
import numpy as np

#| colight: format-html
# This pragma should NOT affect file metadata
x = np.array([1, 2, 3])
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # File metadata should only include the top-level pragma
        assert (
            should_hide_statements(metadata.pragma_tags) is True
        )  # from file-level flags
        assert should_hide_code(metadata.pragma_tags) is True  # from file-level flags
        formats = _get_formats_from_tags(metadata.pragma_tags)
        assert len(formats) == 0  # format-html is per-form, not file-level

        # Find forms
        import_form = None
        array_form = None
        for form in forms:
            if "import numpy" in form.code:
                import_form = form
            elif "x = np.array" in form.code:
                array_form = form

        assert import_form is not None
        assert array_form is not None

        # Import form should have no specific metadata (inherits file defaults)
        assert len(import_form.metadata.pragma_tags) == 0

        # Array form should have the format-html pragma from its per-form annotation
        assert "format-html" in array_form.metadata.pragma_tags

        pathlib.Path(f.name).unlink()


def test_new_pragma_formats():
    """Test new pragma formats with %% and | starters without colight: prefix."""
    content = """# %% hide-code
# Regular comment
import numpy as np

#| format-html
# Another comment
x = np.array([1, 2, 3])

#| hide-statements
y = x * 2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Should have 3 forms
        assert len(forms) == 3

        # First form should have hide_code=True from %% pragma
        import_form = None
        for form in forms:
            if "import numpy" in form.code:
                import_form = form
                break
        assert import_form is not None
        assert should_hide_code(import_form.metadata.pragma_tags) is True

        # Third form should have hide_statements=True from | pragma
        y_form = None
        for form in forms:
            if "y = x * 2" in form.code:
                y_form = form
                break
        assert y_form is not None
        assert should_hide_statements(y_form.metadata.pragma_tags) is True

        pathlib.Path(f.name).unlink()


def test_liberal_tag_matching():
    """Test that tags are matched liberally anywhere in pragma comments."""
    content = """#| hide-statements hide-code flag test

# %% This comment has hide-code somewhere in it
import numpy as np

#| Some text with show-visuals and other words
x = np.array([1, 2, 3])

# %% format-markdown should work too
y = x * 2

z = y + 1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Find the forms and check their metadata
        import_form = None
        x_form = None

        for form in forms:
            if "import numpy" in form.code:
                import_form = form
            elif "x = np.array" in form.code:
                x_form = form

        # Check that liberal matching worked
        assert import_form is not None
        assert should_hide_code(import_form.metadata.pragma_tags) is True

        assert x_form is not None
        assert should_hide_visuals(x_form.metadata.pragma_tags) is False  # show-visuals

        # File-level metadata from file-level flags
        assert should_hide_statements(metadata.pragma_tags) is True
        assert should_hide_code(metadata.pragma_tags) is True

        pathlib.Path(f.name).unlink()


def test_mixed_pragma_formats():
    """Test mixing old and new pragma formats."""
    content = """#| colight: hide-statements
# %% format-html

# Regular comment
import numpy as np

#| show-code
x = np.array([1, 2, 3])
"""

    metadata = parse_file_metadata(content)

    # Both old and new formats should be parsed
    assert should_hide_statements(metadata.pragma_tags) is True
    formats = _get_formats_from_tags(metadata.pragma_tags)
    assert "html" in formats


def test_case_insensitive_matching():
    """Test that tag matching is case insensitive."""
    content = """# %% Hide-Code Format-HTML

import numpy as np

#| SHOW-VISUALS
x = np.array([1, 2, 3])
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        formats = _get_formats_from_tags(metadata.pragma_tags)
        assert "html" in formats
        # File-level hide_code should be in file metadata
        assert should_hide_code(metadata.pragma_tags) is True

        # Find forms and check metadata
        import_form = None
        x_form = None

        for form in forms:
            if "import numpy" in form.code:
                import_form = form
            elif "x = np.array" in form.code:
                x_form = form

        assert import_form is not None

        assert x_form is not None
        assert should_hide_visuals(x_form.metadata.pragma_tags) is False  # show-visuals

        pathlib.Path(f.name).unlink()


def test_plural_singular_support():
    """Test that both plural and singular forms work."""
    content = """# %% hide-statement
import numpy as np

#| hide-statements should also work
x = np.array([1, 2, 3])

# %% show-visual
y = x * 2

#| show-visuals should also work
z = y + 1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Find forms
        import_form = None
        x_form = None
        y_form = None
        z_form = None

        for form in forms:
            if "import numpy" in form.code:
                import_form = form
            elif "x = np.array" in form.code:
                x_form = form
            elif "y = x * 2" in form.code:
                y_form = form
            elif "z = y + 1" in form.code:
                z_form = form

        # Check that both singular and plural forms work
        assert import_form is not None
        assert (
            should_hide_statements(import_form.metadata.pragma_tags) is True
        )  # hide-statement

        assert x_form is not None
        assert (
            should_hide_statements(x_form.metadata.pragma_tags) is True
        )  # hide-statements

        assert y_form is not None
        assert should_hide_visuals(y_form.metadata.pragma_tags) is False  # show-visual

        assert z_form is not None
        assert should_hide_visuals(z_form.metadata.pragma_tags) is False  # show-visuals

        pathlib.Path(f.name).unlink()


def test_regular_comments_not_pragmas():
    """Test that regular comments with tag-like words are not treated as pragmas."""
    content = """# Regular comment with hide-code mentioned
import numpy as np

# This comment mentions show-visuals but should be ignored
x = np.array([1, 2, 3])

# Some text with hide-statements should not affect anything
y = x * 2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # File metadata should be defaults (no pragmas detected)
        assert should_hide_statements(metadata.pragma_tags) is False
        assert should_hide_visuals(metadata.pragma_tags) is False
        assert should_hide_code(metadata.pragma_tags) is False
        formats = _get_formats_from_tags(metadata.pragma_tags)
        assert len(formats) == 0

        # All form metadata should be empty (no pragmas detected)
        for form in forms:
            assert len(form.metadata.pragma_tags) == 0

        pathlib.Path(f.name).unlink()


def test_combined_statements_ending_with_expression():
    """Test that combined statements ending with an expression are treated as expressions."""
    # Test case 1: Statement then expression - should be expression
    content1 = """# Test statement then expression
a = 1
42
"""

    # Test case 2: Multiple statements then expression - should be expression
    content2 = """# Test multiple statements then expression
a = 1
b = 2
a + b
"""

    # Test case 3: Expression then statement - should be statement
    content3 = """# Test expression then statement
42
a = 1
"""

    # Test case 4: Only statements - should be statement
    content4 = """# Test only statements
a = 1
b = 2
"""

    test_cases = [
        (content1, True, "Statement then expression"),
        (content2, True, "Multiple statements then expression"),
        (content3, False, "Expression then statement"),
        (content4, False, "Only statements"),
    ]

    for content, expected_is_expression, description in test_cases:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".colight.py", delete=False
        ) as f:
            f.write(content)
            f.flush()

            forms, metadata = parse_colight_file(pathlib.Path(f.name))

            assert len(forms) >= 1, f"{description}: Expected at least one form"
            form = forms[0]

            assert form.is_expression == expected_is_expression, (
                f"{description}: Expected is_expression={expected_is_expression}, "
                f"got {form.is_expression}"
            )
            assert form.is_statement == (not expected_is_expression), (
                f"{description}: Expected is_statement={not expected_is_expression}, "
                f"got {form.is_statement}"
            )

            pathlib.Path(f.name).unlink()


def test_empty_comment_continuation():
    """Test that empty comments act as continuations within forms."""
    # Test case 1: Empty comment continues the form
    content1 = """#| show-code
a = 1
#
42
"""

    # Test case 2: Non-empty comment breaks the form
    content2 = """#| show-code
a = 1
# This is a comment
42
"""

    # Test case 3: Multiple empty comments
    content3 = """# Test multiple empty comments
a = 1
#
b = 2
#
a + b
"""

    # Test case 4: Blank line breaks the form
    content4 = """# Test blank line
a = 1

42
"""

    test_cases = [
        (content1, 1, "a = 1\n\n42", True, "Empty comment continuation"),
        (content2, 2, "a = 1", False, "Non-empty comment breaks form"),
        (content3, 1, "a = 1\n\nb = 2\n\na + b", True, "Multiple empty comments"),
        (content4, 2, "a = 1", False, "Blank line breaks form"),
    ]

    for (
        content,
        expected_forms,
        expected_first_code,
        expected_is_expr,
        description,
    ) in test_cases:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".colight.py", delete=False
        ) as f:
            f.write(content)
            f.flush()

            forms, metadata = parse_colight_file(pathlib.Path(f.name))

            assert (
                len(forms) == expected_forms
            ), f"{description}: Expected {expected_forms} forms, got {len(forms)}"

            if forms:
                first_form = forms[0]
                assert first_form.code.strip() == expected_first_code, (
                    f"{description}: Expected code {repr(expected_first_code)}, "
                    f"got {repr(first_form.code.strip())}"
                )
                assert first_form.is_expression == expected_is_expr, (
                    f"{description}: Expected is_expression={expected_is_expr}, "
                    f"got {first_form.is_expression}"
                )

            pathlib.Path(f.name).unlink()
