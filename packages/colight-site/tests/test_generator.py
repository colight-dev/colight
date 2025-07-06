"""Test the generator module."""

import pathlib
from typing import List, Optional, Union
from colight_site.generator import MarkdownGenerator
from colight_site.parser import Form, FormElement, FormMetadata
import libcst as cst


def create_test_form(prose: List[str], code: str, start_line: int = 1) -> Form:
    """Helper to create forms for testing."""
    elements = []
    
    # Add prose element if present
    if prose:
        prose_elem = FormElement(
            type="prose",
            content="\n".join(prose),
            line_number=start_line
        )
        elements.append(prose_elem)
    
    # Add code element if present
    if code:
        stmt = cst.parse_statement(code)
        # Determine if it's an expression or statement
        is_expr = (isinstance(stmt, cst.SimpleStatementLine) and 
                   len(stmt.body) == 1 and 
                   isinstance(stmt.body[0], cst.Expr))
        
        code_elem = FormElement(
            type="expression" if is_expr else "statement",
            content=stmt,
            line_number=start_line + len(prose) + 1
        )
        elements.append(code_elem)
    
    return Form(elements=elements, metadata=FormMetadata(), start_line=start_line)

# Get paths
test_dir = pathlib.Path(__file__).parent
examples_dir = test_dir / "examples"
project_root = test_dir.parent.parent.parent  # Go up 3 levels to project root
artifacts_dir = project_root / "test-artifacts" / "colight-site-hide"

# Create test-artifacts directory
artifacts_dir.mkdir(parents=True, exist_ok=True)


def test_markdown_generation():
    """Test basic markdown generation."""
    output_dir = artifacts_dir
    generator = MarkdownGenerator(output_dir)

    # Create mock forms
    forms = [
        create_test_form(["This is a test"], "import numpy as np", 1),
        create_test_form(["Create visualization"], "np.sin(x)", 3),
    ]

    colight_files = [None, pathlib.Path("test.colight")]

    path_context = {"basename": "test"}
    markdown = generator.generate_markdown(
        forms, colight_files, path_context=path_context
    )

    assert "This is a test" in markdown
    assert "Create visualization" in markdown
    assert "```python" in markdown
    assert "import numpy as np" in markdown
    assert "np.sin(x)" in markdown
    assert "data-src=" in markdown and "test_colight/form-" in markdown


def test_html_generation():
    """Test HTML generation."""
    output_dir = artifacts_dir
    generator = MarkdownGenerator(output_dir)

    # Create mock forms
    forms = [
        create_test_form(["This is a test"], "import numpy as np", 1),
        create_test_form(["Create visualization"], "np.sin(x)", 3),
    ]

    colight_files = [None, pathlib.Path("test.colight")]

    path_context = {"basename": "test"}
    html = generator.generate_html(
        forms, colight_files, title="Test Document", path_context=path_context
    )

    assert "<!DOCTYPE html>" in html
    assert "<title>Test Document</title>" in html
    assert "colight-embed" in html
    assert "embed.js" in html
    assert "<p>This is a test</p>" in html


def test_hide_statements_flag():
    """Test the hide_statements flag."""
    output_dir = artifacts_dir
    generator = MarkdownGenerator(output_dir)

    # Create forms with both statements and expressions
    forms = [
        create_test_form(["Import libraries"], "import numpy as np", 1),
        create_test_form(["Create data"], "x = np.linspace(0, 10, 100)", 2),
        create_test_form(["Visualize"], "np.sin(x)", 3),
    ]

    colight_files = [None, None, pathlib.Path("test.colight")]

    # Generate without hiding statements
    path_context = {"basename": "test"}
    markdown = generator.generate_markdown(
        forms, colight_files, path_context=path_context
    )
    assert "import numpy as np" in markdown
    assert "x = np.linspace(0, 10, 100)" in markdown
    assert "np.sin(x)" in markdown

    # Generate with hide_statements=True
    path_context = {"basename": "test"}
    markdown_hidden = generator.generate_markdown(
        forms, colight_files, path_context=path_context, pragma={"hide-statements"}
    )
    assert "import numpy as np" not in markdown_hidden  # statement
    assert "x = np.linspace(0, 10, 100)" not in markdown_hidden  # statement
    assert "np.sin(x)" in markdown_hidden  # expression

    # Importantly, markdown content should still be present
    assert "Import libraries" in markdown_hidden
    assert "Create data" in markdown_hidden
    assert "Visualize" in markdown_hidden


def test_hide_code_flag():
    """Test the hide_code flag."""
    output_dir = artifacts_dir
    generator = MarkdownGenerator(output_dir)

    # Create forms
    forms = [
        create_test_form(["Import libraries"], "import numpy as np", 1),
        create_test_form(["Visualize"], "np.sin(x)", 2),
    ]

    colight_files = [None, pathlib.Path("test.colight")]

    # Generate without hiding code
    path_context = {"basename": "test"}
    markdown = generator.generate_markdown(
        forms, colight_files, path_context=path_context
    )
    assert "```python" in markdown
    assert "import numpy as np" in markdown
    assert "np.sin(x)" in markdown

    # Generate with hide_code=True
    path_context = {"basename": "test"}
    markdown_hidden = generator.generate_markdown(
        forms, colight_files, path_context=path_context, pragma={"hide-code"}
    )
    assert "```python" not in markdown_hidden
    assert "import numpy as np" not in markdown_hidden
    assert "np.sin(x)" not in markdown_hidden
    # But markdown content should still be there
    assert "Import libraries" in markdown_hidden
    assert "Visualize" in markdown_hidden
    # And visualizations should still be there
    assert "colight-embed" in markdown_hidden


def test_hide_visuals_flag():
    """Test the hide_visuals flag."""
    output_dir = artifacts_dir
    generator = MarkdownGenerator(output_dir)

    # Create forms
    forms = [
        create_test_form(["Visualize"], "np.sin(x)", 1),
    ]

    colight_files: List[Optional[Union[bytes, pathlib.Path]]] = [
        pathlib.Path("test.colight")
    ]

    # Generate without hiding visuals
    path_context = {"basename": "test"}
    markdown = generator.generate_markdown(
        forms, colight_files, path_context=path_context
    )
    assert "colight-embed" in markdown
    assert "data-src=" in markdown and "test_colight/form-" in markdown

    # Generate with hide_visuals=True
    path_context = {"basename": "test"}
    markdown_hidden = generator.generate_markdown(
        forms, colight_files, path_context=path_context, pragma={"hide-visuals"}
    )
    assert "colight-embed" not in markdown_hidden
    assert "data-src=" not in markdown_hidden
    # But code and markdown should still be there
    assert "Visualize" in markdown_hidden
    assert "np.sin(x)" in markdown_hidden


def test_per_form_metadata_overrides():
    """Test that per-form metadata overrides file-level settings."""
    output_dir = artifacts_dir
    generator = MarkdownGenerator(output_dir)

    # Create forms with per-form metadata
    form1 = create_test_form(["This form should hide its code"], "import numpy as np", 1)
    form1.metadata.pragma = {"hide-code"}
    
    form2 = create_test_form(["This form should show its code"], "np.sin(x)", 2)
    form2.metadata.pragma = {"show-code"}
    
    forms = [form1, form2]

    colight_files = [None, pathlib.Path("test.colight")]

    # Generate with default settings (should be overridden per-form)
    path_context = {"basename": "test"}
    markdown = generator.generate_markdown(
        forms, colight_files, path_context=path_context
    )

    # First form should hide code due to per-form metadata
    assert "This form should hide its code" in markdown
    assert "import numpy as np" not in markdown

    # Second form should show code due to per-form metadata
    assert "This form should show its code" in markdown
    assert "np.sin(x)" in markdown


def test_show_code_overrides_hide_code():
    """Test that show-code pragma overrides hide-code for specific forms."""
    output_dir = artifacts_dir
    generator = MarkdownGenerator(output_dir)

    # Create test content similar to the user's issue
    import tempfile
    from colight_site.parser import parse_colight_file

    content = """#| hide-all-statements hide-all-code

# First form - should hide code due to file-level flags
import numpy as np

#| colight: show-code
# Second form - should show code despite file-level hide-code
x = np.array([1, 2, 3])

# Third form - should hide code again (back to file defaults)
y = x * 2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Use the file metadata tags directly
        colight_files: List[Optional[Union[bytes, pathlib.Path]]] = [None] * len(forms)

        path_context = {"basename": "test"}
        markdown = generator.generate_markdown(
            forms,
            colight_files,
            path_context=path_context,
            pragma=metadata.pragma,
        )

        # Check results
        assert "import numpy as np" not in markdown  # hidden by file-level hide-code
        assert "x = np.array([1, 2, 3])" in markdown  # shown by show-code override
        assert "y = x * 2" not in markdown  # hidden by file-level hide-code

        pathlib.Path(f.name).unlink()


def test_hide_prose_generation():
    """Test that hide-prose pragma correctly hides markdown prose."""
    import tempfile
    from colight_site.parser import parse_colight_file

    # Test content with hide-prose at file level
    content = """# %% hide-all-prose

# This is a title that should be hidden
# This description should also be hidden

import numpy as np

# This comment should be hidden too
x = np.array([1, 2, 3])

# | show-prose
# But this comment should be visible

y = x * 2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".colight.py", delete=False) as f:
        f.write(content)
        f.flush()

        # Parse the file
        forms, metadata = parse_colight_file(pathlib.Path(f.name))

        # Generate markdown
        output_dir = artifacts_dir / "hide-prose-test"
        generator = MarkdownGenerator(output_dir)

        # No colight data for this test
        colight_data: List[Optional[Union[bytes, pathlib.Path]]] = [None] * len(forms)

        markdown = generator.generate_markdown(
            forms, colight_data, pragma=metadata.pragma
        )

        # Check that prose is hidden/shown correctly
        assert "This is a title that should be hidden" not in markdown
        assert "This description should also be hidden" not in markdown
        assert "This comment should be hidden too" not in markdown
        assert "But this comment should be visible" in markdown

        # Code should still be visible (hide-prose doesn't affect code)
        assert "import numpy as np" in markdown
        assert "x = np.array([1, 2, 3])" in markdown
        assert "y = x * 2" in markdown

        pathlib.Path(f.name).unlink()
