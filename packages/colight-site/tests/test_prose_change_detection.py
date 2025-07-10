"""Test that prose changes are properly detected."""

import pathlib
import tempfile


from colight_live.incremental_executor import IncrementalExecutor
from colight_site.parser import parse_colight_file


def test_prose_change_detection():
    """Test that changes to prose are detected."""
    executor = IncrementalExecutor(verbose=True)

    # Create a document with prose
    content = """# %% [markdown]
# This is some prose
# with multiple lines

# %%
x = 1

# %% [markdown]
# More prose here
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()

        # First execution
        doc = parse_colight_file(pathlib.Path(f.name))
        results1 = executor.execute_incremental(doc)

        # All should be new
        for block, result in results1:
            assert result.content_changed == True

        # Change only prose
        content2 = """# %% [markdown]
# This is CHANGED prose
# with multiple lines

# %%
x = 1

# %% [markdown]
# More prose here
"""
        f2 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f2.write(content2)
        f2.flush()

        doc2 = parse_colight_file(pathlib.Path(f2.name))
        results2 = executor.execute_incremental(doc2)

        results_dict = {str(block.id): result for block, result in results2}

        # First block (prose) should show content changed
        assert results_dict["0"].content_changed == True

        # Second block (code) should not show content changed
        assert results_dict["1"].content_changed == False

        # Third block (prose) should not show content changed
        assert results_dict["2"].content_changed == False

        # Clean up
        pathlib.Path(f.name).unlink()
        pathlib.Path(f2.name).unlink()


def test_trailing_prose_change():
    """Test that trailing prose changes are detected."""
    executor = IncrementalExecutor(verbose=True)

    content = """# %%
1 + 1

# Hello
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()

        doc = parse_colight_file(pathlib.Path(f.name))
        results1 = executor.execute_incremental(doc)

        # Should have 2 blocks
        assert len(results1) == 2

        # Change the trailing prose
        content2 = """# %%
1 + 1

# Hello World
"""
        f2 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f2.write(content2)
        f2.flush()

        doc2 = parse_colight_file(pathlib.Path(f2.name))
        results2 = executor.execute_incremental(doc2)

        results_dict = {str(block.id): result for block, result in results2}

        # Code block should not show content changed
        assert results_dict["0"].content_changed == False

        # Prose block should show content changed
        assert results_dict["1"].content_changed == True

        # Clean up
        pathlib.Path(f.name).unlink()
        pathlib.Path(f2.name).unlink()
