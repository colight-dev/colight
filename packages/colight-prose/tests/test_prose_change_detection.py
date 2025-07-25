"""Test that prose changes affect cache keys."""

import pathlib
import tempfile

from colight_prose.incremental_executor import IncrementalExecutor
from colight_prose.parser import parse_colight_file


def test_prose_change_detection():
    """Test that changes to prose result in different cache keys."""
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
        results1 = list(executor.execute_incremental_streaming(doc))

        # All should be cache misses on first run
        for block, result in results1:
            assert result.cache_hit == False

        # Store block IDs from first run
        block_ids1 = [block.id for block, _ in results1]

        # Change only prose
        content2 = """# %% [markdown]
# This is CHANGED prose
# with multiple lines

# %%
x = 1

# %% [markdown]
# More prose here
"""

        # Parse the changed content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f2:
            f2.write(content2)
            f2.flush()
            doc2 = parse_colight_file(pathlib.Path(f2.name))

        # Get block IDs from changed content
        block_ids2 = [block.id for block in doc2.blocks]

        # First block (prose) should have different ID due to content change
        assert (
            block_ids1[0] != block_ids2[0]
        ), "Prose change should result in different block ID"

        # Second block (code) should have same ID - no change
        assert (
            block_ids1[1] == block_ids2[1]
        ), "Unchanged code should have same block ID"

        # Third block (prose) should have same ID - no change
        assert (
            block_ids1[2] == block_ids2[2]
        ), "Unchanged prose should have same block ID"

        # Clean up
        pathlib.Path(f.name).unlink()
        pathlib.Path(f2.name).unlink()


def test_trailing_prose_change():
    """Test that trailing prose changes affect cache keys."""
    executor = IncrementalExecutor(verbose=True)

    content = """# %%
1 + 1

# Hello
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()

        doc = parse_colight_file(pathlib.Path(f.name))
        results1 = list(executor.execute_incremental_streaming(doc))

        # Should have 2 blocks (code and prose)
        assert len(results1) == 2

        # Store block IDs
        block_ids1 = [block.id for block, _ in results1]

        # Change the trailing prose
        content2 = """# %%
1 + 1

# Hello World
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f2:
            f2.write(content2)
            f2.flush()

            doc2 = parse_colight_file(pathlib.Path(f2.name))
            block_ids2 = [block.id for block in doc2.blocks]

        # Code block should have same ID - no change
        assert (
            block_ids1[0] == block_ids2[0]
        ), "Unchanged code should have same block ID"

        # Prose block should have different ID due to content change
        assert (
            block_ids1[1] != block_ids2[1]
        ), "Changed prose should have different block ID"

        # Clean up
        pathlib.Path(f.name).unlink()
        pathlib.Path(f2.name).unlink()
