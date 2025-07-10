"""Tests for content-addressable cache keys in incremental executor."""

import pathlib
import tempfile


from colight_live.incremental_executor import IncrementalExecutor
from colight_site.parser import parse_colight_file


def test_cache_key_changes_with_dependency():
    """Test that cache keys change when dependencies change."""
    executor = IncrementalExecutor(verbose=True)

    # Create a simple document with dependencies
    content = """
# %%
x = 1

# %%
y = x + 1

# %%
z = y + 1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()

        # Parse and execute
        doc = parse_colight_file(pathlib.Path(f.name))
        results1 = executor.execute_incremental(doc)

        # Get initial cache keys
        initial_keys = {
            block_id: state.cache_key
            for block_id, state in executor.block_states.items()
        }

        # Now change the first block
        content2 = """
# %%
x = 2  # Changed value

# %%
y = x + 1

# %%
z = y + 1
"""
        f2 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f2.write(content2)
        f2.flush()

        # Parse and execute again
        doc2 = parse_colight_file(pathlib.Path(f2.name))
        results2 = executor.execute_incremental(doc2)

        # Get new cache keys
        new_keys = {
            block_id: state.cache_key
            for block_id, state in executor.block_states.items()
        }

        # The first block's key should change (content changed)
        assert initial_keys["0"] != new_keys["0"]

        # The second and third blocks' keys should also change
        # because their dependencies changed
        assert initial_keys["1"] != new_keys["1"]
        assert initial_keys["2"] != new_keys["2"]

        # Clean up
        pathlib.Path(f.name).unlink()
        pathlib.Path(f2.name).unlink()


def test_cache_key_unchanged_for_independent_blocks():
    """Test that cache keys don't change for independent blocks."""
    executor = IncrementalExecutor(verbose=True)

    # Create a document with independent blocks
    content = """
# %%
a = 1

# %%
b = 2  # Independent of a

# %%
c = a + 1  # Depends on a
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()

        # Parse and execute
        doc = parse_colight_file(pathlib.Path(f.name))
        results1 = executor.execute_incremental(doc)

        # Get initial cache keys
        initial_keys = {
            block_id: state.cache_key
            for block_id, state in executor.block_states.items()
        }

        # Change only the first block
        content2 = """
# %%
a = 10  # Changed

# %%
b = 2  # Independent of a

# %%
c = a + 1  # Depends on a
"""
        f2 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f2.write(content2)
        f2.flush()

        # Parse and execute again
        doc2 = parse_colight_file(pathlib.Path(f2.name))
        results2 = executor.execute_incremental(doc2)

        # Get new cache keys
        new_keys = {
            block_id: state.cache_key
            for block_id, state in executor.block_states.items()
        }

        # The first block's key should change
        assert initial_keys["0"] != new_keys["0"]

        # The second block's key should NOT change (independent)
        assert initial_keys["1"] == new_keys["1"]

        # The third block's key SHOULD change (depends on a)
        assert initial_keys["2"] != new_keys["2"]

        # Clean up
        pathlib.Path(f.name).unlink()
        pathlib.Path(f2.name).unlink()


def test_cache_reuse_after_reordering():
    """Test that cache is reused when blocks are reordered but content unchanged."""
    executor = IncrementalExecutor(verbose=True)

    # Create initial document
    content = """
# %%
x = 1

# %%
y = 2

# %%
z = x + y
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()

        # Parse and execute
        doc = parse_colight_file(pathlib.Path(f.name))
        results1 = executor.execute_incremental(doc)

        # Count executions
        initial_cache_size = len(executor.cache_by_key)

        # Reorder blocks (swap first two)
        content2 = """
# %%
y = 2

# %%
x = 1

# %%
z = x + y
"""
        f2 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f2.write(content2)
        f2.flush()

        # Parse and execute again
        doc2 = parse_colight_file(pathlib.Path(f2.name))

        # Clear block states but keep cache_by_key
        executor.block_states.clear()

        results2 = executor.execute_incremental(doc2)

        # Cache size should be the same (reused existing entries)
        assert len(executor.cache_by_key) == initial_cache_size

        # Results should be the same
        assert len(results1) == len(results2)

        # Clean up
        pathlib.Path(f.name).unlink()
        pathlib.Path(f2.name).unlink()
