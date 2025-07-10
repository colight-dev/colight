"""Tests for cache hit and content change tracking."""

import pathlib
import tempfile


from colight_live.incremental_executor import IncrementalExecutor
from colight_site.parser import parse_colight_file


def test_cache_hit_tracking():
    """Test that cache hits are properly tracked."""
    executor = IncrementalExecutor(verbose=True)

    # Create a simple document
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

        # First execution - all cache misses
        doc = parse_colight_file(pathlib.Path(f.name))
        results1 = executor.execute_incremental(doc)

        # All should be cache misses
        for block, result in results1:
            assert hasattr(result, "cache_hit")
            assert result.cache_hit == False
            assert hasattr(result, "content_changed")
            assert result.content_changed == True  # All new blocks

        # Second execution with no changes - all cache hits
        results2 = executor.execute_incremental(doc)

        # All should be cache hits with no content changes
        for block, result in results2:
            assert result.cache_hit == True
            assert result.content_changed == False

        # Change one block
        content2 = """
# %%
x = 2  # Changed

# %%
y = x + 1

# %%
z = y + 1
"""
        f2 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f2.write(content2)
        f2.flush()

        doc2 = parse_colight_file(pathlib.Path(f2.name))
        results3 = executor.execute_incremental(doc2)

        # Check cache hits and content changes
        results_dict = {str(block.id): result for block, result in results3}

        # First block: cache miss (content changed)
        assert results_dict["0"].cache_hit == False
        assert results_dict["0"].content_changed == True

        # Second block: cache miss (dependency changed) but content unchanged
        assert results_dict["1"].cache_hit == False
        assert results_dict["1"].content_changed == False

        # Third block: cache miss (dependency changed) but content unchanged
        assert results_dict["2"].cache_hit == False
        assert results_dict["2"].content_changed == False

        # Clean up
        pathlib.Path(f.name).unlink()
        pathlib.Path(f2.name).unlink()


def test_independent_blocks_cache_hit():
    """Test cache hits for independent blocks."""
    executor = IncrementalExecutor(verbose=True)

    content = """
# %%
a = 1

# %%
b = 2  # Independent

# %%
c = a + 1  # Depends on a
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()

        # First execution
        doc = parse_colight_file(pathlib.Path(f.name))
        results1 = executor.execute_incremental(doc)

        # Change only first block
        content2 = """
# %%
a = 10  # Changed

# %%
b = 2  # Independent

# %%
c = a + 1  # Depends on a
"""
        f2 = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        f2.write(content2)
        f2.flush()

        doc2 = parse_colight_file(pathlib.Path(f2.name))
        results2 = executor.execute_incremental(doc2)

        results_dict = {str(block.id): result for block, result in results2}

        # First block: content changed
        assert results_dict["0"].cache_hit == False
        assert results_dict["0"].content_changed == True

        # Second block: independent, should be cache hit
        assert results_dict["1"].cache_hit == True
        assert results_dict["1"].content_changed == False

        # Third block: depends on a, cache miss but content unchanged
        assert results_dict["2"].cache_hit == False
        assert results_dict["2"].content_changed == False

        # Clean up
        pathlib.Path(f.name).unlink()
        pathlib.Path(f2.name).unlink()
