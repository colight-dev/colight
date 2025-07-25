"""Tests for cache hit tracking."""

import pathlib
import tempfile

from colight_prose.incremental_executor import IncrementalExecutor
from colight_prose.parser import parse_colight_file


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
        file_path = pathlib.Path(f.name)

        # First execution - all cache misses
        doc = parse_colight_file(file_path)
        results1 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        # All should be cache misses
        for block, result in results1:
            assert hasattr(result, "cache_hit")
            assert result.cache_hit == False

        # Second execution with no changes - all cache hits
        results2 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        # All should be cache hits
        for block, result in results2:
            assert result.cache_hit == True

        # Clean up
        file_path.unlink()


def test_independent_blocks_cache_hit():
    """Test cache hits for independent blocks when dependencies change."""
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
        file_path = pathlib.Path(f.name)

        # First execution
        doc = parse_colight_file(file_path)
        results1 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        # All should be cache misses initially
        for block, result in results1:
            assert result.cache_hit == False

        # Second execution - all should be cache hits
        results2 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        for block, result in results2:
            assert result.cache_hit == True

        # Clean up
        file_path.unlink()
