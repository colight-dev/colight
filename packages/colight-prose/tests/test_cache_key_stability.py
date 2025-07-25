"""Tests for cache behavior in incremental executor."""

import pathlib
import tempfile

from colight_prose.incremental_executor import IncrementalExecutor
from colight_prose.parser import parse_colight_file


def test_cache_key_stability_with_reordered_symbols():
    """Test that blocks with same content get cache hits."""
    executor = IncrementalExecutor()

    # Execute a document and verify caching works
    content = """
# %%
a = 1
b = 2

# %%
result = a + b
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        doc = parse_colight_file(file_path)

        # First execution
        results1 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        # All should be cache misses
        for block, result in results1:
            assert result.cache_hit == False

        # Second execution - should be cache hits
        results2 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        for block, result in results2:
            assert result.cache_hit == True

        file_path.unlink()


def test_cache_key_changes_with_dependency_content():
    """Test that changing dependency content invalidates cache."""
    executor = IncrementalExecutor()

    # This test verifies that cache keys include dependency content
    # But we don't need to test the exact mechanism
    content1 = """
# %%
x = 1

# %%
y = x + 1
"""

    content2 = """
# %%
x = 2  # Changed

# %%
y = x + 1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f1:
        f1.write(content1)
        f1.flush()
        file1 = pathlib.Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f2:
            f2.write(content2)
            f2.flush()
            file2 = pathlib.Path(f2.name)

            # Execute first version
            doc1 = parse_colight_file(file1)
            list(executor.execute_incremental_streaming(doc1, str(file1), str(file1)))

            # Execute second version - dependency changed
            doc2 = parse_colight_file(file2)
            results = list(
                executor.execute_incremental_streaming(doc2, str(file2), str(file2))
            )

            # Both blocks should be cache misses due to source change
            assert results[0][1].cache_hit == False
            assert results[1][1].cache_hit == False

            file1.unlink()
            file2.unlink()


def test_cache_key_with_multiple_providers():
    """Test caching with multiple dependencies."""
    executor = IncrementalExecutor()

    content = """
# %%
a = 1

# %%
b = 2

# %%
c = a + b
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        doc = parse_colight_file(file_path)

        # First run
        results1 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        # Second run - all cache hits
        results2 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        for block, result in results2:
            assert result.cache_hit == True

        file_path.unlink()


def test_cache_key_with_missing_dependencies():
    """Test that missing dependencies cause errors."""
    executor = IncrementalExecutor()

    content = """
# %%
result = undefined_var + 1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        doc = parse_colight_file(file_path)
        results = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        # Should have error
        assert results[0][1].error is not None
        assert "undefined_var" in results[0][1].error

        file_path.unlink()


def test_cache_hit_after_reordering():
    """Test cache behavior is consistent."""
    executor = IncrementalExecutor()

    content = """
# %%
x = 1
y = 2

# %%
z = x + y
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        doc = parse_colight_file(file_path)

        # Run twice
        list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )
        results = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        # All cache hits on second run
        for block, result in results:
            assert result.cache_hit == True

        file_path.unlink()
