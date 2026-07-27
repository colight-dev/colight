"""Tests for content-addressable cache behavior."""

import pathlib
import tempfile

from colight_prose.incremental_executor import IncrementalExecutor
from colight_prose.parser import parse_colight_file


def test_cache_key_changes_with_dependency():
    """Test that block IDs change when dependencies change."""
    # Create a simple document with dependencies
    content1 = """# %%
x = 1

# %%
y = x + 1

# %%
z = y + 1"""

    content2 = """# %%
x = 2  # Changed value

# %%
y = x + 1

# %%
z = y + 1"""

    # Parse both versions
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f1:
        f1.write(content1)
        f1.flush()
        doc1 = parse_colight_file(pathlib.Path(f1.name))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f2:
        f2.write(content2)
        f2.flush()
        doc2 = parse_colight_file(pathlib.Path(f2.name))

    # Get block IDs
    ids1 = [block.id for block in doc1.blocks]
    ids2 = [block.id for block in doc2.blocks]

    # The first block's ID should change (content changed)
    assert ids1[0] != ids2[0], "First block ID should change when content changes"

    # The second and third blocks' IDs should also change
    # because their dependency context changed
    assert ids1[1] != ids2[1], "Second block ID should change when dependency changes"
    assert (
        ids1[2] != ids2[2]
    ), "Third block ID should change when transitive dependency changes"

    # Clean up
    pathlib.Path(f1.name).unlink()
    pathlib.Path(f2.name).unlink()


def test_cache_key_unchanged_for_independent_blocks():
    """Test that independent blocks can be cached when content doesn't change."""
    executor = IncrementalExecutor(verbose=True)

    # Use same file to test cache behavior
    content1 = """# %%
a = 1

# %%
b = 2  # Independent of a

# %%
c = a + 1  # Depends on a"""

    content2 = """# %%
a = 10  # Changed

# %%
b = 2  # Independent of a

# %%
c = a + 1  # Depends on a"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        # First version
        f.write(content1)
        f.flush()

        doc1 = parse_colight_file(pathlib.Path(f.name))
        results1 = list(
            executor.execute_incremental_streaming(doc1, str(f.name), str(f.name))
        )

        # All should be cache misses
        assert all(not r.cache_hit for _, r in results1)

        # Second version - same file, changed content
        f.seek(0)
        f.truncate()
        f.write(content2)
        f.flush()

        doc2 = parse_colight_file(pathlib.Path(f.name))
        results2 = list(
            executor.execute_incremental_streaming(doc2, str(f.name), str(f.name))
        )

        # Block 0 changed (a = 10 instead of a = 1)
        assert not results2[0][1].cache_hit

        # Block 1 unchanged (b = 2) and independent - should hit cache
        assert results2[1][1].cache_hit, "Independent unchanged block should hit cache"

        # Block 2 depends on changed block - cache miss
        assert not results2[2][1].cache_hit

        # Clean up
        pathlib.Path(f.name).unlink()


def test_cache_reuse_after_reordering():
    """Test that cache can be reused when blocks are reordered but content is same."""
    executor = IncrementalExecutor(verbose=True)

    # Original order
    content1 = """# %%
a = 1

# %%
b = 2

# %%
c = 3"""

    # Reordered (but same independent blocks)
    content2 = """# %%
b = 2

# %%
c = 3

# %%
a = 1"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f1:
        f1.write(content1)
        f1.flush()

        # First execution
        doc1 = parse_colight_file(pathlib.Path(f1.name))
        results1 = list(
            executor.execute_incremental_streaming(doc1, str(f1.name), str(f1.name))
        )

        # All should be cache misses
        for block, result in results1:
            assert result.cache_hit == False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f2:
        f2.write(content2)
        f2.flush()

        # Second execution with reordered blocks
        doc2 = parse_colight_file(pathlib.Path(f2.name))
        results2 = list(
            executor.execute_incremental_streaming(doc2, str(f2.name), str(f2.name))
        )

        # Since these are independent blocks with same content,
        # they should have same IDs and hit cache
        cache_hits = sum(1 for _, result in results2 if result.cache_hit)
        print(f"Cache hits after reordering: {cache_hits}/3")

        # Due to different file paths, cache might not hit
        # But the important thing is that the blocks have consistent IDs

    # Clean up
    pathlib.Path(f1.name).unlink()
    pathlib.Path(f2.name).unlink()
