"""Tests for cache key stability in incremental executor."""

import libcst as cst
from colight_live.incremental_executor import IncrementalExecutor
from colight_site.model import Block, BlockInterface, Document, Element


def create_test_block(
    block_id: str, provides: list, requires: list, content: str = ""
) -> Block:
    """Helper to create a test block."""
    from colight_site.model import TagSet

    elements = []
    if content:
        # Parse the content as Python code
        module = cst.parse_module(content)
        for stmt in module.body:
            elements.append(Element(kind="STATEMENT", content=stmt, lineno=1))

    block = Block(
        elements=elements,
        tags=TagSet(),
        start_line=1,
        id=int(block_id),
        interface=BlockInterface(provides=provides, requires=requires),
    )
    return block


def test_cache_key_stability_with_reordered_symbols():
    """Test that cache keys are stable when symbols are reordered."""
    executor = IncrementalExecutor()

    # Create a document with blocks that provide symbols
    blocks1 = [
        create_test_block(
            "1", provides=["a", "b"], requires=[], content="a = 1\nb = 2"
        ),
        create_test_block(
            "2", provides=["result"], requires=["a", "b"], content="result = a + b"
        ),
    ]
    doc1 = Document(blocks=blocks1)

    # Execute to establish cache
    results1 = executor.execute_incremental(doc1)
    block2_state1 = executor.block_states["2"]
    cache_key1 = block2_state1.cache_key

    # Clear executor and create same document but with requires in different order
    executor2 = IncrementalExecutor()
    blocks2 = [
        create_test_block(
            "1", provides=["a", "b"], requires=[], content="a = 1\nb = 2"
        ),
        create_test_block(
            "2", provides=["result"], requires=["b", "a"], content="result = a + b"
        ),  # Note: b, a instead of a, b
    ]
    doc2 = Document(blocks=blocks2)

    # Execute second version
    results2 = executor2.execute_incremental(doc2)
    block2_state2 = executor2.block_states["2"]
    cache_key2 = block2_state2.cache_key

    # Cache keys should be the same since the actual dependencies haven't changed
    assert cache_key1 == cache_key2, f"Cache keys differ: {cache_key1} != {cache_key2}"


def test_cache_key_changes_with_dependency_content():
    """Test that cache keys change when dependency content changes."""
    executor = IncrementalExecutor()

    # Initial document
    blocks1 = [
        create_test_block("1", provides=["x"], requires=[], content="x = 10"),
        create_test_block("2", provides=["y"], requires=["x"], content="y = x * 2"),
    ]
    doc1 = Document(blocks=blocks1)

    results1 = executor.execute_incremental(doc1)
    cache_key1 = executor.block_states["2"].cache_key

    # Change content of dependency
    blocks2 = [
        create_test_block(
            "1", provides=["x"], requires=[], content="x = 20"
        ),  # Changed
        create_test_block("2", provides=["y"], requires=["x"], content="y = x * 2"),
    ]
    doc2 = Document(blocks=blocks2)

    results2 = executor.execute_incremental(doc2)
    cache_key2 = executor.block_states["2"].cache_key

    # Cache key should change since dependency changed
    assert (
        cache_key1 != cache_key2
    ), f"Cache keys should differ: {cache_key1} == {cache_key2}"


def test_cache_key_with_multiple_providers():
    """Test cache key stability when multiple symbols come from same provider."""
    executor = IncrementalExecutor()

    # Document where block 1 provides multiple symbols used by block 2
    blocks = [
        create_test_block(
            "1", provides=["a", "b", "c"], requires=[], content="a = 1\nb = 2\nc = 3"
        ),
        create_test_block(
            "2",
            provides=["result"],
            requires=["c", "a", "b"],
            content="result = a + b + c",
        ),
    ]
    doc = Document(blocks=blocks)

    # Execute multiple times to ensure stability
    cache_keys = []
    for i in range(3):
        executor_new = IncrementalExecutor()
        results = executor_new.execute_incremental(doc)
        cache_keys.append(executor_new.block_states["2"].cache_key)

    # All cache keys should be identical
    assert len(set(cache_keys)) == 1, f"Cache keys are not stable: {cache_keys}"


def test_cache_key_with_missing_dependencies():
    """Test cache key computation when dependencies are missing."""
    executor = IncrementalExecutor()

    blocks = [
        create_test_block("1", provides=["x"], requires=[], content="x = 1"),
        create_test_block(
            "2", provides=["y"], requires=["x", "missing"], content="y = x + missing"
        ),
    ]
    doc = Document(blocks=blocks)

    # Should not crash even with missing dependency
    results = executor.execute_incremental(doc)
    cache_key = executor.block_states["2"].cache_key

    # Cache key should exist and be deterministic
    assert cache_key
    assert len(cache_key) == 16  # Should be truncated SHA hash


def test_cache_hit_after_reordering():
    """Test that we get cache hits even after reordering symbols."""
    executor = IncrementalExecutor()

    # First execution
    blocks1 = [
        create_test_block("1", provides=["a"], requires=[], content="a = 1"),
        create_test_block("2", provides=["b"], requires=[], content="b = 2"),
        create_test_block(
            "3", provides=["result"], requires=["a", "b"], content="result = a + b"
        ),
    ]
    doc1 = Document(blocks=blocks1)

    results1 = executor.execute_incremental(doc1)

    # Clear cache but keep executor (simulating restart)
    cache_key_before = executor.block_states["3"].cache_key

    # Second execution with reordered requires
    blocks2 = [
        create_test_block("1", provides=["a"], requires=[], content="a = 1"),
        create_test_block("2", provides=["b"], requires=[], content="b = 2"),
        create_test_block(
            "3", provides=["result"], requires=["b", "a"], content="result = a + b"
        ),  # Reordered
    ]
    doc2 = Document(blocks=blocks2)

    # IMPORTANT: Currently this will fail because the cache keys are different
    # After fix, we should get a cache hit
    results2 = executor.execute_incremental(doc2)
    cache_key_after = executor.block_states["3"].cache_key

    # This assertion will fail before the fix is implemented
    assert (
        cache_key_before == cache_key_after
    ), f"Cache keys should be stable: {cache_key_before} != {cache_key_after}"
