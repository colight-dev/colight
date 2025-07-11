"""Tests to demonstrate the cache key issue with provider ordering."""

import libcst as cst
from colight_live.incremental_executor import IncrementalExecutor
from colight_site.model import Block, BlockInterface, Document, Element, TagSet


def create_test_block(
    block_id: str, provides: list, requires: list, content: str = ""
) -> Block:
    """Helper to create a test block."""
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


def test_cache_key_provider_order_issue():
    """Test cache key stability when providers are in different order but deps are same.

    This tests the specific issue mentioned in the report where dep_keys
    might be in different order if providers are processed differently.
    """
    # First scenario: block IDs are 1, 2, 3
    executor1 = IncrementalExecutor()
    blocks1 = [
        create_test_block("1", provides=["x"], requires=[], content="x = 1"),
        create_test_block("2", provides=["y"], requires=[], content="y = 2"),
        create_test_block(
            "3", provides=["result"], requires=["x", "y"], content="result = x + y"
        ),
    ]
    doc1 = Document(blocks=blocks1)
    results1 = executor1.execute_incremental(doc1)

    # Get cache key components
    cache_key1 = executor1.block_states["3"].cache_key

    # Second scenario: same content but block IDs are different (simulating different execution order)
    executor2 = IncrementalExecutor()
    blocks2 = [
        create_test_block(
            "10", provides=["y"], requires=[], content="y = 2"
        ),  # y comes first
        create_test_block(
            "20", provides=["x"], requires=[], content="x = 1"
        ),  # x comes second
        create_test_block(
            "30", provides=["result"], requires=["x", "y"], content="result = x + y"
        ),
    ]
    doc2 = Document(blocks=blocks2)
    results2 = executor2.execute_incremental(doc2)

    cache_key2 = executor2.block_states["30"].cache_key

    # The cache keys should be the same since the content and dependencies are identical
    print(f"Cache key 1: {cache_key1}")
    print(f"Cache key 2: {cache_key2}")

    # This might fail if the provider block IDs affect the cache key
    assert (
        cache_key1 == cache_key2
    ), f"Cache keys differ when provider order changes: {cache_key1} != {cache_key2}"


def test_dependency_keys_ordering():
    """Test that dependency keys are computed in consistent order."""
    executor = IncrementalExecutor()

    # Create blocks where multiple symbols come from the same provider
    blocks = [
        create_test_block(
            "1", provides=["a", "b", "c"], requires=[], content="a = 1\nb = 2\nc = 3"
        ),
        create_test_block(
            "2", provides=["x", "y", "z"], requires=[], content="x = 4\ny = 5\nz = 6"
        ),
        # This block requires symbols in a specific order
        create_test_block(
            "3",
            provides=["result"],
            requires=["z", "a", "y", "b", "x", "c"],
            content="result = a + b + c + x + y + z",
        ),
    ]
    doc = Document(blocks=blocks)

    # Execute and check the cache key computation
    results = executor.execute_incremental(doc)
    cache_key = executor.block_states["3"].cache_key

    # Now create the same blocks but with requires in different order
    executor2 = IncrementalExecutor()
    blocks2 = [
        create_test_block(
            "1", provides=["a", "b", "c"], requires=[], content="a = 1\nb = 2\nc = 3"
        ),
        create_test_block(
            "2", provides=["x", "y", "z"], requires=[], content="x = 4\ny = 5\nz = 6"
        ),
        # Same symbols but different order
        create_test_block(
            "3",
            provides=["result"],
            requires=["c", "y", "a", "z", "x", "b"],
            content="result = a + b + c + x + y + z",
        ),
    ]
    doc2 = Document(blocks=blocks2)

    results2 = executor2.execute_incremental(doc2)
    cache_key2 = executor2.block_states["3"].cache_key

    # Should be stable despite reordering
    assert (
        cache_key == cache_key2
    ), f"Cache keys differ with reordered requires: {cache_key} != {cache_key2}"


def test_cache_key_with_changing_providers():
    """Test cache key when the same symbol is provided by different blocks over time."""
    executor = IncrementalExecutor()

    # Initial state: block 1 provides 'x'
    blocks1 = [
        create_test_block("1", provides=["x"], requires=[], content="x = 10"),
        create_test_block("2", provides=["y"], requires=["x"], content="y = x * 2"),
    ]
    doc1 = Document(blocks=blocks1)
    results1 = executor.execute_incremental(doc1)
    cache_key1 = executor.block_states["2"].cache_key

    # Add a new block that also provides 'x' (overwrites the previous one)
    blocks2 = [
        create_test_block("1", provides=["x"], requires=[], content="x = 10"),
        create_test_block("2", provides=["y"], requires=["x"], content="y = x * 2"),
        create_test_block(
            "3", provides=["x"], requires=[], content="x = 20"
        ),  # New provider of x
        create_test_block(
            "4", provides=["z"], requires=["x"], content="z = x + 5"
        ),  # Uses new x
    ]
    doc2 = Document(blocks=blocks2)
    results2 = executor.execute_incremental(doc2)

    # Block 2's cache key should not change (still uses block 1's x)
    cache_key2_after = executor.block_states["2"].cache_key
    assert (
        cache_key1 == cache_key2_after
    ), "Block 2's cache key changed but its dependencies didn't"

    # Block 4 should use block 3's x
    cache_key4 = executor.block_states["4"].cache_key

    # Now test if block 2 is moved after block 3, it should get a different cache key
    blocks3 = [
        create_test_block("1", provides=["x"], requires=[], content="x = 10"),
        create_test_block(
            "3", provides=["x"], requires=[], content="x = 20"
        ),  # Overwrites x
        create_test_block(
            "2", provides=["y"], requires=["x"], content="y = x * 2"
        ),  # Now uses block 3's x
        create_test_block("4", provides=["z"], requires=["x"], content="z = x + 5"),
    ]
    doc3 = Document(blocks=blocks3)
    executor3 = IncrementalExecutor()
    results3 = executor3.execute_incremental(doc3)

    cache_key2_moved = executor3.block_states["2"].cache_key
    # Should be different because now block 2 depends on block 3's x instead of block 1's x
    assert (
        cache_key1 != cache_key2_moved
    ), "Block 2's cache key should change when its provider changes"
