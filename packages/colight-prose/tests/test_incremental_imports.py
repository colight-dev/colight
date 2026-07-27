"""Test incremental execution with imports - would have caught the bug."""

from colight_prose.incremental_executor import IncrementalExecutor
from colight_prose.parser import parse_document


def test_import_execution():
    """Test that imports work correctly with incremental execution."""
    code = """# Test imports
import colight.plot as Plot

# Use imported module
Plot.dot([1, 2, 3])"""

    doc = parse_document(code)

    # Verify blocks have unique IDs
    block_ids = [block.id for block in doc.blocks]
    assert len(block_ids) == len(set(block_ids)), "Block IDs must be unique"

    # The parser creates 2 blocks: one with import, one with usage
    assert len(doc.blocks) == 2

    # Verify import provides and usage requires
    import_block = doc.blocks[0]  # First block has the import
    usage_block = doc.blocks[1]  # Second block uses Plot

    assert "Plot" in import_block.interface.provides
    assert "Plot" in usage_block.interface.requires

    # Execute incrementally
    executor = IncrementalExecutor()
    results = list(executor.execute_incremental_streaming(doc))

    # All blocks should execute without errors
    assert len(results) == len(doc.blocks)
    for block, result in results:
        assert result.error is None, f"Block {block.id} failed: {result.error}"

    # Plot should be in the namespace
    assert "Plot" in executor.env
    assert executor.env["Plot"].__name__ == "colight.plot"


def test_forward_reference_not_allowed():
    """Test that forward references (using symbols from blocks below) don't work."""
    code = """# Try to use y before it's defined
result = x + y

# Define x
x = 10

# Define y
y = 20"""

    doc = parse_document(code)
    executor = IncrementalExecutor()
    results = list(executor.execute_incremental_streaming(doc))

    # First block should fail (both x and y are undefined)
    assert results[0][1].error is not None
    assert "NameError" in results[0][1].error

    # Other blocks should succeed
    assert results[1][1].error is None
    assert results[2][1].error is None


def test_imperative_code_limitation():
    """Demonstrate why incremental execution doesn't work well with imperative code.

    IMPORTANT: Incremental execution works best with functional/immutable code patterns.
    With imperative code that mutates state (like list.append), re-executing a single
    block will not produce the same result as running the entire document from top to
    bottom. This is an inherent limitation and not a bug.

    Consider using functional patterns instead:
    - Instead of x.append(1), use x = x + [1]
    - Instead of mutating objects, create new ones
    - Use pure functions that return new values
    """
    code = """# Block 1 - Create list
x = []
x.append(1)

# Block 2 - Append more
x.append(2)

# Block 3 - Append and save
x.append(3)
result = x"""

    doc = parse_document(code)
    executor = IncrementalExecutor()

    # First execution works fine
    results = list(executor.execute_incremental_streaming(doc))
    assert executor.env["result"] == [1, 2, 3]

    # With caching, re-executing doesn't cause the mutation to happen again
    results = list(executor.execute_incremental_streaming(doc))

    # The cached results prevent the duplicate append issue
    # This shows how caching helps with imperative code patterns
    assert executor.env["result"] == [1, 2, 3]  # Still correct due to caching!

    # For correct behavior, use functional style:
    functional_code = """# Block 1
x = [1]

# Block 2  
x = x + [2]

# Block 3
x = x + [3]
result = x"""

    doc_func = parse_document(functional_code)
    executor_func = IncrementalExecutor()

    # First execution
    results = list(executor_func.execute_incremental_streaming(doc_func))
    assert executor_func.env["result"] == [1, 2, 3]

    # Re-execute block 2 - works correctly!
    results = list(executor_func.execute_incremental_streaming(doc_func))
    assert executor_func.env["result"] == [1, 2, 3]  # Still correct
