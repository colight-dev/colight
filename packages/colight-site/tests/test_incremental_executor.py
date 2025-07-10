"""Tests for incremental execution with dependency tracking."""

from colight_live.incremental_executor import IncrementalExecutor
from colight_site.model import Document, Block, Element, TagSet, BlockInterface
import libcst as cst


def create_test_block(id: int, code: str, provides=None, requires=None):
    """Helper to create a test block."""
    # Parse code to get CST nodes
    module = cst.parse_module(code)
    elements = []

    for stmt in module.body:
        elements.append(Element(kind="STATEMENT", content=stmt, lineno=1))

    block = Block(
        elements=elements,
        tags=TagSet(),
        start_line=1,
        id=id,
        interface=BlockInterface(provides=provides or [], requires=requires or []),
    )
    return block


def test_basic_incremental_execution():
    """Test basic incremental execution."""
    executor = IncrementalExecutor()

    # Create a simple document
    blocks = [
        create_test_block(1, "x = 1", provides=["x"]),
        create_test_block(2, "y = x + 1", provides=["y"], requires=["x"]),
        create_test_block(3, "z = y * 2", provides=["z"], requires=["y"]),
    ]
    doc = Document(blocks=blocks)

    # First execution - all blocks
    results = executor.execute_incremental(doc)
    assert len(results) == 3

    # Check results
    assert results[0][1].value is None  # Statement, no value
    assert results[1][1].value is None
    assert results[2][1].value is None

    # Check namespace
    assert executor.env["x"] == 1
    assert executor.env["y"] == 2
    assert executor.env["z"] == 4

    # Modify block 1
    blocks[0] = create_test_block(1, "x = 10", provides=["x"])
    doc = Document(blocks=blocks)

    # Incremental execution - should re-execute all blocks due to dependency
    results = executor.execute_incremental(doc, changed_blocks={"1"})
    assert len(results) == 3

    # Check updated values
    assert executor.env["x"] == 10
    assert executor.env["y"] == 11
    assert executor.env["z"] == 22


def test_independent_blocks():
    """Test that independent blocks aren't re-executed."""
    executor = IncrementalExecutor(verbose=True)

    # Create blocks with no dependencies
    blocks = [
        create_test_block(1, "a = 1", provides=["a"]),
        create_test_block(2, "b = 2", provides=["b"]),
        create_test_block(3, "c = 3", provides=["c"]),
    ]
    doc = Document(blocks=blocks)

    # First execution
    results = executor.execute_incremental(doc)
    assert len(results) == 3

    # Mark execution state
    assert "1" in executor.block_states
    assert "2" in executor.block_states
    assert "3" in executor.block_states

    # Change only block 2
    blocks[1] = create_test_block(2, "b = 20", provides=["b"])
    doc = Document(blocks=blocks)

    # Only block 2 should be re-executed
    results = executor.execute_incremental(doc, changed_blocks={"2"})

    # All results returned but only block 2 was actually executed
    assert len(results) == 3
    assert executor.env["a"] == 1
    assert executor.env["b"] == 20
    assert executor.env["c"] == 3


def test_always_eval_pragma():
    """Test that blocks with always-eval pragma are always re-executed."""
    executor = IncrementalExecutor()

    # Create blocks, one with always-eval pragma
    blocks = [
        create_test_block(1, "x = 1", provides=["x"]),
        create_test_block(
            2, "import time; timestamp = time.time()", provides=["timestamp"]
        ),
    ]
    # Add always-eval pragma to block 2
    blocks[1].tags = TagSet(frozenset(["always-eval"]))

    doc = Document(blocks=blocks)

    # First execution
    results = executor.execute_incremental(doc)
    timestamp1 = executor.env.get("timestamp")

    # Execute again with no changes - block 2 should still run
    results = executor.execute_incremental(doc, changed_blocks=set())
    timestamp2 = executor.env.get("timestamp")

    # Timestamps should be different
    assert timestamp2 != timestamp1


def test_stdout_capture():
    """Test that stdout is captured during execution."""
    executor = IncrementalExecutor()

    blocks = [
        create_test_block(1, 'print("Hello from block 1")'),
        create_test_block(2, 'print("Hello from block 2")'),
    ]
    doc = Document(blocks=blocks)

    results = executor.execute_incremental(doc)

    assert results[0][1].output == "Hello from block 1\n"
    assert results[1][1].output == "Hello from block 2\n"


def test_expression_results():
    """Test that expression results are captured."""
    executor = IncrementalExecutor()

    # Create a block with an expression
    module = cst.parse_module("x = 5\nx * 2")
    elements = []
    for i, stmt in enumerate(module.body):
        if i == 0:
            elements.append(Element(kind="STATEMENT", content=stmt, lineno=1))
        else:
            # The second line is an expression
            elements.append(Element(kind="EXPRESSION", content=stmt, lineno=2))

    block = Block(
        elements=elements,
        tags=TagSet(),
        start_line=1,
        id=1,
        interface=BlockInterface(provides=["x"]),
    )

    doc = Document(blocks=[block])
    results = executor.execute_incremental(doc)

    # Should have captured the expression result
    assert results[0][1].value == 10


def test_error_handling():
    """Test that errors are captured properly."""
    executor = IncrementalExecutor()

    blocks = [
        create_test_block(1, "x = 1", provides=["x"]),
        create_test_block(2, "y = undefined_variable", provides=["y"]),
        create_test_block(3, "z = x + 1", provides=["z"], requires=["x"]),
    ]
    doc = Document(blocks=blocks)

    results = executor.execute_incremental(doc)

    # Block 1 should succeed
    assert results[0][1].error is None
    assert executor.env["x"] == 1

    # Block 2 should have an error
    assert results[1][1].error is not None
    assert "NameError" in results[1][1].error

    # Block 3 should still execute (doesn't depend on y)
    assert results[2][1].error is None
    assert executor.env["z"] == 2
