"""Tests for the new executor implementation."""

import textwrap

import libcst as cst

from colight_prose.executor import BlockExecutor, DocumentExecutor
from colight_prose.model import Block, Element, TagSet
from colight_prose.parser import parse_document


class TestBlockExecutor:
    """Test the BlockExecutor class."""

    def test_init(self):
        """Test executor initialization."""
        executor = BlockExecutor()

        # Check environment is set up
        assert "__name__" in executor.env
        assert "__builtins__" in executor.env
        assert "colight" in executor.env
        assert "np" in executor.env

    def test_execute_simple_statement(self):
        """Test executing a simple statement."""
        # Create a block with a statement
        stmt = cst.parse_module("x = 42").body[0]
        elem = Element(kind="STATEMENT", content=stmt, lineno=1)
        block = Block(elements=[elem], tags=TagSet(), start_line=1)

        executor = BlockExecutor()
        result = executor.execute_block(block)

        assert result.error is None
        assert result.value is None  # Statement has no value
        assert executor.env.get("x") == 42

    def test_execute_expression(self):
        """Test executing an expression."""
        # Create a block with an expression
        expr_stmt = cst.parse_module("2 + 3").body[0]
        elem = Element(kind="EXPRESSION", content=expr_stmt, lineno=1)
        block = Block(elements=[elem], tags=TagSet(), start_line=1)

        executor = BlockExecutor()
        result = executor.execute_block(block)

        assert result.error is None
        assert result.value == 5

    def test_execute_with_output(self):
        """Test capturing stdout."""
        # Create a block that prints
        stmt = cst.parse_module('print("Hello")').body[0]
        elem = Element(kind="EXPRESSION", content=stmt, lineno=1)
        block = Block(elements=[elem], tags=TagSet(), start_line=1)

        executor = BlockExecutor()
        result = executor.execute_block(block)

        assert result.error is None
        assert result.output.strip() == "Hello"

    def test_execute_with_error(self):
        """Test handling execution errors."""
        # Create a block with an error
        stmt = cst.parse_module("1/0").body[0]
        elem = Element(kind="EXPRESSION", content=stmt, lineno=1)
        block = Block(elements=[elem], tags=TagSet(), start_line=1)

        executor = BlockExecutor()
        result = executor.execute_block(block)

        assert result.error is not None
        assert "ZeroDivisionError" in result.error
        assert result.value is None

    def test_persistent_namespace(self):
        """Test that namespace persists between blocks."""
        executor = BlockExecutor()

        # First block: define variable
        stmt1 = cst.parse_module("x = 10").body[0]
        elem1 = Element(kind="STATEMENT", content=stmt1, lineno=1)
        block1 = Block(elements=[elem1], tags=TagSet(), start_line=1)

        # Second block: use variable
        stmt2 = cst.parse_module("x * 2").body[0]
        elem2 = Element(kind="EXPRESSION", content=stmt2, lineno=1)
        block2 = Block(elements=[elem2], tags=TagSet(), start_line=1)

        result1 = executor.execute_block(block1)
        assert result1.error is None

        result2 = executor.execute_block(block2)
        assert result2.error is None
        assert result2.value == 20

    def test_mixed_elements(self):
        """Test block with prose, statements, and expression."""
        prose = Element(kind="PROSE", content="# Calculate", lineno=1)
        stmt = cst.parse_module("y = 5").body[0]
        stmt_elem = Element(kind="STATEMENT", content=stmt, lineno=2)
        expr = cst.parse_module("y + 1").body[0]
        expr_elem = Element(kind="EXPRESSION", content=expr, lineno=3)

        block = Block(
            elements=[prose, stmt_elem, expr_elem], tags=TagSet(), start_line=1
        )

        executor = BlockExecutor()
        result = executor.execute_block(block)

        assert result.error is None
        assert result.value == 6
        assert executor.env.get("y") == 5

    def test_colight_visualization(self):
        """Test Colight visualization capture."""
        # Create a simple list that will be visualized
        code = "[1, 2, 3, 4]"

        doc = parse_document(code)
        executor = BlockExecutor()
        result = executor.execute_block(doc.blocks[0])

        assert result.error is None
        assert result.value == [1, 2, 3, 4]
        assert result.colight_bytes is not None
        assert len(result.colight_bytes) > 0

    def test_reset(self):
        """Test resetting the executor."""
        executor = BlockExecutor()

        # Add some state
        stmt = cst.parse_module("test_var = 123").body[0]
        elem = Element(kind="STATEMENT", content=stmt, lineno=1)
        block = Block(elements=[elem], tags=TagSet(), start_line=1)
        executor.execute_block(block)

        assert "test_var" in executor.env

        # Reset
        executor.reset()

        assert "test_var" not in executor.env
        assert "colight" in executor.env  # Setup should be re-run


class TestDocumentExecutor:
    """Test the DocumentExecutor class."""

    def test_execute_document(self):
        """Test executing a complete document."""
        source = textwrap.dedent("""
            # Setup
            x = 10
            
            # Calculate
            y = x * 2
            y
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        results, namespace = executor.execute(doc)

        assert len(results) == 2
        assert results[0].error is None
        assert results[1].error is None
        assert results[1].value == 20
        assert namespace["x"] == 10
        assert namespace["y"] == 20

    def test_execute_with_error_stops(self):
        """Test that execution stops on error by default."""
        source = textwrap.dedent("""
            x = 1
            
            1/0
            
            y = 2
        """).strip()

        doc = parse_document(source)
        # With blank lines, this creates 3 blocks
        assert len(doc.blocks) == 3

        executor = DocumentExecutor()
        results, _ = executor.execute(doc)

        # Execution should stop at the error in block 2
        assert len(results) == 3  # Always returns results for all blocks
        assert results[0].error is None  # First block succeeds
        assert results[1].error is not None  # Second block has error
        assert "ZeroDivisionError" in results[1].error
        # Third block should have empty result (execution stopped)
        assert results[2].error is None
        assert results[2].output == ""
        assert results[2].value is None

    def test_execute_single(self):
        """Test execute_single method."""
        source = textwrap.dedent("""
            # First visualization
            [1, 2]
            
            # Second visualization
            [3, 4]
        """).strip()

        doc = parse_document(source)
        executor = DocumentExecutor()
        colight_bytes = executor.execute_single(doc)

        assert len(colight_bytes) == 2  # 2 visualizations
        assert colight_bytes[0] is not None
        assert colight_bytes[1] is not None
