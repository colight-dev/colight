"""Tests for the colight_prose.model module."""

import libcst as cst
import pytest

from colight_prose.model import Block, Document, Element, TagSet


class TestTagSet:
    """Test the TagSet class."""

    def test_empty_tagset_shows_everything(self):
        """Empty TagSet should show all content by default."""
        ts = TagSet()
        assert ts.show_code()
        assert ts.show_prose()
        assert ts.show_statements()
        assert ts.show_visuals()

    def test_hide_flags(self):
        """Test hide-* flags work correctly."""
        ts = TagSet(frozenset({"hide-code", "hide-visuals"}))
        assert not ts.show_code()
        assert ts.show_prose()
        assert not ts.show_statements()  # Requires code to be visible
        assert not ts.show_visuals()

    def test_show_flags_override_hide(self):
        """Test that show-* flags override hide-* flags."""
        ts = TagSet(frozenset({"hide-code", "show-code"}))
        assert ts.show_code()

        ts = TagSet(frozenset({"hide-all-visuals", "show-visuals"}))
        assert ts.show_visuals()

    def test_hide_all_flags(self):
        """Test hide-all-* flags work correctly."""
        ts = TagSet(frozenset({"hide-all-statements"}))
        assert ts.show_code()
        assert not ts.show_statements()

    def test_statements_require_code(self):
        """Test that statements are only shown if code is visible."""
        ts = TagSet(frozenset({"hide-code", "show-statements"}))
        assert not ts.show_statements()  # Code is hidden

        ts = TagSet(frozenset({"show-code", "show-statements"}))
        assert ts.show_statements()  # Both visible

    def test_merge_last_writer_wins(self):
        """Test merging with last writer wins semantics."""
        ts1 = TagSet(frozenset({"hide-code", "show-prose"}))
        ts2 = TagSet(frozenset({"show-code", "hide-visuals"}))

        merged = ts1.merged(ts2)
        assert "show-code" in merged.flags
        assert "show-prose" in merged.flags
        assert "hide-visuals" in merged.flags
        assert "hide-code" not in merged.flags  # Overwritten

    def test_merge_operator(self):
        """Test the | operator for merging."""
        ts1 = TagSet(frozenset({"hide-code"}))
        ts2 = TagSet(frozenset({"show-code"}))

        merged = ts1 | ts2
        assert merged.flags == frozenset({"show-code"})

    def test_from_pragma_content(self):
        """Test creating TagSet from pragma content string."""
        ts = TagSet.from_pragma_content("hide-code show-visuals")
        assert ts.flags == frozenset({"hide-code", "show-visuals"})

        # Test normalization of singular to plural
        ts = TagSet.from_pragma_content("hide-statement show-visual")
        assert ts.flags == frozenset({"hide-statements", "show-visuals"})

        # Test hide-all doesn't get pluralized
        ts = TagSet.from_pragma_content("hide-all-statement")
        assert ts.flags == frozenset({"hide-all-statement"})

    def test_immutability(self):
        """Test that TagSet is immutable."""
        ts = TagSet(frozenset({"hide-code"}))
        with pytest.raises(AttributeError):
            ts.flags = frozenset({"show-code"})  # type: ignore


class TestElement:
    """Test the Element class."""

    def test_prose_element(self):
        """Test prose element creation and source extraction."""
        elem = Element(kind="PROSE", content="# Hello World", lineno=1)
        assert elem.get_source() == "# Hello World"

    def test_statement_element(self):
        """Test statement element with CST node."""
        # Create a simple statement: x = 1
        stmt = cst.SimpleStatementLine(
            [
                cst.Assign(
                    targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Integer("1")
                )
            ]
        )
        elem = Element(kind="STATEMENT", content=stmt, lineno=2)
        assert elem.get_source().strip() == "x = 1"

    def test_expression_element(self):
        """Test expression element with CST node."""
        # Create an expression: 2 + 3
        expr = cst.BinaryOperation(
            left=cst.Integer("2"), operator=cst.Add(), right=cst.Integer("3")
        )
        elem = Element(kind="EXPRESSION", content=expr, lineno=3)
        assert elem.get_source().strip() == "2 + 3"

    def test_multiple_statements(self):
        """Test element with multiple statements."""
        stmt1 = cst.SimpleStatementLine(
            [
                cst.Assign(
                    targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Integer("1")
                )
            ]
        )
        stmt2 = cst.SimpleStatementLine(
            [
                cst.Assign(
                    targets=[cst.AssignTarget(cst.Name("y"))], value=cst.Integer("2")
                )
            ]
        )
        elem = Element(kind="STATEMENT", content=[stmt1, stmt2], lineno=4)
        source = elem.get_source()
        assert "x = 1" in source
        assert "y = 2" in source


class TestBlock:
    """Test the Block class."""

    def test_empty_block(self):
        """Test empty block detection."""
        block = Block(elements=[], tags=TagSet(), start_line=1)
        assert block.is_empty
        assert block.last_element is None
        assert not block.has_expression_result

    def test_prose_only_block(self):
        """Test block with only prose elements."""
        elem = Element(kind="PROSE", content="# Title", lineno=1)
        block = Block(elements=[elem], tags=TagSet(), start_line=1)

        assert block.is_empty  # No executable code
        assert len(block.get_prose_elements()) == 1
        assert len(block.get_code_elements()) == 0
        assert not block.has_expression_result

    def test_code_block(self):
        """Test block with code elements."""
        stmt = cst.SimpleStatementLine([cst.Pass()])
        expr = cst.Integer("42")

        elem1 = Element(kind="STATEMENT", content=stmt, lineno=1)
        elem2 = Element(kind="EXPRESSION", content=expr, lineno=2)

        block = Block(elements=[elem1, elem2], tags=TagSet(), start_line=1)

        assert not block.is_empty
        assert block.has_expression_result
        assert block.last_element == elem2
        assert len(block.get_code_elements()) == 2

    def test_compile_once_caching(self):
        """Test that compile_once caches compilation."""
        stmt = cst.SimpleStatementLine(
            [
                cst.Assign(
                    targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Integer("1")
                )
            ]
        )
        elem = Element(kind="STATEMENT", content=stmt, lineno=1)
        block = Block(elements=[elem], tags=TagSet(), start_line=1)

        # First compilation
        block.compile_once()
        exec_code1 = block._exec_code

        # Second compilation should return cached result
        block.compile_once()
        exec_code2 = block._exec_code

        assert exec_code1 is exec_code2  # Same object

    def test_compile_with_expression(self):
        """Test compilation splits statements from final expression."""
        stmt = cst.SimpleStatementLine(
            [
                cst.Assign(
                    targets=[cst.AssignTarget(cst.Name("x"))], value=cst.Integer("1")
                )
            ]
        )
        expr = cst.Name("x")

        elem1 = Element(kind="STATEMENT", content=stmt, lineno=1)
        elem2 = Element(kind="EXPRESSION", content=expr, lineno=2)

        block = Block(elements=[elem1, elem2], tags=TagSet(), start_line=1)
        block.compile_once()

        assert block._exec_code is not None  # Statements compiled
        assert block._eval_code is not None  # Expression compiled


class TestDocument:
    """Test the Document class."""

    def test_empty_document(self):
        """Test empty document."""
        doc = Document(blocks=[])
        assert not doc.has_content
        assert len(doc.get_executable_blocks()) == 0

    def test_document_with_blocks(self):
        """Test document with mixed blocks."""
        # Empty block
        block1 = Block(elements=[], tags=TagSet(), start_line=1)

        # Block with code
        stmt = cst.SimpleStatementLine([cst.Pass()])
        elem = Element(kind="STATEMENT", content=stmt, lineno=2)
        block2 = Block(elements=[elem], tags=TagSet(), start_line=2)

        doc = Document(blocks=[block1, block2], tags=TagSet())

        assert doc.has_content
        assert len(doc.get_executable_blocks()) == 1
        assert doc.get_executable_blocks()[0] == block2

    def test_document_tags(self):
        """Test document with file-level tags."""
        tags = TagSet(frozenset({"hide-all-code"}))
        doc = Document(blocks=[], tags=tags)

        assert not doc.tags.show_code()
