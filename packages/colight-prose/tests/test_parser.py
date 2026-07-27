"""Tests for the new parser implementation."""

import textwrap

from colight_prose.model import EmptyLine, TagSet
from colight_prose.parser import (
    BlankLine,
    # New typed elements
    CodeLine,
    CommentLine,
    PragmaLine,
    RawBlock,
    build_document,
    group_blocks,
    lex,
    parse_document,
)


class TestLexer:
    """Test the lexing phase."""

    def test_lex_empty(self):
        """Test lexing empty source."""
        elements = lex("")
        assert len(elements) == 0

    def test_lex_comments(self):
        """Test lexing different comment types."""
        source = textwrap.dedent("""
            # Regular comment
            #| pragma comment
            #%% another pragma
            
            # Empty comment above
        """).strip()

        elements = lex(source)
        assert len(elements) == 5
        assert isinstance(elements[0], CommentLine)
        assert elements[0].content == "Regular comment"  # Comment marker stripped
        assert isinstance(elements[1], PragmaLine)
        assert elements[1].content == "| pragma comment"
        assert isinstance(elements[2], PragmaLine)
        assert elements[2].content == "%% another pragma"
        assert isinstance(elements[3], BlankLine)
        assert isinstance(elements[4], CommentLine)

    def test_lex_code(self):
        """Test lexing code statements."""
        source = textwrap.dedent("""
            x = 1
            y = 2
            print(x + y)
        """).strip()

        elements = lex(source)
        assert len(elements) == 3
        assert all(isinstance(element, CodeLine) for element in elements)
        # Check that code is parsed as CST nodes
        import libcst as cst

        # Type narrowing - we know all elements are CodeLine from the check above
        code_elements = [elem for elem in elements if isinstance(elem, CodeLine)]
        assert len(code_elements) == len(elements)  # All are CodeLine
        assert all(
            isinstance(element.content, cst.CSTNode) for element in code_elements
        )

    def test_lex_mixed(self):
        """Test lexing mixed content."""
        source = textwrap.dedent("""
            # Title
            #| hide-code
            
            x = 1
            # Comment in code
            y = 2
        """).strip()

        elements = lex(source)
        # Check element types
        assert isinstance(elements[0], CommentLine)  # Title
        assert isinstance(elements[1], PragmaLine)  # hide-code
        assert isinstance(elements[2], BlankLine)
        assert isinstance(elements[3], CodeLine)  # x = 1
        assert isinstance(elements[4], CommentLine)  # Comment in code
        assert isinstance(elements[5], CodeLine)  # y = 2

    def test_lex_markdown_headers(self):
        """Test lexing markdown headers with double #."""
        source = textwrap.dedent("""
            # Regular comment
            # # H1 Header
            # ## H2 Header
            # ### H3 Header
        """).strip()

        elements = lex(source)
        assert len(elements) == 4
        assert all(isinstance(elem, CommentLine) for elem in elements)
        # Type narrow to CommentLine
        comment_elements = [elem for elem in elements if isinstance(elem, CommentLine)]
        assert len(comment_elements) == 4
        assert comment_elements[0].content == "Regular comment"
        assert comment_elements[1].content == "# H1 Header"
        assert comment_elements[2].content == "## H2 Header"
        assert comment_elements[3].content == "### H3 Header"

    def test_lex_pep723_block(self):
        """Test that PEP 723 blocks are skipped."""
        source = textwrap.dedent("""
            # /// script
            # dependencies = ["numpy"]
            # ///
            
            import numpy as np
        """).strip()

        elements = lex(source)
        # Should only have blank line and import
        assert len(elements) == 2
        assert isinstance(elements[0], BlankLine)
        assert isinstance(elements[1], CodeLine)

    def test_lex_trailing_prose(self):
        """Test lexing prose that appears after code (footer content)."""
        source = textwrap.dedent("""
            1 + 1
            
            # hello
        """).strip()

        elements = lex(source)
        assert len(elements) == 3
        assert isinstance(elements[0], CodeLine)  # 1 + 1
        assert isinstance(elements[1], BlankLine)
        assert isinstance(elements[2], CommentLine)  # hello
        assert elements[2].content == "hello"


class TestBlockGrouping:
    """Test the block grouping phase."""

    def test_group_single_block(self):
        """Test grouping a single block."""
        import libcst as cst

        elements = [
            CommentLine(content="Title", lineno=1),
            CodeLine(content=cst.parse_module("x = 1").body[0], lineno=2),
        ]

        blocks = group_blocks(elements)
        assert len(blocks) == 1
        assert len(blocks[0].prose_lines) == 1
        assert len(blocks[0].code_nodes) == 1

    def test_group_multiple_blocks(self):
        """Test grouping with blank line separators."""
        import libcst as cst

        elements = [
            CommentLine(content="Block 1", lineno=1),
            CodeLine(content=cst.parse_module("x = 1").body[0], lineno=2),
            BlankLine(lineno=3),
            CommentLine(content="Block 2", lineno=4),
            CodeLine(content=cst.parse_module("y = 2").body[0], lineno=5),
        ]

        blocks = group_blocks(elements)
        assert len(blocks) == 2
        assert blocks[0].prose_lines == ["Block 1"]
        assert blocks[1].prose_lines == ["Block 2"]

    def test_group_pragma_handling(self):
        """Test that pragmas are collected properly."""
        import libcst as cst

        elements = [
            PragmaLine(content="| hide-code", lineno=1),
            CommentLine(content="Text", lineno=2),
            CodeLine(content=cst.parse_module("x = 1").body[0], lineno=3),
        ]

        blocks = group_blocks(elements)
        assert len(blocks) == 1
        assert blocks[0].pragma_lines == ["| hide-code"]

    def test_group_empty_comment_continuation(self):
        """Test empty comments between code statements."""
        import libcst as cst

        elements = [
            CodeLine(content=cst.parse_module("x = 1").body[0], lineno=1),
            CommentLine(content="", lineno=2),  # Empty comment
            CodeLine(content=cst.parse_module("y = 2").body[0], lineno=3),
        ]

        blocks = group_blocks(elements)
        assert len(blocks) == 1
        assert len(blocks[0].code_nodes) == 3  # Two code nodes + one EmptyLine
        # Check that the middle item is an EmptyLine sentinel
        assert isinstance(blocks[0].code_nodes[1], EmptyLine)


class TestDocumentBuilder:
    """Test the document building phase."""

    def test_build_simple_document(self):
        """Test building a simple document."""
        raw_blocks = [
            RawBlock(
                prose_lines=["Hello"], code_nodes=[], pragma_lines=[], start_line=1
            )
        ]

        doc = build_document(raw_blocks)
        assert len(doc.blocks) == 1
        assert len(doc.blocks[0].elements) == 1
        assert doc.blocks[0].elements[0].kind == "PROSE"
        assert doc.blocks[0].elements[0].content == "Hello"

    def test_build_with_pragmas(self):
        """Test building with pragma tags."""
        raw_blocks = [
            RawBlock(
                prose_lines=["Text"],  # Need some content
                code_nodes=[],
                pragma_lines=["| hide-code show-visuals"],
                start_line=1,
            )
        ]

        doc = build_document(raw_blocks)
        assert len(doc.blocks) == 1
        assert doc.blocks[0].tags.flags == {"hide-code", "show-visuals"}

    def test_build_with_file_pragma(self):
        """Test building with file-level pragma."""
        raw_blocks = []
        file_pragma = TagSet(frozenset({"hide-all-code"}))

        doc = build_document(raw_blocks, file_pragma)
        assert doc.tags.flags == {"hide-all-code"}

    def test_pragma_with_blank_line(self):
        """Test that pragmas work even with blank lines after them."""
        source = textwrap.dedent("""
            #| hide-code
            
            # This is prose
            x = 42
        """).strip()

        doc = parse_document(source)
        assert len(doc.blocks) == 1
        assert "hide-code" in doc.blocks[0].tags.flags
        assert not doc.blocks[0].tags.show_code()

    def test_empty_comments_preserved_with_sentinels(self):
        """Test that empty comments between statements are preserved using EmptyLine sentinels."""
        source = textwrap.dedent("""
            x = 1
            #
            y = 2
            #
            z = 3
        """).strip()

        doc = parse_document(source)
        # All statements are combined into one element with sentinels
        assert len(doc.blocks) == 1
        assert len(doc.blocks[0].elements) == 1
        assert doc.blocks[0].elements[0].kind == "STATEMENT"

        # The content should have EmptyLine sentinels
        content = doc.blocks[0].elements[0].content
        assert isinstance(content, list)
        assert len(content) == 5  # 3 statements + 2 EmptyLine sentinels

        from colight_prose.model import EmptyLine

        assert isinstance(content[1], EmptyLine)
        assert isinstance(content[3], EmptyLine)

        # Check that get_source() properly handles EmptyLine sentinels
        source_output = doc.blocks[0].elements[0].get_source()
        assert source_output == "x = 1\n\ny = 2\n\nz = 3"


class TestEndToEnd:
    """Test the complete parsing pipeline."""

    def test_parse_simple_document(self):
        """Test parsing a simple document."""
        source = textwrap.dedent("""
            # Hello World
            print("Hello, World!")
        """).strip()

        doc = parse_document(source)
        assert len(doc.blocks) == 1
        assert len(doc.blocks[0].elements) == 2
        assert doc.blocks[0].elements[0].kind == "PROSE"
        assert doc.blocks[0].elements[1].kind == "EXPRESSION"

    def test_parse_with_pragmas(self):
        """Test parsing with various pragmas."""
        source = textwrap.dedent("""
            #| hide-all-code
            
            # Document Title
            #| show-code
            x = 42
            x
        """).strip()

        doc = parse_document(source)

        # File-level pragma
        assert "hide-all-code" in doc.tags.flags

        # Without blank line between comment and code, they form one block
        assert len(doc.blocks) == 1

        # Block has prose, then code
        assert doc.blocks[0].elements[0].kind == "PROSE"
        assert (
            doc.blocks[0].elements[0].content == "Document Title"
        )  # Comment marker stripped
        assert doc.blocks[0].elements[1].kind == "STATEMENT"
        assert doc.blocks[0].elements[2].kind == "EXPRESSION"

        # Block has local pragma
        assert "show-code" in doc.blocks[0].tags.flags

    def test_parse_multiple_blocks(self):
        """Test parsing multiple blocks."""
        source = textwrap.dedent("""
            # Block 1
            x = 1
            
            # Block 2
            y = 2
            
            # Block 3
            x + y
        """).strip()

        doc = parse_document(source)
        assert len(doc.blocks) == 3

        # Check last elements
        assert doc.blocks[0].elements[-1].kind == "STATEMENT"
        assert doc.blocks[1].elements[-1].kind == "STATEMENT"
        assert doc.blocks[2].elements[-1].kind == "EXPRESSION"

    def test_parse_code_classification(self):
        """Test that code is correctly classified as statement vs expression."""
        source = textwrap.dedent("""
            # Statements
            x = 1
            y = 2
            
            # Expression
            x + y
            
            # Mixed
            z = 3
            z * 2
        """).strip()

        doc = parse_document(source)

        # First block - all statements
        code_elems = doc.blocks[0].get_code_elements()
        assert all(e.kind == "STATEMENT" for e in code_elems)

        # Second block - expression
        assert doc.blocks[1].elements[-1].kind == "EXPRESSION"

        # Third block - statement then expression
        code_elems = doc.blocks[2].get_code_elements()
        assert code_elems[0].kind == "STATEMENT"
        assert code_elems[1].kind == "EXPRESSION"

    def test_parse_trailing_prose(self):
        """Test parsing document with prose after code (footer content)."""
        source = textwrap.dedent("""
            1 + 1
            
            # hello world
            # this is trailing prose
        """).strip()

        doc = parse_document(source)
        assert len(doc.blocks) == 2

        # First block contains the expression
        assert len(doc.blocks[0].elements) == 1
        assert doc.blocks[0].elements[0].kind == "EXPRESSION"

        # Second block contains the prose
        assert len(doc.blocks[1].elements) == 1
        assert doc.blocks[1].elements[0].kind == "PROSE"
        assert (
            doc.blocks[1].elements[0].content == "hello world\nthis is trailing prose"
        )
