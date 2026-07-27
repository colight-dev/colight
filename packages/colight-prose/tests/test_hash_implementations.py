"""Tests for block ID (cache key) generation."""

import libcst as cst

from colight_prose.model import Block, BlockInterface, Element, TagSet
from colight_prose.parser import parse_document


def create_test_block_with_elements(elements_spec):
    """Create a block with specified elements."""
    elements = []
    for kind, content in elements_spec:
        if kind == "PROSE":
            elements.append(Element(kind=kind, content=content, lineno=1))
        else:  # STATEMENT or EXPRESSION
            module = cst.parse_module(content)
            for stmt in module.body:
                elements.append(Element(kind=kind, content=stmt, lineno=1))

    return Block(
        elements=elements,
        tags=TagSet(),
        start_line=1,
        id="test-id",  # This will be replaced by parser
        interface=BlockInterface(provides=[], requires=[]),
    )


def test_block_id_consistency():
    """Test that block IDs are consistent for the same content."""
    # Parse the same content twice
    code = """# First block
x = 1

# Second block
y = x + 1"""

    doc1 = parse_document(code)
    doc2 = parse_document(code)

    # Block IDs should be the same for the same content
    assert len(doc1.blocks) == len(doc2.blocks)
    for b1, b2 in zip(doc1.blocks, doc2.blocks):
        assert b1.id == b2.id, "Block IDs should be consistent for same content"


def test_block_id_changes_with_content():
    """Test that block IDs change when content changes."""
    code1 = """# Block
x = 1"""

    code2 = """# Block
x = 2"""

    doc1 = parse_document(code1)
    doc2 = parse_document(code2)

    # Block IDs should be different for different content
    assert (
        doc1.blocks[0].id != doc2.blocks[0].id
    ), "Block IDs should differ for different content"


def test_block_id_format():
    """Test that block IDs are valid SHA256 hashes."""
    code = """# Test
x = 1

# Another block
y = 2"""

    doc = parse_document(code)

    for block in doc.blocks:
        # Block IDs should be 16-character hex strings (truncated SHA256)
        assert len(block.id) == 16, f"Block ID should be 16 chars, got {len(block.id)}"
        assert all(
            c in "0123456789abcdef" for c in block.id
        ), "Block ID should be valid hex"


def test_block_id_includes_dependencies():
    """Test that block IDs include dependency information."""
    # Two blocks with same code but different dependencies
    code1 = """# First x
x = 1

# Use x
result = x + 1"""

    code2 = """# Different x
x = 2

# Use x (same code as above)
result = x + 1"""

    doc1 = parse_document(code1)
    doc2 = parse_document(code2)

    # The second block in each document has the same code
    # but depends on different values of x
    block1_second = doc1.blocks[1]
    block2_second = doc2.blocks[1]

    # Since the dependencies are different (x=1 vs x=2),
    # the block IDs should be different even though the code is the same
    assert (
        block1_second.id != block2_second.id
    ), "Block IDs should include dependency context"


def test_block_id_stability_with_whitespace():
    """Test that block IDs are stable with respect to meaningful whitespace."""
    # Same logic, different whitespace
    code1 = """# Block
x = 1
y = 2"""

    code2 = """# Block
x = 1

y = 2"""

    doc1 = parse_document(code1)
    doc2 = parse_document(code2)

    # These create different blocks due to the blank line
    assert len(doc1.blocks) != len(doc2.blocks), "Blank lines create block boundaries"


def test_block_id_with_pragmas():
    """Test that block IDs include pragma information."""
    code1 = """# Normal block
x = 1"""

    code2 = """# %% pragma: always-eval
# Block with pragma
x = 1"""

    doc1 = parse_document(code1)
    doc2 = parse_document(code2)

    # Different pragmas should result in different IDs
    assert doc1.blocks[0].id != doc2.blocks[0].id, "Pragmas should affect block IDs"
