"""Test that error line numbers are improved from generic <string>:1."""

import re

import libcst as cst

from colight_prose.executor import BlockExecutor
from colight_prose.model import Block, Element, TagSet
from colight_prose.parser import parse_document


def test_line_numbers_not_string_1():
    """Test that we don't get unhelpful <string>:1 errors."""
    # Create a block that starts at line 20
    stmt = cst.parse_statement("undefined_variable")
    elem = Element(kind="EXPRESSION", content=stmt, lineno=20)
    block = Block(elements=[elem], tags=TagSet(), start_line=20, id="0")

    executor = BlockExecutor()

    # Without filename - should still show line offset
    result = executor.execute_block(block, "<string>")
    assert result.error is not None
    # Should NOT show line 1 (unless it's the executor line number)
    # We care that the actual code error shows line 20
    assert 'File "<string>", line 1,' not in result.error
    # Should show line 20
    assert re.search(r"line 20", result.error)

    # With filename
    result2 = executor.execute_block(block, "myfile.py")
    assert result2.error is not None
    assert 'File "myfile.py", line 20' in result2.error


def test_multi_line_error_preservation():
    """Test that multi-line constructs preserve line numbers."""
    # Make it a single block by avoiding blank lines
    source = """def process_data(x):
    # Line 2
    result = x * 2
    # Line 4 - error here
    return result / 0
# Line 6 - call the function
value = process_data(10)
"""

    doc = parse_document(source)
    executor = BlockExecutor()

    result = executor.execute_block(doc.blocks[0], "notebook.py")
    assert result.error is not None

    # Should show the traceback with actual line numbers
    # Not just <string>:1 everywhere
    assert "<string>:1" not in result.error
    assert "division by zero" in result.error


def test_real_world_example():
    """Test a realistic example with imports and multiple blocks."""
    source = """# Analysis notebook
import numpy as np

# Generate some data  
data = np.random.randn(100)

# This will cause an error
result = data / np.zeros(100)
"""

    doc = parse_document(source)
    assert len(doc.blocks) >= 1

    executor = BlockExecutor()

    # Execute first block
    result = executor.execute_block(doc.blocks[0], "analysis.py")

    # The error should reference the actual file
    if result.error:
        assert 'File "analysis.py"' in result.error
        # Should not have generic <string>:1
        assert "<string>:1" not in result.error


def test_comparison_with_without_line_tracking():
    """Show the improvement from basic compilation."""
    code = "x = 1 / 0"

    # Old way - compile without line offset
    old_traceback = ""
    try:
        exec(compile(code, "test.py", "exec"))
    except Exception:
        import traceback

        old_traceback = traceback.format_exc()

    # New way - with our block that has line tracking
    stmt = cst.parse_statement(code)
    elem = Element(kind="STATEMENT", content=stmt, lineno=42)
    block = Block(elements=[elem], tags=TagSet(), start_line=42, id="0")

    executor = BlockExecutor()
    result = executor.execute_block(block, "test.py")

    print("Old traceback (always line 1):")
    print(old_traceback)
    print("\nNew traceback (correct line 42):")
    print(result.error)

    # Old shows line 1, new shows line 42
    assert "line 1" in old_traceback
    assert result.error is not None
    assert "line 42" in result.error


if __name__ == "__main__":
    test_line_numbers_not_string_1()
    test_multi_line_error_preservation()
    test_real_world_example()
    test_comparison_with_without_line_tracking()
    print("All tests passed!")
