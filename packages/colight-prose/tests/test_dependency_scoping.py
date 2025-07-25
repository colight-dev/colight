"""Tests for Python-correct scoping in dependency analysis."""

from colight_prose.dependency_analyzer import analyze_block


def test_list_comprehension_scoping():
    """Test that list comprehension variables don't leak out."""
    code = """# Simple comprehension
squares = [x**2 for x in range(10)]

# Nested comprehensions  
matrix = [[x*y for x in row] for y in cols]

# With condition
evens = [n for n in numbers if n % 2 == 0]"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"squares", "matrix", "evens"}
    # x, y, n should NOT be in requires - they're local to comprehensions
    assert "x" not in requires
    assert "y" not in requires
    assert "n" not in requires
    # These are the actual external dependencies
    assert requires == {"row", "cols", "numbers"}


def test_set_comprehension_scoping():
    """Test that set comprehension variables don't leak out."""
    code = """unique_squares = {x**2 for x in data}
filtered = {item for item in items if item > threshold}"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"unique_squares", "filtered"}
    assert "x" not in requires
    assert "item" not in requires
    assert requires == {"data", "items", "threshold"}


def test_dict_comprehension_scoping():
    """Test that dict comprehension variables don't leak out."""
    code = """squared_dict = {k: v**2 for k, v in pairs.items()}
name_map = {person.id: person.name for person in people}"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"squared_dict", "name_map"}
    assert "k" not in requires
    assert "v" not in requires
    assert "person" not in requires
    assert requires == {"pairs", "people"}


def test_generator_expression_scoping():
    """Test that generator expression variables don't leak out."""
    code = """gen = (x*2 for x in sequence)
filtered_gen = (item for item in data if predicate(item))"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"gen", "filtered_gen"}
    assert "x" not in requires
    assert "item" not in requires
    assert requires == {"sequence", "data", "predicate"}


def test_lambda_parameter_scoping():
    """Test that lambda parameters don't leak out."""
    code = """double = lambda x: x * 2
add = lambda a, b: a + b
transform = lambda item: item.value * scale
items.sort(key=lambda obj: obj.priority)"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"double", "add", "transform"}
    assert "x" not in requires
    assert "a" not in requires
    assert "b" not in requires
    assert "item" not in requires
    assert "obj" not in requires
    assert requires == {"scale", "items"}


def test_with_statement_provides():
    """Test that with statement targets are provided."""
    code = """with open('file.txt') as f:
    content = f.read()
    
# f should be available here
print(f)  # f is closed but still exists

with resource_manager() as (conn, cursor):
    cursor.execute(query)"""
    provides, requires, _ = analyze_block(code)
    # with targets should be in provides
    assert "f" in provides
    assert "conn" in provides
    assert "cursor" in provides
    assert "content" in provides
    assert requires == {"resource_manager", "query"}


def test_exception_handler_scoping():
    """Test that exception variables are scoped to their handler."""
    code = """try:
    risky_operation()
except ValueError as e:
    print(e)  # e is available here
    error_msg = str(e)
except (TypeError, KeyError) as err:
    handle_error(err)  # err is available here
    
# e and err should NOT be available here
# print(e) would fail
# print(err) would fail"""
    provides, requires, _ = analyze_block(code)
    # Exception variables should NOT be in provides
    assert "e" not in provides
    assert "err" not in provides
    # But other assignments in except blocks should be
    assert "error_msg" in provides
    assert requires == {"risky_operation", "handle_error"}


def test_star_import_marking():
    """Test that star imports are marked."""
    code = """from numpy import *
import sys
from os.path import *

# These might come from star imports
arr = array([1, 2, 3])
matrix = zeros((3, 3))"""
    provides, requires, _ = analyze_block(code)
    # Should have the star import marker
    assert "__star_import__" in provides
    # Regular imports should still work
    assert "sys" in provides
    # Variables defined in the block
    assert "arr" in provides
    assert "matrix" in provides
    # array and zeros might come from star import, so they're in requires
    assert "array" in requires
    assert "zeros" in requires


def test_complex_scoping_example():
    """Test a complex example with multiple scoping rules."""
    code = """import numpy as np
from collections import defaultdict

# List comprehension with nested scope
data = [process(x) for x in raw_data if x > 0]

# Lambda in a function call
results = list(map(lambda item: item * factor, data))

# With statement
with open('output.txt', 'w') as f:
    f.write(str(results))

# Exception handling
try:
    final = analyze(results, f)  # f is still in scope
except AnalysisError as e:
    print(f"Error: {e}")
    final = None

# Generator with multiple variables
pairs = ((i, val) for i, val in enumerate(data))"""
    provides, requires, _ = analyze_block(code)

    # Imports and assignments
    assert "np" in provides
    assert "defaultdict" in provides
    assert "data" in provides
    assert "results" in provides
    assert "f" in provides  # from with statement
    assert "final" in provides
    assert "pairs" in provides

    # Comprehension/lambda variables should NOT leak
    assert "x" not in requires
    assert "item" not in requires
    assert "i" not in requires
    assert "val" not in requires
    assert "e" not in provides  # exception variable

    # External dependencies
    assert "process" in requires
    assert "raw_data" in requires
    assert "factor" in requires
    assert "analyze" in requires
    assert "AnalysisError" in requires


def test_nested_comprehensions_with_same_variable():
    """Test that nested comprehensions with same variable names work correctly."""
    code = """# Both use 'x' but in different scopes
outer = [sum(x*x for x in row) for x in matrix]"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"outer"}
    assert "x" not in requires  # Neither x should leak
    assert requires == {"row", "matrix"}  # Unclear semantics, but this is what we get
