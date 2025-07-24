"""Tests for dependency analysis."""

from colight_prose.dependency_analyzer import analyze_block


def test_simple_provides():
    """Test detection of provided symbols."""
    code = """
x = 1
y = 2
def foo():
    return x + y
    
class Bar:
    pass
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"x", "y", "foo", "Bar"}
    assert requires == set()


def test_simple_requires():
    """Test detection of required symbols."""
    code = """
result = foo(x, y)
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"result"}
    assert requires == {"foo", "x", "y"}


def test_imports():
    """Test handling of imports."""
    code = """
import numpy as np
from math import sqrt
from collections import defaultdict as dd

array = np.array([1, 2, 3])
root = sqrt(16)
d = dd(list)
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"np", "sqrt", "dd", "array", "root", "d"}
    # array, list are builtins, so not in requires
    assert requires == set()


def test_nested_scopes():
    """Test that function parameters don't leak out."""
    code = """
def process(data):
    return data * 2

result = process(input_data)
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"process", "result"}
    assert requires == {"input_data"}
    # 'data' should not be in requires (it's a parameter)


def test_augmented_assignment():
    """Test augmented assignments require the variable to exist."""
    code = """
x += 1
y = y + 1
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"y"}  # y is assigned
    assert requires == {
        "x"
    }  # only x is required from outside; y = y + 1 is a block-internal issue


def test_tuple_unpacking():
    """Test tuple unpacking in assignments."""
    code = """
a, b, c = get_values()
x, *rest = some_list
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"a", "b", "c", "x", "rest"}
    assert requires == {"get_values", "some_list"}


def test_attribute_access():
    """Test that attribute access tracks base names."""
    code = """
result = np.array([1, 2, 3])
df_sorted = df.sort_values()
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"result", "df_sorted"}
    assert requires == {"np", "df"}


def test_builtins_excluded():
    """Test that builtins are not included in requires."""
    code = """
items = list(range(10))
text = str(42)
size = len(items)
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"items", "text", "size"}
    assert requires == set()  # list, range, str, len are all builtins


def test_class_methods():
    """Test class definitions with methods."""
    code = """
class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, x, y):
        return x * y

calc = Calculator()
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"Calculator", "calc"}
    assert requires == set()


def test_forward_references():
    """Test that forward references within a block are handled."""
    code = """
def caller():
    return callee()

def callee():
    return 42

result = caller()
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"caller", "callee", "result"}
    assert requires == set()  # callee is provided in same block


def test_parse_error_handling():
    """Test that parse errors don't crash analysis."""
    code = """
this is not valid python code
"""
    provides, requires, _ = analyze_block(code)
    assert provides == set()
    assert requires == set()


def test_import_star():
    """Test that import * doesn't crash."""
    code = """
from module import *
import another_module
"""
    provides, requires, _ = analyze_block(code)
    # We can't know what * imports, but regular imports work
    assert "another_module" in provides


def test_expression_vs_statement():
    """Test that both expressions and statements are analyzed."""
    code = """
# Statement
x = compute_value()

# Expression (would show visual)
x + 10
"""
    provides, requires, _ = analyze_block(code)
    assert provides == {"x"}
    assert requires == {"compute_value"}  # x is already provided
