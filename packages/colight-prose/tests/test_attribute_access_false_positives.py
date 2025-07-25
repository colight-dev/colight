"""Tests for attribute access false positives in dependency analysis."""

from colight_prose.dependency_analyzer import analyze_block


def test_attribute_access_basic():
    """Test basic attribute access detection."""
    code = "result = math.sqrt(16)"
    provides, requires, _ = analyze_block(code)

    # Should require 'math', not 'sqrt'
    assert "math" in requires
    assert "sqrt" not in requires
    assert "result" in provides


def test_attribute_access_false_positive():
    """Test the false positive case where foo is both provided and required."""
    # This is the issue mentioned in the report
    code = """
foo = SomeNamespace()
result = foo.bar
"""
    provides, requires, _ = analyze_block(code)

    # Expected behavior: foo is provided but NOT required
    assert "foo" in provides  # Correct
    assert "foo" not in requires  # No false positive - already fixed!
    assert "SomeNamespace" in requires  # Correct
    assert "result" in provides  # Correct

    # The issue mentioned in the report appears to be already fixed


def test_nested_attribute_access():
    """Test nested attribute access."""
    code = "value = obj.attr1.attr2.method()"
    provides, requires, _ = analyze_block(code)

    # Should only require the base object
    assert "obj" in requires
    assert "attr1" not in requires
    assert "attr2" not in requires
    assert "method" not in requires
    assert "value" in provides


def test_attribute_assignment():
    """Test attribute assignment."""
    code = "obj.attr = 42"
    provides, requires, _ = analyze_block(code)

    # Current behavior: doesn't track attribute assignments as requires
    # This could be considered correct (we're not reading obj, just modifying it)
    assert "obj" not in requires  # Not tracked as a requirement
    assert "attr" not in requires
    # Should not provide anything
    assert len(provides) == 0


def test_method_call_on_literal():
    """Test method calls on literals."""
    code = 'result = "hello".upper()'
    provides, requires, _ = analyze_block(code)

    # Should not require anything (literal has the method)
    assert len(requires) == 0
    assert "result" in provides


def test_chained_attribute_access():
    """Test chained attribute access with assignment."""
    code = """
config = get_config()
debug_mode = config.debug.enabled
"""
    provides, requires, _ = analyze_block(code)

    assert "get_config" in requires
    assert "config" in provides
    assert "debug_mode" in provides
    # config should not be in requires (it's defined locally)
    assert "config" not in requires


def test_attribute_in_different_contexts():
    """Test attribute access in various contexts."""
    code = """
# Define object
obj = create_object()

# Use in different ways
print(obj.name)
obj.method()
x = obj.value + 10
if obj.flag:
    pass
"""
    provides, requires, _ = analyze_block(code)

    assert "create_object" in requires
    assert "print" not in requires  # builtins are excluded
    assert "obj" in provides
    assert "x" in provides

    # obj should not be required - and it's not! No false positive
    assert "obj" not in requires


def test_self_attribute_access():
    """Test self attribute access in methods."""
    code = """
def method(self):
    return self.attribute
"""
    provides, requires, _ = analyze_block(code)

    # Should provide the method
    assert "method" in provides
    # Should not require 'self' or 'attribute'
    assert "self" not in requires
    assert "attribute" not in requires


def test_imported_module_attribute():
    """Test attribute access on imported modules."""
    code = """
import os
path = os.path.join("a", "b")
"""
    provides, requires, _ = analyze_block(code)

    assert "os" in provides  # Import provides os
    assert "path" in provides
    # os should not be in requires
    assert "os" not in requires


def test_fix_proposal():
    """Test proposed fix for attribute access false positives."""
    # The fix would check if the base name is in the current block's provides
    # before adding it to requires

    def is_false_positive_attribute_require(name, current_provides):
        """Check if requiring 'name' would be a false positive."""
        return name in current_provides

    # Test the logic
    current_provides = {"foo", "bar"}

    # foo.attr where foo is provided locally
    assert is_false_positive_attribute_require("foo", current_provides)

    # baz.attr where baz is not provided locally
    assert not is_false_positive_attribute_require("baz", current_provides)
