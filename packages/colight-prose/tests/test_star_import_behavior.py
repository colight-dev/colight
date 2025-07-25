"""Tests for star import behavior in dependency analysis."""

from colight_prose.block_graph import BlockGraph
from colight_prose.dependency_analyzer import analyze_block


def test_star_import_sentinel():
    """Test that star imports create the __star_import__ sentinel."""
    # Code with star import
    code = "from math import *"
    provides, requires, _ = analyze_block(code)

    # Should have the sentinel in provides
    assert "__star_import__" in provides
    assert len(requires) == 0


def test_star_import_with_usage():
    """Test star import with symbols that might come from it."""
    # Block 1: star import
    code1 = "from math import *"
    provides1, requires1, _ = analyze_block(code1)

    # Block 2: uses something that might come from the star import
    code2 = "result = sin(pi / 2)"
    provides2, requires2, _ = analyze_block(code2)

    # Current behavior: sin and pi are required but not provided
    assert "sin" in requires2
    assert "pi" in requires2

    # The issue: block_graph doesn't know that __star_import__ might provide these


def test_star_import_in_block_graph():
    """Test how block graph handles star imports."""
    graph = BlockGraph()

    blocks = [
        {
            "id": "1",
            "interface": {
                "provides": ["__star_import__"],  # from math import *
                "requires": [],
            },
        },
        {
            "id": "2",
            "interface": {
                "provides": ["result"],
                "requires": ["sin", "pi"],  # Uses math functions
            },
        },
    ]

    graph.add_blocks(blocks)

    # Current behavior: Block 2 has no dependencies because sin/pi aren't provided
    deps = graph.get_dependencies("2")
    assert len(deps) == 0  # This is the problem!

    # Block 2 will fail at runtime because sin/pi are undefined


def test_explicit_import_works():
    """Test that explicit imports work correctly."""
    graph = BlockGraph()

    blocks = [
        {
            "id": "1",
            "interface": {
                "provides": ["sin", "pi"],  # from math import sin, pi
                "requires": [],
            },
        },
        {"id": "2", "interface": {"provides": ["result"], "requires": ["sin", "pi"]}},
    ]

    graph.add_blocks(blocks)

    # This works correctly
    deps = graph.get_dependencies("2")
    assert "1" in deps


def test_star_import_fallback_proposal():
    """Test proposed fallback behavior for star imports."""
    # This test demonstrates what the fix should do

    # Proposed: when a symbol is required but not found,
    # check if any earlier block has __star_import__

    def find_provider_with_star_fallback(symbol, providers, star_providers):
        """Find provider for a symbol, considering star imports."""
        # First try exact match
        if symbol in providers:
            return providers[symbol]

        # If not found and there are star imports, return the last one
        if star_providers:
            return star_providers[-1]  # Last star import block

        return None

    # Simulate the scenario
    providers = {}  # symbol -> block_id
    star_providers = ["1"]  # Block 1 has star import

    # Block 2 requires 'sin'
    provider = find_provider_with_star_fallback("sin", providers, star_providers)
    assert provider == "1"  # Should fall back to star import block


def test_multiple_star_imports():
    """Test behavior with multiple star imports."""
    graph = BlockGraph()

    blocks = [
        {
            "id": "1",
            "interface": {
                "provides": ["__star_import__"],  # from os import *
                "requires": [],
            },
        },
        {
            "id": "2",
            "interface": {
                "provides": ["__star_import__", "x"],  # from sys import *; x = 1
                "requires": [],
            },
        },
        {
            "id": "3",
            "interface": {
                "provides": ["result"],
                "requires": ["path", "x"],  # path could be from os, x from block 2
            },
        },
    ]

    graph.add_blocks(blocks)

    # Current: only x is resolved (from block 2)
    deps = graph.get_dependencies("3")
    assert len(deps) == 1  # Only x is resolved
    assert "2" in deps  # x comes from block 2

    # With fix: should depend on blocks 1 and 2
    # (though we can't know which block provides 'path')


def test_star_import_with_override():
    """Test star import when symbol is later overridden."""
    graph = BlockGraph()

    blocks = [
        {
            "id": "1",
            "interface": {
                "provides": ["__star_import__"],  # from math import *
                "requires": [],
            },
        },
        {
            "id": "2",
            "interface": {
                "provides": ["pi"],  # pi = 3.14159
                "requires": [],
            },
        },
        {
            "id": "3",
            "interface": {
                "provides": ["result"],
                "requires": ["pi", "sin"],  # pi from block 2, sin maybe from block 1
            },
        },
    ]

    graph.add_blocks(blocks)

    deps = graph.get_dependencies("3")
    # Currently: only depends on block 2 (for pi)
    assert "2" in deps
    assert "1" not in deps  # sin dependency not resolved
