"""Tests for block dependency graph."""

from colight_prose.block_graph import BlockGraph


def test_simple_dependency_chain():
    """Test a simple dependency chain."""
    graph = BlockGraph()

    blocks = [
        {"id": "1", "interface": {"provides": ["x"], "requires": []}},
        {"id": "2", "interface": {"provides": ["y"], "requires": ["x"]}},
        {"id": "3", "interface": {"provides": ["z"], "requires": ["y"]}},
    ]

    graph.add_blocks(blocks)

    # Check execution order
    order = graph.execution_order()
    assert order == ["1", "2", "3"]

    # Check dirty propagation
    dirty = graph.dirty_after({"1"})
    assert set(dirty) == {"1", "2", "3"}

    dirty = graph.dirty_after({"2"})
    assert set(dirty) == {"2", "3"}

    dirty = graph.dirty_after({"3"})
    assert set(dirty) == {"3"}


def test_independent_blocks():
    """Test blocks with no dependencies."""
    graph = BlockGraph()

    blocks = [
        {"id": "1", "interface": {"provides": ["a"], "requires": []}},
        {"id": "2", "interface": {"provides": ["b"], "requires": []}},
        {"id": "3", "interface": {"provides": ["c"], "requires": []}},
    ]

    graph.add_blocks(blocks)

    # Order should preserve input order when no dependencies
    order = graph.execution_order()
    assert order == ["1", "2", "3"]

    # Changing one block shouldn't affect others
    dirty = graph.dirty_after({"2"})
    assert dirty == ["2"]


def test_multiple_dependencies():
    """Test a block that depends on multiple others."""
    graph = BlockGraph()

    blocks = [
        {"id": "1", "interface": {"provides": ["x"], "requires": []}},
        {"id": "2", "interface": {"provides": ["y"], "requires": []}},
        {"id": "3", "interface": {"provides": ["z"], "requires": ["x", "y"]}},
    ]

    graph.add_blocks(blocks)

    order = graph.execution_order()
    # Block 3 must come after both 1 and 2
    assert order.index("3") > order.index("1")
    assert order.index("3") > order.index("2")

    # Changing either dependency should dirty block 3
    assert "3" in graph.dirty_after({"1"})
    assert "3" in graph.dirty_after({"2"})


def test_forward_reference():
    """Test that forward references (to blocks below) are not allowed."""
    graph = BlockGraph()

    blocks = [
        {
            "id": "1",
            "interface": {"provides": ["x"], "requires": ["y"]},
        },  # y not available yet
        {
            "id": "2",
            "interface": {"provides": ["y"], "requires": ["x"]},
        },  # x is available
    ]

    graph.add_blocks(blocks)

    # Block 1 can't see y from block 2 (no forward references)
    # So block 1 has no dependencies
    assert len(graph.reverse_edges.get("1", set())) == 0

    # Block 2 can see x from block 1
    assert "1" in graph.reverse_edges.get("2", set())

    # No cycles because forward references aren't allowed
    cycles = graph.find_circular_dependencies()
    assert len(cycles) == 0

    # Execution order should be sequential
    order = graph.execution_order()
    assert order == ["1", "2"]


def test_missing_dependency():
    """Test handling of missing dependencies."""
    graph = BlockGraph()

    blocks = [
        {"id": "1", "interface": {"provides": ["x"], "requires": ["missing"]}},
        {"id": "2", "interface": {"provides": ["y"], "requires": ["x"]}},
    ]

    graph.add_blocks(blocks)

    # Should still work, just treating 'missing' as external
    order = graph.execution_order()
    assert order == ["1", "2"]


def test_multiple_providers():
    """Test when multiple blocks provide the same symbol."""
    graph = BlockGraph()

    blocks = [
        {"id": "1", "interface": {"provides": ["x"], "requires": []}},
        {
            "id": "2",
            "interface": {"provides": ["x"], "requires": []},
        },  # Also provides x
        {"id": "3", "interface": {"provides": ["y"], "requires": ["x"]}},
    ]

    graph.add_blocks(blocks)

    # Block 3 should depend on the last provider of x (block 2)
    order = graph.execution_order()
    assert order.index("3") > order.index("2")

    # Changing block 2 should dirty block 3
    assert "3" in graph.dirty_after({"2"})

    # But changing block 1 shouldn't (since block 2 overwrites x)
    assert "3" not in graph.dirty_after({"1"})


def test_complex_graph():
    """Test a more complex dependency graph."""
    graph = BlockGraph()

    blocks = [
        {"id": "1", "interface": {"provides": ["a"], "requires": []}},
        {"id": "2", "interface": {"provides": ["b"], "requires": ["a"]}},
        {"id": "3", "interface": {"provides": ["c"], "requires": ["a"]}},
        {"id": "4", "interface": {"provides": ["d"], "requires": ["b", "c"]}},
        {"id": "5", "interface": {"provides": ["e"], "requires": ["d"]}},
    ]

    graph.add_blocks(blocks)

    order = graph.execution_order()

    # Check partial ordering constraints
    assert order.index("1") < order.index("2")
    assert order.index("1") < order.index("3")
    assert order.index("2") < order.index("4")
    assert order.index("3") < order.index("4")
    assert order.index("4") < order.index("5")

    # Changing block 1 should dirty all others
    dirty = set(graph.dirty_after({"1"}))
    assert dirty == {"1", "2", "3", "4", "5"}
