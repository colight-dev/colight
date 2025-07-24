"""Correctness tests for block graph edge building."""

from colight_prose.block_graph import BlockGraph


def test_linear_edge_building_correctness():
    """Test that optimized edge building produces correct results."""
    graph = BlockGraph()

    blocks = [
        {"id": "1", "interface": {"provides": ["a", "b"], "requires": []}},
        {"id": "2", "interface": {"provides": ["c"], "requires": ["a"]}},
        {"id": "3", "interface": {"provides": ["d"], "requires": ["b", "c"]}},
        {"id": "4", "interface": {"provides": ["e"], "requires": ["d"]}},
        {
            "id": "5",
            "interface": {"provides": ["a"], "requires": ["e"]},
        },  # Redefines 'a'
        {
            "id": "6",
            "interface": {"provides": ["f"], "requires": ["a"]},
        },  # Should use block 5's 'a'
    ]

    graph.add_blocks(blocks)

    # Verify edges are correct
    assert graph.edges["1"] == {"2", "3"}  # Block 1 provides a,b used by 2,3
    assert graph.edges["2"] == {"3"}  # Block 2 provides c used by 3
    assert graph.edges["3"] == {"4"}  # Block 3 provides d used by 4
    assert graph.edges["4"] == {"5"}  # Block 4 provides e used by 5
    assert graph.edges["5"] == {"6"}  # Block 5 provides new 'a' used by 6
    assert (
        "6" not in graph.edges or len(graph.edges["6"]) == 0
    )  # Block 6 provides f (not used)

    # Verify reverse edges
    assert len(graph.reverse_edges.get("1", set())) == 0  # No dependencies
    assert graph.reverse_edges["2"] == {"1"}  # Depends on block 1
    assert graph.reverse_edges["3"] == {"1", "2"}  # Depends on blocks 1,2
    assert graph.reverse_edges["4"] == {"3"}  # Depends on block 3
    assert graph.reverse_edges["5"] == {"4"}  # Depends on block 4
    assert graph.reverse_edges["6"] == {"5"}  # Depends on block 5 (not 1!)

    # Verify symbol providers tracking
    assert graph.symbol_providers["a"] == "5"  # Last provider wins
    assert graph.symbol_providers["b"] == "1"
    assert graph.symbol_providers["c"] == "2"
    assert graph.symbol_providers["d"] == "3"
    assert graph.symbol_providers["e"] == "4"
    assert graph.symbol_providers["f"] == "6"


def test_edge_building_with_missing_dependencies():
    """Test edge building when dependencies are missing."""
    graph = BlockGraph()

    blocks = [
        {"id": "1", "interface": {"provides": ["x"], "requires": ["missing1"]}},
        {"id": "2", "interface": {"provides": ["y"], "requires": ["x", "missing2"]}},
        {"id": "3", "interface": {"provides": ["z"], "requires": ["y"]}},
    ]

    graph.add_blocks(blocks)

    # Should only create edges for found dependencies
    assert graph.edges["1"] == {"2"}  # x is used by block 2
    assert graph.edges["2"] == {"3"}  # y is used by block 3
    assert "3" not in graph.edges or len(graph.edges["3"]) == 0

    # Reverse edges should only show actual dependencies
    assert len(graph.reverse_edges.get("1", set())) == 0  # missing1 not found
    assert graph.reverse_edges["2"] == {"1"}  # Only x found, not missing2
    assert graph.reverse_edges["3"] == {"2"}


def test_edge_building_order_preservation():
    """Test that edge building preserves document order for stability."""
    graph = BlockGraph()

    # Create blocks that could have different orderings
    blocks = [
        {"id": "1", "interface": {"provides": ["a"], "requires": []}},
        {"id": "2", "interface": {"provides": ["b"], "requires": []}},
        {"id": "3", "interface": {"provides": ["c"], "requires": ["a", "b"]}},
        {"id": "4", "interface": {"provides": ["d"], "requires": ["c"]}},
        {"id": "5", "interface": {"provides": ["e"], "requires": ["d"]}},
    ]

    graph.add_blocks(blocks)

    # Check execution order handles numeric IDs correctly
    order = graph.execution_order()
    assert order.index("1") < order.index("3")  # 1 provides a needed by 3
    assert order.index("2") < order.index("3")  # 2 provides b needed by 3
    assert order.index("3") < order.index("4")  # 3 provides c needed by 4
    assert order.index("4") < order.index("5")  # 4 provides d needed by 5


def test_symbol_provider_updates():
    """Test that symbol providers are tracked correctly with overwrites."""
    graph = BlockGraph()

    blocks = [
        {"id": "1", "interface": {"provides": ["x", "y"], "requires": []}},
        {"id": "2", "interface": {"provides": ["x"], "requires": []}},  # Overwrites x
        {
            "id": "3",
            "interface": {"provides": ["z"], "requires": ["x"]},
        },  # Should use block 2's x
        {"id": "4", "interface": {"provides": ["y"], "requires": []}},  # Overwrites y
        {
            "id": "5",
            "interface": {"provides": ["w"], "requires": ["y"]},
        },  # Should use block 4's y
    ]

    graph.add_blocks(blocks)

    # Verify final symbol providers
    assert graph.symbol_providers["x"] == "2"
    assert graph.symbol_providers["y"] == "4"
    assert graph.symbol_providers["z"] == "3"
    assert graph.symbol_providers["w"] == "5"

    # Verify dependencies use the correct providers
    assert (
        "2" in graph.edges and "3" in graph.edges["2"]
    )  # Block 3 depends on block 2's x
    assert (
        "1" not in graph.edges or "3" not in graph.edges["1"]
    )  # Block 3 does NOT depend on block 1's x
    assert (
        "4" in graph.edges and "5" in graph.edges["4"]
    )  # Block 5 depends on block 4's y
    assert (
        "1" not in graph.edges or "5" not in graph.edges["1"]
    )  # Block 5 does NOT depend on block 1's y
