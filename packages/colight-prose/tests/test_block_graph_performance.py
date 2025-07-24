"""Performance tests for block dependency graph."""

import time

from colight_prose.block_graph import BlockGraph


def test_edge_building_performance():
    """Test that edge building is O(n) not O(n²)."""
    # Test with increasing block counts to verify linear time
    block_counts = [100, 200, 400, 800]
    times = []

    for count in block_counts:
        # Create blocks where each depends on the previous one
        blocks = []
        for i in range(count):
            if i == 0:
                blocks.append(
                    {
                        "id": str(i),
                        "interface": {"provides": [f"var_{i}"], "requires": []},
                    }
                )
            else:
                blocks.append(
                    {
                        "id": str(i),
                        "interface": {
                            "provides": [f"var_{i}"],
                            "requires": [f"var_{i-1}"],
                        },
                    }
                )

        graph = BlockGraph()
        start_time = time.time()
        graph.add_blocks(blocks)
        elapsed = time.time() - start_time
        times.append(elapsed)

    # Check that time doesn't grow quadratically
    # If O(n²), doubling n should ~4x the time
    # If O(n), doubling n should ~2x the time

    # Compare ratios of consecutive measurements
    ratios = []
    for i in range(1, len(times)):
        ratio = times[i] / times[i - 1]
        ratios.append(ratio)
        print(
            f"Blocks: {block_counts[i-1]} -> {block_counts[i]}, "
            f"Time ratio: {ratio:.2f} (expect ~2.0 for O(n), ~4.0 for O(n²))"
        )

    # Currently this will fail because the implementation is O(n²)
    # After fix, all ratios should be close to 2.0
    avg_ratio = sum(ratios) / len(ratios)
    assert (
        avg_ratio < 3.0
    ), f"Average time ratio {avg_ratio:.2f} suggests O(n²) complexity"


def test_many_providers_performance():
    """Test performance when many blocks provide the same symbols."""
    # Create a scenario where many blocks provide shared symbols
    block_count = 500
    blocks = []

    # First half provides various symbols
    for i in range(block_count // 2):
        blocks.append(
            {
                "id": str(i),
                "interface": {
                    "provides": [f"shared_{i % 10}", f"unique_{i}"],
                    "requires": [],
                },
            }
        )

    # Second half requires those symbols
    for i in range(block_count // 2, block_count):
        blocks.append(
            {
                "id": str(i),
                "interface": {
                    "provides": [f"result_{i}"],
                    "requires": [
                        f"shared_{i % 10}",
                        f"unique_{i % (block_count // 2)}",
                    ],
                },
            }
        )

    graph = BlockGraph()
    start_time = time.time()
    graph.add_blocks(blocks)
    elapsed = time.time() - start_time

    # Should complete quickly even with many symbol lookups
    assert (
        elapsed < 1.0
    ), f"Building graph with {block_count} blocks took {elapsed:.2f}s (too slow)"


def test_wide_dependency_performance():
    """Test performance when blocks have many dependencies."""
    # Create blocks where later blocks depend on many earlier ones
    block_count = 200
    blocks = []

    # First 10 blocks provide base symbols
    for i in range(10):
        blocks.append(
            {"id": str(i), "interface": {"provides": [f"base_{i}"], "requires": []}}
        )

    # Remaining blocks depend on all base symbols
    for i in range(10, block_count):
        blocks.append(
            {
                "id": str(i),
                "interface": {
                    "provides": [f"derived_{i}"],
                    "requires": [f"base_{j}" for j in range(10)],
                },
            }
        )

    graph = BlockGraph()
    start_time = time.time()
    graph.add_blocks(blocks)
    elapsed = time.time() - start_time

    # Should handle wide dependencies efficiently
    assert (
        elapsed < 0.5
    ), f"Building graph with wide dependencies took {elapsed:.2f}s (too slow)"
