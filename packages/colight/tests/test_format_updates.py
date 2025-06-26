import os
import tempfile

import colight.plot as Plot
import numpy as np
from colight.format import parse_file, save_updates, append_update


def test_case_1_single_plot():
    """Test Case 1: A single plot saved to a .colight file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a plot
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        plot = Plot.raster(data)

        # Save the plot
        output_path = os.path.join(tmpdir, "single_plot.colight")
        plot.save_file(output_path)

        # Parse and verify
        initial_data, buffers, updates = parse_file(output_path)
        assert initial_data is not None  # Has initial state
        assert len(buffers) > 0  # Has buffer data
        assert len(updates) == 0  # No updates
        assert "ast" in initial_data
        assert "state" in initial_data


def test_case_2_plot_with_updates():
    """Test Case 2: A single plot with updates appended"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and save initial plot
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        plot = Plot.raster(data)
        output_path = os.path.join(tmpdir, "plot_with_updates.colight")
        plot.save_file(output_path)

        # Create updates (state changes, new visualizations)
        update1 = Plot.State({"zoom": 2.0, "theme": "dark"})
        update2 = Plot.raster(np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32))

        # Append updates to the existing file
        append_update(output_path, update1)
        append_update(output_path, update2)

        # Parse and verify
        initial_data, initial_buffers, updates_list = parse_file(output_path)
        assert initial_data is not None  # Still has initial state
        assert len(initial_buffers) > 0  # Initial plot has buffers
        assert len(updates_list) == 2  # Two updates appended

        # Verify first update (state only)
        update_0 = updates_list[0]
        assert isinstance(update_0, dict)  # Updates are dictionaries
        assert update_0.get("ast") is None  # State-only update
        assert update_0.get("state", {}).get("zoom") == 2.0
        assert update_0.get("state", {}).get("theme") == "dark"

        # Verify second update (new visualization)
        update_1 = updates_list[1]
        assert isinstance(update_1, dict)
        assert update_1.get("ast") is not None  # Has AST for new raster


def test_case_3_updates_only():
    """Test Case 3: Updates only (no initial plot)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create updates
        update1 = Plot.State({"mode": "interactive", "scale": 1.5})
        update2 = Plot.img(np.random.rand(50, 50, 3))  # Image visualization
        update3 = Plot.State({"selection": [0, 0, 10, 10]})

        # Save updates to a new file (no initial plot)
        output_path = os.path.join(tmpdir, "updates_only.colight")
        save_updates(output_path, [update1, update2, update3])

        # Parse and verify
        initial_data, initial_buffers, updates_list = parse_file(output_path)
        assert initial_data is None  # No initial plot
        assert len(initial_buffers) == 0  # No initial buffers
        assert len(updates_list) == 3  # Three updates

        # Verify updates
        for i, update in enumerate(updates_list):
            assert isinstance(update, dict)

            if i == 0:  # First state update
                assert update.get("ast") is None
                assert update.get("state", {}).get("mode") == "interactive"
                assert update.get("state", {}).get("scale") == 1.5
            elif i == 1:  # Image update
                assert update.get("ast") is not None
                # Buffer references are stored in the state, not at top level
            elif i == 2:  # Second state update
                assert update.get("ast") is None
                assert update.get("state", {}).get("selection") == [0, 0, 10, 10]


def test_append_multiple_updates_at_once():
    """Test appending multiple updates in one call"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create initial plot
        plot = Plot.raster(np.eye(3))
        output_path = os.path.join(tmpdir, "multi_append.colight")
        plot.save_file(output_path)

        # Create multiple updates
        updates = [
            Plot.State({"iteration": 1}),
            Plot.State({"iteration": 2}),
            Plot.State({"iteration": 3}),
        ]

        # Append all at once using append_updates (note: this is different from save_updates)
        from colight.format import append_updates

        append_updates(output_path, updates)

        # Verify
        _, _, updates_list = parse_file(output_path)
        assert len(updates_list) == 3
        for i, update_item in enumerate(updates_list):
            assert isinstance(update_item, dict)
            assert update_item.get("state", {}).get("iteration") == i + 1


if __name__ == "__main__":
    test_case_1_single_plot()
    print("✓ Case 1: Single plot")

    test_case_2_plot_with_updates()
    print("✓ Case 2: Plot with updates")

    test_case_3_updates_only()
    print("✓ Case 3: Updates only")

    test_append_multiple_updates_at_once()
    print("✓ Multiple updates append")

    print("\nAll tests passed!")
