"""Test deterministic callback ID generation."""

import colight.plot as Plot


def test_callback_ids_are_deterministic():
    """Callback IDs should be stable across multiple serializations."""

    def my_callback(widget, event):
        print(f"Value: {event['value']}")

    # Create visual with callback
    visual = (
        Plot.State({"x": 0}, sync=True)
        | Plot.onChange({"x": my_callback})
        | Plot.Slider("x", 0, [0, 2])
    )

    # Serialize twice
    bytes1 = visual.to_bytes()
    bytes2 = visual.to_bytes()

    # Should produce identical output (deterministic)
    assert bytes1 == bytes2


def test_callback_ids_include_order():
    """Callback IDs should reflect order encountered during serialization."""

    def callback_a(widget, event):
        pass

    def callback_b(widget, event):
        pass

    visual = (
        Plot.State({"a": 0, "b": 0}, sync=True)
        | Plot.onChange({"a": callback_a, "b": callback_b})
        | Plot.Slider("a", 0, [0, 2])
        | Plot.Slider("b", 0, [0, 2])
    )

    # Get widget and check callback IDs
    widget = visual.widget()

    # IDs should start with cb-0, cb-1, etc.
    callback_ids = list(widget.callback_registry.keys())
    assert len(callback_ids) == 2
    assert any(id.startswith("cb-0-") for id in callback_ids)
    assert any(id.startswith("cb-1-") for id in callback_ids)


def test_same_lambda_different_positions():
    """Lambdas with same code but different positions get different IDs."""

    # Same lambda code in two positions
    visual = (
        Plot.State({"x": 0, "y": 0}, sync=True)
        | Plot.onChange(
            {
                "x": lambda w, e: print(e["value"]),
                "y": lambda w, e: print(e["value"]),
            }
        )
        | Plot.Slider("x", 0, [0, 2])
        | Plot.Slider("y", 0, [0, 2])
    )

    widget = visual.widget()

    # Should have 2 callbacks with different IDs (different positions)
    callback_ids = list(widget.callback_registry.keys())
    assert len(callback_ids) == 2
    assert callback_ids[0] != callback_ids[1]
    # Both should have predictable format
    assert all(id.startswith("cb-") for id in callback_ids)


def test_different_lambdas_different_hashes():
    """Different lambda code should produce different hash components."""

    visual1 = (
        Plot.State({"x": 0}, sync=True)
        | Plot.onChange({"x": lambda w, e: print("A")})
        | Plot.Slider("x", 0, [0, 2])
    )

    visual2 = (
        Plot.State({"x": 0}, sync=True)
        | Plot.onChange({"x": lambda w, e: print("B")})
        | Plot.Slider("x", 0, [0, 2])
    )

    widget1 = visual1.widget()
    widget2 = visual2.widget()

    id1 = list(widget1.callback_registry.keys())[0]
    id2 = list(widget2.callback_registry.keys())[0]

    # Different code should produce different IDs
    # (at least the hash component should differ)
    assert id1 != id2
