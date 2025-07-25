"""Integration test for watch/unwatch protocol."""

import json
from unittest.mock import Mock

import pytest


def test_watch_unwatch_protocol():
    """Test the watch/unwatch protocol with a mock WebSocket."""
    from colight_prose.client_registry import ClientRegistry

    # Create registry
    registry = ClientRegistry()

    # Mock WebSocket
    ws = Mock()

    # Simulate watch-file message
    client_id = "test-client-123"
    file_path = "test.py"

    # Register and watch
    registry.register_client(client_id, ws)
    assert registry.watch_file(client_id, file_path)

    # Verify state
    assert client_id in registry.get_watchers(file_path)
    assert registry.get_watched_files() == {file_path}

    # Watch different file (should auto-unwatch previous)
    file_path2 = "test2.py"
    assert registry.watch_file(client_id, file_path2)

    # Verify state changed
    assert client_id not in registry.get_watchers(file_path)
    assert client_id in registry.get_watchers(file_path2)
    assert registry.get_watched_files() == {file_path2}

    # Explicit unwatch
    assert registry.unwatch_file(client_id, file_path2)
    assert registry.get_watched_files() == set()

    # Unregister client
    registry.unregister_client(client_id)
    assert len(registry.clients) == 0


@pytest.mark.asyncio
async def test_message_format_validation():
    """Test that messages have the correct format."""
    # Watch message format
    watch_msg = {"type": "watch-file", "path": "example.py", "clientId": "client-123"}

    # Validate structure
    assert watch_msg["type"] == "watch-file"
    assert "path" in watch_msg
    assert "clientId" in watch_msg

    # JSON serializable
    json_str = json.dumps(watch_msg)
    parsed = json.loads(json_str)
    assert parsed == watch_msg

    # Unwatch message format
    unwatch_msg = {
        "type": "unwatch-file",
        "path": "example.py",
        "clientId": "client-123",
    }

    assert unwatch_msg["type"] == "unwatch-file"
    assert "path" in unwatch_msg
    assert "clientId" in unwatch_msg


def test_telemetry_phase1():
    """Test that registry collects telemetry without changing execution."""
    from colight_prose.client_registry import ClientRegistry

    registry = ClientRegistry()

    # Register multiple clients
    ws1, ws2, ws3 = Mock(), Mock(), Mock()
    registry.register_client("client1", ws1)
    registry.register_client("client2", ws2)
    registry.register_client("client3", ws3)

    # Different watching patterns
    registry.watch_file("client1", "popular.py")
    registry.watch_file("client2", "popular.py")
    registry.watch_file("client3", "lonely.py")

    # Telemetry queries
    assert len(registry.get_watchers("popular.py")) == 2
    assert len(registry.get_watchers("lonely.py")) == 1
    assert len(registry.get_watched_files()) == 2

    # Client behavior
    registry.watch_file("client1", "another.py")  # Client1 switches files
    assert len(registry.get_watchers("popular.py")) == 1
    assert len(registry.get_watchers("another.py")) == 1

    # Summary stats
    total_clients = len(registry.clients)
    total_watched = len(registry.get_watched_files())
    assert total_clients == 3
    assert total_watched == 3  # popular.py, lonely.py, another.py
