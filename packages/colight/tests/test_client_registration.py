"""Tests for client registration and watch/unwatch protocol."""

import json
import pytest
from unittest.mock import Mock, AsyncMock
from collections import defaultdict


class ClientRegistry:
    """Registry to track which clients are watching which files."""

    def __init__(self):
        self.clients = {}  # clientId -> WebSocket
        self.watched_files = defaultdict(set)  # file_path -> Set[clientId]
        self.client_files = {}  # clientId -> current_file

    def register_client(self, client_id: str, websocket):
        """Register a new client connection."""
        self.clients[client_id] = websocket

    def unregister_client(self, client_id: str):
        """Remove a client and all its watches."""
        if client_id in self.clients:
            del self.clients[client_id]

        # Remove from all watched files
        for file_path in list(self.watched_files.keys()):
            self.watched_files[file_path].discard(client_id)
            if not self.watched_files[file_path]:
                del self.watched_files[file_path]

        # Remove current file tracking
        if client_id in self.client_files:
            del self.client_files[client_id]

    def watch_file(self, client_id: str, file_path: str):
        """Register that a client is watching a file."""
        if client_id not in self.clients:
            return False

        # Unwatch previous file if any
        if client_id in self.client_files:
            old_file = self.client_files[client_id]
            self.watched_files[old_file].discard(client_id)
            if not self.watched_files[old_file]:
                del self.watched_files[old_file]

        # Watch new file
        self.watched_files[file_path].add(client_id)
        self.client_files[client_id] = file_path
        return True

    def unwatch_file(self, client_id: str, file_path: str):
        """Unregister that a client is watching a file."""
        if client_id in self.clients and file_path in self.watched_files:
            self.watched_files[file_path].discard(client_id)
            if not self.watched_files[file_path]:
                del self.watched_files[file_path]

            if self.client_files.get(client_id) == file_path:
                del self.client_files[client_id]
            return True
        return False

    def get_watchers(self, file_path: str) -> set:
        """Get all clients watching a file."""
        return self.watched_files.get(file_path, set())

    def get_watched_files(self) -> set:
        """Get all files being watched by any client."""
        return set(self.watched_files.keys())


class TestClientRegistry:
    """Test the client registry functionality."""

    def test_register_unregister_client(self):
        """Test basic client registration."""
        registry = ClientRegistry()
        ws = Mock()

        # Register
        registry.register_client("client1", ws)
        assert "client1" in registry.clients
        assert registry.clients["client1"] == ws

        # Unregister
        registry.unregister_client("client1")
        assert "client1" not in registry.clients

    def test_watch_unwatch_file(self):
        """Test file watching mechanics."""
        registry = ClientRegistry()
        ws = Mock()
        registry.register_client("client1", ws)

        # Watch a file
        assert registry.watch_file("client1", "file1.py")
        assert "client1" in registry.get_watchers("file1.py")
        assert registry.client_files["client1"] == "file1.py"

        # Watch a different file (should auto-unwatch previous)
        assert registry.watch_file("client1", "file2.py")
        assert "client1" not in registry.get_watchers("file1.py")
        assert "client1" in registry.get_watchers("file2.py")
        assert registry.client_files["client1"] == "file2.py"

        # Unwatch
        assert registry.unwatch_file("client1", "file2.py")
        assert "client1" not in registry.get_watchers("file2.py")
        assert "client1" not in registry.client_files

    def test_multiple_clients_same_file(self):
        """Test multiple clients watching the same file."""
        registry = ClientRegistry()
        ws1, ws2 = Mock(), Mock()

        registry.register_client("client1", ws1)
        registry.register_client("client2", ws2)

        registry.watch_file("client1", "file.py")
        registry.watch_file("client2", "file.py")

        watchers = registry.get_watchers("file.py")
        assert len(watchers) == 2
        assert "client1" in watchers
        assert "client2" in watchers

        # One client unwatches
        registry.unwatch_file("client1", "file.py")
        watchers = registry.get_watchers("file.py")
        assert len(watchers) == 1
        assert "client2" in watchers

    def test_unregister_cleans_watches(self):
        """Test that unregistering a client cleans up all watches."""
        registry = ClientRegistry()
        ws = Mock()

        registry.register_client("client1", ws)
        registry.watch_file("client1", "file1.py")
        registry.watch_file("client1", "file2.py")  # This auto-unwatches file1

        assert registry.get_watched_files() == {"file2.py"}

        registry.unregister_client("client1")
        assert registry.get_watched_files() == set()
        assert not registry.watched_files

    def test_watch_without_registration(self):
        """Test that watching fails without registration."""
        registry = ClientRegistry()

        assert not registry.watch_file("unknown", "file.py")
        assert "unknown" not in registry.get_watchers("file.py")


def test_watch_unwatch_message_format():
    """Test the expected message format for watch/unwatch."""
    # Watch message
    watch_msg = {
        "type": "watch-file",
        "path": "path/to/file.py",
        "clientId": "client-uuid-123",
    }
    assert watch_msg["type"] == "watch-file"
    assert "path" in watch_msg
    assert "clientId" in watch_msg

    # Unwatch message
    unwatch_msg = {
        "type": "unwatch-file",
        "path": "path/to/file.py",
        "clientId": "client-uuid-123",
    }
    assert unwatch_msg["type"] == "unwatch-file"
    assert "path" in unwatch_msg
    assert "clientId" in unwatch_msg


@pytest.mark.asyncio
async def test_message_handler_integration():
    """Test how the registry integrates with message handling."""
    registry = ClientRegistry()
    ws = AsyncMock()

    async def handle_message(websocket, message):
        """Simulate message handler."""
        data = json.loads(message) if isinstance(message, str) else message
        client_id = data.get("clientId")

        if not client_id:
            return {"error": "Missing clientId"}

        # Ensure client is registered
        if client_id not in registry.clients:
            registry.register_client(client_id, websocket)

        if data["type"] == "watch-file":
            success = registry.watch_file(client_id, data["path"])
            return {"type": "watch-ack", "success": success}
        elif data["type"] == "unwatch-file":
            success = registry.unwatch_file(client_id, data["path"])
            return {"type": "unwatch-ack", "success": success}

    # Test watch
    result = await handle_message(
        ws, {"type": "watch-file", "path": "test.py", "clientId": "client1"}
    )
    assert result is not None and result["success"]
    assert "client1" in registry.get_watchers("test.py")

    # Test unwatch
    result = await handle_message(
        ws, {"type": "unwatch-file", "path": "test.py", "clientId": "client1"}
    )
    assert result is not None and result["success"]
    assert "client1" not in registry.get_watchers("test.py")
