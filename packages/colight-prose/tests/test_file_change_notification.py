"""Tests for file change notification feature."""

import asyncio
import json
import pathlib
from unittest.mock import AsyncMock, Mock, patch

import pytest
from colight_prose.server import LiveServer


class TestFileChangeNotification:
    """Test the file change notification system."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project with files."""
        # Create main file
        main_file = tmp_path / "main.py"
        main_file.write_text("""
# Main application
print("Hello world")
""")

        # Create utils file
        utils_file = tmp_path / "utils.py"
        utils_file.write_text("""
def helper():
    return 42
""")

        return tmp_path

    @pytest.mark.asyncio
    async def test_single_file_change_notification(self, temp_project):
        """Test that single file changes trigger notifications."""
        server = LiveServer(
            input_path=temp_project, include=["*.py"], open_url=False, verbose=True
        )

        # Mock WebSocket connections
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        server.connections = {ws1, ws2}

        # Simulate file change
        server._changed_files_buffer.add("main.py")

        # Call the notification method directly
        await server._send_file_change_notification()

        # Verify both connections received the notification
        expected_message = json.dumps(
            {
                "type": "file-changed",
                "path": "main.py",
                "watched": False,  # No clients watching in this test
            }
        )

        ws1.send.assert_called_once_with(expected_message)
        ws2.send.assert_called_once_with(expected_message)

    @pytest.mark.asyncio
    async def test_multiple_file_changes_no_notification(self, temp_project):
        """Test that multiple file changes don't trigger notifications."""
        server = LiveServer(
            input_path=temp_project, include=["*.py"], open_url=False, verbose=False
        )

        # Mock WebSocket connections
        ws1 = AsyncMock()
        server.connections = {ws1}

        # Simulate multiple file changes
        server._changed_files_buffer.add("main.py")
        server._changed_files_buffer.add("utils.py")

        # Call the notification method
        await server._send_file_change_notification()

        # Verify no notification was sent (multiple files changed)
        ws1.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_debounce_behavior(self, temp_project):
        """Test that notifications are debounced properly."""
        server = LiveServer(
            input_path=temp_project, include=["*.py"], open_url=False, verbose=False
        )
        server._notification_delay = 0.1  # Shorter delay for testing

        # Mock WebSocket
        ws = AsyncMock()
        server.connections = {ws}

        # Simulate rapid file changes
        server._changed_files_buffer.add("main.py")
        task1 = asyncio.create_task(server._send_file_change_notification())

        # Add another file before debounce completes
        await asyncio.sleep(0.05)  # Half the debounce time
        server._changed_files_buffer.add("utils.py")

        # Wait for original task to complete
        await task1

        # No notification should be sent (multiple files in buffer)
        ws.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_watched_file_status(self, temp_project):
        """Test that watched status is correctly reported."""
        server = LiveServer(
            input_path=temp_project, include=["*.py"], open_url=False, verbose=False
        )

        # Mock WebSocket
        ws = AsyncMock()
        server.connections = {ws}

        # Register a client watching main.py
        server.client_registry.register_client("client1", Mock())
        server.client_registry.watch_file("client1", "main.py")

        # Simulate file change
        server._changed_files_buffer.add("main.py")
        await server._send_file_change_notification()

        # Verify watched status is True
        expected_message = json.dumps(
            {"type": "file-changed", "path": "main.py", "watched": True}
        )
        ws.send.assert_called_once_with(expected_message)

    @pytest.mark.asyncio
    async def test_relative_path_handling(self, temp_project):
        """Test that relative paths are handled correctly."""
        # Create server with relative path (.)
        server = LiveServer(
            input_path=pathlib.Path("."), include=["*.py"], open_url=False, verbose=True
        )

        # Verify input_path was resolved to absolute
        assert server.input_path.is_absolute()
        assert server.dependency_graph.watched_path.is_absolute()

    @pytest.mark.asyncio
    async def test_integration_with_watch_loop(self, temp_project, monkeypatch):
        """Test integration with the file watch loop."""
        server = LiveServer(
            input_path=temp_project, include=["*.py"], open_url=False, verbose=False
        )
        server._notification_delay = 0.1

        # Mock dependencies
        ws = AsyncMock()
        server.connections = {ws}

        # Register a client watching main.py
        server.client_registry.register_client("test_client", ws)
        server.client_registry.watch_file("test_client", str(temp_project / "main.py"))

        # Mock the awatch generator to simulate file changes
        async def mock_awatch(*args, **kwargs):
            # Yield a single file change (3 = Modified)
            yield [(3, str(temp_project / "main.py"))]
            # Then stop
            server._stop_event.set()

        # Patch various methods
        with patch("colight_prose.server.awatch", mock_awatch):
            with patch.object(
                server, "_send_reload_signal", new_callable=AsyncMock
            ) as mock_reload:
                # Run watch loop
                await server._watch_for_changes()

        # Verify reload signal was sent for the modified file
        mock_reload.assert_called_once()
        # Check that the file path passed to _send_reload_signal ends with main.py
        called_file_path = mock_reload.call_args[0][0]
        assert called_file_path.name == "main.py"
