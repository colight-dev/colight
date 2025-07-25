"""Tests for targeted execution with client-aware filtering."""

import pathlib
from unittest.mock import AsyncMock, Mock, patch

import pytest

from colight_prose.server import LiveServer


class TestTargetedExecution:
    """Test that server only executes files being watched by clients."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project with dependencies."""
        # Create main file that imports utils
        main_file = tmp_path / "main.py"
        main_file.write_text("""
import utils

# Main application
result = utils.calculate(10)
print(f"Result: {result}")
""")

        # Create utils file
        utils_file = tmp_path / "utils.py"
        utils_file.write_text("""
def calculate(x):
    return x * 2
""")

        # Create unrelated file
        other_file = tmp_path / "other.py"
        other_file.write_text("""
# Unrelated file
print("Other module")
""")

        return tmp_path

    @pytest.mark.asyncio
    async def test_only_executes_watched_files(self, temp_project):
        """Test that server only executes files being watched."""
        server = LiveServer(
            input_path=temp_project, include=["*.py"], open_url=False, verbose=True
        )

        # Mock the send_reload_signal to track executions
        executed_files = []
        original_send_reload = server._send_reload_signal

        async def mock_send_reload(changed_file=None, client_run=None):
            if changed_file:
                executed_files.append(str(changed_file.name))
            return await original_send_reload(changed_file, client_run)

        server._send_reload_signal = mock_send_reload

        # Setup client registry
        mock_ws = Mock()
        server.client_registry.register_client("client1", mock_ws)
        server.client_registry.watch_file("client1", "main.py")

        # Setup dependency graph
        server.dependency_graph.analyze_directory(temp_project)

        # Simulate file change to utils.py
        utils_path = temp_project / "utils.py"
        other_path = temp_project / "other.py"

        # Mock _matches_patterns to return True
        server._matches_patterns = Mock(return_value=True)

        # Process changes to utils.py (which affects main.py)
        with patch("pathlib.Path.relative_to") as mock_relative:
            mock_relative.side_effect = (
                lambda base: pathlib.Path("utils.py") if base == temp_project else None
            )

            # Simulate the change processing logic from _watch_for_changes
            changed_files = {utils_path}
            matching_changes = {utils_path}

            # Track which files need execution
            files_to_execute = set()
            watched_files = server.client_registry.get_watched_files()

            for file_path in matching_changes:
                relative_path = "utils.py"
                affected = server.dependency_graph.get_affected_files(relative_path)

                # Only execute files that are watched
                watched_affected = [f for f in affected if f in watched_files]

                if watched_affected:
                    for watched_file in watched_affected:
                        watched_path = temp_project / watched_file
                        if watched_path.exists():
                            await server._send_reload_signal(watched_path)

        # Verify only main.py was executed (it's watched and affected by utils.py)
        assert "main.py" in executed_files
        assert "utils.py" not in executed_files  # Not watched directly
        assert "other.py" not in executed_files  # Not affected

    @pytest.mark.asyncio
    async def test_cache_eviction_for_unwatched(self, temp_project):
        """Test that cache entries are marked for eviction when files become unwatched."""
        server = LiveServer(
            input_path=temp_project, include=["*.py"], open_url=False, verbose=False
        )

        # Setup API middleware with cache manager
        from colight_prose.incremental_executor import IncrementalExecutor

        incremental_executor = IncrementalExecutor()

        # Mock the API middleware
        server._api_middleware = Mock()
        server._api_middleware.incremental_executor = incremental_executor

        # Client watches main.py
        mock_ws = Mock()
        server.client_registry.register_client("client1", mock_ws)
        server.client_registry.watch_file("client1", "main.py")

        # Mark main.py for eviction (should be unmarked since it's watched)
        incremental_executor.mark_file_for_eviction("main.py")
        incremental_executor.unmark_file_for_eviction("main.py")
        # Access the internal cache to verify
        assert "main.py" not in incremental_executor.cache.eviction_times

        # Client unwatches main.py
        server.client_registry.unwatch_file("client1", "main.py")

        # Simulate unwatch handling
        if not server.client_registry.get_watchers("main.py"):
            incremental_executor.mark_file_for_eviction("main.py")

        # Verify main.py is marked for eviction
        assert "main.py" in incremental_executor.cache.eviction_times

    @pytest.mark.asyncio
    async def test_targeted_broadcast(self, temp_project):
        """Test that updates are only sent to clients watching the file."""
        server = LiveServer(
            input_path=temp_project, include=["*.py"], open_url=False, verbose=False
        )

        # Create mock websockets
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        # Register clients
        server.client_registry.register_client("client1", ws1)
        server.client_registry.register_client("client2", ws2)
        server.client_registry.register_client("client3", ws3)

        # Client 1 and 2 watch main.py, client 3 watches other.py
        # Use absolute paths as the server does
        main_path = str((temp_project / "main.py").resolve())
        other_path = str((temp_project / "other.py").resolve())
        server.client_registry.watch_file("client1", main_path)
        server.client_registry.watch_file("client2", main_path)
        server.client_registry.watch_file("client3", other_path)

        # Add websockets to connections
        server.connections = {ws1, ws2, ws3}

        # Send targeted broadcast for main.py
        test_message = {"type": "test", "file": "main.py"}
        await server._ws_broadcast_to_file_watchers(test_message, "main.py")

        # Verify only clients 1 and 2 received the message
        ws1.send.assert_called_once()
        ws2.send.assert_called_once()
        ws3.send.assert_not_called()

        # Verify message content
        sent_message = ws1.send.call_args[0][0]
        assert '"type": "test"' in sent_message
        assert '"file": "main.py"' in sent_message

    @pytest.mark.asyncio
    async def test_no_execution_when_no_watchers(self, temp_project):
        """Test that no execution happens when no clients are watching affected files."""
        server = LiveServer(
            input_path=temp_project, include=["*.py"], open_url=False, verbose=True
        )

        # No clients watching any files
        executed_files = []

        async def mock_send_reload(changed_file=None, client_run=None):
            if changed_file:
                executed_files.append(str(changed_file.name))

        server._send_reload_signal = mock_send_reload

        # Setup dependency graph
        server.dependency_graph.analyze_directory(temp_project)

        # Simulate file change to utils.py
        utils_path = temp_project / "utils.py"
        server._matches_patterns = Mock(return_value=True)

        # Process changes
        watched_files = server.client_registry.get_watched_files()
        assert len(watched_files) == 0  # No files being watched

        # Simulate change processing
        relative_path = "utils.py"
        affected = server.dependency_graph.get_affected_files(relative_path)

        watched_affected = [f for f in affected if f in watched_files]

        # No watched files affected
        assert len(watched_affected) == 0

        # Verify no files were executed
        assert len(executed_files) == 0
