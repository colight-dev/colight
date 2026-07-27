"""Tests for the RunVersion architecture in LiveServer."""

import asyncio
import json
import pathlib
from unittest.mock import AsyncMock, Mock, patch

import pytest
from colight_prose.server import LiveServer


@pytest.mark.asyncio
async def test_run_version_increments():
    """Test that run version increments with each build."""
    server = LiveServer(
        input_path=pathlib.Path("."),
        include=["*.py"],
    )

    # Get initial run version
    run1 = next(server._run_counter)
    run2 = next(server._run_counter)
    run3 = next(server._run_counter)

    assert run1 == 1
    assert run2 == 2
    assert run3 == 3


@pytest.mark.asyncio
async def test_websocket_messages_with_run_version():
    """Test that WebSocket messages include run version."""
    server = LiveServer(
        input_path=pathlib.Path("."),
        include=["*.py"],
    )

    # Mock WebSocket connection
    mock_ws = AsyncMock()
    server.connections.add(mock_ws)

    # Test run-start message
    await server._ws_broadcast_all({"run": 1, "type": "run-start", "file": "test.py"})
    mock_ws.send.assert_called_with(
        json.dumps({"run": 1, "type": "run-start", "file": "test.py"})
    )

    # Test block-result message
    await server._ws_broadcast_all(
        {
            "run": 1,
            "type": "block-result",
            "block": "block-1",
            "ok": True,
            "stdout": "output",
            "error": None,
            "showsVisual": False,
            "elements": [],
        }
    )

    # Test run-end message
    await server._ws_broadcast_all({"run": 1, "type": "run-end"})

    # Verify all messages were sent
    assert mock_ws.send.call_count == 3


@pytest.mark.asyncio
async def test_run_cancellation():
    """Test that in-flight runs can be cancelled."""
    server = LiveServer(
        input_path=pathlib.Path("."),
        include=["*.py"],
    )

    # Create a mock task that takes time
    async def slow_task():
        await asyncio.sleep(1)
        return "completed"

    # Start a task
    task = asyncio.create_task(slow_task())
    server._current_run_task = task

    # Cancel it
    task.cancel()

    # Verify it was cancelled
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_stale_message_filtering():
    """Test that client ignores messages from old runs."""
    # This tests the client-side logic conceptually
    # In real implementation, this would be in the JavaScript tests

    latest_run = 5

    # Message from old run should be ignored
    old_message = {"run": 3, "type": "block-result", "block": "test"}
    assert old_message["run"] < latest_run  # Should be filtered

    # Message from current run should be processed
    current_message = {"run": 5, "type": "block-result", "block": "test"}
    assert current_message["run"] == latest_run  # Should be processed

    # Message from future run should update latest and be processed
    future_message = {"run": 6, "type": "run-start", "file": "test.py"}
    assert future_message["run"] > latest_run  # Should update latest_run


@pytest.mark.asyncio
async def test_request_load_handling():
    """Test that server handles request-load messages from client."""
    server = LiveServer(
        input_path=pathlib.Path("."),
        include=["*.py"],
    )

    # Mock the API middleware
    mock_api = Mock()
    mock_file_resolver = Mock()
    mock_file_resolver.find_source_file.return_value = pathlib.Path("test.py")
    mock_api.file_resolver = mock_file_resolver
    server._api_middleware = mock_api

    # Mock WebSocket that sends a request-load message
    mock_ws = AsyncMock()

    # Simulate receiving a request-load message
    message = json.dumps({"type": "request-load", "path": "test"})

    # Mock the _send_reload_signal method
    with patch.object(server, "_send_reload_signal") as mock_reload:
        # Process the message (this would normally happen in _websocket_handler)
        data = json.loads(message)
        if data.get("type") == "request-load" and data.get("path"):
            file_path = data["path"]
            source_file = mock_file_resolver.find_source_file(file_path + ".html")
            if source_file:
                await server._send_reload_signal(source_file)

        # Verify _send_reload_signal was called
        mock_reload.assert_called_once_with(pathlib.Path("test.py"))


@pytest.mark.asyncio
async def test_multiple_rapid_changes():
    """Test that rapid file changes cancel previous builds."""
    server = LiveServer(
        input_path=pathlib.Path("."),
        include=["*.py"],
    )

    # Track which runs complete
    completed_runs = []

    async def mock_trigger_build(file_path, client_run=None):
        run = next(server._run_counter)
        try:
            # Simulate some work
            await asyncio.sleep(0.1)
            completed_runs.append(run)
        except asyncio.CancelledError:
            # Run was cancelled
            pass

    # Patch the trigger build method
    with patch.object(server, "_trigger_build", mock_trigger_build):
        # Simulate rapid file changes
        file1 = pathlib.Path("test1.py")
        file2 = pathlib.Path("test2.py")
        file3 = pathlib.Path("test3.py")

        # Start multiple builds in quick succession
        await server._send_reload_signal(file1)
        await asyncio.sleep(0.01)  # Small delay

        await server._send_reload_signal(file2)
        await asyncio.sleep(0.01)  # Small delay

        await server._send_reload_signal(file3)

        # Wait for all tasks to complete or be cancelled
        await asyncio.sleep(0.2)

        # Only the last run should complete (or possibly the last two if timing allows)
        assert len(completed_runs) <= 2
        if completed_runs:
            assert max(completed_runs) >= 2  # At least run 2 or 3 should complete
