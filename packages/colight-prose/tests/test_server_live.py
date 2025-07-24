import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from werkzeug.test import Client
from werkzeug.wrappers import Response

from colight_prose.static.server_watch import HtmlFallbackMiddleware, LiveReloadServer


@pytest.fixture
def temp_site(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.html").write_text("<h1>Home</h1>")
    (dist_dir / "about.html").write_text("<h1>About</h1>")
    (dist_dir / "script.js").write_text("console.log('hello')")

    test_artifacts_dir = tmp_path / "test-artifacts" / "colight-prose"
    test_artifacts_dir.mkdir(parents=True)
    (test_artifacts_dir / "extra.html").write_text("<h1>Extra</h1>")

    return {
        "/dist": dist_dir,
        "/": test_artifacts_dir,
    }


@pytest.fixture
def fallback_app(temp_site):
    def not_found_app(environ, start_response):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found"]

    # In a real scenario, SharedDataMiddleware would serve the files.
    # For this test, we simulate its behavior by checking if the path exists.
    def mock_shared_data_app(environ, start_response):
        path = environ.get("PATH_INFO", "").lstrip("/")
        found = False
        for mount, root in temp_site.items():
            if path.startswith(mount.lstrip("/")):
                file_path = root / path[len(mount.lstrip("/")) :].lstrip("/")
                if file_path.exists():
                    found = True
                    break
        if found:
            start_response("200 OK", [("Content-Type", "text/html")])
            return [b"File content"]
        return not_found_app(environ, start_response)

    app = HtmlFallbackMiddleware(mock_shared_data_app, temp_site)
    return Client(app, Response)


def test_html_fallback_serves_html_for_naked_path(fallback_app):
    """Test that a request for '/about' serves 'about.html'"""
    response = fallback_app.get("/dist/about")
    assert response.status_code == 200
    assert response.data == b"File content"


def test_html_fallback_serves_root_index(fallback_app):
    """Test that a request for '/' serves 'index.html'"""
    response = fallback_app.get("/")
    assert response.status_code == 200
    assert response.data == b"File content"


def test_html_fallback_ignores_files_with_extensions(fallback_app):
    """Test that a request for '/script.js' is ignored"""
    response = fallback_app.get("/dist/script.js")
    assert response.status_code == 200  # Served by mock_shared_data_app


def test_html_fallback_handles_not_found(fallback_app):
    """Test that a request for a non-existent file returns 404"""
    response = fallback_app.get("/dist/non-existent")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_live_reload_server_sends_reload_signal(temp_site):
    """Test that the server sends a reload signal on file change."""
    server = LiveReloadServer(roots=temp_site, open_url_delay=False)
    mock_ws = AsyncMock()
    server.connections.add(mock_ws)

    async def mock_awatch(*args, **kwargs):
        yield [("change", "some/file")]
        if hasattr(kwargs.get("stop_event"), "is_set"):
            while not kwargs["stop_event"].is_set():
                await asyncio.sleep(0.1)

    with patch("colight_prose.static.server_watch.awatch", mock_awatch):
        with patch.object(
            server, "_send_reload_signal", new_callable=AsyncMock
        ) as mock_send:
            task = asyncio.create_task(server._watch_for_changes())
            await asyncio.sleep(0.1)  # allow the watcher to run
            server._stop_event.set()
            await task

            mock_send.assert_called_once()


@pytest.mark.asyncio
async def test_websocket_handler_manages_connections(temp_site):
    """Test that websocket connections are properly managed."""
    server = LiveReloadServer(roots=temp_site, open_url_delay=False)
    mock_ws = AsyncMock()
    mock_ws.wait_closed = AsyncMock()

    # Initially empty
    assert len(server.connections) == 0

    # Simulate connection
    await server._websocket_handler(mock_ws)

    # Connection should be removed after handler completes
    assert len(server.connections) == 0
    mock_ws.wait_closed.assert_called_once()


@pytest.mark.asyncio
async def test_send_reload_signal_sends_correct_message(temp_site):
    """Test that reload signal sends the correct JSON message."""
    server = LiveReloadServer(roots=temp_site, open_url_delay=False)
    mock_ws1 = AsyncMock()
    mock_ws2 = AsyncMock()

    server.connections.add(mock_ws1)
    server.connections.add(mock_ws2)

    await server._send_reload_signal()

    expected_message = json.dumps({"type": "reload"})
    mock_ws1.send.assert_called_once_with(expected_message)
    mock_ws2.send.assert_called_once_with(expected_message)


@pytest.mark.asyncio
async def test_send_reload_signal_handles_empty_connections(temp_site):
    """Test that reload signal handles empty connections gracefully."""
    server = LiveReloadServer(roots=temp_site, open_url_delay=False)

    # Should not raise any errors
    await server._send_reload_signal()


def test_server_initialization(temp_site):
    """Test that the server initializes with correct parameters."""
    server = LiveReloadServer(
        roots=temp_site,
        host="localhost",
        http_port=8080,
        ws_port=8081,
        open_url_delay=False,
    )

    assert server.roots == temp_site
    assert server.host == "localhost"
    assert server.http_port == 8080
    assert server.ws_port == 8081
    assert server.open_url_delay is False
    assert len(server.connections) == 0


def test_server_stop_sets_event(temp_site):
    """Test that stopping the server sets the stop event."""
    server = LiveReloadServer(roots=temp_site, open_url_delay=False)

    assert not server._stop_event.is_set()
    server.stop()
    assert server._stop_event.is_set()
