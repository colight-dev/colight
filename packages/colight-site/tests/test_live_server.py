"""Tests for the LiveServer functionality."""

import pytest
from unittest.mock import patch
from werkzeug.test import Client
from werkzeug.wrappers import Response

from colight_live.server import LiveServer, OnDemandMiddleware
from colight_site.builder import BuildConfig


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project with Python files."""
    # Create some test Python files
    (tmp_path / "index.py").write_text("""
import colight.plot as plot
plot.text("# Index Page")
""")

    (tmp_path / "example.py").write_text("""
import colight.plot as plot
plot.text("# Example Page")
""")

    # Create a subdirectory with a file
    subdir = tmp_path / "docs"
    subdir.mkdir()
    (subdir / "guide.py").write_text("""
import colight.plot as plot
plot.text("# Guide Page")
""")

    return tmp_path


@pytest.fixture
def output_dir(tmp_path):
    """Create output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


def test_on_demand_middleware_builds_files(temp_project, output_dir):
    """Test that OnDemandMiddleware builds Python files on demand."""
    config = BuildConfig(formats={"html"})

    # Create a simple app that returns 404
    def not_found_app(environ, start_response):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found"]

    # Create middleware
    middleware = OnDemandMiddleware(
        not_found_app, temp_project, output_dir, config, include=["*.py"], ignore=None
    )

    # Create a test client
    client = Client(middleware, Response)

    # Request a file that doesn't exist yet
    response = client.get("/example.html")

    # Should return 404 because the base app returns 404
    # (In real usage, SharedDataMiddleware would serve the built file)
    assert response.status_code == 404

    # But the file should have been built
    assert (output_dir / "example.html").exists()
    assert "Example Page" in (output_dir / "example.html").read_text()


def test_on_demand_middleware_skips_index(temp_project, output_dir):
    """Test that OnDemandMiddleware skips index.html generation (handled by client)."""
    config = BuildConfig(formats={"html"})

    def not_found_app(environ, start_response):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found"]

    middleware = OnDemandMiddleware(
        not_found_app, temp_project, output_dir, config, include=["*.py"], ignore=None
    )

    client = Client(middleware, Response)

    # Request index.html
    response = client.get("/index.html")

    # Should NOT generate index page (handled by client-side outliner)
    assert response.status_code == 404
    assert not (output_dir / "index.html").exists()


def test_on_demand_middleware_subdirectories(temp_project, output_dir):
    """Test that OnDemandMiddleware handles subdirectories correctly."""
    config = BuildConfig(formats={"html"})

    def not_found_app(environ, start_response):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found"]

    middleware = OnDemandMiddleware(
        not_found_app, temp_project, output_dir, config, include=["*.py"], ignore=None
    )

    client = Client(middleware, Response)

    # Request a file in a subdirectory
    client.get("/docs/guide.html")

    # Should build the file in the correct subdirectory
    assert (output_dir / "docs" / "guide.html").exists()
    assert "Guide Page" in (output_dir / "docs" / "guide.html").read_text()


def test_on_demand_middleware_only_rebuilds_when_needed(temp_project, output_dir):
    """Test that files are only rebuilt when source is newer."""
    config = BuildConfig(formats={"html"})

    def not_found_app(environ, start_response):
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found"]

    middleware = OnDemandMiddleware(
        not_found_app, temp_project, output_dir, config, include=["*.py"], ignore=None
    )

    client = Client(middleware, Response)

    # First request builds the file
    client.get("/example.html")
    assert (output_dir / "example.html").exists()

    # Get initial modification time
    initial_mtime = (output_dir / "example.html").stat().st_mtime

    # Second request should not rebuild
    with patch("colight_site.api.build_file") as mock_build:
        client.get("/example.html")
        mock_build.assert_not_called()

    # Modification time should be unchanged
    assert (output_dir / "example.html").stat().st_mtime == initial_mtime


@pytest.mark.asyncio
async def test_live_server_initialization(temp_project, output_dir):
    """Test LiveServer initialization and index generation."""
    config = BuildConfig(formats={"html"})

    server = LiveServer(
        temp_project,
        output_dir,
        config=config,
        include=["*.py"],
        ignore=None,
        host="127.0.0.1",
        http_port=5555,
        ws_port=5556,
        open_url=False,
    )

    # Test basic initialization
    assert server.input_path == temp_project
    assert server.output_path == output_dir
    assert server.host == "127.0.0.1"
    assert server.http_port == 5555
    assert server.ws_port == 5556
