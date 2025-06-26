"""Tests for watch-serve functionality."""

from unittest.mock import patch

from colight_site.watcher import watch_build_and_serve
from colight_site.builder import BuildConfig


def test_watch_build_and_serve_creates_output_directory(tmp_path):
    """Test that watch_build_and_serve creates the output directory."""
    input_file = tmp_path / "test.colight.py"
    input_file.write_text("import colight as cl\ncl.sphere()")
    output_dir = tmp_path / "output"

    # Mock the server and watch to prevent actual serving
    with patch("colight_site.watcher.LiveReloadServer") as mock_server:
        with patch("colight_site.watcher.watch") as mock_watch:
            with patch("colight_site.watcher.threading.Thread"):
                # Make watch raise KeyboardInterrupt to exit immediately
                mock_watch.side_effect = KeyboardInterrupt()

                try:
                    watch_build_and_serve(
                        input_file,
                        output_dir,
                        config=BuildConfig(verbose=False),
                        open_url=False,
                    )
                except KeyboardInterrupt:
                    pass

                # Check output directory was created
                assert output_dir.exists()

                # Check LiveReloadServer was created with correct params
                # The roots should include the output dir and may include /dist if it exists
                mock_server.assert_called_once()
                call_args = mock_server.call_args
                assert call_args.kwargs["host"] == "127.0.0.1"
                assert call_args.kwargs["http_port"] == 5500
                assert call_args.kwargs["ws_port"] == 5501
                assert call_args.kwargs["open_url_delay"] is False
                assert "/" in call_args.kwargs["roots"]
                assert call_args.kwargs["roots"]["/"] == output_dir


def test_watch_build_and_serve_generates_index_for_directory(tmp_path):
    """Test that watch_build_and_serve generates index.html for directory mode."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create test files
    (input_dir / "test1.colight.py").write_text("import colight as cl\ncl.sphere()")
    (input_dir / "test2.colight.py").write_text("import colight as cl\ncl.cube()")

    output_dir = tmp_path / "output"

    # Mock the server and watch
    with patch("colight_site.watcher.LiveReloadServer"):
        with patch("colight_site.watcher.watch") as mock_watch:
            with patch("colight_site.watcher.threading.Thread"):
                # Make watch raise KeyboardInterrupt to exit immediately
                mock_watch.side_effect = KeyboardInterrupt()

                try:
                    watch_build_and_serve(
                        input_dir,
                        output_dir,
                        config=BuildConfig(verbose=False),
                        open_url=False,
                    )
                except KeyboardInterrupt:
                    pass

                # Check index.html was created
                index_file = output_dir / "index.html"
                assert index_file.exists()

                # Check index contains the expected elements
                index_content = index_file.read_text()
                # The index is now generated as a Colight widget
                assert "application/x-colight" in index_content


def test_watch_build_and_serve_defaults_to_html_format(tmp_path):
    """Test that watch_build_and_serve defaults to HTML format."""
    input_file = tmp_path / "test.colight.py"

    input_file.write_text("import colight as cl\ncl.sphere()")
    output_dir = tmp_path / "output"

    # Mock the server and watch
    with patch("colight_site.watcher.LiveReloadServer"):
        with patch("colight_site.watcher.watch") as mock_watch:
            with patch("colight_site.watcher.threading.Thread"):
                with patch("colight_site.watcher.api.build_file") as mock_build:
                    # Make watch raise KeyboardInterrupt to exit immediately
                    mock_watch.side_effect = KeyboardInterrupt()

                    try:
                        watch_build_and_serve(
                            input_file,
                            output_dir,
                            open_url=False,
                        )
                    except KeyboardInterrupt:
                        pass

                    # Check build was called with HTML format
                    mock_build.assert_called()
                    args, kwargs = mock_build.call_args
                    assert "html" in kwargs["config"].formats
