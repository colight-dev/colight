"""Tests for watch-serve functionality."""

from unittest.mock import patch

from colight_prose.static.builder import BuildConfig
from colight_prose.static.watcher import watch_build_and_serve


def test_watch_build_and_serve_creates_output_directory(tmp_path):
    """Test that watch_build_and_serve creates the output directory."""
    input_file = tmp_path / "test.py"
    input_file.write_text("import colight as cl\ncl.sphere()")
    output_dir = tmp_path / "output"

    # Mock the server and watch to prevent actual serving
    with patch("colight_prose.static.watcher.LiveReloadServer") as mock_server:
        with patch("colight_prose.static.watcher.watch") as mock_watch:
            with patch("colight_prose.static.watcher.threading.Thread"):
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


def test_watch_build_and_serve_builds_files_for_directory(tmp_path):
    """Test that watch_build_and_serve builds individual files for directory mode."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Create test files - use simple content that doesn't require colight
    (input_dir / "test1.py").write_text("# Test 1\nprint('test1')")
    (input_dir / "test2.py").write_text("# Test 2\nprint('test2')")

    output_dir = tmp_path / "output"

    # Mock the server and thread, but let the initial build happen
    with patch("colight_prose.static.watcher.LiveReloadServer"):
        with patch("colight_prose.static.watcher.threading.Thread"):
            # Mock watch to raise KeyboardInterrupt after initial build
            with patch("colight_prose.static.watcher.watch") as mock_watch:
                mock_watch.side_effect = KeyboardInterrupt()

                try:
                    watch_build_and_serve(
                        input_dir,
                        output_dir,
                        config=BuildConfig(verbose=False, formats={"html"}),
                        open_url=False,
                    )
                except KeyboardInterrupt:
                    pass

                # Check individual files were built
                assert (output_dir / "test1.html").exists()
                assert (output_dir / "test2.html").exists()

                # Index is now handled by client-side outliner, not generated as a file
                assert not (output_dir / "index.html").exists()


def test_watch_build_and_serve_defaults_to_html_format(tmp_path):
    """Test that watch_build_and_serve defaults to HTML format."""
    input_file = tmp_path / "test.py"

    input_file.write_text("import colight as cl\ncl.sphere()")
    output_dir = tmp_path / "output"

    # Mock the server and watch
    with patch("colight_prose.static.watcher.LiveReloadServer"):
        with patch("colight_prose.static.watcher.watch") as mock_watch:
            with patch("colight_prose.static.watcher.threading.Thread"):
                with patch(
                    "colight_prose.static.watcher.builder.build_file"
                ) as mock_build:
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
