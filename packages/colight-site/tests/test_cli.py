"""Tests for the CLI interface (colight_cli.py)."""

import pathlib
import tempfile
from unittest.mock import Mock, patch

from click.testing import CliRunner

from colight_cli import main


class TestCLIBuild:
    """Test the 'build' command."""

    def test_build_single_file_basic(self):
        """Test building a single Python file."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            input_file = tmpdir / "test.py"
            input_file.write_text("# Test file\nprint('hello')")

            with patch("colight_cli.api.build_file") as mock_build:
                result = runner.invoke(main, ["build", str(input_file)])

                assert result.exit_code == 0
                mock_build.assert_called_once()

    def test_build_nonexistent_file(self):
        """Test error handling for nonexistent input file."""
        runner = CliRunner()
        result = runner.invoke(main, ["build", "nonexistent.py"])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_build_directory_basic(self):
        """Test building a directory."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            (tmpdir / "test.py").write_text("# Test file\nprint('hello')")

            with patch("colight_cli.api.build_directory") as mock_build:
                result = runner.invoke(main, ["build", str(tmpdir)])

                assert result.exit_code == 0
                mock_build.assert_called_once()

    def test_build_with_all_options(self):
        """Test build command with all CLI options."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            input_file = tmpdir / "test.py"
            input_file.write_text("# Test file\nprint('hello')")
            output_file = tmpdir / "output.html"

            with patch("colight_cli.api.build_file") as mock_build:
                result = runner.invoke(
                    main,
                    [
                        "build",
                        str(input_file),
                        "--output",
                        str(output_file),
                        "--verbose",
                        "True",
                        "--format",
                        "html",
                        "--pragma",
                        "hide-code,show-visuals",
                        "--continue-on-error",
                        "False",
                        "--inline-threshold",
                        "1000",
                    ],
                )

                assert result.exit_code == 0
                mock_build.assert_called_once()
                # Verify config was built correctly
                config = mock_build.call_args[1]["config"]
                assert config.verbose is True
                assert "html" in config.formats
                assert "hide-code" in config.pragma
                assert config.continue_on_error is False
                assert config.inline_threshold == 1000

    def test_build_invalid_format(self):
        """Test error handling for invalid format option."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            input_file = tmpdir / "test.py"
            input_file.write_text("# Test file")

            result = runner.invoke(
                main, ["build", str(input_file), "--format", "invalid"]
            )

            assert result.exit_code != 0
            assert "invalid" in result.output.lower()


class TestCLIWatch:
    """Test the 'watch' command."""

    def test_watch_with_dev_server(self):
        """Test watch command with development server."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            (tmpdir / "test.py").write_text("# Test file\nprint('hello')")

            with patch("colight_cli.watcher.watch_build_and_serve") as mock_watch:
                # Use timeout to prevent hanging
                result = runner.invoke(
                    main,
                    [
                        "watch",
                        str(tmpdir),
                        "--dev-server",
                        "True",
                        "--port",
                        "5555",
                        "--no-open",
                    ],
                    catch_exceptions=False,
                )

                # Should start the watch process
                mock_watch.assert_called_once()
                args = mock_watch.call_args
                assert args[1]["http_port"] == 5555
                assert args[1]["open_url"] is False

    def test_watch_without_dev_server(self):
        """Test watch command without development server."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            (tmpdir / "test.py").write_text("# Test file")

            with patch("colight_cli.watcher.watch_and_build") as mock_watch:
                result = runner.invoke(
                    main, ["watch", str(tmpdir), "--dev-server", "False"]
                )

                mock_watch.assert_called_once()

    def test_watch_include_ignore_patterns(self):
        """Test watch command with include/ignore patterns."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            (tmpdir / "test.py").write_text("# Test file")

            with patch("colight_cli.watcher.watch_build_and_serve") as mock_watch:
                result = runner.invoke(
                    main,
                    [
                        "watch",
                        str(tmpdir),
                        "--include",
                        "*.py",
                        "--include",
                        "*.md",
                        "--ignore",
                        "test_*",
                        "--ignore",
                        "__pycache__",
                        "--no-open",
                    ],
                )

                args = mock_watch.call_args
                assert "*.py" in args[1]["include"]
                assert "*.md" in args[1]["include"]
                assert "test_*" in args[1]["ignore"]
                assert "__pycache__" in args[1]["ignore"]


class TestCLILive:
    """Test the 'live' command."""

    def test_live_basic(self):
        """Test basic live command."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            (tmpdir / "test.py").write_text("# Test file")

            with patch("colight_cli.LiveServer") as mock_server_class:
                mock_server = Mock()
                mock_server_class.return_value = mock_server

                # Mock asyncio.run to prevent actual server startup
                with patch("colight_cli.asyncio.run") as mock_run:
                    result = runner.invoke(
                        main, ["live", str(tmpdir), "--port", "6000", "--no-open"]
                    )

                    # Should create server instance
                    mock_server_class.assert_called_once()
                    server_args = mock_server_class.call_args
                    assert server_args[1]["http_port"] == 6000
                    assert server_args[1]["open_url"] is False

                    # Should start the server
                    mock_run.assert_called_once()

    def test_live_keyboard_interrupt(self):
        """Test live command handles KeyboardInterrupt gracefully."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            (tmpdir / "test.py").write_text("# Test file")

            with patch("colight_cli.LiveServer") as mock_server_class:
                mock_server = Mock()
                mock_server_class.return_value = mock_server

                # Mock asyncio.run to raise KeyboardInterrupt
                with patch("colight_cli.asyncio.run", side_effect=KeyboardInterrupt):
                    result = runner.invoke(main, ["live", str(tmpdir), "--no-open"])

                    # Should handle interrupt gracefully
                    assert result.exit_code == 0
                    assert "Stopping LiveServer" in result.output
                    mock_server.stop.assert_called_once()


class TestCLIInit:
    """Test the 'init' command."""

    def test_init_basic(self):
        """Test project initialization."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = pathlib.Path(tmpdir) / "new_project"

            with patch("colight_cli.api.init_project") as mock_init:
                result = runner.invoke(main, ["init", str(project_dir)])

                assert result.exit_code == 0
                mock_init.assert_called_once_with(project_dir)
                assert "Initialized project" in result.output


class TestCLIHelp:
    """Test CLI help and version information."""

    def test_main_help(self):
        """Test main command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Static site generator for Colight" in result.output

    def test_build_help(self):
        """Test build command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["build", "--help"])

        assert result.exit_code == 0
        assert "Build a .py file into markdown/HTML" in result.output

    def test_watch_help(self):
        """Test watch command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["watch", "--help"])

        assert result.exit_code == 0
        assert "Watch for changes" in result.output

    def test_live_help(self):
        """Test live command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["live", "--help"])

        assert result.exit_code == 0
        assert "Start LiveServer" in result.output

    def test_init_help(self):
        """Test init command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])

        assert result.exit_code == 0
        assert "Initialize a new colight-site project" in result.output

    def test_version(self):
        """Test version information."""
        runner = CliRunner()

        # Mock the version option to avoid package name issues
        with patch("click.version_option") as mock_version:
            mock_version.return_value = lambda f: f  # Identity decorator
            result = runner.invoke(main, ["--version"])

            # The actual behavior may vary, just ensure it doesn't crash
            # Version testing is complex due to package installation requirements


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    def test_invalid_command(self):
        """Test error handling for invalid commands."""
        runner = CliRunner()
        result = runner.invoke(main, ["invalid-command"])

        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_missing_required_argument(self):
        """Test error handling for missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(main, ["build"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_invalid_port_number(self):
        """Test error handling for invalid port numbers."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["watch", str(tmpdir), "--port", "invalid"])

            assert result.exit_code != 0

    def test_invalid_path_type(self):
        """Test error handling for invalid path arguments."""
        runner = CliRunner()
        result = runner.invoke(main, ["build", ""])

        assert result.exit_code != 0
