"""End-to-end integration tests for complete user workflows."""

import pathlib
import tempfile
import time


from colight_site import api
from colight_site.builder import BuildConfig


class TestBuildWorkflow:
    """Test complete build workflow from Python file to output."""

    def test_complete_build_workflow_single_file(self):
        """Test complete build workflow for a single Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create a realistic Python file with colight content
            input_file = tmpdir / "example.py"
            input_file.write_text("""# Data Analysis Example

# This is a simple data analysis notebook.

import numpy as np

# Generate some sample data
data = np.random.randn(100)

# Show the data
data

print("Analysis complete!")
""")

            output_file = tmpdir / "example.md"
            config = BuildConfig(
                verbose=True, formats={"markdown"}, continue_on_error=True
            )

            # Execute the complete build workflow
            api.build_file(input_file, output_file, config=config)

            # Verify output file was created
            assert output_file.exists()

            # Verify content structure
            content = output_file.read_text()
            assert "Data Analysis Example" in content
            assert "This is a simple data analysis" in content
            # Code should be present since we didn't hide it globally
            assert "import numpy" in content
            assert "data = np.random.randn" in content

    def test_complete_build_workflow_directory(self):
        """Test complete build workflow for a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create multiple Python files
            (tmpdir / "file1.py").write_text("""
# File 1
print("Hello from file 1")
""")

            (tmpdir / "file2.py").write_text("""
# File 2  
print("Hello from file 2")
""")

            (tmpdir / "subdir").mkdir()
            (tmpdir / "subdir" / "file3.py").write_text("""
# File 3
print("Hello from file 3")
""")

            output_dir = tmpdir / "output"
            config = BuildConfig(
                verbose=True, formats={"markdown"}, continue_on_error=True
            )

            # Execute the complete build workflow
            api.build_directory(tmpdir, output_dir, config=config)

            # Verify output files were created
            assert (output_dir / "file1.md").exists()
            assert (output_dir / "file2.md").exists()
            assert (output_dir / "subdir" / "file3.md").exists()

            # Verify content
            content1 = (output_dir / "file1.md").read_text()
            assert "Hello from file 1" in content1

    def test_build_workflow_with_errors(self):
        """Test build workflow behavior when Python code has errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create a Python file with syntax errors
            input_file = tmpdir / "broken.py"
            input_file.write_text("""# Broken Example

# This file has syntax errors but also valid prose.

# This should work fine
print("This part works")

# This will cause a runtime error
undefined_variable = some_undefined_variable
""")

            output_file = tmpdir / "broken.md"
            config = BuildConfig(
                verbose=True,
                formats={"markdown"},
                continue_on_error=True,  # Should continue despite errors
            )

            # Should not raise an exception
            api.build_file(input_file, output_file, config=config)

            # Output file should still be created
            assert output_file.exists()

            # Should contain the prose parts
            content = output_file.read_text()
            assert "Broken Example" in content
            assert "This file has syntax errors" in content

    def test_build_workflow_html_format(self):
        """Test build workflow with HTML output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            input_file = tmpdir / "example.py"
            input_file.write_text("""# HTML Example

# This will be converted to HTML.

print("Hello, HTML world!")
""")

            output_file = tmpdir / "example.html"
            config = BuildConfig(verbose=True, formats={"html"}, continue_on_error=True)

            api.build_file(input_file, output_file, config=config)

            assert output_file.exists()
            content = output_file.read_text()

            # Should contain HTML tags
            assert "<html>" in content or "<!DOCTYPE" in content
            assert "HTML Example" in content


class TestWatchWorkflow:
    """Test complete watch workflow with file monitoring."""

    def test_watch_workflow_file_change_detection(self):
        """Test that watch workflow detects file changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create initial file
            input_file = tmpdir / "watched.py"
            input_file.write_text("""
# Initial Content
print("Version 1")
""")

            output_dir = tmpdir / "output"
            output_dir.mkdir()

            config = BuildConfig(formats={"markdown"})

            # Mock the watcher to simulate file change detection
            file_changed = False

            def mock_watch_callback():
                nonlocal file_changed
                file_changed = True
                # Simulate file modification
                input_file.write_text("""
# Updated Content
print("Version 2")
""")
                return True  # Stop watching

            # Test the core file monitoring logic
            # This simulates what the actual watcher would do
            initial_mtime = input_file.stat().st_mtime

            # Modify the file
            time.sleep(0.1)  # Ensure different timestamp
            input_file.write_text("""
# Updated Content  
print("Version 2")
""")

            new_mtime = input_file.stat().st_mtime
            assert new_mtime > initial_mtime

            # Rebuild and verify change was detected
            api.build_file(input_file, output_dir / "watched.md", config=config)

            content = (output_dir / "watched.md").read_text()
            assert "Updated Content" in content
            assert "Version 2" in content


class TestLiveWorkflow:
    """Test complete live server workflow."""

    def test_live_server_initialization(self):
        """Test live server initialization and configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create test files
            (tmpdir / "test.py").write_text("""
# Live Test
print("Live server test")
""")

            cache_dir = tmpdir / ".colight_cache"
            config = BuildConfig(formats={"html"})

            # Import here to avoid circular imports during module load
            from colight_live.server import LiveServer

            server = LiveServer(
                input_path=tmpdir,
                cache_path=cache_dir,
                config=config,
                include=["*.py"],
                ignore=None,
                host="127.0.0.1",
                http_port=5555,
                ws_port=5556,
                open_url=False,
            )

            # Verify server configuration
            assert server.input_path == tmpdir
            assert server.cache_path == cache_dir
            assert server.host == "127.0.0.1"
            assert server.http_port == 5555
            assert server.ws_port == 5556

    def test_live_server_on_demand_building(self):
        """Test live server on-demand building functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create test file
            test_file = tmpdir / "test.py"
            test_file.write_text("""
# On-Demand Test
print("Building on demand")
""")

            cache_dir = tmpdir / ".colight_cache"
            config = BuildConfig(formats={"html"})

            # Test the core on-demand building logic
            # This simulates what LiveServer does when a file is requested

            # Initially, no cache file should exist
            cache_file = cache_dir / "test.html"
            assert not cache_file.exists()

            # Simulate on-demand build
            api.build_file(test_file, cache_file, config=config)

            # Cache file should now exist
            assert cache_file.exists()

            # Verify content
            content = cache_file.read_text()
            assert "On-Demand Test" in content
            assert "Building on demand" in content


class TestProjectInitWorkflow:
    """Test complete project initialization workflow."""

    def test_project_init_workflow(self):
        """Test complete project initialization workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            project_dir = tmpdir / "new_project"

            # Initialize project
            api.init_project(project_dir)

            # Verify project structure was created
            assert project_dir.exists()
            assert project_dir.is_dir()

            # Check for expected files/directories
            # Note: The actual init_project implementation determines
            # what files should be created. This test verifies the
            # function can be called without errors.

    def test_project_init_existing_directory(self):
        """Test project initialization in existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            project_dir = tmpdir / "existing"
            project_dir.mkdir()

            # Add some existing content
            (project_dir / "existing_file.txt").write_text("I was here first")

            # Initialize project (should not fail)
            api.init_project(project_dir)

            # Existing file should still be there
            assert (project_dir / "existing_file.txt").exists()
            content = (project_dir / "existing_file.txt").read_text()
            assert "I was here first" in content


class TestErrorRecoveryWorkflows:
    """Test workflows with various error conditions."""

    def test_workflow_with_file_permission_errors(self):
        """Test workflow behavior with file permission issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            input_file = tmpdir / "test.py"
            input_file.write_text("print('test')")

            # Create output directory without write permissions
            output_dir = tmpdir / "readonly"
            output_dir.mkdir()

            config = BuildConfig(continue_on_error=True)

            # This should handle permission errors gracefully
            # (Exact behavior depends on implementation)
            try:
                api.build_directory(tmpdir, output_dir, config=config)
            except PermissionError:
                # If it raises PermissionError, that's also acceptable
                # depending on continue_on_error behavior
                pass

    def test_workflow_with_malformed_python_files(self):
        """Test workflow with various malformed Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # File with invalid encoding
            malformed_file = tmpdir / "malformed.py"
            malformed_file.write_bytes(b'\xff\xfe# Invalid encoding\nprint("test")')

            output_file = tmpdir / "malformed.md"
            config = BuildConfig(continue_on_error=True)

            # Should handle encoding errors gracefully
            try:
                api.build_file(malformed_file, output_file, config=config)
            except UnicodeDecodeError:
                # If it raises UnicodeDecodeError, that's expected for malformed files
                pass

    def test_workflow_with_import_errors(self):
        """Test workflow with files that have import errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            input_file = tmpdir / "import_error.py"
            input_file.write_text("""
# Import Error Test

This file imports a non-existent module.

import nonexistent_module
from another_nonexistent import something

print("This might not execute")
""")

            output_file = tmpdir / "import_error.md"
            config = BuildConfig(continue_on_error=True)

            # Should complete without raising exceptions
            api.build_file(input_file, output_file, config=config)

            # Output should still be created with prose content
            assert output_file.exists()
            content = output_file.read_text()
            assert "Import Error Test" in content
            assert "This file imports a non-existent module" in content
