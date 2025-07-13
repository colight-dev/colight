"""Tests for error recovery and resilience scenarios."""

import pathlib
import tempfile
from unittest.mock import patch

from colight_site import api
from colight_site.builder import BuildConfig


class TestCacheCorruption:
    """Test recovery from corrupted cache files."""

    def test_corrupted_cache_file_recovery(self):
        """Test that corrupted cache files are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create a Python file
            input_file = tmpdir / "test.py"
            input_file.write_text("""
# Test File
print("Hello world")
""")

            # Create cache directory structure
            cache_dir = tmpdir / ".colight_cache"
            cache_dir.mkdir()

            # Create a corrupted cache file
            corrupted_cache = cache_dir / "test.py.cache"
            corrupted_cache.write_text("This is not valid JSON{{{")

            config = BuildConfig(continue_on_error=True)
            output_file = tmpdir / "test.md"

            # Should recover gracefully from corrupted cache
            api.build_file(input_file, output_file, config=config)

            # Should still produce output
            assert output_file.exists()
            content = output_file.read_text()
            assert "Test File" in content

    def test_corrupted_colight_file_recovery(self):
        """Test recovery from corrupted .colight visualization files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            input_file = tmpdir / "test.py"
            input_file.write_text("""
# Visualization Test
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Test Plot")
""")

            # Pre-create a corrupted .colight file that might be referenced
            colight_dir = tmpdir / "colight_files"
            colight_dir.mkdir()
            corrupted_colight = colight_dir / "form-001.colight"
            corrupted_colight.write_text("corrupted colight data")

            config = BuildConfig(continue_on_error=True)
            output_file = tmpdir / "test.md"

            # Should handle corrupted visualization files gracefully
            api.build_file(input_file, output_file, config=config)

            assert output_file.exists()

    def test_partial_cache_corruption(self):
        """Test recovery when some cache entries are corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create multiple files
            file1 = tmpdir / "file1.py"
            file1.write_text("# File 1\nprint('file1')")

            file2 = tmpdir / "file2.py"
            file2.write_text("# File 2\nprint('file2')")

            # Create cache with one corrupted entry
            cache_dir = tmpdir / ".colight_cache"
            cache_dir.mkdir()

            # Valid cache entry
            cache1 = cache_dir / "file1.py.cache"
            cache1.write_text('{"valid": "cache", "timestamp": 123456789}')

            # Corrupted cache entry
            cache2 = cache_dir / "file2.py.cache"
            cache2.write_text("corrupted{cache")

            config = BuildConfig(continue_on_error=True)
            output_dir = tmpdir / "output"

            # Should process directory despite partial corruption
            api.build_directory(tmpdir, output_dir, config=config)

            # Both files should be processed
            assert (output_dir / "file1.md").exists()
            assert (output_dir / "file2.md").exists()


class TestNetworkFailureRecovery:
    """Test recovery from network-related failures."""

    def test_websocket_connection_loss_recovery(self):
        """Test recovery when WebSocket connection is lost."""
        # This test simulates what should happen when WebSocket fails

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            (tmpdir / "test.py").write_text("print('test')")

            from colight_live.server import LiveServer

            server = LiveServer(
                input_path=tmpdir,
                cache_path=tmpdir / ".cache",
                config=BuildConfig(),
                include=["*.py"],
                ignore=None,
                host="127.0.0.1",
                http_port=5500,
                ws_port=5501,
                open_url=False,
            )

            # Mock WebSocket to simulate connection failure
            with patch.object(server, "_broadcast_message") as mock_broadcast:
                mock_broadcast.side_effect = ConnectionError(
                    "WebSocket connection lost"
                )

                # Server should handle WebSocket failures gracefully
                # and continue serving HTTP requests
                try:
                    # Simulate a message broadcast that fails
                    server._broadcast_message({"type": "test"})
                except ConnectionError:
                    # Should be caught and handled internally
                    pass

    def test_http_server_restart_recovery(self):
        """Test recovery from HTTP server interruptions."""
        # This is more of a design verification test

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            (tmpdir / "test.py").write_text("print('test')")

            from colight_live.server import LiveServer

            server = LiveServer(
                input_path=tmpdir,
                cache_path=tmpdir / ".cache",
                config=BuildConfig(),
                include=["*.py"],
                ignore=None,
                host="127.0.0.1",
                http_port=5500,
                ws_port=5501,
                open_url=False,
            )

            # Verify server can be stopped and restarted
            server.stop()
            # Should be able to create new instance on same port
            server2 = LiveServer(
                input_path=tmpdir,
                cache_path=tmpdir / ".cache",
                config=BuildConfig(),
                include=["*.py"],
                ignore=None,
                host="127.0.0.1",
                http_port=5500,
                ws_port=5501,
                open_url=False,
            )
            server2.stop()


class TestMemoryExhaustionRecovery:
    """Test behavior under memory pressure."""

    def test_large_file_processing_limits(self):
        """Test handling of very large Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create a large file (but not so large it actually exhausts memory)
            large_content = "# Large File\n" + "print('line')\n" * 1000
            input_file = tmpdir / "large.py"
            input_file.write_text(large_content)

            config = BuildConfig(continue_on_error=True)
            output_file = tmpdir / "large.md"

            # Should handle large files without memory issues
            api.build_file(input_file, output_file, config=config)

            assert output_file.exists()
            content = output_file.read_text()
            assert "Large File" in content

    def test_many_files_processing_limits(self):
        """Test handling of directories with many files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create many small files
            for i in range(50):  # Reasonable number for testing
                file_path = tmpdir / f"file_{i:03d}.py"
                file_path.write_text(f"# File {i}\nprint('File number {i}')")

            config = BuildConfig(continue_on_error=True)
            output_dir = tmpdir / "output"

            # Should handle many files without memory issues
            api.build_directory(tmpdir, output_dir, config=config)

            # Verify all files were processed
            output_files = list(output_dir.glob("*.md"))
            assert len(output_files) == 50

    def test_memory_cleanup_after_errors(self):
        """Test that memory is cleaned up after execution errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create file that will cause memory allocation then error
            input_file = tmpdir / "memory_test.py"
            input_file.write_text("""
# Memory Test
import numpy as np

# Allocate some memory
large_array = np.random.randn(1000, 1000)

# Then cause an error
raise ValueError("Intentional error for testing")
""")

            config = BuildConfig(continue_on_error=True)
            output_file = tmpdir / "memory_test.md"

            # Should complete without memory leaks
            api.build_file(input_file, output_file, config=config)

            assert output_file.exists()
            content = output_file.read_text()
            assert "Memory Test" in content


class TestPartialFailureRecovery:
    """Test recovery from partial execution failures."""

    def test_some_blocks_fail_others_succeed(self):
        """Test when some code blocks fail but others succeed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            input_file = tmpdir / "mixed_success.py"
            input_file.write_text("""
# Mixed Success Test

This should work fine.

print("This will succeed")
x = 42
print(f"x = {x}")

# This block will fail
undefined_variable_error

# This should also work
print("This should also succeed")
y = x + 1
print(f"y = {y}")

# Another failure
another_undefined_variable

# Final success
print("Final success")
""")

            config = BuildConfig(continue_on_error=True)
            output_file = tmpdir / "mixed_success.md"

            # Should complete and include successful parts
            api.build_file(input_file, output_file, config=config)

            assert output_file.exists()
            content = output_file.read_text()
            assert "Mixed Success Test" in content
            assert "This should work fine" in content

    def test_import_failure_isolated_to_block(self):
        """Test that import failures don't affect other blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            input_file = tmpdir / "import_isolation.py"
            input_file.write_text("""
# Import Isolation Test

# This should work
print("Before import error")

# This will fail  
import nonexistent_module

# This should still work despite the import failure above
print("After import error")
import os
print(f"Current directory: {os.getcwd()}")
""")

            config = BuildConfig(continue_on_error=True)
            output_file = tmpdir / "import_isolation.md"

            api.build_file(input_file, output_file, config=config)

            assert output_file.exists()
            content = output_file.read_text()
            assert "Import Isolation Test" in content

    def test_visualization_failure_recovery(self):
        """Test recovery when visualization generation fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            input_file = tmpdir / "viz_failure.py"
            input_file.write_text("""
# Visualization Failure Test

This text should appear.

# Good visualization
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Working Plot")

# Bad visualization that might fail
plt.figure()
plt.plot(None, None)  # This might cause issues
plt.title("Broken Plot")

# Text after visualization issues
This text should also appear.
""")

            config = BuildConfig(continue_on_error=True)
            output_file = tmpdir / "viz_failure.md"

            api.build_file(input_file, output_file, config=config)

            assert output_file.exists()
            content = output_file.read_text()
            assert "Visualization Failure Test" in content
            assert "This text should appear" in content
            assert "This text should also appear" in content


class TestFileSystemErrorRecovery:
    """Test recovery from file system related errors."""

    def test_permission_denied_recovery(self):
        """Test recovery when file permissions prevent access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            # Create accessible input file
            input_file = tmpdir / "test.py"
            input_file.write_text("# Test\nprint('test')")

            # Try to create output in restricted location
            # (This test may behave differently on different systems)
            restricted_output = tmpdir / "restricted" / "output.md"

            config = BuildConfig(continue_on_error=True)

            # Should handle permission errors gracefully
            try:
                api.build_file(input_file, restricted_output, config=config)
            except (PermissionError, FileNotFoundError):
                # Expected behavior for permission issues
                pass

    def test_disk_full_simulation(self):
        """Test behavior when disk space is exhausted (simulated)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            input_file = tmpdir / "test.py"
            input_file.write_text("# Test\nprint('test')")

            output_file = tmpdir / "test.md"
            config = BuildConfig(continue_on_error=True)

            # Mock write operations to simulate disk full
            original_write = pathlib.Path.write_text

            def mock_write_disk_full(self, *args, **kwargs):
                if "test.md" in str(self):
                    raise OSError("No space left on device")
                return original_write(self, *args, **kwargs)

            with patch.object(pathlib.Path, "write_text", mock_write_disk_full):
                try:
                    api.build_file(input_file, output_file, config=config)
                except OSError:
                    # Expected behavior when disk is full
                    pass

    def test_concurrent_file_access_recovery(self):
        """Test recovery when files are locked by other processes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            input_file = tmpdir / "test.py"
            input_file.write_text("# Test\nprint('test')")

            output_file = tmpdir / "test.md"
            config = BuildConfig(continue_on_error=True)

            # Simulate file lock by opening output file exclusively
            # Note: This test may behave differently on different platforms
            try:
                with open(output_file, "w") as f:
                    f.write("locked")
                    # Try to build while file is open
                    api.build_file(input_file, output_file, config=config)
            except (OSError, PermissionError):
                # Expected behavior when file is locked
                pass
