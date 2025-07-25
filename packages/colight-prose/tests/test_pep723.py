"""Tests for PEP 723 support."""

import os
import pathlib
import tempfile

from colight_prose.parser import parse_colight_file
from colight_prose.pep723 import detect_pep723_metadata, parse_dependencies
from colight_prose.static.builder import build_file


def test_detect_pep723_metadata():
    """Test detection of PEP 723 metadata."""
    # Valid PEP 723 metadata
    content = """# /// script
# dependencies = [
#   "requests",
#   "pandas>=1.0",
# ]
# ///

import requests
print("hello")
"""
    metadata = detect_pep723_metadata(content)
    assert metadata is not None
    assert "dependencies" in metadata
    assert "requests" in metadata
    assert "pandas>=1.0" in metadata


def test_detect_pep723_metadata_no_metadata():
    """Test detection when no PEP 723 metadata is present."""
    content = """# This is a regular Python file
import requests
print("hello")
"""
    metadata = detect_pep723_metadata(content)
    assert metadata is None


def test_detect_pep723_metadata_incomplete():
    """Test detection with incomplete PEP 723 metadata."""
    # Missing closing marker
    content = """# /// script
# dependencies = [
#   "requests",
# ]

import requests
"""
    metadata = detect_pep723_metadata(content)
    assert metadata is None


def test_parse_dependencies():
    """Test parsing dependencies from PEP 723 metadata."""
    metadata = """dependencies = [
  "requests",
  "pandas>=1.0",
  "numpy"
]"""
    deps = parse_dependencies(metadata)
    assert deps == ["requests", "pandas>=1.0", "numpy"]


def test_parse_dependencies_single_line():
    """Test parsing dependencies in single line format."""
    metadata = 'dependencies = ["requests", "pandas"]'
    deps = parse_dependencies(metadata)
    assert deps == ["requests", "pandas"]


def test_parse_dependencies_empty():
    """Test parsing empty dependencies."""
    metadata = "dependencies = []"
    deps = parse_dependencies(metadata)
    assert deps == []


def test_parse_dependencies_no_deps():
    """Test parsing metadata without dependencies."""
    metadata = "requires-python = '>=3.8'"
    deps = parse_dependencies(metadata)
    assert deps == []


def test_pep723_excluded_from_parser():
    """Test that PEP 723 metadata is excluded from parsed content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""# /// script
# dependencies = [
#   "requests",
# ]
# ///

# This is a regular comment
import requests

# Another comment
print("hello")
""")
        temp_path = pathlib.Path(f.name)

    try:
        document = parse_colight_file(temp_path)

        # Check that PEP 723 metadata doesn't appear in blocks
        all_prose = []
        for block in document.blocks:
            for element in block.elements:
                if element.kind == "PROSE":
                    all_prose.append(element.content)

        # Should not contain PEP 723 markers or dependencies
        assert not any("/// script" in prose for prose in all_prose)
        assert not any("dependencies" in prose for prose in all_prose)

        # Should contain the regular comments
        assert any("This is a regular comment" in prose for prose in all_prose)
        assert any("Another comment" in prose for prose in all_prose)

    finally:
        os.unlink(temp_path)


def test_build_pep723_file(tmp_path):
    """Test building a file with PEP 723 metadata."""
    # Create a test file with PEP 723 metadata
    test_file = tmp_path / "test_pep723.py"
    test_file.write_text("""# /// script
# dependencies = [
#   "tomli",
# ]
# ///

# Test PEP 723 file
import tomli

# Parse some TOML
data = tomli.loads('[tool.test]\\nvalue = 42')
data
""")

    output_file = tmp_path / "test_pep723.md"

    # Build the file with in_subprocess=True to prevent subprocess execution during tests
    build_file(test_file, output_file=output_file, verbose=False, in_subprocess=True)

    # Check output was created
    assert output_file.exists()

    # Read and verify content
    content = output_file.read_text()

    # Should not contain PEP 723 metadata in the prose
    # Note: it might appear in error tracebacks, which is fine
    # Check that it's not in the first part of the document
    prose_section = content.split("```")[0]  # Get content before first code block
    assert "/// script" not in prose_section
    assert "dependencies = [" not in prose_section

    # Should contain the actual content
    assert "Test PEP 723 file" in content
    assert "Parse some TOML" in content
    assert "import tomli" in content


def test_build_regular_file_not_affected(tmp_path):
    """Test that regular files without PEP 723 metadata work normally."""
    # Create a regular test file
    test_file = tmp_path / "test_regular.py"
    test_file.write_text("""# Regular Python file
import math

# Calculate something
result = math.pi * 2
result
""")

    output_file = tmp_path / "test_regular.md"

    # Build the file
    build_file(test_file, output_file=output_file, verbose=False)

    # Check output was created
    assert output_file.exists()

    # Read and verify content
    content = output_file.read_text()

    # Should contain the content
    assert "Regular Python file" in content
    assert "Calculate something" in content
    assert "import math" in content


def test_pep723_subprocess_execution(tmp_path, monkeypatch):
    """Test that PEP 723 files trigger subprocess execution."""
    # Create a test file with PEP 723 metadata
    test_file = tmp_path / "test_subprocess.py"
    test_file.write_text("""# /// script
# dependencies = [
#   "tomli",
# ]
# ///

import tomli
print("test")
""")

    output_file = tmp_path / "test_subprocess.md"

    # Track subprocess calls
    subprocess_calls = []
    original_run = __import__("subprocess").run

    def mock_run(cmd, **kwargs):
        subprocess_calls.append(cmd)

        # Simulate successful execution
        class MockResult:
            returncode = 0
            stderr = ""
            stdout = ""

        return MockResult()

    monkeypatch.setattr("subprocess.run", mock_run)

    # Build the file (should trigger subprocess)
    build_file(test_file, output_file=output_file, verbose=False)

    # Check that subprocess was called
    assert len(subprocess_calls) == 1
    cmd = subprocess_calls[0]

    # Verify the command structure
    assert cmd[0] == "uv"
    assert cmd[1] == "run"
    assert cmd[2] == "--with-editable"
    assert "--with" in cmd
    assert str(test_file) in cmd

    # Find the --with argument for the dependency
    with_idx = cmd.index("--with")
    assert cmd[with_idx + 1] == "tomli"

    # Check that colight-prose build is in the command
    assert "--" in cmd
    dash_idx = cmd.index("--")
    # After the --, should be python -m colight_cli build
    assert "-m" in cmd[dash_idx:]
    assert "colight_cli" in cmd[dash_idx:]
    assert "build" in cmd[dash_idx:]

    # Check that --in-subprocess flag is present
    assert "--in-subprocess" in cmd
