"""Tests for file-level dependency graph."""

import pathlib
import tempfile

import pytest

from colight_prose.file_graph import FileDependencyGraph


@pytest.fixture
def temp_project():
    """Create a temporary project structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = pathlib.Path(tmpdir)

        # Create a simple project structure
        # main.py imports utils and models
        (base / "main.py").write_text("""
import utils
from models import User
from helpers.formatter import format_text

def main():
    user = User("test")
    text = format_text("hello")
    utils.process(user, text)
""")

        # utils.py imports config
        (base / "utils.py").write_text("""
import config

def process(user, text):
    if config.DEBUG:
        print(f"Processing {user} with {text}")
""")

        # models.py has no imports
        (base / "models.py").write_text("""
class User:
    def __init__(self, name):
        self.name = name
""")

        # config.py has no imports
        (base / "config.py").write_text("""
DEBUG = True
API_KEY = "secret"
""")

        # Create helpers package
        (base / "helpers").mkdir()
        (base / "helpers" / "__init__.py").write_text("")
        (base / "helpers" / "formatter.py").write_text("""
from ..config import DEBUG

def format_text(text):
    if DEBUG:
        return f"[DEBUG] {text}"
    return text
""")

        yield base


def test_basic_import_analysis(temp_project):
    """Test basic import analysis."""
    graph = FileDependencyGraph(temp_project)

    # Analyze main.py
    main_imports = graph.analyze_file(temp_project / "main.py")
    assert "utils.py" in main_imports
    assert "models.py" in main_imports
    assert "helpers/formatter.py" in main_imports

    # Analyze utils.py
    utils_imports = graph.analyze_file(temp_project / "utils.py")
    assert "config.py" in utils_imports

    # Analyze models.py (no imports)
    models_imports = graph.analyze_file(temp_project / "models.py")
    assert len(models_imports) == 0


def test_relative_imports(temp_project):
    """Test relative import resolution."""
    graph = FileDependencyGraph(temp_project)

    # Analyze formatter.py which has relative imports
    formatter_imports = graph.analyze_file(temp_project / "helpers" / "formatter.py")
    assert "config.py" in formatter_imports


def test_affected_files(temp_project):
    """Test finding files affected by changes."""
    graph = FileDependencyGraph(temp_project)

    # Analyze all files first
    graph.analyze_directory(temp_project)

    # Changes to config.py affect utils.py and formatter.py
    affected = graph.get_affected_files("config.py")
    assert "config.py" in affected
    assert "utils.py" in affected
    assert "helpers/formatter.py" in affected

    # Changes to utils.py affect main.py
    affected = graph.get_affected_files("utils.py")
    assert "utils.py" in affected
    assert "main.py" in affected

    # Changes to models.py affect main.py
    affected = graph.get_affected_files("models.py")
    assert "models.py" in affected
    assert "main.py" in affected


def test_transitive_dependencies(temp_project):
    """Test transitive dependency detection."""
    graph = FileDependencyGraph(temp_project)

    # Analyze all files
    graph.analyze_directory(temp_project)

    # Changes to config.py should transitively affect main.py
    # config.py -> utils.py -> main.py
    # config.py -> formatter.py -> main.py
    affected = graph.get_affected_files("config.py")
    assert "main.py" in affected


def test_circular_imports(temp_project):
    """Test handling of circular imports."""
    # Create circular import
    (temp_project / "circular_a.py").write_text("""
from circular_b import func_b

def func_a():
    return func_b()
""")

    (temp_project / "circular_b.py").write_text("""
from circular_a import func_a

def func_b():
    return "b"
""")

    graph = FileDependencyGraph(temp_project)
    graph.analyze_file(temp_project / "circular_a.py")
    graph.analyze_file(temp_project / "circular_b.py")

    # Both files should be in each other's affected set
    affected_a = graph.get_affected_files("circular_a.py")
    assert "circular_b.py" in affected_a

    affected_b = graph.get_affected_files("circular_b.py")
    assert "circular_a.py" in affected_b


def test_cache_invalidation(temp_project):
    """Test that cache is invalidated when files change."""
    graph = FileDependencyGraph(temp_project)

    # Initial analysis
    imports1 = graph.analyze_file(temp_project / "main.py")
    assert "utils.py" in imports1

    # Modify the file
    (temp_project / "main.py").write_text("""
import utils
import config  # New import

def main():
    utils.process()
""")

    # Re-analyze - should pick up new import
    imports2 = graph.analyze_file(temp_project / "main.py")
    assert "utils.py" in imports2
    assert "config.py" in imports2


def test_external_imports_ignored(temp_project):
    """Test that external imports are ignored."""
    (temp_project / "external_test.py").write_text("""
import os
import sys
import json
from pathlib import Path
import numpy as np
import requests

import config  # Local import
""")

    graph = FileDependencyGraph(temp_project)
    imports = graph.analyze_file(temp_project / "external_test.py")

    # Only local import should be tracked
    assert imports == {"config.py"}


def test_import_error_handling(temp_project):
    """Test handling of import errors."""
    # Create file with syntax error
    (temp_project / "syntax_error.py").write_text("""
import config
def broken(
    # Missing closing paren
""")

    graph = FileDependencyGraph(temp_project)
    # Should not crash, just return empty set
    imports = graph.analyze_file(temp_project / "syntax_error.py")
    assert imports == set()


def test_nonexistent_imports(temp_project):
    """Test handling of imports that don't resolve to files."""
    (temp_project / "missing_imports.py").write_text("""
import nonexistent_module
from another_missing import something
""")

    graph = FileDependencyGraph(temp_project)
    imports = graph.analyze_file(temp_project / "missing_imports.py")
    # Should be empty since imports don't resolve
    assert imports == set()


def test_graph_stats(temp_project):
    """Test graph statistics."""
    graph = FileDependencyGraph(temp_project)
    graph.analyze_directory(temp_project)

    stats = graph.get_graph_stats()
    assert stats["total_files"] > 0
    assert stats["files_with_imports"] > 0
    assert stats["files_imported"] > 0
    assert stats["total_imports"] > 0


def test_hidden_directories_skipped(temp_project):
    """Test that hidden directories are skipped during analysis."""
    # Create hidden directory with Python files
    hidden_dir = temp_project / ".venv"
    hidden_dir.mkdir()
    (hidden_dir / "test.py").write_text("""
import sys
print("This should not be analyzed")
""")

    # Create another hidden directory
    git_dir = temp_project / ".git"
    git_dir.mkdir()
    (git_dir / "hooks" / "pre-commit").mkdir(parents=True)
    (git_dir / "hooks" / "pre-commit" / "check.py").write_text("""
import os
""")

    # Create a normal file that imports from hidden dir (should fail)
    (temp_project / "bad_import.py").write_text("""
try:
    from .venv import test  # This import won't resolve
except ImportError:
    pass
""")

    graph = FileDependencyGraph(temp_project)
    graph.analyze_directory(temp_project)

    # Get all analyzed files
    stats = graph.get_graph_stats()
    all_files = set(graph.imports.keys()) | set(graph.imported_by.keys())

    # Verify no hidden directory files were analyzed
    for file_path in all_files:
        assert not any(
            part.startswith(".") for part in pathlib.Path(file_path).parts
        ), f"Hidden file {file_path} should not be in graph"

    # The bad_import.py should be analyzed but have no valid imports
    bad_imports = graph.get_dependencies("bad_import.py")
    assert len(bad_imports) == 0


def test_import_resolution_with_different_watched_dir():
    """Test that imports are resolved correctly when watched dir != Python project root.

    This reproduces the issue where watching packages/colight-prose/tests/examples/
    doesn't recognize that visual_update.py depends on visual_update_dep.py.
    """
    # Use the real examples directory
    current_file = pathlib.Path(__file__)
    examples_dir = current_file.parent / "examples"

    # Skip test if examples directory doesn't exist
    if not examples_dir.exists():
        pytest.skip("examples directory not found")

    # Ensure the dependency files exist
    dep_file = examples_dir / "visual_update_dep.py"
    main_file = examples_dir / "visual_update.py"

    if not dep_file.exists() or not main_file.exists():
        pytest.skip("Required test files not found in examples directory")

    # Create dependency graph watching only the examples directory
    graph = FileDependencyGraph(examples_dir)

    # Analyze the main file
    deps = graph.analyze_file(main_file)

    # The main file should depend on visual_update_dep.py
    expected_dep = "visual_update_dep.py"
    assert expected_dep in deps, f"Expected {expected_dep} in dependencies, got {deps}"

    # Check that changes to dep file affect main file
    affected = graph.get_affected_files("visual_update_dep.py")
    assert (
        "visual_update.py" in affected
    ), f"Expected visual_update.py to be affected, got {affected}"
