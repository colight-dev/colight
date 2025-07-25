"""Test performance improvements from Phase 3.5 optimizations."""

import pathlib
import tempfile
import time

from colight_prose.file_graph import FileDependencyGraph


def test_external_imports_not_analyzed():
    """Test that external imports are not analyzed, improving performance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = pathlib.Path(tmpdir)

        # Create a file with many external imports
        test_file = base / "heavy_imports.py"
        test_file.write_text("""
# Standard library imports
import os
import sys
import json
import pathlib
import collections
import itertools
import functools
import typing
import datetime
import asyncio

# Common third-party that might be installed
try:
    import numpy
    import pandas
    import matplotlib
    import requests
    import flask
    import django
except ImportError:
    pass

# Local import
import config
""")

        # Also create the local file
        (base / "config.py").write_text("DEBUG = True")

        graph = FileDependencyGraph(base)

        # Time the analysis
        start = time.time()
        imports = graph.analyze_file(test_file)
        elapsed = time.time() - start

        # Should only find the local import
        assert imports == {"config.py"}

        # Should be very fast since we skip external imports
        # (This is a soft assertion - mainly for visibility)
        print(f"Analysis took {elapsed:.4f} seconds")
        assert elapsed < 0.1  # Should complete in under 100ms


def test_project_scan_performance():
    """Test that scanning a project directory is reasonably fast."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = pathlib.Path(tmpdir)

        # Create a small project structure
        files = [
            "main.py",
            "config.py",
            "utils.py",
            "models/user.py",
            "models/product.py",
            "models/__init__.py",
            "views/home.py",
            "views/api.py",
            "views/__init__.py",
        ]

        # Create directories
        (base / "models").mkdir()
        (base / "views").mkdir()

        # Create files with imports
        (base / "main.py").write_text("""
import config
import utils
from models import user, product
from views import home, api
""")

        (base / "config.py").write_text("DEBUG = True")

        (base / "utils.py").write_text("""
import config
def helper(): pass
""")

        (base / "models" / "__init__.py").write_text("")
        (base / "models" / "user.py").write_text("""
from ..utils import helper
class User: pass
""")

        (base / "models" / "product.py").write_text("""
from .user import User
class Product: pass
""")

        (base / "views" / "__init__.py").write_text("")
        (base / "views" / "home.py").write_text("""
from ..models.user import User
def index(): pass
""")

        (base / "views" / "api.py").write_text("""
from ..models import user, product
from ..utils import helper
def api_endpoint(): pass
""")

        graph = FileDependencyGraph(base)

        # Time the full directory scan
        start = time.time()
        graph.analyze_directory(base)
        elapsed = time.time() - start

        # Check we found all the files
        stats = graph.get_graph_stats()
        assert stats["total_files"] == 9  # All .py files
        assert stats["total_imports"] > 0

        # Should be fast for a small project
        print(
            f"Directory scan took {elapsed:.4f} seconds for {stats['total_files']} files"
        )
        assert elapsed < 0.5  # Should complete in under 500ms


def test_import_spec_fallback():
    """Test that imports that can't be resolved locally still work."""
    # Use fixture directory instead of temp directory
    base = pathlib.Path(__file__).parent / "import-test-fixtures" / "fallback-test"
    test_file = base / "test.py"

    graph = FileDependencyGraph(base)
    imports = graph.analyze_file(test_file)

    # Should only find config.py, missing module is ignored
    assert imports == {"config.py"}
