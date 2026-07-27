"""Test that relative imports work in executed blocks."""

import tempfile
from pathlib import Path

from colight_prose.executor import BlockExecutor
from colight_prose.parser import parse_document


def test_relative_imports_in_package():
    """Test that relative imports work when file is in a package."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a package structure
        pkg_dir = Path(tmpdir) / "mypackage"
        pkg_dir.mkdir()

        # Create __init__.py files
        (pkg_dir / "__init__.py").write_text("")

        # Create a module to import from
        (pkg_dir / "utils.py").write_text("""
def helper():
    return "Hello from utils"
""")

        # Create a subpackage
        sub_dir = pkg_dir / "sub"
        sub_dir.mkdir()
        (sub_dir / "__init__.py").write_text("")

        # Create a .py file with relative imports
        test_file = sub_dir / "example.py"
        test_file.write_text("""
# Test relative imports
from ..utils import helper
result = helper()
print(f"Got: {result}")
# Also test explicit relative import
from . import __init__
""")

        # Parse and execute
        with open(test_file) as f:
            doc = parse_document(f.read())

        executor = BlockExecutor()
        result = executor.execute_block(doc.blocks[0], str(test_file))

        # Should execute without errors
        assert result.error is None
        assert "Got: Hello from utils" in result.output

        # The imported function should be available
        assert "helper" in executor.env
        assert executor.env["helper"]() == "Hello from utils"


def test_relative_imports_no_package():
    """Test that files not in a package still work (no relative imports)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a standalone file
        test_file = Path(tmpdir) / "standalone.py"
        test_file.write_text("""
# No relative imports, just regular code
import os
print(f"Working dir: {os.getcwd()}")
x = 42
""")

        # Parse and execute
        with open(test_file) as f:
            doc = parse_document(f.read())

        executor = BlockExecutor()
        result = executor.execute_block(doc.blocks[0], str(test_file))

        # Should execute without errors
        assert result.error is None
        assert "Working dir:" in result.output
        assert executor.env.get("x") == 42


def test_nested_package_structure():
    """Test deeply nested package structures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested package structure: root/pkg/sub/deep/
        root = Path(tmpdir)
        pkg = root / "pkg"
        sub = pkg / "sub"
        deep = sub / "deep"

        for d in [pkg, sub, deep]:
            d.mkdir()
            (d / "__init__.py").write_text("")

        # Create modules at different levels
        (pkg / "top_utils.py").write_text('TOP = "top level"')
        (sub / "mid_utils.py").write_text('MID = "middle level"')

        # Create test file in deepest level
        test_file = deep / "test.py"
        test_file.write_text("""
# Import from various levels
from ...top_utils import TOP
from ..mid_utils import MID
from . import __init__
print(f"TOP: {TOP}")
print(f"MID: {MID}")
""")

        # Parse and execute
        with open(test_file) as f:
            doc = parse_document(f.read())

        executor = BlockExecutor()
        result = executor.execute_block(doc.blocks[0], str(test_file))

        # Should work with proper package detection
        assert result.error is None
        assert "TOP: top level" in result.output
        assert "MID: middle level" in result.output
