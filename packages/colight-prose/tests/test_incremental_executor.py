"""Tests for incremental execution."""

import pathlib
import tempfile

from colight_prose.incremental_executor import IncrementalExecutor
from colight_prose.parser import parse_colight_file


def test_basic_incremental_execution():
    """Test basic incremental execution."""
    executor = IncrementalExecutor()

    content = """
# %%
x = 1

# %%
y = x + 1

# %%
z = y * 2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        # Parse document
        doc = parse_colight_file(file_path)

        # Execute all blocks
        results = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        assert len(results) == 3

        # All should be cache misses on first run
        for block, result in results:
            assert result.cache_hit == False

        # Clean up
        file_path.unlink()


def test_independent_blocks():
    """Test that independent blocks can be cached separately."""
    executor = IncrementalExecutor()

    content = """
# %%
a = 1

# %%
b = 2

# %%
c = a + b
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        # Parse and execute
        doc = parse_colight_file(file_path)
        results1 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        # Execute again - should all be cache hits
        results2 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        for block, result in results2:
            assert result.cache_hit == True

        # Clean up
        file_path.unlink()


def test_always_eval_pragma_skip():
    """Test that blocks with always-eval pragma are never cached."""
    executor = IncrementalExecutor()

    content = """
# %% pragma: always-eval
import time
t = time.time()

# %%
x = 1
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        # Parse and execute
        doc = parse_colight_file(file_path)

        # Debug: Print block tags
        print(f"Block 0 tags: {doc.blocks[0].tags.flags}")
        print(f"Block 1 tags: {doc.blocks[1].tags.flags}")

        # Execute twice
        results1 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )
        results2 = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        # First block should never be cached due to always-eval
        assert results2[0][1].cache_hit == False
        # Second block should be cached
        assert results2[1][1].cache_hit == True

        # Clean up
        file_path.unlink()


def test_stdout_capture():
    """Test that stdout is properly captured."""
    executor = IncrementalExecutor()

    content = """
# %%
print("Hello, World!")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        doc = parse_colight_file(file_path)
        results = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        assert len(results) == 1
        assert results[0][1].output == "Hello, World!\n"

        # Clean up
        file_path.unlink()


def test_expression_results():
    """Test that expression results are captured."""
    executor = IncrementalExecutor()

    content = """
# %%
x = 42

# %%
x * 2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        doc = parse_colight_file(file_path)
        results = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        assert len(results) == 2
        # First block is statement, no value
        assert results[0][1].value is None
        # Second block is expression, should have value
        assert results[1][1].value == 84

        # Clean up
        file_path.unlink()


def test_error_handling():
    """Test that errors are properly captured."""
    executor = IncrementalExecutor()

    content = """
# %%
x = 1

# %%
y = undefined_var
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        file_path = pathlib.Path(f.name)

        doc = parse_colight_file(file_path)
        results = list(
            executor.execute_incremental_streaming(doc, str(file_path), str(file_path))
        )

        assert len(results) == 2
        # First block should succeed
        assert results[0][1].error is None
        # Second block should have error
        assert results[1][1].error is not None
        assert "undefined_var" in results[1][1].error

        # Clean up
        file_path.unlink()
