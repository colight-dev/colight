"""Test that block IDs are consistent between manifest and results."""

import pathlib
import tempfile

from colight_prose.incremental_executor import IncrementalExecutor
from colight_prose.json_api import JsonDocumentGenerator
from colight_prose.parser import parse_colight_file


def test_block_id_format():
    """Test that block IDs match between manifest and execution results."""

    content = """# %%
x = 1

# %%
y = 2

# %%
z = x + y
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        source_path = pathlib.Path(f.name)

        # Parse document
        document = parse_colight_file(source_path)

        # Get block IDs (which are now cache keys)
        block_ids = [block.id for block in document.blocks]

        # Execute using JSON generator
        executor = IncrementalExecutor()
        generator = JsonDocumentGenerator(incremental_executor=executor)

        # Collect block IDs from execution
        result_ids = []
        for block_id, result in generator.execute_incremental_with_results(source_path):
            result_ids.append(block_id)

        # Verify they match
        assert len(block_ids) == len(
            result_ids
        ), f"Block count mismatch: {len(block_ids)} vs {len(result_ids)}"

        # Block IDs should be non-empty strings (cache keys)
        for block_id in result_ids:
            assert isinstance(
                block_id, str
            ), f"Block ID should be string, got {type(block_id)}"
            assert len(block_id) > 0, "Block ID should not be empty"

        # The block IDs from parsing and execution should match
        assert (
            block_ids == result_ids
        ), f"Block IDs don't match: {block_ids} vs {result_ids}"

        # Clean up
        source_path.unlink()


if __name__ == "__main__":
    test_block_id_format()
    print("Block ID consistency test passed!")
