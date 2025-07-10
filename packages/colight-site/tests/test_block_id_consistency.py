"""Test that block IDs are consistent between manifest and results."""

import hashlib
import pathlib
import tempfile

from colight_live.incremental_executor import IncrementalExecutor
from colight_live.json_generator import JsonFormGenerator
from colight_site.builder import BuildConfig
from colight_site.parser import parse_colight_file


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

        # Generate file hash (same as server does)
        file_hash = hashlib.sha256(str(source_path).encode()).hexdigest()[:6]

        # Generate block IDs as server does
        block_ids = []
        for i, block in enumerate(document.blocks):
            unique_id = block.id if block.id != 0 else i
            stable_id = f"{file_hash}-B{unique_id:05d}"
            block_ids.append(stable_id)

        # Execute using JSON generator
        config = BuildConfig()
        executor = IncrementalExecutor()
        generator = JsonFormGenerator(config=config, incremental_executor=executor)

        # Collect block IDs from execution
        result_ids = []
        for block_id, result in generator.execute_incremental_with_results(source_path):
            result_ids.append(block_id)

        # Verify they match
        assert len(block_ids) == len(
            result_ids
        ), f"Block count mismatch: {len(block_ids)} vs {len(result_ids)}"

        for expected, actual in zip(block_ids, result_ids):
            assert (
                expected == actual
            ), f"Block ID mismatch: expected {expected}, got {actual}"

        # Clean up
        source_path.unlink()


if __name__ == "__main__":
    test_block_id_format()
    print("Block ID consistency test passed!")
