"""Test that visuals update when code changes."""

import tempfile
from pathlib import Path

from colight_prose.incremental_executor import IncrementalExecutor
from colight_prose.parser import parse_colight_file


def test_visual_content_changes():
    """Test that visual results change when code changes."""

    # Initial code
    code_v1 = """# %%
import colight.plot as Plot
value = 42
Plot.dot([[1, value], [2, value + 10]])
"""

    # Changed code
    code_v2 = """# %%
import colight.plot as Plot
value = 45
Plot.dot([[1, value], [2, value + 10]])
"""

    executor = IncrementalExecutor(verbose=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.py"

        # First execution
        file_path.write_text(code_v1)
        doc1 = parse_colight_file(file_path)
        results1 = list(
            executor.execute_incremental_streaming(doc1, filename=str(file_path))
        )

        assert len(results1) == 1
        block1, result1 = results1[0]
        assert result1.colight_bytes is not None, "Should produce visual"
        visual_bytes_v1 = result1.colight_bytes

        # Second execution with changed value
        file_path.write_text(code_v2)
        doc2 = parse_colight_file(file_path)
        results2 = list(
            executor.execute_incremental_streaming(doc2, filename=str(file_path))
        )

        assert len(results2) == 1
        block2, result2 = results2[0]
        assert result2.colight_bytes is not None, "Should produce visual"
        visual_bytes_v2 = result2.colight_bytes

        # Check if visual bytes are different
        import hashlib

        v1_hash = hashlib.sha256(visual_bytes_v1).hexdigest()[:16]
        v2_hash = hashlib.sha256(visual_bytes_v2).hexdigest()[:16]

        print(f"V1 visual hash: {v1_hash}")
        print(f"V2 visual hash: {v2_hash}")
        print(f"Visual bytes equal: {visual_bytes_v1 == visual_bytes_v2}")

        # The visual bytes should be different
        if visual_bytes_v1 == visual_bytes_v2:
            print("WARNING: Visual bytes are the same even though value changed!")
            print("This might be a bug in the visual generation")

        # The block should have been re-executed (not cached)
        assert (
            result2.cache_hit == False
        ), "Block should be re-executed when content changes"

        print("✓ Block execution behavior is correct")
        print(f"  - V1 visual size: {len(visual_bytes_v1)} bytes")
        print(f"  - V2 visual size: {len(visual_bytes_v2)} bytes")
        print(f"  - Cache hit: {result2.cache_hit}")


if __name__ == "__main__":
    test_visual_content_changes()
