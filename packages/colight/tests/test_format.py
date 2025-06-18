import os
import tempfile

import colight.plot as Plot
import numpy as np
from colight.format import parse_file, MAGIC_BYTES, HEADER_SIZE

data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
p = Plot.raster(data)


def test_colight_file():
    """Test that export_colight creates a valid .colight file"""

    # Test without example file - also create in test-artifacts for JS tests
    test_artifacts_dir = os.path.join(os.path.dirname(__file__), "test-artifacts")
    os.makedirs(test_artifacts_dir, exist_ok=True)

    # Create test file in artifacts directory for JS tests to use
    artifact_path = os.path.join(test_artifacts_dir, "test-raster.colight")
    result_path = p.save_file(artifact_path)

    # Check that the file exists
    assert os.path.exists(result_path)

    # Test the new binary format
    with open(result_path, "rb") as f:
        content = f.read()

    # Check header
    assert len(content) >= HEADER_SIZE
    magic = content[:8]
    assert magic == MAGIC_BYTES

    # Parse using our parser
    json_data, buffers = parse_file(result_path)

    # Verify buffer layout
    assert "bufferLayout" in json_data
    assert "offsets" in json_data["bufferLayout"]
    assert "lengths" in json_data["bufferLayout"]
    assert "count" in json_data["bufferLayout"]
    assert "totalSize" in json_data["bufferLayout"]

    # Verify we have buffers
    assert len(buffers) > 0
    buffer_layout = json_data["bufferLayout"]
    assert len(buffers) == buffer_layout["count"]

    # Test with example file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test2.colight")
        colight_path = p.save_file(output_path)
        assert os.path.exists(colight_path)
