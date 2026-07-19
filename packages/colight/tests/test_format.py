import os
import struct
import tempfile

import colight.plot as Plot
import numpy as np
from colight.format import (
    HEADER_SIZE,
    MAGIC_BYTES,
    append_update,
    create_file,
    parse_file,
    parse_file_with_updates,
)
from colight.widget import to_json_with_state

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
    json_data, buffers, updates = parse_file(result_path)

    # Verify buffer layout
    assert json_data is not None
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


def test_alignment_artifact():
    """Multi-entry file whose first entry has an odd unpadded length.

    Regression for entry padding (format v2): before padding was introduced,
    the appended entry started at a misaligned absolute offset and the JS
    reader's zero-copy Float64Array construction threw a RangeError. The JS
    test format-alignment.test.js consumes the artifact written here.
    """
    test_artifacts_dir = os.path.join(os.path.dirname(__file__), "test-artifacts")
    os.makedirs(test_artifacts_dir, exist_ok=True)
    path = os.path.join(test_artifacts_dir, "test-alignment.colight")

    # First entry: single 3-byte uint8 buffer -> odd unpadded entry size.
    json_data, buffers = to_json_with_state(
        {"tiny": np.array([1, 2, 3], dtype=np.uint8)}
    )
    create_file(json_data, buffers, path)
    # Appended entry: float64 buffer, the dtype most sensitive to alignment.
    big = np.array([1.5, -2.5, 3.5, 4.5], dtype=np.float64)
    append_update(path, Plot.State({"big": big}))

    content = open(path, "rb").read()
    binary_offset, binary_length = struct.unpack_from("<2Q", content, 32)
    # The padding matters: the first entry's unpadded size is not 8-aligned...
    unpadded = binary_offset + binary_length
    assert unpadded % 8 != 0
    # ...but the second entry starts at the padded, 8-aligned offset.
    second_offset = (unpadded + 7) & ~7
    assert content[second_offset : second_offset + 8] == MAGIC_BYTES

    initial, _initial_buffers, updates = parse_file_with_updates(path)
    assert initial is not None
    envelope = updates[0]["data"]["state"]["big"]
    assert envelope["dtype"] == "float64"
    assert bytes(updates[0]["buffers"][envelope["__buffer_index__"]]) == big.tobytes()
