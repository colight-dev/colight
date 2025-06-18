"""
Colight file format writer.

The .colight format is a self-contained binary format inspired by PNG and SQLite:

Header Structure (96 bytes):
- Bytes 0-7:   Magic bytes "COLIGHT\x00"
- Bytes 8-15:  Version number (uint64, little-endian)
- Bytes 16-23: JSON section offset (uint64, little-endian)
- Bytes 24-31: JSON section length (uint64, little-endian)
- Bytes 32-39: Binary section offset (uint64, little-endian)
- Bytes 40-47: Binary section length (uint64, little-endian)
- Bytes 48-55: Number of buffers (uint64, little-endian)
- Bytes 56-95: Reserved for future use (40 bytes, zeroed)

After header:
- JSON section: Contains AST and metadata
- Binary section: Concatenated binary buffers with 8-byte alignment

Alignment guarantees:
- The binary section starts at an 8-byte aligned offset from the file beginning
- Each buffer within the binary section starts at an 8-byte aligned offset
- This ensures zero-copy typed array creation for all standard numeric types

The JSON includes buffer layout with offsets and lengths for each buffer.
Buffer references in the AST keep using the existing index system.
"""

import struct
import json
from typing import List, Dict, Any, Union
from pathlib import Path

# File format constants
MAGIC_BYTES = b"COLIGHT\x00"
CURRENT_VERSION = 1
HEADER_SIZE = 96


def create_bytes(
    json_data: Dict[str, Any], buffers: List[Union[bytes, bytearray, memoryview]]
) -> bytes:
    """
    Create the bytes for a .colight file.

    Args:
        json_data: The JSON data containing AST and metadata (with existing buffer indexes)
        buffers: List of binary buffers

    Returns:
        Complete file content as bytes
    """
    # Calculate buffer layout (offsets and lengths within binary section)
    buffer_offsets = []
    buffer_lengths = []
    current_offset = 0

    # Alignment requirement (8 bytes covers all typed arrays)
    ALIGNMENT = 8

    for buffer in buffers:
        # Ensure offset is aligned
        if current_offset % ALIGNMENT != 0:
            padding = ALIGNMENT - (current_offset % ALIGNMENT)
            current_offset += padding

        buffer_offsets.append(current_offset)
        buffer_length = len(buffer)
        buffer_lengths.append(buffer_length)
        current_offset += buffer_length

    # Add buffer layout to JSON data
    json_data_with_layout = json_data.copy()
    json_data_with_layout["bufferLayout"] = {
        "offsets": buffer_offsets,
        "lengths": buffer_lengths,
        "count": len(buffers),
        "totalSize": current_offset,
    }

    # Serialize JSON
    json_bytes = json.dumps(json_data_with_layout, separators=(",", ":")).encode(
        "utf-8"
    )

    # Calculate file layout with alignment
    json_offset = HEADER_SIZE
    json_length = len(json_bytes)

    # Ensure binary section starts at an 8-byte aligned offset
    unaligned_binary_offset = json_offset + json_length
    binary_offset = (unaligned_binary_offset + 7) & ~7  # Round up to 8-byte boundary
    json_padding = binary_offset - unaligned_binary_offset

    binary_length = current_offset
    num_buffers = len(buffers)

    # Create header
    header = bytearray(HEADER_SIZE)
    struct.pack_into("<8s", header, 0, MAGIC_BYTES)
    struct.pack_into("<Q", header, 8, CURRENT_VERSION)
    struct.pack_into("<Q", header, 16, json_offset)
    struct.pack_into("<Q", header, 24, json_length)
    struct.pack_into("<Q", header, 32, binary_offset)
    struct.pack_into("<Q", header, 40, binary_length)
    struct.pack_into("<Q", header, 48, num_buffers)
    # Bytes 56-95 remain zeroed (reserved)

    # Combine all sections
    result = bytearray()
    result.extend(header)
    result.extend(json_bytes)
    result.extend(b"\x00" * json_padding)  # Padding after JSON to align binary section

    # Write buffers with alignment padding
    written_offset = 0
    for i, buffer in enumerate(buffers):
        # Add padding if needed
        expected_offset = buffer_offsets[i]
        if written_offset < expected_offset:
            padding_size = expected_offset - written_offset
            result.extend(b"\x00" * padding_size)
            written_offset = expected_offset

        result.extend(buffer)
        written_offset += len(buffer)

    return bytes(result)


def create_file(
    json_data: Dict[str, Any],
    buffers: List[Union[bytes, bytearray, memoryview]],
    output_path: Union[str, Path],
) -> str:
    """
    Create a .colight file.

    Args:
        json_data: The JSON data containing AST and metadata (with existing buffer indexes)
        buffers: List of binary buffers
        output_path: Path to write the file

    Returns:
        Path to the created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use create_bytes to generate the complete file content
    file_content = create_bytes(json_data, buffers)

    # Write file
    with open(output_path, "wb") as f:
        f.write(file_content)

    return str(output_path)


def parse_file(file_path: Union[str, Path]) -> tuple[Dict[str, Any], List[bytes]]:
    """
    Parse a .colight file and return JSON data and buffers (for testing).

    Args:
        file_path: Path to the .colight file

    Returns:
        Tuple of (json_data, buffers_list)

    Raises:
        ValueError: If file format is invalid
    """
    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        # Read and validate header
        header = f.read(HEADER_SIZE)
        if len(header) != HEADER_SIZE:
            raise ValueError("Invalid .colight file: Header too short")

        # Parse header
        magic = struct.unpack_from("<8s", header, 0)[0]
        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid .colight file: Wrong magic bytes {magic}")

        version = struct.unpack_from("<Q", header, 8)[0]
        if version > CURRENT_VERSION:
            raise ValueError(f"Unsupported .colight file version: {version}")

        json_offset = struct.unpack_from("<Q", header, 16)[0]
        json_length = struct.unpack_from("<Q", header, 24)[0]
        binary_offset = struct.unpack_from("<Q", header, 32)[0]
        binary_length = struct.unpack_from("<Q", header, 40)[0]
        num_buffers = struct.unpack_from("<Q", header, 48)[0]

        # Read JSON section
        f.seek(json_offset)
        json_bytes = f.read(json_length)
        if len(json_bytes) != json_length:
            raise ValueError("Invalid .colight file: JSON section truncated")

        json_data = json.loads(json_bytes.decode("utf-8"))

        # Read binary section
        f.seek(binary_offset)
        binary_data = f.read(binary_length)
        if len(binary_data) != binary_length:
            raise ValueError("Invalid .colight file: Binary section truncated")

    # Extract individual buffers based on buffer layout in JSON
    buffers = []
    buffer_layout = json_data.get("bufferLayout", {})
    buffer_offsets = buffer_layout.get("offsets", [])
    buffer_lengths = buffer_layout.get("lengths", [])

    if len(buffer_offsets) != num_buffers or len(buffer_lengths) != num_buffers:
        raise ValueError("Invalid .colight file: Buffer layout mismatch")

    for i in range(num_buffers):
        offset = buffer_offsets[i]
        length = buffer_lengths[i]
        if offset + length > binary_length:
            raise ValueError(
                f"Invalid .colight file: Buffer {i} extends beyond binary section"
            )
        buffer = binary_data[offset : offset + length]
        buffers.append(buffer)

    return json_data, buffers
