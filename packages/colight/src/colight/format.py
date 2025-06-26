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

For updates: Multiple complete .colight entries can be appended to a file.
The parser reads entries sequentially until EOF.
"""

import struct
import json
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from colight.widget import to_json_with_state

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
    if buffers:  # Only add buffer layout if there are buffers
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

    # Calculate layout
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
    Create a .colight file with initial state.

    Args:
        json_data: The JSON data containing AST and metadata
        buffers: List of binary buffers
        output_path: Path to write the file

    Returns:
        Path to the created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        file_content = create_bytes(json_data, buffers)
        f.write(file_content)

    return str(output_path)


def parse_entry(f, offset: int = 0) -> tuple[Dict[str, Any], List[bytes], int]:
    """
    Parse a single entry from the file.

    Returns:
        Tuple of (json_data, buffers, entry_size)
    """
    f.seek(offset)

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
    f.seek(offset + json_offset)
    json_bytes = f.read(json_length)
    if len(json_bytes) != json_length:
        raise ValueError("Invalid .colight file: JSON section truncated")

    json_data = json.loads(json_bytes.decode("utf-8"))

    # Read binary section
    buffers = []
    if binary_length > 0:
        f.seek(offset + binary_offset)
        binary_data = f.read(binary_length)
        if len(binary_data) != binary_length:
            raise ValueError("Invalid .colight file: Binary section truncated")

        # Extract individual buffers based on buffer layout in JSON
        buffer_layout = json_data.get("bufferLayout", {})
        buffer_offsets = buffer_layout.get("offsets", [])
        buffer_lengths = buffer_layout.get("lengths", [])

        if len(buffer_offsets) != num_buffers or len(buffer_lengths) != num_buffers:
            raise ValueError("Invalid .colight file: Buffer layout mismatch")

        for i in range(num_buffers):
            offset_in_binary = buffer_offsets[i]
            length = buffer_lengths[i]
            if offset_in_binary + length > binary_length:
                raise ValueError(
                    f"Invalid .colight file: Buffer {i} extends beyond binary section"
                )
            buffer = binary_data[offset_in_binary : offset_in_binary + length]
            buffers.append(buffer)

    # Calculate total entry size
    entry_size = binary_offset + binary_length

    return json_data, buffers, entry_size


def parse_file(
    file_path: Union[str, Path],
) -> tuple[
    Optional[Dict[str, Any]], List[bytes], List[List[Union[Dict[str, Any], List[Any]]]]
]:
    """
    Parse a .colight file and return all entries.

    Args:
        file_path: Path to the .colight file

    Returns:
        Tuple of (initial_json_data, initial_buffers, updates_list)
        If file contains only updates, initial_json_data will be None

    Raises:
        ValueError: If file format is invalid
    """
    file_path = Path(file_path)
    file_size = file_path.stat().st_size

    initial_data = None
    initial_buffers = []
    updates = []

    with open(file_path, "rb") as f:
        offset = 0
        first_entry = True

        while offset < file_size:
            try:
                json_data, buffers, entry_size = parse_entry(f, offset)

                if first_entry and "updates" not in json_data:
                    # First entry without updates is the initial state
                    initial_data = json_data
                    initial_buffers = buffers
                else:
                    # Entry with updates field is an update entry
                    if "updates" in json_data:
                        updates.append(json_data["updates"])

                first_entry = False
                offset += entry_size
            except Exception:
                # If we can't parse an entry, we've reached the end
                break

    return initial_data, initial_buffers, updates


def append_updates(
    file_path: Union[str, Path],
    update_items: List[Any],
) -> str:
    """
    Append multiple updates to an existing .colight file.

    Args:
        file_path: Path to the existing .colight file
        update_items: List of LayoutItems or objects to serialize as updates

    Returns:
        Path to the updated file
    """
    file_path = Path(file_path)

    for update_item in update_items:
        append_update(file_path, update_item)

    return str(file_path)


def save_updates(
    output_path: Union[str, Path],
    update_items: List[Any],
) -> str:
    """
    Save updates to a new .colight file (without initial state).

    Args:
        update_items: List of LayoutItems or objects to serialize as updates
        output_path: Path to write the file

    Returns:
        Path to the created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create empty file
    with open(output_path, "wb"):
        pass

    # Append updates
    return append_updates(output_path, update_items)


def append_update(
    file_path: Union[str, Path],
    update_item: Any,
) -> str:
    """
    Append a single update to an existing .colight file.

    Args:
        file_path: Path to the existing .colight file
        update_item: A LayoutItem or any object that can be serialized to create an update

    Returns:
        Path to the updated file
    """

    # Serialize the update item to get JSON and buffers
    update_json, update_buffers = to_json_with_state(update_item)

    # Wrap the update data
    wrapped_json = {"updates": update_json}

    # Create the update entry with its buffers
    update_content = create_bytes(wrapped_json, update_buffers)

    file_path = Path(file_path)

    # Append to the file
    with open(file_path, "ab") as f:
        f.write(update_content)

    return str(file_path)
