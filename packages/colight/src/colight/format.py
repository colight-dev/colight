"""
Colight file format writer.

The authoritative specification of the .colight format lives in
docs/src/colight_docs/format.md; keep this module and that document in sync.

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
- Every entry's total size is padded to a multiple of 8, so appended entries
  (and therefore every buffer's absolute file offset) stay 8-byte aligned
- This ensures zero-copy typed array creation for all standard numeric types

The JSON includes buffer layout with offsets and lengths for each buffer.
Buffer references in the AST keep using the existing index system.

For updates: Multiple complete .colight entries can be appended to a file.
The parser reads entries sequentially until EOF.

Streaming (see format.md section 4.1):
- A .colight file is an append-only stream. A producer appends whole entries and
  never rewrites what it already wrote.
- Use ColightWriter to hold the file open across many appends (the fast path for
  a long-running producer), or append_update/append_updates for the one-shot
  open-append-close form.
- Readers tolerate a torn tail: a reader that arrives mid-append sees every
  complete entry and silently drops the incomplete one, so a file may be read
  while it is still being written.
"""

import json
import os
import struct
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
from colight.widget import to_json_with_state

# File format constants
MAGIC_BYTES = b"COLIGHT\x00"
CURRENT_VERSION = 2
HEADER_SIZE = 96

__all__ = [
    # Constants
    "MAGIC_BYTES",
    "CURRENT_VERSION",
    "HEADER_SIZE",
    # Writing
    "align8",
    "create_bytes",
    "create_file",
    "create_update_bytes",
    # Appending (the streaming API)
    "ColightWriter",
    "append_update",
    "append_updates",
    "save_updates",
    # Reading
    "parse_entry",
    "parse_file",
    "parse_file_with_updates",
]


def align8(n: int) -> int:
    """Round ``n`` up to the next multiple of 8."""
    return (n + 7) & ~7


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
    binary_offset = align8(unaligned_binary_offset)
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

    # Pad the entry to an 8-byte boundary so that any appended entry (and
    # therefore every buffer's absolute file offset) stays 8-byte aligned.
    result.extend(b"\x00" * (align8(len(result)) - len(result)))

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
    if version != CURRENT_VERSION:
        raise ValueError(
            f"Unsupported .colight file version: found {version}, "
            f"this reader supports version {CURRENT_VERSION}"
        )

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

    # Total entry size, including the trailing padding that keeps the next
    # entry 8-byte aligned.
    entry_size = align8(binary_offset + binary_length)

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
                # A malformed first entry means the file itself is invalid:
                # surface the error (wrong magic, unsupported version, ...).
                if first_entry:
                    raise
                # After the first entry, a parse failure means we've reached
                # the end of the valid entries (e.g. a partially appended
                # update); stop reading.
                break

    return initial_data, initial_buffers, updates


def parse_file_with_updates(
    file_path: Union[str, Path],
) -> tuple[Optional[Dict[str, Any]], List[bytes], List[Dict[str, Any]]]:
    """
    Parse a .colight file and return update entries with buffers.

    Args:
        file_path: Path to the .colight file

    Returns:
        Tuple of (initial_json_data, initial_buffers, update_entries)
        update_entries is a list of {"data": <updates>, "buffers": <bytes[]>}
    """
    file_path = Path(file_path)
    file_size = file_path.stat().st_size

    initial_data = None
    initial_buffers: List[bytes] = []
    update_entries: List[Dict[str, Any]] = []

    with open(file_path, "rb") as f:
        offset = 0
        first_entry = True

        while offset < file_size:
            try:
                json_data, buffers, entry_size = parse_entry(f, offset)

                if first_entry and "updates" not in json_data:
                    initial_data = json_data
                    initial_buffers = buffers
                elif "updates" in json_data:
                    update_entries.append(
                        {"data": json_data["updates"], "buffers": buffers}
                    )

                first_entry = False
                offset += entry_size
            except Exception:
                if first_entry:
                    raise
                break

    return initial_data, initial_buffers, update_entries


def create_update_bytes(update_item: Any) -> bytes:
    """
    Serialize one update entry to bytes, ready to append to a .colight file.

    This is the unit of the streaming contract: the returned bytes are a whole,
    self-describing entry whose length is a multiple of 8, so appending them to
    a conforming file leaves the file conforming.

    Args:
        update_item: A LayoutItem, ``Plot.State({...})``, or any object the
            widget serializer accepts.

    Returns:
        The encoded update entry.
    """
    update_json, update_buffers = to_json_with_state(update_item)
    return create_bytes({"updates": update_json}, update_buffers)


def _check_appendable(file_path: Path) -> None:
    """
    Raise if appending to ``file_path`` would misalign the appended entry.

    Every buffer's absolute file offset must stay 8-byte aligned so readers can
    build zero-copy typed arrays over them (see the module docstring). A
    conforming writer always pads entries to a multiple of 8, so this can only
    fail on data of unknown provenance -- a truncated file, a hand-edited one.
    """
    if not file_path.exists():
        return
    size = file_path.stat().st_size
    if size % 8 != 0:
        raise ValueError(
            f"Cannot append to {file_path}: its length {size} is not a multiple "
            "of 8, so an appended entry's buffers would not be 8-byte aligned."
        )


class ColightWriter:
    """
    A .colight file held open across many appends.

    A long-running producer -- a simulation, a training loop, an instrument, an
    agent loop -- opens the artifact once and appends a state update per tick.
    This mirrors ``ColightFileWriter`` in the JavaScript ``@colight/format``
    package; the two implementations are kept deliberately parallel.

    Streaming contract:

    - **Append-only.** Entries are only ever added at the end; existing bytes
      are never rewritten. There is no seek, no in-place edit, no rewrite of
      the header.
    - **8-byte alignment is preserved.** Every entry is padded to a multiple of
      8, so each appended entry -- and therefore every buffer inside it --
      starts at an 8-byte aligned absolute file offset.
    - **Readers tolerate a torn tail.** A reader that opens the file while an
      entry is half-written stops at that entry and returns everything before
      it, without error (see :func:`parse_file_with_updates`). Concurrent
      reading is therefore safe at any moment: a reader sees a monotonically
      growing, never-corrupt prefix.

    Durability: an :meth:`append` reaches the OS immediately (each entry is one
    ``write`` call to an unbuffered file object), so another process reading the
    file sees it right away. It is not on stable storage until :meth:`flush` --
    which calls ``os.fsync`` -- or :meth:`close` returns. If the process is
    killed mid-``append`` the file ends in a torn entry, which readers already
    discard; nothing earlier is damaged.

    Example:
        >>> with ColightWriter.create("run.colight", Plot.State({"t": 0})) as w:
        ...     for t in range(1, 100):
        ...         w.append(Plot.State({"t": t}))

    Args:
        path: Path to the artifact being written.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._file = open(self.path, "ab", buffering=0)

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        initial_item: Optional[Any] = None,
    ) -> "ColightWriter":
        """
        Create (or truncate) ``path`` and open it for appending.

        Args:
            path: Path to write. Parent directories are created.
            initial_item: An optional LayoutItem to write as the initial-state
                entry. If omitted the file starts empty and its first entry
                will be an update entry, which readers accept as an
                updates-only file.

        Returns:
            An open writer.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            if initial_item is not None:
                json_data, buffers = to_json_with_state(initial_item)
                f.write(create_bytes(json_data, buffers))
        return cls(path)

    @classmethod
    def open(cls, path: Union[str, Path]) -> "ColightWriter":
        """
        Open an existing .colight file for further appends.

        Args:
            path: Path to an existing conforming .colight file.

        Returns:
            An open writer positioned at the end of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file does not start with the .colight magic
                bytes, or if its length is not a multiple of 8 (which would
                misalign every buffer in the appended entry).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such .colight file: {path}")
        _check_appendable(path)
        with open(path, "rb") as f:
            magic = f.read(len(MAGIC_BYTES))
        if magic and magic != MAGIC_BYTES:
            raise ValueError(
                f"{path} does not start with the .colight magic bytes "
                f'"COLIGHT\\x00".'
            )
        return cls(path)

    def append(self, update_item: Any) -> None:
        """
        Append one update entry.

        The entry is written whole in a single ``write``, so a concurrent reader
        never observes a partially applied update beyond the torn-tail case the
        format already tolerates.

        Args:
            update_item: A LayoutItem, ``Plot.State({...})``, or any object the
                widget serializer accepts.
        """
        self._file_or_raise().write(create_update_bytes(update_item))

    def append_all(self, update_items: List[Any]) -> None:
        """
        Append several update entries, in order.

        Args:
            update_items: LayoutItems or objects to serialize as updates.
        """
        f = self._file_or_raise()
        for update_item in update_items:
            f.write(create_update_bytes(update_item))

    def flush(self) -> None:
        """Force everything written so far to stable storage (``fsync``)."""
        f = self._file_or_raise()
        f.flush()
        os.fsync(f.fileno())

    def close(self) -> None:
        """Close the file. Idempotent."""
        if self._file is not None:
            self._file.close()
            self._file = None  # type: ignore[assignment]

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has been called."""
        return self._file is None

    def __enter__(self) -> "ColightWriter":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def _file_or_raise(self):
        if self._file is None:
            raise ValueError(f"This ColightWriter ({self.path}) is closed.")
        return self._file


def append_update(
    file_path: Union[str, Path],
    update_item: Any,
) -> str:
    """
    Append a single update to an existing .colight file, opening and closing it.

    The one-shot form of :class:`ColightWriter`. Prefer the writer when
    appending repeatedly from a long-running producer: it avoids re-opening the
    file per entry, which costs roughly 30-40% of throughput at small entry
    sizes. Use this when appends are occasional or the producer cannot keep a
    handle open. Both honour the same streaming contract -- append-only,
    alignment preserved, readers tolerate a torn tail.

    Args:
        file_path: Path to the existing .colight file
        update_item: A LayoutItem or any object that can be serialized to create an update

    Returns:
        Path to the updated file

    Raises:
        ValueError: If the file's length is not a multiple of 8, which would
            misalign the appended entry's buffers.
    """
    file_path = Path(file_path)
    _check_appendable(file_path)

    with open(file_path, "ab") as f:
        f.write(create_update_bytes(update_item))

    return str(file_path)


def append_updates(
    file_path: Union[str, Path],
    update_items: List[Any],
) -> str:
    """
    Append several updates to an existing .colight file in one open/close.

    Args:
        file_path: Path to the existing .colight file
        update_items: List of LayoutItems or objects to serialize as updates

    Returns:
        Path to the updated file

    Raises:
        ValueError: If the file's length is not a multiple of 8.
    """
    file_path = Path(file_path)
    _check_appendable(file_path)

    with open(file_path, "ab") as f:
        for update_item in update_items:
            f.write(create_update_bytes(update_item))

    return str(file_path)


def save_updates(
    output_path: Union[str, Path],
    update_items: List[Any],
) -> str:
    """
    Save updates to a new .colight file (without initial state).

    Readers accept a file whose first entry is an update entry; the result is a
    stream of states with no initial visual, which the viewer's scrubber steps
    through.

    Args:
        output_path: Path to write the file
        update_items: List of LayoutItems or objects to serialize as updates

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
