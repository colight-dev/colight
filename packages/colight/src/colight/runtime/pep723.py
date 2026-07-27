"""Support for PEP 723 inline script metadata."""

import re
from typing import List, Optional


def detect_pep723_metadata(content: str) -> Optional[str]:
    """
    Detect PEP 723 metadata block in Python file content.

    Returns the metadata content if found, None otherwise.
    """
    # PEP 723 metadata format:
    # # /// script
    # # dependencies = [
    # #   "package1",
    # #   "package2>=1.0",
    # # ]
    # # ///

    # Use a regex to find the first PEP 723 metadata block.
    # The `(?m)` flag enables multi-line mode, so `^` matches the start of a line.
    # The `(?s)` flag (re.DOTALL) allows `.` to match newlines.
    match = re.search(
        r"^[ \t]*# /// script\n(.*?)\n[ \t]*# ///[ \t]*$",
        content,
        re.MULTILINE | re.DOTALL,
    )

    if not match:
        return None

    metadata_block = match.group(1)
    metadata_lines = []

    # Process the captured metadata content.
    for line in metadata_block.split("\n"):
        # Per PEP 723, lines must be empty or start with "# ".
        if line.startswith("# "):
            metadata_lines.append(line[2:])
        elif line.strip() == "":
            # The spec allows empty lines.
            metadata_lines.append("")
        elif line.strip() == "#":
            # Allow lines with only a # as a comment.
            metadata_lines.append("")
        else:
            # Invalid line format within the block.
            return None

    return "\n".join(metadata_lines)


def parse_dependencies(metadata: str) -> List[str]:
    """Extract dependencies from PEP 723 metadata."""
    # Simple regex to extract dependencies list
    # This handles both single-line and multi-line formats
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", metadata, re.DOTALL)

    if not match:
        return []

    deps_str = match.group(1)

    # Extract individual dependencies (quoted strings)
    deps = re.findall(r'["\']([^"\']+)["\']', deps_str)

    return deps
