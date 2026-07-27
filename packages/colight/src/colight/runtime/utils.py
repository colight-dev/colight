"""Shared utilities for colight.runtime package."""

import hashlib

from .model import Block


def hash_block_content(block: Block) -> str:
    """Generate a content hash for a block.

    This provides a consistent way to hash block content across the codebase.
    The hash includes element kinds and content to ensure changes are detected.

    Args:
        block: The block to hash

    Returns:
        SHA256 hex digest of the block content
    """
    content_parts = []
    for elem in block.elements:
        content_parts.append(f"{elem.kind}:{elem.get_source()}")
    content = "\n".join(content_parts)
    return hashlib.sha256(content.encode()).hexdigest()
