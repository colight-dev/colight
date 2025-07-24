"""Shared utilities for colight_prose package."""

import hashlib
from typing import List, Optional

from .constants import DEFAULT_IGNORE_PATTERNS
from .model import Block


def merge_ignore_patterns(user_patterns: Optional[List[str]] = None) -> List[str]:
    """Merge user ignore patterns with default patterns.

    Args:
        user_patterns: Optional list of user-provided ignore patterns

    Returns:
        Combined list with user patterns first, then defaults
    """
    combined = list(user_patterns) if user_patterns else []
    combined.extend(DEFAULT_IGNORE_PATTERNS)
    return combined


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
