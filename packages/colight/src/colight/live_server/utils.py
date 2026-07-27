"""Shared utilities for colight.live_server package."""

from typing import List, Optional

from .constants import DEFAULT_IGNORE_PATTERNS


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
