"""Tests for ignore pattern handling."""

from typing import List, Optional

from colight_prose.constants import DEFAULT_IGNORE_PATTERNS


def merge_ignore_patterns(
    user_patterns: Optional[List[str]] = None,
    default_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Merge user and default ignore patterns.

    This is what we'll extract to a shared utility.
    """
    if default_patterns is None:
        default_patterns = DEFAULT_IGNORE_PATTERNS

    combined = list(user_patterns) if user_patterns else []
    combined.extend(default_patterns)
    return combined


def test_merge_ignore_patterns_with_user_patterns():
    """Test merging user patterns with defaults."""
    user_patterns = ["*.tmp", "build/"]
    result = merge_ignore_patterns(user_patterns)

    # Should include both user patterns and defaults
    assert "*.tmp" in result
    assert "build/" in result

    # Should also include defaults
    for pattern in DEFAULT_IGNORE_PATTERNS:
        assert pattern in result


def test_merge_ignore_patterns_without_user_patterns():
    """Test merging with no user patterns."""
    result = merge_ignore_patterns(None)

    # Should only include defaults
    assert result == list(DEFAULT_IGNORE_PATTERNS)


def test_merge_ignore_patterns_empty_user_patterns():
    """Test merging with empty user patterns."""
    result = merge_ignore_patterns([])

    # Should only include defaults
    assert result == list(DEFAULT_IGNORE_PATTERNS)


def test_merge_ignore_patterns_preserves_order():
    """Test that pattern order is preserved (user patterns first)."""
    user_patterns = ["user1", "user2"]
    result = merge_ignore_patterns(user_patterns)

    # User patterns should come first
    assert result[0] == "user1"
    assert result[1] == "user2"

    # Then defaults
    assert result[2:] == list(DEFAULT_IGNORE_PATTERNS)


def test_ignore_pattern_matching():
    """Test that the patterns work correctly with fnmatch."""
    import fnmatch

    patterns = merge_ignore_patterns(["*.tmp", "secret*"])

    # Test files that should be ignored
    ignored_files = [
        "__pycache__",  # Default pattern
        ".git",  # Default pattern
        "file.tmp",  # User pattern
        "secret_key.py",  # User pattern
        "node_modules",  # Default pattern
        ".hidden_file",  # Default pattern (.*)
    ]

    for filename in ignored_files:
        # Check if any pattern matches
        matched = any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)
        assert matched, f"{filename} should be ignored"

    # Test files that should NOT be ignored
    allowed_files = [
        "main.py",
        "src/module.py",
        "README.md",
    ]

    for filename in allowed_files:
        matched = any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)
        assert not matched, f"{filename} should not be ignored"


def test_current_duplication_behavior():
    """Test that the current duplicated implementations behave identically."""

    # Simulate ApiMiddleware._get_combined_ignore_patterns
    def api_middleware_version(ignore):
        combined_ignore = list(ignore) if ignore else []
        combined_ignore.extend(DEFAULT_IGNORE_PATTERNS)
        return combined_ignore

    # Simulate LiveServer._get_combined_ignore_patterns
    def live_server_version(ignore):
        combined_ignore = list(ignore) if ignore else []
        combined_ignore.extend(DEFAULT_IGNORE_PATTERNS)
        return combined_ignore

    # Test with various inputs
    test_cases = [
        None,
        [],
        ["*.tmp"],
        ["*.tmp", "build/", "*.log"],
    ]

    for test_input in test_cases:
        result1 = api_middleware_version(test_input)
        result2 = live_server_version(test_input)

        # Both implementations should produce identical results
        assert result1 == result2, f"Implementations differ for input: {test_input}"
