"""Centralized comment and pragma handling for the parser."""

from typing import Iterable, Iterator, Literal, Set, TypedDict, Union

import libcst as cst


class CommentParse(TypedDict):
    """Parsed comment with classification."""

    kind: Literal["COMMENT", "PRAGMA"]
    text: str


def is_pragma_comment(comment_text: str) -> bool:
    """Check if a comment is a pragma comment.

    Accepts comments starting with | or %% as pragma markers.
    """
    text = comment_text.strip()
    return text.startswith("|") or text.startswith("%%")


def extract_pragma_content(comment_text: str) -> str:
    """Extract the pragma content from a comment."""
    text = comment_text.strip()

    if text.startswith("|"):
        return text[1:].strip()
    elif text.startswith("%%"):
        return text[2:].strip()
    else:
        return text


def parse_comment_line(raw_comment: str) -> CommentParse:
    """Return canonicalized text and whether it's a pragma.

    Args:
        raw_comment: Raw comment string including # prefix

    Returns:
        CommentParse dict with kind and processed text
    """
    # Strip Python comment marker and whitespace
    body = raw_comment.lstrip("#").strip()

    if is_pragma_comment(body):
        return {"kind": "PRAGMA", "text": body}

    # For prose comments, handle markdown headers (## Title -> # Title)
    if raw_comment.startswith("#"):
        # Remove the first # and any single space after it
        if len(raw_comment) > 1 and raw_comment[1] == " ":
            prose_text = raw_comment[2:]  # Skip "# "
        else:
            prose_text = raw_comment[1:]  # Skip just "#"
    else:
        prose_text = raw_comment

    return {"kind": "COMMENT", "text": prose_text}


def iter_comment_lines(
    lines: Iterable[cst.EmptyLine],
) -> Iterator[tuple[int, CommentParse]]:
    """Process a sequence of CST EmptyLine objects for comments.

    Args:
        lines: Iterable of CST EmptyLine objects

    Yields:
        Tuples of (relative_lineno, CommentParse)
    """
    for rel_lineno, line in enumerate(lines, start=1):
        if line.comment:
            yield rel_lineno, parse_comment_line(line.comment.value)


def parse_pragma_arg(pragma: Union[str, Set[str], None]) -> Set[str]:
    """Parse pragma tags from comma-separated string.

    Args:
        pragma: Either a string of comma-separated tags, a set of tags, or None

    Returns:
        Set of pragma tag strings
    """
    if not pragma:
        return set()
    if isinstance(pragma, set):
        return pragma
    return {tag.strip() for tag in pragma.split(",") if tag.strip()}
