"""Clean parser implementation for colight files."""

import hashlib
import os
import pathlib
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, List, Literal, Optional, Union

import libcst as cst
from colight_prose.dependency_analyzer import (
    analyze_block,
)
from libcst.metadata import MetadataWrapper, PositionProvider

from .model import Block, BlockInterface, Document, Element, EmptyLine, TagSet
from .pragma import (
    extract_pragma_content,
    parse_comment_line,
)
from .utils import hash_block_content


# Element type enumeration
class ElementKind(Enum):
    """Types of source elements."""

    CODE = auto()
    COMMENT = auto()
    PRAGMA = auto()
    BLANK = auto()


# Base class for all source elements
@dataclass
class SourceElement:
    """Base class for source elements."""

    lineno: int


# Specialized element types
@dataclass
class CodeLine(SourceElement):
    """A line containing Python code."""

    content: cst.CSTNode


@dataclass
class CommentLine(SourceElement):
    """A line containing a comment."""

    content: str
    is_pragma: bool = False

    @property
    def is_empty_comment(self) -> bool:
        """Check if this is an empty comment line."""
        return self.content == ""


@dataclass
class PragmaLine(SourceElement):
    """A line containing a pragma directive."""

    content: str


@dataclass
class BlankLine(SourceElement):
    """A blank line."""


# Union type for all parsed lines
ParsedLine = Union[CodeLine, CommentLine, PragmaLine, BlankLine]


@dataclass
class RawBlock:
    """A block before final processing."""

    prose_lines: List[str]
    code_nodes: List[Union[cst.CSTNode, EmptyLine]]
    pragma_lines: List[str]
    start_line: int


# State machine for block grouping
class _BlockState(Enum):
    """States for the block grouping state machine."""

    SEARCHING = auto()
    IN_BLOCK = auto()


def _compute_block_cache_key(
    block: Block,
    symbol_providers: dict[str, str],
    project_root: Optional[str] = None,
) -> str:
    """Compute cache key for a block based on content and dependencies.

    The cache key includes:
    - Block's content hash
    - Cache keys of dependencies
    - File modification times
    """
    # Start with content hash
    content = hash_block_content(block).encode()

    # Add dependency cache keys
    dep_keys = []
    for dep_name in sorted(block.interface.requires):
        if dep_name in symbol_providers:
            dep_keys.append(f"{dep_name}:{symbol_providers[dep_name]}")

    # Add file dependencies with mtimes
    file_deps = []
    if (
        hasattr(block.interface, "file_dependencies")
        and block.interface.file_dependencies
    ):
        for file_path in sorted(block.interface.file_dependencies):
            # File path is relative to project root
            if project_root:
                abs_path = os.path.join(project_root, file_path)
            else:
                abs_path = file_path

            try:
                mtime = os.path.getmtime(abs_path)
                file_deps.append(f"file:{file_path}:{mtime}")
            except OSError:
                # File doesn't exist or can't be accessed
                file_deps.append(f"file:{file_path}:missing")

    # Combine all parts
    combined = (
        content
        + b"::"
        + "\n".join(dep_keys).encode()
        + b"::"
        + "\n".join(file_deps).encode()
    )
    cache_key = hashlib.sha256(combined).hexdigest()[:16]
    return cache_key


def _classify_code_node(node: cst.CSTNode) -> Literal["STATEMENT", "EXPRESSION"]:
    """Classify a code node as statement or expression."""
    if isinstance(node, cst.SimpleStatementLine):
        if len(node.body) == 1 and isinstance(node.body[0], cst.Expr):
            return "EXPRESSION"
    return "STATEMENT"


def _is_literal_value(node: cst.BaseExpression) -> bool:
    """Check if a CST node represents a literal value."""
    # Simple literals
    if isinstance(
        node, (cst.Integer, cst.Float, cst.SimpleString, cst.FormattedString)
    ):
        return True

    # Boolean/None literals
    if isinstance(node, cst.Name) and node.value in ("True", "False", "None"):
        return True

    # Unary operations on numeric literals
    if isinstance(node, cst.UnaryOperation):
        if isinstance(node.operator, (cst.Minus, cst.Plus)):
            return _is_literal_value(node.expression)

    # Collections with only literal contents
    if isinstance(node, (cst.List, cst.Tuple, cst.Set)):
        return all(
            _is_literal_value(elem.value)
            for elem in node.elements
            if isinstance(elem, cst.Element)
        )

    if isinstance(node, cst.Dict):
        return all(
            _is_literal_value(elem.key) and _is_literal_value(elem.value)
            for elem in node.elements
            if isinstance(elem, cst.DictElement) and elem.key is not None
        )

    return False


def _strip_leading_comments(node: cst.CSTNode) -> cst.CSTNode:
    """Create a copy of the node without leading comments."""
    if not isinstance(node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
        return node

    # Filter out comment lines, keeping only whitespace-only lines
    new_leading_lines = []
    for line in node.leading_lines:
        if not line.comment:
            new_leading_lines.append(line)

    return node.with_changes(leading_lines=new_leading_lines)


# Main parser functions
def lex(source_code: str) -> List[ParsedLine]:
    """Step 1: Extract all lines from source code and classify them."""
    elements: List[ParsedLine] = []

    # Parse with LibCST
    module = cst.parse_module(source_code)

    # Enable position tracking
    wrapper = MetadataWrapper(module)
    positions = wrapper.resolve(PositionProvider)

    # Process header comments first
    current_line = 1
    in_pep723_block = False

    if hasattr(module, "header") and module.header:
        for line in module.header:
            if line.comment:
                # Get the raw comment value and stripped version
                raw_comment = line.comment.value
                stripped_text = raw_comment.lstrip("#").strip()

                # Check for PEP 723 markers
                if stripped_text == "/// script":
                    in_pep723_block = True
                    current_line += 1
                    continue
                elif stripped_text == "///":
                    in_pep723_block = False
                    current_line += 1
                    continue

                # Skip content inside PEP 723 block
                if in_pep723_block:
                    current_line += 1
                    continue

                # Parse the comment
                parsed = parse_comment_line(raw_comment)

                # Add the parsed element
                if parsed["kind"] == "PRAGMA":
                    elements.append(
                        PragmaLine(content=parsed["text"], lineno=current_line)
                    )
                else:
                    elements.append(
                        CommentLine(
                            content=parsed["text"], lineno=current_line, is_pragma=False
                        )
                    )
            elif line.whitespace.value.strip() == "":
                # Blank line
                if not in_pep723_block:
                    elements.append(BlankLine(lineno=current_line))
            current_line += 1

    # Process body statements
    for stmt in module.body:
        # Get position
        stmt_line = positions.get(stmt, None)
        if stmt_line:
            stmt_line_num = stmt_line.start.line
        else:
            stmt_line_num = current_line

        # Extract leading comments
        if hasattr(stmt, "leading_lines"):
            comment_line = stmt_line_num - len(stmt.leading_lines)
            for line in stmt.leading_lines:
                if line.comment:
                    # Get the raw comment value and stripped version
                    raw_comment = line.comment.value
                    stripped_text = raw_comment.lstrip("#").strip()

                    # Parse the comment
                    parsed = parse_comment_line(raw_comment)

                    if parsed["kind"] == "PRAGMA":
                        elements.append(
                            PragmaLine(content=parsed["text"], lineno=comment_line)
                        )
                    else:
                        elements.append(
                            CommentLine(
                                content=parsed["text"],
                                lineno=comment_line,
                                is_pragma=False,
                            )
                        )
                elif line.whitespace.value.strip() == "":
                    elements.append(BlankLine(lineno=comment_line))
                comment_line += 1

        # Add the code statement (without leading comments)
        clean_stmt = _strip_leading_comments(stmt)
        elements.append(CodeLine(content=clean_stmt, lineno=stmt_line_num))

        # Update current line
        if stmt_line:
            current_line = stmt_line.end.line + 1

    # Process footer (trailing comments/blank lines after last statement)
    if hasattr(module, "footer") and module.footer:
        for line in module.footer:
            if line.comment:
                # Get the raw comment value
                raw_comment = line.comment.value

                # Parse the comment
                parsed = parse_comment_line(raw_comment)

                if parsed["kind"] == "PRAGMA":
                    elements.append(
                        PragmaLine(content=parsed["text"], lineno=current_line)
                    )
                else:
                    elements.append(
                        CommentLine(
                            content=parsed["text"],
                            lineno=current_line,
                            is_pragma=False,
                        )
                    )
            elif line.whitespace.value.strip() == "":
                # Blank line
                elements.append(BlankLine(lineno=current_line))
            current_line += 1

    return elements


def _group_blocks_generator(elements: List[ParsedLine]) -> Iterator[RawBlock]:
    """Step 2: Group lines into blocks based on blank line separators using FSM."""
    state = _BlockState.SEARCHING
    current: Optional[RawBlock] = None
    i = 0

    def _start_block(el: ParsedLine) -> RawBlock:
        """Start a new block with the given element."""
        block = RawBlock(
            prose_lines=[],
            code_nodes=[],
            pragma_lines=[],
            start_line=el.lineno,
        )
        _append_to_block(block, el)
        return block

    def _append_to_block(block: RawBlock, el: ParsedLine) -> None:
        """Add an element to the current block."""
        if isinstance(el, CommentLine):
            if el.is_empty_comment and block.code_nodes:
                # Empty comment within code - add sentinel
                block.code_nodes.append(EmptyLine())
            else:
                # Regular comment - add to prose
                block.prose_lines.append(el.content)
        elif isinstance(el, PragmaLine):
            block.pragma_lines.append(el.content)
        elif isinstance(el, CodeLine):
            block.code_nodes.append(el.content)
        # BlankLine is handled separately in state machine

    def _blank_ends_block(current_block: RawBlock) -> bool:
        """Check if a blank line should end the current block."""
        # If we only have pragmas so far, don't end the block
        if (
            current_block.pragma_lines
            and not current_block.prose_lines
            and not current_block.code_nodes
        ):
            return False

        # Blank lines always end blocks
        return True

    # Process all elements
    while i < len(elements):
        el = elements[i]

        # State machine transitions
        match (state, type(el).__name__):
            case (_BlockState.SEARCHING, "BlankLine"):
                pass  # Ignore leading blanks

            case (_BlockState.SEARCHING, _):
                current = _start_block(el)
                state = _BlockState.IN_BLOCK

            case (_BlockState.IN_BLOCK, "BlankLine"):
                if current and _blank_ends_block(current):
                    if current.prose_lines or current.code_nodes:
                        yield current
                    current = None
                    state = _BlockState.SEARCHING

            case (_BlockState.IN_BLOCK, _):
                if current:
                    _append_to_block(current, el)

        i += 1

    # Yield final block if any
    if current and (current.prose_lines or current.code_nodes):
        yield current


def group_blocks(elements: List[ParsedLine]) -> List[RawBlock]:
    """Group lines into blocks (backward compatibility wrapper)."""
    return list(_group_blocks_generator(elements))


def build_document(
    raw_blocks: List[RawBlock],
    file_pragma: Optional[TagSet] = None,
    current_file: Optional[str] = None,
    project_root: Optional[str] = None,
) -> Document:
    """Step 3: Convert raw blocks into a Document with proper Elements and Tags.

    Args:
        raw_blocks: List of raw blocks to process
        file_pragma: Optional file-level pragma tags
        current_file: Optional path to current file (relative to project root)
        project_root: Optional path to project root directory
    """
    if file_pragma is None:
        file_pragma = TagSet()

    blocks = []

    # Track which block provides each symbol (for dependency resolution)
    symbol_providers = {}  # symbol -> block_cache_key

    for block_idx, raw_block in enumerate(raw_blocks):
        # Parse block-level pragmas
        block_pragma = TagSet()
        for pragma_line in raw_block.pragma_lines:
            if isinstance(pragma_line, str):
                pragma_content = extract_pragma_content(pragma_line)
                block_pragma = block_pragma | TagSet.from_pragma_content(pragma_content)

        # Build elements list
        elements = []

        # Add prose element if present
        if raw_block.prose_lines:
            # Join prose lines with newlines
            prose_content = "\n".join(raw_block.prose_lines)
            prose_elem = Element(
                kind="PROSE", content=prose_content, lineno=raw_block.start_line
            )
            elements.append(prose_elem)

        # Process code nodes
        if raw_block.code_nodes:
            if len(raw_block.code_nodes) == 1:
                # Single code node
                node = raw_block.code_nodes[0]
                if isinstance(node, cst.CSTNode):
                    elem_kind = _classify_code_node(node)
                    code_elem = Element(
                        kind=elem_kind,
                        content=node,
                        lineno=raw_block.start_line + len(raw_block.prose_lines) + 1,
                    )
                    elements.append(code_elem)
            else:
                # Multiple code nodes - check if we have any EmptyLine sentinels
                has_empty_lines = any(
                    isinstance(node, EmptyLine) for node in raw_block.code_nodes
                )
                if has_empty_lines:
                    # We have empty lines - preserve them in the sequence
                    # Determine kind based on last non-EmptyLine node
                    last_code_node = None
                    for node in reversed(raw_block.code_nodes):
                        if isinstance(node, cst.CSTNode):
                            last_code_node = node
                            break
                    last_kind = (
                        _classify_code_node(last_code_node)
                        if last_code_node
                        else "STATEMENT"
                    )
                    code_elem = Element(
                        kind=last_kind,
                        content=raw_block.code_nodes,
                        lineno=raw_block.start_line + len(raw_block.prose_lines) + 1,
                    )
                    elements.append(code_elem)
                else:
                    # No empty lines - check if all statements
                    all_statements = all(
                        _classify_code_node(node) == "STATEMENT"
                        for node in raw_block.code_nodes
                        if isinstance(node, cst.CSTNode)
                    )

                    if all_statements:
                        # All statements - combine them
                        code_elem = Element(
                            kind="STATEMENT",
                            content=raw_block.code_nodes,
                            lineno=raw_block.start_line
                            + len(raw_block.prose_lines)
                            + 1,
                        )
                        elements.append(code_elem)
                    else:
                        # Mixed or has expressions - add separately
                        line_offset = (
                            raw_block.start_line + len(raw_block.prose_lines) + 1
                        )
                        for i, node in enumerate(raw_block.code_nodes):
                            if isinstance(node, cst.CSTNode):
                                elem_kind = _classify_code_node(node)
                                code_elem = Element(
                                    kind=elem_kind, content=node, lineno=line_offset + i
                                )
                                elements.append(code_elem)

        # Create block if it has content
        if elements or raw_block.prose_lines or raw_block.code_nodes:
            # For empty blocks (prose only with no real content), add a dummy element
            if not elements:
                dummy_stmt = cst.SimpleStatementLine([cst.Pass()])
                elements = [
                    Element(
                        kind="STATEMENT",
                        content=dummy_stmt,
                        lineno=raw_block.start_line,
                    )
                ]

            # Create block initially with temporary ID
            block = Block(
                elements=elements,
                tags=block_pragma,
                start_line=raw_block.start_line,
                id=f"temp-{block_idx}",  # Temporary ID
            )

            # Analyze dependencies for blocks with code
            if block.get_code_elements():
                try:
                    code_text = block.get_code_text()
                    provides, requires, file_deps = analyze_block(
                        code_text, current_file, project_root
                    )
                    block.interface = BlockInterface(
                        provides=sorted(list(provides)),
                        requires=sorted(list(requires)),
                        file_dependencies=file_deps,
                    )
                except Exception:
                    # If analysis fails, just use empty interface
                    pass

            # Compute cache key as the block's ID
            cache_key = _compute_block_cache_key(block, symbol_providers, project_root)
            block.id = cache_key

            # Update symbol providers with this block's provides
            for symbol in block.interface.provides:
                symbol_providers[symbol] = cache_key

            blocks.append(block)

    return Document(blocks=blocks, tags=file_pragma)


def extract_file_pragma(elements: List[ParsedLine]) -> TagSet:
    """Extract file-level pragma tags (hide-all-*) from the beginning of the file."""
    pragma = TagSet()

    # Look for hide-all-* pragmas before the first code element
    for element in elements:
        if isinstance(element, CodeLine):
            break
        elif isinstance(element, PragmaLine):
            content = extract_pragma_content(element.content)
            tags = TagSet.from_pragma_content(content)
            # Only keep hide-all-* tags for file-level
            file_tags = TagSet(
                frozenset(tag for tag in tags.flags if tag.startswith("hide-all-"))
            )
            pragma = pragma | file_tags

    return pragma


def parse_document(
    source_code: str,
    current_file: Optional[str] = None,
    project_root: Optional[str] = None,
) -> Document:
    """Parse source code into a Document.

    This is the main API for the new parser.

    Args:
        source_code: The source code to parse
        current_file: Optional path to current file (relative to project root)
        project_root: Optional path to project root directory
    """
    # Step 1: Lex into classified elements
    elements = lex(source_code)

    # Step 2: Extract file-level pragma
    file_pragma = extract_file_pragma(elements)

    # Step 3: Group into blocks
    raw_blocks = list(_group_blocks_generator(elements))

    # Step 4: Build document
    document = build_document(raw_blocks, file_pragma, current_file, project_root)

    return document


def find_project_root(
    start_path: pathlib.Path,
    markers: tuple[str, ...] = (".git", "pyproject.toml", "setup.py"),
) -> pathlib.Path:
    """Find project root by looking for marker files/directories."""
    for parent in [start_path] + list(start_path.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    return start_path.parent  # Default to parent directory


def parse_file(
    file_path: pathlib.Path, project_root: Optional[pathlib.Path] = None
) -> Document:
    """Parse a colight file into a Document."""
    source_code = file_path.read_text(encoding="utf-8")

    # Determine project root if not provided
    if project_root is None:
        project_root = find_project_root(file_path)

    # Pass file path information for dependency tracking
    try:
        relative_path = str(file_path.relative_to(project_root))
    except ValueError:
        relative_path = str(file_path)

    return parse_document(
        source_code, current_file=relative_path, project_root=str(project_root)
    )


# Convenience exports for compatibility
def parse_colight_file(
    file_path: pathlib.Path, project_root: Optional[pathlib.Path] = None
) -> Document:
    """Parse a colight file (compatibility alias)."""
    return parse_file(file_path, project_root=project_root)
