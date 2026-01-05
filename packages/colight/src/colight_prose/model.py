"""Core data model for colight-prose parser and execution pipeline."""

import re
from dataclasses import dataclass, field
from types import CodeType
from typing import FrozenSet, List, Literal, Optional, Sequence, Set, Union, cast

import libcst as cst

# Constants for pragma handling
_HIDE_ALL_PREFIX = "hide-all-"
_HIDE_PREFIX = "hide-"
_SHOW_PREFIX = "show-"


def _base(tag: str) -> str:
    """Return the category part of a pragma tag."""
    return re.sub(rf"^({_SHOW_PREFIX}|{_HIDE_PREFIX}|{_HIDE_ALL_PREFIX})", "", tag)


@dataclass(slots=True)
class EmptyLine:
    """Sentinel for an empty line between code elements.

    IMPORTANT: This is a critical feature that allows multiple statements to be
    grouped together in a single Element while preserving their visual structure.
    Empty lines (blank lines with no comments) between code statements are preserved
    using these sentinels, allowing the code to maintain its formatting while still
    being executed as a cohesive unit.

    DO NOT REMOVE OR CHANGE THIS BEHAVIOR - it is intentional and essential for
    the proper functioning of the colight-prose parser.
    """

    pass


@dataclass(frozen=True, slots=True)
class TagSet:
    """Immutable set of pragma flags with visibility behavior.

    Supports merging with "last writer wins" semantics per category.
    For example, if one TagSet has "hide-code" and another has "show-code",
    the merged result will have only "show-code" (last one wins).
    """

    flags: FrozenSet[str] = frozenset()

    def _shows(self, cat: str) -> bool:
        """Generic 'should X be shown?' query. Last matching flag wins."""
        # Walk through precedence: show-X > hide-X > hide-all-X
        for prefix in (_SHOW_PREFIX, _HIDE_PREFIX, _HIDE_ALL_PREFIX):
            flag = prefix + cat
            if flag in self.flags:
                return prefix == _SHOW_PREFIX
        return True  # default: visible

    def show_code(self) -> bool:
        """Check if code should be shown."""
        return self._shows("code")

    def show_prose(self) -> bool:
        """Check if prose should be shown."""
        return self._shows("prose")

    def show_statements(self) -> bool:
        """Check if statements should be shown (requires code visible)."""
        return self.show_code() and self._shows("statements")

    def show_visuals(self) -> bool:
        """Check if visuals should be shown."""
        return self._shows("visuals")

    def merged(self, *others: "TagSet") -> "TagSet":
        """Merge multiple TagSets with last writer wins semantics per category.

        Example:
            ts1 = TagSet({"hide-code", "show-prose"})
            ts2 = TagSet({"show-code", "hide-visuals"})
            ts3 = ts1.merged(ts2)
            # ts3.flags == {"show-code", "show-prose", "hide-visuals"}
        """
        current = {}  # cat -> full_flag

        # Process all TagSets in order
        for ts in (self, *others):
            for flag in ts.flags:
                cat = _base(flag)
                current[cat] = flag  # overwrite same category

        return TagSet(frozenset(current.values()))

    def __or__(self, other: "TagSet") -> "TagSet":
        """Syntactic sugar for merging: ts3 = ts1 | ts2"""
        return self.merged(other)

    @classmethod
    def from_pragma_content(cls, content: str) -> "TagSet":
        """Create TagSet from pragma content string.

        Example:
            TagSet.from_pragma_content("hide-code show-visuals")
            # Returns TagSet with {"hide-code", "show-visuals"}
            TagSet.from_pragma_content("pragma: always-eval")
            # Returns TagSet with {"always-eval"}
        """
        # Handle different pragma formats
        normalized_content = content.lower().strip()

        # Handle "pragma: <flag>" format
        if normalized_content.startswith("pragma:"):
            flag = normalized_content[7:].strip()
            return cls(frozenset({flag}) if flag else frozenset())

        # Extract all pragma tags using regex for hide/show format
        tags = set(re.findall(r"\b(?:hide|show)(?:-all)?-\w+\b", normalized_content))

        # Normalize singular to plural for consistency
        normalized = set()
        for tag in tags:
            if tag.endswith(("statement", "visual")) and not tag.startswith("hide-all"):
                normalized.add(tag + "s")
            else:
                normalized.add(tag)

        return cls(frozenset(normalized))


@dataclass(slots=True)
class Element:
    """An item within a block.

    Elements can be:
    - PROSE: Markdown/comment text
    - STATEMENT: Python statement(s)
    - EXPRESSION: Python expression that may produce a visual

    The content field can be:
    - str: For prose elements
    - cst.CSTNode: For a single code statement/expression
    - Sequence[Union[cst.CSTNode, EmptyLine]]: For multiple statements joined by
      empty lines. This allows preserving blank lines between statements while
      keeping them as a single executable unit.
    """

    kind: Literal["PROSE", "STATEMENT", "EXPRESSION"]
    content: Union[
        str, cst.CSTNode, Sequence[Union[cst.CSTNode, EmptyLine]]
    ]  # str for prose, CST for code
    lineno: int

    def get_source(self) -> str:
        """Get the source representation of this element."""
        if self.kind == "PROSE":
            return self.content if isinstance(self.content, str) else ""

        # Handle code elements
        if isinstance(self.content, list):
            # Multiple statements/expressions with possible empty lines
            lines = []
            for item in self.content:
                if isinstance(item, EmptyLine):
                    lines.append("")  # Empty line
                elif isinstance(item, cst.CSTNode):
                    # Ensure the item is a proper statement/compound statement
                    if isinstance(
                        item, (cst.SimpleStatementLine, cst.BaseCompoundStatement)
                    ):
                        lines.append(cst.Module(body=[item]).code.strip())
                    else:
                        # Wrap other nodes in a statement
                        if isinstance(item, cst.BaseExpression):
                            stmt = cst.SimpleStatementLine([cst.Expr(item)])
                            lines.append(cst.Module(body=[stmt]).code.strip())
                        else:
                            # Fallback for other node types
                            lines.append(str(item).strip())
                else:
                    lines.append(str(item).strip())
            return "\n".join(lines)
        elif isinstance(self.content, cst.CSTNode):
            # Single node
            if isinstance(
                self.content, (cst.SimpleStatementLine, cst.BaseCompoundStatement)
            ):
                return cst.Module(body=[self.content]).code.strip()
            elif isinstance(self.content, cst.BaseExpression):
                # Wrap expression in statement
                stmt = cst.SimpleStatementLine([cst.Expr(self.content)])
                return cst.Module(body=[stmt]).code.strip()
            else:
                # Ensure the content is a proper statement/compound statement
                if isinstance(
                    self.content, (cst.SimpleStatementLine, cst.BaseCompoundStatement)
                ):
                    return cst.Module(body=[self.content]).code.strip()
                else:
                    # Wrap in statement if it's an expression
                    if isinstance(self.content, cst.BaseExpression):
                        stmt = cst.SimpleStatementLine([cst.Expr(self.content)])
                        return cst.Module(body=[stmt]).code.strip()
                    else:
                        # Fallback for other node types
                        return str(self.content).strip()
        else:
            return str(self.content)


@dataclass(slots=True)
class BlockInterface:
    """Interface information for a block (provides/requires symbols)."""

    provides: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    file_dependencies: Set[str] = field(default_factory=set)


@dataclass(slots=True)
class Block:
    """Execution/render unit containing elements and metadata.

    Blocks are separated by blank lines in the source.
    Each block has its own TagSet and can be executed independently.
    """

    elements: List[Element]
    tags: TagSet
    start_line: int
    id: str = field(default="")  # Cache key as ID
    interface: BlockInterface = field(default_factory=BlockInterface)

    # Cached compiled code (populated on demand)
    _exec_code: Optional[CodeType] = field(default=None, init=False, repr=False)
    _eval_code: Optional[CodeType] = field(default=None, init=False, repr=False)

    @property
    def last_element(self) -> Optional[Element]:
        """Get the last element in the block."""
        return self.elements[-1] if self.elements else None

    @property
    def has_expression_result(self) -> bool:
        """Check if the block ends with an expression that could produce a visual."""
        last = self.last_element
        return last is not None and last.kind == "EXPRESSION"

    @property
    def is_empty(self) -> bool:
        """Check if this block has no executable code."""
        return not any(
            elem.kind in ("STATEMENT", "EXPRESSION") for elem in self.elements
        )

    def get_prose_elements(self) -> List[Element]:
        """Get all prose elements in order."""
        return [elem for elem in self.elements if elem.kind == "PROSE"]

    def get_code_elements(self) -> List[Element]:
        """Get all code elements (statements and expressions) in order."""
        return [
            elem for elem in self.elements if elem.kind in ("STATEMENT", "EXPRESSION")
        ]

    def get_code_text(self) -> str:
        """Get all code as a single string for dependency analysis."""
        code_parts = []
        for elem in self.elements:
            if elem.kind in ("STATEMENT", "EXPRESSION"):
                code_parts.append(elem.get_source())
        return "\n".join(code_parts)

    def compile_once(self, filename: str = "<string>") -> None:
        """Compile code elements for execution (caches result).

        This separates statements to execute from the final expression to evaluate.
        """
        if self._exec_code is not None:  # Already compiled
            return

        code_elements = self.get_code_elements()
        if not code_elements:
            return

        # Use the line number of the first code element, not the block start
        # This handles cases where prose comes before code
        first_code_line = code_elements[0].lineno
        line_offset = first_code_line - 1  # Convert to 0-based
        # Note: The rest of the function handles the actual compilation

        # Special handling when we have a single element with mixed content
        if len(code_elements) == 1 and isinstance(code_elements[0].content, list):
            # This element contains multiple CST nodes - we need to split them properly
            nodes = code_elements[0].content
            statements = []
            expression = None

            for node in nodes:
                if isinstance(node, EmptyLine):
                    continue  # Skip empty lines for execution
                elif isinstance(node, cst.CSTNode):
                    # Check if this is an expression (single Expr in a SimpleStatementLine)
                    is_expression = (
                        isinstance(node, cst.SimpleStatementLine)
                        and len(node.body) == 1
                        and isinstance(node.body[0], cst.Expr)
                    )
                    if is_expression and node is nodes[-1]:
                        # Last node is an expression - we know it's a SimpleStatementLine
                        stmt_node = cast(cst.SimpleStatementLine, node)
                        expression = cst.Module(body=[stmt_node]).code.strip()
                    else:
                        # Statement - ensure proper type
                        if isinstance(
                            node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)
                        ):
                            statements.append(cst.Module(body=[node]).code.strip())
                        else:
                            # This shouldn't happen, but handle gracefully
                            continue

            if statements:
                # Prepend empty lines to maintain line numbers
                padded_source = "\n" * line_offset + "\n".join(statements)
                self._exec_code = compile(padded_source, filename, "exec")
            else:
                self._exec_code = None

            if expression:
                # For expressions, we need to account for all previous lines
                # Count lines in statements plus the offset
                total_lines = line_offset + sum(s.count("\n") + 1 for s in statements)
                padded_expr = "\n" * total_lines + expression
                self._eval_code = compile(padded_expr, filename, "eval")
            else:
                self._eval_code = None

        else:
            # Normal case - separate elements
            code_sources = []
            for elem in code_elements:
                source = elem.get_source()
                code_sources.append(source)

            if self.has_expression_result:
                # Last element is the expression to evaluate
                *statements, expression = code_sources

                if statements:
                    # Compile all statements for execution
                    exec_source = "\n".join(statements)
                    padded_source = "\n" * line_offset + exec_source
                    self._exec_code = compile(padded_source, filename, "exec")
                else:
                    self._exec_code = None

                # Compile the expression for evaluation
                # The expression might be multi-line, so we keep it intact
                # Count lines in statements to get the expression's actual line
                total_lines = line_offset + sum(s.count("\n") + 1 for s in statements)
                padded_expr = "\n" * total_lines + expression
                self._eval_code = compile(padded_expr, filename, "eval")
            else:
                # All statements, no expression
                exec_source = "\n".join(code_sources)
                padded_source = "\n" * line_offset + exec_source
                self._exec_code = compile(padded_source, filename, "exec")
                self._eval_code = None


@dataclass(slots=True)
class Document:
    """Top-level container for a parsed colight file.

    Contains blocks and file-level tags.
    This is the main data structure passed through the execution pipeline.
    """

    blocks: List[Block]
    tags: TagSet = field(default_factory=TagSet)

    @property
    def has_content(self) -> bool:
        """Check if document has any non-empty blocks."""
        return any(not block.is_empty for block in self.blocks)

    def get_executable_blocks(self) -> List[Block]:
        """Get only blocks that contain code to execute."""
        return [block for block in self.blocks if not block.is_empty]

    def get_cache_keys(self) -> List[str]:
        """Get all cache keys (block IDs) in document order."""
        return [block.id for block in self.blocks]
