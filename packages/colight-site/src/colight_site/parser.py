"""Parse .colight.py files using LibCST."""

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
from dataclasses import dataclass, field
from typing import List, Union, Optional, Literal
import pathlib
import re


def _extract_pragma_tags(text: str) -> set[str]:
    """Extract all pragma tags from text using a single regex pattern."""
    # Single comprehensive pattern to match all our tag types
    tags = set(re.findall(r"\b(?:hide|show|format)-\w+\b", text.lower()))

    # Normalize to consistent forms
    normalized = set()
    for tag in tags:
        # Convert singular to plural for consistency
        if tag.endswith(("statement", "visual")):
            normalized.add(tag + "s")
        else:
            normalized.add(tag)

    return normalized


def should_hide_statements(tags: set[str]) -> bool:
    """Check if statements should be hidden based on tags."""
    # show- tags override hide- tags
    if "show-statements" in tags:
        return False
    return "hide-statements" in tags


def should_hide_visuals(tags: set[str]) -> bool:
    """Check if visuals should be hidden based on tags."""
    # show- tags override hide- tags
    if "show-visuals" in tags:
        return False
    return "hide-visuals" in tags


def should_hide_code(tags: set[str]) -> bool:
    """Check if code should be hidden based on tags."""
    # show- tags override hide- tags
    if "show-code" in tags:
        return False
    return "hide-code" in tags


def _strip_leading_comments(
    node: Union[cst.SimpleStatementLine, cst.BaseCompoundStatement],
) -> Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]:
    """Create a copy of the node without leading comments."""
    # Filter out comment lines, keeping only whitespace-only lines
    new_leading_lines = []
    for line in node.leading_lines:
        if not line.comment:
            new_leading_lines.append(line)

    # Create a new node with filtered leading lines
    return node.with_changes(leading_lines=new_leading_lines)


def _get_formats_from_tags(tags: set[str]) -> set[str]:
    """Extract formats from pragma tags."""
    formats = set()
    if "format-html" in tags:
        formats.add("html")
    if "format-markdown" in tags:
        formats.add("markdown")
    return formats


def _is_pragma_comment(comment_text: str) -> bool:
    """Check if a comment is a pragma comment.

    Only accepts comments starting with | or %% as pragma starters.
    """
    # Remove leading whitespace
    text = comment_text.strip()

    # Only check for explicit pragma starters: | or %%
    return text.startswith("|") or text.startswith("%%")


def _extract_pragma_content(comment_text: str) -> str:
    """Extract the pragma content from a comment.

    Removes the | or %% prefix and returns the content.
    """
    text = comment_text.strip()

    if text.startswith("|"):
        return text[1:].strip()
    elif text.startswith("%%"):
        return text[2:].strip()
    else:
        # This shouldn't happen if _is_pragma_comment is used correctly
        return text


@dataclass
class FormMetadata:
    """Metadata extracted from per-form pragma annotations."""

    pragma_tags: set[str] = field(default_factory=set)

    def resolve_with_defaults(self, default_tags: set[str]) -> set[str]:
        """Resolve form metadata with default tags."""
        # Form-specific tags override defaults
        return self.pragma_tags if self.pragma_tags else default_tags


@dataclass
class FileMetadata:
    """Metadata extracted from file-level pragma annotations."""

    pragma_tags: set[str] = field(default_factory=set)

    def merge_with_cli_options(
        self,
        hide_statements: bool = False,
        hide_visuals: bool = False,
        hide_code: bool = False,
        format: Optional[str] = None,
    ) -> tuple[set[str], set[str]]:
        """Merge file metadata with CLI options. CLI options take precedence."""
        # Start with file metadata tags
        result_tags = self.pragma_tags.copy()

        # CLI options override/add to file metadata
        if hide_statements:
            result_tags.add("hide-statements")
        if hide_visuals:
            result_tags.add("hide-visuals")
        if hide_code:
            result_tags.add("hide-code")

        # Get formats from file metadata
        result_formats = _get_formats_from_tags(self.pragma_tags)

        # CLI format option overrides file metadata formats
        if format:
            result_formats = {format}
        elif not result_formats:
            # Default to markdown if no format specified
            result_formats = {"markdown"}

        return result_tags, result_formats


class CombinedCode:
    """A pseudo-node that combines multiple consecutive code elements (statements and/or expressions)."""

    def __init__(
        self,
        code_elements: List[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]],
        empty_comment_positions: Optional[List[int]] = None,
    ):
        self.code_elements = code_elements
        # Track positions where empty comments appeared (0-based, between code elements)
        self.empty_comment_positions = empty_comment_positions or []

    def code(self) -> str:
        """Generate combined code from all code elements."""
        lines = []
        for i, stmt in enumerate(self.code_elements):
            # Add blank line if there was an empty comment before this statement
            if i in self.empty_comment_positions:
                lines.append("")

            if isinstance(stmt, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
                # Strip leading comments since they're handled separately
                node_without_comments = _strip_leading_comments(stmt)
                lines.append(cst.Module(body=[node_without_comments]).code.strip())
            else:
                # For other node types, wrap in a SimpleStatementLine if it's an expression
                if isinstance(stmt, cst.BaseExpression):
                    wrapped = cst.SimpleStatementLine([cst.Expr(stmt)])
                    lines.append(cst.Module(body=[wrapped]).code.strip())
                else:
                    lines.append(str(stmt).strip())
        return "\n".join(lines)


@dataclass
class Form:
    """A form represents a comment block + code statement."""

    markdown: List[str]
    node: Union[cst.CSTNode, CombinedCode]
    start_line: int
    metadata: FormMetadata = field(default_factory=FormMetadata)

    @property
    def code(self) -> str:
        """Get the source code for this form's node."""
        # Handle CombinedCode specially
        if isinstance(self.node, CombinedCode):
            return self.node.code()

        # Handle different node types properly for Module creation
        if isinstance(self.node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
            # Create a copy of the node without leading comments since they're already in markdown
            node_without_comments = _strip_leading_comments(self.node)
            return cst.Module(body=[node_without_comments]).code.strip()
        else:
            # For other node types, wrap in a SimpleStatementLine if it's an expression
            if isinstance(self.node, cst.BaseExpression):
                stmt = cst.SimpleStatementLine([cst.Expr(self.node)])
                return cst.Module(body=[stmt]).code.strip()
            # For other cases, convert to string directly
            return str(self.node).strip()

    @property
    def is_expression(self) -> bool:
        """Check if this form is a standalone expression."""
        # For CombinedCode, check if the last code element is an expression
        if isinstance(self.node, CombinedCode):
            if self.node.code_elements:
                last_stmt = self.node.code_elements[-1]
                if isinstance(last_stmt, cst.SimpleStatementLine):
                    if len(last_stmt.body) == 1:
                        return isinstance(last_stmt.body[0], cst.Expr)
            return False

        if isinstance(self.node, cst.SimpleStatementLine):
            if len(self.node.body) == 1:
                return isinstance(self.node.body[0], cst.Expr)
        return False

    @property
    def is_statement(self) -> bool:
        """Check if this form is a statement (not a standalone expression)."""
        # If it's not a dummy form and not an expression, it's a statement
        return not self.is_dummy_form and not self.is_expression

    @property
    def is_literal(self) -> bool:
        """Check if this form contains only literal values."""
        if not self.is_expression:
            return False

        if isinstance(self.node, cst.SimpleStatementLine) and len(self.node.body) == 1:
            expr = self.node.body[0]
            if isinstance(expr, cst.Expr):
                return self._is_literal_value(expr.value)
        return False

    def _is_literal_value(self, node: cst.BaseExpression) -> bool:
        """Check if a CST node represents a literal value."""
        # Simple literals
        if isinstance(
            node, (cst.Integer, cst.Float, cst.SimpleString, cst.FormattedString)
        ):
            return True

        # Boolean literals (Name nodes for True/False)
        if isinstance(node, cst.Name) and node.value in ("True", "False", "None"):
            return True

        # Unary operations on numeric literals (e.g., -42, +3.14)
        if isinstance(node, cst.UnaryOperation):
            if isinstance(node.operator, (cst.Minus, cst.Plus)):
                return self._is_literal_value(node.expression)
            return False

        # Literal collections (lists, tuples, sets, dicts with only literal contents)
        if isinstance(node, cst.List):
            return all(
                self._is_literal_value(elem.value)
                for elem in node.elements
                if isinstance(elem, cst.Element)
            )

        if isinstance(node, cst.Tuple):
            return all(
                self._is_literal_value(elem.value)
                for elem in node.elements
                if isinstance(elem, cst.Element)
            )

        if isinstance(node, cst.Set):
            return all(
                self._is_literal_value(elem.value)
                for elem in node.elements
                if isinstance(elem, cst.Element)
            )

        if isinstance(node, cst.Dict):
            return all(
                self._is_literal_value(elem.key) and self._is_literal_value(elem.value)
                for elem in node.elements
                if isinstance(elem, cst.DictElement) and elem.key is not None
            )

        # Bytes literals and concatenated strings
        if isinstance(node, cst.ConcatenatedString):
            return all(
                isinstance(part, (cst.SimpleString, cst.FormattedString))
                for part in [node.left, node.right]
            )

        return False

    @property
    def is_dummy_form(self) -> bool:
        """Check if this form is a dummy form (markdown-only with pass statement)."""
        # Check if the node is a SimpleStatementLine with a single Pass statement
        if isinstance(self.node, cst.SimpleStatementLine):
            if len(self.node.body) == 1 and isinstance(self.node.body[0], cst.Pass):
                return True

        # Check if it's a CombinedCode with only dummy pass statements
        if isinstance(self.node, CombinedCode):
            for stmt in self.node.code_elements:
                if isinstance(stmt, cst.SimpleStatementLine):
                    if len(stmt.body) == 1 and isinstance(stmt.body[0], cst.Pass):
                        continue  # This is a dummy pass
                    else:
                        return False  # Has real code
                else:
                    return False  # Has real code
            return True  # All statements are dummy passes

        return False


# New clean parser implementation
@dataclass
class RawElement:
    """A single element from the source file."""

    type: Literal["comment", "pragma", "code", "blank_line"]
    content: Union[
        str, cst.CSTNode
    ]  # Can be string for comments/pragmas or CSTNode for code
    line_number: int


@dataclass
class RawForm:
    """A form before metadata processing."""

    markdown_lines: List[str]
    code_statements: List[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]]
    pragma_comments: List[str]  # Only pragmas directly associated with this form
    start_line: int
    empty_comment_positions: List[int] = field(default_factory=list)


def extract_raw_elements(source_code: str) -> List[RawElement]:
    """Step 1: Extract all elements from source code."""
    elements = []

    # Parse with LibCST to get the AST
    module = cst.parse_module(source_code)

    # Enable position tracking
    wrapper = MetadataWrapper(module)
    positions = wrapper.resolve(PositionProvider)

    # First, extract header comments
    current_line = 1
    if hasattr(module, "header") and module.header:
        for line in module.header:
            if line.comment:
                comment_text = line.comment.value.lstrip("#").strip()
                if _is_pragma_comment(comment_text):
                    elements.append(RawElement("pragma", comment_text, current_line))
                else:
                    elements.append(RawElement("comment", comment_text, current_line))
            elif line.whitespace.value.strip() == "":
                elements.append(RawElement("blank_line", "", current_line))
            current_line += 1

    # Then extract statements and their leading comments
    for stmt in module.body:
        # Get the position of this statement
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
                    comment_text = line.comment.value.lstrip("#").strip()
                    if _is_pragma_comment(comment_text):
                        elements.append(
                            RawElement("pragma", comment_text, comment_line)
                        )
                    else:
                        elements.append(
                            RawElement("comment", comment_text, comment_line)
                        )
                elif line.whitespace.value.strip() == "":
                    elements.append(RawElement("blank_line", "", comment_line))
                comment_line += 1

        # Add the code statement
        elements.append(RawElement("code", stmt, stmt_line_num))

        # Update current line for next iteration
        if stmt_line:
            current_line = stmt_line.end.line + 1

    return elements


def group_into_forms(elements: List[RawElement]) -> List[RawForm]:
    """Step 2: Group elements into forms with clear rules."""
    forms = []

    # Skip file-level pragmas at the beginning
    # File-level pragmas are those at the top followed by at least one empty line
    i = 0

    # Count consecutive pragmas at the beginning
    while i < len(elements) and elements[i].type == "pragma":
        i += 1

    # Check if these pragmas are followed by a blank line
    if i > 0 and i < len(elements) and elements[i].type == "blank_line":
        # These are file-level pragmas, skip past the blank line
        i += 1
    else:
        # No blank line after pragmas, so they're not file-level
        # Reset to beginning
        i = 0

    while i < len(elements):
        current_markdown = []
        current_pragmas = []
        current_code = []
        start_line = None  # Track the line where this form starts

        # Collect comments and pragmas
        while i < len(elements) and elements[i].type in [
            "comment",
            "pragma",
            "blank_line",
        ]:
            elem = elements[i]
            if start_line is None and elem.type != "blank_line":
                start_line = elem.line_number
            if elem.type == "comment":
                current_markdown.append(elem.content)
            elif elem.type == "pragma":
                current_pragmas.append(elem.content)
            elif elem.type == "blank_line":
                if current_markdown and current_markdown[-1]:  # Add paragraph break
                    current_markdown.append("")
            i += 1

        # Collect code statements, allowing empty comments as continuations
        empty_comment_positions = []
        while i < len(elements):
            if elements[i].type == "code":
                if start_line is None:
                    start_line = elements[i].line_number
                stmt = elements[i].content
                if isinstance(
                    stmt, (cst.SimpleStatementLine, cst.BaseCompoundStatement)
                ):
                    current_code.append(stmt)
                i += 1
            elif (
                elements[i].type == "comment"
                and isinstance(elements[i].content, str)
                and elements[i].content == ""
            ):
                # Empty comment acts as continuation - record position for blank line
                if current_code:  # Only track if we have code already
                    empty_comment_positions.append(len(current_code))
                i += 1
            else:
                # Non-empty comment or other element - stop collecting code
                break

        # Create form
        if current_markdown or current_code:
            # Default start line if we haven't found one yet
            if start_line is None:
                start_line = 1

            # If we have code, create a regular form
            if current_code:
                forms.append(
                    RawForm(
                        markdown_lines=current_markdown,
                        code_statements=current_code,
                        pragma_comments=current_pragmas,
                        start_line=start_line,
                        empty_comment_positions=empty_comment_positions,
                    )
                )
            # If we have only markdown, create a dummy form
            elif current_markdown:
                dummy_stmt = cst.SimpleStatementLine([cst.Pass()])
                forms.append(
                    RawForm(
                        markdown_lines=current_markdown,
                        code_statements=[dummy_stmt],
                        pragma_comments=current_pragmas,
                        start_line=start_line,
                    )
                )

    return forms


def parse_file_metadata_clean(elements: List[RawElement]) -> FileMetadata:
    """Extract file-level metadata from elements at the start."""
    metadata = FileMetadata()

    # File-level pragmas are those at the top followed by at least one empty line
    i = 0

    # Collect consecutive pragmas at the beginning
    pragma_contents = []
    while i < len(elements) and elements[i].type == "pragma":
        if isinstance(elements[i].content, str):
            pragma_contents.append(elements[i].content)
        i += 1

    # Check if these pragmas are followed by a blank line
    if i < len(elements) and elements[i].type == "blank_line":
        # These are file-level pragmas
        for pragma_content in pragma_contents:
            content = _extract_pragma_content(pragma_content)
            tags = _extract_pragma_tags(content)
            metadata.pragma_tags.update(tags)

    return metadata


def apply_metadata_clean(raw_forms: List[RawForm]) -> List[Form]:
    """Step 3: Convert RawForm to Form with proper metadata."""
    forms = []

    for raw_form in raw_forms:
        # Parse form-level metadata from pragma comments
        form_metadata = FormMetadata()
        for pragma in raw_form.pragma_comments:
            if isinstance(pragma, str):
                pragma_content = _extract_pragma_content(pragma)
                tags = _extract_pragma_tags(pragma_content)
                form_metadata.pragma_tags.update(tags)

        # Convert code statements to proper node
        if len(raw_form.code_statements) == 1:
            node = raw_form.code_statements[0]
        else:
            # Multiple statements - need to combine them
            node = CombinedCode(
                raw_form.code_statements, raw_form.empty_comment_positions
            )

        # Create final form
        form = Form(
            markdown=raw_form.markdown_lines,
            node=node,
            start_line=raw_form.start_line,
            metadata=form_metadata,
        )
        forms.append(form)

    return forms


def parse_colight_file(file_path: pathlib.Path) -> tuple[List[Form], FileMetadata]:
    """Parse a .colight.py file and extract forms and metadata."""
    source_code = file_path.read_text(encoding="utf-8")

    # Step 1: Extract raw elements
    elements = extract_raw_elements(source_code)

    # Step 2: Parse file metadata
    file_metadata = parse_file_metadata_clean(elements)

    # Step 3: Group into forms
    raw_forms = group_into_forms(elements)

    # Step 4: Apply metadata
    forms = apply_metadata_clean(raw_forms)

    return forms, file_metadata


def parse_file_metadata(source_code: str) -> FileMetadata:
    """Parse file-level pragma annotations from source code."""
    # Use the new clean implementation
    elements = extract_raw_elements(source_code)
    return parse_file_metadata_clean(elements)


def is_colight_file(file_path: pathlib.Path) -> bool:
    """Check if a file is a .colight.py file."""
    return file_path.suffix == ".py" and ".colight" in file_path.name
