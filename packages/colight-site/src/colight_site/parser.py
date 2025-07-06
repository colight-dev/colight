"""Parse .colight.py files using LibCST."""

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider
from dataclasses import dataclass, field
from typing import List, Union, Optional, Literal, Tuple
import pathlib
import re


def _extract_pragma(text: str) -> set[str]:
    """Extract all pragma tags from text using a single regex pattern."""
    # Pattern to match hide/show tags (no format tags)
    # Also matches hide-all-* for file-level pragmas
    tags = set(re.findall(r"\b(?:hide|show)(?:-all)?-\w+\b", text.lower()))

    # Normalize to consistent forms
    normalized = set()
    for tag in tags:
        # Convert singular to plural for consistency
        if tag.endswith(("statement", "visual")) and not tag.startswith("hide-all"):
            normalized.add(tag + "s")
        else:
            normalized.add(tag)

    return normalized


def should_show_statements(tags: set[str]) -> bool:
    """Check if statements should be shown based on tags."""
    # show- tags override hide- tags
    if "show-statements" in tags:
        return True
    return not ("hide-statements" in tags or "hide-all-statements" in tags)


def should_show_visuals(tags: set[str]) -> bool:
    """Check if visuals should be shown based on tags."""
    # show- tags override hide- tags
    if "show-visuals" in tags:
        return True
    return not ("hide-visuals" in tags or "hide-all-visuals" in tags)


def should_show_code(tags: set[str]) -> bool:
    """Check if code should be shown based on tags."""
    # show- tags override hide- tags
    if "show-code" in tags:
        return True
    return not ("hide-code" in tags or "hide-all-code" in tags)


def should_show_prose(tags: set[str]) -> bool:
    """Check if prose (markdown content) should be shown based on tags."""
    # show- tags override hide- tags
    if "show-prose" in tags:
        return True
    return not ("hide-prose" in tags or "hide-all-prose" in tags)


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


def _is_pragma_comment(comment_text: str) -> bool:
    """Check if a comment is a pragma comment.

    Accepts comments starting with | or %% as pragma starters.
    Handles variations like #|, # |, #%%, # %%
    """
    # Remove leading whitespace after stripping
    text = comment_text.strip()

    # Check for pragma starters
    return text.startswith("|") or text.startswith("%")


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
class FormElement:
    """A single element within a form."""
    
    type: Literal["prose", "statement", "expression"]
    content: Union[str, cst.CSTNode, List[cst.CSTNode], "CombinedCode"]  # str for prose, CSTNode(s) for code
    line_number: int
    
    def get_code(self) -> str:
        """Get the source code for this element."""
        if self.type == "prose":
            return self.content if isinstance(self.content, str) else ""
        
        # Handle CombinedCode specially
        if isinstance(self.content, CombinedCode):
            return self.content.code()
        
        # Handle code elements
        if isinstance(self.content, list):
            # Multiple statements combined
            lines = []
            for node in self.content:
                if isinstance(node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
                    stripped = _strip_leading_comments(node)
                    lines.append(cst.Module(body=[stripped]).code.strip())
                else:
                    lines.append(str(node).strip())
            return "\n".join(lines)
        elif isinstance(self.content, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
            stripped = _strip_leading_comments(self.content)
            return cst.Module(body=[stripped]).code.strip()
        elif isinstance(self.content, cst.BaseExpression):
            stmt = cst.SimpleStatementLine([cst.Expr(self.content)])
            return cst.Module(body=[stmt]).code.strip()
        else:
            return str(self.content).strip()


@dataclass
class FormMetadata:
    """Metadata extracted from per-form pragma annotations."""

    pragma: set[str] = field(default_factory=set)

    def resolve_with_defaults(self, default_tags: set[str]) -> set[str]:
        """Resolve form metadata with default tags."""
        # Form-specific tags override defaults
        return self.pragma if self.pragma else default_tags


@dataclass
class FileMetadata:
    """Metadata extracted from file-level pragma annotations."""

    pragma: set[str] = field(default_factory=set)


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


def _is_expression_node(node: Union[cst.CSTNode, cst.BaseExpression]) -> bool:
    """Check if a node represents an expression."""
    if isinstance(node, cst.SimpleStatementLine):
        if len(node.body) == 1 and isinstance(node.body[0], cst.Expr):
            return True
    elif isinstance(node, cst.BaseExpression):
        return True
    return False


def _is_literal_value(node: cst.BaseExpression) -> bool:
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
            return _is_literal_value(node.expression)
        return False

    # Literal collections (lists, tuples, sets, dicts with only literal contents)
    if isinstance(node, cst.List):
        return all(
            _is_literal_value(elem.value)
            for elem in node.elements
            if isinstance(elem, cst.Element)
        )

    if isinstance(node, cst.Tuple):
        return all(
            _is_literal_value(elem.value)
            for elem in node.elements
            if isinstance(elem, cst.Element)
        )

    if isinstance(node, cst.Set):
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

    # Bytes literals and concatenated strings
    if isinstance(node, cst.ConcatenatedString):
        return all(
            isinstance(part, (cst.SimpleString, cst.FormattedString))
            for part in [node.left, node.right]
        )

    return False


@dataclass
class Form:
    """A form represents a group of related content (prose, statements, expressions)."""

    elements: List[FormElement]
    metadata: FormMetadata = field(default_factory=FormMetadata)
    start_line: int = 1

    @property
    def last_element(self) -> Optional[FormElement]:
        """Get the last element in the form."""
        return self.elements[-1] if self.elements else None
    
    def should_show_element(self, element: FormElement) -> bool:
        """Determine if element should be shown based on form pragma."""
        pragma = self.metadata.pragma
        
        if element.type == "prose":
            return should_show_prose(pragma)
        elif element.type == "statement":
            # Show statements if both code and statements are shown
            return should_show_code(pragma) and should_show_statements(pragma)
        elif element.type == "expression":
            # Show expressions if code is shown (NOT affected by hide-statements)
            return should_show_code(pragma)
        
        return True
    
    def should_show_visual(self) -> bool:
        """Check if visuals should be shown for expressions in this form."""
        return should_show_visuals(self.metadata.pragma)

    @property
    def markdown(self) -> List[str]:
        """Get markdown content for backward compatibility."""
        lines = []
        for elem in self.elements:
            if elem.type == "prose" and isinstance(elem.content, str):
                lines.extend(elem.content.split("\n"))
        return lines
    
    def get_all_code(self) -> str:
        """Get all code content as a string (for testing/debugging only)."""
        code_parts = []
        for elem in self.elements:
            if elem.type in ("statement", "expression"):
                code = elem.get_code()
                if code:
                    code_parts.append(code)
        return "\n".join(code_parts)

    @property
    def code(self) -> str:
        """DEPRECATED: Use elements directly or get_all_code() for testing."""
        return self.get_all_code()

    @property
    def is_expression(self) -> bool:
        """Check if the last element is an expression (backward compatibility)."""
        last = self.last_element
        return last is not None and last.type == "expression"

    @property
    def is_statement(self) -> bool:
        """Check if the last element is a statement (backward compatibility)."""
        last = self.last_element
        return last is not None and last.type == "statement"

    @property
    def is_dummy_form(self) -> bool:
        """Check if this form has no real code content."""
        for elem in self.elements:
            if elem.type in ("statement", "expression"):
                # Check if it's a pass statement
                if isinstance(elem.content, cst.SimpleStatementLine):
                    if len(elem.content.body) == 1 and isinstance(elem.content.body[0], cst.Pass):
                        continue
                return False
        return True

    @property
    def is_literal(self) -> bool:
        """Check if the last element is a literal expression."""
        last = self.last_element
        if last is None or last.type != "expression":
            return False
            
        node = last.content
        if isinstance(node, cst.SimpleStatementLine) and len(node.body) == 1:
            expr = node.body[0]
            if isinstance(expr, cst.Expr):
                return _is_literal_value(expr.value)
        elif isinstance(node, cst.BaseExpression):
            return _is_literal_value(node)
        return False

    @property
    def node(self) -> Union[cst.CSTNode, "CombinedCode", None]:
        """Get the code node for backward compatibility."""
        code_elements = [elem for elem in self.elements if elem.type in ("statement", "expression")]
        if not code_elements:
            # Return a dummy pass statement for forms with no code
            return cst.SimpleStatementLine([cst.Pass()])
        elif len(code_elements) == 1:
            return code_elements[0].content
        else:
            # Multiple code elements - return CombinedCode
            nodes = []
            for elem in code_elements:
                if isinstance(elem.content, list):
                    nodes.extend(elem.content)
                else:
                    nodes.append(elem.content)
            return CombinedCode(nodes)


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
    in_pep723_block = False

    if hasattr(module, "header") and module.header:
        for line in module.header:
            if line.comment:
                comment_text = line.comment.value.lstrip("#").strip()

                # Check for PEP 723 markers
                if comment_text == "/// script":
                    in_pep723_block = True
                    current_line += 1
                    continue
                elif comment_text == "///" and in_pep723_block:
                    in_pep723_block = False
                    current_line += 1
                    continue

                # Skip content inside PEP 723 block
                if in_pep723_block:
                    current_line += 1
                    continue

                # Normal comment processing
                if _is_pragma_comment(comment_text):
                    elements.append(RawElement("pragma", comment_text, current_line))
                else:
                    elements.append(RawElement("comment", comment_text, current_line))
            elif line.whitespace.value.strip() == "":
                # Skip blank lines in PEP 723 block
                if not in_pep723_block:
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

    # Start from the beginning - no need to skip file-level pragmas
    # since they're identified by hide-all-* prefix
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
    """Extract file-level metadata from elements.

    File-level pragmas use hide-all-* prefix and can appear anywhere before code.
    """
    pragma = set()

    # Look for hide-all-* pragmas anywhere before the first code element
    for elem in elements:
        if elem.type == "code":
            # Stop when we hit code
            break
        elif elem.type == "pragma" and isinstance(elem.content, str):
            content = _extract_pragma_content(elem.content)
            tags = _extract_pragma(content)
            # Only keep hide-all-* tags for file metadata
            file_level_tags = {tag for tag in tags if tag.startswith("hide-all-")}
            pragma.update(file_level_tags)

    return FileMetadata(pragma)


def _classify_code_node(node: cst.CSTNode) -> Literal["statement", "expression"]:
    """Classify a code node as statement or expression."""
    if isinstance(node, cst.SimpleStatementLine):
        if len(node.body) == 1 and isinstance(node.body[0], cst.Expr):
            return "expression"
    return "statement"


def apply_metadata_clean(raw_forms: List[RawForm]) -> List[Form]:
    """Step 3: Convert RawForm to Form with proper metadata."""
    forms = []

    for raw_form in raw_forms:
        # Parse form-level metadata from pragma comments
        form_metadata = FormMetadata()
        for pragma in raw_form.pragma_comments:
            if isinstance(pragma, str):
                pragma_content = _extract_pragma_content(pragma)
                tags = _extract_pragma(pragma_content)
                form_metadata.pragma.update(tags)

        # Build elements list
        elements = []
        
        # Add prose element if present
        if raw_form.markdown_lines:
            prose_content = "\n".join(raw_form.markdown_lines)
            prose_elem = FormElement(
                type="prose",
                content=prose_content,
                line_number=raw_form.start_line
            )
            elements.append(prose_elem)
        
        # Add code elements
        if raw_form.code_statements:
            # Determine if we should group statements or keep them separate
            # For now, group consecutive statements of the same type
            if len(raw_form.code_statements) == 1:
                # Single statement - classify and add
                stmt = raw_form.code_statements[0]
                elem_type = _classify_code_node(stmt)
                code_elem = FormElement(
                    type=elem_type,
                    content=stmt,
                    line_number=raw_form.start_line + len(raw_form.markdown_lines) + 1
                )
                elements.append(code_elem)
            else:
                # Multiple statements - check if we can group them
                all_statements = all(
                    _classify_code_node(stmt) == "statement" 
                    for stmt in raw_form.code_statements
                )
                
                if all_statements:
                    # All are statements - group them
                    # Use CombinedCode if we have empty comment positions to preserve
                    if raw_form.empty_comment_positions:
                        combined = CombinedCode(raw_form.code_statements, raw_form.empty_comment_positions)
                        code_elem = FormElement(
                            type="statement",
                            content=combined,
                            line_number=raw_form.start_line + len(raw_form.markdown_lines) + 1
                        )
                    else:
                        code_elem = FormElement(
                            type="statement",
                            content=raw_form.code_statements,
                            line_number=raw_form.start_line + len(raw_form.markdown_lines) + 1
                        )
                    elements.append(code_elem)
                else:
                    # Mixed types or expressions - need to preserve empty comment positions
                    if raw_form.empty_comment_positions:
                        # Use CombinedCode to preserve blank lines
                        combined = CombinedCode(raw_form.code_statements, raw_form.empty_comment_positions)
                        # Determine type based on last statement
                        last_type = _classify_code_node(raw_form.code_statements[-1])
                        code_elem = FormElement(
                            type=last_type,  # Use the type of the last element
                            content=combined,
                            line_number=raw_form.start_line + len(raw_form.markdown_lines) + 1
                        )
                        elements.append(code_elem)
                    else:
                        # No empty comments - add separately
                        line_offset = raw_form.start_line + len(raw_form.markdown_lines) + 1
                        for i, stmt in enumerate(raw_form.code_statements):
                            elem_type = _classify_code_node(stmt)
                            code_elem = FormElement(
                                type=elem_type,
                                content=stmt,
                                line_number=line_offset + i
                            )
                            elements.append(code_elem)

        # Create final form with elements
        form = Form(
            elements=elements,
            metadata=form_metadata,
            start_line=raw_form.start_line
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
