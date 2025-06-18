"""Parse .colight.py files using LibCST."""

import libcst as cst
from dataclasses import dataclass, field
from typing import List, Union, Optional, Literal
import pathlib


@dataclass
class FormMetadata:
    """Metadata extracted from per-form pragma annotations."""

    hide_statements: Optional[bool] = None  # None means inherit from parent
    hide_visuals: Optional[bool] = None
    hide_code: Optional[bool] = None

    def resolve_with_defaults(
        self,
        default_hide_statements: bool = False,
        default_hide_visuals: bool = False,
        default_hide_code: bool = False,
    ) -> dict:
        """Resolve form metadata with default values."""
        return {
            "hide_statements": self.hide_statements
            if self.hide_statements is not None
            else default_hide_statements,
            "hide_visuals": self.hide_visuals
            if self.hide_visuals is not None
            else default_hide_visuals,
            "hide_code": self.hide_code
            if self.hide_code is not None
            else default_hide_code,
        }


@dataclass
class FileMetadata:
    """Metadata extracted from file-level pragma annotations."""

    hide_statements: bool = False
    hide_visuals: bool = False
    hide_code: bool = False
    format: Optional[str] = None  # 'html' or 'markdown'

    def merge_with_cli_options(
        self,
        hide_statements: bool = False,
        hide_visuals: bool = False,
        hide_code: bool = False,
        format: Optional[str] = None,
        mostly_prose: bool = False,
    ) -> dict:
        """Merge file metadata with CLI options. CLI options take precedence."""
        # Start with file metadata
        result = {
            "hide_statements": self.hide_statements,
            "hide_visuals": self.hide_visuals,
            "hide_code": self.hide_code,
            "format": self.format,
        }

        # CLI options override file metadata
        if hide_statements:
            result["hide_statements"] = True
        if hide_visuals:
            result["hide_visuals"] = True
        if hide_code:
            result["hide_code"] = True
        if format:
            result["format"] = format
        if mostly_prose:
            result["hide_statements"] = True
            result["hide_code"] = True

        return result


class CombinedStatements:
    """A pseudo-node that combines multiple consecutive statements."""

    def __init__(
        self,
        statements: List[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]],
    ):
        self.statements = statements

    def code(self) -> str:
        """Generate combined code from all statements."""
        lines = []
        for stmt in self.statements:
            if isinstance(stmt, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
                # Strip leading comments since they're handled separately
                node_without_comments = self._strip_leading_comments(stmt)
                lines.append(cst.Module(body=[node_without_comments]).code.strip())
            else:
                # For other node types, wrap in a SimpleStatementLine if it's an expression
                if isinstance(stmt, cst.BaseExpression):
                    wrapped = cst.SimpleStatementLine([cst.Expr(stmt)])
                    lines.append(cst.Module(body=[wrapped]).code.strip())
                else:
                    lines.append(str(stmt).strip())
        return "\n".join(lines)

    def _strip_leading_comments(
        self, node: Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]
    ) -> Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]:
        """Create a copy of the node without leading comments."""
        if isinstance(node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
            # Filter out comment lines, keeping only whitespace-only lines
            new_leading_lines = []
            for line in node.leading_lines:
                if not line.comment:
                    new_leading_lines.append(line)

            # Create a new node with filtered leading lines
            return node.with_changes(leading_lines=new_leading_lines)
        return node


@dataclass
class Form:
    """A form represents a comment block + code statement."""

    markdown: List[str]
    node: Union[cst.CSTNode, CombinedStatements]
    start_line: int
    metadata: FormMetadata = field(default_factory=FormMetadata)

    @property
    def code(self) -> str:
        """Get the source code for this form's node."""
        # Handle CombinedStatements specially
        if isinstance(self.node, CombinedStatements):
            return self.node.code()

        # Handle different node types properly for Module creation
        if isinstance(self.node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
            # Create a copy of the node without leading comments since they're already in markdown
            node_without_comments = self._strip_leading_comments(self.node)
            return cst.Module(body=[node_without_comments]).code.strip()
        else:
            # For other node types, wrap in a SimpleStatementLine if it's an expression
            if isinstance(self.node, cst.BaseExpression):
                stmt = cst.SimpleStatementLine([cst.Expr(self.node)])
                return cst.Module(body=[stmt]).code.strip()
            # For other cases, convert to string directly
            return str(self.node).strip()

    def _strip_leading_comments(
        self, node: Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]
    ) -> Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]:
        """Create a copy of the node without leading comments."""
        if isinstance(node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
            # Filter out comment lines, keeping only whitespace-only lines
            new_leading_lines = []
            for line in node.leading_lines:
                if not line.comment:
                    new_leading_lines.append(line)

            # Create a new node with filtered leading lines
            return node.with_changes(leading_lines=new_leading_lines)
        return node

    @property
    def is_expression(self) -> bool:
        """Check if this form is a standalone expression."""
        # CombinedStatements are never single expressions
        if isinstance(self.node, CombinedStatements):
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

        # Check if it's a CombinedStatements with only dummy pass statements
        if isinstance(self.node, CombinedStatements):
            for stmt in self.node.statements:
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


def extract_raw_elements(source_code: str) -> List[RawElement]:
    """Step 1: Extract all elements from source code."""
    elements = []

    # Parse with LibCST to get the AST
    module = cst.parse_module(source_code)

    # First, extract header comments
    if hasattr(module, "header") and module.header:
        for line in module.header:
            if line.comment:
                comment_text = line.comment.value.lstrip("#").strip()
                if comment_text.strip().startswith("|") and "colight:" in comment_text:
                    elements.append(
                        RawElement("pragma", comment_text, 1)
                    )  # TODO: real line numbers
                else:
                    elements.append(RawElement("comment", comment_text, 1))
            elif line.whitespace.value.strip() == "":
                elements.append(RawElement("blank_line", "", 1))

    # Then extract statements and their leading comments
    for stmt in module.body:
        # Extract leading comments
        if hasattr(stmt, "leading_lines"):
            for line in stmt.leading_lines:
                if line.comment:
                    comment_text = line.comment.value.lstrip("#").strip()
                    if (
                        comment_text.strip().startswith("|")
                        and "colight:" in comment_text
                    ):
                        elements.append(RawElement("pragma", comment_text, 1))
                    else:
                        elements.append(RawElement("comment", comment_text, 1))
                elif line.whitespace.value.strip() == "":
                    elements.append(RawElement("blank_line", "", 1))

        # Add the code statement
        elements.append(RawElement("code", stmt, 1))

    return elements


def group_into_forms(elements: List[RawElement]) -> List[RawForm]:
    """Step 2: Group elements into forms with clear rules."""
    forms = []

    i = 0
    while i < len(elements):
        current_markdown = []
        current_pragmas = []
        current_code = []
        start_line = 1  # TODO: track real line numbers

        # Collect comments and pragmas
        while i < len(elements) and elements[i].type in [
            "comment",
            "pragma",
            "blank_line",
        ]:
            elem = elements[i]
            if elem.type == "comment":
                current_markdown.append(elem.content)
            elif elem.type == "pragma":
                current_pragmas.append(elem.content)
            elif elem.type == "blank_line":
                if current_markdown and current_markdown[-1]:  # Add paragraph break
                    current_markdown.append("")
            i += 1

        # Collect consecutive code statements
        while i < len(elements) and elements[i].type == "code":
            stmt = elements[i].content
            if isinstance(stmt, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
                current_code.append(stmt)
            i += 1

        # Create form
        if current_markdown or current_code:
            # If we have code, create a regular form
            if current_code:
                forms.append(
                    RawForm(
                        markdown_lines=current_markdown,
                        code_statements=current_code,
                        pragma_comments=current_pragmas,
                        start_line=start_line,
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

    # Only process consecutive pragma elements at the very beginning
    in_file_pragma_section = True

    for elem in elements:
        if in_file_pragma_section and elem.type == "pragma":
            if isinstance(elem.content, str):
                options_str = elem.content.split("colight:", 1)[1].strip()
                options = _parse_pragma_options(options_str)

                # Apply options to metadata
                if options["hide_statements"] is not None:
                    metadata.hide_statements = options["hide_statements"]
                if options["hide_visuals"] is not None:
                    metadata.hide_visuals = options["hide_visuals"]
                if options["hide_code"] is not None:
                    metadata.hide_code = options["hide_code"]
                if options["format"] is not None:
                    metadata.format = options["format"]
                if options["mostly_prose"]:
                    metadata.hide_statements = True
                    metadata.hide_code = True
        elif elem.type in ["comment", "blank_line"]:
            # Comments and blank lines end the file pragma section
            in_file_pragma_section = False
        else:
            # Hit code or other content, definitely done with file pragmas
            break

    return metadata


def apply_metadata_clean(raw_forms: List[RawForm]) -> List[Form]:
    """Step 3: Convert RawForm to Form with proper metadata."""
    forms = []

    for raw_form in raw_forms:
        # Parse form-level metadata from pragma comments
        form_metadata = FormMetadata()
        for pragma in raw_form.pragma_comments:
            if isinstance(pragma, str) and "colight:" in pragma:
                options_str = pragma.split("colight:", 1)[1].strip()
                options = _parse_pragma_options(options_str)

                # Apply options to form metadata (no format or mostly_prose for forms)
                if options["hide_statements"] is not None:
                    form_metadata.hide_statements = options["hide_statements"]
                if options["hide_visuals"] is not None:
                    form_metadata.hide_visuals = options["hide_visuals"]
                if options["hide_code"] is not None:
                    form_metadata.hide_code = options["hide_code"]

        # Convert code statements to proper node
        if len(raw_form.code_statements) == 1:
            node = raw_form.code_statements[0]
        else:
            # Multiple statements - need to combine them
            node = CombinedStatements(raw_form.code_statements)

        # Create final form
        form = Form(
            markdown=raw_form.markdown_lines,
            node=node,
            start_line=raw_form.start_line,
            metadata=form_metadata,
        )
        forms.append(form)

    return forms


def _parse_pragma_options(options_str: str) -> dict:
    """Parse pragma options string into a dictionary of settings.

    Supports both hide- and show- options. show- options override hide- options.
    """
    result = {
        "hide_statements": None,
        "hide_visuals": None,
        "hide_code": None,
        "format": None,
        "mostly_prose": False,
    }

    # Split by comma and process each option
    options = [opt.strip() for opt in options_str.split(",")]

    for option in options:
        option = option.strip()

        if option == "hide-statements":
            result["hide_statements"] = True
        elif option == "show-statements":
            result["hide_statements"] = False
        elif option == "hide-visuals":
            result["hide_visuals"] = True
        elif option == "show-visuals":
            result["hide_visuals"] = False
        elif option == "hide-code":
            result["hide_code"] = True
        elif option == "show-code":
            result["hide_code"] = False
        elif option == "mostly-prose":
            result["mostly_prose"] = True
        elif option == "format-html":
            result["format"] = "html"
        elif option == "format-markdown":
            result["format"] = "markdown"
        # Silently ignore unknown options for forward compatibility

    return result


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
