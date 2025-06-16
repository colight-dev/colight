"""Parse .colight.py files using LibCST."""

import libcst as cst
from dataclasses import dataclass
from typing import List
import pathlib


@dataclass
class Form:
    """A form represents a comment block + code statement."""

    markdown: List[str]
    node: cst.CSTNode
    start_line: int

    @property
    def code(self) -> str:
        """Get the source code for this form's node."""
        # Handle different node types properly for Module creation
        if isinstance(self.node, (cst.SimpleStatementLine, cst.BaseCompoundStatement)):
            return cst.Module([self.node]).code.strip()
        else:
            # For other node types, wrap in a SimpleStatementLine if it's an expression
            if isinstance(self.node, cst.BaseExpression):
                stmt = cst.SimpleStatementLine([cst.Expr(self.node)])
                return cst.Module([stmt]).code.strip()
            # For other cases, convert to string directly
            return str(self.node).strip()

    @property
    def is_expression(self) -> bool:
        """Check if this form is a standalone expression."""
        if isinstance(self.node, cst.SimpleStatementLine):
            if len(self.node.body) == 1:
                return isinstance(self.node.body[0], cst.Expr)
        return False


class FormExtractor(cst.CSTVisitor):
    """Extract forms (comment + code blocks) from a CST."""

    def __init__(self):
        self.forms: List[Form] = []
        self.pending_markdown: List[str] = []
        self.current_line = 1

    def visit_Module(self, node: cst.Module) -> None:
        """Process the module body."""
        # First, handle any header comments
        if hasattr(node, "header") and node.header:
            for line in node.header:
                if line.comment:
                    comment_text = line.comment.value.lstrip("#").strip()
                    self.pending_markdown.append(comment_text)

        for stmt in node.body:
            self._process_statement(stmt)

        # Handle any remaining markdown at the end
        if self.pending_markdown:
            # Create a dummy form for trailing comments
            dummy_node = cst.SimpleStatementLine([cst.Pass()])
            self.forms.append(
                Form(
                    markdown=self.pending_markdown.copy(),
                    node=dummy_node,
                    start_line=self.current_line,
                )
            )
            self.pending_markdown.clear()

    def _process_statement(self, stmt: cst.CSTNode) -> None:
        """Process a single statement and its leading comments."""
        # Extract leading comments
        leading_lines = []
        if hasattr(stmt, "leading_lines"):
            leading_lines_attr = getattr(stmt, "leading_lines", None)
            if leading_lines_attr is not None:
                leading_lines = leading_lines_attr

        # Process comments in leading lines
        for line in leading_lines:
            if line.comment:
                comment_text = line.comment.value.lstrip("#").strip()
                if (
                    comment_text or self.pending_markdown
                ):  # Keep empty comment lines if we have content
                    self.pending_markdown.append(comment_text)
            elif line.whitespace.value.strip() == "":
                # Empty line - split markdown blocks only if it's truly empty
                # (no comment and only whitespace)
                if self.pending_markdown and self.pending_markdown[-1]:
                    self.pending_markdown.append("")  # Add paragraph break

        # Create form with collected markdown
        form = Form(
            markdown=self.pending_markdown.copy(),
            node=stmt,
            start_line=self.current_line,
        )
        self.forms.append(form)
        self.pending_markdown.clear()

        # Update line counter (approximate)
        if hasattr(stmt, "body"):
            self.current_line += len(str(stmt).split("\n"))
        else:
            self.current_line += 1


def parse_colight_file(file_path: pathlib.Path) -> List[Form]:
    """Parse a .colight.py file and extract forms."""
    source_code = file_path.read_text(encoding="utf-8")

    # Parse with LibCST
    module = cst.parse_module(source_code)

    # Extract forms
    extractor = FormExtractor()
    module.visit(extractor)

    return extractor.forms


def is_colight_file(file_path: pathlib.Path) -> bool:
    """Check if a file is a .colight.py file."""
    return file_path.suffix == ".py" and ".colight" in file_path.name
