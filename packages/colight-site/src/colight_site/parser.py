"""Parse .colight.py files using LibCST."""

import libcst as cst
from dataclasses import dataclass
from typing import List, Union
import pathlib


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


class FormExtractor(cst.CSTVisitor):
    """Extract forms (comment + code blocks) from a CST."""

    def __init__(self):
        self.forms: List[Form] = []
        self.pending_markdown: List[str] = []
        self.current_line = 1
        self.pending_statements: List[
            Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]
        ] = []  # For grouping consecutive statements

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

        # Handle any remaining statements and markdown at the end
        self._flush_pending_statements()
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

        # Post-process forms to group consecutive code blocks
        self.forms = self._merge_consecutive_forms(self.forms)

    def _process_statement(
        self, stmt: Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]
    ) -> None:
        """Process a single statement and its leading comments."""
        # Extract leading comments
        leading_lines = []
        if hasattr(stmt, "leading_lines"):
            leading_lines_attr = getattr(stmt, "leading_lines", None)
            if leading_lines_attr is not None:
                leading_lines = leading_lines_attr

        # Process comments in leading lines
        has_new_markdown = False
        for line in leading_lines:
            if line.comment:
                comment_text = line.comment.value.lstrip("#").strip()
                if (
                    comment_text or self.pending_markdown
                ):  # Keep empty comment lines if we have content
                    self.pending_markdown.append(comment_text)
                    has_new_markdown = True
            elif line.whitespace.value.strip() == "":
                # Empty line - split markdown blocks only if it's truly empty
                # (no comment and only whitespace)
                if self.pending_markdown and self.pending_markdown[-1]:
                    self.pending_markdown.append("")  # Add paragraph break

        # Decide whether to start a new form or group with previous statements
        if has_new_markdown or self.pending_markdown:
            # New markdown found - flush any pending statements first
            self._flush_pending_statements()

            # Create form with markdown + this statement
            form = Form(
                markdown=self.pending_markdown.copy(),
                node=stmt,
                start_line=self.current_line,
            )
            self.forms.append(form)
            self.pending_markdown.clear()
        else:
            # No new markdown - add to pending statements for grouping
            self.pending_statements.append(stmt)

        # Update line counter (approximate)
        if hasattr(stmt, "body"):
            self.current_line += len(str(stmt).split("\n"))
        else:
            self.current_line += 1

    def _flush_pending_statements(self) -> None:
        """Create a combined form from pending statements."""
        if not self.pending_statements:
            return

        if len(self.pending_statements) == 1:
            # Single statement - create normal form
            form = Form(
                markdown=[],
                node=self.pending_statements[0],
                start_line=self.current_line,
            )
            self.forms.append(form)
        else:
            # Multiple statements - combine them into a compound statement
            combined_node = self._combine_statements(self.pending_statements)
            form = Form(
                markdown=[],
                node=combined_node,
                start_line=self.current_line,
            )
            self.forms.append(form)

        self.pending_statements.clear()

    def _combine_statements(
        self,
        statements: List[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]],
    ) -> CombinedStatements:
        """Combine multiple statements into a single compound node."""
        # Create a copy of the statements list to avoid issues with mutation
        return CombinedStatements(statements.copy())

    def _merge_consecutive_forms(self, forms: List[Form]) -> List[Form]:
        """Post-process forms to merge consecutive code blocks that should be grouped."""
        if not forms:
            return forms

        merged_forms = []
        i = 0

        while i < len(forms):
            current_form = forms[i]
            has_meaningful_markdown = current_form.markdown and any(
                line.strip() for line in current_form.markdown
            )

            if has_meaningful_markdown:
                # Look ahead to see if we should group this with following forms
                group = [current_form]
                j = i + 1

                # Collect consecutive forms with no meaningful markdown
                while j < len(forms):
                    next_form = forms[j]
                    next_has_markdown = next_form.markdown and any(
                        line.strip() for line in next_form.markdown
                    )

                    if not next_has_markdown:
                        group.append(next_form)
                        j += 1
                    else:
                        break

                # Create merged form if we have multiple forms
                if len(group) > 1:
                    merged_form = self._create_merged_form(group, [])
                    merged_forms.append(merged_form)
                else:
                    merged_forms.append(current_form)

                i = j
            else:
                # Form with no markdown - should be merged with previous or following forms
                # This case should be rare due to our grouping logic, but handle it
                merged_forms.append(current_form)
                i += 1

        return merged_forms

    def _create_merged_form(self, forms: List[Form], markdown: List[str]) -> Form:
        """Create a merged form from a list of forms."""
        if len(forms) == 1:
            return forms[0]

        # Extract all statements from the forms
        statements = []
        for form in forms:
            if isinstance(form.node, CombinedStatements):
                statements.extend(form.node.statements)
            else:
                statements.append(form.node)

        # Create combined node
        combined_node = CombinedStatements(statements)

        # Use the first form's start line and combine markdown from all forms
        all_markdown = []
        for form in forms:
            all_markdown.extend(form.markdown)

        return Form(
            markdown=all_markdown, node=combined_node, start_line=forms[0].start_line
        )


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
