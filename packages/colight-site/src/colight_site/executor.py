"""Execute Python forms and capture Colight visualizations."""

import ast
from typing import Any, Dict, Optional
import sys
import io
import contextlib

from .parser import Form, CombinedCode
from colight.inspect import inspect
import libcst as cst


class FormExecutor:
    """Execute forms in a persistent namespace."""

    def __init__(self, verbose: bool = False):
        self.env: Dict[str, Any] = {}
        self.form_counter = 0
        self.verbose = verbose

        # Setup basic imports
        self._setup_environment()

    def _setup_environment(self):
        """Setup the execution environment with common imports."""
        # Import colight and common scientific libraries
        setup_code = """
import colight
import numpy as np
import pathlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
try:
    import pandas as pd
except ImportError:
    pass
"""
        exec(setup_code, self.env)

    def execute_form(self, form: Form, filename: str = "<string>") -> Optional[Any]:
        """Execute a form and return its result if it's an expression."""
        self.form_counter += 1

        # Get the code to execute
        code = form.code
        if not code.strip():
            return None

        try:
            # Special handling for CombinedCode that ends with an expression
            if form.is_expression and hasattr(form.node, "code_elements"):
                # This is CombinedCode ending with an expression
                # Execute all statements first, then evaluate the last expression
                if isinstance(form.node, CombinedCode) and form.node.code_elements:
                    # Execute all but the last element as statements
                    for stmt in form.node.code_elements[:-1]:
                        # Convert CST node to code
                        stmt_code = cst.Module(body=[stmt]).code.strip()
                        compiled = compile(stmt_code, filename, "exec")
                        exec(compiled, self.env)

                    # Evaluate the last element as an expression
                    last_elem = form.node.code_elements[-1]
                    # Extract just the expression from the last statement
                    if (
                        isinstance(last_elem, cst.SimpleStatementLine)
                        and len(last_elem.body) == 1
                    ):
                        if isinstance(last_elem.body[0], cst.Expr):
                            expr_code = cst.Module(body=[last_elem]).code.strip()
                            parsed = ast.parse(expr_code, filename, mode="eval")
                            compiled = compile(parsed, filename, "eval")
                            result = eval(compiled, self.env)
                            return result

                # Fallback to original behavior
                parsed = ast.parse(code, filename, mode="eval")
                compiled = compile(parsed, filename, "eval")
                result = eval(compiled, self.env)
                return result
            elif form.is_expression:
                # Parse and compile as expression
                parsed = ast.parse(code, filename, mode="eval")
                compiled = compile(parsed, filename, "eval")
                result = eval(compiled, self.env)
                return result
            else:
                # Execute as statement
                compiled = compile(code, filename, "exec")
                exec(compiled, self.env)
                return None

        except Exception as e:
            print(f"Error executing form {self.form_counter}: {e}", file=sys.stderr)
            print(f"Code: {code}", file=sys.stderr)
            raise

    def get_colight_bytes(self, value: Any) -> Optional[bytes]:
        """Get Colight visualization as bytes."""
        if value is None:
            return None

        try:
            # Let inspect() handle all the complexity internally
            visual = inspect(value)
            if visual is None:
                return None
            return visual.to_bytes()

        except Exception as e:
            if self.verbose:
                print(
                    f"Warning: Could not create Colight visualization: {e}",
                    file=sys.stderr,
                )
            return None


class SafeFormExecutor(FormExecutor):
    """A safer version that captures stdout/stderr."""

    def execute_form(self, form: Form, filename: str = "<string>") -> Optional[Any]:
        """Execute form with output capture."""
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with (
            contextlib.redirect_stdout(stdout_capture),
            contextlib.redirect_stderr(stderr_capture),
        ):
            try:
                result = super().execute_form(form, filename)
                return result
            except Exception:
                # Print captured output to actual stderr
                captured_stderr = stderr_capture.getvalue()
                if captured_stderr:
                    print(captured_stderr, file=sys.stderr)
                raise
