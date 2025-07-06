"""Execute Python forms and capture Colight visualizations."""

import ast
from typing import Any, Dict, Optional
import sys
import io
import contextlib
import libcst as cst

from .parser import Form, CombinedCode
from colight.inspect import inspect


class FormExecutor:
    """Execute forms in a persistent namespace."""

    def __init__(self, verbose: bool = False):
        self.env: Dict[str, Any] = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
        }
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

        # Set __file__ in the environment if we have a real filename
        if filename != "<string>":
            self.env["__file__"] = filename

        try:
            result = None
            
            # Execute each code element in order
            for elem in form.elements:
                if elem.type in ("statement", "expression"):
                    # Special handling for CombinedCode elements
                    if isinstance(elem.content, CombinedCode) and elem.type == "expression" and elem == form.last_element:
                        # This is CombinedCode ending with an expression
                        # Execute all statements first, then evaluate the last expression
                        code_elements = elem.content.code_elements
                        if code_elements:
                            # Execute all but the last element as statements
                            for stmt in code_elements[:-1]:
                                stmt_code = cst.Module(body=[stmt]).code.strip()
                                if stmt_code:
                                    exec(compile(stmt_code, filename, "exec"), self.env)
                            
                            # Evaluate the last element as an expression
                            last_elem = code_elements[-1]
                            if isinstance(last_elem, cst.SimpleStatementLine) and len(last_elem.body) == 1:
                                if isinstance(last_elem.body[0], cst.Expr):
                                    expr_code = cst.Module(body=[last_elem]).code.strip()
                                    result = eval(compile(expr_code, filename, 'eval'), self.env)
                    else:
                        # Normal code element
                        code = elem.get_code().strip()
                        if not code:
                            continue
                        
                        if elem.type == "expression" and elem == form.last_element:
                            # This is the last element and it's an expression - evaluate it
                            result = eval(compile(code, filename, 'eval'), self.env)
                        else:
                            # Execute as a statement
                            exec(compile(code, filename, 'exec'), self.env)
            
            return result

        except Exception as e:
            print(f"Error executing form {self.form_counter}: {e}", file=sys.stderr)
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
