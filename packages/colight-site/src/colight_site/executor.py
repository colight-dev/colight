"""Execute Python forms and capture Colight visualizations."""

import ast
import pathlib
from typing import Any, Dict, Optional
import sys
import io
import contextlib

from .parser import Form
from colight.inspect import inspect


class FormExecutor:
    """Execute forms in a persistent namespace."""

    def __init__(self, output_dir: pathlib.Path):
        self.env: Dict[str, Any] = {}
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.form_counter = 0

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
            # Try to parse as an expression first
            if form.is_expression:
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

    def save_colight_visualization(
        self, value: Any, form_index: int
    ) -> Optional[pathlib.Path]:
        """Save a Colight visualization to a .colight file."""
        if value is None:
            return None

        output_path = self.output_dir / f"form-{form_index:03d}.colight"

        try:
            # Let inspect() handle all the complexity internally
            visual = inspect(value)
            if visual is None:
                return None
            visual.save_file(str(output_path))
            return output_path

        except Exception as e:
            print(
                f"Warning: Could not save Colight visualization: {e}", file=sys.stderr
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
