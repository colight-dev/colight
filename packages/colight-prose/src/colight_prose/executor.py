"""Block-based executor for colight documents."""

import contextlib
import io
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from colight.inspect import inspect

from .model import Block, Document


@dataclass
class ExecutionResult:
    """Result of executing a block."""

    value: Any = None
    output: str = ""
    error: Optional[str] = None
    colight_bytes: Optional[bytes] = None
    cache_hit: bool = False
    content_changed: bool = False


class BlockExecutor:
    """Execute blocks in a persistent namespace."""

    def __init__(self, verbose: bool = False):
        self.env: Dict[str, Any] = {
            "__name__": "__main__",
            "__builtins__": __builtins__,
        }
        self.verbose = verbose
        self._setup_environment()

    def _get_package_from_filename(self, filename: str) -> Optional[str]:
        """Determine the package name from a filename.

        Args:
            filename: The absolute or relative path to the file

        Returns:
            The package name or None if not in a package
        """
        from pathlib import Path

        filepath = Path(filename).resolve()

        # Walk up the directory tree looking for __init__.py files
        package_parts = []
        current = filepath.parent

        while current != current.parent:  # Stop at root
            init_file = current / "__init__.py"
            if init_file.exists():
                package_parts.append(current.name)
                current = current.parent
            else:
                break

        if package_parts:
            # Reverse to get correct order (parent.child.subchild)
            package_parts.reverse()
            return ".".join(package_parts)

        return None

    def _setup_environment(self):
        """Setup the execution environment with common imports."""
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

    def execute_block(
        self, block: Block, filename: str = "<string>"
    ) -> ExecutionResult:
        """Execute a single block and return its result.

        Args:
            block: The block to execute
            filename: The filename for error reporting

        Returns:
            ExecutionResult with value, output, error, and colight_bytes
        """
        # Set __file__ in the environment if we have a real filename
        if filename != "<string>":
            self.env["__file__"] = filename

            # Set __package__ to support relative imports
            package = self._get_package_from_filename(filename)
            if package is not None:
                self.env["__package__"] = package

                # Also ensure the package root is in sys.path
                from pathlib import Path

                filepath = Path(filename).resolve()
                # Find the root of the package (parent of topmost package)
                package_root = filepath.parent
                for _ in package.split("."):
                    package_root = package_root.parent

                if str(package_root) not in sys.path:
                    sys.path.insert(0, str(package_root))

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        result = ExecutionResult()

        with (
            contextlib.redirect_stdout(stdout_capture),
            contextlib.redirect_stderr(stderr_capture),
        ):
            try:
                # Compile the block if not already done
                block.compile_once(filename)

                # Execute statements if any
                if block._exec_code:
                    exec(block._exec_code, self.env)

                # Evaluate expression if any
                if block._eval_code:
                    result.value = eval(block._eval_code, self.env)

                    # Try to create Colight visualization
                    if result.value is not None:
                        try:
                            visual = inspect(result.value)
                            if visual is not None:
                                result.colight_bytes = visual.to_bytes()
                        except Exception as e:
                            if self.verbose:
                                print(
                                    f"Warning: Could not create Colight visualization: {e}",
                                    file=sys.stderr,
                                )

                # Capture output
                result.output = stdout_capture.getvalue()

            except Exception:
                # Capture error with better formatting
                import traceback

                # Get the raw traceback
                tb = traceback.format_exc()

                # If we have a real filename (not <string>), the traceback will
                # already show correct line numbers due to our padding in compile_once
                result.error = tb

                # Also capture any stderr output
                stderr_output = stderr_capture.getvalue()
                if stderr_output:
                    result.error = stderr_output + "\n" + result.error

        return result

    def execute_document(
        self, document: Document, filename: str = "<string>"
    ) -> List[ExecutionResult]:
        """Execute all blocks in a document.

        Args:
            document: The document to execute
            filename: The filename for error reporting

        Returns:
            List of ExecutionResult, one per block
        """
        results = []

        for i, block in enumerate(document.blocks):
            if self.verbose:
                print(
                    f"Executing block {i + 1}/{len(document.blocks)}...",
                    file=sys.stderr,
                )

            result = self.execute_block(block, filename)
            results.append(result)

            # Stop execution if there was an error (unless we want to continue)
            if result.error and not self.verbose:
                # Fill remaining blocks with empty results
                for _ in range(len(document.blocks) - i - 1):
                    results.append(ExecutionResult())
                break

        return results

    def reset(self):
        """Reset the execution environment."""
        self.env.clear()
        self.env.update(
            {
                "__name__": "__main__",
                "__builtins__": __builtins__,
            }
        )
        self._setup_environment()


class DocumentExecutor:
    """High-level executor for documents with additional features."""

    def __init__(self, verbose: bool = False, capture_output: bool = True):
        self.executor = BlockExecutor(verbose=verbose)
        self.capture_output = capture_output

    def execute(
        self, document: Document, filename: str = "<string>"
    ) -> Tuple[List[ExecutionResult], Dict[str, Any]]:
        """Execute a document and return results plus final namespace.

        Args:
            document: The document to execute
            filename: The filename for error reporting

        Returns:
            Tuple of (results, namespace)
        """
        results = self.executor.execute_document(document, filename)

        # Return a copy of the namespace (excluding builtins)
        namespace = {
            k: v
            for k, v in self.executor.env.items()
            if k not in ("__builtins__", "__name__", "__file__", "__package__")
        }

        return results, namespace

    def execute_single(
        self, document: Document, filename: str = "<string>"
    ) -> List[Optional[bytes]]:
        """Execute document and return only Colight bytes for each block.

        This provides a simpler API for cases that only care about visualizations.

        Args:
            document: The document to execute
            filename: The filename for error reporting

        Returns:
            List of bytes (or None) for each block's visualization
        """
        results = self.executor.execute_document(document, filename)
        return [r.colight_bytes for r in results]
