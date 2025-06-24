"""Support for PEP 723 inline script metadata."""

import re
import tempfile
import subprocess
import pathlib
from typing import Optional, List, Tuple


def detect_pep723_metadata(content: str) -> Optional[str]:
    """
    Detect PEP 723 metadata block in Python file content.

    Returns the metadata content if found, None otherwise.
    """
    # PEP 723 metadata format:
    # # /// script
    # # dependencies = [
    # #   "package1",
    # #   "package2>=1.0",
    # # ]
    # # ///

    # Look for the start marker
    lines = content.split("\n")
    start_idx = None

    for i, line in enumerate(lines):
        if line.strip() == "# /// script":
            start_idx = i
            break

    if start_idx is None:
        return None

    # Look for the end marker
    end_idx = None
    for i in range(start_idx + 1, len(lines)):
        if lines[i].strip() == "# ///":
            end_idx = i
            break

    if end_idx is None:
        return None

    # Extract metadata content (remove comment markers)
    metadata_lines = []
    for i in range(start_idx + 1, end_idx):
        line = lines[i]
        # Remove leading '# ' if present
        if line.startswith("# "):
            metadata_lines.append(line[2:])
        elif line.strip() == "#":
            metadata_lines.append("")
        else:
            # Invalid format
            return None

    return "\n".join(metadata_lines)


def parse_dependencies(metadata: str) -> List[str]:
    """Extract dependencies from PEP 723 metadata."""
    # Simple regex to extract dependencies list
    # This handles both single-line and multi-line formats
    match = re.search(r"dependencies\s*=\s*\[(.*?)\]", metadata, re.DOTALL)

    if not match:
        return []

    deps_str = match.group(1)

    # Extract individual dependencies (quoted strings)
    deps = re.findall(r'["\']([^"\']+)["\']', deps_str)

    return deps


def run_with_pep723_env(
    file_path: pathlib.Path, python_code: str, verbose: bool = False
) -> Tuple[subprocess.CompletedProcess, pathlib.Path]:
    """
    Run a Python file with PEP 723 dependencies using uv.

    Returns the completed process and the temporary file path used.
    """
    # Create a temporary file with the Python code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=file_path.parent
    ) as tmp_file:
        tmp_file.write(python_code)
        tmp_path = pathlib.Path(tmp_file.name)

    try:
        # Run with uv run --script
        cmd = ["uv", "run", "--script", str(tmp_path)]

        if verbose:
            print(f"Running with PEP 723 dependencies: {cmd}")

        # Run the command
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=file_path.parent
        )

        return result, tmp_path

    except Exception:
        # Clean up temp file on error
        tmp_path.unlink(missing_ok=True)
        raise


def execute_pep723_code(
    code: str, env: dict, filename: str = "<pep723>", verbose: bool = False
) -> None:
    """
    Execute code that was run in a PEP 723 environment.

    This is used to execute individual forms after the initial
    PEP 723 setup has been done.
    """
    # Execute directly in the provided environment
    exec(compile(code, filename, "exec"), env)
