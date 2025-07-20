"""Module name to file path resolution."""

import pathlib
from pathlib import Path
from typing import Optional


def resolve_module_to_file(
    module_name: str, current_file: str, project_root: str, relative_level: int = 0
) -> Optional[str]:
    """Resolve a module name to a file path within the project.

    Args:
        module_name: The module name to resolve (e.g., 'mymodule', 'package.submodule')
        current_file: The file containing the import statement (relative to project root)
        project_root: The root directory of the project
        relative_level: Number of dots for relative imports (0 for absolute imports)

    Returns:
        Relative path from project root to the module file, or None if:
        - Module is external (stdlib or third-party)
        - Module file doesn't exist
    """
    project_root_path = Path(project_root)
    current_file_path = project_root_path / current_file

    # Handle relative imports
    if relative_level > 0:
        # Start from the directory containing the current file
        start_dir = current_file_path.parent

        # Go up the directory tree based on the number of dots
        for _ in range(relative_level - 1):
            start_dir = start_dir.parent

        # If module_name is empty (e.g., "from . import x"), we're importing from __init__.py
        if not module_name:
            target_path = start_dir / "__init__.py"
        else:
            # Convert module name to path
            module_parts = module_name.split(".")
            target_path = start_dir.joinpath(*module_parts)
    else:
        # Absolute import - start from project root
        if not module_name:
            return None

        module_parts = module_name.split(".")
        target_path = project_root_path.joinpath(*module_parts)

    # Build list of candidates
    candidates = []
    parts = module_parts  # Use the module_parts we already split

    # 1. Try relative to the importing file's directory (most common for local imports)
    from_dir = current_file_path.parent
    candidates.append(from_dir / pathlib.Path(*parts).with_suffix(".py"))
    candidates.append(from_dir / pathlib.Path(*parts) / "__init__.py")

    # 2. Check if the importing file is in a package (has __init__.py files)
    # Walk up directory tree to find package roots
    current = from_dir
    while current.is_relative_to(project_root_path):
        parent = current.parent
        if (current / "__init__.py").exists():
            # This is a package, try importing relative to its parent
            candidates.append(parent / pathlib.Path(*parts).with_suffix(".py"))
            candidates.append(parent / pathlib.Path(*parts) / "__init__.py")
            current = parent
        else:
            break

    # 3. Try from project root (absolute import)
    candidates.append(target_path.with_suffix(".py"))
    candidates.append(target_path / "__init__.py")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            # Return relative path from project root
            try:
                rel_path = str(candidate.relative_to(project_root_path))
                return rel_path
            except ValueError:
                # File is outside project root
                # print(f"Outside project: {candidate}")
                return None

    # Module not found in project - likely external
    return None


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module name refers to a standard library module.

    This is a simple heuristic - a more complete implementation would
    check against the actual list of stdlib modules.
    """
    # Common stdlib modules
    stdlib_modules = {
        "os",
        "sys",
        "json",
        "math",
        "random",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "typing",
        "re",
        "ast",
        "dataclasses",
        "enum",
        "abc",
        "asyncio",
        "subprocess",
        "shutil",
        "tempfile",
        "unittest",
        "logging",
        "argparse",
        "configparser",
        "csv",
        "sqlite3",
        "urllib",
        "http",
        "email",
        "html",
        "xml",
    }

    # Check if it's a known stdlib module or submodule
    top_level = module_name.split(".")[0]
    return top_level in stdlib_modules
