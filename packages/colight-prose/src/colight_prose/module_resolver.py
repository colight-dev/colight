"""Module name to file path resolution."""

import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Optional


def is_stdlib_module(module_name: str) -> bool:
    """Check if a module name refers to a standard library module."""
    if module_name in sys.builtin_module_names:
        return True
    try:
        spec = find_spec(module_name)
        if spec is None:
            return False  # Not found, can't be stdlib
        # This is a robust way to check for stdlib modules
        return (
            spec.origin is not None and "lib" in spec.origin and "python" in spec.origin
        )
    except (ImportError, ValueError, AttributeError):
        return False


def resolve_module_to_file(
    module_name: str, current_file: str, project_root: str, relative_level: int = 0
) -> Optional[str]:
    """Resolve a module name to a file path within the project."""
    log_messages = []
    tried_paths = []
    is_debug_target = True  # can be used for debugging a specific issue

    def log(message):
        if is_debug_target:
            log_messages.append(message)

    log(
        f"🔍 Resolving '{module_name}' from {Path(current_file).name} (level={relative_level})"
    )

    project_root_path = Path(project_root).resolve()
    current_file_path = (project_root_path / current_file).resolve()

    if relative_level == 0 and is_stdlib_module(module_name):
        return None

    search_paths = []
    if relative_level > 0:
        start_dir = current_file_path.parent
        for _ in range(relative_level - 1):
            start_dir = start_dir.parent
        search_paths.append(start_dir)
    else:
        search_paths.append(current_file_path.parent)
        search_paths.append(project_root_path)
        for p in sys.path:
            try:
                resolved_p = Path(p).resolve()
                if (
                    project_root_path in resolved_p.parents
                    or project_root_path == resolved_p
                ):
                    if resolved_p not in search_paths:
                        search_paths.append(resolved_p)
            except (FileNotFoundError, Exception):
                continue

    module_parts = module_name.split(".") if module_name else []

    # Primary search
    for search_path in search_paths:
        candidate_base = search_path.joinpath(*module_parts)

        candidate_pkg = candidate_base / "__init__.py"
        tried_paths.append(candidate_pkg)
        if candidate_pkg.is_file() and candidate_pkg.is_relative_to(project_root_path):
            resolved_path = str(candidate_pkg.relative_to(project_root_path))
            return resolved_path

        candidate_file = candidate_base.with_suffix(".py")
        tried_paths.append(candidate_file)
        if candidate_file.is_file() and candidate_file.is_relative_to(
            project_root_path
        ):
            resolved_path = str(candidate_file.relative_to(project_root_path))
            return resolved_path

    # Fallback: Upward traversal for absolute imports
    if relative_level == 0:
        try:
            spec = find_spec(module_name)
            if spec is not None:
                return None
        except (ImportError, ValueError, AttributeError):
            pass

        current_dir = current_file_path.parent
        while current_dir != project_root_path and current_dir.parent != current_dir:
            candidate_base = current_dir.joinpath(*module_parts)

            candidate_pkg = candidate_base / "__init__.py"
            tried_paths.append(candidate_pkg)
            if candidate_pkg.is_file() and candidate_pkg.is_relative_to(
                project_root_path
            ):
                resolved_path = str(candidate_pkg.relative_to(project_root_path))
                return resolved_path

            candidate_file = candidate_base.with_suffix(".py")
            tried_paths.append(candidate_file)
            if candidate_file.is_file() and candidate_file.is_relative_to(
                project_root_path
            ):
                resolved_path = str(candidate_file.relative_to(project_root_path))
                return resolved_path

            current_dir = current_dir.parent

    log(f"❌ Not found. Tried: {', '.join(str(p) for p in tried_paths)}")
    if is_debug_target:
        print("\n".join(log_messages))

    if relative_level == 0:
        try:
            spec = find_spec(module_name)
            if (
                spec
                and spec.origin
                and not Path(spec.origin).is_relative_to(project_root_path)
            ):
                return None
        except (ImportError, ValueError, AttributeError):
            pass

    return None
