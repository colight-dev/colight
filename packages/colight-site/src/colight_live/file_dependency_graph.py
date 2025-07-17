"""File-level dependency graph for tracking import relationships."""

import ast
import logging
import pathlib
from collections import defaultdict
from importlib.util import find_spec
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class FileDependencyGraph:
    """Track dependencies between Python files based on imports.

    watched_path: The directory being watched for changes
    """

    def __init__(self, watched_path: pathlib.Path):
        self.watched_path = watched_path.resolve()
        # Forward dependencies: file -> files it imports
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        # Reverse dependencies: file -> files that import it
        self.imported_by: Dict[str, Set[str]] = defaultdict(set)
        # Cache of analyzed files
        self._cache: Dict[str, Tuple[Set[str], float]] = {}

    def analyze_file(self, file_path: pathlib.Path) -> Set[str]:
        """Analyze a Python file and extract its import dependencies.

        Args:
            file_path: Path to Python file

        Returns:
            Set of file paths this file imports
        """
        file_path = file_path.resolve()
        relative_path = self._get_relative_path(file_path)

        # Check cache
        if relative_path in self._cache:
            mtime = file_path.stat().st_mtime
            cached_imports, cached_mtime = self._cache[relative_path]
            if mtime <= cached_mtime:
                return cached_imports

        try:
            logger.debug(f"Analyzing file: {file_path}")
            imports = self._extract_imports(file_path)
            logger.debug(f"File {relative_path} imports: {imports}")
            # Update cache
            self._cache[relative_path] = (imports, file_path.stat().st_mtime)
            # Update graph
            self._update_graph(relative_path, imports)
            return imports
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return set()

    def _extract_imports(self, file_path: pathlib.Path) -> Set[str]:
        """Extract import statements from a Python file."""
        imports = set()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_file = self._resolve_import(alias.name, file_path)
                        if module_file:
                            imports.add(module_file)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Absolute import
                        module_file = self._resolve_import(node.module, file_path)
                        if module_file:
                            imports.add(module_file)
                    elif node.level > 0:
                        # Relative import
                        module_file = self._resolve_relative_import(
                            node.level, getattr(node, "module", None), file_path
                        )
                        if module_file:
                            imports.add(module_file)

        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")

        return imports

    def _resolve_import(
        self, module_name: str, from_file: pathlib.Path
    ) -> Optional[str]:
        """Resolve an import to a file path as Python would, using sys.path and the importing file's directory."""
        import sys

        parts = module_name.split(".")
        candidates = []

        # DEBUG: Print what we're trying to resolve
        logger.debug(f"Resolving import '{module_name}' from {from_file}")

        # 1. Try relative to the importing file's directory (most common for local imports)
        from_dir = from_file.parent
        candidates.append(from_dir / pathlib.Path(*parts).with_suffix(".py"))
        candidates.append(from_dir / pathlib.Path(*parts) / "__init__.py")

        # 2. Check if the importing file is in a package (has __init__.py files)
        # Walk up directory tree to find package roots
        current = from_dir
        while self._is_within_base(current):
            parent = current.parent
            if (current / "__init__.py").exists():
                # This is a package, try importing relative to its parent
                candidates.append(parent / pathlib.Path(*parts).with_suffix(".py"))
                candidates.append(parent / pathlib.Path(*parts) / "__init__.py")
                current = parent
            else:
                break

        # 3. Try all sys.path entries (mimics Python's import system)
        for sys_path_entry in sys.path:
            if not sys_path_entry:  # Empty string means current directory
                sys_path_entry = "."
            try:
                sys_path_dir = pathlib.Path(sys_path_entry).resolve()
            except Exception:
                continue
            candidates.append(sys_path_dir / pathlib.Path(*parts).with_suffix(".py"))
            candidates.append(sys_path_dir / pathlib.Path(*parts) / "__init__.py")

        # 4. Try watched_path as a fallback (for compatibility)
        candidates.append(self.watched_path / pathlib.Path(*parts).with_suffix(".py"))
        candidates.append(self.watched_path / pathlib.Path(*parts) / "__init__.py")

        for candidate in candidates:
            logger.debug(f"Checking candidate: {candidate}")
            if candidate.exists() and candidate.is_file():
                # Only return files within our watched directory
                if self._is_within_base(candidate):
                    result = self._get_relative_path(candidate)
                    logger.debug(f"Found import: {module_name} -> {result}")
                    return result
                else:
                    logger.debug(f"Found external import: {module_name} -> {candidate}")
                    return None

        # If we couldn't find it locally, check if it's an external import
        if self._is_external_import(module_name):
            logger.debug(f"External import detected: {module_name}")
            return None
        logger.debug(f"Import not found: {module_name}")
        return None

    def _resolve_relative_import(
        self, level: int, module: Optional[str], from_file: pathlib.Path
    ) -> Optional[str]:
        """Resolve a relative import to a file path.

        Args:
            level: Number of dots in the relative import
            module: Module name after the dots (can be None)
            from_file: The file doing the importing

        Returns:
            Relative path to the imported file, or None if not found
        """
        # Start from the importing file's directory
        current_dir = from_file.parent

        # Go up 'level' directories
        for _ in range(level - 1):
            current_dir = current_dir.parent
            if not self._is_within_base(current_dir):
                return None

        if module:
            # from ..foo import bar
            parts = module.split(".")
            candidates = [
                current_dir / pathlib.Path(*parts).with_suffix(".py"),
                current_dir / pathlib.Path(*parts) / "__init__.py",
            ]
        else:
            # from .. import something (import from parent package)
            candidates = [current_dir / "__init__.py"]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return self._get_relative_path(candidate)

        return None

    def _is_external_import(self, module_name: str) -> bool:
        """Check if an import is external (outside our project directory).

        Returns True if the module lives outside our base directory, meaning
        we can safely ignore it since users won't edit stdlib or venv files.
        """
        # Quick checks to avoid expensive find_spec calls
        first_part = module_name.split(".")[0]

        # Common patterns that are definitely external
        if first_part.startswith("_"):  # _thread, _collections, etc.
            return True

        # If it contains no dots and starts with lowercase, it might be stdlib
        # But we can't be sure without checking, so continue

        try:
            # Only use find_spec as a last resort
            spec = find_spec(module_name)
            if spec is None or spec.origin is None:
                # Namespace package or built-in - safe to ignore
                return True

            # Get the actual file path of the module
            module_path = pathlib.Path(spec.origin).resolve()

            # Check if it's within our base directory
            return not self._is_within_base(module_path)
        except (ImportError, ValueError, AttributeError):
            # If we can't find the module or something goes wrong,
            # assume it's external to be safe
            return True

    def _is_within_base(self, path: pathlib.Path) -> bool:
        """Check if a path is within the watched directory."""
        try:
            path.relative_to(self.watched_path)
            return True
        except ValueError:
            return False

    def _get_relative_path(self, path: pathlib.Path) -> str:
        """Get relative path from watched directory."""
        try:
            return str(path.relative_to(self.watched_path))
        except ValueError:
            return str(path)

    def _update_graph(self, file_path: str, imports: Set[str]):
        """Update the dependency graph with new import information."""
        # Clear old reverse dependencies
        old_imports = self.imports.get(file_path, set())
        for old_import in old_imports:
            self.imported_by[old_import].discard(file_path)

        # Filter imports to only include files within our base directory
        # This is a safety check to ensure we never track external files
        valid_imports = set()
        for imp in imports:
            # Check if imp is already an absolute path
            if pathlib.Path(imp).is_absolute():
                abs_path = pathlib.Path(imp)
            else:
                # Convert relative path to absolute for checking
                abs_path = (self.watched_path / imp).resolve()

            if self._is_within_base(abs_path):
                # Store as relative path within watched directory
                valid_imports.add(self._get_relative_path(abs_path))
            else:
                logger.debug(f"Skipping external import reference: {imp}")

        # Update forward dependencies
        self.imports[file_path] = valid_imports

        # Update reverse dependencies
        for imported in valid_imports:
            self.imported_by[imported].add(file_path)

    def get_affected_files(self, changed_file: str) -> Set[str]:
        """Get all files affected by changes to the given file.

        This includes:
        1. The changed file itself
        2. All files that import the changed file (directly or indirectly)

        Args:
            changed_file: Relative path to the changed file

        Returns:
            Set of all affected file paths
        """
        affected = {changed_file}
        to_check = [changed_file]

        while to_check:
            current = to_check.pop()
            # Find all files that import the current file
            importers = self.imported_by.get(current, set())
            for importer in importers:
                if importer not in affected:
                    affected.add(importer)
                    to_check.append(importer)

        return affected

    def get_dependencies(self, file_path: str) -> Set[str]:
        """Get all files that the given file depends on.

        Args:
            file_path: Relative path to the file

        Returns:
            Set of file paths this file imports
        """
        return self.imports.get(file_path, set())

    def clear_cache(self):
        """Clear the analysis cache."""
        self._cache.clear()

    def analyze_directory(self, directory: pathlib.Path):
        """Analyze all Python files in a directory recursively.

        Skips hidden directories (starting with .) to avoid scanning
        .venv, .git, .tox, etc.
        """
        directory = directory.resolve()

        # Count files for logging
        analyzed_count = 0
        skipped_count = 0

        for py_file in directory.rglob("*.py"):
            # Skip files in hidden directories
            if any(
                part.startswith(".") for part in py_file.relative_to(directory).parts
            ):
                skipped_count += 1
                continue

            # Skip files outside our watched directory (safety check)
            if not self._is_within_base(py_file):
                skipped_count += 1
                continue

            if py_file.is_file():
                self.analyze_file(py_file)
                analyzed_count += 1

        logger.info(
            f"Analyzed {analyzed_count} files, skipped {skipped_count} files in hidden directories"
        )

    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the dependency graph."""
        total_files = len(set(self.imports.keys()) | set(self.imported_by.keys()))
        total_imports = sum(len(deps) for deps in self.imports.values())

        return {
            "total_files": total_files,
            "total_imports": total_imports,
            "files_with_imports": len(self.imports),
            "files_imported": len(self.imported_by),
        }
