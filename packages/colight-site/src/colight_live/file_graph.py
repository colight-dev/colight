"""File-level dependency graph for tracking import relationships."""

import ast
import logging
import pathlib
from collections import defaultdict
from typing import Dict, Set, Tuple

from .module_resolver import resolve_module_to_file

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
        relative_file_path = self._get_relative_path(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_file = resolve_module_to_file(
                            alias.name,
                            relative_file_path,
                            str(self.watched_path),
                        )
                        if module_file:
                            imports.add(module_file)

                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    level = node.level or 0
                    if level > 0 or (module_name and not module_name.startswith(".")):
                        module_file = resolve_module_to_file(
                            module_name,
                            relative_file_path,
                            str(self.watched_path),
                            level,
                        )
                        if module_file:
                            imports.add(module_file)

        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")

        return imports

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
        # Normalize the changed file path - it should match what's in our dictionaries
        # The keys in imported_by are relative to base_path
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

    def remove_file(self, file_path: str):
        """Remove a file and its dependencies from the graph."""
        try:
            relative_path = self._get_relative_path(pathlib.Path(file_path))
        except ValueError:
            relative_path = file_path

        # Remove from cache
        if relative_path in self._cache:
            del self._cache[relative_path]

        # Get the files this file imported and remove the reverse dependency
        imports_to_clear = self.imports.get(relative_path, set())
        for imported_file in imports_to_clear:
            if imported_file in self.imported_by:
                self.imported_by[imported_file].discard(relative_path)
                if not self.imported_by[imported_file]:
                    del self.imported_by[imported_file]

        # Remove the file's own entries
        if relative_path in self.imports:
            del self.imports[relative_path]

        logger.debug(f"Removed {relative_path} from dependency graph.")
