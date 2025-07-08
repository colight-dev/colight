"""File resolution and pattern matching utilities for colight-site."""

import pathlib
from typing import List, Optional


class FileResolver:
    """Handles file resolution, pattern matching, and output path generation."""

    def __init__(
        self,
        input_path: pathlib.Path,
        include: List[str],
        ignore: Optional[List[str]] = None,
    ):
        """Initialize FileResolver.

        Args:
            input_path: Base input path (file or directory)
            include: List of include patterns
            ignore: Optional list of ignore patterns
        """
        self.input_path = input_path
        self.include = include
        self.ignore = ignore or []
        self._is_single_file = input_path.is_file()

    def find_source_file(self, requested_path: str) -> Optional[pathlib.Path]:
        """Find the source .py file for a requested path.

        Args:
            requested_path: The requested path (may or may not have .html extension)

        Returns:
            Path to the source file if found and matches patterns, None otherwise
        """
        # Remove .html extension if present
        clean_path = requested_path.removesuffix(".html")

        # Handle single file mode
        if self._is_single_file:
            if clean_path == self.input_path.stem or clean_path == "":
                return (
                    self.input_path if self.matches_patterns(self.input_path) else None
                )
            return None

        # Try different variations for directory mode
        possible_paths = [
            self.input_path / f"{clean_path}.py",
            self.input_path / f"{clean_path}.colight.py",
            self.input_path / clean_path / "__init__.py",
        ]

        for source_path in possible_paths:
            if (
                source_path.exists()
                and source_path.is_file()
                and self.matches_patterns(source_path)
            ):
                return source_path

        return None

    def get_output_path(
        self, source_file: pathlib.Path, output_base: pathlib.Path
    ) -> pathlib.Path:
        """Get the output path for a source file.

        Args:
            source_file: Path to the source file
            output_base: Base output directory

        Returns:
            Path where the output file should be written
        """
        if self._is_single_file:
            # Single file mode - output goes directly to output directory
            return output_base / source_file.with_suffix(".html").name

        # Directory mode - mirror the structure
        try:
            rel_path = source_file.relative_to(self.input_path)
            return output_base / rel_path.with_suffix(".html")
        except ValueError:
            # Fallback if not relative
            return output_base / source_file.with_suffix(".html").name

    def matches_patterns(self, file_path: pathlib.Path) -> bool:
        """Check if file matches include/ignore patterns.

        Args:
            file_path: Path to check

        Returns:
            True if file matches include patterns and doesn't match ignore patterns
        """
        from .index_generator import matches_patterns
        from .constants import DEFAULT_IGNORE_PATTERNS

        # Combine user ignore patterns with defaults
        combined_ignore = list(self.ignore)
        combined_ignore.extend(DEFAULT_IGNORE_PATTERNS)

        return matches_patterns(file_path, self.include, combined_ignore)

    def get_all_files(self) -> List[str]:
        """Get list of all matching files as relative paths.

        Returns:
            List of relative paths without extensions
        """
        from .index_generator import find_colight_files
        from .constants import DEFAULT_IGNORE_PATTERNS

        # Combine user ignore patterns with defaults
        combined_ignore = list(self.ignore)
        combined_ignore.extend(DEFAULT_IGNORE_PATTERNS)

        files = find_colight_files(self.input_path, self.include, combined_ignore)

        # Convert to relative paths without extensions
        if self._is_single_file:
            return [self.input_path.stem]
        else:
            paths = []
            for f in files:
                rel_path = str(f.relative_to(self.input_path))
                # Remove .py extension
                if rel_path.endswith(".py"):
                    rel_path = rel_path[:-3]
                # Remove .colight.py extension
                if rel_path.endswith(".colight"):
                    rel_path = rel_path[:-8]
                paths.append(rel_path)
            return sorted(paths)
