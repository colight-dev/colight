"""MkDocs plugin for processing .py files with Colight.

This plugin processes Python files during the build without creating intermediate files,
similar to how mkdocs-jupyter works.
"""

import pathlib
import traceback
from functools import cached_property
from typing import Optional

from colight_publish import api
from colight_publish.constants import DEFAULT_INLINE_THRESHOLD
from mkdocs.config import Config
from mkdocs.config.config_options import DictOfItems, ListOfItems, Type
from mkdocs.plugins import BasePlugin
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page


class ColightFile(File):
    """Custom File subclass for handling Python/Colight files."""

    def __init__(
        self, path: str, src_dir: str, dest_dir: str, use_directory_urls: bool
    ):
        # Keep the original path - don't change extension
        super().__init__(path, src_dir, dest_dir, use_directory_urls)
        self._content = None
        self._content_bytes = None
        self.colight_files = []  # Store paths to generated .colight files

    def is_documentation_page(self) -> bool:
        """Tell MkDocs this is a documentation page."""
        return True

    @cached_property
    def abs_src_path(self) -> str:
        """Return the absolute source path."""
        # Use the parent class's implementation but ensure it returns a string
        parent_path = super().abs_src_path
        if parent_path is None:
            # Fallback if parent returns None
            if self.src_dir:
                return str(pathlib.Path(self.src_dir) / self.src_path)
            return self.src_path
        return str(parent_path)

    @property
    def content_string(self) -> str:
        """Return the processed content."""
        if self._content is None:
            # This shouldn't happen as we process in on_files
            return ""
        # Ensure we always return a string
        if isinstance(self._content, bytes):
            return self._content.decode("utf-8")
        # Cast to str to ensure correct type
        return str(self._content)

    @content_string.setter
    def content_string(self, value: str):
        """Set the processed content."""
        self._content = value

    @property
    def content_bytes(self) -> bytes:
        """Return the content as bytes."""
        if self._content_bytes is None:
            return self.content_string.encode("utf-8")
        return self._content_bytes

    @content_bytes.setter
    def content_bytes(self, value: bytes):
        """Set the content bytes."""
        self._content_bytes = value


class SitePlugin(BasePlugin):
    """MkDocs plugin that processes Python files in-memory without intermediate files."""

    config_scheme = (
        ("verbose", Type(bool, default=False)),
        ("format", Type(str, default="markdown")),
        ("include", ListOfItems(Type(str), default=["*.py"])),
        ("ignore", ListOfItems(Type(str), default=[])),
        ("pragma", Type(str, default="")),
        (
            "colight_output_path",
            Type(str, default="./form-{form:03d}.colight"),
        ),
        ("colight_embed_path", Type(str, default="form-{form:03d}.colight")),
        ("file_options", DictOfItems(Type(dict), default={})),
        ("inline_threshold", Type(int, default=DEFAULT_INLINE_THRESHOLD)),
    )

    def should_include_file(self, file: File) -> bool:
        """Check if a file should be processed by this plugin."""
        src_path = file.src_path

        # Check if it's a Python file
        if not src_path.endswith(".py"):
            return False

        # Always ignore __pycache__ and __init__.py
        if "__pycache__" in src_path or pathlib.Path(src_path).name == "__init__.py":
            return False

        # Check include patterns
        included = False
        for pattern in self.config["include"]:
            if pathlib.Path(src_path).match(pattern):
                included = True
                break

        if not included:
            return False

        # Check ignore patterns
        for pattern in self.config["ignore"]:
            if pathlib.Path(src_path).match(pattern):
                return False

        return True

    def on_files(self, files: Files, *, config: Config) -> Files:
        """Process Python files and convert them to ColightFile instances."""
        new_files = Files([])

        # Get the site directory from config
        self.site_dir = pathlib.Path(config["site_dir"])

        if self.config["verbose"]:
            print("[colight] Starting file processing...")

        for file in files:
            if self.should_include_file(file):
                # Replace with our custom file class
                # Handle potential None values
                src_dir = file.src_dir or ""
                dest_dir = file.dest_dir or ""
                colight_file = ColightFile(
                    file.src_path, src_dir, dest_dir, file.use_directory_urls
                )
                # Preserve any existing attributes
                colight_file.page = file.page
                colight_file.inclusion = file.inclusion

                # Process the file immediately
                file_path = pathlib.Path(colight_file.abs_src_path)

                if self.config["verbose"]:
                    print(f"[colight] Processing: {file.src_path}")
                    print(f"[colight]   Dest: {colight_file.dest_path}")

                try:
                    # Get file-specific options
                    options = self._get_file_options(file_path)

                    # Process the file to markdown
                    content = self._process_file(
                        file_path, colight_file, config, **options
                    )

                    # Set the content directly
                    colight_file.content_string = content

                except Exception as e:
                    if self.config["verbose"]:
                        print(f"[colight] Error processing {file_path}: {e}")
                        traceback.print_exc()
                    # Set error content
                    colight_file.content_string = f"# Error Processing File\n\nCould not process `{file_path.name}`: {e}\n"

                new_files.append(colight_file)
            else:
                new_files.append(file)

        return new_files

    def on_page_read_source(self, *, page: Page, config: Config) -> Optional[str]:
        """Return the pre-processed content for ColightFile instances."""
        # Only handle our custom ColightFile instances
        if isinstance(page.file, ColightFile):
            if self.config["verbose"]:
                print(f"[colight] Providing content for: {page.file.src_path}")
            # Content was already processed in on_files
            return page.file.content_string

        return None

    def _get_file_options(self, file_path: pathlib.Path) -> dict:
        """Get file-specific options based on patterns."""
        options = {
            "pragma": self.config["pragma"],
            "format": self.config["format"],
        }

        # Apply file-specific overrides
        if self.config["file_options"]:
            for pattern, file_opts in self.config["file_options"].items():
                if pattern in str(file_path) or file_path.match(pattern):
                    options.update(file_opts)

        return options

    def _process_file(
        self,
        file_path: pathlib.Path,
        colight_file: ColightFile,
        config: Config,
        **options,
    ) -> str:
        """Process a Python file and return markdown content."""

        # Write .colight files directly to the site directory
        # Calculate the destination path in the site directory
        site_dir = pathlib.Path(config["site_dir"])

        # Get relative path from docs_dir to the file
        docs_dir = pathlib.Path(config["docs_dir"])
        rel_path = file_path.relative_to(docs_dir)

        # Create the output directory in the site based on the file's destination
        # Match the MkDocs page structure
        if rel_path.name.endswith(".py"):
            output_name = rel_path.name[:-3]  # Remove .py
        else:
            output_name = rel_path.stem

        # Build the colight output directory path in the site
        colight_dir = site_dir / rel_path.parent / output_name
        colight_dir.mkdir(parents=True, exist_ok=True)

        if self.config["verbose"]:
            print(f"[colight]   Writing .colight files to: {colight_dir}")

        # Process the colight file using the public API
        result = api.evaluate_python(
            file_path,
            output_dir=colight_dir,
            inline_threshold=self.config["inline_threshold"],
            format=options.get("format", "markdown"),
            verbose=self.config["verbose"],
            pragma=options.get("pragma", ""),
            output_path_template=self.config["colight_output_path"],
            embed_path_template=self.config["colight_embed_path"],
        )

        # Track colight files for this file
        for block in result.blocks:
            if isinstance(block.visual_data, pathlib.Path):
                colight_file.colight_files.append(str(block.visual_data))
            elif block.visual_data is not None:
                # Bytes were returned - we need to save them
                colight_path = (
                    colight_dir / f"form-{len(colight_file.colight_files):03d}.colight"
                )
                colight_path.write_bytes(block.visual_data)
                colight_file.colight_files.append(str(colight_path))
                if self.config["verbose"]:
                    print(
                        f"[colight]   Saved inline visualization to {colight_path.name}"
                    )

        # Return the markdown content from the API result
        return result.markdown_content
