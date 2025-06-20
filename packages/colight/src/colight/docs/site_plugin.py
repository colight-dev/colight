"""MkDocs plugin for processing .py and .colight.py files in-memory.

This plugin processes Python files during the build without creating intermediate files,
similar to how mkdocs-jupyter works.
"""

import pathlib
import traceback
from typing import Optional

from mkdocs.config import Config
from mkdocs.plugins import BasePlugin
from mkdocs.config.config_options import Type, DictOfItems, ListOfItems
from mkdocs.structure.files import Files, File
from mkdocs.structure.pages import Page

# Import colight-site modules
import sys

colight_site_path = (
    pathlib.Path(__file__).parent.parent.parent.parent.parent / "colight-site" / "src"
)
if colight_site_path.exists():
    sys.path.insert(0, str(colight_site_path))

try:
    from colight_site.parser import parse_colight_file
    from colight_site.executor import SafeFormExecutor
    from colight_site.generator import MarkdownGenerator
except ImportError as e:
    raise ImportError(f"colight-site package is required: {e}")


class ColightFiles(Files):
    """Custom Files collection that handles .py to .md mapping."""

    def __init__(self, files):
        super().__init__(files)
        self._py_to_md_map = {}

    def add_py_mapping(self, py_path: str, md_path: str):
        """Add a mapping from .py path to .md path."""
        self._py_to_md_map[py_path] = md_path

    def get_file_from_path(self, path: str) -> Optional[File]:
        """Override to handle .py lookups that map to .md files."""
        # First try the standard lookup
        file = super().get_file_from_path(path)
        if file:
            return file

        # If not found and it's a .py file, try the mapped .md version
        if path.endswith(".py") and path in self._py_to_md_map:
            md_path = self._py_to_md_map[path]
            return super().get_file_from_path(md_path)

        return None


class ColightFile(File):
    """Custom File subclass for handling Python/Colight files."""

    def __init__(
        self, path: str, src_dir: str, dest_dir: str, use_directory_urls: bool
    ):
        # Store original path before modification
        self._original_src_path = path

        # Change the source path to end with .md so MkDocs treats it as markdown
        if path.endswith(".py"):
            # Change to .md for MkDocs processing
            modified_path = path[:-3] + ".md"
        else:
            modified_path = path

        super().__init__(modified_path, src_dir, dest_dir, use_directory_urls)
        self._content = None
        self._content_bytes = None
        self.colight_files = []  # Store paths to generated .colight files

    @property
    def abs_src_path(self) -> str:
        """Return the original source path for processing."""
        # Use the original Python file path for reading
        return str(pathlib.Path(self.src_dir) / self._original_src_path)

    @property
    def content_string(self) -> str:
        """Return the processed content."""
        if self._content is None:
            # This shouldn't happen as we process in on_files
            return ""
        return self._content

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
        ("include", ListOfItems(Type(str), default=["*.py", "*.colight.py"])),
        ("ignore", ListOfItems(Type(str), default=[])),
        ("hide_statements", Type(bool, default=False)),
        ("hide_visuals", Type(bool, default=False)),
        ("hide_code", Type(bool, default=False)),
        (
            "colight_output_path",
            Type(str, default="./form-{form:03d}.colight"),
        ),
        ("colight_embed_path", Type(str, default="form-{form:03d}.colight")),
        ("file_options", DictOfItems(Type(dict), default={})),
    )

    def should_include_file(self, file: File) -> bool:
        """Check if a file should be processed by this plugin."""
        # For ColightFile instances, check the original path
        if isinstance(file, ColightFile):
            src_path = file._original_src_path
        else:
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
        new_files = ColightFiles([])

        # Get the site directory from config
        self.site_dir = pathlib.Path(config["site_dir"])

        for file in files:
            if self.should_include_file(file):
                # Replace with our custom file class
                colight_file = ColightFile(
                    file.src_path, file.src_dir, file.dest_dir, file.use_directory_urls
                )
                # Preserve any existing attributes
                colight_file.page = file.page
                colight_file.inclusion = file.inclusion

                # Add mapping from .py to .md
                new_files.add_py_mapping(file.src_path, colight_file.src_path)

                # Process the file immediately
                file_path = pathlib.Path(colight_file.abs_src_path)

                if self.config["verbose"]:
                    print(f"[colight] Processing: {file.src_path}")
                    print(f"[colight]   Original: {colight_file._original_src_path}")
                    print(f"[colight]   Modified: {colight_file.src_path}")
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

    def on_page_read_source(self, page: Page, *, config: Config) -> Optional[str]:
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
            "hide_statements": self.config["hide_statements"],
            "hide_visuals": self.config["hide_visuals"],
            "hide_code": self.config["hide_code"],
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
        # Parse the file
        forms, file_metadata = parse_colight_file(file_path)

        # Skip the verbose logging here as colight_dir is not defined yet
        if self.config["verbose"]:
            print(f"[colight] Found {len(forms)} forms in {file_path.name}")

        # Merge metadata with options
        merged_options = file_metadata.merge_with_cli_options(**options)
        final_format = merged_options.pop("format", "markdown")

        # Write .colight files directly to the site directory
        # Calculate the destination path in the site directory
        site_dir = pathlib.Path(config["site_dir"])

        # Get relative path from docs_dir to the file
        docs_dir = pathlib.Path(config["docs_dir"])
        rel_path = file_path.relative_to(docs_dir)

        # Create the output directory in the site based on the file's destination
        # For .colight.py files, we want to match the MkDocs page structure
        if rel_path.name.endswith(".colight.py"):
            # e.g., sliders.colight.py -> sliders.colight
            output_name = rel_path.name[:-3]  # Remove just .py, keep .colight
        elif rel_path.name.endswith(".py"):
            output_name = rel_path.name[:-3]  # Remove .py
        else:
            output_name = rel_path.stem

        # Build the colight output directory path in the site
        colight_dir = site_dir / rel_path.parent / output_name
        colight_dir.mkdir(parents=True, exist_ok=True)

        if self.config["verbose"]:
            print(f"[colight]   Writing .colight files to: {colight_dir}")

        # Setup executor
        executor = SafeFormExecutor(
            colight_dir, output_path_template=self.config["colight_output_path"]
        )

        # Prepare path context for template substitution
        # For basename, remove .colight extension if present
        basename = (
            output_name.replace(".colight", "")
            if output_name.endswith(".colight")
            else output_name
        )
        path_context = {
            "basename": basename,
            "filename": file_path.name,
            "reldir": str(rel_path.parent) if str(rel_path.parent) != "." else "",
            "relpath": str(rel_path.with_suffix("")),
            "abspath": str(file_path.absolute()),
            "absdir": str(file_path.parent.absolute()),
        }

        # Execute forms and collect visualizations
        colight_files = []
        for i, form in enumerate(forms):
            try:
                result = executor.execute_form(form, str(file_path))
                # Pass colight_dir as the output_path so relative paths work correctly
                # The executor expects output_path to be the file being generated
                # So we create a dummy path in the colight_dir
                dummy_output_path = colight_dir / "index.md"
                colight_output = executor.save_colight_visualization(
                    result, i, output_path=dummy_output_path, path_context=path_context
                )
                colight_files.append(colight_output)

                if self.config["verbose"] and colight_output:
                    print(
                        f"[colight]   Form {i}: saved visualization to {colight_output.name}"
                    )

                # Store colight file paths in the file object
                if colight_output:
                    colight_file.colight_files.append(str(colight_output))

            except Exception as e:
                if self.config["verbose"]:
                    print(f"[colight]   Form {i}: execution failed: {e}")
                colight_files.append(None)

        # Generate markdown content
        # The embed path should be relative to the page location
        # Since .colight files are in the same directory as the HTML, use simple paths
        embed_path_template = self.config["colight_embed_path"]

        generator = MarkdownGenerator(
            colight_dir, embed_path_template=embed_path_template
        )

        title = file_path.stem.replace(".colight", "").replace("_", " ").title()

        # Generate content based on format
        if final_format == "html":
            # For now, we'll generate markdown and let MkDocs convert it
            content = generator.generate_markdown(
                forms, colight_files, title, file_path, path_context, **merged_options
            )
        else:
            content = generator.generate_markdown(
                forms, colight_files, title, file_path, path_context, **merged_options
            )

        return content
