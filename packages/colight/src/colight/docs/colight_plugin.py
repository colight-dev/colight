"""MkDocs plugin for processing .colight.py files.

This plugin integrates colight-site functionality into MkDocs, allowing
.colight.py files to be processed and rendered as part of the documentation.
"""

import pathlib
from typing import Optional, Dict, Any

from mkdocs.config.base import Config
from mkdocs.config.config_options import Type, Choice, DictOfItems
from mkdocs.plugins import BasePlugin
from mkdocs.structure.files import File, Files
from mkdocs.structure.pages import Page

# Import colight-site components
# Since this is in the same monorepo, we can import directly
import sys
from pathlib import Path

# Add colight-site to path if it's not already there
colight_site_path = (
    Path(__file__).parent.parent.parent.parent.parent / "colight-site" / "src"
)
if colight_site_path.exists():
    sys.path.insert(0, str(colight_site_path))

try:
    from colight_site import api
    from colight_site.parser import (
        is_colight_file,
    )  # Still need this for file detection
    from colight_site.generator import MarkdownGenerator  # For custom subclass only
except ImportError as e:
    raise ImportError(
        f"colight-site package is required for the colight MkDocs plugin: {e}. "
        "Make sure it's installed or available in the Python path."
    )


class ColightMarkdownGenerator(MarkdownGenerator):
    """Custom MarkdownGenerator for MkDocs integration."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        base_output_dir: str,
        inline_threshold: int = 50000,
    ):
        super().__init__(output_dir, inline_threshold=inline_threshold)
        self.base_output_dir = base_output_dir

    def _get_relative_path(
        self, colight_file: pathlib.Path, output_path: Optional[pathlib.Path]
    ) -> str:
        """Get relative path for colight file embeds in MkDocs context."""
        if colight_file is None:
            return ""

        # For MkDocs, we want paths relative to the site root
        # The colight files will be in the site_dir under base_output_dir
        relative_path = f"{self.base_output_dir}/{colight_file.name}"
        return relative_path


class ColightPlugin(BasePlugin):
    """MkDocs plugin for processing .colight.py files."""

    config_scheme = (
        ("output_dir", Type(str, default="colight_files")),
        ("format", Choice(["markdown", "html"], default="markdown")),
        ("hide_statements", Type(bool, default=False)),
        ("hide_visuals", Type(bool, default=False)),
        ("hide_code", Type(bool, default=False)),
        ("verbose", Type(bool, default=False)),
        ("inline_threshold", Type(int, default=50000)),
        ("file_options", DictOfItems(Type(dict), default={})),
    )

    def on_config(self, config: Config):  # pyright: ignore
        """Add source files to watch list during development."""
        if not hasattr(config, "watch"):
            setattr(config, "watch", [])

        # Watch for .colight.py files in docs directory
        docs_dir = pathlib.Path(config["docs_dir"])
        for colight_file in docs_dir.rglob("*.colight.py"):
            config["watch"].append(str(colight_file))

        return config

    def on_files(self, files: Files, *, config: Config) -> Files:
        """Process .colight.py files and add generated files to the collection."""
        docs_dir = pathlib.Path(config["docs_dir"])
        site_dir = pathlib.Path(config["site_dir"])

        # Find all .colight.py files
        colight_files_to_process = []
        for file in files:
            if is_colight_file(pathlib.Path(file.src_path)):
                colight_files_to_process.append(file)

        # Process each .colight.py file
        for src_file in colight_files_to_process:
            self._process_colight_file(src_file, files, docs_dir, site_dir, config)

        return files

    def _process_colight_file(
        self,
        src_file: File,
        files: Files,
        docs_dir: pathlib.Path,
        site_dir: pathlib.Path,
        config: Config,
    ):
        """Process a single .colight.py file."""
        src_path = docs_dir / src_file.src_path

        # Determine output path (replace .colight.py with .md)
        if src_file.src_path.endswith(".colight.py"):
            output_rel_path = (
                src_file.src_path[:-11] + ".md"
            )  # Remove .colight.py, add .md
        else:
            output_rel_path = src_file.src_path + ".md"

        output_path = docs_dir / output_rel_path

        # Get file-specific options
        file_options = self._get_file_options(src_file.src_path)

        if self.config["verbose"]:
            print(f"Processing {src_path} -> {output_path}")

        try:
            # Get file-specific options to override defaults
            merged_options = file_options  # file_options already contains overrides

            # Setup output directory for colight files
            colight_output_dir = (
                site_dir
                / self.config["output_dir"]
                / pathlib.Path(output_rel_path).parent
            )
            colight_output_dir.mkdir(parents=True, exist_ok=True)

            # Process the colight file using the public API
            result = api.process_colight_file(
                src_path,
                output_dir=colight_output_dir,
                inline_threshold=self.config["inline_threshold"],
                format="markdown",
                verbose=self.config["verbose"],
                hide_statements=merged_options.get(
                    "hide_statements", self.config["hide_statements"]
                ),
                hide_visuals=merged_options.get(
                    "hide_visuals", self.config["hide_visuals"]
                ),
                hide_code=merged_options.get("hide_code", self.config["hide_code"]),
            )

            # Extract visualization files from the result
            # For MkDocs, we need actual file paths, so save any inline data
            colight_files = []
            for i, pf in enumerate(result.forms):
                if isinstance(pf.visualization_data, pathlib.Path):
                    colight_files.append(pf.visualization_data)
                elif pf.visualization_data is not None:
                    # It's bytes - save to disk for MkDocs compatibility
                    colight_file = colight_output_dir / f"form-{i:03d}.colight"
                    colight_file.write_bytes(pf.visualization_data)
                    colight_files.append(colight_file)
                else:
                    colight_files.append(None)

            # For MkDocs, we may need custom path handling, so use the custom generator
            # to override how paths are generated in the markdown
            generator = ColightMarkdownGenerator(
                colight_output_dir,
                self.config["output_dir"],
                self.config["inline_threshold"],
            )

            # Extract forms from the result
            forms = [pf.form for pf in result.forms]

            # Generate markdown without title (MkDocs handles titles)
            markdown_content = generator.generate_markdown(
                forms, colight_files, title=None
            )

            # Write the generated markdown file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown_content)

            # Remove the original .colight.py file from files collection
            files.remove(src_file)

            # Add the generated markdown file
            new_file = File(
                path=output_rel_path,
                src_dir=str(docs_dir),
                dest_dir=str(site_dir),
                use_directory_urls=config["use_directory_urls"],
            )
            files.append(new_file)

        except Exception as e:
            if self.config["verbose"]:
                print(f"Error processing {src_path}: {e}")
            # Create error output
            error_content = f"# Error Processing Colight File\n\nCould not process {src_path.name}: {e}\n"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(error_content)

    def _get_file_options(self, src_path: str) -> Dict[str, Any]:
        """Get file-specific options from configuration."""
        for pattern, options in self.config.get("file_options", {}).items():
            if pattern in src_path or pathlib.Path(src_path).match(pattern):
                return options
        return {}

    def _merge_options(
        self, file_metadata, file_options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge options with priority: file_options > plugin config > file metadata."""
        # Start with plugin config defaults
        result = {
            "hide_statements": self.config["hide_statements"],
            "hide_visuals": self.config["hide_visuals"],
            "hide_code": self.config["hide_code"],
        }

        # Apply file metadata
        merged = file_metadata.merge_with_cli_options(**result)

        # Apply file-specific options (highest priority)
        for key in ["hide_statements", "hide_visuals", "hide_code"]:
            if key in file_options:
                merged[key] = file_options[key]

        # Remove format from the options (not needed for markdown generation)
        merged.pop("format", None)

        return merged

    def on_page_content(
        self, html: str, *, page: Page, config: Config, files: Files
    ) -> str:
        """Post-process the HTML to ensure colight embeds work correctly."""
        # Check if this page was generated from a .colight.py file
        if hasattr(page.file, "_was_colight"):
            # Add the embed script if not already present
            embed_script = f'<script src="{config["site_url"]}/dist/embed.js"></script>'
            if embed_script not in html:
                # Add before closing body tag
                html = html.replace("</body>", f"{embed_script}\n</body>")

        return html

    def on_post_build(self, *, config: Config) -> None:
        """Copy colight visualization files to the site directory."""
        # The files are already in the right place from execution
        if self.config["verbose"]:
            print(
                f"Colight files saved to {config['site_dir']}/{self.config['output_dir']}"
            )
