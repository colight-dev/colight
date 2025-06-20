"""Simplified MkDocs plugin for processing .colight.py files.

This plugin uses colight-site's builder module directly to process files
before MkDocs builds the documentation.
"""

import pathlib
from mkdocs.config import Config
from mkdocs.plugins import BasePlugin
from mkdocs.config.config_options import Type, DictOfItems, ListOfItems

# Import colight-site builder
import sys

colight_site_path = (
    pathlib.Path(__file__).parent.parent.parent.parent.parent / "colight-site" / "src"
)
if colight_site_path.exists():
    sys.path.insert(0, str(colight_site_path))

try:
    from colight_site import builder
except ImportError as e:
    raise ImportError(f"colight-site package is required: {e}")


class SitePlugin(BasePlugin):
    """Simplified MkDocs plugin that processes .colight.py files in-place."""

    config_scheme = (
        ("verbose", Type(bool, default=False)),
        ("output_dir", Type(str, default="content")),
        ("include", ListOfItems(Type(str), default=["*.colight.py"])),
        ("ignore", ListOfItems(Type(str), default=[])),
        ("hide_statements", Type(bool, default=False)),
        ("hide_visuals", Type(bool, default=False)),
        ("hide_code", Type(bool, default=False)),
        (
            "colight_output_path",
            Type(str, default="./{basename}/form-{form:03d}.colight"),
        ),
        ("colight_embed_path", Type(str, default="form-{form:03d}.colight")),
        ("file_options", DictOfItems(Type(dict), default={})),
    )

    def on_pre_build(self, *, config: Config) -> None:
        """Process all Python files matching include patterns before MkDocs builds."""
        docs_dir = pathlib.Path(config["docs_dir"])
        output_dir = docs_dir / self.config["output_dir"]

        if self.config["verbose"]:
            print(f"Processing Python files in {docs_dir}")
            print(f"Include patterns: {self.config['include']}")
            print(f"Ignore patterns: {self.config['ignore']}")
            print(f"Output directory: {output_dir}")

        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)

        # Find all Python files matching include patterns
        python_files = []
        for include_pattern in self.config["include"]:
            for py_file in docs_dir.rglob(include_pattern):
                if py_file.suffix == ".py":
                    # Always ignore __pycache__ directories and __init__.py files
                    if "__pycache__" in py_file.parts or py_file.name == "__init__.py":
                        continue

                    # Check if file should be ignored by user patterns
                    should_ignore = False
                    for ignore_pattern in self.config["ignore"]:
                        if py_file.match(ignore_pattern):
                            should_ignore = True
                            break
                    if not should_ignore:
                        python_files.append(py_file)

        # Remove duplicates and sort
        python_files = sorted(set(python_files))

        if self.config["verbose"]:
            print(f"Found {len(python_files)} Python files to process")

        # Process each file
        for py_file in python_files:
            try:
                # Calculate relative output path
                rel_path = py_file.relative_to(docs_dir)
                output_file = output_dir / builder._get_output_path(
                    rel_path, "markdown"
                )

                if self.config["verbose"]:
                    print(f"Processing {py_file}")

                builder.build_file(
                    py_file,
                    output_file,
                    verbose=self.config["verbose"],
                    format="markdown",
                    hide_statements=self.config["hide_statements"],
                    hide_visuals=self.config["hide_visuals"],
                    hide_code=self.config["hide_code"],
                    colight_output_path=self.config["colight_output_path"],
                    colight_embed_path=self.config["colight_embed_path"],
                )
            except Exception as e:
                if self.config["verbose"]:
                    print(f"Error processing {py_file}: {e}")
                    import traceback

                    traceback.print_exc()

        # Apply file-specific options
        if self.config["file_options"]:
            for pattern, options in self.config["file_options"].items():
                for py_file in python_files:
                    if pattern in str(py_file) or py_file.match(pattern):
                        # Calculate output path relative to output_dir
                        rel_path = py_file.relative_to(docs_dir)
                        output_file = output_dir / builder._get_output_path(
                            rel_path, "markdown"
                        )

                        if self.config["verbose"]:
                            print(f"  Re-processing {py_file} with custom options")

                        try:
                            builder.build_file(
                                py_file,
                                output_file,
                                verbose=self.config["verbose"],
                                format="markdown",
                                colight_output_path=self.config["colight_output_path"],
                                colight_embed_path=self.config["colight_embed_path"],
                                **options,
                            )
                        except Exception as e:
                            if self.config["verbose"]:
                                print(
                                    f"Error processing {py_file} with custom options: {e}"
                                )
