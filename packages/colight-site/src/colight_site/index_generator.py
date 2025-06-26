"""Generate index pages for colight-site projects."""

import pathlib
from typing import Dict, List, Optional
import fnmatch


def generate_index_html(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    include: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
) -> None:
    """Generate an index.html file listing all colight files in a directory structure.

    Args:
        input_path: The directory being watched
        output_path: The output directory where index.html will be created
        include: List of glob patterns to include
        ignore: List of glob patterns to ignore
    """
    if include is None:
        include = ["*.py"]

    # First generate a Python file that will be processed by colight-site
    index_py_path = output_path / "index.py"
    generate_index_python(input_path, index_py_path, include, ignore)

    # Then process it through colight-site to generate the HTML
    from . import api
    from .builder import BuildConfig

    config = BuildConfig(
        formats={"html"},
        pragma_tags={"hide-all-code"},  # Hide the code that generates the index
        verbose=False,
    )

    index_html_path = output_path / "index.html"
    api.build_file(index_py_path, index_html_path, config=config)

    # Clean up the intermediate Python file
    index_py_path.unlink()


def find_colight_files(
    input_path: pathlib.Path,
    include: List[str],
    ignore: Optional[List[str]] = None,
) -> List[pathlib.Path]:
    """Find all files matching the include patterns."""
    files = []

    if input_path.is_file():
        # Single file mode
        return [input_path] if matches_patterns(input_path, include, ignore) else []

    # Directory mode
    for pattern in include:
        for file_path in input_path.rglob(pattern):
            if file_path.is_file() and matches_patterns(file_path, include, ignore):
                files.append(file_path)

    return sorted(set(files))  # Remove duplicates and sort


def matches_patterns(
    file_path: pathlib.Path,
    include_patterns: List[str],
    ignore_patterns: Optional[List[str]] = None,
) -> bool:
    """Check if file matches include patterns and doesn't match ignore patterns."""
    file_str = str(file_path)

    # Check if file matches any include pattern
    matches_include = any(
        fnmatch.fnmatch(file_str, pattern) or fnmatch.fnmatch(file_path.name, pattern)
        for pattern in include_patterns
    )

    if not matches_include:
        return False

    # Check if file matches any ignore pattern
    if ignore_patterns:
        matches_ignore = any(
            fnmatch.fnmatch(file_str, pattern)
            or fnmatch.fnmatch(file_path.name, pattern)
            for pattern in ignore_patterns
        )
        if matches_ignore:
            return False

    return True


def build_file_tree(
    files: List[pathlib.Path],
    input_path: pathlib.Path,
    output_path: pathlib.Path,
) -> Dict:
    """Build a nested dictionary representing the file tree."""
    tree = {"name": input_path.name, "children": {}, "files": []}

    for file_path in files:
        # Get relative path from input directory
        try:
            rel_path = file_path.relative_to(input_path)
        except ValueError:
            # File is not relative to input path (single file mode)
            rel_path = pathlib.Path(file_path.name)

        # Calculate HTML path
        html_path = pathlib.Path(str(rel_path).replace(".py", ".html"))

        # Add to tree
        current = tree
        parts = list(rel_path.parts)

        # Navigate/create directory structure
        for part in parts[:-1]:
            if part not in current["children"]:
                current["children"][part] = {"name": part, "children": {}, "files": []}
            current = current["children"][part]

        # Add file
        current["files"].append(
            {
                "name": parts[-1],
                "html_path": str(html_path),
                "source_path": str(rel_path),
            }
        )

    return tree


def generate_index_python(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    include: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
) -> None:
    """Generate a Python file that creates an index using Colight."""
    if include is None:
        include = ["*.py"]

    # Find all matching files
    files = find_colight_files(input_path, include, ignore)

    # Build tree structure
    tree = build_file_tree(files, input_path, output_path.parent)

    # Generate Python code
    code = generate_python_code(tree, input_path)

    # Write the Python file
    output_path.write_text(code)


def generate_python_code(tree: Dict, input_path: pathlib.Path) -> str:
    """Generate Python code that uses Colight to create the index."""
    # Build the list items
    items = []
    _build_items_list(tree, items)

    # Format the items as Python list literals
    items_code = ",\n            ".join(items)

    code = f"""# | hide-all-code

import colight.plot as Plot

Plot.html(
    [
        "div.p-6",
        ["h1.text-2xl.font-bold.mb-6", "Colight Examples Directory"],
        ["p.mb-4", "Click on any example file to view it:"],
        [
            "ul.list-disc.pl-6",
            {items_code}
        ],
    ]
)
"""

    return code


def _build_items_list(node: Dict, items: List[str], level: int = 0) -> None:
    """Recursively build list items for the Python code."""
    # Add files
    for file_info in sorted(node.get("files", []), key=lambda x: x["name"]):
        file_name = file_info["name"]
        display_name = file_name.replace(".colight.py", "").replace(".py", "")
        href = file_info["html_path"]

        item = f"""[
                "li.mb-2",
                [
                    "a.text-blue-600.hover:underline",
                    {{"href": "{href}"}},
                    "{display_name}",
                ],
            ]"""
        items.append(item)

    # Add subdirectories
    for name, child in sorted(node.get("children", {}).items()):
        if child.get("files") or child.get("children"):  # Only show non-empty dirs
            # Add directory header
            dir_item = f"""[
                "li.mb-2",
                ["span.font-semibold", "{name}/"],
                ["ul.list-disc.pl-6.mt-2", """

            # Add child items
            child_items = []
            _build_items_list(child, child_items, level + 1)

            if child_items:
                dir_item += "\n                    " + ",\n                    ".join(
                    child_items
                )

            dir_item += """
                ],
            ]"""
            items.append(dir_item)
