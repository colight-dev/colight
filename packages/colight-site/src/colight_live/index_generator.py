"""Generate index pages for colight-site projects."""

import json
import pathlib
from typing import Any, Dict, List, Optional

from colight_site.file_resolver import find_files


def generate_index_json(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    include: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
) -> None:
    """Generate an index.json file listing all colight files in a directory structure.

    Args:
        input_path: The directory being watched
        output_path: The output directory where index.json will be created
        include: List of glob patterns to include
        ignore: List of glob patterns to ignore
    """
    if include is None:
        include = ["*.py"]

    # Find all matching files
    files = find_files(input_path, include, ignore)

    # Build tree structure
    tree = build_file_tree_json(files, input_path)

    # Write JSON file
    index_json_path = output_path / "index.json"
    with open(index_json_path, "w") as f:
        json.dump(tree, f, indent=2)


def build_file_tree_json(
    files: List[pathlib.Path],
    input_path: pathlib.Path,
) -> Dict[str, Any]:
    """Build a nested dictionary representing the file tree in JSON format.

    Returns a structure like:
    {
        "name": "root",
        "path": "/",
        "type": "directory",
        "children": [
            {
                "name": "example.py",
                "path": "example.py",
                "type": "file",
                "htmlPath": "example.html"
            },
            {
                "name": "subfolder",
                "path": "subfolder/",
                "type": "directory",
                "children": [...]
            }
        ]
    }
    """
    root = {
        "name": input_path.name or "root",
        "path": "/",
        "type": "directory",
        "children": [],
    }

    # Build a temporary nested dict structure
    tree_dict = {}

    for file_path in files:
        # Get relative path from input directory
        try:
            rel_path = file_path.relative_to(input_path)
        except ValueError:
            # File is not relative to input path (single file mode)
            rel_path = pathlib.Path(file_path.name)

        # Build nested structure
        parts = rel_path.parts
        current_dict = tree_dict

        # Create nested directories
        for i, part in enumerate(parts[:-1]):
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]

        # Add file
        file_name = parts[-1]
        html_path = str(rel_path).replace(".py", ".html")
        current_dict[file_name] = {"type": "file", "htmlPath": html_path}

    # Convert the dict structure to the desired JSON format
    def dict_to_tree(d: dict, path: str = "") -> List[Dict[str, Any]]:
        items = []

        for name, value in sorted(d.items()):
            current_path = f"{path}/{name}" if path else name

            if isinstance(value, dict):
                if value.get("type") == "file":
                    # It's a file
                    items.append(
                        {
                            "name": name,
                            "path": current_path,
                            "type": "file",
                            "htmlPath": value["htmlPath"],
                        }
                    )
                else:
                    # It's a directory
                    children = dict_to_tree(value, current_path)
                    if children:  # Only add non-empty directories
                        # Check if we should collapse this directory
                        if len(children) == 1 and children[0]["type"] == "directory":
                            # This directory has only one subdirectory, merge them
                            child = children[0]
                            items.append(
                                {
                                    "name": f"{name}/{child['name']}",
                                    "path": child["path"],
                                    "type": "directory",
                                    "children": child["children"],
                                }
                            )
                        else:
                            # Normal directory with multiple children or files
                            items.append(
                                {
                                    "name": name,
                                    "path": current_path + "/",
                                    "type": "directory",
                                    "children": children,
                                }
                            )

        return items

    root["children"] = dict_to_tree(tree_dict)
    return root
