"""Generate index pages for colight-site projects."""

import json
import pathlib
from typing import Dict, List, Optional, Any
import fnmatch


# Deprecated - use generate_index_json instead
def generate_index_html(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    include: Optional[List[str]] = None,
    ignore: Optional[List[str]] = None,
) -> None:
    """Deprecated: Generate an index.html file. Use generate_index_json instead."""
    raise DeprecationWarning(
        "generate_index_html is deprecated. Use generate_index_json instead."
    )


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

    # First check ignore patterns - check all parts of the path
    if ignore_patterns:
        for part in file_path.parts:
            for pattern in ignore_patterns:
                if fnmatch.fnmatch(part, pattern):
                    return False

        # Also check the full path
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(file_str, pattern) or fnmatch.fnmatch(
                file_path.name, pattern
            ):
                return False

    # Check if file matches any include pattern
    matches_include = any(
        fnmatch.fnmatch(file_str, pattern) or fnmatch.fnmatch(file_path.name, pattern)
        for pattern in include_patterns
    )

    return matches_include










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
    files = find_colight_files(input_path, include, ignore)

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
        "children": []
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
        current_dict[file_name] = {
            "type": "file",
            "htmlPath": html_path
        }

    # Convert the dict structure to the desired JSON format
    def dict_to_tree(d: dict, path: str = "") -> List[Dict[str, Any]]:
        items = []
        
        for name, value in sorted(d.items()):
            current_path = f"{path}/{name}" if path else name
            
            if isinstance(value, dict):
                if value.get("type") == "file":
                    # It's a file
                    items.append({
                        "name": name,
                        "path": current_path,
                        "type": "file",
                        "htmlPath": value["htmlPath"]
                    })
                else:
                    # It's a directory
                    children = dict_to_tree(value, current_path)
                    if children:  # Only add non-empty directories
                        items.append({
                            "name": name,
                            "path": current_path + "/",
                            "type": "directory",
                            "children": children
                        })
        
        return items

    root["children"] = dict_to_tree(tree_dict)
    return root
