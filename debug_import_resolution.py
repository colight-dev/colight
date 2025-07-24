#!/usr/bin/env python3
"""Debug the import resolution issue."""

import pathlib
import sys
import ast
sys.path.insert(0, "packages/colight-prose/src")

from colight_prose.file_graph import FileDependencyGraph
from colight_prose.module_resolver import resolve_module_to_file

def debug_import_resolution():
    base = pathlib.Path("packages/colight-prose/tests/import-test-fixtures/fallback-test").resolve()
    test_file = base / "test.py"
    
    print(f"Base directory: {base}")
    print(f"Test file: {test_file}")
    print(f"Files in directory: {list(base.glob('*.py'))}")
    print(f"Test file exists: {test_file.exists()}")
    print(f"Config file exists: {(base / 'config.py').exists()}")
    
    # Test resolve_module_to_file directly
    print("\n--- Testing resolve_module_to_file directly ---")
    relative_test_path = "test.py"  # This is what FileDependencyGraph would pass
    result = resolve_module_to_file("config", relative_test_path, str(base))
    print(f"resolve_module_to_file('config', '{relative_test_path}', '{base}') = {result}")
    
    # Test with FileDependencyGraph
    print("\n--- Testing FileDependencyGraph ---")
    graph = FileDependencyGraph(base)
    
    # Check what _get_relative_path returns
    relative_path = graph._get_relative_path(test_file)
    print(f"_get_relative_path({test_file}) = {relative_path}")
    
    # Parse the file manually to see the AST
    print("\n--- Manual AST parsing ---")
    with open(test_file, 'r') as f:
        content = f.read()
        print(f"File content:\n{content}")
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    print(f"Found import: {alias.name}")
    
    # Run the actual analyze_file
    imports = graph.analyze_file(test_file)
    print(f"\nFileDependencyGraph found imports: {imports}")

if __name__ == "__main__":
    debug_import_resolution()