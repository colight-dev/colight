#!/usr/bin/env python3

from colight_site.parser import parse_colight_file
from colight_site.generator import MarkdownGenerator
import pathlib
from typing import List, Optional

file_path = pathlib.Path("tests/examples/01_basic_numpy.colight.py")

print("=== Testing user's actual file ===")
forms, metadata = parse_colight_file(file_path)

print(f"File metadata: {metadata}")
print(f"Found {len(forms)} forms")

for i, form in enumerate(forms):
    if not form.is_dummy_form:
        print(f"Form {i}: {repr(form.code[:40])}...")
        print(f"  metadata: {form.metadata}")

# Test with generator
output_dir = pathlib.Path("/tmp")
generator = MarkdownGenerator(output_dir)

merged_options = metadata.merge_with_cli_options()
generator_options = {k: v for k, v in merged_options.items() if k != "format"}

colight_files: List[Optional[pathlib.Path]] = [None] * len(forms)

markdown = generator.generate_markdown(
    forms, colight_files, title="Test", **generator_options
)

print("\n=== Generated Markdown ===")
print(markdown)
