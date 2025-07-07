"""Generate JSON representation of documents for LiveServer."""

import json
import base64
import pathlib
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .parser import parse_colight_file
from .executor import DocumentExecutor
from .model import Block, TagSet
from .builder import BuildConfig


@dataclass
class JsonDocumentGenerator:
    """Convert parsed documents to JSON representation for client-side rendering."""

    config: BuildConfig = field(default_factory=BuildConfig)

    def generate_json(self, source_path: pathlib.Path) -> str:
        """Generate JSON representation of a Python file."""
        # Parse the file
        document = parse_colight_file(source_path)

        # Apply config pragma if any
        if self.config.pragma:
            document.tags = document.tags | TagSet(frozenset(self.config.pragma))

        # Execute document
        executor = DocumentExecutor(verbose=self.config.verbose)
        results, _ = executor.execute(document, str(source_path))

        # Build JSON structure
        doc = {
            "file": str(source_path.name),
            "metadata": {
                "pragma": sorted(list(document.tags.flags)),
                "title": source_path.stem,
            },
            "blocks": [],
        }

        # Convert each block to JSON
        for i, (block, result) in enumerate(zip(document.blocks, results)):
            json_block = self._block_to_json(block, result, document.tags, i)
            if json_block:  # Skip empty blocks
                doc["blocks"].append(json_block)

        return json.dumps(doc, indent=2)

    def _block_to_json(
        self, block: Block, result, file_tags: TagSet, block_id: int
    ) -> Optional[Dict[str, Any]]:
        """Convert a single block to JSON representation."""

        # Merge file and block tags
        tags = file_tags | block.tags

        # Build elements array with visibility flags
        elements = []

        # Process each element in order
        for elem in block.elements:
            elem_data: Dict[str, Any] = {"type": elem.kind.lower()}

            if elem.kind == "PROSE":
                prose_text = elem.content if isinstance(elem.content, str) else ""
                elem_data["value"] = prose_text.strip()
                elem_data["show"] = tags.show_prose()

            elif elem.kind in ("STATEMENT", "EXPRESSION"):
                code = elem.get_source().strip()
                elem_data["value"] = code

                # Determine visibility
                if elem.kind == "STATEMENT":
                    elem_data["show"] = tags.show_statements()
                else:  # EXPRESSION
                    elem_data["show"] = tags.show_code()

            # Skip empty elements
            if elem_data.get("value"):
                elements.append(elem_data)

        # Add visual data to last element if it's an expression with a result
        if (
            block.has_expression_result
            and result.colight_bytes is not None
            and tags.show_visuals()
        ):
            # Find the last expression element in our elements array
            for elem in reversed(elements):
                if elem["type"] == "expression":
                    # Base64 encode the visual data
                    elem["visual"] = base64.b64encode(result.colight_bytes).decode(
                        "ascii"
                    )
                    break

        # Skip blocks with no visible content
        if not elements:
            return None

        return {
            "id": f"block-{block_id:03d}",
            "elements": elements,
            "error": result.error if result.error else None,
            "showsVisual": block.has_expression_result and tags.show_visuals(),
        }


# For backwards compatibility, keep the old name as an alias
JsonFormGenerator = JsonDocumentGenerator
