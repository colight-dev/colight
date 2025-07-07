"""Generate JSON representation of documents for LiveServer."""

import json
import base64
import hashlib
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

        # Generate file hash for unique block IDs
        file_hash = hashlib.sha256(str(source_path).encode()).hexdigest()[:6]

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
            json_block = self._block_to_json(
                block, result, document.tags, i, file_hash, source_path
            )
            if json_block:  # Skip empty blocks
                doc["blocks"].append(json_block)

        return json.dumps(doc, indent=2)

    def _block_to_json(
        self,
        block: Block,
        result,
        file_tags: TagSet,
        block_id: int,
        file_hash: str,
        source_path: pathlib.Path,
    ) -> Optional[Dict[str, Any]]:
        """Convert a single block to JSON representation with stable IDs and hashing."""

        # Merge file and block tags
        tags = file_tags | block.tags

        # Build elements array with visibility flags
        elements = []
        interface_parts = []
        content_parts = []

        # Process each element in order
        for elem in block.elements:
            elem_data: Dict[str, Any] = {"type": elem.kind.lower()}

            if elem.kind == "PROSE":
                prose_text = elem.content if isinstance(elem.content, str) else ""
                elem_data["value"] = prose_text.strip()
                elem_data["show"] = tags.show_prose()
                content_parts.append(prose_text)

            elif elem.kind in ("STATEMENT", "EXPRESSION"):
                code = elem.get_source().strip()
                elem_data["value"] = code
                content_parts.append(code)

                # Determine visibility
                if elem.kind == "STATEMENT":
                    elem_data["show"] = tags.show_statements()
                    # Extract key identifiers for interface hash (simplified)
                    # TODO: Use LibCST metadata for more accurate extraction
                    if code.startswith(("def ", "class ", "import ", "from ")):
                        # Extract the first identifier after the keyword
                        parts = code.split()
                        if len(parts) > 1:
                            interface_parts.append(f"stmt:{parts[1].split('(')[0]}")
                else:  # EXPRESSION
                    elem_data["show"] = tags.show_code()
                    # Include expression in interface hash
                    interface_parts.append(f"expr:{code}")

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

        # Generate stable block ID
        # Use block.id if available (should be unique), otherwise use ordinal
        unique_id = block.id if block.id != 0 else block_id
        stable_id = f"{file_hash}-B{unique_id:05d}"

        # Generate interface hash (for future dependency tracking)
        interface_text = "\n".join(sorted(interface_parts)) if interface_parts else ""
        interface_hash = hashlib.sha256(interface_text.encode()).hexdigest()[:16]

        # Generate content hash (for change detection)
        content_text = "\n".join(content_parts)
        content_hash = hashlib.sha256(content_text.encode()).hexdigest()[:16]

        return {
            "id": stable_id,
            "interface_hash": interface_hash,
            "content_hash": content_hash,
            "line": block.start_line,
            "ordinal": block_id,
            "elements": elements,
            "error": result.error if result.error else None,
            "showsVisual": block.has_expression_result and tags.show_visuals(),
        }


# For backwards compatibility, keep the old name as an alias
JsonFormGenerator = JsonDocumentGenerator
