"""Generate JSON representation of documents and file indexes for the frontend."""

import base64
import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .executor import DocumentExecutor
from .incremental_executor import IncrementalExecutor
from .model import Block, TagSet
from .parser import parse_colight_file

# Threshold for inline vs external visual storage
VISUAL_INLINE_THRESHOLD = 50 * 1024  # 50KB


@dataclass
class JsonDocumentGenerator:
    """Convert parsed documents to JSON representation for client-side rendering."""

    verbose: bool = False
    pragma: set[str] = field(default_factory=set)
    visual_store: Optional[Dict[str, bytes]] = None  # Storage for large visuals
    incremental_executor: Optional[IncrementalExecutor] = (
        None  # For incremental execution
    )

    def generate_json(
        self, source_path: pathlib.Path, changed_blocks: Optional[set] = None
    ) -> str:
        """Generate JSON representation of a Python file.

        Args:
            source_path: Path to the source file
            changed_blocks: Set of block IDs that have changed (for incremental execution)
        """
        # Parse the file
        document = parse_colight_file(source_path)

        # Apply pragma if any
        if self.pragma:
            document.tags = document.tags | TagSet(frozenset(self.pragma))

        # Execute document incrementally if we have an executor
        if self.incremental_executor:
            block_results = list(
                self.incremental_executor.execute_incremental_streaming(
                    document, changed_blocks, str(source_path), str(source_path)
                )
            )
            # Extract just the results in order
            results = [result for block, result in block_results]
        else:
            # Fall back to regular execution
            executor = DocumentExecutor(verbose=self.verbose)
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
            # Use block's ID (which is its cache key)
            json_block = self._block_to_json(
                block, result, document.tags, i, block.id, source_path
            )
            if json_block:  # Skip empty blocks
                doc["blocks"].append(json_block)

        return json.dumps(doc, indent=2)

    def execute_incremental_with_results(
        self, source_path: pathlib.Path, document=None
    ):
        """Execute a document incrementally and yield (block_id, result_dict) pairs.

        This method is used by the LiveServer to stream results as they are executed.

        Args:
            source_path: Path to the source file
            document: Optional pre-parsed document (if not provided, will parse the file)
        """
        # Parse the file if document not provided
        if document is None:
            document = parse_colight_file(source_path)

        # Apply pragma if any
        if self.pragma:
            document.tags = document.tags | TagSet(frozenset(self.pragma))

        # Execute incrementally in document order (streaming)
        if self.incremental_executor:
            # Use the new streaming method that executes in document order
            i = 0
            for (
                block,
                result,
            ) in self.incremental_executor.execute_incremental_streaming(
                document, str(source_path), str(source_path)
            ):
                # Use block's ID (which is its cache key)
                json_block = self._block_to_json(
                    block, result, document.tags, i, block.id, source_path
                )
                if json_block:
                    # Extract the parts we need for the WebSocket message
                    result_dict = {
                        "ok": not bool(json_block.get("error")),
                        "stdout": json_block.get("stdout", ""),
                        "error": json_block.get("error"),
                        "showsVisual": json_block.get("showsVisual", False),
                        "elements": json_block.get("elements", []),
                        "cache_hit": getattr(result, "cache_hit", False),
                        "content_changed": getattr(result, "content_changed", False),
                        "ordinal": i,  # Position in document
                    }
                    yield block.id, result_dict
                i += 1
        else:
            # Fall back to regular execution
            executor = DocumentExecutor(verbose=self.verbose)
            results, _ = executor.execute(document, str(source_path))

            # For non-incremental execution
            for i, (block, result) in enumerate(zip(document.blocks, results)):
                # Use block's ID (cache key)
                json_block = self._block_to_json(
                    block, result, document.tags, i, block.id, source_path
                )
                if json_block:
                    result_dict = {
                        "ok": not bool(json_block.get("error")),
                        "stdout": json_block.get("stdout", ""),
                        "error": json_block.get("error"),
                        "showsVisual": json_block.get("showsVisual", False),
                        "elements": json_block.get("elements", []),
                        "cache_hit": False,
                        "content_changed": False,
                        "ordinal": i,  # Position in document
                    }
                    yield block.id, result_dict

    def _block_to_json(
        self,
        block: Block,
        result,
        file_tags: TagSet,
        block_ordinal: int,
        block_id: str,
        source_path: pathlib.Path,
    ) -> Optional[Dict[str, Any]]:
        """Convert a single block to JSON representation using cache key as ID."""

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
                    visual_size = len(result.colight_bytes)

                    # Include metadata
                    elem["visual_meta"] = {"size": visual_size, "format": "colight"}

                    if visual_size <= VISUAL_INLINE_THRESHOLD:
                        # Inline small visuals as before
                        elem["visual"] = base64.b64encode(result.colight_bytes).decode(
                            "ascii"
                        )
                    else:
                        # Use content hash for deduplication and caching
                        visual_hash = hashlib.sha256(result.colight_bytes).hexdigest()[
                            :16
                        ]
                        visual_key = f"{visual_hash}"

                        # Store large visuals externally (deduped by hash)
                        if self.visual_store is not None:
                            self.visual_store[visual_key] = result.colight_bytes

                        # Store reference for large visuals
                        elem["visual_ref"] = {
                            "id": visual_key,
                            "url": f"/api/visual/{visual_key}",
                            "size": visual_size,
                            "format": "colight",
                        }
                    break

        # Skip blocks with no visible content
        if not elements:
            return None

        # Generate interface hash (for future dependency tracking)
        interface_text = "\n".join(sorted(interface_parts)) if interface_parts else ""
        interface_hash = hashlib.sha256(interface_text.encode()).hexdigest()[:16]

        # Generate content hash (for change detection)
        content_text = "\n".join(content_parts)
        content_hash = hashlib.sha256(content_text.encode()).hexdigest()[:16]

        return {
            "id": block_id,  # Now using cache key as ID
            "interface": {
                "provides": block.interface.provides,
                "requires": block.interface.requires,
            },
            "interface_hash": interface_hash,
            "content_hash": content_hash,
            "line": block.start_line,
            "ordinal": block_ordinal,
            "elements": elements,
            "error": result.error if result.error else None,
            "stdout": result.output if result.output else None,
            "showsVisual": block.has_expression_result and tags.show_visuals(),
        }


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
        html_path = str(rel_path.with_suffix(".html"))
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
                        items.append(
                            {
                                "name": name,
                                "path": current_path,
                                "type": "directory",
                                "children": children,
                            }
                        )

        return items

    root["children"] = dict_to_tree(tree_dict)
    return root
