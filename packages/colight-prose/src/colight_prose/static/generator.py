"""Document-based generators for Markdown and HTML output."""

import base64
import pathlib
from typing import Dict, List, Optional

import markdown
from colight.env import VERSIONED_CDN_DIST_URL

from colight_prose.constants import DEFAULT_INLINE_THRESHOLD
from colight_prose.executor import ExecutionResult
from colight_prose.model import Block, Document, Element

EMBED_URL = (
    VERSIONED_CDN_DIST_URL + "/embed.js" if VERSIONED_CDN_DIST_URL else "/dist/embed.js"
)


class MarkdownGenerator:
    """Generate Markdown from Documents and execution results."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        embed_path_template: Optional[str] = None,
        inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    ):
        self.output_dir = output_dir
        self.embed_path_template = (
            embed_path_template or "{basename}_colight/block-{block:03d}.colight"
        )
        self.inline_threshold = inline_threshold

    def generate(
        self,
        document: Document,
        results: List[ExecutionResult],
        path_context: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate Markdown content from document and execution results.

        Args:
            document: The parsed document
            results: Execution results, one per block
            path_context: Context for embed path template

        Returns:
            Generated Markdown as string
        """
        lines = []

        for block_idx, (block, result) in enumerate(zip(document.blocks, results)):
            # Merge block tags with document tags
            tags = document.tags | block.tags

            # Process elements in order
            for elem in block.elements:
                if elem.kind == "PROSE" and tags.show_prose():
                    prose_text = elem.content if isinstance(elem.content, str) else ""
                    processed = self._process_prose(prose_text)
                    if processed.strip():
                        lines.append(processed)
                        lines.append("")

                elif elem.kind in ("STATEMENT", "EXPRESSION"):
                    # Check visibility
                    show_element = False
                    if elem.kind == "STATEMENT":
                        show_element = tags.show_statements()
                    elif elem.kind == "EXPRESSION":
                        show_element = tags.show_code()

                    if show_element:
                        code = elem.get_source().strip()
                        # Skip empty code and literal expressions
                        if code and not (
                            elem.kind == "EXPRESSION"
                            and self._is_literal_expr(elem, block)
                        ):
                            lines.append("```python")
                            lines.append(code)
                            lines.append("```")
                            lines.append("")

            # Check for execution error
            if result.error:
                lines.append("```")
                lines.append(result.error.strip())
                lines.append("```")
                lines.append("")

            # Add visualization if applicable
            elif (
                tags.show_visuals()
                and result.colight_bytes
                and block.has_expression_result
                and not block.is_empty
            ):
                lines.append(
                    self._embed_visualization(
                        result.colight_bytes, block_idx, path_context
                    )
                )
                lines.append("")

        return "\n".join(lines).rstrip()

    def _process_prose(self, text: str) -> str:
        """Process prose text (currently just returns as-is)."""
        # In the future, could handle special markdown processing here
        return text

    def _is_literal_expr(self, elem: Element, block: Block) -> bool:
        """Check if element is a literal expression (should be hidden)."""
        # For now, simplified check - could use parser's _is_literal_value
        if elem != block.last_element:
            return False

        code = elem.get_source().strip()
        # Very basic literal detection
        try:
            import ast

            ast.literal_eval(code)
            return True
        except (SyntaxError, ValueError):
            return False

    def _embed_visualization(
        self,
        colight_bytes: bytes,
        block_idx: int,
        path_context: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate embed code for visualization."""
        if len(colight_bytes) < self.inline_threshold:
            # Small visualization - inline as base64
            base64_data = base64.b64encode(colight_bytes).decode("ascii")
            return f'<script type="application/x-colight">\n{base64_data}\n</script>'
        else:
            # Large visualization - external reference
            context = {**(path_context or {}), "block": block_idx}
            embed_path = self.embed_path_template.format(**context)
            return f'<div class="colight-embed" data-src="{embed_path}"></div>'


class HTMLGenerator:
    """Generate HTML from Documents and execution results."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        embed_path_template: Optional[str] = None,
        inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    ):
        self.markdown_generator = MarkdownGenerator(
            output_dir, embed_path_template, inline_threshold
        )

    def generate(
        self,
        document: Document,
        results: List[ExecutionResult],
        title: Optional[str] = None,
        path_context: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate HTML content from document and execution results.

        Args:
            document: The parsed document
            results: Execution results, one per block
            title: HTML page title
            path_context: Context for embed path template

        Returns:
            Generated HTML as string
        """
        # First generate Markdown
        markdown_content = self.markdown_generator.generate(
            document, results, path_context
        )

        # Convert to HTML
        md = markdown.Markdown(extensions=["codehilite", "fenced_code", "md_in_html"])
        html_content = md.convert(markdown_content)

        # Wrap in template
        return self._wrap_template(html_content, title or "Colight Document")

    def _wrap_template(self, content: str, title: str) -> str:
        """Wrap content in HTML template."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
        }}
        
        .prose > * {{
            margin-top: 1em;
            margin-bottom: 1em;
        }}
        
        .prose {{
            font-size: 14px;
        }}
        
        .prose pre {{
            background: #f4f4f4;
            color: #333;
            padding: 1em;
            border-radius: 4px;
            overflow-x: auto;
        }}
        
        code {{
            background: #f4f4f4;
            color: #333;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        
        .colight-embed {{
            margin: 1em 0;
        }}
    </style>
    <script src="{EMBED_URL}"></script>
    <script>colight.api.tw("prose")</script>
</head>
<body>
    <div class='prose'>
        {content}
    </div>
</body>
</html>"""


def write_colight_files(
    output_dir: pathlib.Path,
    results: List[ExecutionResult],
    basename: str = "block",
) -> List[Optional[pathlib.Path]]:
    """Write Colight visualization files to output directory.

    Args:
        output_dir: Directory to write files to
        results: Execution results containing colight_bytes
        basename: Base name for files

    Returns:
        List of paths (or None) for each visualization
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i, result in enumerate(results):
        if result.colight_bytes:
            path = output_dir / f"{basename}-{i:03d}.colight"
            path.write_bytes(result.colight_bytes)
            paths.append(path)
        else:
            paths.append(None)

    return paths
