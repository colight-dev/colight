"""Generate Markdown and HTML output from executed forms."""

import pathlib
from typing import List, Optional, Dict, Union
import markdown
import base64

from colight_site.parser import (
    Form,
    should_hide_statements,
    should_hide_visuals,
    should_hide_code,
)
from colight.env import VERSIONED_CDN_DIST_URL
from .constants import DEFAULT_INLINE_THRESHOLD

EMBED_URL = (
    VERSIONED_CDN_DIST_URL + "/embed.js" if VERSIONED_CDN_DIST_URL else "/dist/embed.js"
)


class MarkdownGenerator:
    """Generate Markdown from forms and their execution results."""

    def __init__(
        self,
        output_dir: pathlib.Path,
        embed_path_template: Optional[str] = None,
        inline_threshold: int = DEFAULT_INLINE_THRESHOLD,
    ):
        self.output_dir = output_dir
        self.output_file_dir = None  # Will be set when generating
        self.embed_path_template = (
            embed_path_template or "{basename}_colight/form-{form:03d}.colight"
        )
        self.inline_threshold = inline_threshold

    def generate_markdown(
        self,
        forms: List[Form],
        colight_data: List[Optional[Union[bytes, pathlib.Path]]],
        title: Optional[str] = None,
        output_path: Optional[pathlib.Path] = None,
        path_context: Optional[Dict[str, str]] = None,
        pragma_tags: Optional[set[str]] = None,
        execution_errors: Optional[List[Optional[str]]] = None,
    ) -> str:
        """Generate complete Markdown document."""
        if pragma_tags is None:
            pragma_tags = set()

        lines = []

        # Add title if provided
        if title:
            lines.append(f"# {title}")
            lines.append("")

        # Process each form
        for i, (form, colight_item) in enumerate(zip(forms, colight_data)):
            # Check if this is a dummy form (markdown-only)
            is_dummy_form = self._is_dummy_form(form)

            # Resolve form-specific settings: per-form metadata overrides file/CLI defaults
            resolved_tags = form.metadata.resolve_with_defaults(pragma_tags)

            # Always add markdown content from comments (separated by blank lines)
            if form.markdown:
                markdown_content = self._process_markdown_lines(form.markdown)
                if markdown_content.strip():
                    lines.append(markdown_content)
                    lines.append("")

            # Determine if we should show the code block
            # If this form explicitly has show-code, it overrides hide_statements
            explicit_show_code = "show-code" in resolved_tags

            if explicit_show_code:
                # show-code pragma overrides both hide_code and hide_statements for this form
                show_code_block = True
            else:
                # Normal logic: skip if hide_code is True OR if hide_statements is True and this is a statement
                show_code_block = not should_hide_code(resolved_tags) and not (
                    should_hide_statements(resolved_tags) and form.is_statement
                )

            if show_code_block:
                # Add code block (but skip dummy forms and literals)
                if not is_dummy_form:
                    code = form.code.strip()
                    show_code = code and not form.is_literal

                    if show_code:
                        lines.append("```python")
                        lines.append(code)
                        lines.append("```")
                        lines.append("")

            # Check for execution errors
            if execution_errors and i < len(execution_errors) and execution_errors[i]:
                # Display error message
                lines.append("```")
                lines.append(execution_errors[i])
                lines.append("```")
                lines.append("")
            # Skip visuals if hide_visuals is True
            elif not should_hide_visuals(resolved_tags):
                # Add colight embed if we have a visualization
                if colight_item and not is_dummy_form:
                    if isinstance(colight_item, bytes):
                        # Already in memory - embed as script tag
                        base64_data = base64.b64encode(colight_item).decode("ascii")
                        lines.append(
                            f'<script type="application/x-colight">\n{base64_data}\n</script>'
                        )
                    elif isinstance(colight_item, pathlib.Path):
                        # It's a Path - check file size to determine embedding method
                        try:
                            file_size = colight_item.stat().st_size
                        except FileNotFoundError:
                            # In tests, the file might not exist - treat as external reference
                            file_size = float("inf")

                        if file_size < self.inline_threshold:
                            # Small file - read and embed as script tag
                            with open(colight_item, "rb") as f:
                                colight_bytes = f.read()
                            base64_data = base64.b64encode(colight_bytes).decode(
                                "ascii"
                            )
                            lines.append(
                                f'<script type="application/x-colight">\n{base64_data}\n</script>'
                            )
                        else:
                            # Large file - use external reference
                            context = {**(path_context or {}), "form": i}
                            embed_path = self.embed_path_template.format(**context)
                            lines.append(
                                f'<div class="colight-embed" data-src="{embed_path}"></div>'
                            )
                    else:
                        # Should not happen, but satisfies type checker
                        raise TypeError(
                            f"Unexpected type for colight_item: {type(colight_item)}"
                        )
                    lines.append("")

        return "\n".join(lines)

    def _is_dummy_form(self, form: Form) -> bool:
        """Check if this form is a dummy form (markdown-only with pass statement)."""
        return form.is_dummy_form

    def _get_relative_path(
        self, colight_file: pathlib.Path, output_path: Optional[pathlib.Path]
    ) -> str:
        """Get relative path from output file to colight file."""
        if output_path:
            try:
                return str(colight_file.relative_to(output_path.parent))
            except ValueError:
                # If relative_to fails, construct path manually
                colight_dir_name = self.output_dir.name
                return str(pathlib.Path(colight_dir_name) / colight_file.name)
        else:
            # Fallback to directory name + filename
            colight_dir_name = self.output_dir.name
            return str(pathlib.Path(colight_dir_name) / colight_file.name)

    def _process_markdown_lines(self, markdown_lines: List[str]) -> str:
        """Process markdown lines from comments."""
        if not markdown_lines:
            return ""

        # Join lines and handle paragraph breaks
        result_lines = []
        current_paragraph = []

        for line in markdown_lines:
            if line.strip() == "":
                # Empty line - end current paragraph
                if current_paragraph:
                    # Check if we should preserve line breaks (e.g., for headers)
                    if self._should_preserve_line_breaks(current_paragraph):
                        result_lines.extend(current_paragraph)
                    else:
                        result_lines.append(" ".join(current_paragraph))
                    current_paragraph = []
                    result_lines.append("")  # Add paragraph break
            else:
                current_paragraph.append(line)

        # Add final paragraph
        if current_paragraph:
            # Check if we should preserve line breaks
            if self._should_preserve_line_breaks(current_paragraph):
                result_lines.extend(current_paragraph)
            else:
                result_lines.append(" ".join(current_paragraph))

        return "\n".join(result_lines)

    # Markdown patterns that require preserved line breaks
    MARKDOWN_PATTERNS = [
        lambda line: line.startswith("#"),  # Headers
        lambda line: all(c in "=-" for c in line) and len(line) >= 3,  # Underlines
        lambda line: line.startswith(("-", "*", "+")) and len(line) > 1,  # Lists
        lambda line: line.split(".")[0].isdigit(),  # Numbered lists
        lambda line: line.startswith(">"),  # Blockquotes
        lambda line: line.startswith(("```", "~~~")),  # Code fences
        lambda line: "|" in line,  # Tables
    ]

    def _should_preserve_line_breaks(self, lines: List[str]) -> bool:
        """Check if line breaks should be preserved for this block of lines."""
        for line in lines:
            stripped = line.strip()
            if stripped and any(
                pattern(stripped) for pattern in self.MARKDOWN_PATTERNS
            ):
                return True
        return False

    def write_markdown_file(self, content: str, output_path: pathlib.Path):
        """Write markdown content to a file."""
        self.output_file_dir = output_path.parent
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    def write_html_file(self, content: str, output_path: pathlib.Path):
        """Write HTML content to a file."""
        self.output_file_dir = output_path.parent
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    def generate_html(
        self,
        forms: List[Form],
        colight_data: List[Optional[Union[bytes, pathlib.Path]]],
        title: Optional[str] = None,
        output_path: Optional[pathlib.Path] = None,
        path_context: Optional[Dict[str, str]] = None,
        pragma_tags: Optional[set[str]] = None,
        execution_errors: Optional[List[Optional[str]]] = None,
    ) -> str:
        """Generate complete HTML document with embedded visualizations."""
        # First generate markdown content
        markdown_content = self.generate_markdown(
            forms,
            colight_data,
            title,
            output_path,
            path_context,
            pragma_tags=pragma_tags,
            execution_errors=execution_errors,
        )

        # Convert markdown to HTML
        # Use md_in_html extension to preserve raw HTML (like script tags)
        md = markdown.Markdown(extensions=["codehilite", "fenced_code", "md_in_html"])
        html_content = md.convert(markdown_content)

        # Wrap in HTML template
        return self._wrap_html_template(html_content, title or "Colight Document")

    def _wrap_html_template(self, content: str, title: str) -> str:
        """Wrap content in HTML template with Colight embed support."""
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
        }}

        
        code {{
            background: #f4f4f4;
            color: #333;
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
