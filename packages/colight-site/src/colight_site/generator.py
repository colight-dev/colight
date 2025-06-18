"""Generate Markdown and HTML output from executed forms."""

import pathlib
from typing import List, Optional
import markdown

from colight_site.parser import Form
from colight.env import VERSIONED_CDN_DIST_URL

EMBED_URL = (
    VERSIONED_CDN_DIST_URL + "/embed.js" if VERSIONED_CDN_DIST_URL else "/dist/embed.js"
)


class MarkdownGenerator:
    """Generate Markdown from forms and their execution results."""

    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir
        self.output_file_dir = None  # Will be set when generating

    def generate_markdown(
        self,
        forms: List[Form],
        colight_files: List[Optional[pathlib.Path]],
        title: Optional[str] = None,
        output_path: Optional[pathlib.Path] = None,
    ) -> str:
        """Generate complete Markdown document."""
        lines = []

        # Add title if provided
        if title:
            lines.append(f"# {title}")
            lines.append("")

        # Process each form
        for i, (form, colight_file) in enumerate(zip(forms, colight_files)):
            # Add markdown content from comments
            if form.markdown:
                markdown_content = self._process_markdown_lines(form.markdown)
                if markdown_content.strip():
                    lines.append(markdown_content)
                    lines.append("")

            # Add code block
            code = form.code.strip()
            if code:
                lines.append("```python")
                lines.append(code)
                lines.append("```")
                lines.append("")

                # Add colight embed if we have a visualization
                if colight_file:
                    embed_path = self._get_relative_path(colight_file, output_path)
                    lines.append(
                        f'<div class="colight-embed" data-src="{embed_path}"></div>'
                    )
                    lines.append("")

        return "\n".join(lines)

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
                    result_lines.append(" ".join(current_paragraph))
                    current_paragraph = []
                    result_lines.append("")  # Add paragraph break
            else:
                current_paragraph.append(line)

        # Add final paragraph
        if current_paragraph:
            result_lines.append(" ".join(current_paragraph))

        return "\n".join(result_lines)

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
        colight_files: List[Optional[pathlib.Path]],
        title: Optional[str] = None,
        output_path: Optional[pathlib.Path] = None,
    ) -> str:
        """Generate complete HTML document with embedded visualizations."""
        # First generate markdown content
        markdown_content = self.generate_markdown(
            forms, colight_files, title, output_path
        )

        # Convert markdown to HTML
        md = markdown.Markdown(extensions=["codehilite", "fenced_code"])
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
        
        pre {{
            background: #f4f4f4;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }}
        
        code {{
            background: #f4f4f4;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        
        pre code {{
            background: none;
            padding: 0;
        }}
    </style>
    <script src="{EMBED_URL}"></script>
    
</head>
<body>
    <div class='prose'>
        {content}
    </div>
</body>
</html>"""
