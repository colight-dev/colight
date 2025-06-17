"""Generate Markdown output from executed forms."""

import pathlib
from typing import List, Optional

from .parser import Form


class MarkdownGenerator:
    """Generate Markdown from forms and their execution results."""

    def __init__(self, output_dir: pathlib.Path):
        self.output_dir = output_dir

    def generate_markdown(
        self,
        forms: List[Form],
        colight_files: List[Optional[pathlib.Path]],
        title: Optional[str] = None,
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
                    # Use just the filename for simplicity and predictability
                    lines.append(
                        f'<div class="colight-embed" data-src="{colight_file.name}"></div>'
                    )
                    lines.append("")

        return "\n".join(lines)

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

    def generate_html_template(
        self, markdown_content: str, title: str = "Colight Document"
    ) -> str:
        """Generate a complete HTML document with Colight embed support."""
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
        
        .colight-embed {{
            margin: 1rem 0;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #f9f9f9;
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
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div id="content"></div>
    
    <script>
        // Parse markdown and render
        const markdownContent = `{markdown_content}`;
        document.getElementById('content').innerHTML = marked.parse(markdownContent);
        
        // Initialize colight embeds
        document.addEventListener('DOMContentLoaded', function() {{
            const embeds = document.querySelectorAll('.colight-embed');
            embeds.forEach(embed => {{
                const src = embed.getAttribute('data-src');
                if (src) {{
                    // Load and display colight file
                    fetch(src)
                        .then(response => response.json())
                        .then(data => {{
                            embed.innerHTML = `<pre>${{JSON.stringify(data, null, 2)}}</pre>`;
                        }})
                        .catch(error => {{
                            embed.innerHTML = `<p>Error loading visualization: ${{src}}</p>`;
                        }});
                }}
            }});
        }});
    </script>
</body>
</html>"""
