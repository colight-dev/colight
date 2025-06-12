import os
import sys
from pathlib import Path
from typing import Union

# Add src to path for colight imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from colight.env import WIDGET_URL, PARENT_PATH


def create_embed_example(
    colight_path: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
    use_local_embed: bool = False,
) -> str:
    """Create a minimal HTML example demonstrating .colight embedding."""
    colight_path = Path(colight_path)

    if output_dir is None:
        output_dir = colight_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    rel_path = os.path.relpath(colight_path, output_dir)

    # Determine script URL
    script_url = "https://cdn.jsdelivr.net/npm/@colight/core/embed.js"  # Default

    if use_local_embed:
        local_embed_path = PARENT_PATH / "dist/embed.js"
        if local_embed_path.exists():
            script_url = f"./{os.path.relpath(local_embed_path, output_dir)}"
        else:
            print("Warning: Local embed.js not found. Using CDN.")
    elif isinstance(WIDGET_URL, str):
        script_url = str(WIDGET_URL).replace("widget.mjs", "embed.js")

    example_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Colight Example</title>
</head>
<body>
    <h1>Colight Visualization</h1>

    <!-- Simple embedding -->
    <div class="colight-embed" data-src="{rel_path}"></div>

    <!-- Programmatic embedding -->
    <div id="viz"></div>
    <button onclick="addAnother()">Add Another</button>
    <div id="container"></div>

    <script type="module">
        import {{ loadVisual, loadVisuals }} from "{script_url}";

        // Load into specific element
        loadVisual("#viz", "{rel_path}");

        // Add button functionality
        window.addAnother = () => {{
            const div = document.createElement("div");
            div.className = "colight-embed";
            div.setAttribute("data-src", "{rel_path}");
            document.getElementById("container").appendChild(div);
            loadVisuals();
        }};
    </script>
</body>
</html>"""

    example_path = output_dir / f"{colight_path.stem}_example.html"
    with open(example_path, "w") as f:
        f.write(example_html)
    return str(example_path)
