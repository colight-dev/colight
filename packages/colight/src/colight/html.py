import base64
import json
import os
import uuid

import colight.env as env
from colight.format import create_bytes
from colight.util import read_file
from colight.widget import to_json_with_initialState


def encode_string(s):
    return base64.b64encode(s.encode("utf-8")).decode("utf-8")


def encode_buffers(buffers):
    """
    Encode binary buffers as base64 strings for inclusion in JavaScript.

    This function takes a list of binary buffers and returns a JavaScript array literal
    containing the base64-encoded versions of these buffers.

    Args:
        buffers: List of binary buffers to encode

    Returns:
        A string representation of a JavaScript array containing the base64-encoded buffers
    """
    # Encode each buffer as base64
    buffer_entries = [base64.b64encode(buffer).decode("utf-8") for buffer in buffers]

    # Return a proper JSON array of strings
    return json.dumps(buffer_entries)


def get_script_content():
    """Get the JS content either from CDN or local file"""
    if isinstance(env.WIDGET_URL, str):  # It's a CDN URL
        return f'import {{ render }} from "{env.WIDGET_URL}";'
    else:  # It's a local Path
        # Create a blob URL for the module
        content = read_file(env.WIDGET_URL)

        return f"""
            const encodedContent = "{encode_string(content)}";
            const decodedContent = atob(encodedContent);
            const moduleBlob = new Blob([decodedContent], {{ type: 'text/javascript' }});
            const moduleUrl = URL.createObjectURL(moduleBlob);
            const {{ render }} = await import(moduleUrl);
            URL.revokeObjectURL(moduleUrl);
        """


def get_widget_script_url(use_cdn=True, output_dir=None):
    """Get the appropriate script URL for the widget, handling both CDN and local cases."""
    if use_cdn:
        if isinstance(env.WIDGET_URL, str):
            return env.WIDGET_URL
        else:
            return "https://cdn.jsdelivr.net/npm/@colight/core/widget.mjs"
    else:
        # Local development mode
        local_widget_path = env.DIST_PATH / "widget.mjs"
        if local_widget_path.exists():
            if output_dir:
                return f"./{os.path.relpath(local_widget_path, output_dir)}"
            else:
                return str(local_widget_path)
        else:
            raise FileNotFoundError("Local widget.mjs not found. Run `yarn dev`")


def html_snippet(ast, id=None, use_cdn=True, output_dir=None):
    id = id or f"colight-widget-{uuid.uuid4().hex}"
    data, buffers = to_json_with_initialState(ast, buffers=[])

    colight_data = create_bytes(data, buffers)
    colight_base64 = base64.b64encode(colight_data).decode("utf-8")

    # Get the appropriate script URL
    script_url = get_widget_script_url(use_cdn, output_dir)

    html_content = f"""
    <div class="bg-white p3" id="{id}"></div>

    <script type="application/x-colight" data-target="{id}">
        {colight_base64}
    </script>

    <script type="module">
        import {{ render, parseColightScript }} from "{script_url}";;
        const container = document.getElementById('{id}');
        const colightData = parseColightScript(container.nextElementSibling);
        render(container, colightData, '{id}');
    </script>
    """

    return html_content


def html_page(ast, id=None, use_cdn=True, output_dir=None):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Colight Visual</title>
    </head>
    <body>
        {html_snippet(ast, id, use_cdn=use_cdn, output_dir=output_dir)}
    </body>
    </html>
    """
