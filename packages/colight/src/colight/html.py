import base64
import json
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
    if env.VERSIONED_CDN_DIST_URL:
        return (
            f'import {{ render }} from "{env.VERSIONED_CDN_DIST_URL + "/widget.mjs"}";'
        )
    else:  # It's a local Path
        # Create a blob URL for the module
        content = read_file(env.WIDGET_PATH)

        return f"""
            const encodedContent = "{encode_string(content)}";
            const decodedContent = atob(encodedContent);
            const moduleBlob = new Blob([decodedContent], {{ type: 'text/javascript' }});
            const moduleUrl = URL.createObjectURL(moduleBlob);
            const {{ render }} = await import(moduleUrl);
            URL.revokeObjectURL(moduleUrl);
        """


def html_snippet(ast, id=None, dist_url=None):
    id = id or f"colight-widget-{uuid.uuid4().hex}"
    data, buffers = to_json_with_initialState(ast, buffers=[])

    colight_data = create_bytes(data, buffers)
    colight_base64 = base64.b64encode(colight_data).decode("utf-8")

    # Get the appropriate script URL
    script_url = (
        dist_url or env.VERSIONED_CDN_DIST_URL or env.UNVERSIONED_CDN_DIST_URL
    ) + "/widget.mjs"

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


def html_page(ast, id=None, dist_url=None):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Colight</title>
    </head>
    <body>
        {html_snippet(ast, id, dist_url=dist_url)}
    </body>
    </html>
    """
