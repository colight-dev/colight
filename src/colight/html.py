import base64
import json
import uuid
from colight.util import read_file
from colight.widget import to_json_with_initialState
from colight.env import WIDGET_URL, CSS_URL


def encode_string(s):
    return base64.b64encode(s.encode("utf-8")).decode("utf-8")


def encode_buffers(buffers):
    buffer_entries = [
        f"'{base64.b64encode(buffer).decode('utf-8')}'" for buffer in buffers
    ]
    return "[" + ",".join(buffer_entries) + "]"


def get_script_content():
    """Get the JS content either from CDN or local file"""
    if isinstance(WIDGET_URL, str):  # It's a CDN URL
        return f'import {{ renderData }} from "{WIDGET_URL}";'
    else:  # It's a local Path
        # Create a blob URL for the module
        content = read_file(WIDGET_URL)

        return f"""
            const encodedContent = "{encode_string(content)}";
            const decodedContent = atob(encodedContent);
            const moduleBlob = new Blob([decodedContent], {{ type: 'text/javascript' }});
            const moduleUrl = URL.createObjectURL(moduleBlob);
            const {{ renderData }} = await import(moduleUrl);
            URL.revokeObjectURL(moduleUrl);
        """


def get_style_content():
    """Get the CSS content either from CDN or local file"""
    if isinstance(CSS_URL, str):  # It's a CDN URL
        return f'@import "{CSS_URL}";'
    else:  # It's a local Path
        with open(CSS_URL, "r") as css_file:
            return css_file.read()


def html_snippet(ast, id=None):
    id = id or f"colight-widget-{uuid.uuid4().hex}"
    data, buffers = to_json_with_initialState(ast, buffers=[])

    # Get JS and CSS content
    js_content = get_script_content()
    css_content = get_style_content()

    html_content = f"""
    <style>{css_content}</style>
    <div class="bg-white p3" id="{id}"></div>

    <script type="application/json">
        {json.dumps(data)}
    </script>

    <script type="module">
        {js_content};

        const container = document.getElementById('{id}');
        const jsonString = container.nextElementSibling.textContent;
        let data;
        try {{
            data = JSON.parse(jsonString);
        }} catch (error) {{
            console.error('Failed to parse JSON:', error);
        }}
        window.colight.renderData(container, data, {encode_buffers(buffers)});
    </script>
    """

    return html_content


def html_page(ast, id=None):
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Colight Widget</title>
    </head>
    <body>
        {html_snippet(ast, id)}
    </body>
    </html>
    """
