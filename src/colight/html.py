import base64
import json
import uuid
from pathlib import Path
from typing import Union

from colight.util import read_file
from colight.widget import to_json_with_initialState
from colight.env import WIDGET_URL


# Binary data delimiter
BINARY_DELIMITER = b"\n---BINARY_DATA---\n"


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
    """CSS is now embedded in JS bundles, no separate CSS needed"""
    return ""


def html_snippet(ast, id=None):
    id = id or f"colight-widget-{uuid.uuid4().hex}"
    data, buffers = to_json_with_initialState(ast, buffers=[])

    # Get JS and CSS content
    js_content = get_script_content()
    css_content = get_style_content()

    # For embedded HTML, we have to use base64 since we can't use binary
    # This is only for HTML embedding; the .colight file format uses raw binary
    encoded_buffers = [base64.b64encode(buffer).decode("utf-8") for buffer in buffers]

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
        window.colight.renderData(container, data, {json.dumps(encoded_buffers)});
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


def export_colight(
    ast,
    output_path: Union[str, Path],
    create_example: bool = True,
    use_local_embed: bool = False,
) -> Union[str, tuple[str, str]]:
    """
    Export a visualization as a single .colight file that can be embedded in another site.

    The .colight file format consists of:
    1. A JSON header with visualization data and buffer layout information
    2. A delimiter (BINARY_DELIMITER)
    3. Concatenated binary buffers

    Args:
        ast: The visualization AST to export
        output_path: Path to write the .colight file to
        create_example: Whether to create an example HTML file showing how to embed the visualization
        use_local_embed: Whether to use the local embed.js file in the example (for testing)

    Returns:
        If create_example is False: Path to the created .colight file
        If create_example is True: Tuple of (colight_path, example_path)
    """
    output_path = Path(output_path)

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get data and buffers
    data, buffers = to_json_with_initialState(ast, buffers=[])

    # Calculate buffer offsets
    offsets = []
    current_offset = 0

    for buffer in buffers:
        offsets.append(current_offset)
        current_offset += len(buffer)

    # Add buffer layout to the data
    data["bufferLayout"] = {
        "offsets": offsets,
        "count": len(buffers),
        "totalSize": current_offset,
    }

    # Serialize the JSON header
    json_header = json.dumps(data).encode("utf-8")

    # Write the file
    with open(output_path, "wb") as f:
        # Write JSON header
        f.write(json_header)

        # Write delimiter
        f.write(BINARY_DELIMITER)

        # Write concatenated buffers
        for buffer in buffers:
            f.write(buffer)

    colight_path = str(output_path)

    # Create an example HTML file if requested
    if create_example:
        # Import here to avoid circular imports and keep example functionality separate
        try:
            from notebooks.embed_examples import create_embed_example

            example_path = create_embed_example(
                output_path, use_local_embed=use_local_embed
            )
            return colight_path, example_path
        except ImportError:
            print(
                "Warning: Could not import embed examples. Install with notebooks dependencies."
            )
            return colight_path
    else:
        return colight_path
