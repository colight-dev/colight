import asyncio
import json
import threading
import webbrowser
from pathlib import Path

import websockets
from watchfiles import awatch
from werkzeug.middleware.shared_data import SharedDataMiddleware
from werkzeug.serving import make_server


class HtmlFallbackMiddleware:
    def __init__(self, app, roots):
        self.app = app
        self.roots = roots

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")

        # Check if it's a naked path (no extension)
        if not Path(path).suffix and not path.endswith("/"):
            for mount, root in self.roots.items():
                if path.startswith(mount):
                    relative_path = path[len(mount) :].lstrip("/")
                    html_file = root / f"{relative_path}.html"

                    if html_file.exists():
                        environ["PATH_INFO"] = f"{path}.html"
                        return self.app(environ, start_response)

        # Check for directory paths and serve index.html
        elif path.endswith("/") or path == "" or path == "/":
            # Normalize empty path to "/"
            if path == "":
                path = "/"

            for mount, root in self.roots.items():
                # Check if the path matches this mount point
                if (
                    path == mount
                    or path.startswith(mount + "/")
                    or (mount == "/" and path == "/")
                ):
                    if mount == "/":
                        relative_path = path[1:] if len(path) > 1 else ""
                    else:
                        relative_path = path[len(mount) :].lstrip("/")

                    index_file = (
                        root / relative_path / "index.html"
                        if relative_path
                        else root / "index.html"
                    )
                    if index_file.exists():
                        index_path = (
                            f"{path}index.html"
                            if path.endswith("/")
                            else f"{path}/index.html"
                        )
                        environ["PATH_INFO"] = index_path.replace("//", "/")
                        return self.app(environ, start_response)

        return self.app(environ, start_response)


class LiveReloadMiddleware:
    """Middleware to inject live reload script into HTML responses."""

    def __init__(self, app, ws_port, use_live_js=False):
        self.app = app
        self.ws_port = ws_port
        self.use_live_js = use_live_js

        if use_live_js:
            # For LiveServer, inject the enhanced live.js script
            self.reload_script = """<script src="/dist/live.js"></script>"""
        else:
            # For watch-serve, use the simple inline script
            self.reload_script = f"""<script>
(function() {{
    let reloadScheduled = false;
    
    function connect() {{
        const ws = new WebSocket('ws://127.0.0.1:{ws_port}');
        
        ws.onopen = function() {{
            console.log('LiveReload connected');
        }};
        
        ws.onmessage = function(event) {{
            const data = JSON.parse(event.data);
            if (data.type === 'reload' && !reloadScheduled) {{
                reloadScheduled = true;
                window.location.reload();
            }}
        }};
        
        ws.onerror = function(error) {{
            console.log('LiveReload connection error:', error);
        }};
        
        ws.onclose = function() {{
            console.log('LiveReload disconnected. Reconnecting in 2s...');
            setTimeout(connect, 2000);
        }};
    }}
    
    connect();
}})();
</script>
"""

    def __call__(self, environ, start_response):
        from io import BytesIO
        from typing import List, Tuple

        # Capture the response
        response_body = BytesIO()
        response_status = "200 OK"  # Default status
        response_headers: List[Tuple[str, str]] = []  # Default empty headers

        def capturing_start_response(status, headers, exc_info=None):
            nonlocal response_status, response_headers
            response_status = status
            response_headers = list(headers)  # Make a copy
            # Return a dummy write function
            return lambda data: response_body.write(data)

        # Call the app
        app_iter = self.app(environ, capturing_start_response)

        # Consume the iterator
        try:
            for data in app_iter:
                response_body.write(data)
        finally:
            if hasattr(app_iter, "close"):
                app_iter.close()

        # Get the complete response body
        body = response_body.getvalue()

        # Check if it's HTML
        is_html = False
        for name, value in response_headers:
            if name.lower() == "content-type" and "text/html" in value.lower():
                is_html = True
                break

        # If it's HTML, inject the script
        if is_html:
            if b"</body>" in body:
                body = body.replace(
                    b"</body>", self.reload_script.encode() + b"</body>"
                )
            elif b"</html>" in body:
                body = body.replace(
                    b"</html>", self.reload_script.encode() + b"</html>"
                )
            else:
                body += self.reload_script.encode()

            # Update Content-Length
            new_headers = []
            for name, value in response_headers:
                if name.lower() == "content-length":
                    new_headers.append((name, str(len(body))))
                else:
                    new_headers.append((name, value))

            # Add Content-Length if missing
            if not any(h[0].lower() == "content-length" for h in new_headers):
                new_headers.append(("Content-Length", str(len(body))))

            response_headers = new_headers

        # Send the response
        start_response(response_status, response_headers)
        return [body]


class LiveReloadServer:
    """
    A development server that provides live reloading functionality.

    This server watches for file changes in specified directories and
    notifies connected clients to reload. It consists of two main parts:
    1. An HTTP server to serve static files.
    2. A WebSocket server to send reload signals to the client.
    """

    def __init__(
        self,
        roots,
        host="127.0.0.1",
        http_port=5500,
        ws_port=5501,
        open_url_delay=True,
    ):
        self.roots = roots
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.open_url_delay = open_url_delay
        self.connections = set()
        self._http_server_thread = None
        self._stop_event = asyncio.Event()

    async def _websocket_handler(self, websocket):
        """Handles new WebSocket connections."""
        self.connections.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.connections.remove(websocket)

    async def _send_reload_signal(self):
        """Sends a 'reload' message to all connected clients."""
        if not self.connections:
            return
        message = json.dumps({"type": "reload"})
        await asyncio.gather(*(ws.send(message) for ws in self.connections))

    async def _watch_for_changes(self):
        """Watches for file changes and triggers a reload."""
        paths_to_watch = [str(root) for root in self.roots.values()]
        async for _ in awatch(*paths_to_watch, stop_event=self._stop_event):
            print("Changes detected, reloading...")
            await self._send_reload_signal()

    def _run_http_server(self):
        """Runs the Werkzeug HTTP server in a separate thread."""
        # The app serves static files from the specified roots.
        app = SharedDataMiddleware(
            # A simple 404 handler for files not found.
            lambda environ, start_response: (
                start_response("404 Not Found", [("Content-Type", "text/plain")]),
                [b"Not Found"],
            )[1],
            {mount: str(root) for mount, root in self.roots.items()},
        )
        app = HtmlFallbackMiddleware(app, self.roots)  # Add HTML fallback
        app = LiveReloadMiddleware(app, self.ws_port)  # Add live reload
        # The server runs indefinitely until the program is stopped.
        server = make_server(self.host, self.http_port, app)
        server.serve_forever()

    async def serve(self):
        """Starts the file watcher, WebSocket server, and HTTP server."""
        # Run the HTTP server in a background thread.
        self._http_server_thread = threading.Thread(
            target=self._run_http_server, daemon=True
        )
        self._http_server_thread.start()

        # Start the WebSocket server.
        ws_server = await websockets.serve(
            self._websocket_handler, self.host, self.ws_port
        )
        print(f"WebSocket server started at ws://{self.host}:{self.ws_port}")
        print(f"HTTP server started at http://{self.host}:{self.http_port}")

        if self.open_url_delay:
            url = f"http://{self.host}:{self.http_port}"
            # Open the URL in a new browser tab after a short delay.
            threading.Timer(1, lambda: webbrowser.open(url)).start()

        # Start watching for file changes.
        await self._watch_for_changes()

        # Clean up when the server is stopped.
        ws_server.close()
        await ws_server.wait_closed()

    def stop(self):
        """Stops the server."""
        self._stop_event.set()


def serve_live(
    roots, host="127.0.0.1", http_port=5500, ws_port=5501, open_url_delay=True
):
    """Convenience function to create and run a LiveReloadServer."""
    server = LiveReloadServer(
        roots=roots,
        host=host,
        http_port=http_port,
        ws_port=ws_port,
        open_url_delay=open_url_delay,
    )
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("Stopping server...")
        server.stop()


if __name__ == "__main__":
    # Example usage:
    # This will serve files from the 'dist' and 'test-artifacts/colight-prose'
    # directories and watch for changes within them.
    project_roots = {
        "/dist": Path("dist").resolve(),
        "/": Path("test-artifacts/colight-prose").resolve(),
    }
    serve_live(project_roots)
