import http.server
import socketserver
import threading
import os
import mimetypes
from typing import Dict, Union, Any


class ColightRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    Custom request handler for Colight.
    Serves files from multiple sources in a specific order:
    1. Dynamically added files/buffers (self.server_instance.served_files).
       - Buffers with a specific prefix are served once and then removed.
    2. Static files from a configured directory (self.server_instance.static_dir).
    3. (Optional) Files from the current working directory (if self.server_instance.serve_cwd is True).
    """

    def __init__(
        self, request, client_address, server, server_instance: "ColightHTTPServer"
    ):
        self.server_instance = server_instance
        if self.server_instance.static_dir and not self.server_instance.serve_cwd:
            # If serving from static_dir and not CWD, pass static_dir to super
            super().__init__(
                request,
                client_address,
                server,
                directory=self.server_instance.static_dir,
            )
        else:
            # If serving CWD or static_dir is None, default SimpleHTTPRequestHandler behavior (serves from CWD)
            super().__init__(request, client_address, server)

    def log_message(self, format: str, *args: Any) -> None:
        if self.server_instance.debug:
            # Use system default logging if debug is on
            sys_format = f"[{self.__class__.__name__}/{self.server_instance.actual_port}] {format}"
            http.server.SimpleHTTPRequestHandler.log_message(self, sys_format, *args)
        # Suppress logging if debug is off

    def guess_type(self, path: os.PathLike | str) -> str:
        """Guess the type of a file based on its extension, with additions."""
        # Prioritize mimetypes.guess_type for broader coverage
        ctype, encoding = mimetypes.guess_type(path)
        if ctype:
            return ctype

        # Fallback for some common types not always in mimetypes db or for overrides
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".js":
            return "application/javascript"
        if ext == ".wasm":
            return "application/wasm"
        return "application/octet-stream"  # Default fallback

    def do_GET(self) -> None:
        """Handles GET requests by trying different serving strategies."""
        path = self.path.lstrip("/")
        print(f"GET: {path}")
        if not path:  # If root path, try to serve index.html
            path = "index.html"

        # 1. Check for dynamically served buffers/files
        if path in self.server_instance.served_files:
            content = self.server_instance.served_files[path]
            is_buffer_to_pop = path.startswith(self.server_instance.buffer_prefix)

            if is_buffer_to_pop:
                content_bytes = self.server_instance.served_files.pop(path)
                if isinstance(content_bytes, str):
                    content_bytes = content_bytes.encode()
            else:
                content_bytes = (
                    content.encode() if isinstance(content, str) else content
                )

            self.send_response(200)
            self.send_header("Content-type", self.guess_type(path))
            self.send_header("Content-Length", str(len(content_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")  # CORS
            self.end_headers()
            self.wfile.write(content_bytes)
            return

        # 2. Check for static files from self.server_instance.static_dir
        #    SimpleHTTPRequestHandler handles this if `directory` was passed to its __init__
        #    and self.server_instance.serve_cwd is False.
        if self.server_instance.static_dir and not self.server_instance.serve_cwd:
            return super().do_GET()

        # 3. Fallback to SimpleHTTPRequestHandler's default CWD serving if serve_cwd is True
        if self.server_instance.serve_cwd:
            return super().do_GET()

        self.send_error(404, "File not found")


class ColightHTTPServer:
    """
    A general-purpose HTTP server for Colight.
    Can serve dynamic files, static files from a directory, and optionally from CWD.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 0,
        debug: bool = False,
        static_dir: Union[str, None] = None,
        serve_cwd: bool = False,
        buffer_prefix: str = "served_buffer_",
    ):
        self.host = host
        self.requested_port = port  # Store the requested port
        self.actual_port = port  # Will be updated after server starts if port was 0
        self.debug = debug
        self.static_dir = os.path.abspath(static_dir) if static_dir else None
        self.serve_cwd = serve_cwd
        self.buffer_prefix = buffer_prefix

        self.served_files: Dict[str, Union[str, bytes]] = {}
        self.httpd: Union[socketserver.TCPServer, None] = None
        self.server_thread: Union[threading.Thread, None] = None

        if not mimetypes.inited:
            mimetypes.init()  # Ensure mimetypes db is loaded

    def add_served_file(self, path: str, content: Union[str, bytes]) -> None:
        """Adds a file or buffer to be served dynamically. Replaces if path exists."""
        self.served_files[path] = content
        if self.debug:
            print(f"[ColightServer] Added/Updated served file: {path}")

    def remove_served_file(self, path: str) -> None:
        """Removes a file or buffer from the dynamic registry if it exists."""
        if path in self.served_files:
            del self.served_files[path]
            if self.debug:
                print(f"[ColightServer] Removed served file: {path}")

    def get_url(self, path: str = "") -> str:
        """Returns the full HTTP URL for a given path on this server."""
        return f"http://{self.host}:{self.actual_port}/{path.lstrip('/')}"

    def start(self) -> None:
        if self.server_thread and self.server_thread.is_alive():
            if self.debug:
                print(f"[ColightServer] Server already running on {self.get_url()}")
            return

        def handler_class(request, client_address, server):
            return ColightRequestHandler(
                request, client_address, server, server_instance=self
            )

        # socketserver.TCPServer.allow_reuse_address = True # Useful for quick restarts
        self.httpd = socketserver.TCPServer(
            (self.host, self.requested_port), handler_class
        )
        self.httpd.allow_reuse_address = True  # Set before server_bind

        self.actual_port = self.httpd.server_address[
            1
        ]  # Get the actual port, especially if requested_port was 0

        if self.debug:
            print(
                f"[ColightServer] Starting HTTP server on {self.host}:{self.actual_port}"
            )
            if self.static_dir:
                print(
                    f"[ColightServer] Serving static files from (if not overridden by dynamic): {self.static_dir}"
                )
            if self.serve_cwd:
                print(
                    "[ColightServer] CWD serving fallback enabled (for paths not in dynamic/static)."
                )
            if self.served_files:
                print(
                    f"[ColightServer] Pre-registered served files: {list(self.served_files.keys())}"
                )

        self.server_thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.1}
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        if self.debug:
            print(f"[ColightServer] Server started. Access at {self.get_url()}")

    def stop(self) -> None:
        if self.httpd:
            if self.debug:
                print(
                    f"[ColightServer] Shutting down HTTP server on port {self.actual_port}"
                )
            self.httpd.shutdown()  # Stop serve_forever loop
            if self.server_thread:
                self.server_thread.join(timeout=1.0)  # Wait for thread to finish
                if self.server_thread.is_alive() and self.debug:
                    print("[ColightServer] Server thread did not join cleanly.")
            self.httpd.server_close()  # Release the port
            self.httpd = None
            self.server_thread = None
            if self.debug:
                print("[ColightServer] Server stopped.")
        elif self.debug:
            print(
                "[ColightServer] Stop called but server not running or already stopped."
            )

    def __enter__(self) -> "ColightHTTPServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


if __name__ == "__main__":
    # Example Usage:
    # 1. Server for dynamic content and CWD (like chrome_devtools)
    print("Starting server for dynamic content and CWD (port 8000)...")
    server1_path = os.path.join(os.getcwd(), "test_server1_file.html")
    with open(server1_path, "w") as f:
        f.write("<h1>Hello from CWD via Server 1!</h1>")

    with ColightHTTPServer(port=8000, debug=True, serve_cwd=True) as server1:
        server1.add_served_file("dynamic.html", "<h1>Hello from Dynamic File!</h1>")
        server1.add_served_file(
            server1.buffer_prefix + "mybuffer.txt", b"This is a one-time buffer."
        )

        print(
            f"Server 1 URLs: \n  Dynamic: {server1.get_url('dynamic.html')}\n  Buffer: {server1.get_url(server1.buffer_prefix + 'mybuffer.txt')}\n  CWD: {server1.get_url('test_server1_file.html')}"
        )
        print(
            "Try accessing the URLs. Press Ctrl+C to stop server1 (if not on Windows, may need to kill process)."
        )
        try:
            while True:
                threading.Event().wait(1)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt server1...")
        finally:
            os.remove(server1_path)

    # 2. Server for static directory (like sidecar app shell)
    print("\nStarting server for static directory (port 8001)...")
    static_example_dir = "temp_static_site"
    static_index_path = os.path.join(static_example_dir, "index.html")
    static_asset_path = os.path.join(static_example_dir, "app.js")

    os.makedirs(static_example_dir, exist_ok=True)
    with open(static_index_path, "w") as f:
        f.write("<h1>Hello from Static Site!</h1><script src='app.js'></script>")
    with open(static_asset_path, "w") as f:
        f.write("console.log('Static app.js loaded!');")

    with ColightHTTPServer(
        port=8001, debug=True, static_dir=static_example_dir
    ) as server2:
        print(
            f"Server 2 URLs: \n  Static Index: {server2.get_url('index.html')} or {server2.get_url('')}\n  Static JS: {server2.get_url('app.js')}"
        )
        print("Try accessing the URLs. Press Ctrl+C to stop server2.")
        try:
            while True:
                threading.Event().wait(1)  # Keep alive for testing
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt server2...")
        finally:
            os.remove(static_index_path)
            os.remove(static_asset_path)
            os.rmdir(static_example_dir)

    print("\nExample servers finished.")
