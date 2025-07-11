"""LiveServer: On-demand development server for colight-site."""

import asyncio
import fnmatch
import itertools
import json
import pathlib
import threading
import webbrowser
from typing import Any, List, Optional, Set

import websockets
from watchfiles import awatch
from werkzeug.middleware.shared_data import SharedDataMiddleware
from werkzeug.serving import make_server
from werkzeug.wrappers import Request, Response

from colight_site.build_helper import BuildHelper
from colight_site.builder import BuildConfig
from colight_site.file_resolver import FileResolver, find_files
from colight_site.utils import merge_ignore_patterns

from .incremental_executor import IncrementalExecutor
from .index_generator import build_file_tree_json

# DocumentCache removed - no longer needed with RunVersion architecture


class SpaMiddleware:
    """Middleware that serves the SPA for all non-API, non-static routes."""

    def __init__(self, app, spa_html: str):
        self.app = app
        self.spa_html = spa_html

    def __call__(self, environ, start_response):
        request = Request(environ)
        path = request.path

        # Let API and static files pass through
        if (
            path.startswith("/api/")
            or path.startswith("/dist/")
            or path.endswith(".js")
            or path.endswith(".css")
        ):
            return self.app(environ, start_response)

        # Serve SPA for all other routes
        response = Response(self.spa_html, mimetype="text/html")
        return response(environ, start_response)


class ApiMiddleware:
    """Middleware that handles API requests for the SPA."""

    def __init__(
        self,
        app,
        input_path: pathlib.Path,
        output_path: pathlib.Path,
        config: BuildConfig,
        include: List[str],
        ignore: Optional[List[str]] = None,
    ):
        self.app = app
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.include = include
        self.ignore = ignore
        self.visual_store = {}  # Store visual data by ID
        self.file_resolver = FileResolver(input_path, include, ignore)
        self.incremental_executor = IncrementalExecutor(verbose=config.verbose)

    def _get_files(self) -> List[str]:
        """Get list of all matching Python files."""
        return self.file_resolver.get_all_files()

    def __call__(self, environ, start_response):
        """Handle API requests."""
        request = Request(environ)
        path = request.path

        # Handle /api/files endpoint
        if path == "/api/files":
            files = self._get_files()

            response_data = json.dumps({"files": sorted(files)})
            response = Response(response_data, mimetype="application/json")
            return response(environ, start_response)

        # Handle /api/index endpoint
        if path == "/api/index":
            files = find_files(
                self.input_path, self.include, merge_ignore_patterns(self.ignore)
            )

            # Build the tree structure
            tree = build_file_tree_json(files, self.input_path)

            response_data = json.dumps(tree, indent=2)
            response = Response(response_data, mimetype="application/json")
            return response(environ, start_response)

        if path.startswith("/api/document/"):
            file_path = path[14:]  # Remove /api/document/

            # Find source file
            source_file = self.file_resolver.find_source_file(file_path + ".html")
            if source_file:
                try:
                    # Generate JSON directly from source
                    from .json_generator import JsonFormGenerator

                    generator = JsonFormGenerator(
                        config=self.config,
                        visual_store=self.visual_store,
                        incremental_executor=self.incremental_executor,
                    )
                    json_content = generator.generate_json(source_file, None)
                    doc = json.loads(json_content)

                    response = Response(
                        json.dumps(doc, indent=2), mimetype="application/json"
                    )
                    return response(environ, start_response)
                except Exception as e:
                    error_data = json.dumps({"error": str(e), "type": "build_error"})
                    response = Response(
                        error_data, status=500, mimetype="application/json"
                    )
                    return response(environ, start_response)

            # File not found
            response = Response(
                json.dumps({"error": "File not found", "type": "not_found"}),
                status=404,
                mimetype="application/json",
            )
            return response(environ, start_response)

        # Handle /api/visual/<visual_id> endpoint
        if path.startswith("/api/visual/"):
            visual_id = path[12:]  # Remove /api/visual/

            if visual_id in self.visual_store:
                visual_data = self.visual_store[visual_id]
                response = Response(
                    visual_data,
                    mimetype="application/octet-stream",
                    headers={
                        "Cache-Control": "public, max-age=31536000, immutable",  # Cache forever
                        "Content-Type": "application/x-colight",
                    },
                )
                return response(environ, start_response)
            else:
                response = Response(
                    json.dumps({"error": "Visual not found", "id": visual_id}),
                    status=404,
                    mimetype="application/json",
                )
                return response(environ, start_response)

        # Not an API request, pass through
        return self.app(environ, start_response)


class OnDemandMiddleware:
    """Middleware that builds .py files to .html on-demand."""

    def __init__(
        self,
        app,
        input_path: pathlib.Path,
        output_path: pathlib.Path,
        config: BuildConfig,
        include: List[str],
        ignore: Optional[List[str]] = None,
    ):
        self.app = app
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.include = include
        self.ignore = ignore
        self._ensure_output_dir()

        self.file_resolver = FileResolver(input_path, include, ignore)

    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        self.output_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, environ, start_response):
        """Handle requests, building files on-demand."""
        request = Request(environ)
        path = request.path

        # Debug logging
        if self.config.verbose:
            print(f"OnDemandMiddleware handling: {path}")

        # Only handle .html requests or paths without extensions
        if not (path.endswith(".html") or ("." not in pathlib.Path(path).name)):
            return self.app(environ, start_response)

        # Special handling for index.html - no longer auto-generated
        if path == "/index.html" or path == "index.html":
            # Index is now handled by the client-side outliner
            # Let it be served normally if it exists
            return self.app(environ, start_response)

        # Normalize path to always end with .html
        if not path.endswith(".html"):
            path = f"{path}.html"

        # Try to find source file using FileResolver
        # Remove leading slash for consistency
        clean_path = path.lstrip("/")
        source_file = self.file_resolver.find_source_file(clean_path)

        if self.config.verbose:
            print(f"Looking for source file for path: {path}")
            print(f"Found source file: {source_file}")

        if source_file:
            output_file = self.file_resolver.get_output_path(
                source_file, self.output_path
            )

            if self.config.verbose:
                print(f"Output file will be: {output_file}")

            error_result = BuildHelper.build_file_if_stale(
                source_file, output_file, self.config, error_format="html"
            )

            if error_result:
                # Build failed, return error
                response = Response(error_result, status=500, mimetype="text/html")
                return response(environ, start_response)

        # Continue with normal app handling (will serve from cache or 404)
        return self.app(environ, start_response)


class LiveServer:
    """On-demand development server with live reload."""

    def __init__(
        self,
        input_path: pathlib.Path,
        output_path: pathlib.Path,
        config: BuildConfig,
        include: List[str],
        ignore: Optional[List[str]] = None,
        host: str = "127.0.0.1",
        http_port: int = 5500,
        ws_port: int = 5501,
        open_url: bool = True,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.include = include
        self.ignore = ignore
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.open_url = open_url

        self.connections: Set[Any] = set()  # WebSocket connections
        self._http_server = None
        self._http_thread = None
        self._stop_event = asyncio.Event()
        self._run_counter = itertools.count(1)  # Monotonic run version counter
        self._current_run_task: Optional[asyncio.Task] = None  # Current execution task
        self._api_middleware: Optional[ApiMiddleware] = (
            None  # Reference to API middleware
        )

    def _get_spa_html(self):
        """Get the SPA HTML template."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LiveServer</title>
</head>
<body>
    <div id="root"></div>
    <script src="/dist/live.js"></script>
</body>
</html>"""

    def _create_app(self):
        """Create the WSGI application with all middleware."""
        # Set up roots - only serve dist directory for JS/CSS
        roots = {}

        # Find dist directory - check both current dir and project root
        possible_dist_dirs = [
            pathlib.Path("dist"),  # If run from root
            pathlib.Path(__file__).parent.parent.parent.parent.parent
            / "dist",  # If run from packages/colight-site
        ]

        dist_dir = None
        for path in possible_dist_dirs:
            resolved = path.resolve()
            if resolved.exists() and resolved.is_dir():
                dist_dir = resolved
                break

        if dist_dir:
            roots["/dist"] = str(dist_dir)
            print(f"Serving assets from {dist_dir}")
        else:
            print("WARNING: Could not find dist directory for assets")

        # Base app that serves static files
        app = SharedDataMiddleware(
            lambda environ, start_response: (
                start_response("404 Not Found", [("Content-Type", "text/plain")]),
                [b"Not Found"],
            )[1],
            roots,
        )

        # Add API middleware
        self._api_middleware = ApiMiddleware(
            app,
            self.input_path,
            self.output_path,
            self.config,
            self.include,
            self.ignore,
        )
        app = self._api_middleware

        # Add SPA middleware (serves the React app)
        app = SpaMiddleware(app, self._get_spa_html())

        # Don't add live reload script injection - the SPA handles WebSocket connection

        return app

    def _run_http_server(self):
        """Run the HTTP server in a separate thread."""
        try:
            app = self._create_app()
            self._http_server = make_server(self.host, self.http_port, app)
            print(f"HTTP server thread started on {self.host}:{self.http_port}")
            self._http_server.serve_forever()
        except Exception as e:
            print(f"ERROR in HTTP server thread: {e}")
            import traceback

            traceback.print_exc()

    async def _websocket_handler(self, websocket):
        """Handle WebSocket connections."""
        self.connections.add(websocket)
        try:
            # Listen for messages from the client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "request-load" and data.get("path"):
                        # Client is requesting to load a file
                        file_path = data["path"]
                        client_run = data.get(
                            "clientRun", 0
                        )  # Get client's current run version
                        # Find the actual source file
                        if self._api_middleware:
                            source_file = (
                                self._api_middleware.file_resolver.find_source_file(
                                    file_path + ".html"
                                )
                            )
                            if source_file:
                                # Trigger a build for this file with client run info
                                await self._send_reload_signal(source_file, client_run)
                except json.JSONDecodeError:
                    pass  # Ignore invalid messages
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connections.remove(websocket)

    async def _ws_broadcast(self, message: dict):
        """Broadcast a message to all connected WebSocket clients."""
        if not self.connections:
            return

        message_str = json.dumps(message)
        await asyncio.gather(
            *(ws.send(message_str) for ws in self.connections), return_exceptions=True
        )

    async def _trigger_build(
        self, file_path: pathlib.Path, client_run: Optional[int] = None
    ):
        """Trigger a build for a changed file."""
        run = next(self._run_counter)

        # If client_run is provided and less than current run, send full data
        force_full_data = client_run is not None and client_run < run

        # Convert to relative path without extension
        if self.input_path.is_file():
            html_path = self.input_path.stem
        else:
            try:
                # Make sure we have absolute paths for comparison
                abs_file = file_path if file_path.is_absolute() else file_path.resolve()
                abs_input = (
                    self.input_path
                    if self.input_path.is_absolute()
                    else self.input_path.resolve()
                )

                rel_path = abs_file.relative_to(abs_input)
                html_path = str(rel_path)
                # Remove .py extension
                if html_path.endswith(".py"):
                    html_path = html_path[:-3]
            except ValueError:
                # File is not relative to input path
                html_path = file_path.stem

        # Parse document to get block information
        from colight_site.model import TagSet
        from colight_site.parser import parse_colight_file

        source_file = None

        try:
            # Find source file
            if self._api_middleware:
                source_file = self._api_middleware.file_resolver.find_source_file(
                    html_path + ".html"
                )
                if not source_file:
                    await self._ws_broadcast(
                        {"run": run, "type": "run-start", "file": html_path}
                    )
                    await self._ws_broadcast(
                        {"run": run, "type": "run-end", "error": "File not found"}
                    )
                    return

                # Parse the document
                document = parse_colight_file(source_file)

                # Apply config pragma if any
                if self.config.pragma:
                    document.tags = document.tags | TagSet(
                        frozenset(self.config.pragma)
                    )

                # Generate file hash for stable block IDs (same as in JSON generator)
                import hashlib

                file_hash = hashlib.sha256(str(source_file).encode()).hexdigest()[:6]

                # Get block IDs in document order with proper format
                block_ids = []
                for i, block in enumerate(document.blocks):
                    unique_id = block.id if block.id != 0 else i
                    stable_id = f"{file_hash}-B{unique_id:05d}"
                    block_ids.append(stable_id)

                # Determine which blocks are dirty (will be re-executed)
                if self._api_middleware.incremental_executor:
                    dirty_blocks_raw = (
                        self._api_middleware.incremental_executor.get_dirty_blocks(
                            document
                        )
                    )
                    # Convert raw block IDs to stable IDs
                    dirty_blocks = []
                    for i, block in enumerate(document.blocks):
                        if str(block.id) in dirty_blocks_raw:
                            unique_id = block.id if block.id != 0 else i
                            stable_id = f"{file_hash}-B{unique_id:05d}"
                            dirty_blocks.append(stable_id)
                else:
                    # If no incremental executor, all blocks are dirty
                    dirty_blocks = block_ids

                # Send enhanced run-start message with block manifest
                await self._ws_broadcast(
                    {
                        "run": run,
                        "type": "run-start",
                        "file": html_path,
                        "blocks": block_ids,  # All blocks in document order
                        "dirty": dirty_blocks,  # Blocks that will be re-executed
                    }
                )
            else:
                # Fallback to simple run-start
                await self._ws_broadcast(
                    {"run": run, "type": "run-start", "file": html_path}
                )

            # Now continue with execution (we already have document from above)
            if self._api_middleware and "document" in locals() and source_file:
                from .json_generator import JsonFormGenerator

                generator = JsonFormGenerator(
                    config=self.config,
                    visual_store=self._api_middleware.visual_store,
                    incremental_executor=self._api_middleware.incremental_executor,
                )

                # Execute incrementally and stream results
                for block_id, result in generator.execute_incremental_with_results(
                    source_file
                ):
                    # Check if task was cancelled
                    task = asyncio.current_task()
                    if task and task.cancelled():
                        return

                    # Check if block result is unchanged
                    cache_hit = result.get("cache_hit", False)
                    content_changed = result.get("content_changed", False)
                    unchanged = cache_hit and not content_changed

                    # Always send block-result for dirty blocks, but optimize payload
                    # If client is behind (force_full_data), always send full data
                    if unchanged and not force_full_data:
                        # Send lightweight message for unchanged blocks
                        await self._ws_broadcast(
                            {
                                "run": run,
                                "type": "block-result",
                                "block": block_id,
                                "unchanged": True,
                                # Minimal fields - client keeps existing results
                            }
                        )
                    else:
                        # Send full message for changed blocks
                        await self._ws_broadcast(
                            {
                                "run": run,
                                "type": "block-result",
                                "block": block_id,
                                "ok": result.get("ok", True),
                                "stdout": result.get("stdout", ""),
                                "error": result.get("error"),
                                "showsVisual": result.get("showsVisual", False),
                                "elements": result.get("elements", []),
                                "cache_hit": cache_hit,
                                "content_changed": content_changed,
                            }
                        )

                # Send run-end message
                await self._ws_broadcast({"run": run, "type": "run-end"})
            else:
                # This shouldn't happen in normal operation
                await self._ws_broadcast(
                    {"run": run, "type": "run-end", "error": "Server not initialized"}
                )

        except asyncio.CancelledError:
            # Task was cancelled, clean up if needed
            await self._ws_broadcast({"run": run, "type": "run-end", "cancelled": True})
            raise
        except Exception as e:
            # Send error and run-end
            await self._ws_broadcast({"run": run, "type": "run-end", "error": str(e)})
            if self.config.verbose:
                import traceback

                traceback.print_exc()

    async def _send_reload_signal(self, changed_file=None, client_run=None):
        """Send reload signal to all connected clients."""
        # This method is now simplified - just trigger a build
        if changed_file:
            # Cancel any in-flight build
            if self._current_run_task and not self._current_run_task.done():
                self._current_run_task.cancel()
                # Wait a bit for cancellation to complete
                try:
                    await asyncio.wait_for(self._current_run_task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

            # Start new build task
            self._current_run_task = asyncio.create_task(
                self._trigger_build(changed_file, client_run)
            )
        else:
            # General reload (shouldn't happen in new architecture)
            await self._ws_broadcast({"type": "reload"})

    async def _watch_for_changes(self):
        """Watch for file changes."""
        paths_to_watch = [str(self.input_path)]

        async for changes in awatch(*paths_to_watch, stop_event=self._stop_event):
            changed_files = {pathlib.Path(path) for _, path in changes}

            # Filter for matching files
            matching_changes = {f for f in changed_files if self._matches_patterns(f)}

            if matching_changes:
                # Log changes
                for file_path in matching_changes:
                    print(f"File changed: {file_path}")
                    # Send reload signal for each changed file
                    await self._send_reload_signal(file_path)

    def _matches_patterns(self, file_path: pathlib.Path) -> bool:
        """Check if file matches include/ignore patterns."""
        file_str = str(file_path)

        # Get combined ignore patterns
        combined_ignore = merge_ignore_patterns(self.ignore)

        # First check ignore patterns - check all parts of the path
        for part in file_path.parts:
            for pattern in combined_ignore:
                if fnmatch.fnmatch(part, pattern):
                    return False

        # Also check the full path against ignore patterns
        for pattern in combined_ignore:
            if fnmatch.fnmatch(file_str, pattern) or fnmatch.fnmatch(
                file_path.name, pattern
            ):
                return False

        # Check include patterns
        matches_include = any(
            fnmatch.fnmatch(file_str, pattern)
            or fnmatch.fnmatch(file_path.name, pattern)
            for pattern in self.include
        )

        return matches_include

    async def serve(self):
        """Start the server."""
        # Start HTTP server in background thread
        self._http_thread = threading.Thread(target=self._run_http_server, daemon=True)
        self._http_thread.start()

        # Start WebSocket server
        ws_server = await websockets.serve(
            self._websocket_handler, self.host, self.ws_port
        )

        print(f"LiveServer running at http://{self.host}:{self.http_port}")
        print(f"WebSocket server at ws://{self.host}:{self.ws_port}")
        print("Building files on-demand...")

        # Open browser if requested
        if self.open_url:
            url = f"http://{self.host}:{self.http_port}"
            threading.Timer(1, lambda: webbrowser.open(url)).start()

        try:
            # Watch for changes
            await self._watch_for_changes()
        finally:
            # Cleanup
            ws_server.close()
            await ws_server.wait_closed()
            if self._http_server:
                self._http_server.shutdown()

    def stop(self):
        """Stop the server."""
        self._stop_event.set()
        if self._http_server:
            self._http_server.shutdown()
