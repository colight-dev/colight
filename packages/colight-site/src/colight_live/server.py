"""LiveServer: On-demand development server for colight-site."""

import asyncio
import fnmatch
import json
import pathlib
import threading
import time
import webbrowser
from typing import Any, Dict, List, Optional, Set, Tuple

import websockets
from watchfiles import awatch
from werkzeug.middleware.shared_data import SharedDataMiddleware
from werkzeug.serving import make_server
from werkzeug.wrappers import Request, Response

from colight_site.build_helper import BuildHelper
from colight_site.builder import BuildConfig
from colight_site.constants import DEFAULT_IGNORE_PATTERNS
from colight_site.file_resolver import FileResolver, find_files

from .incremental_executor import IncrementalExecutor
from .index_generator import build_file_tree_json


class DocumentCache:
    """Cache for document JSON data with version tracking for incremental updates."""

    def __init__(self):
        self.cache: Dict[str, Tuple[dict, int]] = {}  # path -> (document, version)
        self.versions: Dict[str, int] = {}  # path -> current version

    def get_changes(self, path: str, new_doc: dict) -> dict:
        """Compare cached document with new document and return changes."""
        old_doc, old_version = self.cache.get(path, ({}, 0))
        new_version = old_version + 1

        old_blocks = {b["id"]: b for b in old_doc.get("blocks", [])}
        new_blocks = {b["id"]: b for b in new_doc.get("blocks", [])}

        changes = {
            "modified": [],
            "removed": [],
            "moved": [],
            "total": len(new_blocks),
            "version": new_version,
            "full": False,
        }

        # Track ordinals for move detection
        old_ordinals = {
            b["id"]: b.get("ordinal", i)
            for i, b in enumerate(old_doc.get("blocks", []))
        }
        new_ordinals = {
            b["id"]: b.get("ordinal", i)
            for i, b in enumerate(new_doc.get("blocks", []))
        }

        # Detect changes
        for bid, block in new_blocks.items():
            if bid in old_blocks:
                old_block = old_blocks[bid]
                # Content changed
                if old_block.get("content_hash") != block.get("content_hash"):
                    changes["modified"].append(block)
                # Position changed (moved)
                elif old_ordinals.get(bid) != new_ordinals.get(bid):
                    changes["moved"].append(
                        {
                            "block": block,
                            "old_ordinal": old_ordinals.get(bid),
                            "new_ordinal": new_ordinals.get(bid),
                        }
                    )
            else:
                # New block
                changes["modified"].append(block)

        # Track removed blocks (with full data for animations)
        for bid, block in old_blocks.items():
            if bid not in new_blocks:
                changes["removed"].append(block)

        # Decide if full reload is better
        if self._should_full_reload(new_doc, changes):
            changes["full"] = True

        # Update cache
        self.cache[path] = (new_doc, new_version)
        self.versions[path] = new_version

        return changes

    def _should_full_reload(self, doc: dict, changes: dict) -> bool:
        """Determine if a full reload is safer/better."""
        total_blocks = changes["total"]

        # No blocks at all
        if total_blocks == 0:
            return False

        # Parse error in document
        if doc.get("error"):
            return True

        # More than 30% of blocks changed
        modified_count = len(changes["modified"]) + len(changes["removed"])
        if total_blocks > 0 and modified_count > total_blocks * 0.3:
            return True

        # TODO: Add detection for large contiguous changes

        return False

    def clear(self, path: Optional[str] = None):
        """Clear cache for a specific path or all paths."""
        if path:
            self.cache.pop(path, None)
            self.versions.pop(path, None)
        else:
            self.cache.clear()
            self.versions.clear()


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
        document_cache: Optional[DocumentCache] = None,
    ):
        self.app = app
        self.input_path = input_path
        self.output_path = output_path
        self.config = config
        self.include = include
        self.ignore = ignore
        self.document_cache = document_cache or DocumentCache()
        self.visual_store = {}  # Store visual data by ID
        self.file_resolver = FileResolver(input_path, include, ignore)
        self.incremental_executor = IncrementalExecutor(verbose=config.verbose)

    def _get_files(self) -> List[str]:
        """Get list of all matching Python files."""
        return self.file_resolver.get_all_files()

    def _get_combined_ignore_patterns(self) -> List[str]:
        """Get combined default and user ignore patterns."""

        combined_ignore = list(self.ignore) if self.ignore else []
        combined_ignore.extend(DEFAULT_IGNORE_PATTERNS)
        return combined_ignore

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
            # Get all colight files
            combined_ignore = self._get_combined_ignore_patterns()
            files = find_files(self.input_path, self.include, combined_ignore)

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

                    # Get previous document to detect changed blocks
                    old_doc = self.document_cache.cache.get(file_path, (None, None))[0]
                    changed_blocks = None

                    if old_doc:
                        # Detect which blocks have changed based on content_hash
                        changed_blocks = set()
                        old_blocks = {b["id"]: b for b in old_doc.get("blocks", [])}

                        # We'll need to parse to get current block IDs and hashes
                        # For now, let the incremental executor figure it out
                        changed_blocks = None

                    generator = JsonFormGenerator(
                        config=self.config,
                        visual_store=self.visual_store,
                        incremental_executor=self.incremental_executor,
                    )
                    json_content = generator.generate_json(source_file, changed_blocks)
                    doc = json.loads(json_content)

                    # Get changes from cache
                    changes = self.document_cache.get_changes(file_path, doc)

                    # Add changes info to response
                    doc["_changes"] = {
                        "version": changes["version"],
                        "full": changes["full"],
                        "modified": [b["id"] for b in changes["modified"]]
                        if not changes["full"]
                        else None,
                        "removed": [b["id"] for b in changes["removed"]]
                        if not changes["full"]
                        else None,
                    }

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
        self._document_cache = DocumentCache()  # Cache for incremental updates

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

        # Add API middleware with document cache
        app = ApiMiddleware(
            app,
            self.input_path,
            self.output_path,
            self.config,
            self.include,
            self.ignore,
            self._document_cache,
        )

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
            await websocket.wait_closed()
        finally:
            self.connections.remove(websocket)

    async def _send_reload_signal(self, changed_file=None):
        """Send reload signal to all connected clients."""
        if not self.connections:
            return

        # Send file-specific change notification
        if changed_file:
            # Convert to relative path without extension
            if self.input_path.is_file():
                html_path = self.input_path.stem
            else:
                try:
                    # Make sure we have absolute paths for comparison
                    abs_file = (
                        changed_file
                        if changed_file.is_absolute()
                        else changed_file.resolve()
                    )
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
                    # Remove .colight.py extension
                    elif html_path.endswith(".colight.py"):
                        html_path = html_path[:-11]
                except ValueError:
                    # File is not relative to input path
                    html_path = changed_file.stem

            # For now, just send file-changed notification
            # The client will fetch the document and check _changes field
            message = json.dumps(
                {"type": "file-changed", "path": html_path, "timestamp": time.time()}
            )
        else:
            # General reload
            message = json.dumps({"type": "reload"})

        # Send to all connections, ignoring failures
        await asyncio.gather(
            *(ws.send(message) for ws in self.connections), return_exceptions=True
        )

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
        combined_ignore = self._get_combined_ignore_patterns()

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

    def _get_combined_ignore_patterns(self) -> List[str]:
        """Get combined default and user ignore patterns."""

        combined_ignore = list(self.ignore) if self.ignore else []
        combined_ignore.extend(DEFAULT_IGNORE_PATTERNS)
        return combined_ignore

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
