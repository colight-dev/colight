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

from colight_site.file_resolver import FileResolver, find_files
from colight_site.model import TagSet
from colight_site.parser import parse_colight_file
from colight_site.pragma import parse_pragma_arg
from colight_site.utils import merge_ignore_patterns

from .incremental_executor import IncrementalExecutor
from .index_generator import build_file_tree_json
from .json_generator import JsonDocumentGenerator
from .client_registry import ClientRegistry
from .file_dependency_graph import FileDependencyGraph
from .cache_manager import CacheManager

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
        include: List[str],
        ignore: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.app = app
        self.input_path = input_path.resolve()  # Always use absolute path
        self.include = include
        self.ignore = ignore
        self.visual_store = {}  # Store visual data by ID
        self.file_resolver = FileResolver(self.input_path, include, ignore)
        # Create CacheManager with reasonable defaults
        self.cache_manager = CacheManager(max_size_mb=500, hot_threshold_seconds=300)
        self.incremental_executor = IncrementalExecutor(
            verbose=verbose, cache_manager=self.cache_manager
        )
        self.verbose = verbose

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
                    generator = JsonDocumentGenerator(
                        verbose=self.verbose,
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


class LiveServer:
    """On-demand development server with live reload."""

    def __init__(
        self,
        input_path: pathlib.Path,
        include: List[str],
        ignore: Optional[List[str]] = None,
        host: str = "127.0.0.1",
        http_port: int = 5500,
        ws_port: int = 5501,
        open_url: bool = True,
        verbose: bool = False,
        pragma: Optional[str | set] = set(),
    ):
        self.input_path = input_path.resolve()  # Always use absolute path
        self.verbose = verbose
        self.pragma = parse_pragma_arg(pragma)
        self.include = include
        self.ignore = ignore
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.open_url = open_url

        self.connections: Set[Any] = set()  # WebSocket connections
        self.client_registry = ClientRegistry()  # Track client file watches
        self.dependency_graph = FileDependencyGraph(
            self.input_path  # Use resolved path
        )  # Track file dependencies
        self._http_server = None
        self._http_thread = None
        self._stop_event = asyncio.Event()
        self._run_counter = itertools.count(1)  # Monotonic run version counter
        self._current_run_task: Optional[asyncio.Task] = None  # Current execution task
        self._api_middleware: Optional[ApiMiddleware] = (
            None  # Reference to API middleware
        )
        self._eviction_task: Optional[asyncio.Task] = None  # Periodic eviction task

        # File change notification state
        self._changed_files_buffer = set()
        self._notification_task = None
        self._notification_delay = 0.1  # 100ms throttle

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
            app, self.input_path, self.include, self.ignore, self.verbose
        )
        app = self._api_middleware

        # Add SPA middleware (serves the React app)
        app = SpaMiddleware(app, self._get_spa_html())

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
        client_id = None
        try:
            # Listen for messages from the client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")

                    # Handle watch-file message
                    if message_type == "watch-file":
                        client_id = data.get("clientId")
                        file_path = data.get("path")
                        if client_id and file_path:
                            # Register client if not already registered
                            if client_id not in self.client_registry.clients:
                                self.client_registry.register_client(
                                    client_id, websocket
                                )
                            # Watch the file
                            self.client_registry.watch_file(client_id, file_path)
                            # Log current status
                            self.client_registry.log_status()
                            # Unmark file from eviction if it was marked
                            if (
                                self._api_middleware
                                and self._api_middleware.incremental_executor
                            ):
                                self._api_middleware.incremental_executor.unmark_file_for_eviction(
                                    file_path
                                )

                    # Handle unwatch-file message
                    elif message_type == "unwatch-file":
                        client_id = data.get("clientId")
                        file_path = data.get("path")
                        if client_id and file_path:
                            self.client_registry.unwatch_file(client_id, file_path)
                            self.client_registry.log_status()
                            # Mark file for potential eviction if no one is watching it
                            if not self.client_registry.get_watchers(file_path):
                                if (
                                    self._api_middleware
                                    and self._api_middleware.incremental_executor
                                ):
                                    self._api_middleware.incremental_executor.mark_file_for_eviction(
                                        file_path
                                    )

                    # Handle existing request-load message
                    elif message_type == "request-load" and data.get("path"):
                        # Client is requesting to load a file
                        file_path = data["path"]
                        client_run = data.get(
                            "clientRun", 0
                        )  # Get client's current run version
                        # Find the actual source file
                        if self._api_middleware:
                            # Remove .py extension if present, then add .html for file resolver
                            html_path = file_path.removesuffix(".py") + ".html"
                            source_file = (
                                self._api_middleware.file_resolver.find_source_file(
                                    html_path
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
            # Unregister client if registered
            if client_id:
                self.client_registry.unregister_client(client_id)

    async def _ws_broadcast_to_file_watchers(self, message: dict, file_path: str):
        """Broadcast a message only to clients watching a specific file."""
        # Get clients watching this file
        watching_clients = self.client_registry.get_watchers(file_path)

        if not watching_clients:
            return

        message_str = json.dumps(message)
        # Send only to watching clients
        websockets_to_send = []
        for client_id in watching_clients:
            ws = self.client_registry.clients.get(client_id)
            if ws and ws in self.connections:
                websockets_to_send.append(ws)

        if websockets_to_send:
            await asyncio.gather(
                *(ws.send(message_str) for ws in websockets_to_send),
                return_exceptions=True,
            )

    async def _trigger_build(
        self, file_path: pathlib.Path, client_run: Optional[int] = None
    ):
        """Trigger a build for a changed file."""
        run = next(self._run_counter)

        # If client_run is provided and less than current run, send full data
        force_full_data = client_run is not None and client_run < run

        # Convert to relative path - keep .py extension for file_path
        if self.input_path.is_file():
            file_path_str = self.input_path.name  # Keep .py extension
            html_path = self.input_path.stem  # Remove .py for HTML path
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
                file_path_str = str(rel_path)  # Keep .py extension
                html_path = str(rel_path)
                # Remove .py extension for HTML path only
                if html_path.endswith(".py"):
                    html_path = html_path[:-3]
            except ValueError:
                # File is not relative to input path
                file_path_str = file_path.name  # Keep .py extension
                html_path = file_path.stem

        source_file = None

        try:
            # Find source file
            if self._api_middleware:
                source_file = self._api_middleware.file_resolver.find_source_file(
                    html_path + ".html"
                )
                if not source_file:
                    await self._ws_broadcast_to_file_watchers(
                        {"run": run, "type": "run-start", "file": file_path_str},
                        file_path_str,
                    )
                    await self._ws_broadcast_to_file_watchers(
                        {"run": run, "type": "run-end", "error": "File not found"},
                        file_path_str,
                    )
                    return

                # Parse the document
                document = parse_colight_file(source_file)

                # Apply config pragma if any
                if self.pragma:
                    document.tags = document.tags | TagSet(frozenset(self.pragma))

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
                await self._ws_broadcast_to_file_watchers(
                    {
                        "run": run,
                        "type": "run-start",
                        "file": file_path_str,
                        "blocks": block_ids,  # All blocks in document order
                        "dirty": dirty_blocks,  # Blocks that will be re-executed
                    },
                    file_path_str,
                )
            else:
                # Fallback to simple run-start
                await self._ws_broadcast_to_file_watchers(
                    {"run": run, "type": "run-start", "file": file_path_str},
                    file_path_str,
                )

            # Now continue with execution (we already have document from above)
            if self._api_middleware and "document" in locals() and source_file:
                from .json_generator import JsonDocumentGenerator

                generator = JsonDocumentGenerator(
                    verbose=self.verbose,
                    pragma=self.pragma,
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
                        await self._ws_broadcast_to_file_watchers(
                            {
                                "run": run,
                                "type": "block-result",
                                "block": block_id,
                                "unchanged": True,
                                # Minimal fields - client keeps existing results
                            },
                            file_path_str,
                        )
                    else:
                        # Send full message for changed blocks
                        await self._ws_broadcast_to_file_watchers(
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
                            },
                            file_path_str,
                        )

                # Send run-end message
                await self._ws_broadcast_to_file_watchers(
                    {"run": run, "type": "run-end"}, file_path_str
                )
            else:
                # This shouldn't happen in normal operation
                await self._ws_broadcast_to_file_watchers(
                    {"run": run, "type": "run-end", "error": "Server not initialized"},
                    file_path_str,
                )

        except asyncio.CancelledError:
            # Task was cancelled, clean up if needed
            await self._ws_broadcast_to_file_watchers(
                {"run": run, "type": "run-end", "cancelled": True}, file_path_str
            )
            raise
        except Exception as e:
            # Send error and run-end
            await self._ws_broadcast_to_file_watchers(
                {"run": run, "type": "run-end", "error": str(e)}, file_path_str
            )
            if self.verbose:
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
            # General reload - send to all clients watching any file
            # This is a fallback that shouldn't happen in normal operation
            for file_path in self.client_registry.get_watched_files():
                await self._ws_broadcast_to_file_watchers({"type": "reload"}, file_path)

    async def _send_file_change_notification(self):
        """Send notification about changed files after debounce period."""
        await asyncio.sleep(self._notification_delay)

        # Get the files that changed
        changed_files = list(self._changed_files_buffer)
        self._changed_files_buffer.clear()

        # Only notify if a single file changed
        if len(changed_files) == 1:
            file_path = changed_files[0]
            # Send to all connected clients
            message = {
                "type": "file-changed",
                "path": file_path,
                "watched": file_path in self.client_registry.get_watched_files(),
            }

            # Send to all connected clients (not just watchers)
            if self.connections:
                message_str = json.dumps(message)
                await asyncio.gather(
                    *(ws.send(message_str) for ws in self.connections),
                    return_exceptions=True,
                )

                if self.verbose:
                    print(f"Sent file-changed notification for {file_path}")

    async def _periodic_cache_eviction(self):
        """Periodically evict cache entries for unwatched files."""
        while not self._stop_event.is_set():
            try:
                # Wait 30 seconds between eviction runs
                await asyncio.sleep(30)

                if self._api_middleware and self._api_middleware.incremental_executor:
                    # Evict cache entries for unwatched files
                    self._api_middleware.incremental_executor.evict_unwatched_files(
                        force=False
                    )

                    # Log cache stats if verbose
                    if self.verbose:
                        stats = (
                            self._api_middleware.incremental_executor.get_cache_stats()
                        )
                        print(
                            f"Cache stats: {stats['total_entries']} entries, "
                            f"{stats['size_mb']:.1f}MB, "
                            f"hit rate: {stats['hit_rate']:.2%}"
                        )
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.verbose:
                    print(f"Error in cache eviction: {e}")

    async def _watch_for_changes(self):
        """Watch for file changes."""
        paths_to_watch = [str(self.input_path)]

        # Initial dependency graph build
        print("Building initial dependency graph...")
        self.dependency_graph.analyze_directory(self.input_path)
        graph_stats = self.dependency_graph.get_graph_stats()
        print(
            f"Dependency graph: {graph_stats['total_files']} files, {graph_stats['total_imports']} imports"
        )

        async for changes in awatch(*paths_to_watch, stop_event=self._stop_event):
            changed_files = {pathlib.Path(path) for _, path in changes}

            # Filter for matching files
            matching_changes = {f for f in changed_files if self._matches_patterns(f)}

            if matching_changes:
                # Update dependency graph for changed files
                for file_path in matching_changes:
                    if file_path.suffix == ".py":
                        self.dependency_graph.analyze_file(file_path)

                # Buffer changed files for notification
                for file_path in matching_changes:
                    try:
                        relative_path = str(file_path.relative_to(self.input_path))
                        self._changed_files_buffer.add(relative_path)
                    except ValueError:
                        # File is outside input_path, skip it
                        if self.verbose:
                            print(f"Ignoring file outside project: {file_path}")
                        continue

                # Cancel any pending notification task and start a new one
                if self._notification_task and not self._notification_task.done():
                    self._notification_task.cancel()
                self._notification_task = asyncio.create_task(
                    self._send_file_change_notification()
                )

                # Track which files need execution
                files_to_execute = set()
                watched_files = self.client_registry.get_watched_files()

                # Find affected files and filter by what's being watched
                for file_path in matching_changes:
                    try:
                        relative_path = str(file_path.relative_to(self.input_path))
                    except ValueError:
                        # File is outside input_path, skip it
                        if self.verbose:
                            print(f"Ignoring file outside project: {file_path}")
                        continue
                    affected = self.dependency_graph.get_affected_files(relative_path)

                    # Only execute files that are watched (or affect watched files)
                    watched_affected = [f for f in affected if f in watched_files]

                    if watched_affected:
                        # Log with dependency info
                        if len(affected) > 1:
                            print(
                                f"File changed: {file_path} (affects {len(affected)} files, {len(watched_affected)} watched)"
                            )
                            print(f"  Executing for: {', '.join(watched_affected)}")
                        else:
                            print(f"File changed: {file_path} (watched)")

                        # Add watched affected files to execution set
                        for watched_file in watched_affected:
                            # Convert back to Path for execution
                            watched_path = self.input_path / watched_file
                            if watched_path.exists():
                                files_to_execute.add(watched_path)
                    else:
                        # File changed but nothing watched is affected
                        print(
                            f"File changed: {file_path} (no watched files affected, skipping)"
                        )

                        # Mark cache entries for eviction if file not watched
                        if (
                            self._api_middleware
                            and self._api_middleware.incremental_executor
                        ):
                            self._api_middleware.incremental_executor.mark_file_for_eviction(
                                relative_path
                            )

                # Execute only the watched files that are affected
                for file_path in files_to_execute:
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

        # Start periodic cache eviction task
        self._eviction_task = asyncio.create_task(self._periodic_cache_eviction())

        try:
            # Watch for changes
            await self._watch_for_changes()
        finally:
            # Cleanup
            if self._eviction_task and not self._eviction_task.done():
                self._eviction_task.cancel()
                try:
                    await self._eviction_task
                except asyncio.CancelledError:
                    pass

            ws_server.close()
            await ws_server.wait_closed()
            if self._http_server:
                self._http_server.shutdown()

    def stop(self):
        """Stop the server."""
        self._stop_event.set()
        if self._http_server:
            self._http_server.shutdown()
