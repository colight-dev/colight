"""LiveServer: On-demand development server for colight-prose."""

import asyncio
import fnmatch
import itertools
import json
import logging
import pathlib
import sys
import threading
import webbrowser
from typing import Any, List, Optional, Set

import websockets
from colight.env import DIST_LOCAL_PATH
from watchfiles import awatch
from werkzeug.middleware.shared_data import SharedDataMiddleware
from werkzeug.serving import make_server
from werkzeug.wrappers import Request, Response

from .client_registry import ClientRegistry
from .file_graph import FileDependencyGraph
from .file_resolver import FileResolver, find_files
from .incremental_executor import IncrementalExecutor
from .json_api import JsonDocumentGenerator, build_file_tree_json
from .model import TagSet
from .parser import parse_file
from .pragma import parse_pragma_arg
from .utils import merge_ignore_patterns

logger = logging.getLogger(__name__)

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
        self.incremental_executor = IncrementalExecutor(verbose=verbose)
        self.incremental_executor.project_root = str(self.input_path)
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(level=logging.INFO)

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
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
        self.pragma = parse_pragma_arg(pragma)
        self.include = include
        self.ignore = ignore
        self.host = host
        self.http_port = http_port
        self.ws_port = ws_port
        self.open_url = open_url

        self.connections: Set[Any] = set()  # WebSocket connections
        self.client_registry = (
            ClientRegistry()
        )  # Track client file watches (absolute paths)
        self.dependency_graph = FileDependencyGraph(
            self.input_path
        )  # Track file dependencies (absolute paths)
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

        roots["/dist"] = str(DIST_LOCAL_PATH)

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
                    print("Received message", data)

                    # Handle watch-file message
                    if message_type == "watch-file":
                        client_id = data.get("clientId")
                        rel_path = data.get("path")
                        if client_id and rel_path:
                            abs_path = (self.input_path / rel_path).resolve()
                            # Register client if not already registered
                            if client_id not in self.client_registry.clients:
                                self.client_registry.register_client(
                                    client_id, websocket
                                )
                            # Watch the file (absolute path)
                            self.client_registry.watch_file(client_id, str(abs_path))
                            self.client_registry.log_status()
                            if self.verbose:
                                print(
                                    f"DEBUG: Client {client_id} watching path: '{abs_path}'"
                                )
                            # Unmark file from eviction if it was marked
                            if (
                                self._api_middleware
                                and self._api_middleware.incremental_executor
                            ):
                                self._api_middleware.incremental_executor.unmark_file_for_eviction(
                                    str(abs_path)
                                )

                    # Handle unwatch-file message
                    elif message_type == "unwatch-file":
                        client_id = data.get("clientId")
                        rel_path = data.get("path")
                        if client_id and rel_path:
                            abs_path = (self.input_path / rel_path).resolve()
                            self.client_registry.unwatch_file(client_id, str(abs_path))
                            self.client_registry.log_status()
                            # Mark file for potential eviction if no one is watching it
                            if not self.client_registry.get_watchers(str(abs_path)):
                                if (
                                    self._api_middleware
                                    and self._api_middleware.incremental_executor
                                ):
                                    # Use the absolute Python file path directly
                                    self._api_middleware.incremental_executor.mark_file_for_eviction(
                                        str(abs_path)
                                    )

                    # Handle existing request-load message
                    elif message_type == "request-load" and data.get("path"):
                        rel_path = data["path"]
                        client_run = data.get("clientRun", 0)
                        if self._api_middleware:
                            # Convert relative path to absolute Python file path
                            abs_path = (self.input_path / rel_path).resolve()
                            if (
                                abs_path.exists()
                                and self._api_middleware.file_resolver.matches_patterns(
                                    abs_path
                                )
                            ):
                                await self._send_reload_signal(abs_path, client_run)
                except json.JSONDecodeError:
                    pass  # Ignore invalid messages
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.connections.remove(websocket)
            if client_id:
                self.client_registry.unregister_client(client_id)

    async def _ws_broadcast_to_file_watchers(self, message: dict, file_path: str):
        """Broadcast a message only to clients watching a specific file."""
        # file_path is relative; convert to absolute for registry lookup
        abs_path = (self.input_path / file_path).resolve()
        watching_clients = self.client_registry.get_watchers(str(abs_path))

        if not watching_clients:
            return

        # Convert any absolute paths in outgoing message to relative (for client)
        if "file" in message:
            try:
                abs_file = (self.input_path / message["file"]).resolve()
                message["file"] = str(abs_file.relative_to(self.input_path))
            except Exception:
                pass

        message_str = json.dumps(message)
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

    async def _ws_broadcast_all(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.connections:
            return

        message_str = json.dumps(message)
        await asyncio.gather(
            *(ws.send(message_str) for ws in self.connections),
            return_exceptions=True,
        )

    async def _process_multiple_files(self, files_to_execute: Set[pathlib.Path]):
        """Process multiple files that need execution."""
        # Cancel any in-flight build
        if self._current_run_task and not self._current_run_task.done():
            self._current_run_task.cancel()
            try:
                await asyncio.wait_for(self._current_run_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Separate watched and unwatched files
        watched_files = self.client_registry.get_watched_files()
        watched_to_execute = []
        unwatched_to_execute = []

        for file_path in files_to_execute:
            if str(file_path) in watched_files:
                watched_to_execute.append(file_path)
            else:
                unwatched_to_execute.append(file_path)

        # Clear Python's import cache for changed files
        for file_path in files_to_execute:
            # Remove from sys.modules to force fresh import
            abs_path = str(file_path.resolve())
            modules_to_remove = []
            for module_name, module in sys.modules.items():
                if module and hasattr(module, "__file__") and module.__file__:
                    if (
                        module.__file__ == abs_path
                        or module.__file__ == abs_path.replace(".py", ".pyc")
                    ):
                        modules_to_remove.append(module_name)

            for module_name in modules_to_remove:
                print(f"  Removing {module_name} from sys.modules")
                del sys.modules[module_name]

        # Then, fully execute watched files with visual tracking
        if watched_to_execute:
            # Trigger normal builds for watched files
            for file_path in watched_to_execute:
                await self._send_reload_signal(file_path)

    async def _trigger_build(
        self, file_path: pathlib.Path, client_run: Optional[int] = None
    ):
        """Trigger a build for a changed file."""
        run = next(self._run_counter)

        # If client_run is provided and less than current run, send full data
        force_full_data = client_run is not None and client_run < run

        source_file = file_path if file_path.is_absolute() else file_path.resolve()

        # Convert to relative path for client messages
        if self.input_path.is_file():
            file_path_str = self.input_path.name
        else:
            try:
                abs_file = source_file
                abs_input = (
                    self.input_path
                    if self.input_path.is_absolute()
                    else self.input_path.resolve()
                )
                rel_path = abs_file.relative_to(abs_input)
                file_path_str = str(rel_path)
            except ValueError:
                # File is not relative to input path
                file_path_str = source_file.name

        try:
            # Verify the file exists and matches patterns
            if self._api_middleware:
                if (
                    not source_file.exists()
                    or not self._api_middleware.file_resolver.matches_patterns(
                        source_file
                    )
                ):
                    # TODO, do not use run-start here, use an error message that shows itself..
                    await self._ws_broadcast_to_file_watchers(
                        {"run": run, "type": "run-start", "file": file_path_str},
                        file_path_str,
                    )
                    await self._ws_broadcast_to_file_watchers(
                        {"run": run, "type": "run-end", "error": "File not found"},
                        file_path_str,
                    )
                    return
                doc = parse_file(source_file, project_root=self.input_path)

                # Apply pragma if any
                if self.pragma:
                    doc.tags = doc.tags | TagSet(frozenset(self.pragma))

                # Get block IDs from analyzed document
                all_block_ids = doc.get_cache_keys()

                # Clean up cache entries for blocks that no longer exist
                if self._api_middleware and self._api_middleware.incremental_executor:
                    # IMPORTANT: Must use absolute path to match what incremental_executor stores
                    # source_file is the absolute path that will be used as current_file
                    if self.verbose:
                        logger.info(
                            f"Cleaning stale entries for: source_file={source_file}, file_path_str={file_path_str}"
                        )
                    self._api_middleware.incremental_executor.clean_stale_entries(
                        str(source_file), set(all_block_ids)
                    )

                # Send run-start message with block manifest
                # Note: We don't pre-compute dirty blocks anymore - cache hit/miss during
                # execution will determine what actually runs
                await self._ws_broadcast_to_file_watchers(
                    {
                        "run": run,
                        "type": "run-start",
                        "file": file_path_str,
                        "block_ids": all_block_ids,  # All block IDs (cache keys) in document order
                    },
                    file_path_str,
                )

                # Execute incrementally and stream results
                from .json_api import JsonDocumentGenerator

                generator = JsonDocumentGenerator(
                    verbose=self.verbose,
                    pragma=self.pragma,
                    visual_store=self._api_middleware.visual_store,
                    incremental_executor=self._api_middleware.incremental_executor,
                )

                # Pass the analyzed document
                for block_id, result in generator.execute_incremental_with_results(
                    source_file, doc
                ):
                    # Check if task was cancelled
                    task = asyncio.current_task()
                    if task and task.cancelled():
                        return

                    # Check if block result is unchanged
                    cache_hit = result.get("cache_hit", False)
                    content_changed = result.get("content_changed", False)
                    unchanged = cache_hit and not content_changed

                    # Optimize payload for unchanged blocks
                    # ONLY send lightweight message if:
                    # 1. Block execution was unchanged (cache hit)
                    # 2. Client is not behind (force_full_data is false)
                    # 3. AND this is not the first run after a reload
                    #
                    # Send lightweight message when block is unchanged and client is up to date
                    send_lightweight = unchanged and not force_full_data

                    if send_lightweight:
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
                                "ordinal": result.get(
                                    "ordinal"
                                ),  # Position in document
                            },
                            file_path_str,
                        )

                # Send run-end message
                await self._ws_broadcast_to_file_watchers(
                    {"run": run, "type": "run-end"}, file_path_str
                )
            else:
                # Fallback to simple run-start when no API middleware
                await self._ws_broadcast_to_file_watchers(
                    {"run": run, "type": "run-start", "file": file_path_str},
                    file_path_str,
                )
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

    async def _send_reload_signal(self, changed_file, client_run=None):
        """Trigger a build for the changed file."""
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

    async def _send_file_change_notification(self):
        """Send notification about changed files after debounce period."""
        await asyncio.sleep(self._notification_delay)

        # Get the files that changed
        changed_files = list(self._changed_files_buffer)
        self._changed_files_buffer.clear()

        # Only notify if a single file changed
        if len(changed_files) == 1:
            abs_path = changed_files[0]
            try:
                rel_path = str(pathlib.Path(abs_path).relative_to(self.input_path))
            except Exception:
                rel_path = str(abs_path)
            message = {
                "type": "file-changed",
                "path": rel_path,
                "watched": str(abs_path) in self.client_registry.get_watched_files(),
            }

            if self.connections:
                message_str = json.dumps(message)
                await asyncio.gather(
                    *(ws.send(message_str) for ws in self.connections),
                    return_exceptions=True,
                )

                if self.verbose:
                    print(f"Sent file-changed notification for {rel_path}")

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
        if self.verbose:
            print(f"\nDependency graph watching: {self.dependency_graph.watched_path}")
            print("\nDependency graph imports:")
            for file, imports in sorted(self.dependency_graph.imports.items()):
                if imports:
                    print(f"  {file} -> {imports}")
            print("\nDependency graph imported_by:")
            for file, imported_by in sorted(self.dependency_graph.imported_by.items()):
                if imported_by:
                    print(f"  {file} <- {imported_by}")

        graph_stats = self.dependency_graph.get_graph_stats()
        print(
            f"Dependency graph: {graph_stats['total_files']} files, {graph_stats['total_imports']} imports"
        )

        async for changes in awatch(*paths_to_watch, stop_event=self._stop_event):
            directory_changed = False
            files_to_execute = set()

            # First, update the dependency graph based on the changes
            for change_type, path_str in changes:
                file_path = pathlib.Path(path_str)
                if not self._matches_patterns(file_path):
                    continue

                if change_type == 1:  # Added
                    directory_changed = True
                    if file_path.suffix == ".py" and file_path.exists():
                        self.dependency_graph.analyze_file(file_path)
                elif change_type == 2:  # Deleted
                    # Some editors (like VSCode) delete and recreate files when saving
                    # Check if the file actually exists - if so, treat as modified
                    if file_path.suffix == ".py":
                        if file_path.exists():
                            # File was recreated immediately - treat as modification
                            self.dependency_graph.analyze_file(file_path)
                        else:
                            # File was actually deleted
                            directory_changed = True
                            self.dependency_graph.remove_file(str(file_path))
                elif change_type == 3:  # Modified
                    if file_path.suffix == ".py" and file_path.exists():
                        self.dependency_graph.analyze_file(file_path)

            # If the directory structure changed, notify all clients
            if directory_changed:
                print("Directory structure changed, notifying clients.")
                await self._ws_broadcast_all({"type": "directory-changed"})

            # Determine which files need re-execution based on modifications
            modified_files = set()
            for change_type, path_str in changes:
                file_path = pathlib.Path(path_str)
                if self._matches_patterns(file_path):
                    if change_type == 3:  # Normal modification
                        modified_files.add(file_path.resolve())
                    elif (
                        change_type == 2 and file_path.exists()
                    ):  # Delete+recreate pattern
                        modified_files.add(file_path.resolve())

            if not modified_files:
                continue

            # Find all files affected by the modifications
            if self.verbose and modified_files:
                print(f"\nProcessing {len(modified_files)} modified file(s)...")
            for file_path in modified_files:
                try:
                    relative_path = str(file_path.relative_to(self.input_path))
                except ValueError:
                    if self.verbose:
                        print(f"  Skipping {file_path} - not relative to input path")
                    continue

                if self.verbose:
                    print(f"  Modified: {relative_path}")

                affected = self.dependency_graph.get_affected_files(relative_path)
                if self.verbose:
                    print(f"    Affected files: {affected}")

                for affected_file in affected:
                    affected_path = (self.input_path / affected_file).resolve()
                    if affected_path.exists():
                        files_to_execute.add(affected_path)
                        if self.verbose:
                            print(f"    Will execute: {affected_file}")

            # Process all affected files
            if files_to_execute:
                await self._process_multiple_files(files_to_execute)

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
