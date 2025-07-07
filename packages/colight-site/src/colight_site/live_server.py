"""LiveServer: On-demand development server for colight-site."""

import asyncio
import json
import pathlib
import threading
import time
import webbrowser
from typing import Optional, List, Set, Dict, Any, Tuple
import fnmatch

import websockets
from werkzeug.serving import make_server
from werkzeug.middleware.shared_data import SharedDataMiddleware
from werkzeug.wrappers import Request, Response
from watchfiles import awatch

import colight_site.api as api
from colight_site.builder import BuildConfig


def _get_metadata_path(html_path: pathlib.Path) -> pathlib.Path:
    """Get the metadata file path for an HTML file."""
    return html_path.with_suffix(".meta.json")


def _save_build_metadata(
    html_path: pathlib.Path, source_path: pathlib.Path, config: BuildConfig
) -> None:
    """Save build metadata for caching."""
    metadata = {
        "source_mtime": source_path.stat().st_mtime,
        "pragma": sorted(list(config.pragma)),
        "source_path": str(source_path),
    }

    meta_path = _get_metadata_path(html_path)
    meta_path.write_text(json.dumps(metadata, indent=2))


def _load_build_metadata(html_path: pathlib.Path) -> Optional[Dict]:
    """Load build metadata if it exists."""
    meta_path = _get_metadata_path(html_path)
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _should_rebuild_with_metadata(
    source_path: pathlib.Path, html_path: pathlib.Path, config: BuildConfig
) -> bool:
    """Check if rebuild is needed considering metadata."""
    if not html_path.exists():
        return True

    metadata = _load_build_metadata(html_path)
    if not metadata:
        # No metadata, fall back to mtime check
        return source_path.stat().st_mtime > html_path.stat().st_mtime

    # Check if source file changed
    current_mtime = source_path.stat().st_mtime
    if current_mtime > metadata.get("source_mtime", 0):
        return True

    # Check if pragma tags changed
    current_pragmas = sorted(list(config.pragma))
    cached_pragmas = metadata.get("pragma", [])
    if current_pragmas != cached_pragmas:
        return True

    return False


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

    def _get_files(self) -> List[str]:
        """Get list of all matching Python files."""
        from .index_generator import find_colight_files

        combined_ignore = self._get_combined_ignore_patterns()
        files = find_colight_files(self.input_path, self.include, combined_ignore)

        # Convert to relative paths without extensions
        if self.input_path.is_file():
            return [self.input_path.stem]
        else:
            paths = []
            for f in files:
                rel_path = str(f.relative_to(self.input_path))
                # Remove .py extension
                if rel_path.endswith(".py"):
                    rel_path = rel_path[:-3]
                # Remove .colight.py extension
                if rel_path.endswith(".colight"):
                    rel_path = rel_path[:-8]
                paths.append(rel_path)
            return paths

    def _get_combined_ignore_patterns(self) -> List[str]:
        """Get combined default and user ignore patterns."""
        default_ignore = [
            ".*",  # Hidden files/dirs
            "__pycache__",
            "*.pyc",
            "__init__.py",
            "node_modules",
            ".git",
            ".venv",
            "venv",
            ".env",
            "env",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".coverage",
            "htmlcov",
            ".tox",
            "*.egg-info",
            ".idea",
            ".vscode",
            ".colight_cache",
        ]

        combined_ignore = list(self.ignore) if self.ignore else []
        combined_ignore.extend(default_ignore)
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

        # Handle /api/content/<path> endpoint (legacy HTML)
        if path.startswith("/api/content/"):
            file_path = path[13:]  # Remove /api/content/

            # Add .html extension if not present (for finding source)
            if not file_path.endswith(".html"):
                file_path = file_path + ".html"

            # Build the file if needed (reuse OnDemandMiddleware logic)
            source_file = self._find_source_file(file_path)
            if source_file:
                output_file = self._get_output_path(source_file)

                if _should_rebuild_with_metadata(source_file, output_file, self.config):
                    try:
                        # Ensure output directory exists
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        # Build the file
                        api.build_file(source_file, output_file, config=self.config)
                        # Save metadata
                        _save_build_metadata(output_file, source_file, self.config)
                    except Exception as e:
                        error_html = f'<div style="color: red; white-space: pre-wrap; padding: 20px; font-family: monospace;">Build Error:\n{str(e)}</div>'
                        response = Response(
                            error_html, status=500, mimetype="text/html"
                        )
                        return response(environ, start_response)

                # Read and return the built HTML
                if output_file.exists():
                    content = output_file.read_text()
                    response = Response(content, mimetype="text/html")
                    return response(environ, start_response)

            # File not found
            response = Response("File not found", status=404)
            return response(environ, start_response)

        # Handle /api/document/<path> endpoint (new JSON)
        if path.startswith("/api/document/"):
            file_path = path[14:]  # Remove /api/document/

            # Find source file
            source_file = self._find_source_file(file_path + ".html")
            if source_file:
                try:
                    # Generate JSON directly from source
                    from .json_generator import JsonFormGenerator

                    generator = JsonFormGenerator(config=self.config)
                    json_content = generator.generate_json(source_file)
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

        # Not an API request, pass through
        return self.app(environ, start_response)

    def _find_source_file(self, requested_path: str) -> Optional[pathlib.Path]:
        """Find the source .py file for a requested path."""
        # Remove .html extension if present
        clean_path = requested_path.removesuffix(".html")

        # Check if we're in single file mode
        if self.input_path.is_file():
            if clean_path == self.input_path.stem or clean_path == "":
                return (
                    self.input_path if self._matches_patterns(self.input_path) else None
                )
            return None

        # Try different variations
        possible_paths = [
            self.input_path / f"{clean_path}.py",
            self.input_path / f"{clean_path}.colight.py",
            self.input_path / clean_path / "__init__.py",
        ]

        for source_path in possible_paths:
            if (
                source_path.exists()
                and source_path.is_file()
                and self._matches_patterns(source_path)
            ):
                return source_path

        return None

    def _get_output_path(self, source_file: pathlib.Path) -> pathlib.Path:
        """Get the output path for a source file."""
        if self.input_path.is_file():
            return self.output_path / source_file.with_suffix(".html").name

        try:
            rel_path = source_file.relative_to(self.input_path)
            return self.output_path / rel_path.with_suffix(".html")
        except ValueError:
            return self.output_path / source_file.with_suffix(".html").name

    def _matches_patterns(self, file_path: pathlib.Path) -> bool:
        """Check if file matches include/ignore patterns."""
        from .index_generator import matches_patterns

        combined_ignore = self._get_combined_ignore_patterns()
        return matches_patterns(file_path, self.include, combined_ignore)


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

    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _matches_patterns(self, file_path: pathlib.Path) -> bool:
        """Check if file matches include patterns and doesn't match ignore patterns."""
        # Use the same pattern matching logic as LiveServer
        from .index_generator import matches_patterns

        # Get default ignore patterns
        default_ignore = [
            ".*",  # Hidden files/dirs
            "__pycache__",
            "*.pyc",
            "__init__.py",  # Ignore __init__ files
            "node_modules",
            ".git",
            ".venv",
            "venv",
            ".env",
            "env",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".coverage",
            "htmlcov",
            ".tox",
            "*.egg-info",
            ".idea",
            ".vscode",
            ".colight_cache",
        ]

        combined_ignore = list(self.ignore) if self.ignore else []
        combined_ignore.extend(default_ignore)

        return matches_patterns(file_path, self.include, combined_ignore)

    def _find_source_file(self, requested_path: str) -> Optional[pathlib.Path]:
        """Find the source .py file for a requested .html path."""
        # Remove leading slash and .html extension
        clean_path = requested_path.lstrip("/").removesuffix(".html")

        # Check if we're in single file mode
        if self.input_path.is_file():
            # For single file mode, only serve if the name matches
            if clean_path == self.input_path.stem or clean_path == "":
                return (
                    self.input_path if self._matches_patterns(self.input_path) else None
                )
            return None

        # If empty path, don't try to find source (let index.html be served)
        if not clean_path:
            return None

        # Try different variations
        possible_paths = [
            self.input_path / f"{clean_path}.py",
            self.input_path / f"{clean_path}.colight.py",
            self.input_path / clean_path / "__init__.py",
        ]

        for source_path in possible_paths:
            if (
                source_path.exists()
                and source_path.is_file()
                and self._matches_patterns(source_path)
            ):
                return source_path

        return None

    def _get_output_path(self, source_file: pathlib.Path) -> pathlib.Path:
        """Get the output path for a source file, mirroring directory structure."""
        if self.input_path.is_file():
            # Single file mode
            return self.output_path / source_file.with_suffix(".html").name

        # Directory mode - mirror the structure
        try:
            rel_path = source_file.relative_to(self.input_path)
            return self.output_path / rel_path.with_suffix(".html")
        except ValueError:
            # Fallback if not relative
            return self.output_path / source_file.with_suffix(".html").name

    def _get_combined_ignore_patterns(self) -> List[str]:
        """Get combined default and user ignore patterns."""
        default_ignore = [
            ".*",  # Hidden files/dirs
            "__pycache__",
            "*.pyc",
            "__init__.py",  # Ignore __init__ files
            "node_modules",
            ".git",
            ".venv",
            "venv",
            ".env",
            "env",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".coverage",
            "htmlcov",
            ".tox",
            "*.egg-info",
            ".idea",
            ".vscode",
            ".colight_cache",
        ]

        combined_ignore = list(self.ignore) if self.ignore else []
        combined_ignore.extend(default_ignore)
        return combined_ignore

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

        # Special handling for index.html
        if path == "/index.html" or path == "index.html":
            # Check if index.html needs to be regenerated
            index_file = self.output_path / "index.html"
            if not index_file.exists() and not self.input_path.is_file():
                # Regenerate index
                from .index_generator import generate_index_html

                print("Regenerating index.html...")
                combined_ignore = self._get_combined_ignore_patterns()
                generate_index_html(
                    self.input_path, self.output_path, self.include, combined_ignore
                )
            # Let it be served normally
            return self.app(environ, start_response)

        # Normalize path to always end with .html
        if not path.endswith(".html"):
            path = f"{path}.html"

        # Try to find source file
        source_file = self._find_source_file(path)

        if self.config.verbose:
            print(f"Looking for source file for path: {path}")
            print(f"Found source file: {source_file}")

        if source_file:
            output_file = self._get_output_path(source_file)

            if self.config.verbose:
                print(f"Output file will be: {output_file}")

            # Build if needed
            if _should_rebuild_with_metadata(source_file, output_file, self.config):
                try:
                    print(f"Building {source_file} -> {output_file}")

                    # Ensure output directory exists
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    # Build the file
                    api.build_file(source_file, output_file, config=self.config)
                    # Save metadata
                    _save_build_metadata(output_file, source_file, self.config)

                    if self.config.verbose:
                        print(f"Built {source_file} successfully")

                except Exception as e:
                    print(f"Error building {source_file}: {e}")
                    error_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Build Error</title>
                        <style>
                            body {{ font-family: monospace; margin: 40px; }}
                            .error {{ color: red; white-space: pre-wrap; }}
                        </style>
                    </head>
                    <body>
                        <h1>Build Error</h1>
                        <p>Failed to build: {source_file}</p>
                        <pre class="error">{str(e)}</pre>
                    </body>
                    </html>
                    """
                    response = Response(error_html, status=500, mimetype="text/html")
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
        default_ignore = [
            ".*",  # Hidden files/dirs
            "__pycache__",
            "*.pyc",
            "__init__.py",  # Ignore __init__ files
            "node_modules",
            ".git",
            ".venv",
            "venv",
            ".env",
            "env",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".coverage",
            "htmlcov",
            ".tox",
            "*.egg-info",
            ".idea",
            ".vscode",
            ".colight_cache",
        ]

        combined_ignore = list(self.ignore) if self.ignore else []
        combined_ignore.extend(default_ignore)
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
