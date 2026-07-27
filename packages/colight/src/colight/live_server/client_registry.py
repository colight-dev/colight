"""Client registry for tracking which clients are watching which files."""

from collections import defaultdict
from typing import Any, Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


class ClientRegistry:
    """Registry to track which clients are watching which files."""

    def __init__(self):
        self.clients: Dict[str, Any] = {}  # clientId -> WebSocket
        self.watched_files: Dict[str, Set[str]] = defaultdict(
            set
        )  # file_path -> Set[clientId]
        self.client_files: Dict[str, str] = {}  # clientId -> current_file

    def register_client(self, client_id: str, websocket):
        """Register a new client connection."""
        self.clients[client_id] = websocket
        logger.info(f"Registered client: {client_id}")

    def unregister_client(self, client_id: str):
        """Remove a client and all its watches."""
        if client_id in self.clients:
            del self.clients[client_id]

        # Remove from all watched files
        for file_path in list(self.watched_files.keys()):
            self.watched_files[file_path].discard(client_id)
            if not self.watched_files[file_path]:
                del self.watched_files[file_path]

        # Remove current file tracking
        if client_id in self.client_files:
            del self.client_files[client_id]

        logger.info(f"Unregistered client: {client_id}")

    def watch_file(self, client_id: str, file_path: str) -> bool:
        """Register that a client is watching a file."""
        if client_id not in self.clients:
            logger.warning(
                f"Client {client_id} not registered, cannot watch {file_path}"
            )
            return False

        # Unwatch previous file if any
        if client_id in self.client_files:
            old_file = self.client_files[client_id]
            self.watched_files[old_file].discard(client_id)
            if not self.watched_files[old_file]:
                del self.watched_files[old_file]
            logger.debug(f"Client {client_id} unwatched {old_file}")

        # Watch new file
        self.watched_files[file_path].add(client_id)
        self.client_files[client_id] = file_path
        logger.info(f"Client {client_id} watching {file_path}")
        return True

    def unwatch_file(self, client_id: str, file_path: str) -> bool:
        """Unregister that a client is watching a file."""
        if client_id in self.clients and file_path in self.watched_files:
            self.watched_files[file_path].discard(client_id)
            if not self.watched_files[file_path]:
                del self.watched_files[file_path]

            if self.client_files.get(client_id) == file_path:
                del self.client_files[client_id]

            logger.info(f"Client {client_id} unwatched {file_path}")
            return True
        return False

    def get_watchers(self, file_path: str) -> Set[str]:
        """Get all clients watching a file."""
        return self.watched_files.get(file_path, set())

    def get_watched_files(self) -> Set[str]:
        """Get all files being watched by any client."""
        return set(self.watched_files.keys())

    def get_client_websocket(self, client_id: str) -> Optional[Any]:
        """Get the WebSocket for a client."""
        return self.clients.get(client_id)

    def log_status(self):
        """Log current registry status for debugging."""
        logger.info(
            f"Registry status: {len(self.clients)} clients, {len(self.watched_files)} watched files"
        )
        for file_path, watchers in self.watched_files.items():
            logger.debug(f"  {file_path}: {len(watchers)} watchers")
