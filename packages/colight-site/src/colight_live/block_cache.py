"""Cache management for incremental execution with file-aware eviction."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Set

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    cache_key: str
    file_path: str
    size_bytes: int
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    is_hot: bool = False  # Frequently accessed

    def access(self):
        """Record an access to this cache entry."""
        self.last_access = time.time()
        self.access_count += 1
        # Mark as hot if accessed frequently
        if self.access_count > 5:
            self.is_hot = True


class BlockCache:
    """Manages block execution cache with file-aware eviction policies."""

    def __init__(self, max_size_mb: int = 500, hot_threshold_seconds: int = 300):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.hot_threshold_seconds = hot_threshold_seconds

        # Core data structures
        self.entries: Dict[str, CacheEntry] = {}  # cache_key -> CacheEntry
        self.file_entries: Dict[str, Set[str]] = defaultdict(
            set
        )  # file_path -> Set[cache_key]
        self.marked_for_eviction: Set[str] = set()  # file paths marked for eviction

        # Statistics
        self.total_size = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def add_entry(self, cache_key: str, file_path: str, size_bytes: int):
        """Add a new cache entry."""
        # Remove old entry if exists
        if cache_key in self.entries:
            self.remove_entry(cache_key)

        # Check if we need to make space BEFORE adding
        if self.total_size + size_bytes > self.max_size_bytes:
            self._evict_to_make_space(size_bytes)

        entry = CacheEntry(cache_key, file_path, size_bytes)
        self.entries[cache_key] = entry
        self.file_entries[file_path].add(cache_key)
        self.total_size += size_bytes

    def access_entry(self, cache_key: str) -> bool:
        """Record access to a cache entry. Returns True if found."""
        if cache_key in self.entries:
            self.entries[cache_key].access()
            self.hit_count += 1
            return True
        else:
            self.miss_count += 1
            return False

    def remove_entry(self, cache_key: str):
        """Remove a cache entry."""
        if cache_key not in self.entries:
            return

        entry = self.entries[cache_key]
        self.total_size -= entry.size_bytes
        self.file_entries[entry.file_path].discard(cache_key)
        if not self.file_entries[entry.file_path]:
            del self.file_entries[entry.file_path]
        del self.entries[cache_key]

    def mark_file_for_eviction(self, file_path: str):
        """Mark all entries from a file for potential eviction."""
        self.marked_for_eviction.add(file_path)
        logger.debug(f"Marked {file_path} for cache eviction")

    def unmark_file_for_eviction(self, file_path: str):
        """Remove eviction mark from a file (e.g., it became watched again)."""
        self.marked_for_eviction.discard(file_path)

    def evict_marked_files(self, force: bool = False):
        """Evict cache entries from marked files.

        Args:
            force: If True, evict even hot entries. If False, keep hot entries.
        """
        evicted = 0
        for file_path in list(self.marked_for_eviction):
            if file_path in self.file_entries:
                for cache_key in list(self.file_entries[file_path]):
                    entry = self.entries.get(cache_key)
                    if entry and (force or not self._is_hot(entry)):
                        self.remove_entry(cache_key)
                        evicted += 1
                        self.eviction_count += 1

            # Remove from marked set if all entries evicted
            if file_path not in self.file_entries:
                self.marked_for_eviction.discard(file_path)

        if evicted > 0:
            logger.info(f"Evicted {evicted} cache entries from unwatched files")

    def _is_hot(self, entry: CacheEntry) -> bool:
        """Check if an entry is hot (recently/frequently accessed)."""
        if entry.is_hot:
            return True

        # Only consider it hot if it's been accessed (not just created)
        if entry.access_count == 0:
            return False

        # Check if accessed recently
        age = time.time() - entry.last_access
        return age < self.hot_threshold_seconds

    def _evict_to_make_space(self, needed_bytes: int):
        """Evict entries to make space for new entry."""
        target_size = self.max_size_bytes - needed_bytes

        # First try evicting from marked files
        self.evict_marked_files(force=False)

        if self.total_size <= target_size:
            return

        # Sort entries for eviction (least valuable first)
        # Priority: non-hot & never-accessed < non-hot & accessed < hot
        def eviction_priority(entry):
            if entry.is_hot:
                return (2, -entry.last_access)  # Hot entries last
            elif entry.access_count > 0:
                return (1, -entry.last_access)  # Accessed entries second
            else:
                return (0, -entry.last_access)  # Never accessed first

        sorted_entries = sorted(self.entries.values(), key=eviction_priority)

        # Evict until we have enough space
        for entry in sorted_entries:
            if self.total_size <= target_size:
                break

            # Skip hot entries unless desperate
            if entry.is_hot and self.total_size < self.max_size_bytes * 1.2:
                continue

            self.remove_entry(entry.cache_key)
            self.eviction_count += 1

        if self.total_size > target_size:
            logger.warning(
                f"Could not evict enough entries. Need {needed_bytes / 1024 / 1024:.1f}MB, current size: {self.total_size / 1024 / 1024:.1f}MB"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.entries)
        hot_entries = sum(1 for e in self.entries.values() if self._is_hot(e))
        watched_entries = sum(
            1
            for e in self.entries.values()
            if e.file_path not in self.marked_for_eviction
        )

        hit_rate = (
            self.hit_count / (self.hit_count + self.miss_count)
            if (self.hit_count + self.miss_count) > 0
            else 0
        )

        return {
            "total_entries": total_entries,
            "hot_entries": hot_entries,
            "watched_entries": watched_entries,
            "marked_files": len(self.marked_for_eviction),
            "size_mb": self.total_size / 1024 / 1024,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
        }

    def clear_file_cache(self, file_path: str):
        """Clear all cache entries for a specific file."""
        if file_path in self.file_entries:
            for cache_key in list(self.file_entries[file_path]):
                self.remove_entry(cache_key)
            logger.debug(f"Cleared cache for {file_path}")

    def estimate_entry_size(self, result) -> int:
        """Estimate the memory size of a cache entry.

        This is a rough estimate based on the result content.
        """
        # Base size
        size = 1024  # 1KB base

        # Add size for output
        if hasattr(result, "output") and result.output:
            size += len(result.output.encode("utf-8"))

        # Add size for error
        if hasattr(result, "error") and result.error:
            size += len(result.error.encode("utf-8"))

        # Add size for colight bytes
        if hasattr(result, "colight_bytes") and result.colight_bytes:
            size += len(result.colight_bytes)

        # Add overhead estimate
        size += 512  # Python object overhead

        return size
