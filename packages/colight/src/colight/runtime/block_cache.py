"""Unified cache for incremental execution with delayed file eviction.

IMPORTANT: Cache keys are based ONLY on the source code of blocks (transitively),
not on runtime values. This means:
- A block with the same source will always have the same cache key
- We never attempt to serialize/hash Python runtime values
- Changing any source code invalidates the cache for affected blocks
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from .executor import ExecutionResult

logger = logging.getLogger(__name__)

logger.setLevel("INFO")


@dataclass
class CachedResult:
    """A cached execution result."""

    cache_key: str  # Based on source code only
    file_path: str
    result: ExecutionResult
    timestamp: float = field(default_factory=time.time)


class BlockCache:
    """Single cache for block execution results with delayed file eviction.

    The cache stores execution results keyed by source-based cache keys.
    Files marked for eviction are not immediately cleared to handle
    temporary unwatching (e.g., browser reload).
    """

    def __init__(self, eviction_delay_seconds: int = 30):
        self.eviction_delay_seconds = eviction_delay_seconds

        # Core data structure - single source of truth
        self.cache: Dict[str, CachedResult] = {}  # cache_key -> CachedResult

        # Indexes for efficient lookups
        self.file_entries: Dict[str, Set[str]] = defaultdict(
            set
        )  # file_path -> Set[cache_key]
        self.eviction_times: Dict[str, float] = {}  # file_path -> time when marked

        # Statistics
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def get(self, cache_key: str) -> Optional[ExecutionResult]:
        """Get a cached result."""
        if cache_key in self.cache:
            self.hit_count += 1
            return self.cache[cache_key].result
        else:
            self.miss_count += 1
            return None

    def put(self, cache_key: str, file_path: str, result: ExecutionResult):
        """Add or update a cache entry."""
        # Remove old entry if exists
        if cache_key in self.cache:
            self._remove_entry(cache_key)

        # Add new entry
        entry = CachedResult(cache_key, file_path, result)
        self.cache[cache_key] = entry
        self.file_entries[file_path].add(cache_key)

    def remove(self, cache_key: str) -> bool:
        """Remove a cache entry. Returns True if entry existed."""
        if cache_key in self.cache:
            self._remove_entry(cache_key)
            return True
        return False

    def _remove_entry(self, cache_key: str):
        """Internal method to remove an entry and update indexes."""
        entry = self.cache[cache_key]
        self.file_entries[entry.file_path].discard(cache_key)
        if not self.file_entries[entry.file_path]:
            del self.file_entries[entry.file_path]
        del self.cache[cache_key]

    def clear_file(self, file_path: str) -> Set[str]:
        """Clear all cache entries for a file. Returns removed keys."""
        if file_path not in self.file_entries:
            return set()

        removed_keys = set()
        for cache_key in list(self.file_entries[file_path]):
            self._remove_entry(cache_key)
            removed_keys.add(cache_key)

        logger.debug(f"Cleared {len(removed_keys)} entries for {file_path}")
        return removed_keys

    def clean_stale_entries(
        self, file_path: str, current_block_ids: Set[str]
    ) -> Set[str]:
        """Remove entries for blocks no longer in the document. Returns removed keys.

        NOTE: This method has a known limitation - if a block is deleted and then
        recreated with identical source code, it will have the same cache key and
        won't be considered "stale". This can cause the server to send "unchanged"
        messages for blocks the client doesn't have. Consider using clear_file()
        instead when the file has been modified.
        """
        if file_path not in self.file_entries:
            return set()

        # Find stale entries
        file_cache_keys = self.file_entries[file_path].copy()
        stale_keys = file_cache_keys - current_block_ids

        # Remove stale entries
        removed_keys = set()
        for cache_key in stale_keys:
            self._remove_entry(cache_key)
            removed_keys.add(cache_key)

        if True or removed_keys:
            print(f"Removed {len(removed_keys)} stale entries for {file_path}")
            print(f"Stale keys removed: {[k[:8] + '...' for k in removed_keys]}")
            print(f"Current block IDs: {[k[:8] + '...' for k in current_block_ids]}")
            print(
                f"Remaining cached: {[k[:8] + '...' for k in self.file_entries.get(file_path, set())]}"
            )

        return removed_keys

    def mark_file_for_eviction(self, file_path: str):
        """Mark a file for delayed eviction."""
        self.eviction_times[file_path] = time.time()
        logger.debug(f"Marked {file_path} for eviction")

    def unmark_file_for_eviction(self, file_path: str):
        """Cancel pending eviction for a file."""
        self.eviction_times.pop(file_path, None)

    def evict_marked_files(self, force: bool = False) -> Set[str]:
        """Evict entries from files marked for eviction after delay.

        Args:
            force: If True, evict immediately regardless of delay
        """
        evicted_keys = set()
        current_time = time.time()

        for file_path, mark_time in list(self.eviction_times.items()):
            # Check if enough time has passed or force eviction
            if force or (current_time - mark_time) >= self.eviction_delay_seconds:
                # Evict all entries for this file
                if file_path in self.file_entries:
                    keys_to_evict = list(self.file_entries[file_path])
                    for cache_key in keys_to_evict:
                        self._remove_entry(cache_key)
                        evicted_keys.add(cache_key)
                        self.eviction_count += 1

                # Remove from eviction times
                del self.eviction_times[file_path]

        if evicted_keys:
            logger.info(f"Evicted {len(evicted_keys)} entries from unwatched files")

        return evicted_keys

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        watched_entries = sum(
            1 for e in self.cache.values() if e.file_path not in self.eviction_times
        )

        hit_rate = (
            self.hit_count / (self.hit_count + self.miss_count)
            if (self.hit_count + self.miss_count) > 0
            else 0
        )

        return {
            "total_entries": total_entries,
            "watched_entries": watched_entries,
            "files_marked_for_eviction": len(self.eviction_times),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "eviction_count": self.eviction_count,
        }
