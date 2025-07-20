"""Simplified incremental executor using cache-key-based block IDs."""

from typing import Any, Dict, Iterator, Optional, Set, Tuple

from colight_site.executor import BlockExecutor, ExecutionResult
from colight_site.model import Block, Document

from .block_cache import BlockCache


class IncrementalExecutor(BlockExecutor):
    """Execute blocks incrementally using cache-key-based IDs."""

    def __init__(self, verbose: bool = False, block_cache: Optional[BlockCache] = None):
        super().__init__(verbose)
        self.cache: Dict[str, ExecutionResult] = {}  # cache_key -> result
        self.block_cache = block_cache or BlockCache()
        self.current_file: Optional[str] = None
        self.project_root: Optional[str] = None

    def execute_incremental_streaming(
        self,
        document: Document,
        changed_blocks: Optional[Set[str]] = None,
        filename: str = "<string>",
        source_file: Optional[str] = None,
    ) -> Iterator[Tuple[Block, ExecutionResult]]:
        """Execute blocks incrementally, yielding results as they complete.

        Since blocks already have cache keys as IDs, we just check the cache
        and execute if needed.
        """
        # Set current file for cache tracking
        self.current_file = source_file

        # Execute blocks in document order
        for block in document.blocks:
            # Check cache using the block's ID (which is its cache key)
            cache_key = block.id

            if cache_key in self.cache:
                # Cache hit
                if self.verbose:
                    print(f"Cache hit for block {cache_key[:8]}... in {filename}")
                result = self.cache[cache_key]
                result.cache_hit = True

                # Record cache access
                self.block_cache.access_entry(cache_key)
            else:
                # Cache miss - execute the block
                if self.verbose:
                    print(
                        f"Cache miss - executing block {cache_key[:8]}... in {filename}"
                    )

                result = self.execute_block(block, filename)
                result.cache_hit = False

                # Store in cache
                self.cache[cache_key] = result

                # Add to cache manager
                if self.current_file:
                    size = self.block_cache.estimate_entry_size(result)
                    self.block_cache.add_entry(cache_key, self.current_file, size)

            # Add cache key to result
            result.cache_key = cache_key

            # Yield the result
            yield block, result

    def clear_cache(self):
        """Clear all cached results."""
        self.cache.clear()

    def evict_cached_entries(self, file_path: str):
        """Evict cache entries for a specific file."""
        evicted = self.block_cache.evict_file(file_path)

        # Remove from local cache
        for cache_key in evicted:
            self.cache.pop(cache_key, None)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.block_cache.get_stats()

    def mark_file_for_eviction(self, file_path: str):
        """Mark a file's cache entries for potential eviction."""
        self.block_cache.mark_file_for_eviction(file_path)

    def unmark_file_for_eviction(self, file_path: str):
        """Remove eviction mark from a file."""
        self.block_cache.unmark_file_for_eviction(file_path)

    def evict_unwatched_files(self, force: bool = False):
        """Evict cache entries for files marked for eviction."""
        evicted = self.block_cache.evict_marked_files(force)
        # Remove from local cache
        for cache_key in evicted:
            self.cache.pop(cache_key, None)
