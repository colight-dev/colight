"""Simplified incremental executor using cache-key-based block IDs."""

from typing import Iterator, Optional, Set, Tuple

from colight_site.executor import BlockExecutor, ExecutionResult
from colight_site.model import Block, Document

from .block_cache import BlockCache


class IncrementalExecutor(BlockExecutor):
    """Execute blocks incrementally using cache-key-based IDs."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.cache = BlockCache(eviction_delay_seconds=30)
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

            # Try to get from cache
            result = self.cache.get(cache_key)

            if result is not None:
                # Cache hit
                if self.verbose:
                    print(f"Cache hit for block {cache_key[:8]}... in {filename}")
                result.cache_hit = True
            else:
                # Cache miss - execute the block
                if self.verbose:
                    print(
                        f"Cache miss - executing block {cache_key[:8]}... in {filename}"
                    )

                result = self.execute_block(block, filename)
                result.cache_hit = False

                # Store in cache
                if self.current_file:
                    self.cache.put(cache_key, self.current_file, result)

            # Add cache key to result
            result.cache_key = cache_key

            # Yield the result
            yield block, result

    def clear_cache(self):
        """Clear all cached results."""
        self.cache = BlockCache()

    def evict_cached_entries(self, file_path: str):
        """Evict cache entries for a specific file."""
        self.cache.clear_file(file_path)

    def get_cache_stats(self):
        """Get cache statistics."""
        return self.cache.get_stats()

    def mark_file_for_eviction(self, file_path: str):
        """Mark a file's cache entries for potential eviction."""
        self.cache.mark_file_for_eviction(file_path)

    def unmark_file_for_eviction(self, file_path: str):
        """Remove eviction mark from a file."""
        self.cache.unmark_file_for_eviction(file_path)

    def evict_unwatched_files(self, force: bool = False):
        """Evict cache entries for files marked for eviction."""
        self.cache.evict_marked_files(force)

    def clean_stale_entries(self, file_path: str, current_block_ids: Set[str]):
        """Remove cache entries for blocks no longer in the document."""
        self.cache.clean_stale_entries(file_path, current_block_ids)
