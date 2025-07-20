"""Incremental executor using dependency graph for smart re-execution."""

import hashlib
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from colight_site.executor import BlockExecutor, ExecutionResult
from colight_site.model import Block, Document
from colight_site.utils import hash_block_content

from .block_cache import BlockCache
from .block_graph import BlockGraph


@dataclass
class BlockState:
    """State of a block for incremental execution."""

    content_hash: str
    cache_key: str  # Content-addressable key including dependencies
    result: ExecutionResult
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)


class IncrementalExecutor(BlockExecutor):
    """Execute blocks incrementally based on dependency graph."""

    def __init__(self, verbose: bool = False, block_cache: Optional[BlockCache] = None):
        super().__init__(verbose)
        self.block_states: Dict[str, BlockState] = {}  # block_id -> BlockState
        self.cache_by_key: Dict[str, ExecutionResult] = {}  # cache_key -> result
        self.block_graph: Optional[BlockGraph] = None
        self.block_cache = block_cache or BlockCache()
        self.current_file: Optional[str] = None  # Track current file being executed
        self.project_root: Optional[str] = None  # Track project root directory

    def _get_content_hash(self, block: Block) -> str:
        """Get a hash of block's content including prose."""
        return hash_block_content(block)

    def _compute_cache_key(self, block: Block, dependency_keys: Dict[str, str]) -> str:
        """Compute content-addressable cache key for a block.

        The key is based on:
        - The block's own content (including prose)
        - The cache keys of all its dependencies
        - The modification times of file dependencies

        This ensures that if any dependency changes, this block's key changes too.
        """
        # Start with block's own content hash
        content = hash_block_content(block).encode()

        # Add sorted dependency cache keys
        dep_keys = []
        for dep_name in sorted(block.interface.requires):
            # Find which block provides this dependency
            dep_block_id = None
            if self.block_graph and dep_name in self.block_graph.symbol_providers:
                dep_block_id = self.block_graph.symbol_providers[dep_name]

            if dep_block_id and dep_block_id in dependency_keys:
                dep_keys.append(f"{dep_name}:{dependency_keys[dep_block_id]}")

        # Add file dependencies (modification times)
        file_deps = []
        if hasattr(block.interface, "file_dependencies"):
            for file_path in sorted(block.interface.file_dependencies):
                # File path is relative to project root
                if self.project_root:
                    abs_path = os.path.join(self.project_root, file_path)
                else:
                    abs_path = file_path

                try:
                    mtime = os.path.getmtime(abs_path)
                    file_deps.append(f"file:{file_path}:{mtime}")
                except OSError:
                    # File doesn't exist or can't be accessed
                    file_deps.append(f"file:{file_path}:missing")

        # Combine content, dependencies, and file dependencies
        combined = (
            content
            + b"::"
            + "\n".join(dep_keys).encode()
            + b"::"
            + "\n".join(file_deps).encode()
        )
        return hashlib.sha256(combined).hexdigest()[:16]  # Use first 16 chars

    def _has_always_eval_pragma(self, block: Block) -> bool:
        """Check if block has the always-eval pragma."""
        return "always-eval" in block.tags.flags

    def execute_incremental(
        self,
        document: Document,
        changed_blocks: Optional[Set[str]] = None,
        filename: str = "<string>",
        source_file: Optional[str] = None,
    ) -> List[Tuple[Block, ExecutionResult]]:
        """Execute blocks incrementally based on changes.

        Args:
            document: The document to execute
            changed_blocks: Set of block IDs that have changed (None = all changed)
            filename: The filename for error reporting
            source_file: The source file path for cache tracking

        Returns:
            List of (block, result) tuples in document order
        """
        # Set current file for cache tracking
        self.current_file = source_file

        # Set project root for file dependency resolution
        if source_file:
            project_root = None
            for parent in pathlib.Path(source_file).parents:
                if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                    project_root = str(parent)
                    break
            self.project_root = project_root

        # Build dependency graph
        self.block_graph = BlockGraph()
        # Convert blocks to dict format expected by BlockGraph
        block_dicts = []
        for block in document.blocks:
            block_dict = {
                "id": str(block.id),
                "interface": {
                    "provides": block.interface.provides,
                    "requires": block.interface.requires,
                },
            }
            block_dicts.append(block_dict)
        self.block_graph.add_blocks(block_dicts)

        # Determine which blocks need re-execution
        blocks_to_execute = set()

        # If no specific changes, check all blocks
        if changed_blocks is None:
            for block in document.blocks:
                block_id = str(block.id)
                content_hash = self._get_content_hash(block)

                # Check if block is new or changed
                if (
                    block_id not in self.block_states
                    or self.block_states[block_id].content_hash != content_hash
                    or self._has_always_eval_pragma(block)
                ):
                    blocks_to_execute.add(block_id)
        else:
            # Start with explicitly changed blocks
            blocks_to_execute = changed_blocks.copy()

        # Add always-eval blocks
        for block in document.blocks:
            if self._has_always_eval_pragma(block):
                blocks_to_execute.add(str(block.id))

        # Find all downstream dependencies
        all_dirty = set(self.block_graph.dirty_after(blocks_to_execute))

        if self.verbose:
            print(f"Executing {len(all_dirty)} blocks out of {len(document.blocks)}")

        # Get execution order for dirty blocks
        execution_order = self.block_graph.execution_order()

        # Execute blocks in dependency order
        results = []
        block_map = {str(block.id): block for block in document.blocks}

        # First, compute content hashes for all blocks and detect changes
        content_hashes = {}
        content_changed = set()  # Blocks whose content actually changed
        for block in document.blocks:
            block_id = str(block.id)
            new_hash = self._get_content_hash(block)
            content_hashes[block_id] = new_hash

            # Check if content changed from last execution
            if block_id in self.block_states:
                if self.block_states[block_id].content_hash != new_hash:
                    content_changed.add(block_id)
            else:
                # New block
                content_changed.add(block_id)

        # Then compute cache keys in dependency order
        dependency_keys = {}
        for block_id in execution_order:
            block = block_map.get(block_id)
            if not block:
                continue

            # For dirty blocks or blocks with changed content, compute new cache key
            if block_id in all_dirty or (
                block_id in self.block_states
                and self.block_states[block_id].content_hash != content_hashes[block_id]
            ):
                cache_key = self._compute_cache_key(block, dependency_keys)
            else:
                # Reuse existing cache key if block hasn't changed
                cache_key = self.block_states.get(
                    block_id, BlockState("", "", ExecutionResult())
                ).cache_key
                if not cache_key:
                    cache_key = self._compute_cache_key(block, dependency_keys)

            dependency_keys[block_id] = cache_key

        # Now execute blocks using the computed cache keys
        for block_id in execution_order:
            block = block_map.get(block_id)
            if not block:
                continue

            cache_key = dependency_keys[block_id]

            # Check if we have a cached result for this cache key
            if cache_key in self.cache_by_key and block_id not in all_dirty:
                # Reuse cached result
                if self.verbose:
                    print(f"Cache hit for block {block_id} (key: {cache_key})")
                result = self.cache_by_key[cache_key]
                # Mark this as a cache hit
                result.cache_hit = True
                result.content_changed = block_id in content_changed
                result.cache_key = cache_key
                results.append((block, result))
                # Record cache access
                self.block_cache.access_entry(cache_key)
            else:
                # Execute this block
                if self.verbose:
                    print(f"Executing block {block_id} (key: {cache_key})")

                result = self.execute_block(block, filename)
                # Mark this as a cache miss
                result.cache_hit = False
                result.content_changed = block_id in content_changed

                # Store in cache
                self.cache_by_key[cache_key] = result
                result.cache_key = cache_key
                results.append((block, result))
                # Add to cache manager
                if self.current_file:
                    size = self.block_cache.estimate_entry_size(result)
                    self.block_cache.add_entry(cache_key, self.current_file, size)

            # Update block state
            self.block_states[block_id] = BlockState(
                content_hash=content_hashes[block_id],
                cache_key=cache_key,
                result=result,
                dependencies=set(block.interface.requires),
                dependents=set(),  # Will be updated by graph
            )

        # Sort results back to document order
        block_order = {str(block.id): i for i, block in enumerate(document.blocks)}
        results.sort(key=lambda x: block_order.get(str(x[0].id), 0))

        return results

    def clear_cache(self):
        """Clear all cached execution results."""
        self.block_states.clear()
        self.cache_by_key.clear()
        # Cache manager keeps its own copy, so we don't clear it here

    def clear_file_cache(self, file_path: str):
        """Clear all cache entries for a specific file.

        This is called when a file's dependencies change.
        """
        # Clear from cache manager (which tracks by file)
        self.block_cache.clear_file_cache(file_path)

        # Also remove from our local cache
        # We need to track which cache keys belong to which file
        # For now, clear entries that are no longer in cache manager
        for cache_key in list(self.cache_by_key.keys()):
            if cache_key not in self.block_cache.entries:
                del self.cache_by_key[cache_key]

    def mark_file_for_eviction(self, file_path: str):
        """Mark a file's cache entries for potential eviction."""
        self.block_cache.mark_file_for_eviction(file_path)

    def unmark_file_for_eviction(self, file_path: str):
        """Remove eviction mark from a file."""
        self.block_cache.unmark_file_for_eviction(file_path)

    def evict_unwatched_files(self, force: bool = False):
        """Evict cache entries from unwatched files."""
        self.block_cache.evict_marked_files(force=force)

        # Also remove from our local cache
        for cache_key in list(self.cache_by_key.keys()):
            if cache_key not in self.block_cache.entries:
                del self.cache_by_key[cache_key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.block_cache.get_stats()

    def get_dependent_blocks(self, block_id: str) -> Set[str]:
        """Get all blocks that depend on the given block."""
        if self.block_graph:
            return set(self.block_graph.dirty_after({block_id}))
        return set()

    def get_dirty_blocks(self, document: Document) -> Set[str]:
        """Determine which blocks in the document need re-execution.

        Returns a set of block IDs that are dirty (need re-execution).
        """
        # Build dependency graph
        temp_graph = BlockGraph()
        block_dicts = []
        for block in document.blocks:
            block_dict = {
                "id": str(block.id),
                "interface": {
                    "provides": block.interface.provides,
                    "requires": block.interface.requires,
                },
            }
            block_dicts.append(block_dict)
        temp_graph.add_blocks(block_dicts)

        # Determine which blocks need re-execution
        blocks_to_execute = set()

        for block in document.blocks:
            block_id = str(block.id)
            content_hash = self._get_content_hash(block)

            # Check if block is new or changed
            if (
                block_id not in self.block_states
                or self.block_states[block_id].content_hash != content_hash
                or self._has_always_eval_pragma(block)
            ):
                blocks_to_execute.add(block_id)

        # Find all downstream dependencies
        all_dirty = set(temp_graph.dirty_after(blocks_to_execute))

        return all_dirty

    def execute_incremental_streaming(
        self,
        document: Document,
        changed_blocks: Optional[Set[str]] = None,
        filename: str = "<string>",
        source_file: Optional[str] = None,
    ):
        """Execute blocks incrementally in document order, yielding results as they complete.

        This method executes blocks in document order (not dependency order) to maintain
        the appearance of top-to-bottom execution while still using caching for efficiency.

        Args:
            document: The document to execute
            changed_blocks: Set of block IDs that have changed (None = all changed)
            filename: The filename for error reporting
            source_file: The source file path for cache tracking

        Yields:
            Tuples of (block, result) in document order as they are executed
        """
        # Set current file for cache tracking
        self.current_file = source_file

        # Set project root for file dependency resolution
        if source_file:
            project_root = None
            for parent in pathlib.Path(source_file).parents:
                if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                    project_root = str(parent)
                    break
            self.project_root = project_root

        # Build dependency graph
        self.block_graph = BlockGraph()
        block_dicts = []
        for block in document.blocks:
            block_dict = {
                "id": str(block.id),
                "interface": {
                    "provides": block.interface.provides,
                    "requires": block.interface.requires,
                },
            }
            block_dicts.append(block_dict)
        self.block_graph.add_blocks(block_dicts)

        # Determine which blocks need re-execution (same as before)
        blocks_to_execute = set()

        if changed_blocks is None:
            for block in document.blocks:
                block_id = str(block.id)
                content_hash = self._get_content_hash(block)

                if (
                    block_id not in self.block_states
                    or self.block_states[block_id].content_hash != content_hash
                    or self._has_always_eval_pragma(block)
                ):
                    blocks_to_execute.add(block_id)
        else:
            blocks_to_execute = changed_blocks.copy()

        # Add always-eval blocks
        for block in document.blocks:
            if self._has_always_eval_pragma(block):
                blocks_to_execute.add(str(block.id))

        # Find all downstream dependencies
        all_dirty = set(self.block_graph.dirty_after(blocks_to_execute))

        if self.verbose:
            print(f"Executing {len(all_dirty)} blocks out of {len(document.blocks)}")

        # Compute content hashes and dependency keys for all blocks
        content_hashes = {}
        content_changed = set()
        for block in document.blocks:
            block_id = str(block.id)
            new_hash = self._get_content_hash(block)
            content_hashes[block_id] = new_hash

            if block_id in self.block_states:
                if self.block_states[block_id].content_hash != new_hash:
                    content_changed.add(block_id)
            else:
                content_changed.add(block_id)

        # Process blocks in document order (not dependency order!)
        dependency_keys = {}

        for block in document.blocks:
            block_id = str(block.id)

            # Always compute cache key - it includes file modification times
            # which may have changed even if block content hasn't
            cache_key = self._compute_cache_key(block, dependency_keys)

            dependency_keys[block_id] = cache_key

            # Check if we have a cached result
            if cache_key in self.cache_by_key and block_id not in all_dirty:
                # Reuse cached result
                if self.verbose:
                    print(f"Cache hit for block {block_id} (key: {cache_key})")
                result = self.cache_by_key[cache_key]
                result.cache_hit = True
                result.content_changed = block_id in content_changed
                # Record cache access
                self.block_cache.access_entry(cache_key)
            else:
                # Execute this block
                if self.verbose:
                    print(f"Executing block {block_id} (key: {cache_key})")

                result = self.execute_block(block, filename)
                result.cache_hit = False
                result.content_changed = block_id in content_changed

                # Store in cache
                self.cache_by_key[cache_key] = result
                # Add to cache manager
                if self.current_file:
                    size = self.block_cache.estimate_entry_size(result)
                    self.block_cache.add_entry(cache_key, self.current_file, size)

            # Update block state
            self.block_states[block_id] = BlockState(
                content_hash=content_hashes[block_id],
                cache_key=cache_key,
                result=result,
                dependencies=set(block.interface.requires),
                dependents=set(),
            )

            # Add cache key to result for use as block ID
            result.cache_key = cache_key

            # Yield the result immediately
            yield block, result


class DocumentExecutor:
    """High-level document executor with incremental support."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.incremental_executor = IncrementalExecutor(verbose)

    def execute(
        self,
        document: Document,
        filename: str = "<string>",
        incremental: bool = True,
        changed_blocks: Optional[Set[str]] = None,
    ) -> Tuple[List[ExecutionResult], List[str]]:
        """Execute a document, optionally incrementally.

        Args:
            document: The document to execute
            filename: The filename for error reporting
            incremental: Whether to use incremental execution
            changed_blocks: Set of changed block IDs (for incremental mode)

        Returns:
            Tuple of (results, errors) where results are in document order
        """
        if incremental:
            block_results = self.incremental_executor.execute_incremental(
                document, changed_blocks, filename
            )
            results = [result for _, result in block_results]
        else:
            # Full execution using base executor
            executor = BlockExecutor(self.verbose)
            results = executor.execute_document(document, filename)

        # Collect any errors
        errors = []
        for i, result in enumerate(results):
            if result.error:
                errors.append(f"Block {i}: {result.error}")

        return results, errors
