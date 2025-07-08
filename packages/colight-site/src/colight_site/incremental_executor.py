"""Incremental executor using dependency graph for smart re-execution."""

from typing import Dict, Optional, List, Set, Tuple
from dataclasses import dataclass, field
import hashlib

from .model import Block, Document
from .executor import ExecutionResult, BlockExecutor
from .block_graph import BlockGraph


@dataclass
class BlockState:
    """State of a block for incremental execution."""

    content_hash: str
    result: ExecutionResult
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)


class IncrementalExecutor(BlockExecutor):
    """Execute blocks incrementally based on dependency graph."""

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.block_states: Dict[str, BlockState] = {}
        self.block_graph: Optional[BlockGraph] = None

    def _get_content_hash(self, block: Block) -> str:
        """Get a hash of block's executable content."""
        content = block.get_code_text()
        return hashlib.sha256(content.encode()).hexdigest()

    def _has_always_eval_pragma(self, block: Block) -> bool:
        """Check if block has the always-eval pragma."""
        return "always-eval" in block.tags.flags

    def execute_incremental(
        self,
        document: Document,
        changed_blocks: Optional[Set[str]] = None,
        filename: str = "<string>",
    ) -> List[Tuple[Block, ExecutionResult]]:
        """Execute blocks incrementally based on changes.

        Args:
            document: The document to execute
            changed_blocks: Set of block IDs that have changed (None = all changed)
            filename: The filename for error reporting

        Returns:
            List of (block, result) tuples in document order
        """
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

        for block_id in execution_order:
            block = block_map.get(block_id)
            if not block:
                continue

            if block_id in all_dirty:
                # Execute this block
                if self.verbose:
                    print(f"Executing block {block_id}")

                result = self.execute_block(block, filename)

                # Update state
                self.block_states[block_id] = BlockState(
                    content_hash=self._get_content_hash(block),
                    result=result,
                    dependencies=set(block.interface.requires),
                    dependents=set(),  # Will be updated by graph
                )

                results.append((block, result))
            else:
                # Reuse cached result
                if block_id in self.block_states:
                    if self.verbose:
                        print(f"Reusing cached result for block {block_id}")
                    results.append((block, self.block_states[block_id].result))
                else:
                    # Shouldn't happen, but handle gracefully
                    if self.verbose:
                        print(
                            f"Warning: No cached result for block {block_id}, executing"
                        )
                    result = self.execute_block(block, filename)
                    self.block_states[block_id] = BlockState(
                        content_hash=self._get_content_hash(block), result=result
                    )
                    results.append((block, result))

        # Sort results back to document order
        block_order = {str(block.id): i for i, block in enumerate(document.blocks)}
        results.sort(key=lambda x: block_order.get(str(x[0].id), 0))

        return results

    def clear_cache(self):
        """Clear all cached execution results."""
        self.block_states.clear()

    def get_dependent_blocks(self, block_id: str) -> Set[str]:
        """Get all blocks that depend on the given block."""
        if self.block_graph:
            return set(self.block_graph.dirty_after({block_id}))
        return set()


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
