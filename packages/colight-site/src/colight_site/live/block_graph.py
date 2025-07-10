"""Block dependency graph for topological execution ordering."""

from typing import Dict, List, Set
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class BlockInterface:
    """Interface information for a block."""

    provides: List[str]  # Symbols this block provides
    requires: List[str]  # Symbols this block requires


class BlockGraph:
    """Dependency graph for code blocks in a document."""

    def __init__(self):
        self.blocks: Dict[str, Dict] = {}  # block_id -> block data
        self.provides: Dict[str, Set[str]] = {}  # block_id -> provided symbols
        self.requires: Dict[str, Set[str]] = {}  # block_id -> required symbols
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # block -> dependents
        self.reverse_edges: Dict[str, Set[str]] = defaultdict(
            set
        )  # block -> dependencies
        self.symbol_providers: Dict[
            str, str
        ] = {}  # symbol -> block_id that provides it

    def add_blocks(self, blocks: List[Dict]) -> None:
        """Add blocks to the graph and build dependency edges.

        Args:
            blocks: List of block dictionaries with 'id' and 'interface' fields
        """
        # Clear existing data
        self.blocks.clear()
        self.provides.clear()
        self.requires.clear()
        self.edges.clear()
        self.reverse_edges.clear()
        self.symbol_providers.clear()

        # First pass: collect all blocks and their interfaces
        for block in blocks:
            block_id = block.get("id")
            if not block_id:
                continue

            self.blocks[block_id] = block

            # Extract interface information
            interface = block.get("interface", {})
            provides = set(interface.get("provides", []))
            requires = set(interface.get("requires", []))

            self.provides[block_id] = provides
            self.requires[block_id] = requires

        # Sort blocks by ID to ensure proper order
        ordered_blocks = sorted(self.blocks.keys(), key=lambda x: int(x))

        # Second pass: build dependency edges respecting execution order
        for i, block_id in enumerate(ordered_blocks):
            required_symbols = self.requires.get(block_id, set())

            for symbol in required_symbols:
                # Find the most recent provider ABOVE this block
                provider = None
                for j in range(i):  # Only look at blocks before this one
                    earlier_block = ordered_blocks[j]
                    if symbol in self.provides.get(earlier_block, set()):
                        provider = earlier_block  # Keep the last (most recent) provider

                if provider:
                    # This block depends on the provider
                    self.reverse_edges[block_id].add(provider)
                    # The provider has this block as a dependent
                    self.edges[provider].add(block_id)

    def execution_order(self) -> List[str]:
        """Return topological sort of all blocks.

        Returns:
            List of block IDs in execution order.
            If cycles exist, returns blocks in original order.
        """
        # Kahn's algorithm for topological sort with stable ordering
        in_degree = {
            bid: len(self.reverse_edges.get(bid, set())) for bid in self.blocks
        }

        # Start with blocks that have no dependencies, sorted by ID to maintain order
        initial_blocks = [bid for bid, degree in in_degree.items() if degree == 0]
        initial_blocks.sort(key=lambda x: int(x))
        queue = deque(initial_blocks)
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # Process all dependents of current block, maintaining order
            dependents = list(self.edges.get(current, set()))
            dependents.sort(
                key=lambda x: int(x)
            )  # Sort by ID to maintain document order

            for dependent in dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Check if we processed all blocks (no cycles)
        if len(result) != len(self.blocks):
            # Cycle detected - return blocks in original order
            # Could also raise an exception or return cycle information
            return sorted(self.blocks.keys(), key=lambda x: int(x))

        return result

    def dirty_after(self, changed_ids: Set[str]) -> List[str]:
        """Return blocks that must be re-executed after changes.

        Args:
            changed_ids: Set of block IDs that have changed

        Returns:
            List of block IDs that need re-execution, in topological order
        """
        # Find all blocks affected by the changes
        dirty = set(changed_ids)
        stack = list(changed_ids)

        while stack:
            current = stack.pop()
            # Find all blocks that depend on the current block
            for dependent in self.edges.get(current, set()):
                if dependent not in dirty:
                    dirty.add(dependent)
                    stack.append(dependent)

        # Return dirty blocks in execution order
        execution_order = self.execution_order()
        return [bid for bid in execution_order if bid in dirty]

    def find_circular_dependencies(self) -> List[List[str]]:
        """Find all circular dependency chains.

        Returns:
            List of cycles, where each cycle is a list of block IDs
        """
        # Use DFS with recursion stack to detect cycles
        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Check all dependencies
            for neighbor in self.reverse_edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)

        # Check each component
        for node in self.blocks:
            if node not in visited:
                dfs(node, [])

        return cycles

    def get_dependencies(self, block_id: str) -> Set[str]:
        """Get all blocks that a given block depends on.

        Args:
            block_id: Block to check

        Returns:
            Set of block IDs that the given block depends on
        """
        return self.reverse_edges.get(block_id, set())

    def get_dependents(self, block_id: str) -> Set[str]:
        """Get all blocks that depend on a given block.

        Args:
            block_id: Block to check

        Returns:
            Set of block IDs that depend on the given block
        """
        return self.edges.get(block_id, set())

    def get_undefined_symbols(self) -> Dict[str, Set[str]]:
        """Find symbols that are required but not provided.

        Returns:
            Dict mapping block_id to set of undefined symbols
        """
        undefined = {}

        for block_id, required_symbols in self.requires.items():
            missing = set()
            for symbol in required_symbols:
                if symbol not in self.symbol_providers:
                    missing.add(symbol)
            if missing:
                undefined[block_id] = missing

        return undefined

    def to_dict(self) -> Dict:
        """Convert graph to dictionary for serialization."""
        return {
            "blocks": list(self.blocks.keys()),
            "edges": {k: list(v) for k, v in self.edges.items()},
            "provides": {k: list(v) for k, v in self.provides.items()},
            "requires": {k: list(v) for k, v in self.requires.items()},
            "symbol_providers": self.symbol_providers,
        }
