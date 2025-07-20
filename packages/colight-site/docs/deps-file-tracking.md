# Minimal File-Level Block Dependency Tracking

## Executive Summary

This document outlines a minimal approach to improve cross-module dependency tracking by having blocks track which external files they depend on. This preserves the current high-performance architecture where only watched files are parsed block-by-block, while providing more granular invalidation.

## Current Problem

When file B changes, ALL blocks in file A that imports from B are re-executed, even if they don't use anything from B:

```python
# module_a.py
from .module_b import x  # Block 0 - depends on module_b.py

x + 1  # Block 1 - uses x, indirectly depends on module_b.py

y = 42  # Block 2 - doesn't depend on module_b.py at all!
```

Currently: ALL blocks re-execute when module_b.py changes.
Goal: Only blocks 0 and 1 re-execute.

## Proposed Solution

### Core Concept

Enhance `BlockInterface` to track file dependencies:

```python
@dataclass
class BlockInterface:
    provides: List[str]
    requires: List[str]
    # NEW: External files this block depends on
    file_dependencies: Set[str] = field(default_factory=set)
```

### Implementation

#### 1. Enhanced Dependency Analysis

Modify `dependency_analyzer.py` to detect import statements and track file dependencies:

```python
class DependencyVisitor(libcst.CSTVisitor):
    def visit_Import(self, node: libcst.Import) -> None:
        # Track that this block depends on the imported module
        for name in node.names:
            module_name = name.name.value
            file_path = self._resolve_module_to_file(module_name)
            if file_path:
                self.file_dependencies.add(file_path)

    def visit_ImportFrom(self, node: libcst.ImportFrom) -> None:
        # Track file dependency from 'from X import Y' statements
        if node.module:
            module_name = self._get_module_name(node)
            file_path = self._resolve_module_to_file(module_name,
                                                     node.relative)
            if file_path:
                self.file_dependencies.add(file_path)
```

#### 2. Module Resolution

Add a lightweight module resolver that:

- Resolves module names to file paths
- Handles relative imports
- Only tracks files within the project (ignores stdlib/third-party)

```python
def resolve_module_to_file(module_name: str,
                          current_file: str,
                          is_relative: bool = False) -> Optional[str]:
    """Resolve a module name to a file path within the project."""
    # Implementation would:
    # 1. Handle relative imports based on current_file
    # 2. Try .py file, then /__init__.py for packages
    # 3. Return None for external modules
    # 4. Return relative path from project root
```

#### 3. Block Cache Key Enhancement

Modify cache key generation to include file dependency hashes:

```python
def _compute_cache_key(self, block_id: str, content: str,
                      dependencies: Dict[str, str],
                      file_dependencies: Set[str]) -> str:
    """Compute cache key including file dependencies."""
    components = [content]

    # Existing: dependency cache keys
    for dep_id in sorted(dependencies.keys()):
        components.append(f"{dep_id}:{dependencies[dep_id]}")

    # NEW: Add file modification times/hashes
    for file_path in sorted(file_dependencies):
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            components.append(f"file:{file_path}:{mtime}")

    combined = "\n".join(components)
    return hashlib.sha256(combined.encode()).hexdigest()
```

#### 4. Incremental Executor Changes

Update `IncrementalExecutor` to:

1. Pass current file path to dependency analyzer
2. Include file dependencies in block interfaces
3. Mark blocks dirty when their file dependencies change

```python
def _analyze_block(self, block: Block, current_file: str) -> BlockInterface:
    """Analyze a block's dependencies including file dependencies."""
    provides, requires = analyze_block(block.get_code())
    file_deps = analyze_file_dependencies(block.get_code(), current_file)

    return BlockInterface(
        provides=list(provides),
        requires=list(requires),
        file_dependencies=file_deps
    )
```

### Benefits

1. **Minimal Changes**: Preserves existing architecture
2. **Performance**: Still only parse watched files block-by-block
3. **Better Granularity**: Only re-execute blocks that actually import from changed files
4. **Simple Implementation**: No need for cross-module symbol resolution

### Limitations (Accepted Trade-offs)

1. **Conservative**: If ANY part of an imported file changes, dependent blocks re-execute
2. **No Symbol Tracking**: Can't tell if a block uses specific symbols from imports
3. **Transitive Dependencies**: Block using imported symbols doesn't know about transitive file deps

These limitations are acceptable because:

- Most import statements are at the top of files (few blocks affected)
- Still much better than current file-level invalidation
- Maintains high performance for large codebases

### Implementation Steps

#### Phase 1: Dependency Detection (2-3 days)

1. Enhance `DependencyVisitor` to track import statements
2. Implement module-to-file resolution
3. Add `file_dependencies` to `BlockInterface`

#### Phase 2: Cache Integration (1-2 days)

1. Update cache key computation to include file dependencies
2. Add file modification tracking
3. Test cache invalidation

#### Phase 3: Executor Integration (1-2 days)

1. Update `IncrementalExecutor` to populate file dependencies
2. Mark blocks dirty when file dependencies change
3. Integration testing

### Example Walkthrough

Given our test case:

```python
# module_a.py
from .module_b import x  # Block 0
x + 1                    # Block 1
y = 42                   # Block 2
```

After analysis:

- Block 0: `file_dependencies = {"module_b.py"}`, `provides = ["x"]`
- Block 1: `file_dependencies = {}`, `requires = ["x"]`
- Block 2: `file_dependencies = {}`, `requires = []`

When module_b.py changes:

1. File watcher detects change
2. Executor checks each block's file dependencies
3. Block 0 is marked dirty (has module_b.py in file_dependencies)
4. Block 1 is marked dirty (depends on block 0 via 'x')
5. Block 2 remains clean (no dependencies)

### Testing Strategy

1. **Unit Tests**:

   - Import detection in various forms
   - Module resolution logic
   - File dependency tracking

2. **Integration Tests**:

   - Multi-file projects with imports
   - Cache invalidation on file changes
   - Performance benchmarks

3. **Edge Cases**:
   - Circular imports
   - Missing files
   - Dynamic imports (marked as depending on all files)

### Success Metrics

1. **Correctness**: All existing tests pass
2. **Performance**:
   - 30-50% reduction in unnecessary block re-executions
   - Negligible overhead in dependency analysis (<5ms per file)
3. **Simplicity**: Implementation in <500 lines of code

## Conclusion

This minimal approach provides significant improvement in granularity without architectural changes. By tracking file-level dependencies per block, we can avoid re-executing blocks that don't import from changed files, while maintaining the current high-performance design that only parses watched files in detail.
