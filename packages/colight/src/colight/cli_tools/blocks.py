"""Block graph introspection: stable IDs, line ranges, dependency edges.

Stable block IDs
----------------
The runtime's internal block ids are *transitive cache keys*: a hash of the
block's source, the cache keys of its dependency providers, and the mtimes of
imported local files (see ``parser._compute_block_cache_key``). Those keys
change whenever an upstream block changes, so they cannot serve as a stable
identity across edits.

The CLI therefore derives stable ids as short hashes of the block's own
normalized source (``hash_block_content``), disambiguated with a ``-N``
suffix when identical blocks repeat. A block's id survives edits to *other*
blocks; the transitive cache key is still used (as ``cache_key``) to decide
whether a block needs re-execution.
"""

import pathlib
from typing import Any, Dict, List, Optional, Tuple

from colight.runtime.model import Block, Document
from colight.runtime.parser import parse_colight_file
from colight.runtime.utils import hash_block_content

BlockPairs = List[Tuple[Block, str]]


def assign_stable_ids(document: Document) -> BlockPairs:
    """Assign content-stable ids to every block in the document.

    Args:
        document: Parsed colight document.

    Returns:
        List of (block, stable_id) pairs in document order.
    """
    counts: Dict[str, int] = {}
    pairs: BlockPairs = []
    for block in document.blocks:
        digest = hash_block_content(block)[:12]
        occurrence = counts.get(digest, 0) + 1
        counts[digest] = occurrence
        stable_id = digest if occurrence == 1 else f"{digest}-{occurrence}"
        pairs.append((block, stable_id))
    return pairs


def pair_by_stable_id(
    current_ids: List[str], previous: List[Dict[str, Any]]
) -> Tuple[List[Optional[Dict[str, Any]]], List[Dict[str, Any]]]:
    """Pair current stable ids with previous entries (dicts with an ``id``).

    Exact id matches win first; leftover current ids are then paired with
    leftover previous entries in order, so an *edited* block (whose stable
    id changed with its source) keeps its identity instead of degrading to
    "new" + "removed".

    Args:
        current_ids: Current stable ids in document order.
        previous: Previous entries (each with an ``id`` key) in order.

    Returns:
        Tuple of (per-current-id matches, ``None`` where unpaired; orphaned
        previous entries).
    """
    previous_by_id = {entry["id"]: entry for entry in previous}
    matches: List[Optional[Dict[str, Any]]] = [
        previous_by_id.get(sid) for sid in current_ids
    ]
    used = {id(entry) for entry in matches if entry is not None}
    leftovers = iter(entry for entry in previous if id(entry) not in used)
    for index, entry in enumerate(matches):
        if entry is None:
            entry = next(leftovers, None)
            if entry is None:
                break
            matches[index] = entry
            used.add(id(entry))
    orphaned = [entry for entry in previous if id(entry) not in used]
    return matches, orphaned


def block_lines(block: Block) -> Tuple[int, int]:
    """Compute the (start, end) line range of a block (1-based, inclusive)."""
    end = block.start_line
    for elem in block.elements:
        source = elem.get_source()
        line_count = max(1, len(source.splitlines())) if source else 1
        end = max(end, elem.lineno + line_count - 1)
    return block.start_line, end


def block_kind(block: Block) -> str:
    """Classify a block as ``prose``, ``code`` or ``both``."""
    has_code = bool(block.get_code_elements())
    has_prose = bool(block.get_prose_elements())
    if has_code and has_prose:
        return "both"
    return "code" if has_code else "prose"


def dependency_edges(
    pairs: BlockPairs,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build dependency edges between blocks using stable ids.

    Uses the same last-provider-in-document-order semantics as the runtime's
    cache key computation.

    Args:
        pairs: (block, stable_id) pairs in document order.

    Returns:
        Tuple of (depends_on, dependents) adjacency maps keyed by stable id.
    """
    depends_on: Dict[str, List[str]] = {sid: [] for _, sid in pairs}
    dependents: Dict[str, List[str]] = {sid: [] for _, sid in pairs}
    last_provider: Dict[str, str] = {}
    for block, sid in pairs:
        upstream: List[str] = []
        for symbol in block.interface.requires:
            provider = last_provider.get(symbol)
            if provider is not None and provider != sid and provider not in upstream:
                upstream.append(provider)
        depends_on[sid] = upstream
        for provider in upstream:
            dependents[provider].append(sid)
        for symbol in block.interface.provides:
            last_provider[symbol] = sid
    return depends_on, dependents


def transitive_dependents(dependents: Dict[str, List[str]], root: str) -> List[str]:
    """All blocks downstream of ``root`` (excluding ``root``), in stable order."""
    seen: List[str] = []
    stack = list(dependents.get(root, []))
    while stack:
        current = stack.pop(0)
        if current in seen:
            continue
        seen.append(current)
        stack.extend(dependents.get(current, []))
    return seen


def block_infos(document: Document, pairs: BlockPairs) -> List[Dict[str, Any]]:
    """Build per-block info dicts (shared by ``blocks`` and ``run``)."""
    depends_on, dependents = dependency_edges(pairs)
    infos: List[Dict[str, Any]] = []
    for block, sid in pairs:
        start, end = block_lines(block)
        infos.append(
            {
                "id": sid,
                "cache_key": block.id,
                "lines": [start, end],
                "kind": block_kind(block),
                "provides": list(block.interface.provides),
                "requires": list(block.interface.requires),
                "depends_on": depends_on[sid],
                "dependents": dependents[sid],
                "pragma": sorted(block.tags.flags),
                "ends_with_expression": block.has_expression_result,
            }
        )
    return infos


def describe_file(file_path: pathlib.Path) -> Dict[str, Any]:
    """Produce the ``colight blocks`` payload for a file.

    Args:
        file_path: Path to a notebook-style ``.py`` file.

    Returns:
        Dict with ``file``, ``pragma`` and ``blocks`` keys.
    """
    document = parse_colight_file(file_path)
    pairs = assign_stable_ids(document)
    return {
        "file": str(file_path),
        "pragma": sorted(document.tags.flags),
        "blocks": block_infos(document, pairs),
    }
