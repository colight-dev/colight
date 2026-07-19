"""Headless incremental evaluation with a persistent per-file run record.

Each ``colight run`` invocation executes a file's blocks and compares the
outcome against a compact fingerprint record persisted from the previous
invocation (in ``.colight_cache/cli-run/`` under the project root, the same
gitignored cache directory the live/publish servers use).

Because each invocation is a fresh process, runtime values cannot be carried
across runs; instead the runtime's *transitive cache keys* (source + upstream
sources + local import mtimes) decide which blocks are known-unchanged. Those
blocks are skipped entirely — mirroring the live server's cache-hit semantics
where side effects also do not re-run — unless an executed block needs the
symbols they provide, in which case they re-execute to rebuild the namespace.

Statuses:
    cached:        cache key matched the previous record; block not executed.
    ran:unchanged: executed; result fingerprint equals the previous one.
    ran:changed:   executed; result fingerprint differs.
    new:           executed; block has no counterpart in the previous record.
    removed:       block from the previous record no longer in the file.
    error:         executed and raised.
"""

import hashlib
import json
import pathlib
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from colight.runtime.executor import BlockExecutor, ExecutionResult
from colight.runtime.model import Block
from colight.runtime.parser import find_project_root, parse_colight_file

from . import blocks as blocks_mod
from . import summaries

RECORD_VERSION = 1
MAX_STDOUT = 2000

Pair = Tuple[Block, str]


def record_path(file_path: pathlib.Path, project_root: pathlib.Path) -> pathlib.Path:
    """Location of the persistent run record for ``file_path``.

    Records live in ``<project_root>/.colight_cache/cli-run/`` — the same
    cache directory convention used by ``colight publish --serve``.
    """
    digest = hashlib.sha256(str(file_path.resolve()).encode()).hexdigest()[:12]
    return (
        project_root / ".colight_cache" / "cli-run" / f"{file_path.stem}-{digest}.json"
    )


def load_record(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Load a previous run record, tolerating absence or corruption."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if record.get("version") != RECORD_VERSION:
        return None
    return record


def save_record(path: pathlib.Path, record: Dict[str, Any]) -> None:
    """Persist the run record, ensuring the cache dir exists and is gitignored."""
    cache_dir = path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    gitignore = cache_dir.parent / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")
    path.write_text(json.dumps(record, indent=1), encoding="utf-8")


def _match_previous(
    pairs: List[Pair], prev_entries: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Map current stable ids to previous record entries.

    Exact id matches first; leftover current/previous blocks are then paired
    in document order so an edited block keeps its identity (and can report
    ``ran:changed`` instead of ``new`` + ``removed``).
    """
    prev_by_id = {entry["id"]: entry for entry in prev_entries}
    matched: Dict[str, Dict[str, Any]] = {}
    used_prev_ids: Set[str] = set()
    for _block, sid in pairs:
        if sid in prev_by_id:
            matched[sid] = prev_by_id[sid]
            used_prev_ids.add(sid)
    unmatched_current = [sid for _b, sid in pairs if sid not in matched]
    unmatched_prev = [e for e in prev_entries if e["id"] not in used_prev_ids]
    for sid, entry in zip(unmatched_current, unmatched_prev):
        matched[sid] = entry
    return matched


def _plan_execution(pairs: List[Pair], matched: Dict[str, Dict[str, Any]]) -> Set[str]:
    """Decide which blocks must execute.

    A block is skippable when its transitive cache key matches its previous
    record entry and that entry did not record an error (errors are never
    cacheable results). Skippable blocks still run when an executed
    downstream block (transitively) requires a symbol they provide, since a
    fresh process has no namespace to reuse.

    Returns:
        Set of stable ids that will execute.
    """
    run_set: Set[str] = set()
    needed_symbols: Set[str] = set()
    for block, sid in reversed(pairs):
        prev = matched.get(sid)
        prev_errored = (
            prev is not None and (prev.get("summary") or {}).get("kind") == "error"
        )
        cache_stable = (
            prev is not None and prev.get("cache_key") == block.id and not prev_errored
        )
        must_run = (
            not cache_stable
            or "always-eval" in block.tags.flags
            or bool(set(block.interface.provides) & needed_symbols)
        )
        if must_run:
            run_set.add(sid)
            needed_symbols.update(block.interface.requires)
    return run_set


def _error_payload(result: ExecutionResult) -> Optional[Dict[str, Any]]:
    """Structured error info for JSON output."""
    if not result.error:
        return None
    if result.error_info:
        return result.error_info
    return {"type": "Exception", "message": result.error.strip(), "frames": []}


def run_file(
    file_path: pathlib.Path,
    focus_block: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a file's blocks and diff against the previous invocation.

    Args:
        file_path: Notebook-style ``.py`` file to execute.
        focus_block: Optional stable block id; full detail is reported only
            for that block and its transitive dependents (other blocks get
            one-line statuses).

    Returns:
        The ``colight run`` JSON payload (see the CLI ``--help`` for schema).

    Raises:
        ValueError: If ``focus_block`` is not a known block id.
    """
    file_path = file_path.resolve()
    project_root = find_project_root(file_path)
    document = parse_colight_file(file_path, project_root=project_root)
    pairs = blocks_mod.assign_stable_ids(document)
    infos = {info["id"]: info for info in blocks_mod.block_infos(document, pairs)}
    _depends_on, dependents = blocks_mod.dependency_edges(pairs)

    detail_ids: Optional[Set[str]] = None
    if focus_block is not None:
        if focus_block not in infos:
            known = ", ".join(sid for _b, sid in pairs)
            raise ValueError(f"Unknown block id: {focus_block}. Known ids: {known}")
        detail_ids = {
            focus_block,
            *blocks_mod.transitive_dependents(dependents, focus_block),
        }

    rec_path = record_path(file_path, project_root)
    previous = load_record(rec_path)
    prev_entries: List[Dict[str, Any]] = previous.get("blocks", []) if previous else []
    matched = _match_previous(pairs, prev_entries)
    run_set = _plan_execution(pairs, matched)

    executor = BlockExecutor()
    out_blocks: List[Dict[str, Any]] = []
    new_entries: List[Dict[str, Any]] = []
    error_count = 0

    for block, sid in pairs:
        info = infos[sid]
        prev = matched.get(sid)
        entry: Dict[str, Any] = {"id": sid, "lines": info["lines"]}

        if sid in run_set:
            started = time.monotonic()
            result = executor.execute_block(block, str(file_path))
            duration_ms = int((time.monotonic() - started) * 1000)
            fingerprint = summaries.result_fingerprint(result)
            summary = summaries.summarize_result(result)
            if result.error:
                status = "error"
                error_count += 1
            elif prev is None:
                status = "new"
            elif fingerprint != prev.get("fingerprint"):
                status = "ran:changed"
            else:
                status = "ran:unchanged"
            entry.update(status=status, executed=True, duration_ms=duration_ms)
            if detail_ids is None or sid in detail_ids:
                entry["summary"] = summary
                if result.output:
                    entry["stdout"] = result.output[:MAX_STDOUT]
                error = _error_payload(result)
                if error:
                    entry["error"] = error
            elif result.error:
                # Errors are always surfaced, even outside the focus set.
                entry["error"] = _error_payload(result)
        else:
            assert prev is not None  # skip only happens on a cache-key match
            fingerprint = prev.get("fingerprint")
            summary = prev.get("summary")
            entry.update(status="cached", executed=False)
            if (detail_ids is None or sid in detail_ids) and summary is not None:
                entry["summary"] = summary

        out_blocks.append(entry)
        new_entries.append(
            {
                "id": sid,
                "cache_key": block.id,
                "fingerprint": fingerprint,
                "summary": summary,
                "lines": info["lines"],
            }
        )

    # Previous blocks with no counterpart in the current document.
    matched_prev_ids = {id(entry) for entry in matched.values()}
    for entry in prev_entries:
        if id(entry) not in matched_prev_ids:
            out_blocks.append(
                {
                    "id": entry["id"],
                    "status": "removed",
                    "lines": entry.get("lines"),
                    "executed": False,
                }
            )

    save_record(
        rec_path,
        {
            "version": RECORD_VERSION,
            "file": str(file_path),
            "updated": time.time(),
            "blocks": new_entries,
        },
    )

    return {
        "file": str(file_path),
        "ok": error_count == 0,
        "errors": error_count,
        "blocks": out_blocks,
    }
