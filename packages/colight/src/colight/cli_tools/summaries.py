"""Token-frugal summaries and stable fingerprints for block results.

Summaries never serialize full data: arrays become dtype/shape (+ min/max),
scalars become truncated reprs, visuals become component types + counts.

Fingerprints are content hashes of a *canonicalized* result so that two runs
producing the same output hash identically even though widget/state ids are
regenerated on every run.
"""

import hashlib
import io
import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np

import colight.format as colight_format

# Ids regenerated on every run (widget ids, uuid state keys) must not leak
# into fingerprints. The pattern lives beside the mint sites in colight.ids.
from colight.ids import VOLATILE_ID_RE
from colight.runtime.executor import ExecutionResult

from .structure import component_label, iter_component_paths

MAX_REPR = 120


def truncated_repr(value: Any, limit: int = MAX_REPR) -> str:
    """Return ``repr(value)`` truncated to ``limit`` characters."""
    try:
        text = repr(value)
    except Exception:
        text = f"<unreprable {type(value).__name__}>"
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def array_stats(values: np.ndarray) -> Dict[str, Any]:
    """Cheap statistics for a numeric array (min/max/nan/inf counts).

    Args:
        values: Array to summarize.

    Returns:
        Dict with any of ``min``, ``max``, ``nan``, ``inf`` that apply.
    """
    stats: Dict[str, Any] = {}
    if values.size == 0 or not np.issubdtype(values.dtype, np.number):
        return stats
    data = np.asarray(values)
    if np.issubdtype(data.dtype, np.complexfloating):
        return stats
    finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data
    if finite.size:
        stats["min"] = float(np.min(finite))
        stats["max"] = float(np.max(finite))
    if np.issubdtype(data.dtype, np.floating):
        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
        if nan_count:
            stats["nan"] = nan_count
        if inf_count:
            stats["inf"] = inf_count
    return stats


def summarize_value(value: Any) -> Dict[str, Any]:
    """Summarize an arbitrary Python value without serializing its data.

    Args:
        value: The value produced by a block's trailing expression.

    Returns:
        A small JSON-safe dict; always has a ``kind`` key.
    """
    if value is None:
        return {"kind": "none"}
    if isinstance(value, np.ndarray):
        summary: Dict[str, Any] = {
            "kind": "array",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        summary.update(array_stats(value))
        return summary
    if isinstance(value, (bool, int, float, str)):
        return {
            "kind": "scalar",
            "type": type(value).__name__,
            "repr": truncated_repr(value),
        }
    if isinstance(value, (list, tuple, set, dict)):
        return {
            "kind": type(value).__name__,
            "length": len(value),
            "repr": truncated_repr(value),
        }
    return {
        "kind": "object",
        "type": f"{type(value).__module__}.{type(value).__name__}",
        "repr": truncated_repr(value),
    }


def parse_colight_bytes(data: bytes) -> Tuple[Dict[str, Any], List[bytes]]:
    """Parse in-memory ``.colight`` bytes into (json_data, buffers)."""
    json_data, buffers, _size = colight_format.parse_entry(io.BytesIO(data))
    return json_data, buffers


_MASKED_ID = "<id>"


def _mask_ids(text: str) -> str:
    return VOLATILE_ID_RE.sub(_MASKED_ID, text)


def _collect_volatile_ids(node: Any, path: str, first_paths: Dict[str, str]) -> None:
    """Map each volatile id to the masked path of its first occurrence.

    Paths use dict keys (volatile ids masked) and ``[*]`` for list indices,
    so an id's first-use path does not shift when siblings are inserted.
    Dict insertion order makes the traversal deterministic.
    """
    if isinstance(node, str):
        for match in VOLATILE_ID_RE.findall(node):
            first_paths.setdefault(match, path)
    elif isinstance(node, dict):
        for k, v in node.items():
            _collect_volatile_ids(k, path, first_paths)
            child = f"{path}.{_mask_ids(k)}" if isinstance(k, str) else f"{path}.?"
            _collect_volatile_ids(v, child, first_paths)
    elif isinstance(node, (list, tuple)):
        for item in node:
            _collect_volatile_ids(item, f"{path}[*]", first_paths)


def _volatile_id_mapping(trimmed: Dict[str, Any]) -> Dict[str, str]:
    """Assign a stable placeholder to every volatile id in the payload.

    The placeholder is ``id-<hash>`` where the hash covers the id's
    first-occurrence *context*: the shallow kind of its state entry (its
    component label, e.g. ``MarkSpec:dot``, or its value type) plus the
    masked path of its first use. That context is stable both when *other*
    stateful nodes (Refs, widgets, MarkSpec state) are inserted or removed
    and when the entry's own data is edited — so diffs neither report
    unrelated state keys as removed+added nor lose pairing on content
    changes. Ids with identical context are disambiguated by
    first-occurrence order, which keeps the mapping deterministic for
    fingerprinting.
    """
    first_paths: Dict[str, str] = {}
    _collect_volatile_ids(trimmed, "", first_paths)
    state = trimmed.get("state")
    state = state if isinstance(state, dict) else {}

    mapping: Dict[str, str] = {}
    used: Dict[str, int] = {}
    for volatile_id, first_path in first_paths.items():
        kind = ""
        if volatile_id in state:
            value = state[volatile_id]
            if isinstance(value, dict):
                kind = component_label(value) or "dict"
            elif isinstance(value, (list, tuple)):
                kind = "list"
            else:
                kind = type(value).__name__
        signature = f"{kind}|{first_path}"
        base = f"id-{hashlib.sha256(signature.encode()).hexdigest()[:8]}"
        count = used.get(base, 0) + 1
        used[base] = count
        mapping[volatile_id] = base if count == 1 else f"{base}-{count}"
    return mapping


def _replace_volatile_ids(node: Any, mapping: Dict[str, str]) -> Any:
    """Rewrite volatile ids using the collected mapping."""
    if isinstance(node, str):
        return VOLATILE_ID_RE.sub(lambda m: mapping[m.group(0)], node)
    if isinstance(node, dict):
        return {
            _replace_volatile_ids(k, mapping): _replace_volatile_ids(v, mapping)
            for k, v in node.items()
        }
    if isinstance(node, (list, tuple)):
        return [_replace_volatile_ids(item, mapping) for item in node]
    return node


def canonicalize_visual_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize a visual's JSON payload (as data, not text).

    Drops per-run identifiers (top-level ``id``, ``bufferLayout``) and maps
    generated uuid state keys / widget ids to stable content-derived
    placeholders (see :func:`_volatile_id_mapping`). Buffer index references
    are left untouched.
    """
    trimmed = {k: v for k, v in data.items() if k not in ("id", "bufferLayout")}
    return _replace_volatile_ids(trimmed, _volatile_id_mapping(trimmed))


def canonicalize_visual_json(data: Dict[str, Any]) -> str:
    """Canonicalize a visual's JSON payload for fingerprinting."""
    return json.dumps(
        canonicalize_visual_data(data), sort_keys=True, separators=(",", ":")
    )


def visual_fingerprint(colight_bytes: bytes) -> str:
    """Fingerprint a visual: canonical JSON + buffer contents."""
    data, buffers = parse_colight_bytes(colight_bytes)
    hasher = hashlib.sha256()
    hasher.update(canonicalize_visual_json(data).encode())
    for buffer in buffers:
        hasher.update(hashlib.sha256(bytes(buffer)).digest())
    return hasher.hexdigest()[:16]


def summarize_visual(colight_bytes: bytes) -> Dict[str, Any]:
    """Summarize a visual as component types + counts (no data)."""
    data, buffers = parse_colight_bytes(colight_bytes)
    paths = iter_component_paths({"ast": data.get("ast"), "state": data.get("state")})
    counts: Dict[str, int] = {}
    for path in paths:
        counts[path] = counts.get(path, 0) + 1
    return {
        "kind": "visual",
        "components": [{"path": p, "count": n} for p, n in counts.items()],
        "buffers": len(buffers),
        "size": len(colight_bytes),
    }


def summarize_result(result: ExecutionResult) -> Dict[str, Any]:
    """Summarize a block's execution result (token-frugal)."""
    if result.error:
        info = result.error_info or {}
        return {
            "kind": "error",
            "type": info.get("type", "Exception"),
            "message": truncated_repr(info.get("message", ""), 200).strip("'\""),
        }
    if result.colight_bytes is not None:
        return summarize_visual(result.colight_bytes)
    return summarize_value(result.value)


def result_fingerprint(result: ExecutionResult) -> str:
    """Stable fingerprint of a block's observable output.

    Combines error (type+message), stdout, and either the canonicalized
    visual or the value summary.
    """
    hasher = hashlib.sha256()
    if result.error:
        info = result.error_info or {}
        hasher.update(f"error:{info.get('type')}:{info.get('message')}".encode())
    hasher.update(b"stdout:" + result.output.encode())
    if result.colight_bytes is not None:
        hasher.update(b"visual:" + visual_fingerprint(result.colight_bytes).encode())
    elif isinstance(result.value, np.ndarray):
        array = np.ascontiguousarray(result.value)
        hasher.update(f"ndarray:{array.dtype}:{array.shape}:".encode())
        hasher.update(array.tobytes())
    elif result.value is not None:
        # Strip memory addresses so default object reprs stay stable across runs.
        text = re.sub(r"0x[0-9a-fA-F]+", "0x0", truncated_repr(result.value, 100_000))
        hasher.update(b"value:" + text.encode())
    return hasher.hexdigest()[:16]


__all__ = [
    "array_stats",
    "canonicalize_visual_data",
    "canonicalize_visual_json",
    "parse_colight_bytes",
    "result_fingerprint",
    "summarize_result",
    "summarize_value",
    "summarize_visual",
    "truncated_repr",
    "visual_fingerprint",
]
