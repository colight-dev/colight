"""Semantic diff of colight visuals — change magnitude without rendering.

Compares two targets (each a ``.colight`` artifact or a ``.py`` file that is
evaluated headlessly) and reports, per visual: components added / removed /
type-changed, per-array shape/dtype changes and magnitude statistics
(max |Δ|, mean |Δ|, fraction of elements changed beyond epsilon, bounds
drift), scalar value changes, state-key changes, buffer count/bytes deltas
and the sanity-warning delta.

Everything is linear in the data already unpacked into memory; no rendering
and no quadratic work.
"""

import difflib
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import colight.format as colight_format

from . import inspect_tools, summaries

DEFAULT_EPSILON = 1e-9

_SKIP_KEYS = ("__type__", "path", "bufferLayout", "id")


def load_target(file_path: pathlib.Path) -> Dict[str, Any]:
    """Load a diff target into a uniform shape.

    Args:
        file_path: A ``.colight`` artifact or notebook-style ``.py`` file.

    Returns:
        Dict with ``file``, ``kind``, ``visuals`` (each with canonicalized
        ``data`` + ``buffers``, plus ``block``/``lines`` for ``.py``),
        ``errors`` and, for artifacts, ``updates``.

    Raises:
        ValueError: If the target has an unsupported extension or a
            ``.colight`` file has no initial state entry.
    """
    if file_path.suffix == ".colight":
        data, buffers, updates = colight_format.parse_file(file_path)
        if data is None:
            raise ValueError(f"file contains no initial state entry: {file_path}")
        return {
            "file": str(file_path),
            "kind": "colight",
            "updates": len(updates),
            "visuals": [{"data": data, "buffers": buffers}],
            "errors": [],
        }
    if file_path.suffix == ".py":
        visuals, errors = inspect_tools.evaluate_python_visuals(file_path)
        return {
            "file": str(file_path),
            "kind": "py",
            "visuals": visuals,
            "errors": errors,
        }
    raise ValueError(f"Unsupported target (expected .colight or .py): {file_path}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten_leaves(
    node: Any, path: str, key: Optional[str], out: Dict[str, Any]
) -> None:
    """Collect scalar leaves with paths, mirroring the array walk's skips.

    ndarray nodes and inline numeric lists are excluded — those are covered
    by the array diff.
    """
    if isinstance(node, dict):
        if node.get("__type__") == "ndarray" and "__buffer_index__" in node:
            return
        for k, v in node.items():
            if k in _SKIP_KEYS:
                continue
            _flatten_leaves(v, f"{path}.{k}" if path else k, k, out)
    elif isinstance(node, list):
        if key is not None and inspect_tools._coerce_inline_array(node) is not None:
            return
        for i, item in enumerate(node):
            _flatten_leaves(item, f"{path}[{i}]", None, out)
    else:
        out[path] = node


def _diff_components(
    a: List[Any], b: List[Any]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Match component sequences by label/position (difflib opcodes)."""
    added: List[Dict[str, str]] = []
    removed: List[Dict[str, str]] = []
    changed: List[Dict[str, str]] = []
    labels_a = [c.path for c in a]
    labels_b = [c.path for c in b]
    matcher = difflib.SequenceMatcher(a=labels_a, b=labels_b, autojunk=False)
    for op, a0, a1, b0, b1 in matcher.get_opcodes():
        if op == "equal":
            continue
        pairs = min(a1 - a0, b1 - b0) if op == "replace" else 0
        for offset in range(pairs):
            changed.append(
                {
                    "path": a[a0 + offset].display_path,
                    "from": labels_a[a0 + offset],
                    "to": labels_b[b0 + offset],
                }
            )
        for idx in range(a0 + pairs, a1):
            removed.append({"path": a[idx].display_path, "type": labels_a[idx]})
        for idx in range(b0 + pairs, b1):
            added.append({"path": b[idx].display_path, "type": labels_b[idx]})
    return added, removed, changed


def _numeric_delta(
    a: np.ndarray, b: np.ndarray, epsilon: float
) -> Optional[Dict[str, Any]]:
    """Magnitude stats for two same-shape numeric arrays; None if within epsilon."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    nan_a = np.isnan(av)
    nan_b = np.isnan(bv)
    both_nan = nan_a & nan_b
    delta = np.abs(bv - av)
    delta[both_nan] = 0.0
    nan_mismatch = nan_a ^ nan_b
    finite = np.isfinite(delta)
    changed_mask = (finite & (delta > epsilon)) | nan_mismatch | (~finite & ~both_nan)
    changed_count = int(np.count_nonzero(changed_mask))
    if changed_count == 0:
        return None
    finite_deltas = delta[finite]
    stats: Dict[str, Any] = {
        "changed_fraction": round(changed_count / delta.size, 6),
    }
    if finite_deltas.size:
        stats["max_abs_delta"] = float(np.max(finite_deltas))
        stats["mean_abs_delta"] = float(np.mean(finite_deltas))
    if int(np.count_nonzero(nan_mismatch)):
        stats["nan_mismatch"] = int(np.count_nonzero(nan_mismatch))
    return stats


def _bounds(values: Optional[np.ndarray]) -> Optional[List[float]]:
    if values is None:
        return None
    stats = summaries.array_stats(values)
    if "min" not in stats:
        return None
    return [stats["min"], stats["max"]]


def _diff_arrays(
    a_records: List[Any], b_records: List[Any], epsilon: float
) -> Dict[str, List[Dict[str, Any]]]:
    """Pair arrays by path (positionally within a path) and diff each pair."""

    def by_path(records: List[Any]) -> Dict[str, List[Any]]:
        grouped: Dict[str, List[Any]] = {}
        for record in records:
            grouped.setdefault(record.path, []).append(record)
        return grouped

    grouped_a = by_path(a_records)
    grouped_b = by_path(b_records)
    added: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    changed: List[Dict[str, Any]] = []

    for path in list(grouped_a.keys()) + [
        p for p in grouped_b.keys() if p not in grouped_a
    ]:
        list_a = grouped_a.get(path, [])
        list_b = grouped_b.get(path, [])
        for rec_a, rec_b in zip(list_a, list_b):
            entry: Dict[str, Any] = {"path": path}
            if rec_a.dtype != rec_b.dtype:
                entry["dtype"] = [rec_a.dtype, rec_b.dtype]
            if rec_a.shape != rec_b.shape:
                entry["shape"] = [rec_a.shape, rec_b.shape]
            same_shape_numeric = (
                rec_a.shape == rec_b.shape
                and rec_a.values is not None
                and rec_b.values is not None
                and np.issubdtype(rec_a.values.dtype, np.number)
                and np.issubdtype(rec_b.values.dtype, np.number)
                and not np.issubdtype(rec_a.values.dtype, np.complexfloating)
                and not np.issubdtype(rec_b.values.dtype, np.complexfloating)
            )
            if same_shape_numeric and rec_a.values.size:
                stats = _numeric_delta(rec_a.values, rec_b.values, epsilon)
                if stats is not None:
                    entry.update(stats)
                    bounds_a = _bounds(rec_a.values)
                    bounds_b = _bounds(rec_b.values)
                    if bounds_a != bounds_b:
                        entry["bounds"] = {"from": bounds_a, "to": bounds_b}
            if len(entry) > 1:
                changed.append(entry)
        for rec in list_a[len(list_b) :]:
            removed.append({"path": path, "dtype": rec.dtype, "shape": rec.shape})
        for rec in list_b[len(list_a) :]:
            added.append({"path": path, "dtype": rec.dtype, "shape": rec.shape})

    return {"added": added, "removed": removed, "changed": changed}


def _diff_leaves(
    a: Dict[str, Any], b: Dict[str, Any], epsilon: float
) -> Dict[str, List[Any]]:
    """Diff scalar leaves by path (numeric leaves use epsilon)."""
    added = [p for p in b if p not in a]
    removed = [p for p in a if p not in b]
    changed: List[Dict[str, Any]] = []
    for path, value_a in a.items():
        if path not in b:
            continue
        value_b = b[path]
        if _is_number(value_a) and _is_number(value_b):
            if abs(float(value_b) - float(value_a)) <= epsilon:
                continue
        elif value_a == value_b:
            continue
        changed.append(
            {
                "path": path,
                "from": summaries.truncated_repr(value_a, 60),
                "to": summaries.truncated_repr(value_b, 60),
            }
        )
    return {"added": added, "removed": removed, "changed": changed}


def _diff_state(
    data_a: Dict[str, Any],
    data_b: Dict[str, Any],
    arrays: Dict[str, List[Dict[str, Any]]],
    leaves: Dict[str, List[Any]],
) -> Dict[str, List[str]]:
    """State keys added/removed/changed (via array + leaf diffs under state.*)."""
    keys_a = set((data_a.get("state") or {}).keys())
    keys_b = set((data_b.get("state") or {}).keys())

    def state_key(path: Any) -> Optional[str]:
        text = path if isinstance(path, str) else path.get("path", "")
        if not text.startswith("state."):
            return None
        rest = text[len("state.") :]
        for sep in (".", "[", "/"):
            rest = rest.split(sep)[0]
        return rest

    changed = set()
    for section in (arrays["changed"], leaves["changed"]):
        for item in section:
            key = state_key(item)
            if key is not None and key in keys_a and key in keys_b:
                changed.add(key)
    return {
        "added": sorted(keys_b - keys_a),
        "removed": sorted(keys_a - keys_b),
        "changed": sorted(changed),
    }


def _diff_warnings(
    warnings_a: List[Dict[str, str]], warnings_b: List[Dict[str, str]]
) -> Dict[str, List[Dict[str, str]]]:
    """Warnings introduced/resolved, identified by (code, path)."""
    keys_a = {(w["code"], w["path"]) for w in warnings_a}
    keys_b = {(w["code"], w["path"]) for w in warnings_b}
    introduced = [w for w in warnings_b if (w["code"], w["path"]) not in keys_a]
    resolved = [w for w in warnings_a if (w["code"], w["path"]) not in keys_b]
    return {"introduced": introduced, "resolved": resolved}


def diff_visual_pair(
    visual_a: Dict[str, Any], visual_b: Dict[str, Any], epsilon: float
) -> Dict[str, Any]:
    """Diff two visuals (canonicalized payloads + buffers).

    Args:
        visual_a: Dict with ``data`` and ``buffers`` (old side).
        visual_b: Dict with ``data`` and ``buffers`` (new side).
        epsilon: Elementwise threshold below which numeric changes are
            considered identical.

    Returns:
        Per-pair diff payload; ``identical`` is True when no change section
        has entries.
    """
    data_a = summaries.canonicalize_visual_data(visual_a["data"])
    data_b = summaries.canonicalize_visual_data(visual_b["data"])
    buffers_a = visual_a["buffers"]
    buffers_b = visual_b["buffers"]

    structure_a = inspect_tools.collect_structure(data_a, buffers_a)
    structure_b = inspect_tools.collect_structure(data_b, buffers_b)

    comp_added, comp_removed, comp_changed = _diff_components(
        structure_a.components, structure_b.components
    )
    arrays = _diff_arrays(structure_a.arrays, structure_b.arrays, epsilon)

    leaves_a: Dict[str, Any] = {}
    leaves_b: Dict[str, Any] = {}
    _flatten_leaves({"ast": data_a.get("ast")}, "", None, leaves_a)
    _flatten_leaves({"state": data_a.get("state")}, "", None, leaves_a)
    _flatten_leaves({"ast": data_b.get("ast")}, "", None, leaves_b)
    _flatten_leaves({"state": data_b.get("state")}, "", None, leaves_b)
    leaves = _diff_leaves(leaves_a, leaves_b, epsilon)

    state = _diff_state(data_a, data_b, arrays, leaves)

    _, warnings_a = inspect_tools.inspect_visual_data(visual_a["data"], buffers_a)
    _, warnings_b = inspect_tools.inspect_visual_data(visual_b["data"], buffers_b)

    result: Dict[str, Any] = {
        "components": {
            "added": comp_added,
            "removed": comp_removed,
            "changed": comp_changed,
        },
        "arrays": arrays,
        "values": leaves,
        "state": state,
        "buffers": {
            "count": [len(buffers_a), len(buffers_b)],
            "total_bytes": [
                sum(len(b) for b in buffers_a),
                sum(len(b) for b in buffers_b),
            ],
        },
        "warnings": _diff_warnings(warnings_a, warnings_b),
    }
    result["identical"] = not any(
        (
            comp_added,
            comp_removed,
            comp_changed,
            arrays["added"],
            arrays["removed"],
            arrays["changed"],
            leaves["added"],
            leaves["removed"],
            leaves["changed"],
            state["added"],
            state["removed"],
            state["changed"],
        )
    )
    return result


def _target_info(target: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "file": target["file"],
        "kind": target["kind"],
        "visuals": len(target["visuals"]),
    }
    if target["kind"] == "colight":
        info["updates"] = target["updates"]
    if target["errors"]:
        info["errors"] = target["errors"]
    return info


def diff_targets(
    path_a: pathlib.Path,
    path_b: pathlib.Path,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict[str, Any]:
    """Diff two targets (.colight or .py each), pairing visuals by position.

    Args:
        path_a: Old side.
        path_b: New side.
        epsilon: Elementwise threshold for "unchanged" numeric values.

    Returns:
        Full diff payload with per-pair diffs, unpaired visuals, evaluation
        errors, a magnitude summary and a top-level ``identical`` flag.
    """
    target_a = load_target(path_a)
    target_b = load_target(path_b)

    pairs: List[Dict[str, Any]] = []
    for index, (visual_a, visual_b) in enumerate(
        zip(target_a["visuals"], target_b["visuals"])
    ):
        pair = {"index": index}
        for side, visual in (("a", visual_a), ("b", visual_b)):
            if "block" in visual:
                pair[f"{side}_block"] = visual["block"]
        pair.update(diff_visual_pair(visual_a, visual_b, epsilon))
        pairs.append(pair)

    def unpaired(target: Dict[str, Any], start: int) -> List[Dict[str, Any]]:
        extras = []
        for offset, visual in enumerate(target["visuals"][start:]):
            extra = {k: visual[k] for k in ("block", "lines") if k in visual}
            extra["index"] = start + offset
            extras.append(extra)
        return extras

    paired = min(len(target_a["visuals"]), len(target_b["visuals"]))
    only_a = unpaired(target_a, paired)
    only_b = unpaired(target_b, paired)

    arrays_changed = 0
    max_abs_delta: Optional[float] = None
    max_abs_delta_path: Optional[str] = None
    for pair in pairs:
        for entry in pair["arrays"]["changed"]:
            arrays_changed += 1
            delta = entry.get("max_abs_delta")
            if delta is not None and (max_abs_delta is None or delta > max_abs_delta):
                max_abs_delta = delta
                max_abs_delta_path = entry["path"]

    summary: Dict[str, Any] = {"arrays_changed": arrays_changed}
    if max_abs_delta is not None:
        summary["max_abs_delta"] = max_abs_delta
        summary["max_abs_delta_path"] = max_abs_delta_path

    identical = (
        all(pair["identical"] for pair in pairs)
        and not only_a
        and not only_b
        and not target_a["errors"]
        and not target_b["errors"]
    )
    payload: Dict[str, Any] = {
        "a": _target_info(target_a),
        "b": _target_info(target_b),
        "epsilon": epsilon,
        "identical": identical,
        "pairs": pairs,
        "summary": summary,
    }
    if only_a or only_b:
        payload["unpaired"] = {"a": only_a, "b": only_b}
    return payload


def verdict_line(payload: Dict[str, Any]) -> str:
    """One-line human verdict for a diff payload."""
    if payload["identical"]:
        return f"identical (within epsilon {payload['epsilon']:g})"
    parts: List[str] = []
    summary = payload["summary"]
    if summary["arrays_changed"]:
        text = f"{summary['arrays_changed']} array(s) changed"
        if "max_abs_delta" in summary:
            text += (
                f", max |Δ| {summary['max_abs_delta']:.4g}"
                f" in {summary['max_abs_delta_path']}"
            )
        parts.append(text)
    component_changes = sum(
        len(pair["components"][k])
        for pair in payload["pairs"]
        for k in ("added", "removed", "changed")
    )
    if component_changes:
        parts.append(f"{component_changes} component change(s)")
    value_changes = sum(
        len(pair["values"][k])
        for pair in payload["pairs"]
        for k in ("added", "removed", "changed")
    )
    if value_changes:
        parts.append(f"{value_changes} value change(s)")
    state_changes = sum(
        len(pair["state"][k])
        for pair in payload["pairs"]
        for k in ("added", "removed", "changed")
    )
    if state_changes:
        parts.append(f"{state_changes} state key change(s)")
    unpaired = payload.get("unpaired")
    if unpaired:
        parts.append(
            f"visuals only in A: {len(unpaired['a'])}, only in B: {len(unpaired['b'])}"
        )
    for side in ("a", "b"):
        errors = payload[side].get("errors")
        if errors:
            parts.append(f"{len(errors)} error(s) in {side.upper()}")
    return "; ".join(parts) or "differences found"


__all__ = [
    "DEFAULT_EPSILON",
    "diff_targets",
    "diff_visual_pair",
    "load_target",
    "verdict_line",
]
