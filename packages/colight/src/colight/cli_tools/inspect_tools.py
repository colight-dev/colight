"""Structural inspection of visuals with sanity warnings — no rendering.

Parses ``.colight`` artifacts (or evaluates a ``.py`` file and inspects the
visuals it produces) and reports component structure, per-array schema, state
keys and callbacks, plus warnings that catch "why is my scene blank" class
problems: empty arrays, NaN/Inf values, near-zero alphas, degenerate bounds
and mismatched per-instance attribute lengths.
"""

import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import colight.format as colight_format
from colight.runtime.executor import BlockExecutor
from colight.runtime.parser import parse_colight_file

from . import blocks as blocks_mod
from . import summaries

# Per-instance attribute names and the number of scalars per instance when
# the array is flat (scene3d convention).
PER_INSTANCE_COMPONENTS = {
    "centers": 3,
    "positions": 3,
    "colors": 3,
    "half_sizes": 3,
    "half_size": 3,
    "quaternions": 4,
    "alphas": 1,
    "sizes": 1,
    "scales": 1,
}

ALPHA_KEYS = {"alpha", "alphas", "opacity", "fill_opacity", "fillOpacity"}
BOUNDS_KEYS = {"centers", "positions", "points", "x", "y", "z"}


@dataclass
class _ArrayRecord:
    path: str
    key: Optional[str]
    values: Optional[np.ndarray]
    dtype: str
    shape: List[int]
    inline: bool = False


@dataclass
class _ComponentRecord:
    path: str
    display_path: str
    arrays: List[_ArrayRecord] = field(default_factory=list)


@dataclass
class _WalkState:
    buffers: List[bytes]
    arrays: List[_ArrayRecord] = field(default_factory=list)
    components: List[_ComponentRecord] = field(default_factory=list)
    stack: List[_ComponentRecord] = field(default_factory=list)


def _decode_ndarray(node: Dict[str, Any], buffers: List[bytes]) -> Optional[np.ndarray]:
    """Decode an ndarray buffer reference; None if undecodable."""
    try:
        index = node["__buffer_index__"]
        dtype = node.get("dtype", "float64")
        shape = node.get("shape")
        values = np.frombuffer(buffers[index], dtype=dtype)
        if shape:
            values = values.reshape(shape)
        return values
    except Exception:
        return None


def _is_numeric_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def _coerce_inline_array(value: Any) -> Optional[np.ndarray]:
    """Coerce a flat or rectangular nested numeric list to an ndarray."""
    if _is_numeric_list(value):
        return np.asarray(value)
    if (
        isinstance(value, list)
        and len(value) > 0
        and all(_is_numeric_list(row) for row in value)
        and len({len(row) for row in value}) == 1
    ):
        return np.asarray(value)
    return None


def _record_array(
    state: _WalkState,
    path: str,
    key: Optional[str],
    values: Optional[np.ndarray],
    dtype: str,
    shape: List[int],
    inline: bool,
) -> None:
    record = _ArrayRecord(
        path=path, key=key, values=values, dtype=dtype, shape=shape, inline=inline
    )
    state.arrays.append(record)
    if state.stack:
        state.stack[-1].arrays.append(record)


def _walk(node: Any, path: str, key: Optional[str], state: _WalkState) -> None:
    """Recursive walk collecting components and arrays with display paths."""
    if isinstance(node, dict):
        node_type = node.get("__type__")
        if node_type == "ndarray" and "__buffer_index__" in node:
            values = _decode_ndarray(node, state.buffers)
            _record_array(
                state,
                path,
                key,
                values,
                str(node.get("dtype", "?")),
                list(node.get("shape") or []),
                inline=False,
            )
            return
        if node_type in ("function", "js_ref") and isinstance(node.get("path"), str):
            label = node["path"]
            if label == "MarkSpec":
                args = node.get("args") or []
                if args and isinstance(args[0], str):
                    label = f"MarkSpec:{args[0]}"
            component = _ComponentRecord(path=label, display_path=f"{path}/{label}")
            state.components.append(component)
            state.stack.append(component)
            for arg in node.get("args") or []:
                _walk(arg, f"{path}/{label}", None, state)
            state.stack.pop()
            return
        for k, v in node.items():
            if k in ("__type__", "path", "bufferLayout", "id"):
                continue
            _walk(v, f"{path}.{k}" if path else k, k, state)
    elif isinstance(node, list):
        if key is not None:
            values = _coerce_inline_array(node)
            if values is not None:
                _record_array(
                    state,
                    path,
                    key,
                    values,
                    str(values.dtype),
                    list(values.shape),
                    inline=True,
                )
                return
        for i, item in enumerate(node):
            _walk(item, f"{path}[{i}]", key, state)


def _implied_instance_count(record: _ArrayRecord) -> Optional[int]:
    """Number of instances implied by a per-instance attribute array."""
    if record.key not in PER_INSTANCE_COMPONENTS:
        return None
    divisor = PER_INSTANCE_COMPONENTS[record.key]
    shape = record.shape
    if not shape:
        return None
    if len(shape) > 1:
        return shape[0]
    if divisor > 1 and shape[0] % divisor == 0:
        return shape[0] // divisor
    return shape[0]


def _array_warnings(record: _ArrayRecord) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    values = record.values
    size = int(np.prod(record.shape)) if record.shape else 0
    if size == 0:
        warnings.append(
            {
                "code": "empty-array",
                "path": record.path,
                "message": f"array is empty (shape {record.shape})",
            }
        )
        return warnings
    if values is None or not np.issubdtype(values.dtype, np.number):
        return warnings
    if np.issubdtype(values.dtype, np.floating):
        nan_count = int(np.isnan(values).sum())
        inf_count = int(np.isinf(values).sum())
        if nan_count:
            warnings.append(
                {
                    "code": "nan-values",
                    "path": record.path,
                    "message": f"{nan_count}/{values.size} values are NaN",
                }
            )
        if inf_count:
            warnings.append(
                {
                    "code": "inf-values",
                    "path": record.path,
                    "message": f"{inf_count}/{values.size} values are Inf",
                }
            )
    if record.key in ALPHA_KEYS:
        finite = values[np.isfinite(values)]
        if finite.size and float(np.max(finite)) <= 1e-6:
            warnings.append(
                {
                    "code": "alphas-zero",
                    "path": record.path,
                    "message": "all alpha/opacity values are ~0 (invisible)",
                }
            )
    if record.key in BOUNDS_KEYS and values.size > 1:
        divisor = PER_INSTANCE_COMPONENTS.get(record.key or "", 1)
        points = values
        if points.ndim == 1 and divisor > 1 and points.size % divisor == 0:
            points = points.reshape(-1, divisor)
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if points.shape[0] > 1:
            finite_rows = points[np.all(np.isfinite(points), axis=1)]
            if finite_rows.shape[0] > 1:
                extents = np.max(finite_rows, axis=0) - np.min(finite_rows, axis=0)
                if bool(np.all(extents == 0)):
                    warnings.append(
                        {
                            "code": "degenerate-bounds",
                            "path": record.path,
                            "message": (
                                f"{points.shape[0]} points but zero extent "
                                "on every axis (all points identical)"
                            ),
                        }
                    )
    return warnings


def _component_warnings(component: _ComponentRecord) -> List[Dict[str, str]]:
    counts: Dict[str, int] = {}
    for record in component.arrays:
        implied = _implied_instance_count(record)
        if implied is not None and record.key is not None:
            counts[record.key] = implied
    if len(set(counts.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        return [
            {
                "code": "length-mismatch",
                "path": component.display_path,
                "message": f"per-instance attributes imply different counts: {detail}",
            }
        ]
    return []


def collect_structure(data: Dict[str, Any], buffers: List[bytes]) -> _WalkState:
    """Walk a visual's payload, collecting components and arrays.

    Args:
        data: The visual's JSON envelope (``ast``, ``state``, ...).
        buffers: The visual's binary buffers.

    Returns:
        The populated walk state (components and array records in order).
    """
    state = _WalkState(buffers=buffers)
    _walk({"ast": data.get("ast")}, "", None, state)
    _walk({"state": data.get("state")}, "", None, state)
    return state


def inspect_visual_data(
    data: Dict[str, Any], buffers: List[bytes]
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Inspect a visual's JSON payload + buffers.

    Args:
        data: The visual's JSON envelope (``ast``, ``state``, ...).
        buffers: The visual's binary buffers.

    Returns:
        Tuple of (structure payload, warnings list).
    """
    state = collect_structure(data, buffers)

    component_counts: Dict[str, int] = {}
    component_instances: Dict[str, Optional[int]] = {}
    for component in state.components:
        component_counts[component.path] = component_counts.get(component.path, 0) + 1
        implied = None
        for record in component.arrays:
            implied = _implied_instance_count(record)
            if implied is not None:
                break
        if component.path not in component_instances or implied is not None:
            component_instances[component.path] = implied

    arrays_out: List[Dict[str, Any]] = []
    warnings: List[Dict[str, str]] = []
    for record in state.arrays:
        item: Dict[str, Any] = {
            "path": record.path,
            "dtype": record.dtype,
            "shape": record.shape,
        }
        if record.values is not None:
            item.update(summaries.array_stats(record.values))
        arrays_out.append(item)
        warnings.extend(_array_warnings(record))
    for component in state.components:
        warnings.extend(_component_warnings(component))

    state_dict = data.get("state") or {}
    listeners = data.get("listeners") or {}
    py_listeners = data.get("py_listeners") or {}

    payload = {
        "components": [
            {
                "path": path,
                "count": count,
                "instances": component_instances.get(path),
            }
            for path, count in component_counts.items()
        ],
        "arrays": arrays_out,
        "state_keys": list(state_dict.keys()),
        "synced_keys": list(data.get("syncedKeys") or []),
        "listeners": list(listeners.keys()),
        "py_listeners": list(py_listeners.keys()),
        "buffers": {
            "count": len(buffers),
            "total_bytes": sum(len(b) for b in buffers),
        },
    }
    return payload, warnings


def inspect_colight_file(file_path: pathlib.Path) -> Dict[str, Any]:
    """Inspect a ``.colight`` artifact directly (no evaluation)."""
    data, buffers, updates = colight_format.parse_file(file_path)
    if data is None:
        return {
            "file": str(file_path),
            "kind": "colight",
            "error": {
                "type": "ValueError",
                "message": "file contains no initial state entry",
            },
            "updates": len(updates),
        }
    payload, warnings = inspect_visual_data(data, buffers)
    return {
        "file": str(file_path),
        "kind": "colight",
        "updates": len(updates),
        "visual": payload,
        "warnings": warnings,
    }


def evaluate_python_visuals(
    file_path: pathlib.Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate a ``.py`` file headlessly and collect the visuals it produces.

    Args:
        file_path: Path to a notebook-style ``.py`` file.

    Returns:
        Tuple of (visuals, errors). Each visual dict has ``block`` (stable
        id), ``lines``, ``data`` (JSON envelope) and ``buffers``; each error
        dict has ``block``, ``lines`` and a structured ``error``.
    """
    file_path = file_path.resolve()
    document = parse_colight_file(file_path)
    pairs = blocks_mod.assign_stable_ids(document)
    executor = BlockExecutor()

    visuals: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for block, sid in pairs:
        result = executor.execute_block(block, str(file_path))
        lines = list(blocks_mod.block_lines(block))
        if result.error:
            errors.append(
                {
                    "block": sid,
                    "lines": lines,
                    "error": result.error_info
                    or {"type": "Exception", "message": result.error.strip()},
                }
            )
            continue
        if result.colight_bytes is None:
            continue
        data, buffers = summaries.parse_colight_bytes(result.colight_bytes)
        visuals.append({"block": sid, "lines": lines, "data": data, "buffers": buffers})
    return visuals, errors


def inspect_python_file(file_path: pathlib.Path) -> Dict[str, Any]:
    """Evaluate a ``.py`` file and inspect every visual it produces."""
    file_path = file_path.resolve()
    produced, errors = evaluate_python_visuals(file_path)

    visuals: List[Dict[str, Any]] = []
    for item in produced:
        payload, warnings = inspect_visual_data(item["data"], item["buffers"])
        visuals.append(
            {
                "block": item["block"],
                "lines": item["lines"],
                "visual": payload,
                "warnings": warnings,
            }
        )

    out: Dict[str, Any] = {
        "file": str(file_path),
        "kind": "py",
        "visuals": visuals,
    }
    if errors:
        out["errors"] = errors
    return out


def inspect_target(file_path: pathlib.Path) -> Dict[str, Any]:
    """Dispatch on target type (``.colight`` vs ``.py``)."""
    if file_path.suffix == ".colight":
        return inspect_colight_file(file_path)
    if file_path.suffix == ".py":
        return inspect_python_file(file_path)
    raise ValueError(f"Unsupported target (expected .colight or .py): {file_path}")
