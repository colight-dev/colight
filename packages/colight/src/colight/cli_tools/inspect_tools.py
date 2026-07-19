"""Structural inspection of visuals with sanity warnings — no rendering.

Parses ``.colight`` artifacts (or evaluates a ``.py`` file and inspects the
visuals it produces) and reports component structure, per-array schema, state
keys and callbacks, plus warnings that catch "why is my scene blank" class
problems: empty arrays, NaN/Inf values, near-zero alphas, degenerate bounds
and mismatched per-instance attribute lengths.
"""

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import summaries, targets
from .structure import ArrayRecord, ComponentRecord, WalkState, collect_structure

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


def _implied_instance_count(record: ArrayRecord) -> Optional[int]:
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


def _array_warnings(record: ArrayRecord) -> List[Dict[str, str]]:
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


def _component_warnings(component: ComponentRecord) -> List[Dict[str, str]]:
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


def structure_warnings(state: WalkState) -> List[Dict[str, str]]:
    """Sanity warnings for an already-collected walk state."""
    warnings: List[Dict[str, str]] = []
    for record in state.arrays:
        warnings.extend(_array_warnings(record))
    for component in state.components:
        warnings.extend(_component_warnings(component))
    return warnings


def legend_payload(
    color_by: Dict[str, Any], component: Optional[str] = None
) -> Dict[str, Any]:
    """Machine-readable legend entry for a component's ``color_by`` spec.

    Lean on purpose (no color tables): agents read what the colors encode —
    ``{"component"?, "label"?, "cmap", "domain"?, "categorical",
    "categories"?}``.
    """
    entry: Dict[str, Any] = {}
    if component is not None:
        entry["component"] = component
    if "label" in color_by:
        entry["label"] = color_by["label"]
    entry["cmap"] = color_by.get("cmap")
    if "domain" in color_by:
        entry["domain"] = color_by["domain"]
    entry["categorical"] = bool(color_by.get("categorical"))
    if "categories" in color_by:
        entry["categories"] = color_by["categories"]
    return entry


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
    for record in state.arrays:
        item: Dict[str, Any] = {
            "path": record.path,
            "dtype": record.dtype,
            "shape": record.shape,
        }
        if record.values is not None:
            item.update(summaries.array_stats(record.values))
        arrays_out.append(item)
    warnings = structure_warnings(state)

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
    legends = [
        legend_payload(component.color_by, component=component.path)
        for component in state.components
        if component.color_by is not None
    ]
    if legends:
        payload["legends"] = legends
    return payload, warnings


def inspect_target(file_path: pathlib.Path) -> Dict[str, Any]:
    """Inspect a target (``.colight`` or ``.py``), one payload per kind.

    Raises:
        ValueError: Unsupported target or empty ``.colight`` file.
    """
    loaded = targets.load_target(file_path)
    if loaded.kind == "colight":
        visual = loaded.visuals[0]
        payload, warnings = inspect_visual_data(visual["data"], visual["buffers"])
        return {
            "file": str(file_path),
            "kind": "colight",
            "updates": loaded.updates,
            "visual": payload,
            "warnings": warnings,
        }

    visuals: List[Dict[str, Any]] = []
    for item in loaded.visuals:
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
        "file": str(file_path.resolve()),
        "kind": "py",
        "visuals": visuals,
    }
    if loaded.errors:
        out["errors"] = loaded.errors
    return out
