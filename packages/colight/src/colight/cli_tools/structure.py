"""The one structural walk over a visual's JSON payload.

Collects component nodes (``{"__type__": "function"|"js_ref", ...}``) and
array records (typed buffer references and inline numeric lists) with unique
paths. Everything that needs to traverse a visual — summaries, inspection,
diffing — goes through this walker so component labeling and path layout
cannot drift between tools.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def component_label(node: Dict[str, Any]) -> Optional[str]:
    """Label for a component node, or None if ``node`` is not a component.

    Component nodes are ``{"__type__": "function"|"js_ref", "path": ...}``.
    ``MarkSpec`` nodes are qualified with their mark name (e.g.
    ``MarkSpec:dot``).
    """
    if node.get("__type__") not in ("function", "js_ref"):
        return None
    path = node.get("path")
    if not isinstance(path, str):
        return None
    if path == "MarkSpec":
        args = node.get("args") or []
        if args and isinstance(args[0], str):
            return f"MarkSpec:{args[0]}"
    return path


@dataclass
class ArrayRecord:
    path: str
    key: Optional[str]
    values: Optional[np.ndarray]
    dtype: str
    shape: List[int]
    inline: bool = False


@dataclass
class ComponentRecord:
    path: str
    display_path: str
    arrays: List[ArrayRecord] = field(default_factory=list)


@dataclass
class WalkState:
    buffers: List[bytes]
    arrays: List[ArrayRecord] = field(default_factory=list)
    components: List[ComponentRecord] = field(default_factory=list)
    stack: List[ComponentRecord] = field(default_factory=list)


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


def coerce_inline_array(value: Any) -> Optional[np.ndarray]:
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
    state: WalkState,
    path: str,
    key: Optional[str],
    values: Optional[np.ndarray],
    dtype: str,
    shape: List[int],
    inline: bool,
) -> None:
    record = ArrayRecord(
        path=path, key=key, values=values, dtype=dtype, shape=shape, inline=inline
    )
    state.arrays.append(record)
    if state.stack:
        state.stack[-1].arrays.append(record)


def _walk(node: Any, path: str, key: Optional[str], state: WalkState) -> None:
    """Recursive walk collecting components and arrays with unique paths."""
    if isinstance(node, dict):
        if node.get("__type__") == "ndarray" and "__buffer_index__" in node:
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
        label = component_label(node)
        if label is not None:
            component = ComponentRecord(path=label, display_path=f"{path}/{label}")
            state.components.append(component)
            state.stack.append(component)
            # Args are indexed so sibling args (and their contents) never
            # share a path; the array diff pairs arrays by path.
            for index, arg in enumerate(node.get("args") or []):
                _walk(arg, f"{path}/{label}[{index}]", None, state)
            state.stack.pop()
            return
        for k, v in node.items():
            if k in ("__type__", "path", "bufferLayout", "id"):
                continue
            _walk(v, f"{path}.{k}" if path else k, k, state)
    elif isinstance(node, list):
        if key is not None:
            values = coerce_inline_array(node)
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


def collect_structure(data: Dict[str, Any], buffers: List[bytes]) -> WalkState:
    """Walk a visual's payload, collecting components and arrays.

    Args:
        data: The visual's JSON envelope (``ast``, ``state``, ...).
        buffers: The visual's binary buffers.

    Returns:
        The populated walk state (components and array records in order).
    """
    state = WalkState(buffers=buffers)
    _walk({"ast": data.get("ast")}, "", None, state)
    _walk({"state": data.get("state")}, "", None, state)
    return state


def iter_component_paths(node: Any) -> List[str]:
    """List component labels appearing in an AST/state payload, in order."""
    state = WalkState(buffers=[])
    _walk(node, "", None, state)
    return [component.path for component in state.components]


__all__ = [
    "ArrayRecord",
    "ComponentRecord",
    "WalkState",
    "coerce_inline_array",
    "collect_structure",
    "component_label",
    "iter_component_paths",
]
