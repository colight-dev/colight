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
    # Colormap legend spec ({cmap, domain, label, categorical, ...}) when
    # the component was colored via color_by (see colight/colormaps.py).
    color_by: Optional[Dict[str, Any]] = None
    # Per-instance filter spec ({min, max, label, ...}) when the component
    # declared filter_by (see colight/scene3d.py). The `values` buffer is
    # dropped; only the reportable thresholds are kept.
    filter_by: Optional[Dict[str, Any]] = None
    # Switchable color channels ({active, channels: [{name, label, kind}]})
    # when the component declared color_channels. The per-channel value arrays
    # and LUTs are dropped; only the discoverable channel roster is kept.
    color_channels: Optional[Dict[str, Any]] = None
    # Parameterized channels driving props on this component: one entry per
    # `Plot.channel(...)` of {parameter, rule, domain, samples, prop}. See
    # colight/core.py; the resampler itself is not a component.
    channels: List[Dict[str, Any]] = field(default_factory=list)


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


def _scene_config_label(node: Dict[str, Any], state: WalkState) -> Optional[str]:
    """Label for a nested scene3d component config dict, or None.

    Scene components nested inside another component's props (e.g. a Group's
    ``children``) serialize as plain ``{"type": "Mesh", ...}`` dicts rather
    than function nodes. Recognize them only inside a scene3d subtree so an
    unrelated dict that happens to carry a ``type`` key is never misread as a
    component.
    """
    type_name = node.get("type")
    if not isinstance(type_name, str) or "__type__" in node:
        return None
    if not (state.stack and state.stack[-1].path.startswith("scene3d.")):
        return None
    return f"scene3d.{type_name}"


CHANNEL_PATH = "colight.resampleChannel"


def _channel_config(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The config of a ``Plot.channel`` JSCall node, or None.

    A channel is a declaration on the prop it drives, not a component of its
    own: recognized here so the walker records it on the enclosing component
    instead of adding ``colight.resampleChannel`` to the component roster.
    """
    if node.get("__type__") != "function" or node.get("path") != CHANNEL_PATH:
        return None
    args = node.get("args") or []
    config = args[0] if args else None
    return config if isinstance(config, dict) else None


def _channel_summary(
    config: Dict[str, Any], prop: Optional[str], at: Optional[np.ndarray]
) -> Dict[str, Any]:
    """Reportable shape of one channel: what it is indexed by and how."""
    entry: Dict[str, Any] = {
        "parameter": config.get("parameter"),
        "rule": config.get("rule"),
        "prop": prop,
    }
    if at is not None and at.size:
        entry["domain"] = [float(at.flat[0]), float(at.flat[-1])]
        entry["samples"] = int(at.shape[0])
    return entry


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
        config = _channel_config(node)
        if config is not None:
            # Record the declaration on the component the channel drives, then
            # walk the config's arrays under the prop's own path so `at` and
            # `values` stay in the array records (real shipped data, and the
            # dominant cost of a dense table) with a stable path for diffing.
            at_values: Optional[np.ndarray] = None
            for config_key, config_value in config.items():
                if config_key in ("parameter", "rule", "value"):
                    continue
                if config_key == "at":
                    at_values = (
                        _decode_ndarray(config_value, state.buffers)
                        if isinstance(config_value, dict)
                        else coerce_inline_array(config_value)
                    )
                _walk(config_value, f"{path}.{config_key}", config_key, state)
            if state.stack:
                state.stack[-1].channels.append(
                    _channel_summary(config, key, at_values)
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
        scene_label = _scene_config_label(node, state)
        if scene_label is not None:
            # Record the component but keep walking with the UNCHANGED path:
            # array paths must stay stable (diff pairs arrays by path), so the
            # component roster gains an entry without renaming anything under it.
            component = ComponentRecord(
                path=scene_label, display_path=f"{path}/{scene_label}"
            )
            state.components.append(component)
            state.stack.append(component)
            _walk_dict_items(node, path, state)
            state.stack.pop()
            return
        _walk_dict_items(node, path, state)
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


def _walk_dict_items(node: Dict[str, Any], path: str, state: WalkState) -> None:
    """Walk a dict's items, routing metadata keys to the enclosing component."""
    for k, v in node.items():
        if k in ("__type__", "path", "bufferLayout", "id"):
            continue
        # A color_by spec is legend metadata, not data: record it on the
        # enclosing component and don't descend (its stops/colors lists
        # would otherwise pollute the array records).
        if k == "color_by" and isinstance(v, dict) and "cmap" in v:
            if state.stack:
                state.stack[-1].color_by = v
            continue
        # filter_by is per-instance filter metadata: record min/max/label on
        # the enclosing component and don't descend (its `values` buffer
        # would otherwise pollute the array records).
        if k == "filter_by" and isinstance(v, dict) and "values" in v:
            if state.stack:
                state.stack[-1].filter_by = {
                    key: val for key, val in v.items() if key != "values"
                }
            continue
        # color_channels is switchable-coloring metadata: record the
        # discoverable channel roster (name/label/kind) on the enclosing
        # component and don't descend (per-channel `values`/`lut` buffers
        # would otherwise pollute the array records).
        if k == "color_channels" and isinstance(v, dict) and v:
            if state.stack:
                roster = []
                for name, chan in v.items():
                    if not isinstance(chan, dict):
                        continue
                    colorizer = chan.get("colorizer") or {}
                    roster.append(
                        {
                            "name": name,
                            "label": chan.get("label", name),
                            "kind": colorizer.get("kind", "continuous"),
                        }
                    )
                state.stack[-1].color_channels = {"channels": roster}
            continue
        if k == "active_channel" and isinstance(v, str):
            if state.stack and state.stack[-1].color_channels is not None:
                state.stack[-1].color_channels["active"] = v
            continue
        _walk(v, f"{path}.{k}" if path else k, k, state)


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
