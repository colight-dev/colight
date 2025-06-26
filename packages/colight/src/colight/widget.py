import datetime
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, Tuple
from types import SimpleNamespace

import anywidget
import numpy as np
import traitlets
import warnings

from colight.env import CONFIG, ANYWIDGET_PATH
from colight.protocols import Collector
from colight.binary_serialization import serialize_binary_data, replace_buffers
from colight.state_operations import entry_id, normalize_updates, apply_updates


# Serialization registry
SKIP = object()  # Sentinel value for "skip this serializer"
_SERIALIZERS: List[Callable] = []


def list_serializers() -> List[str]:
    """List registered serializer names in order."""
    return [getattr(s, "_serializer_name", s.__name__) for s in _SERIALIZERS]


def register_serializer(
    name: Optional[str] = None,
    before: Optional[str] = None,
    after: Optional[str] = None,
):
    """Decorator to register a custom serializer with optional positioning.

    Args:
        name: Name for this serializer (defaults to function name)
        before: Insert before the serializer with this name
        after: Insert after the serializer with this name
    """

    def decorator(func: Callable) -> Callable:
        func._serializer_name = name or func.__name__

        if before or after:
            # Find insertion point at registration time (one-time cost)
            insert_idx = None
            for i, serializer in enumerate(_SERIALIZERS):
                if before and getattr(serializer, "_serializer_name", None) == before:
                    insert_idx = i
                    break
                elif after and getattr(serializer, "_serializer_name", None) == after:
                    insert_idx = i + 1
                    break

            if insert_idx is not None:
                _SERIALIZERS.insert(insert_idx, func)
            else:
                _SERIALIZERS.append(func)
        else:
            _SERIALIZERS.append(func)
        return func

    return decorator


class SubscriptableNamespace(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class CollectedState:
    # collect initial state while serializing data.
    def __init__(self, buffers=None, widget=None):
        self.syncedKeys = set()
        self.state = {}
        self.stateJSON = {}
        self.listeners = {"py": {}, "js": {}}
        self.imports = []
        self.animateBy = []
        self.buffers = buffers or []
        self.widget = widget

    def state_entry(self, state_key, value, sync=False):
        if sync:
            self.syncedKeys.add(state_key)
        if state_key not in self.stateJSON:
            self.state[state_key] = value
            self.stateJSON[state_key] = to_json(value, collected_state=self)
        return {"__type__": "ref", "state_key": state_key}

    def add_import(self, spec: dict):
        """Add an import specification.

        Args:
            spec: Import specification with format, source info, and options
        """
        self.imports.append(spec)
        return None

    def _add_listener(self, state_key, listener):
        listeners = [listener] if not isinstance(listener, list) else listener
        for listener in listeners:
            target = "py" if callable(listener) else "js"
            self.listeners[target].setdefault(state_key, []).append(listener)

    def add_listeners(self, listeners):
        for state_key, listener in listeners.items():
            self.syncedKeys.add(state_key)
            self._add_listener(state_key, listener)
        return None


@register_serializer()
def serialize_collector(data: Any, collected_state: Optional[CollectedState]) -> Any:
    """Serializer for Collector protocol objects."""
    if not isinstance(data, Collector):
        return SKIP
    assert collected_state is not None
    return data.collect(collected_state)


@register_serializer()
def serialize_for_json(data: Any, collected_state: Optional[CollectedState]) -> Any:
    """Serializer for objects with for_json() method."""
    if not hasattr(data, "for_json"):
        return SKIP
    return to_json(data.for_json(), collected_state=collected_state)


@register_serializer()
def serialize_callable(data: Any, collected_state: Optional[CollectedState]) -> Any:
    """Serializer for callable objects."""
    if not callable(data):
        return SKIP
    assert collected_state is not None
    if collected_state and collected_state.widget:
        id = str(uuid.uuid4())
        collected_state.widget.callback_registry[id] = data
        return {"__type__": "callback", "id": id}
    warnings.warn(
        "Callback encountered but no widget context available - callback will be elided",
        UserWarning,
    )
    return None


@register_serializer()
def serialize_numpy_array(data: Any, collected_state: Optional[CollectedState]) -> Any:
    """Serializer for numpy and jax arrays."""
    if not (
        isinstance(data, np.ndarray)
        or type(data).__name__
        in (
            "DeviceArray",
            "Array",
            "ArrayImpl",
        )
    ):
        return SKIP

    assert collected_state is not None
    try:
        if data.ndim == 0:  # It's a scalar
            return data.item()
    except AttributeError:
        pass

    bytes_data = data.tobytes()
    return serialize_binary_data(
        collected_state.buffers,
        {
            "__type__": "ndarray",
            "data": bytes_data,
            "dtype": str(data.dtype),
            "shape": data.shape,
        },
    )


@register_serializer()
def serialize_attributes_dict(
    data: Any, collected_state: Optional[CollectedState]
) -> Any:
    """Serializer for objects with attributes_dict() method."""
    if not (hasattr(data, "attributes_dict") and callable(data.attributes_dict)):
        return SKIP
    return to_json(data.attributes_dict(), collected_state=collected_state)


def to_json(
    data: Any,
    collected_state: Optional[CollectedState] = None,
) -> Any:
    # Handle NaN at top level
    if isinstance(data, float):
        if np.isnan(data):
            return None
        return data

    # Handle basic JSON-serializable types first since they're most common
    if isinstance(data, (str, int, bool)):
        return data

    # Handle None case
    if data is None:
        return None

    # Handle binary data
    if isinstance(data, (bytes, bytearray, memoryview)):
        assert collected_state is not None
        # Store binary data in buffers and return reference
        buffer_index = len(collected_state.buffers)
        collected_state.buffers.append(data)
        return {"__type__": "buffer", "index": buffer_index}

    # Handle datetime objects early since isinstance check is fast
    if isinstance(data, (datetime.date, datetime.datetime)):
        return {"__type__": "datetime", "value": data.isoformat()}

    # Use extensible serializer system for complex types
    for serializer in _SERIALIZERS:
        result = serializer(data, collected_state)
        if result is not SKIP:
            return result

    # Handle containers
    if isinstance(data, dict):
        return {k: to_json(v, collected_state) for k, v in data.items()}

    if isinstance(data, (list, tuple)):
        return [to_json(x, collected_state) for x in data]

    if isinstance(data, Iterable):
        if not hasattr(data, "__len__") and not hasattr(data, "__getitem__"):
            warnings.warn(
                "Potentially exhaustible iterator encountered: generator", UserWarning
            )
        return [to_json(x, collected_state) for x in data]

    # Raise error for unsupported types
    raise TypeError(f"Object of type {type(data)} is not JSON serializable")


def resolve_animate_by(collected_state):
    """Resolve the actual range from metadata, handling rangeFrom cases"""

    metadatas = collected_state.animateBy
    out = []
    for metadata in metadatas:
        if metadata is None:
            continue

        range_val = None

        if metadata.get("range") is not None:
            range_val = metadata.get("range")
            if isinstance(range_val, int):
                range_val = [0, range_val - 1]
            range_val = range_val

        if metadata.get("rangeFrom") is not None:
            rangeFrom = metadata["rangeFrom"]
            # Determine the state key to look up
            if isinstance(rangeFrom, str):
                state_key = rangeFrom
            elif hasattr(rangeFrom, "_state_key"):
                state_key = rangeFrom._state_key
            else:
                state_key = None

            # Look up the value in state and return range if it has length
            if state_key is not None:
                value = collected_state.state.get(state_key)
                if value is not None and hasattr(value, "__len__"):
                    range_val = [0, len(value) - 1]
        if range_val is not None:
            out.append(
                {
                    "key": metadata["key"],
                    "range": range_val,
                    "fps": metadata["fps"],
                    "step": metadata.get("step"),
                }
            )
    return out


def to_json_with_state(
    layout_item: Any,
    widget: "Widget | None" = None,
    buffers: List[bytes | bytearray | memoryview] | None = None,
) -> Union[Any, Tuple[Any, List[bytes | bytearray | memoryview]]]:
    collected_state = CollectedState(widget=widget, buffers=buffers or [])
    id = layout_item.get_id() if hasattr(layout_item, "get_id") else None
    ast = to_json(layout_item, collected_state=collected_state)

    json = to_json(
        {
            "ast": ast,
            "id": id,
            "state": collected_state.stateJSON,
            "syncedKeys": collected_state.syncedKeys,
            "listeners": collected_state.listeners["js"],
            "imports": collected_state.imports,
            "animateBy": resolve_animate_by(collected_state),
            **CONFIG,
        },
    )

    if widget is not None:
        widget.state.init_state(collected_state)
    return json, collected_state.buffers


class WidgetState:
    def __init__(self, widget):
        self._state = {}
        self._widget = widget
        self._syncedKeys = set()
        self._listeners = {}
        self._processing_listeners = (
            set()
        )  # Track which listeners are currently processing

    def __getattr__(self, name):
        if name in self._state:
            return self._state[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._state[name] = value
            self.update([name, "reset", value])

    def notify_listeners(self, updates: List[List[Any]]) -> None:
        for name, operation, value in updates:
            for listener in self._listeners.get(name, []):
                # Skip if this listener is already being processed
                if listener in self._processing_listeners:
                    continue
                try:
                    self._processing_listeners.add(listener)
                    listener(
                        self._widget,
                        SubscriptableNamespace(id=name, value=self._state[name]),
                    )
                finally:
                    self._processing_listeners.remove(listener)

    # update values from python - send to js
    def update(self, *updates: Union[List[Any], Dict[str, Any]]) -> None:
        normalized_updates = normalize_updates(updates)

        # apply updates locally for synced state
        synced_updates = [
            [name, op, payload]
            for name, op, payload in normalized_updates
            if entry_id(name) in self._syncedKeys
        ]
        apply_updates(self._state, synced_updates)

        # send all updates to JS regardless of sync status
        collected_state = CollectedState(widget=self._widget)
        json_updates = to_json(normalized_updates, collected_state=collected_state)
        self._widget.send(
            {"type": "update_state", "updates": json_updates},
            buffers=collected_state.buffers,
        )

        self.notify_listeners(synced_updates)

    # accept updates from js - notify callbacks
    def accept_js_updates(self, updates: List[List[Any]]) -> None:
        apply_updates(self._state, updates)
        self.notify_listeners(updates)

    def init_state(self, collected_state):
        self._listeners = collected_state.listeners["py"]
        self._syncedKeys = syncedKeys = collected_state.syncedKeys

        for key, value in collected_state.state.items():
            if key in syncedKeys and key not in self._state:
                self._state[key] = value


class Widget(anywidget.AnyWidget):
    _esm = ANYWIDGET_PATH
    # CSS is now embedded in the JS bundle
    callback_registry: Dict[str, Callable] = {}
    data = traitlets.Any().tag(sync=True, to_json=to_json_with_state)

    def __init__(self, ast: Any):
        self.state = WidgetState(self)
        super().__init__()
        self.data = ast

    def set_ast(self, ast: Any):
        self.data = ast

    def _repr_mimebundle_(self, **kwargs):  # type: ignore
        return super()._repr_mimebundle_(**kwargs)

    @anywidget.experimental.command  # type: ignore
    def handle_callback(
        self, params: dict[str, Any], buffers: list[bytes]
    ) -> tuple[str, list[bytes]]:
        f = self.callback_registry[params["id"]]
        if f is not None:
            event = replace_buffers(params["event"], buffers)
            print(event)
            event = SubscriptableNamespace(**event)
            f(self, event)
        return "ok", []

    @anywidget.experimental.command  # type: ignore
    def handle_updates(
        self, params: dict[str, Any], buffers: list[bytes]
    ) -> tuple[str, list[bytes]]:
        updates = replace_buffers(params["updates"], buffers)
        self.state.accept_js_updates(updates)
        return "ok", []
