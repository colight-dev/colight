"""State update operations for colight widgets."""

from typing import Any, Dict, Iterable, List, Union


def entry_id(key: Union[str, Any]) -> str:
    """Extract state key from string or object.

    Args:
        key: State key as string or object with _state_key attribute

    Returns:
        String state key

    Raises:
        TypeError: If key is not a string or doesn't have _state_key
    """
    if isinstance(key, str):
        return key
    elif hasattr(key, "_state_key"):
        return key._state_key
    else:
        raise TypeError(f"Expected str or object with _state_key, got {type(key)}")


def normalize_updates(
    updates: Iterable[Union[List[Any], Dict[str, Any]]],
) -> List[List[Any]]:
    """Normalize various update formats into consistent list format.

    Args:
        updates: Mixed format updates (dicts or lists)

    Returns:
        Normalized list of [key, operation, value] tuples
    """
    out = []
    for entry in updates:
        if isinstance(entry, dict):
            for key, value in entry.items():
                out.append([entry_id(key), "reset", value])
        else:
            out.append([entry_id(entry[0]), entry[1], entry[2]])
    return out


def apply_updates(state: Dict[str, Any], updates: List[List[Any]]) -> None:
    """Apply state updates in-place.

    Args:
        state: State dictionary to update
        updates: List of [name, operation, payload] updates

    Supported operations:
        - reset: Replace value
        - append: Add single item to list
        - concat: Add multiple items to list
        - setAt: Replace item at index
    """
    for name, operation, payload in updates:
        if operation == "append":
            if name not in state:
                state[name] = []
            state[name] = state[name] + [payload]
        elif operation == "concat":
            if name not in state:
                state[name] = []
            state[name] = state[name] + list(payload)
        elif operation == "reset":
            state[name] = payload
        elif operation == "setAt":
            index, value = payload
            if name not in state:
                state[name] = []
            state[name] = state[name][:index] + [value] + state[name][index + 1 :]
        else:
            raise ValueError(f"Unknown operation: {operation}")
