"""Uniform loading of CLI targets (``.colight`` artifacts and ``.py`` files).

Every CLI tool that accepts a target (inspect, diff, screenshot, verify)
dispatches through :func:`load_target`, so the supported extensions and the
error messages for unsupported/empty targets have a single origin.
"""

import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import colight.format as colight_format
from colight.runtime.executor import BlockExecutor
from colight.runtime.parser import parse_colight_file

from . import blocks as blocks_mod
from . import summaries


@dataclass
class LoadedTarget:
    """A target resolved to its visuals.

    Attributes:
        kind: ``"colight"`` or ``"py"``.
        visuals: Each visual has ``data`` (JSON envelope) and ``buffers``;
            ``.py`` visuals also carry ``block`` (stable id) and ``lines``.
        errors: Evaluation errors (``.py`` targets only), each with
            ``block``, ``lines`` and a structured ``error``.
        updates: Number of update entries (``.colight`` targets only).
    """

    kind: str
    visuals: List[Dict[str, Any]]
    errors: List[Dict[str, Any]] = field(default_factory=list)
    updates: int = 0


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


def load_target(file_path: pathlib.Path) -> LoadedTarget:
    """Load a target's visuals, dispatching on its extension.

    Args:
        file_path: A ``.colight`` artifact (parsed directly; update entries
            are counted but not applied) or a notebook-style ``.py`` file
            (evaluated headlessly).

    Returns:
        The loaded target.

    Raises:
        ValueError: If the target has an unsupported extension or a
            ``.colight`` file has no initial state entry.
    """
    if file_path.suffix == ".colight":
        data, buffers, updates = colight_format.parse_file(file_path)
        if data is None:
            raise ValueError(f"file contains no initial state entry: {file_path}")
        return LoadedTarget(
            kind="colight",
            visuals=[{"data": data, "buffers": buffers}],
            updates=len(updates),
        )
    if file_path.suffix == ".py":
        visuals, errors = evaluate_python_visuals(file_path)
        return LoadedTarget(kind="py", visuals=visuals, errors=errors)
    raise ValueError(f"Unsupported target (expected .colight or .py): {file_path}")


__all__ = ["LoadedTarget", "evaluate_python_visuals", "load_target"]
