"""TypeScript interface generation from Python dataclasses.

This module provides utilities to generate TypeScript type definitions
from Python dataclasses, enabling type-safe communication between
Python backends and TypeScript frontends via colight-serde.

Example:
    from dataclasses import dataclass
    from colight_serde import generate_typescript, Shape

    @dataclass
    class Point3D:
        x: float
        y: float
        z: float

    @dataclass
    class Pose:
        position: Point3D
        quaternion: tuple[float, float, float, float]

    # Generate .d.ts content
    print(generate_typescript(Point3D, Pose))

Shape annotations:
    from typing import Annotated
    import numpy as np
    from numpy.typing import NDArray

    @dataclass
    class Trajectory:
        # Shape encoded in type -> NdArrayView<Float32Array, [number, 7]>
        posquats: Annotated[NDArray[np.float32], Shape(None, 7)]
"""

from __future__ import annotations

import dataclasses
from typing import (
    Any,
    List,
    Set,
    Type,
    get_type_hints,
)

from .handlers import find_handler_for_hint


def _python_type_to_ts(hint: Any, seen: Set[str], known_names: Set[str]) -> str:
    """Convert a Python type hint to TypeScript type string.

    Uses the unified handler registry from handlers.py to ensure
    type generation stays in sync with serialization.

    Args:
        hint: The Python type hint to convert.
        seen: Set of type names that have been referenced (for dependency tracking).
        known_names: Set of type names that are being generated (for resolving references).
    """
    handler = find_handler_for_hint(hint)
    if handler:
        return handler.to_typescript(
            hint, recurse=_python_type_to_ts, seen=seen, known_names=known_names
        )

    # Fallback
    return "any"


def _generate_interface(cls: Type, seen: Set[str], known_names: Set[str]) -> str:
    """Generate TypeScript interface for a single dataclass."""
    name = cls.__name__
    lines = [f"export interface {name} {{"]

    # Always include __serde__ tag for round-tripping
    lines.append(f'  __serde__: "{name}";')

    try:
        # include_extras=True preserves Annotated wrappers
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        # Fallback to __annotations__ if get_type_hints fails
        hints = getattr(cls, "__annotations__", {})

    for field in dataclasses.fields(cls):
        hint = hints.get(field.name, Any)
        ts_type = _python_type_to_ts(hint, seen, known_names)

        # Check if field has default or default_factory (optional in TS)
        optional = ""
        if (
            field.default is not dataclasses.MISSING
            or field.default_factory is not dataclasses.MISSING
        ):
            optional = "?"

        lines.append(f"  {field.name}{optional}: {ts_type};")

    lines.append("}")
    return "\n".join(lines)


def _generate_constructor(cls: Type, known_names: Set[str]) -> str:
    """Generate TypeScript constructor function for a single dataclass.

    The constructor simply creates an object with the __serde__ tag and
    the provided fields. Buffer extraction happens later via packMessage.
    """
    name = cls.__name__

    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = getattr(cls, "__annotations__", {})

    # Build parameter list
    params = []
    for field in dataclasses.fields(cls):
        field_name = field.name
        hint = hints.get(field_name, Any)
        ts_type = _python_type_to_ts(hint, set(), known_names)
        params.append(f"{field_name}: {ts_type}")

    params_str = ", ".join(params)
    fields_str = ", ".join(f.name for f in dataclasses.fields(cls))

    return f'export function {name}({params_str}): {name} {{\n  return {{ __serde__: "{name}", {fields_str} }};\n}}'


def generate_typescript(
    *classes: Type,
    include_imports: bool = True,
) -> str:
    """Generate TypeScript definitions for the specified dataclasses.

    Args:
        *classes: Dataclasses to generate types for.
        include_imports: Whether to include import statement for NdArrayView.

    Returns:
        TypeScript definition file content as a string.

    Example:
        generate_typescript(Point3D, Pose, Trajectory)
    """
    if not classes:
        return "// No types specified\n"

    # Build set of known type names for resolving references
    known_names: Set[str] = {cls.__name__ for cls in classes}

    seen: Set[str] = set()
    interfaces: List[str] = []
    constructors: List[str] = []

    # Generate interfaces in dependency order
    pending = list(classes)
    generated: Set[str] = set()

    while pending:
        cls = pending.pop(0)
        name = cls.__name__

        if name in generated:
            continue

        # Generate interface and constructor
        interfaces.append(_generate_interface(cls, seen, known_names))
        constructors.append(_generate_constructor(cls, known_names))
        generated.add(name)

    # Build output
    parts = []

    # Header
    parts.append("// Auto-generated by colight-serde. Do not edit manually.")
    parts.append("")

    # Imports
    if include_imports:
        parts.append('import type { NdArrayView } from "@colight/serde";')
        parts.append("")

    # Interfaces
    parts.extend(interfaces)

    # Constructors
    if constructors:
        parts.append("")
        parts.extend(constructors)

    return "\n".join(parts) + "\n"


def write_typescript(
    path: str,
    *classes: Type,
    include_imports: bool = True,
) -> None:
    """Write TypeScript definitions to a file.

    Args:
        path: Output file path (typically .d.ts or .ts)
        *classes: Dataclasses to generate types for.
        include_imports: Whether to include import statements.

    Example:
        write_typescript("types.ts", Point3D, Pose, Trajectory)
    """
    content = generate_typescript(
        *classes,
        include_imports=include_imports,
    )
    with open(path, "w") as f:
        f.write(content)
