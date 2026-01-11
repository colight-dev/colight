# colight-serde: Python-to-TypeScript Serialization Guide

A developer's guide to the colight-serde system for efficient, type-safe data transfer between Python backends and TypeScript frontends.

## Overview

colight-serde provides an end-to-end solution for serializing complex Python data structures—especially those containing large numpy arrays—and consuming them in TypeScript with type safety. The system is designed for interactive visualization and real-time applications where:

- **Performance matters**: Binary array data avoids JSON encoding/decoding overhead. On the Python side, arrays are sent as raw bytes via `memoryview` (no copy). On the TypeScript side, `NdArrayView` wraps the received `ArrayBuffer` directly as a TypedArray without copying the data.
- **Type safety matters**: TypeScript interfaces are auto-generated from Python dataclasses
- **Developer experience matters**: The API is minimal and declarative

## Architecture

The system has two components that share a **unified handler registry**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Python Backend                                │
│                                                                         │
│   @dataclass             ──────►  pack_message()  ──────►  JSON envelope│
│   class Scenario:                 (runtime)                + binary     │
│       poses: NDArray[float32]          │                     buffers    │
│                                        │                                │
│                              ┌─────────┴─────────┐                      │
│                              │  Handler Registry │                      │
│                              │  (handlers.py)    │                      │
│                              └─────────┬─────────┘                      │
│                                        │                                │
│   generate_typescript()  ──────────────┘         ──────►  types.ts      │
│   (build time)                                            interface ... │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ WebSocket / HTTP
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TypeScript Frontend                             │
│                                                                         │
│   interface Scenario {      unpackMessage()  ◄──────  JSON envelope     │
│     poses: NdArrayView<Float32Array>;            + binary buffers       │
│   }                            │                                        │
│                         scenario.poses[0][3]  // indexed access         │
│                         scenario.poses.flat   // raw Float32Array       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: The serializer (`pack_message`) and type generator (`generate_typescript`) both use the same **handler registry**. Each handler knows how to serialize values _and_ generate TypeScript types, ensuring they stay in sync.

## Shared Type Conventions

Both the serializer and type generator follow these mapping rules:

| Python Type                           | Wire Format                     | TypeScript Type             |
| ------------------------------------- | ------------------------------- | --------------------------- |
| `NDArray[np.float32]`                 | binary buffer + dtype="float32" | `NdArrayView<Float32Array>` |
| `NDArray[np.float64]`                 | binary buffer + dtype="float64" | `NdArrayView<Float64Array>` |
| `NDArray[np.int32]`                   | binary buffer + dtype="int32"   | `NdArrayView<Int32Array>`   |
| `NDArray[np.int8]`                    | binary buffer + dtype="int8"    | `NdArrayView<Int8Array>`    |
| `NDArray[np.uint8]`                   | binary buffer + dtype="uint8"   | `NdArrayView<Uint8Array>`   |
| `Annotated[NDArray[...], Shape(...)]` | binary buffer                   | `NdArrayView<T, [shape]>`   |
| `int`, `float`                        | JSON number                     | `number`                    |
| `str`                                 | JSON string                     | `string`                    |
| `bool`                                | JSON boolean                    | `boolean`                   |
| `list[T]`                             | JSON array                      | `T[]`                       |
| `tuple[A, B, ...]`                    | JSON array                      | `[A, B, ...]`               |
| `dict[K, V]`                          | JSON object                     | `{ [key: K]: V }`           |
| `Optional[T]`                         | JSON value or null              | `T \| null`                 |
| `Union[A, B]`                         | JSON value                      | `A \| B`                    |
| `Literal["a", "b"]`                   | JSON value                      | `"a" \| "b"`                |
| `bytes`, `bytearray`                  | binary buffer                   | `ArrayBuffer`               |
| `@dataclass`                          | JSON object                     | `interface`                 |
| `Any`                                 | JSON value                      | `any`                       |

**Important**: Use `NDArray[np.dtype]` type hints to get precise TypeScript types. Generic `np.ndarray` produces untyped `NdArrayView`.

## Wire Protocol

Messages are sent as a JSON envelope followed by binary buffer frames:

```
Frame 1: JSON envelope
{
  "message_id": "abc123",
  "buffer_count": 3,
  "payload": {
    "poses": {
      "__type__": "ndarray",
      "__buffer_index__": 0,
      "dtype": "float32",
      "shape": [10, 7],
      "strides": [28, 4],
      "order": "C"
    },
    ...
  }
}

Frame 2: Binary buffer 0 (poses data)
Frame 3: Binary buffer 1 (...)
Frame 4: Binary buffer 2 (...)
```

## Python API

### 1. Define Types as Dataclasses

Use `NDArray[dtype]` for precise TypeScript types:

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class Pose:
    """6DOF pose: position + quaternion."""
    posquat: NDArray[np.float32]  # -> NdArrayView<Float32Array>

@dataclass
class Keypoints:
    """Tracked keypoints with indices."""
    positions: NDArray[np.float32]   # -> NdArrayView<Float32Array>
    object_ids: NDArray[np.int32]    # -> NdArrayView<Int32Array>

@dataclass
class Scenario:
    """Multi-camera tracking scenario."""
    camera_poses: Pose
    object_poses: Pose
    keypoints: Keypoints
```

Any dataclass can be serialized—no decorator required.

### 2. Serialize with `pack_message()`

```python
from colight_serde import pack_message
import numpy as np

scenario = Scenario(
    camera_poses=Pose(posquat=np.zeros((10, 2, 7), dtype=np.float32)),
    object_poses=Pose(posquat=np.zeros((10, 5, 7), dtype=np.float32)),
    keypoints=Keypoints(
        positions=np.random.randn(100, 3).astype(np.float32),
        object_ids=np.arange(100, dtype=np.int32),
    ),
)

envelope, buffers = pack_message(scenario)
# envelope: JSON-serializable dict
# buffers: list of memoryview objects (raw array data)
```

### 3. Generate TypeScript Types

Pass your dataclasses directly to `generate_typescript()`:

```python
from colight_serde import generate_typescript, write_typescript

# Generate to string
ts_code = generate_typescript(Pose, Keypoints, Scenario)

# Or write directly to file
write_typescript("web/types.ts", Pose, Keypoints, Scenario)
```

Output:

```typescript
// Auto-generated by colight-serde. Do not edit manually.

import type { NdArrayView } from "@colight/serde";

export interface Pose {
  __serde__: "Pose";
  posquat: NdArrayView<Float32Array>;
}
export interface Keypoints {
  __serde__: "Keypoints";
  positions: NdArrayView<Float32Array>;
  object_ids: NdArrayView<Int32Array>;
}
export interface Scenario {
  __serde__: "Scenario";
  camera_poses: Pose;
  object_poses: Pose;
  keypoints: Keypoints;
}

export function Pose(posquat: NdArrayView<Float32Array>): Pose {
  return { __serde__: "Pose", posquat };
}
export function Keypoints(
  positions: NdArrayView<Float32Array>,
  object_ids: NdArrayView<Int32Array>,
): Keypoints {
  return { __serde__: "Keypoints", positions, object_ids };
}
export function Scenario(
  camera_poses: Pose,
  object_poses: Pose,
  keypoints: Keypoints,
): Scenario {
  return { __serde__: "Scenario", camera_poses, object_poses, keypoints };
}
```

The `__serde__` tag enables round-trip deserialization back to Python dataclasses.

### 4. Array Shape Annotations

Use `Shape` with `Annotated` to include array shape information in TypeScript types:

```python
from typing import Annotated
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from colight_serde import Shape

@dataclass
class Pose:
    # Fixed shape [7] -> NdArrayView<Float32Array, [7]>
    posquat: Annotated[NDArray[np.float32], Shape(7)]

@dataclass
class PointCloud:
    # Dynamic first dim, fixed second -> NdArrayView<Float32Array, [number, 3]>
    points: Annotated[NDArray[np.float32], Shape(None, 3)]

    # Single dynamic dimension -> NdArrayView<Float32Array, [number]>
    weights: Annotated[NDArray[np.float32], Shape(None)]
```

The `Shape` annotation is purely for TypeScript generation—it doesn't affect serialization (actual shapes come from the numpy array at runtime).

### 5. Array Framework Support

colight-serde automatically handles arrays from JAX, PyTorch, TensorFlow, and Warp:

```python
import jax.numpy as jnp

scenario = Scenario(
    camera_poses=Pose(posquat=jnp.zeros((10, 2, 7))),  # JAX array - auto-converted
    ...
)
```

For dynamically-imported frameworks:

```python
from colight_serde import register_jax, register_torch
register_jax()   # Enable JAX support
register_torch() # Enable PyTorch support
```

## TypeScript API

### 1. Install the Package

```json
{
  "dependencies": {
    "@colight/serde": "file:path/to/colight-serde"
  }
}
```

### 2. Deserialize Messages

```typescript
import { unpackMessage } from "@colight/serde";
import type { Scenario } from "./types";

// Receive envelope and buffers (e.g., over WebSocket)
const scenario = unpackMessage(envelope, buffers) as Scenario;
```

### 3. Use NdArrayView

`NdArrayView` provides zero-copy method-based access to multidimensional arrays:

```typescript
const posquat = scenario.camera_poses.posquat;

// Direct element access via .get()
const x = posquat.get(0, 0, 0); // number
const y = posquat.get(0, 0, 1); // number

// Sub-views via .slice() (no data copy)
const firstCamera = posquat.slice(0); // NdArrayView<Float32Array, [2, 7]>
const firstPose = posquat.slice(0).slice(0); // NdArrayView<Float32Array, [7]>

// Raw typed array access (for WebGL, computations, etc.)
const flat: Float32Array = posquat.flat;

// Shape and metadata
posquat.shape; // [10, 2, 7]
posquat.strides; // [14, 7, 1]
posquat.ndim; // 3
posquat.length; // 140 (total elements)

// Fast iteration via callbacks
const sum = posquat.reduce((acc, v) => acc + v, 0);
posquat.forEach((value, i, j, k) => {
  /* ... */
});

// Fast 2D row iteration (for matrix operations)
const rowSums = matrix.mapRows((row) => row.reduce((a, v) => a + v, 0));
```

### 4. WebSocket Integration

```typescript
function useWireSocket(url: string) {
  const [data, setData] = useState<Scenario | null>(null);
  const pending = useRef<{ envelope: any; buffers: ArrayBuffer[] } | null>(
    null,
  );

  useEffect(() => {
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onmessage = (event) => {
      if (typeof event.data === "string") {
        const envelope = JSON.parse(event.data);
        if (envelope.buffer_count > 0) {
          pending.current = { envelope, buffers: [] };
        } else {
          setData(unpackMessage(envelope, []) as Scenario);
        }
        return;
      }

      // Binary frame
      pending.current!.buffers.push(event.data);
      if (
        pending.current!.buffers.length ===
        pending.current!.envelope.buffer_count
      ) {
        const { envelope, buffers } = pending.current!;
        pending.current = null;
        setData(unpackMessage(envelope, buffers) as Scenario);
      }
    };

    return () => ws.close();
  }, [url]);

  return data;
}
```

## Testing

### Python Tests

```python
from colight_serde import pack_message, unpack_message
import numpy as np

def test_roundtrip():
    scenario = Scenario(...)
    envelope, buffers = pack_message(scenario)
    restored = unpack_message(envelope, buffers)

    np.testing.assert_array_equal(
        restored["camera_poses"]["posquat"],
        scenario.camera_poses.posquat
    )
```

### TypeScript Tests

```typescript
import { describe, it, expect } from "vitest";
import { unpackMessage } from "@colight/serde";
import type { Scenario } from "../web/types";

describe("Scenario deserialization", () => {
  it("deserializes with correct typed arrays", () => {
    const envelope = require("./fixtures/scenario.json");
    const buffers = readBuffers("./fixtures/scenario.bin");

    const scenario = unpackMessage(envelope, buffers) as Scenario;

    // Shape is correct
    expect(scenario.camera_poses.posquat.shape).toEqual([10, 2, 7]);

    // Underlying array has correct type
    expect(scenario.camera_poses.posquat.flat).toBeInstanceOf(Float32Array);
    expect(scenario.keypoints.object_ids.flat).toBeInstanceOf(Int32Array);
  });
});
```

## Best Practices

1. **Use typed array hints**: `NDArray[np.float32]` instead of bare `np.ndarray` for precise TypeScript types.

2. **Prefer float32/int32**: These map directly to TypeScript typed arrays. int64/uint64 require BigInt conversion.

3. **Batch data when possible**: Instead of sending many small messages, batch arrays along a leading dimension.

4. **Generate types at build time**: Run `write_typescript()` as part of your build process, not at runtime.

5. **Test the full roundtrip**: Write tests that serialize in Python and deserialize in TypeScript to catch schema drift early.

6. **Leverage the unified handler system**: Serialization and TypeScript generation use the same handlers, so they stay in sync automatically.

## Handler System (Advanced)

The handler registry in `handlers.py` is the core of colight-serde. Each handler implements both serialization and TypeScript generation for a specific type category.

### Built-in Handlers

| Handler              | Matches                               | Serialization                 | TypeScript               |
| -------------------- | ------------------------------------- | ----------------------------- | ------------------------ |
| `ArrayHandler`       | numpy arrays, JAX, PyTorch, etc.      | Binary buffer with metadata   | `NdArrayView<T>`         |
| `DataclassHandler`   | `@dataclass` instances                | Recursive field serialization | Interface reference      |
| `PrimitiveHandler`   | `int`, `float`, `str`, `bool`, `None` | Pass-through                  | `number`, `string`, etc. |
| `NumpyScalarHandler` | `np.float32`, `np.int64`, etc.        | `.item()` conversion          | `number`                 |
| `BytesHandler`       | `bytes`, `bytearray`, `memoryview`    | Binary buffer                 | `ArrayBuffer`            |
| `DictHandler`        | `dict`                                | Recursive serialization       | `Record<K, V>`           |
| `ListHandler`        | `list`, `tuple`                       | Recursive serialization       | `T[]` or `[A, B]`        |
| `UnionHandler`       | `Union`, `Optional`                   | (type-level only)             | `A \| B`                 |
| `AnnotatedHandler`   | `Annotated[T, Shape(...)]`            | (delegates to inner)          | `NdArrayView<T, shape>`  |
| `LiteralHandler`     | `Literal["a", "b"]`                   | (type-level only)             | `"a" \| "b"`             |
| `ForwardRefHandler`  | Forward references                    | (type-level only)             | Type name                |
| `AnyHandler`         | `Any`                                 | (type-level only)             | `any`                    |

### Handler Interface

Each handler implements `TypeHandler`:

```python
class TypeHandler(ABC):
    def matches_value(self, value: Any) -> bool:
        """Check if this handler can serialize the given value."""
        ...

    def matches_hint(self, hint: Any) -> bool:
        """Check if this handler can generate TypeScript for the given hint."""
        ...

    def serialize(self, value, collector, recurse) -> Any:
        """Serialize value to wire format."""
        ...

    def to_typescript(self, hint, recurse, seen, known_names) -> str:
        """Generate TypeScript type string."""
        ...

    def deserialize(self, value, hint, buffers, recurse) -> Any:
        """Deserialize wire format back to Python value."""
        ...
```

The `recurse` parameter allows handlers to delegate nested types back to the registry.

## Bi-directional Communication

colight-serde supports full round-trip serialization. Messages created in TypeScript can be deserialized back to Python dataclasses.

### Register Types for Deserialization

```python
from colight_serde import register_types, unpack_message

# Register your dataclasses
register_types(Pose, Keypoints, Scenario)

# Now unpack_message can reconstruct dataclass instances
envelope, buffers = receive_from_websocket()
scenario = unpack_message(envelope, buffers)  # Returns Scenario instance, not dict
```

### TypeScript → Python Flow

```typescript
import { packMessage, Scenario, Pose, Keypoints } from "./types";

// Use generated constructor functions
const scenario = Scenario(
  Pose(new Float32Array([0, 0, 0, 0, 0, 0, 1])),
  Pose(new Float32Array([1, 0, 0, 0, 0, 0, 1])),
  Keypoints(new Float32Array([...]), new Int32Array([...]))
);

// Pack and send
const [envelope, buffers] = packMessage(scenario);
ws.send(JSON.stringify(envelope));
buffers.forEach(b => ws.send(b));
```

```python
# Python receives and reconstructs the dataclass
scenario = unpack_message(envelope, buffers)
assert isinstance(scenario, Scenario)
assert isinstance(scenario.camera_poses, Pose)
```
