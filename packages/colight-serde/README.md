# @colight/serde

A wire protocol for sending NumPy arrays and binary data alongside JSON.

## What it does

When you need to send structured data containing arrays from Python to JavaScript (or vice versa), you typically have two choices:

1. Serialize arrays as JSON (slow, bloated - each float becomes a string)
2. Use a binary format for everything (loses JSON's flexibility for metadata)

`@colight/serde` takes a hybrid approach: JSON for structure and metadata, binary buffers sent out-of-band for array data. The JSON contains references to buffer indices, and the receiver reassembles them.

```
Python: {"points": np.array([[1,2,3], [4,5,6]], dtype=float32)}
           ↓
Wire:   JSON: {"points": {"__type__": "ndarray", "__buffer_index__": 0,
                          "dtype": "float32", "shape": [2,3], ...}}
        + Binary: <24 bytes of raw float32 data>
           ↓
JS:     {"points": [[1,2,3], [4,5,6]]}
```

## Key properties

- **Zero-copy on JS side**: Binary buffers map directly to TypedArrays
- **Transparent**: Arrays serialize/deserialize automatically in nested structures
- **Transport-agnostic**: Works over WebSocket, HTTP multipart, or any channel that can carry JSON + binary blobs
- **Bi-directional**: Both Python and JS can pack/unpack messages
- **Type-safe**: Auto-generate TypeScript interfaces from Python dataclasses

## Usage

### Python → JavaScript

```python
from colight_serde import pack_message
import numpy as np

data = {
    "vertices": np.random.rand(1000, 3).astype(np.float32),
    "indices": np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32),
    "name": "mesh",
}
envelope, buffers = pack_message(data)
# Send envelope as JSON text, then each buffer as binary
```

```javascript
import { unpackMessage } from "@colight/serde";

// Receive envelope and buffers from transport
const data = unpackMessage(envelope, buffers);
// data.vertices: NdArrayView (zero-copy Float32Array wrapper)
// data.indices: NdArrayView (zero-copy Uint32Array wrapper)
// data.name: "mesh"
```

### TypeScript Generation

Generate TypeScript interfaces from Python dataclasses:

```python
from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
from colight_serde import generate_typescript, Shape
from typing import Annotated

@dataclass
class Mesh:
    vertices: NDArray[np.float32]
    indices: NDArray[np.uint32]
    name: str

# Generate TypeScript
print(generate_typescript(Mesh))
# interface Mesh {
#   vertices: NdArrayView<Float32Array>;
#   indices: NdArrayView<Uint32Array>;
#   name: string;
# }
```

### JavaScript → Python

```javascript
import { packMessage } from "@colight/serde";

const [envelope, buffers] = packMessage({
    samples: new Float32Array([0.1, 0.2, 0.3, 0.4]),
});
```

```python
from colight_serde import unpack_message

data = unpack_message(envelope, buffers)
# data["samples"] is a numpy array
```

## Wire format

See [WIRE_PROTOCOL.md](./WIRE_PROTOCOL.md) for the specification.

## Install

```bash
npm install @colight/serde   # JavaScript
pip install colight-serde    # Python
```

## License

Apache-2.0
