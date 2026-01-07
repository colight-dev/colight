# colight-serde

Wire protocol utilities for sending NumPy arrays and binary buffers over JSON +
binary channels. Deserialization on the JavaScript side is zero-copy; Python
serialization may materialize contiguous bytes.

See `packages/colight-serde/WIRE_PROTOCOL.md` for the draft spec.

## Quick Usage

Python:

```python
from colight_serde import pack_message, unpack_message
import numpy as np

payload = {"points": np.random.rand(4, 3).astype(np.float32)}
envelope, buffers = pack_message(payload)

# Send envelope JSON, then `buffers` as binary frames.
restored = unpack_message(envelope, buffers)
```

JavaScript:

```ts
import { packMessage, unpackMessage } from "@colight/serde";

const [envelope, buffers] = packMessage({
  points: new Float32Array([0, 1, 2, 3, 4, 5]),
});

const restored = unpackMessage(envelope, buffers);
```
