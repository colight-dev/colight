# Wire Protocol Draft

This document defines the draft JSON+binary wire format used by
`colight-wire-protocol`.

## Overview

- JSON carries structure and metadata.
- Binary buffers carry raw bytes out-of-band (WebSocket binary frames, HTTP
  multipart, etc.).
- A single JSON envelope references one or more buffers.

## Message Envelope

Every JSON message should be wrapped in an envelope so receivers can associate
the correct number of binary frames with the JSON payload.

```json
{
  "message_id": "7f3c3f2b-8f62-4f8a-8d0e-5e4c2a1f7b9a",
  "buffer_count": 2,
  "payload": { "type": "update", "data": { "...": "..." } }
}
```

- `message_id` (string | number): Unique per message on a connection. A UUID or
  monotonic counter are both fine.
- `buffer_count` (number): Number of binary buffers that immediately follow
  this JSON message on the transport.
- `payload` (any): Application-level JSON data. This is where ndarray metadata
  and other structured data live.

## Binary Buffer References

Any JSON value can be replaced with a buffer reference. The referenced buffer
is stored in the out-of-band buffer list at `__buffer_index__`.

```json
{ "__buffer_index__": 0 }
```

## Ndarray References

ndarray values include both a buffer reference and array metadata.

```json
{
  "__type__": "ndarray",
  "__buffer_index__": 0,
  "dtype": "float32",
  "shape": [100, 3],
  "order": "C",
  "strides": [12, 4]
}
```

- `dtype`: NumPy dtype string (e.g. `float32`, `int64`).
- `shape`: Array shape.
- `order`: `"C"` or `"F"`, describing how bytes are laid out in the buffer.
- `strides`: Byte strides for each dimension. If omitted, consumers should
  assume the buffer is contiguous in `order`.

### Contiguity Notes

- Senders may materialize a contiguous buffer when the input view is
  non-contiguous. In that case, `strides` should describe the contiguous buffer
  layout, not the original view.
- JavaScript consumers can reconstruct nested arrays using `shape` and
  `strides`, but non-contiguous views require copying.

## Transport Expectations

- The JSON envelope is always sent first.
- The next `buffer_count` frames are the raw buffers, in index order.
- For WebSockets, the common pattern is:
  1. Send JSON text frame.
  2. Send `buffer_count` binary frames.

