# The `.colight` File Format

This document is the authoritative specification of the `.colight` binary
format, **version 2**. It is the language-neutral boundary between the Python
authoring side and the JS/WebGPU rendering side. A reader for a new language
should be implementable from this document alone, without consulting the
Colight source code.

Reference implementations:

- Writer/reader (Python): `packages/colight/src/colight/format.py`
- Reader (JavaScript): `packages/colight/src/js/format.js`

## 1. Overview

A `.colight` file is a sequence of one or more self-contained **entries**
concatenated back to back. Each entry carries:

1. a fixed-size 96-byte **header**,
2. a UTF-8 **JSON section** (the payload: AST, state, metadata, and the
   buffer layout table),
3. an optional **binary section** containing raw byte buffers referenced from
   the JSON by index.

The first entry normally holds the _initial state_ of a visual. Additional
entries appended later are _updates_ (used by `colight render` video frames
and live/tailing workflows). Appending an update never rewrites existing
bytes — an update entry is simply written to the end of the file.

All multi-byte integers in the header are **unsigned 64-bit little-endian**
(`uint64le`). All offsets inside an entry are relative to the start of that
entry unless stated otherwise.

## 2. Entry layout

### 2.1 Header (96 bytes)

| Offset | Size | Type     | Field           | Value / meaning                                                                                                                               |
| ------ | ---- | -------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 0      | 8    | bytes    | `magic`         | `"COLIGHT\x00"` — hex `43 4F 4C 49 47 48 54 00`                                                                                               |
| 8      | 8    | uint64le | `version`       | Format version. Currently `2`. See [Versioning](#5-versioning).                                                                               |
| 16     | 8    | uint64le | `json_offset`   | Offset of the JSON section from the entry start. The writer always emits `96` (JSON immediately follows the header).                          |
| 24     | 8    | uint64le | `json_length`   | Length of the JSON section in bytes (exact, excludes padding).                                                                                |
| 32     | 8    | uint64le | `binary_offset` | Offset of the binary section from the entry start. Always `(json_offset + json_length)` rounded **up** to the next multiple of 8.             |
| 40     | 8    | uint64le | `binary_length` | Length of the binary section in bytes.                                                                                                        |
| 48     | 8    | uint64le | `num_buffers`   | Number of buffers in the binary section. Must equal the lengths of `bufferLayout.offsets` and `bufferLayout.lengths` in the JSON (see below). |
| 56     | 40   | bytes    | reserved        | Writers MUST write zeros. Readers MUST ignore.                                                                                                |

### 2.2 Section order and framing

Sections MUST appear in this order, with these invariants:

```text
entry_start
├── header            96 bytes
├── JSON section      json_length bytes of UTF-8 JSON (one JSON object)
├── zero padding      (binary_offset − json_offset − json_length) bytes, all 0x00
├── binary section    binary_length bytes
├── zero padding      to the next multiple of 8, all 0x00
entry_end = entry_start + align8(binary_offset + binary_length)
```

(`align8(n)` rounds `n` up to the next multiple of 8.)

- `json_offset = 96` (readers use the header value, but the writer never
  emits anything else).
- `binary_offset ≥ json_offset + json_length`, and `binary_offset` is a
  multiple of 8. The gap is zero padding. The padding is written even when
  the entry has no buffers.
- **Entry size** is defined as `align8(binary_offset + binary_length)`: the
  writer pads every entry with trailing zeros to an 8-byte boundary so that
  every entry — including appended updates — starts at an absolute offset
  that is a multiple of 8. The next entry (if any) begins at exactly
  `entry_start + entry_size`. There is no entry count field and no
  terminator: readers walk entries sequentially until end of data.
- The JSON section is a single JSON **object**, encoded as UTF-8 without a
  BOM. The Python writer uses compact separators (`","`/`":"`) but readers
  MUST NOT rely on any particular JSON whitespace.
- Practical limit: although header fields are uint64, JavaScript readers
  convert them to IEEE doubles, so all offsets and lengths must stay below
  2^53. In practice files are far smaller.

### 2.3 Binary section and alignment

The binary section is a concatenation of `num_buffers` raw buffers, in
buffer-index order, with zero padding between them such that:

- each buffer starts at an offset that is a multiple of 8 **relative to the
  start of the binary section**, and
- every entry starts at a multiple of 8 relative to the start of the file
  (entry sizes are padded — see 2.2), the header is 96 bytes, and the JSON
  section is padded, so the binary section of **every** entry starts at a
  multiple of 8 **relative to the start of the file**.

Together these guarantee that every buffer of every entry is 8-byte aligned
relative to the file start, which allows zero-copy typed-array views over an
in-memory copy of the file for every element type up to `float64`. The
reference JS reader relies on this: it always constructs typed arrays
directly over the file bytes without copying.

Buffer positions are described by the `bufferLayout` object in the same
entry's JSON:

```json
"bufferLayout": {
  "offsets": [0, 48],   // per-buffer offset relative to the binary section start
  "lengths": [40, 3],   // per-buffer exact length in bytes (excludes padding)
  "count": 2,           // number of buffers (informational; must equal num_buffers)
  "totalSize": 51       // last offset + last length (informational)
}
```

`bufferLayout` is present **iff the entry has at least one buffer**. Readers
MUST error if `offsets` or `lengths` length differs from the header's
`num_buffers`, or if any `offset + length` exceeds `binary_length`.

## 3. JSON envelope

### 3.1 Initial entries vs update entries

An entry whose JSON object contains a top-level `"updates"` key is an
**update entry**; any other entry is an **initial-state entry**.

File-level semantics (both reference readers implement exactly this):

- The **first** entry, if it is not an update entry, provides the initial
  state. A file may also _start_ with an update entry, in which case there is
  no initial state (an "updates-only" file, e.g. produced for live tailing).
- Every update entry, wherever it appears, is collected in order into a list
  of updates, each paired with its own entry's buffers.
- A non-first entry that is _not_ an update entry is silently ignored.
- A malformed **first** entry (bad magic, unsupported version, truncation)
  is an error. A malformed entry **after** the first terminates the walk
  without error — this tolerance is deliberate, so that a reader can consume
  a file whose latest update is still being appended.

### 3.2 Initial-state envelope

The initial entry's JSON object as produced by the Python writer
(`colight.widget.to_json_with_state`) has these keys:

| Key                             | Type          | Meaning                                                                    |
| ------------------------------- | ------------- | -------------------------------------------------------------------------- |
| `ast`                           | any           | The serialized visual (Colight's evaluatable AST).                         |
| `id`                            | string\|null  | Widget/visual DOM id.                                                      |
| `state`                         | object        | Initial state map: state key → serialized value (may contain buffer refs). |
| `syncedKeys`                    | array<string> | State keys mirrored back to Python when changed.                           |
| `listeners`                     | object        | JS-side state listeners (serialized).                                      |
| `py_listeners`                  | object        | Python-side state listeners (callback refs).                               |
| `imports`                       | array         | JS import specifications to load before evaluating.                        |
| `animateBy`                     | array         | Animation metadata, see [3.5](#35-animateby-metadata).                     |
| `display_as`, `dev`, `defaults` | —             | Environment configuration flags merged in from Python-side config.         |
| `bufferLayout`                  | object        | See [2.3](#23-binary-section-and-alignment). Present iff buffers exist.    |

Of these, only `bufferLayout` and the buffer-reference conventions below are
part of the _container_ contract; the rest are the Colight client's payload
schema. A minimal foreign reader that only wants the raw data needs
`bufferLayout` plus the `__buffer_index__` / `ndarray` conventions; a full
renderer needs the whole envelope.

### 3.3 Buffer references and ndarray encoding

Anywhere inside the JSON payload, two object shapes reference the entry's
binary buffers by zero-based index:

**Raw bytes** (Python `bytes`/`bytearray`/`memoryview`):

```json
{ "__buffer_index__": 0 }
```

**N-dimensional array** (NumPy and NumPy-convertible arrays):

```json
{
  "__type__": "ndarray",
  "data": null,
  "dtype": "float32",
  "shape": [2, 5],
  "__buffer_index__": 1
}
```

- `dtype` — one of: `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`,
  `uint32`, `uint64`, `float32`, `float64`. The buffer holds the elements
  **little-endian**, densely packed (no strides), in **C (row-major) order**.
  The Python writer enforces this: big-endian NumPy arrays are converted to
  little-endian at write time (a cheap copy, only when needed), and the
  `dtype` string is always the canonical name above — never a
  byte-order-qualified spelling like `">f4"`. Readers MUST error loudly on a
  `dtype` they do not recognize rather than guessing an element type.
- `shape` — array of non-negative integers; the product of the dimensions
  times the element size equals the buffer's length in bytes.
- `data` — always `null` in files. (The same envelope is used on transports
  where binary buffers travel out-of-band, e.g. Jupyter widget comms; `data`
  is the pre-extraction slot and files never populate it.)
- Zero-dimensional arrays are never written; Python serializes them as plain
  JSON scalars.
- JavaScript notes: `int64`/`uint64` decode to `BigInt64Array`/
  `BigUint64Array` and are then converted to plain numbers, so values with
  magnitude above 2^53 lose precision in JS clients.

### 3.4 Other `__type__` tags

The AST and state values use additional tagged objects. These are
application-level (they do not reference the binary section) but are listed
here so a reader knows the vocabulary:

| Tag                    | Shape                                                           | Meaning                                                          |
| ---------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| `datetime`             | `{"__type__", "value": ISO-8601 string}`                        | Date or datetime.                                                |
| `ref`                  | `{"__type__", "state_key": string}`                             | Reference to an entry in `state`.                                |
| `callback`             | `{"__type__", "id": string}`                                    | Python callback (live widget only; inert when rendering a file). |
| `function`             | `{"__type__", "path", "args", ...}`                             | Colight AST node: JS function call.                              |
| `js_ref` / `js_source` | `{"__type__", "path"}` / `{"__type__", "expression"?, "value"}` | Colight AST nodes: JS references / inline JS.                    |

### 3.5 `animateBy` metadata

`animateBy` is a list (usually empty or singleton) describing sliders marked
for offline animation. Each element:

```json
{ "key": "frame", "range": [0, 29], "fps": 24, "step": 1 }
```

- `key` — the state key the animation drives.
- `range` — inclusive `[from, to]` integer range (already resolved; a
  Python-side `rangeFrom` is resolved against the initial state length at
  write time).
- `fps` — suggested frames per second (may be `null`; consumers default to 24).
- `step` — increment between frames (may be `null`, meaning 1).

`colight render video` uses this when a file has no update entries: it
generates one state update `{key: i}` per value of `i` in
`range(from, to+1, step)` and captures a frame after each. If a file has
multiple `animateBy` entries, the render CLI refuses and asks for explicit
updates.

### 3.6 Update entries

An update entry's JSON object is:

```json
{
  "updates": {
    /* same envelope shape as 3.2, without bufferLayout */
  },
  "bufferLayout": {
    /* references THIS entry's binary section, if any */
  }
}
```

The value of `updates` is produced by the same serializer as the initial
envelope, so it contains `ast`, `state`, `syncedKeys`, `listeners`,
`imports`, `animateBy`, etc. Buffer references inside `updates` index into
the **update entry's own** buffers (indices restart at 0 for each entry).

Semantics of an update, in order of application:

1. `imports` are loaded and `state` entries are merged into the environment
   (new keys added; listeners merged).
2. If `ast` is `null`, the update is **state-only**: the consumer applies
   every entry of `state` as a `"reset"` (overwriting existing keys, not
   just backfilling new ones).
3. If `ast` is a new visual AST, the consumer re-renders with it.
4. If `ast` is a list of update operations `[key, op, payload]` — `op` one
   of `"reset"` (replace value), `"append"` (add one item to a list),
   `"concat"` (add many items), `"setAt"` (`payload = [index, value]`) —
   they are applied to state.

The `colight render` CLI treats each update entry as one video frame.

## 4. Writing

Writer algorithm (what `colight.format.create_bytes` does):

1. Lay out buffers: walk the buffer list, padding the running offset up to a
   multiple of 8 before each buffer; record `offsets`/`lengths`.
2. If there are buffers, add `bufferLayout` (with `count` and `totalSize`) to
   the JSON object.
3. Encode the JSON as compact UTF-8; `json_offset = 96`;
   `binary_offset = align8(96 + json_length)`.
4. Emit header, JSON bytes, zero padding to `binary_offset`, then each buffer
   preceded by its zero padding.
5. Pad the entry with zeros to `align8(binary_offset + binary_length)`, so
   the next appended entry starts 8-byte aligned.

To append an update: serialize the update payload, wrap it as
`{"updates": ...}`, run the same algorithm, and append the resulting bytes to
the file (`colight.format.append_update`).

## 5. Versioning

- The format version is the single `uint64le` at header bytes 8–15 of
  **every** entry. The current version is **2**. A single integer (rather
  than semver) is used deliberately: the format is pre-release and every
  change is breaking-by-default, so there is nothing for minor/patch
  components to express.
- Readers MUST accept exactly the version(s) they implement and MUST fail
  loudly otherwise, naming both the found and the supported version. The
  reference readers raise
  `Unsupported .colight file version: found N, this reader supports version 2`.
- Policy (pre-release): any change to the byte layout, the framing rules, or
  the required JSON contract (`bufferLayout`, buffer-reference envelopes,
  `updates` wrapping) requires bumping `CURRENT_VERSION` in
  `colight/format.py` **and** `js/format.js`, plus updating this document.
  No silent format drift: if the bytes change, the number changes.
- Backward-compatibility windows may be introduced after release; today
  there is exactly one supported version. Version 2 readers deliberately do
  **not** read version 1 files: at the time of the bump no `.colight` files
  existed outside locally regenerable artifacts (nothing is committed to the
  repository and there are no external consumers), so a compatibility window
  would have been dead code. Regenerate any local v1 artifact from its
  source.

Version history:

- **2** (2026-07) — entry sizes are padded to a multiple of 8 so appended
  entries (and all their buffers) stay 8-byte aligned absolutely; the writer
  normalizes array byte order to little-endian and always emits canonical
  dtype names.
- **1** — initial format.

## 6. Embedding variants

The same bytes travel in three ways besides a standalone `.colight` file:

- **Inline script tag** — the whole file (all entries) base64-encoded as the
  text content of `<script type="application/x-colight">`, optionally with a
  `data-target="<element id>"` attribute naming the container to render
  into. Whitespace around the base64 text is ignored. Produced by
  `colight.html.html_snippet` / the static site generator; consumed by
  `parseColightScript` in `js/format.js`.
- **External fetch** — `<div class="colight-embed" data-src="url">`; the URL
  returns the raw file bytes (served with content type
  `application/x-colight`). Consumed by `js/embed.js`.
- **Jupyter widget comms** — the JSON envelope is sent as the anywidget
  model with buffers out-of-band; the container format (header, sections) is
  not used on this path, but the buffer-reference envelopes of 3.3 are
  identical.

## 7. Reading a `.colight` file in a new language

A complete reader, from this document alone:

1. **Load** the bytes. If they came from a `<script type="application/x-colight">`
   tag, base64-decode the trimmed text content first.
2. **Walk entries.** Set `offset = 0`, `first = true`. While at least 96
   bytes remain past `offset`:
   1. Verify bytes `offset..offset+8` equal `43 4F 4C 49 47 48 54 00`.
   2. Read the six `uint64le` header fields at offsets 8, 16, 24, 32, 40, 48
      (version, json_offset, json_length, binary_offset, binary_length,
      num_buffers).
   3. Verify `version == 2`; otherwise fail with found-vs-supported versions.
   4. Decode `json_length` bytes at `offset + json_offset` as UTF-8 and
      parse as JSON → `entry_json`.
   5. If `num_buffers > 0`: read `entry_json.bufferLayout`; verify
      `offsets` and `lengths` both have `num_buffers` elements; for each
      `i`, take the byte range
      `[offset + binary_offset + offsets[i], … + lengths[i])` (verify it
      lies inside the binary section) → `buffers[i]`.
   6. If any of the above fails and `first` is true, raise; if `first` is
      false, stop walking (partially appended tail).
   7. If `first` and `entry_json` has no `"updates"` key: this is the
      initial state (`initial = entry_json`, `initial_buffers = buffers`).
      Else if `entry_json` has `"updates"`: append
      `{data: entry_json.updates, buffers}` to the updates list. Else:
      ignore the entry.
   8. `offset += align8(binary_offset + binary_length)`; `first = false`.
3. **Materialize arrays.** Walk `initial` (and each update's `data`)
   recursively; replace `{"__buffer_index__": i}` with `buffers[i]`, and
   ndarray envelopes with an array of element type `dtype` (little-endian,
   C-order) over `buffers[i]`, reshaped to `shape`.
4. Interpret `ast`/`state`/`animateBy` per sections 3.2–3.6 as far as your
   application requires.

A reader that stops after step 3 already has every number and byte the file
contains; steps beyond that are Colight application semantics.

### What `@colight/scene3d` consumes

The standalone `@colight/scene3d` npm package does **not** parse `.colight`
files. It accepts JavaScript typed arrays / nested arrays as React props
(`packages/colight/src/js/scene3d/coercion.ts` normalizes them). In the full
Colight client, the pipeline is: `js/format.js` parses the container →
`js/eval.js` resolves buffer references and decodes ndarray envelopes
(`js/binary.ts`) → the resulting typed arrays are handed to scene3d. So
scene3d depends on this spec only indirectly, via the ndarray decoding rules
in section 3.3.

## 8. Known discrepancies

Differences between this spec's intent and the current implementations, kept
here rather than silently fixed. Each requires an explicit decision (and, if
bytes change, a version bump).

1. **`bufferLayout.count` and `totalSize` are unvalidated.** Both readers
   validate `num_buffers` against the _lengths of_ `offsets`/`lengths` but
   never check `count` or `totalSize`; they are informational duplicates
   (as is `num_buffers` itself relative to the JSON).
2. **`int64`/`uint64` lose precision in JS** beyond ±2^53 (converted from
   BigInt arrays to plain numbers).
3. **Python `parse_file` drops update buffers.** `parse_file` returns update
   JSON only; callers that need an update's buffers must use
   `parse_file_with_updates`. Internal API asymmetry, not a wire-format
   issue.
4. **Historical version tolerance.** Before 2026-07 both readers accepted
   any version `≤ 1` (i.e. also the never-issued version 0) and Python's
   `parse_file` swallowed _all_ first-entry errors, returning an empty
   result for arbitrary garbage. Both readers now require exactly the
   current version and propagate first-entry errors; error tolerance after
   the first entry (the walk rule in section 3.1) is retained by design.

Resolved in version 2 (2026-07): appended entries breaking absolute 8-byte
alignment (entries are now padded, section 2.2); state-only update entries
crashing the JS widget (`applyUpdateEntries` now applies `ast: null` entries
as state resets, section 3.6); unenforced endianness (the writer now converts
big-endian arrays to little-endian, section 3.3); unknown dtypes silently
decoding as `float64` in JS (`evaluateNdarray` now errors loudly,
section 3.3).
