# `.colight` format writer (JavaScript/TypeScript)

A dependency-free writer for the [`.colight` binary container format][spec],
version 2. It has no dependency on the Colight widget or any other Colight
JavaScript, so a Node producer can take it standalone.

Colight's Python side has always been able to write `.colight` files and its
JavaScript side to read them. This package closes the loop: **JavaScript can now
write them too**, which is what a non-Python producer (a Node service, a browser
tool, a CLI in another runtime) needs in order to emit Colight artifacts.

This implementation was written from [`docs/src/colight_docs/format.md`][spec]
alone, and is validated against the Python reference implementation in both
directions by the conformance suite in `tests/`.

## Writing a file

```ts
import { createFile, ndarray, rawBuffer } from "@colight/format";

const bytes = createFile(
  // The initial-state entry.
  {
    ast: null,
    state: {
      points: ndarray(new Float32Array([0, 1, 2, 3, 4, 5]), [3, 2]),
      blob: rawBuffer(new Uint8Array([0xde, 0xad])),
    },
  },
  // Zero or more update entries.
  [
    { ast: null, state: { frame: 1 } },
    { ast: null, state: { frame: 2 } },
  ],
);
```

Anywhere an `ndarray(...)` or `rawBuffer(...)` appears in the payload, the writer
hoists its bytes into the entry's binary section and leaves the corresponding
buffer-reference envelope in the JSON. Buffer indices are assigned in traversal
order and restart at zero in every entry, as the format requires.

## Appending

The format is an append-only stream: an update is written to the end of the file
and existing bytes are never rewritten. In memory:

```ts
import { appendUpdates } from "@colight/format";

const grown = appendUpdates(bytes, [{ ast: null, state: { frame: 3 } }]);
```

On disk, with the file held open across many appends — the shape a long-running
producer wants:

```ts
import { ColightFileWriter } from "@colight/format/node";

const writer = ColightFileWriter.create("out.colight", {
  ast: null,
  state: {},
});
try {
  for (let frame = 0; frame < 100; frame++) {
    writer.append({ ast: null, state: { frame } });
  }
} finally {
  writer.close();
}
```

`ColightFileWriter.open(path)` reopens an existing file for further appends, and
`appendUpdatesToFile(path, updates)` is the one-shot open-append-close form.
Holding the file open is 1.8–4.5x faster and is what you should reach for by
default; see [the streaming contract in format.md §4.1][streaming] for the
measurements and for the rules a conforming producer must honour:

- **Append-only.** Nothing already written is ever modified.
- **Alignment preserved.** Every entry is padded to a multiple of 8, so each
  appended entry — and every buffer in it — stays 8-byte aligned. Both
  `open` and `appendUpdatesToFile` refuse a file whose length is not, rather
  than write a tail no reader can view without copying.
- **Readers tolerate a torn tail.** A reader arriving mid-append sees every
  complete entry and drops the incomplete one, so the file can be read while
  it is still being written — a consumer always observes a monotonically
  growing, never-corrupt prefix.

`append()` reaches the OS immediately, so another process sees it at once; call
`flush()` to force it to stable storage (`fsync`). Neither reference
implementation fsyncs per append — it costs roughly three orders of magnitude,
and a truncated stream is a readable artifact anyway.

A worked producer, with the CLI commands that measure its output, is in
[`examples/streaming/`](../../examples/streaming/).

## Arrays

`ndarray(typedArray, shape)` infers the dtype from the typed array. For dtypes
with no corresponding JS typed array, or to supply plain numbers, use the
explicit form:

```ts
ndarray({ dtype: "uint16", shape: [2, 2], data: [1, 2, 3, 4] });
boolArray([true, false, true]); // dtype "bool", one byte per element
```

Elements are written little-endian in C (row-major) order regardless of the host
platform's byte order.

**64-bit integers.** JavaScript numbers cannot represent every `int64`/`uint64`
value: above 2^53 they lose precision. Rather than silently writing a wrong
number, this writer **throws** when a plain JS number outside ±(2^53−1) is given
for a 64-bit element. Pass a `BigInt` to write the full range:

```ts
ndarray({ dtype: "int64", data: [2n ** 62n] }); // fine
ndarray({ dtype: "int64", data: [2 ** 62] }); // throws
```

Note that Colight's _readers_ still convert 64-bit integers to JS numbers, so a
value above 2^53 written here will read back imprecisely in a JS client. The
throw protects the write path only.

## Byte-identity with the Python writer

For every fixture in the conformance suite this package emits bytes identical to
Colight's Python writer, which makes fixtures diffable and outputs
content-addressable. Achieving that required pinning JSON spelling choices the
format does not constrain — compact separators, ASCII-escaped non-ASCII, and
Python's numeric formatting rules. See [format.md §2.2][spec] for what is
canonical and what is implementation freedom.

One difference is not resolvable from JavaScript: Python distinguishes an `int`
`1` (written `1`) from a `float` `1.0` (written `1.0`), and JavaScript has one
number type. Wrap a value in `pyFloat(...)` to request the float spelling when
matching a Python-authored payload byte-for-byte matters. It has no effect on the
parsed value.

## API

Core (`@colight/format`, environment-free):

| Export                                           | Purpose                                                |
| ------------------------------------------------ | ------------------------------------------------------ |
| `createFile(initial, updates?)`                  | Whole file; `initial: null` gives an updates-only file |
| `createEntry(payload)`                           | One initial-state entry                                |
| `createUpdateEntry(payload)`                     | One update entry (wrapped as `{updates: …}`)           |
| `appendUpdates(existing, updates)`               | Append entries to existing bytes                       |
| `ndarray`, `boolArray`, `rawBuffer`              | Buffer-carrying payload values                         |
| `pyFloat`                                        | Request Python's float spelling for a number           |
| `layoutBuffers`, `align8`, `assertAppendable`    | Layout primitives                                      |
| `MAGIC_BYTES`, `HEADER_SIZE`, `CURRENT_VERSION`  | Format constants                                       |
| `DTYPE_BYTES`, `isDtype`, `assertDtype`          | Dtype table and validation                             |
| `encodeJson`, `encodeJsonString`, `encodeNumber` | The JSON section encoder                               |

Node (`@colight/format/node`):

| Export                            | Purpose                             |
| --------------------------------- | ----------------------------------- |
| `writeColightFile(path, initial)` | Write a new single-entry file       |
| `appendUpdatesToFile(path, ups)`  | One-shot append to a file           |
| `ColightFileWriter`               | Open once, append many times, close |

`ColightFileWriter` mirrors `colight.format.ColightWriter` on the Python side
(`create` / `open` / `append` / `append_all` / `flush` / `close`); the two are
kept deliberately parallel so a producer can be ported between them.

## Reading

This package writes; it does not read. Colight's readers are
`packages/colight/src/js/format.js` (JavaScript) and `colight.format` (Python).

[spec]: ../../docs/src/colight_docs/format.md
[streaming]: ../../docs/src/colight_docs/format.md#41-appending-the-streaming-contract
