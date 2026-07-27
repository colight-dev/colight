/**
 * The `.colight` version 2 writer.
 *
 * Implements the entry framing of spec §2 and the writer algorithm of §4:
 * lay out buffers 8-byte aligned, add `bufferLayout` when there is at least one
 * buffer, encode compact UTF-8 JSON at offset 96, pad to an 8-byte
 * `binary_offset`, emit the buffers, and pad the whole entry to a multiple of 8
 * so the next appended entry starts aligned.
 */

import {
  CURRENT_VERSION,
  HEADER_SIZE,
  MAGIC_BYTES,
  align8,
} from "./constants.js";
import { PyFloat, encodeJson, type JsonValue } from "./json.js";
import { NDArray, RawBuffer } from "./values.js";

/** The `bufferLayout` table written into an entry's JSON (spec §2.3). */
export interface BufferLayout {
  /** Per-buffer offset relative to the binary section start. */
  offsets: number[];
  /** Per-buffer exact length in bytes, excluding padding. */
  lengths: number[];
  /** Number of buffers; equals the header's `num_buffers`. */
  count: number;
  /** Last offset plus last length. `0` when there are no buffers. */
  totalSize: number;
}

/**
 * A payload value handed to the writer. Anywhere a {@link NDArray} or
 * {@link RawBuffer} appears it is hoisted into the binary section and replaced
 * by its buffer-reference envelope.
 */
export type Payload =
  | null
  | boolean
  | number
  | bigint
  | string
  | PyFloat
  | NDArray
  | RawBuffer
  | readonly Payload[]
  | { readonly [key: string]: Payload };

interface Extracted {
  json: JsonValue;
  buffers: Uint8Array[];
}

/**
 * Replaces every {@link NDArray}/{@link RawBuffer} in `payload` with its JSON
 * envelope, collecting the bytes in buffer-index order (indices are assigned in
 * depth-first traversal order, restarting at 0 for each entry — spec §3.6).
 */
function extractBuffers(payload: Payload): Extracted {
  const buffers: Uint8Array[] = [];

  const walk = (value: Payload): JsonValue => {
    if (value instanceof NDArray) {
      const index = buffers.push(value.bytes) - 1;
      return {
        __type__: "ndarray",
        // Always null in files: `data` is the pre-extraction slot used by
        // out-of-band transports (spec §3.3).
        data: null,
        dtype: value.dtype,
        shape: value.shape as number[],
        __buffer_index__: index,
      };
    }
    if (value instanceof RawBuffer) {
      const index = buffers.push(value.bytes) - 1;
      return { __buffer_index__: index };
    }
    if (value === null || typeof value !== "object") {
      return value as JsonValue;
    }
    // A float-spelling marker is a leaf; the JSON encoder handles it.
    if (value instanceof PyFloat) return value;
    if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
      throw new Error(
        "Raw typed arrays and ArrayBuffers are not payload values. Wrap them " +
          "with ndarray(...) to write an ndarray envelope, or rawBuffer(...) " +
          "to write opaque bytes.",
      );
    }
    if (Array.isArray(value)) return value.map(walk);

    const out: Record<string, JsonValue> = {};
    for (const [key, entry] of Object.entries(
      value as { [key: string]: Payload },
    )) {
      if (entry === undefined) continue;
      out[key] = walk(entry);
    }
    return out;
  };

  return { json: walk(payload), buffers };
}

/**
 * Computes buffer offsets, padding the running offset up to a multiple of 8
 * before each buffer (spec §4.1).
 */
export function layoutBuffers(buffers: readonly Uint8Array[]): {
  layout: BufferLayout;
  binaryLength: number;
} {
  const offsets: number[] = [];
  const lengths: number[] = [];
  let cursor = 0;
  for (const buffer of buffers) {
    cursor = align8(cursor);
    offsets.push(cursor);
    lengths.push(buffer.byteLength);
    cursor += buffer.byteLength;
  }
  const totalSize =
    buffers.length === 0
      ? 0
      : offsets[offsets.length - 1] + lengths[lengths.length - 1];
  return {
    layout: { offsets, lengths, count: buffers.length, totalSize },
    // The binary section's own length excludes trailing padding; the entry is
    // padded to align8 separately (spec §2.2).
    binaryLength: cursor,
  };
}

const TEXT_ENCODER = new TextEncoder();

/**
 * Serializes one entry: header, JSON section, padding, binary section, padding.
 *
 * This is the `.colight` equivalent of Python's `colight.format.create_bytes`.
 * `payload` is the entry's JSON object; any buffer-carrying values inside it are
 * hoisted automatically. A `bufferLayout` key is added iff at least one buffer
 * was found — supplying your own is an error, since only the writer knows the
 * final offsets.
 */
export function createEntry(payload: Payload): Uint8Array {
  if (
    payload === null ||
    typeof payload !== "object" ||
    Array.isArray(payload)
  ) {
    throw new Error(
      "A .colight entry's JSON section must be a single JSON object (spec §2.2).",
    );
  }

  const { json, buffers } = extractBuffers(payload);
  const object = json as Record<string, JsonValue>;
  if ("bufferLayout" in object) {
    throw new Error(
      "Do not set bufferLayout yourself; the writer computes it from the " +
        "ndarray/rawBuffer values it finds in the payload.",
    );
  }

  const { layout, binaryLength } = layoutBuffers(buffers);
  const jsonObject: Record<string, JsonValue> =
    buffers.length > 0
      ? { ...object, bufferLayout: layout as unknown as JsonValue }
      : object;

  const jsonBytes = TEXT_ENCODER.encode(encodeJson(jsonObject));
  const jsonOffset = HEADER_SIZE;
  const jsonLength = jsonBytes.byteLength;
  const binaryOffset = align8(jsonOffset + jsonLength);
  const entrySize = align8(binaryOffset + binaryLength);

  const entry = new Uint8Array(entrySize);
  const view = new DataView(entry.buffer);

  entry.set(MAGIC_BYTES, 0);
  view.setBigUint64(8, BigInt(CURRENT_VERSION), true);
  view.setBigUint64(16, BigInt(jsonOffset), true);
  view.setBigUint64(24, BigInt(jsonLength), true);
  view.setBigUint64(32, BigInt(binaryOffset), true);
  view.setBigUint64(40, BigInt(binaryLength), true);
  view.setBigUint64(48, BigInt(buffers.length), true);
  // Bytes 56..96 stay zero: writers MUST write zeros there (spec §2.1).

  entry.set(jsonBytes, jsonOffset);
  for (let i = 0; i < buffers.length; i++) {
    entry.set(buffers[i], binaryOffset + layout.offsets[i]);
  }
  return entry;
}

/**
 * Serializes an update entry: the payload wrapped as `{"updates": ...}`
 * (spec §3.6). Buffer indices restart at 0 within the entry.
 */
export function createUpdateEntry(update: Payload): Uint8Array {
  return createEntry({ updates: update });
}

/**
 * Serializes a whole file: an optional initial-state entry followed by zero or
 * more update entries, concatenated.
 *
 * Pass `initial: null` for an "updates-only" file (spec §3.1).
 */
export function createFile(
  initial: Payload | null,
  updates: readonly Payload[] = [],
): Uint8Array {
  const entries: Uint8Array[] = [];
  if (initial !== null) entries.push(createEntry(initial));
  for (const update of updates) entries.push(createUpdateEntry(update));
  return concat(entries);
}

/**
 * Appends update entries to existing `.colight` bytes.
 *
 * Appending never rewrites existing bytes (spec §1): the result is the original
 * buffer followed verbatim by the new entries. Because every entry is padded to
 * a multiple of 8, the appended entries stay absolutely 8-byte aligned as long
 * as `existing` was produced by a conforming writer — which this function
 * verifies before appending.
 */
export function appendUpdates(
  existing: Uint8Array,
  updates: readonly Payload[],
): Uint8Array {
  assertAppendable(existing);
  return concat([existing, ...updates.map(createUpdateEntry)]);
}

/**
 * Checks that `bytes` ends on an 8-byte boundary, so that an entry appended
 * after it starts at an absolute multiple of 8 (spec §2.2/§2.3). Files written
 * by a version 2 writer always do; a truncated or hand-edited file may not, and
 * appending to it would silently misalign every buffer of the new entry.
 */
export function assertAppendable(bytes: Uint8Array): void {
  if (bytes.byteLength % 8 !== 0) {
    throw new Error(
      `Cannot append to .colight data of length ${bytes.byteLength}: it is not ` +
        `a multiple of 8, so an appended entry's buffers would not be 8-byte ` +
        `aligned. The data is truncated or was not written by a version 2 writer.`,
    );
  }
}

function concat(parts: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const part of parts) total += part.byteLength;
  const out = new Uint8Array(total);
  let at = 0;
  for (const part of parts) {
    out.set(part, at);
    at += part.byteLength;
  }
  return out;
}
