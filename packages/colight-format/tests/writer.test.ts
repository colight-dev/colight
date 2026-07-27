/**
 * Byte-level structure of what the writer emits, checked directly against the
 * invariants of format.md §2 rather than against another implementation.
 */

import { describe, expect, it } from "vitest";

import {
  CURRENT_VERSION,
  HEADER_SIZE,
  MAGIC_BYTES,
  align8,
  appendUpdates,
  boolArray,
  createEntry,
  createFile,
  createUpdateEntry,
  layoutBuffers,
  ndarray,
  rawBuffer,
} from "@colight/format";

/** Reads the six uint64le header fields of the entry at `offset` (spec §2.1). */
function header(bytes: Uint8Array, offset = 0) {
  const view = new DataView(bytes.buffer, bytes.byteOffset + offset);
  return {
    magic: bytes.subarray(offset, offset + 8),
    version: Number(view.getBigUint64(8, true)),
    jsonOffset: Number(view.getBigUint64(16, true)),
    jsonLength: Number(view.getBigUint64(24, true)),
    binaryOffset: Number(view.getBigUint64(32, true)),
    binaryLength: Number(view.getBigUint64(40, true)),
    numBuffers: Number(view.getBigUint64(48, true)),
    reserved: bytes.subarray(offset + 56, offset + HEADER_SIZE),
  };
}

function jsonOf(bytes: Uint8Array, offset = 0) {
  const head = header(bytes, offset);
  const text = new TextDecoder().decode(
    bytes.subarray(
      offset + head.jsonOffset,
      offset + head.jsonOffset + head.jsonLength,
    ),
  );
  return JSON.parse(text);
}

describe("entry header", () => {
  it("writes the magic bytes, version and fixed json offset", () => {
    const entry = createEntry({ ast: null, state: {} });
    const head = header(entry);
    expect(Array.from(head.magic)).toEqual(Array.from(MAGIC_BYTES));
    expect(head.version).toBe(CURRENT_VERSION);
    expect(head.version).toBe(2);
    expect(head.jsonOffset).toBe(HEADER_SIZE);
  });

  it("zeroes the 40 reserved bytes", () => {
    const entry = createEntry({
      state: { a: ndarray(new Float64Array([1]), [1]) },
    });
    expect(Array.from(header(entry).reserved)).toEqual(new Array(40).fill(0));
  });

  it("reports json_length excluding padding", () => {
    const payload = { ast: null, state: {} };
    const entry = createEntry(payload);
    const head = header(entry);
    const text = new TextDecoder().decode(
      entry.subarray(HEADER_SIZE, HEADER_SIZE + head.jsonLength),
    );
    expect(JSON.parse(text)).toEqual(payload);
    expect(text.endsWith("}")).toBe(true);
  });
});

describe("section framing and padding", () => {
  it("pads the gap between JSON and the binary section with zeros", () => {
    const entry = createEntry({ x: rawBuffer(new Uint8Array([1, 2, 3])) });
    const head = header(entry);
    expect(head.binaryOffset).toBe(align8(HEADER_SIZE + head.jsonLength));
    expect(head.binaryOffset % 8).toBe(0);
    const gap = entry.subarray(
      HEADER_SIZE + head.jsonLength,
      head.binaryOffset,
    );
    expect(Array.from(gap)).toEqual(new Array(gap.length).fill(0));
  });

  it("pads the JSON section even when the entry has no buffers", () => {
    const entry = createEntry({ ast: null });
    const head = header(entry);
    expect(head.numBuffers).toBe(0);
    expect(head.binaryLength).toBe(0);
    expect(head.binaryOffset).toBe(align8(HEADER_SIZE + head.jsonLength));
    expect(entry.byteLength).toBe(head.binaryOffset);
  });

  it("pads every entry to a multiple of 8", () => {
    // A payload whose JSON length lands on each residue mod 8.
    for (let padding = 0; padding < 16; padding++) {
      const entry = createEntry({ k: "x".repeat(padding) });
      expect(entry.byteLength % 8).toBe(0);
      const head = header(entry);
      expect(entry.byteLength).toBe(
        align8(head.binaryOffset + head.binaryLength),
      );
    }
  });

  it("starts every entry of a multi-entry file at a multiple of 8", () => {
    const file = createFile({ state: { a: rawBuffer(new Uint8Array([1])) } }, [
      { state: { b: rawBuffer(new Uint8Array([1, 2, 3])) } },
      { state: { c: ndarray(new Float64Array([1, 2]), [2]) } },
      { state: {} },
    ]);
    let offset = 0;
    let entries = 0;
    while (offset + HEADER_SIZE <= file.byteLength) {
      expect(offset % 8).toBe(0);
      const head = header(file, offset);
      expect(Array.from(head.magic)).toEqual(Array.from(MAGIC_BYTES));
      // Every buffer is 8-byte aligned relative to the *file* start (§2.3).
      expect((offset + head.binaryOffset) % 8).toBe(0);
      offset += align8(head.binaryOffset + head.binaryLength);
      entries++;
    }
    expect(entries).toBe(4);
    expect(offset).toBe(file.byteLength);
  });
});

describe("bufferLayout", () => {
  it("is absent when the entry has no buffers", () => {
    const entry = createEntry({ ast: null, state: { n: 1 } });
    expect(jsonOf(entry)).not.toHaveProperty("bufferLayout");
    expect(header(entry).numBuffers).toBe(0);
  });

  it("aligns each buffer to 8 bytes within the binary section", () => {
    const entry = createEntry({
      a: rawBuffer(new Uint8Array([1, 2, 3])),
      b: rawBuffer(new Uint8Array([4, 5, 6, 7, 8])),
      c: ndarray(new Float64Array([9]), [1]),
    });
    const layout = jsonOf(entry).bufferLayout;
    expect(layout.offsets).toEqual([0, 8, 16]);
    expect(layout.lengths).toEqual([3, 5, 8]);
    for (const offset of layout.offsets) expect(offset % 8).toBe(0);
  });

  it("reports count and totalSize, and count matches num_buffers", () => {
    const entry = createEntry({
      a: rawBuffer(new Uint8Array(3)),
      b: rawBuffer(new Uint8Array(5)),
    });
    const layout = jsonOf(entry).bufferLayout;
    expect(layout.count).toBe(2);
    expect(layout.count).toBe(header(entry).numBuffers);
    // totalSize is the last offset plus the last length (§2.3).
    expect(layout.totalSize).toBe(13);
  });

  it("reports binary_length covering the last buffer without trailing padding", () => {
    const entry = createEntry({ a: rawBuffer(new Uint8Array([1, 2, 3])) });
    expect(header(entry).binaryLength).toBe(3);
  });

  it("computes offsets for a zero-length buffer without skipping an index", () => {
    const { layout, binaryLength } = layoutBuffers([
      new Uint8Array(0),
      new Uint8Array(1),
      new Uint8Array(0),
      new Uint8Array(2),
    ]);
    expect(layout.offsets).toEqual([0, 0, 8, 8]);
    expect(layout.lengths).toEqual([0, 1, 0, 2]);
    expect(layout.count).toBe(4);
    expect(layout.totalSize).toBe(10);
    expect(binaryLength).toBe(10);
  });

  it("rejects a caller-supplied bufferLayout", () => {
    expect(() =>
      createEntry({
        bufferLayout: { offsets: [0], lengths: [1], count: 1, totalSize: 1 },
      }),
    ).toThrow(/Do not set bufferLayout/);
  });
});

describe("ndarray envelopes", () => {
  it("writes the canonical envelope with data null", () => {
    const entry = createEntry({
      a: ndarray(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]),
    });
    expect(jsonOf(entry).a).toEqual({
      __type__: "ndarray",
      data: null,
      dtype: "float32",
      shape: [2, 3],
      __buffer_index__: 0,
    });
  });

  it("writes elements little-endian in C order", () => {
    const entry = createEntry({ a: ndarray(new Uint16Array([1, 256]), [2]) });
    const head = header(entry);
    const bytes = entry.subarray(
      head.binaryOffset,
      head.binaryOffset + head.binaryLength,
    );
    expect(Array.from(bytes)).toEqual([0x01, 0x00, 0x00, 0x01]);
  });

  it("infers a dtype from each typed array kind", () => {
    const pairs: [ArrayBufferView, string][] = [
      [new Int8Array(1), "int8"],
      [new Int16Array(1), "int16"],
      [new Int32Array(1), "int32"],
      [new Uint8Array(1), "uint8"],
      [new Uint16Array(1), "uint16"],
      [new Uint32Array(1), "uint32"],
      [new Float32Array(1), "float32"],
      [new Float64Array(1), "float64"],
      [new BigInt64Array(1), "int64"],
      [new BigUint64Array(1), "uint64"],
    ];
    for (const [array, dtype] of pairs) {
      expect(ndarray(array, [1]).dtype).toBe(dtype);
    }
  });

  it("writes bool as one byte per element", () => {
    const array = boolArray([true, false, true]);
    expect(array.dtype).toBe("bool");
    expect(Array.from(array.bytes)).toEqual([1, 0, 1]);
  });

  it("rejects a shape that disagrees with the data length", () => {
    expect(() => ndarray(new Float32Array(6), [2, 2])).toThrow(
      /needs 16 bytes, got 24/,
    );
  });

  it("rejects zero-dimensional arrays", () => {
    expect(() => ndarray(new Float32Array(1), [])).toThrow(/Zero-dimensional/);
  });

  it("rejects an unknown dtype", () => {
    expect(() => ndarray({ dtype: "float16", data: [1, 2] })).toThrow(
      /Unknown \.colight dtype "float16"/,
    );
  });

  it("rejects a bare typed array in the payload", () => {
    expect(() => createEntry({ a: new Float32Array([1]) as never })).toThrow(
      /Wrap them with ndarray/,
    );
  });
});

describe("64-bit integer precision", () => {
  it("accepts BigInt values across the full 64-bit range", () => {
    const big = 2n ** 62n + 12345n;
    const array = ndarray({ dtype: "int64", data: [big] });
    const view = new DataView(array.bytes.buffer, array.bytes.byteOffset);
    expect(view.getBigInt64(0, true)).toBe(big);
  });

  it("throws rather than silently losing precision on a large plain number", () => {
    expect(() =>
      ndarray({ dtype: "int64", data: [Number.MAX_SAFE_INTEGER + 2] }),
    ).toThrow(/outside the exactly-representable range/);
    expect(() => ndarray({ dtype: "uint64", data: [2 ** 60] })).toThrow(
      /outside the exactly-representable range/,
    );
  });

  it("accepts plain numbers inside the safe range", () => {
    const array = ndarray({ dtype: "uint64", data: [Number.MAX_SAFE_INTEGER] });
    const view = new DataView(array.bytes.buffer, array.bytes.byteOffset);
    expect(view.getBigUint64(0, true)).toBe(BigInt(Number.MAX_SAFE_INTEGER));
  });
});

describe("update entries", () => {
  it("wraps the payload under an updates key", () => {
    const entry = createUpdateEntry({ ast: null, state: { frame: 3 } });
    expect(jsonOf(entry)).toEqual({
      updates: { ast: null, state: { frame: 3 } },
    });
  });

  it("puts bufferLayout beside updates, not inside it", () => {
    const entry = createUpdateEntry({
      ast: null,
      state: { a: rawBuffer(new Uint8Array([1])) },
    });
    const json = jsonOf(entry);
    expect(json.bufferLayout).toBeDefined();
    expect(json.updates.bufferLayout).toBeUndefined();
  });

  it("restarts buffer indices at zero in each entry", () => {
    const file = createFile({ a: rawBuffer(new Uint8Array([1])) }, [
      { state: { b: rawBuffer(new Uint8Array([2])) } },
    ]);
    const firstSize = createEntry({
      a: rawBuffer(new Uint8Array([1])),
    }).byteLength;
    expect(jsonOf(file).a.__buffer_index__).toBe(0);
    expect(jsonOf(file, firstSize).updates.state.b.__buffer_index__).toBe(0);
  });
});

describe("appending", () => {
  it("leaves the existing bytes byte-for-byte unchanged", () => {
    const original = createEntry({ ast: null, state: {} });
    const appended = appendUpdates(original, [{ ast: null, state: { n: 1 } }]);
    expect(Array.from(appended.subarray(0, original.byteLength))).toEqual(
      Array.from(original),
    );
    expect(appended.byteLength).toBeGreaterThan(original.byteLength);
  });

  it("produces the same bytes as writing the whole file at once", () => {
    const initial = {
      ast: null,
      state: { a: rawBuffer(new Uint8Array([1, 2, 3])) },
    };
    const updates = [
      { ast: null, state: { frame: 1 } },
      { ast: null, state: { b: ndarray(new Float64Array([1.5]), [1]) } },
    ];
    const atOnce = createFile(initial, updates);
    let incremental = createEntry(initial);
    for (const update of updates)
      incremental = appendUpdates(incremental, [update]);
    expect(Array.from(incremental)).toEqual(Array.from(atOnce));
  });

  it("refuses to append to misaligned bytes", () => {
    const original = createEntry({ ast: null });
    expect(() =>
      appendUpdates(original.subarray(0, original.byteLength - 1), [{}]),
    ).toThrow(/not a multiple of 8/);
  });
});

describe("payload validation", () => {
  it("requires the entry payload to be an object", () => {
    expect(() => createEntry([1, 2, 3] as never)).toThrow(
      /must be a single JSON object/,
    );
    expect(() => createEntry(null as never)).toThrow(
      /must be a single JSON object/,
    );
  });

  it("refuses non-finite numbers", () => {
    expect(() => createEntry({ a: NaN })).toThrow(/non-finite/);
    expect(() => createEntry({ a: Infinity })).toThrow(/non-finite/);
  });
});
