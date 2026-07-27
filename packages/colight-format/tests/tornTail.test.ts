/**
 * Torn-tail tolerance (spec §3.1).
 *
 * A `.colight` file is an append-only stream, so a reader may see it while the
 * latest entry is still being written. The documented behavior is: a malformed
 * *first* entry is an error; a malformed entry *after* the first terminates the
 * walk without error, yielding everything read so far.
 *
 * Both reference readers are exercised here against files the JS writer
 * produced and then truncated mid-entry.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { HEADER_SIZE, createEntry, createFile, ndarray } from "@colight/format";

import { jsReadFile } from "./jsReader.js";
import { pythonReadFile, tryPythonReadFile } from "./python.js";

let workdir: string;

beforeAll(() => {
  workdir = mkdtempSync(join(tmpdir(), "colight-format-torn-"));
});

afterAll(() => {
  rmSync(workdir, { recursive: true, force: true });
});

/** A three-entry file: initial state plus two updates, all carrying buffers. */
function buildStream(): { bytes: Uint8Array; firstEntryLength: number } {
  const initial = {
    ast: null,
    state: { base: ndarray(new Float32Array([1, 2, 3]), [3]) },
  };
  const bytes = createFile(initial, [
    {
      ast: null,
      state: { frame: 1, data: ndarray(new Int32Array([9, 8]), [2]) },
    },
    { ast: null, state: { frame: 2 } },
  ]);
  return { bytes, firstEntryLength: createEntry(initial).byteLength };
}

/** Truncates `bytes` to `length` and writes it out for the Python reader. */
function truncateTo(bytes: Uint8Array, length: number, name: string): string {
  const path = join(workdir, name);
  writeFileSync(path, bytes.subarray(0, length));
  return path;
}

describe("torn tail after the first entry", () => {
  const cases: {
    label: string;
    cut: (first: number, total: number) => number;
  }[] = [
    {
      label: "mid-header of the second entry",
      cut: (first) => first + 40,
    },
    {
      label: "just past the second entry's header, before its JSON",
      cut: (first) => first + HEADER_SIZE,
    },
    {
      label: "mid-JSON of the second entry",
      cut: (first) => first + HEADER_SIZE + 16,
    },
    {
      label: "mid-binary-section of the second entry",
      cut: (first, total) => Math.floor((first + total) / 2),
    },
  ];

  it.each(cases)(
    "both readers stop at the first entry when truncated $label",
    ({ label, cut }) => {
      const { bytes, firstEntryLength } = buildStream();
      const length = cut(firstEntryLength, bytes.byteLength);
      expect(length).toBeGreaterThan(firstEntryLength);
      expect(length).toBeLessThan(bytes.byteLength);

      const truncated = bytes.subarray(0, length);

      // The existing JS reader, called as a black box.
      const fromJs = jsReadFile(truncated);
      expect(fromJs.initial).not.toBeNull();
      expect(
        (fromJs.initial as { state: { base: unknown } }).state.base,
      ).toEqual({
        __array__: { dtype: "float32", shape: [3], values: [1, 2, 3] },
      });

      // Python, via `parse_file_with_updates`.
      const path = truncateTo(
        bytes,
        length,
        `${label.replace(/\W+/g, "-")}.colight`,
      );
      const fromPython = pythonReadFile(path);

      // Neither reader errors, and both agree on how much survived: everything
      // up to the torn entry, and nothing of the torn entry itself.
      expect(fromPython.initial).toEqual(fromJs.initial);
      expect(fromPython.updates).toEqual(fromJs.updates);
      expect(fromJs.updates.length).toBeLessThan(2);
    },
  );

  it("still reads the last entry when only its trailing padding is missing", () => {
    // An entry's content ends at `binary_offset + binary_length`; the bytes
    // after that up to the 8-byte boundary are padding whose only purpose is to
    // align the *next* entry (spec §2.2). Losing them costs nothing.
    const { bytes } = buildStream();
    const path = truncateTo(
      bytes,
      bytes.byteLength - 1,
      "padding-torn.colight",
    );

    const fromJs = jsReadFile(bytes.subarray(0, bytes.byteLength - 1));
    const fromPython = pythonReadFile(path);

    expect(fromJs.updates).toHaveLength(2);
    expect(fromPython.updates).toEqual(fromJs.updates);
  });

  it("recovers every complete entry and drops only the torn one", () => {
    const { bytes, firstEntryLength } = buildStream();
    // Cut one byte into the third entry: the first two entries are intact.
    const secondEntryEnd = findSecondEntryEnd(bytes, firstEntryLength);
    const path = truncateTo(bytes, secondEntryEnd + 1, "third-torn.colight");

    const fromPython = pythonReadFile(path);
    const fromJs = jsReadFile(bytes.subarray(0, secondEntryEnd + 1));

    expect(fromJs.updates).toHaveLength(1);
    expect(fromPython.updates).toEqual(fromJs.updates);
    expect(fromJs.updates[0]).toEqual({
      ast: null,
      state: {
        frame: 1,
        data: { __array__: { dtype: "int32", shape: [2], values: [9, 8] } },
      },
    });
  });
});

describe("a malformed first entry is an error", () => {
  it("both readers reject a truncated first entry", () => {
    const { bytes } = buildStream();
    const path = truncateTo(bytes, 40, "first-torn.colight");

    expect(() => jsReadFile(bytes.subarray(0, 40))).toThrow();

    const result = tryPythonReadFile(path);
    expect(result.ok).toBe(false);
  });

  it("both readers reject wrong magic bytes in the first entry", () => {
    const { bytes } = buildStream();
    const corrupted = bytes.slice();
    corrupted[0] = 0x58; // "X"
    const path = truncateTo(
      corrupted,
      corrupted.byteLength,
      "bad-magic.colight",
    );

    expect(() => jsReadFile(corrupted)).toThrow();
    expect(tryPythonReadFile(path).ok).toBe(false);
  });

  it("both readers reject an unsupported version in the first entry", () => {
    const { bytes } = buildStream();
    const corrupted = bytes.slice();
    new DataView(corrupted.buffer, corrupted.byteOffset).setBigUint64(
      8,
      99n,
      true,
    );
    const path = truncateTo(
      corrupted,
      corrupted.byteLength,
      "bad-version.colight",
    );

    expect(() => jsReadFile(corrupted)).toThrow();
    expect(tryPythonReadFile(path).ok).toBe(false);
  });
});

/** Reads the second entry's end offset out of its own header (spec §2.2). */
function findSecondEntryEnd(
  bytes: Uint8Array,
  firstEntryLength: number,
): number {
  const view = new DataView(bytes.buffer, bytes.byteOffset + firstEntryLength);
  const binaryOffset = Number(view.getBigUint64(32, true));
  const binaryLength = Number(view.getBigUint64(40, true));
  const size = Math.ceil((binaryOffset + binaryLength) / 8) * 8;
  return firstEntryLength + size;
}
