/**
 * Streaming contract from the JavaScript side: appending while reading.
 *
 * `tornTail.test.ts` proves both readers survive a half-written entry. This
 * suite proves the *writer* keeps that promise in motion: a producer holding
 * the file open, appending entry after entry, must leave a file that a reader
 * can consume at any instant — seeing a prefix that is complete, correct, and
 * never shrinks.
 *
 * Both readers are exercised: the existing JavaScript one, and Python's
 * `parse_file_with_updates`, so a JS producer and a Python consumer are
 * verified as an actual pair.
 */

import {
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { createUpdateEntry, ndarray } from "@colight/format";
import {
  ColightFileWriter,
  appendUpdatesToFile,
  writeColightFile,
} from "@colight/format/node";

import { jsReadFile } from "./jsReader.js";
import { pythonReadFile } from "./python.js";

const TICKS = 12;

let workdir: string;

beforeAll(() => {
  workdir = mkdtempSync(join(tmpdir(), "colight-streaming-"));
});

afterAll(() => {
  rmSync(workdir, { recursive: true, force: true });
});

/** A distinct, checkable array per tick. */
const positions = (tick: number) =>
  ndarray(new Float32Array([tick, tick, tick, tick, tick, tick]), [2, 3]);

const initial = { ast: null, state: { positions: positions(0) } };
const update = (tick: number) => ({
  ast: null,
  state: { tick, positions: positions(tick) },
});

/** The `tick` of every update entry a reader can see in `path`. */
function ticksVisible(path: string): number[] {
  const read = jsReadFile(readFileSync(path));
  return read.updates.map(
    (entry) => (entry as { state: { tick: number } }).state.tick,
  );
}

describe("a held-open writer leaves the file readable after every append", () => {
  it("grows the visible prefix by exactly one entry per append", () => {
    const path = join(workdir, "grow.colight");
    const writer = ColightFileWriter.create(path, initial);
    try {
      const seen: number[] = [];
      for (let tick = 1; tick <= TICKS; tick++) {
        writer.append(update(tick));

        const visible = ticksVisible(path);
        expect(visible).toEqual(Array.from({ length: tick }, (_, i) => i + 1));
        seen.push(visible.length);
        // The prefix never shrinks.
        expect(seen).toEqual([...seen].sort((a, b) => a - b));
      }
    } finally {
      writer.close();
    }

    // Python agrees with the JS reader on the finished artifact.
    expect(pythonReadFile(path).updates).toEqual(
      jsReadFile(readFileSync(path)).updates,
    );
  });

  it("keeps the file 8-byte aligned at every append, so appending stays legal", () => {
    const path = join(workdir, "aligned.colight");
    const writer = ColightFileWriter.create(path, initial);
    try {
      expect(statSync(path).size % 8).toBe(0);
      for (let tick = 1; tick <= TICKS; tick++) {
        writer.append(update(tick));
        expect(statSync(path).size % 8).toBe(0);
      }
    } finally {
      writer.close();
    }
    // The alignment invariant is exactly what `open` re-checks.
    expect(() => ColightFileWriter.open(path).close()).not.toThrow();
  });

  it("survives being reopened partway through", () => {
    const path = join(workdir, "reopen.colight");
    const first = ColightFileWriter.create(path, initial);
    first.append(update(1));
    first.close();

    const second = ColightFileWriter.open(path);
    second.append(update(2));
    second.close();

    expect(ticksVisible(path)).toEqual([1, 2]);
  });

  it("refuses to append to a misaligned file rather than corrupt it", () => {
    const path = join(workdir, "misaligned.colight");
    writeColightFile(path, initial);
    writeFileSync(path, Buffer.concat([readFileSync(path), Buffer.alloc(1)]));

    expect(() => ColightFileWriter.open(path)).toThrow(/multiple of 8/);
    expect(() => appendUpdatesToFile(path, [update(1)])).toThrow(
      /multiple of 8/,
    );
  });

  it("rejects appends after close", () => {
    const path = join(workdir, "closed.colight");
    const writer = ColightFileWriter.create(path, initial);
    writer.close();
    expect(writer.closed).toBe(true);
    writer.close(); // idempotent
    expect(() => writer.append(update(1))).toThrow(/closed/);
  });
});

describe("a reader landing mid-append sees the complete prefix", () => {
  it("shows exactly the finished entries at every byte of an in-flight append", () => {
    // Rather than racing a real writer (flaky), replay one real append a chunk
    // at a time, so every mid-append instant is covered — the technique
    // `tornTail.test.ts` uses, applied along a growing file instead of a
    // truncated one.
    const path = join(workdir, "inflight-base.colight");
    const writer = ColightFileWriter.create(path, initial);
    writer.append(update(1));
    writer.append(update(2));
    writer.close();

    const complete = readFileSync(path);
    const tail = createUpdateEntry(update(3));
    const partial = join(workdir, "inflight.colight");

    for (let cut = 0; cut < tail.byteLength; cut += 8) {
      writeFileSync(
        partial,
        Buffer.concat([complete, Buffer.from(tail.subarray(0, cut))]),
      );
      expect(ticksVisible(partial), `cut=${cut}`).toEqual([1, 2]);
      // Python sees the same prefix, and does not error either.
      expect(pythonReadFile(partial).updates).toHaveLength(2);
    }

    // The moment the last byte lands, the third entry appears whole.
    writeFileSync(partial, Buffer.concat([complete, Buffer.from(tail)]));
    expect(ticksVisible(partial)).toEqual([1, 2, 3]);
  });
});

describe("holding open versus reopening per entry", () => {
  it("produces byte-identical artifacts", () => {
    // The choice is purely about throughput: the bytes do not depend on it.
    const held = join(workdir, "held.colight");
    const reopened = join(workdir, "reopened.colight");

    const writer = ColightFileWriter.create(held, initial);
    for (let tick = 1; tick <= 4; tick++) writer.append(update(tick));
    writer.close();

    writeColightFile(reopened, initial);
    for (let tick = 1; tick <= 4; tick++) {
      appendUpdatesToFile(reopened, [update(tick)]);
    }

    expect(readFileSync(held)).toEqual(readFileSync(reopened));
  });
});
