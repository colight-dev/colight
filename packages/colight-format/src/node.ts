/**
 * Node filesystem helpers for writing `.colight` files.
 *
 * Kept in its own module so the core writer stays free of any environment
 * assumptions: `@colight/format`'s main entry point runs anywhere a
 * `TextEncoder` exists.
 */

import {
  closeSync,
  fsyncSync,
  openSync,
  readSync,
  statSync,
  writeSync,
} from "node:fs";

import { MAGIC_BYTES } from "./constants.js";
import { createEntry, createUpdateEntry, type Payload } from "./writer.js";

/** Writes a fresh `.colight` file containing a single initial-state entry. */
export function writeColightFile(path: string, initial: Payload): void {
  const fd = openSync(path, "w");
  try {
    writeSync(fd, createEntry(initial));
  } finally {
    closeSync(fd);
  }
}

/**
 * Appends update entries to an existing `.colight` file, in place.
 *
 * The file is opened for append, the new entries are written, and the file is
 * closed. Existing bytes are never rewritten (spec §1). Throws if the file's
 * current length is not a multiple of 8, which would misalign the new entries.
 */
export function appendUpdatesToFile(
  path: string,
  updates: readonly Payload[],
): void {
  const size = statSync(path).size;
  if (size % 8 !== 0) {
    throw new Error(
      `Cannot append to ${path}: its length ${size} is not a multiple of 8, so ` +
        `an appended entry's buffers would not be 8-byte aligned.`,
    );
  }
  const fd = openSync(path, "a");
  try {
    for (const update of updates) writeSync(fd, createUpdateEntry(update));
  } finally {
    closeSync(fd);
  }
}

/**
 * A `.colight` file held open for repeated appends.
 *
 * This is the recommended shape for a long-running producer — a simulation, a
 * training loop, an instrument, an agent loop. Opening once and appending many
 * times is roughly 3x faster than re-opening per entry (see "Appending" in
 * `format.md`); {@link appendUpdatesToFile} is the open-append-close form for
 * occasional writes.
 *
 * Streaming contract:
 *
 * - **Append-only.** Entries are only ever added at the end; existing bytes are
 *   never rewritten.
 * - **8-byte alignment is preserved.** Every entry is padded to a multiple of 8,
 *   so each appended entry — and every buffer inside it — starts at an 8-byte
 *   aligned absolute file offset.
 * - **Readers tolerate a torn tail.** A reader arriving mid-append sees every
 *   complete entry and silently drops the incomplete one, so the file may be
 *   read while it is still being written: a reader always observes a
 *   monotonically growing, never-corrupt prefix.
 *
 * Durability: {@link append} issues one `writeSync` per entry, so another
 * process sees the bytes immediately. They are not on stable storage until
 * {@link flush} (`fsync`) or {@link close} returns. A process killed mid-append
 * leaves a torn final entry, which readers already discard; nothing earlier is
 * damaged.
 */
export class ColightFileWriter {
  #fd: number | null;

  private constructor(fd: number) {
    this.#fd = fd;
  }

  /** Creates (or truncates) `path` and writes the initial-state entry. */
  static create(path: string, initial: Payload): ColightFileWriter {
    const fd = openSync(path, "w");
    try {
      writeSync(fd, createEntry(initial));
    } catch (error) {
      closeSync(fd);
      throw error;
    }
    return new ColightFileWriter(fd);
  }

  /**
   * Opens an existing conforming `.colight` file for further appends.
   *
   * Verifies the magic bytes and the 8-byte tail alignment before returning.
   */
  static open(path: string): ColightFileWriter {
    const size = statSync(path).size;
    if (size % 8 !== 0) {
      throw new Error(
        `Cannot append to ${path}: its length ${size} is not a multiple of 8.`,
      );
    }
    const probe = openSync(path, "r");
    try {
      const magic = new Uint8Array(8);
      readSync(probe, magic, 0, 8, 0);
      if (!magic.every((byte, i) => byte === MAGIC_BYTES[i])) {
        throw new Error(
          `${path} does not start with the .colight magic bytes "COLIGHT\\0".`,
        );
      }
    } finally {
      closeSync(probe);
    }
    return new ColightFileWriter(openSync(path, "a"));
  }

  /** Appends one update entry. */
  append(update: Payload): void {
    writeSync(this.#fdOrThrow(), createUpdateEntry(update));
  }

  /** Appends several update entries in order. */
  appendAll(updates: readonly Payload[]): void {
    const fd = this.#fdOrThrow();
    for (const update of updates) writeSync(fd, createUpdateEntry(update));
  }

  /** Forces everything written so far to stable storage (`fsync`). */
  flush(): void {
    fsyncSync(this.#fdOrThrow());
  }

  /** Whether {@link close} has been called. */
  get closed(): boolean {
    return this.#fd === null;
  }

  /** Closes the file. Idempotent. */
  close(): void {
    if (this.#fd !== null) {
      closeSync(this.#fd);
      this.#fd = null;
    }
  }

  #fdOrThrow(): number {
    if (this.#fd === null) {
      throw new Error("This ColightFileWriter is closed.");
    }
    return this.#fd;
  }
}
