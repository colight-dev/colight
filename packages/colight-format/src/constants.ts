/**
 * Constants of the `.colight` container format, version 2.
 *
 * Written from the specification in `docs/src/colight_docs/format.md` alone.
 */

/** `"COLIGHT\0"` — the 8 magic bytes at offset 0 of every entry. */
export const MAGIC_BYTES: Uint8Array = new Uint8Array([
  0x43, 0x4f, 0x4c, 0x49, 0x47, 0x48, 0x54, 0x00,
]);

/** Fixed header size in bytes (spec §2.1). */
export const HEADER_SIZE = 96;

/** The format version this package writes (spec §5). */
export const CURRENT_VERSION = 2;

/**
 * Every offset and length in the format is aligned to this many bytes so that
 * typed-array views can be taken over the file bytes without copying (spec §2.3).
 */
export const ALIGNMENT = 8;

/** Rounds `n` up to the next multiple of 8 (spec §2.2, `align8`). */
export function align8(n: number): number {
  return Math.ceil(n / ALIGNMENT) * ALIGNMENT;
}
