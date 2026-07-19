import { describe, it, expect } from "vitest";
import { parseColightData, CURRENT_VERSION } from "../../src/js/format.js";

// Build a minimal single-entry .colight file with the given version.
// Layout per docs/src/colight_docs/format.md.
function createEntry(jsonData, version) {
  const encoder = new TextEncoder();
  const HEADER_SIZE = 96;
  const jsonBytes = encoder.encode(JSON.stringify(jsonData));
  const jsonLength = jsonBytes.length;
  // align8(json end), as the Python writer does
  const binaryOffset = (HEADER_SIZE + jsonLength + 7) & ~7;

  const result = new Uint8Array(binaryOffset);
  const view = new DataView(result.buffer);

  result.set(encoder.encode("COLIGHT\0"), 0);
  view.setBigUint64(8, version, true);
  view.setBigUint64(16, BigInt(HEADER_SIZE), true);
  view.setBigUint64(24, BigInt(jsonLength), true);
  view.setBigUint64(32, BigInt(binaryOffset), true);
  view.setBigUint64(40, 0n, true); // binary length
  view.setBigUint64(48, 0n, true); // num buffers
  result.set(jsonBytes, HEADER_SIZE);

  return result;
}

describe("Colight format versioning", () => {
  it("supports exactly version 2", () => {
    expect(CURRENT_VERSION).toBe(2n);
  });

  it("accepts the current version", () => {
    const data = createEntry({ ast: null, state: {} }, CURRENT_VERSION);
    const result = parseColightData(data);
    expect(result.state).toEqual({});
    expect(result.buffers).toEqual([]);
  });

  it("rejects a bumped version, naming found and supported versions", () => {
    const data = createEntry({ ast: null, state: {} }, CURRENT_VERSION + 1n);
    expect(() => parseColightData(data)).toThrow(
      `Unsupported .colight file version: found ${CURRENT_VERSION + 1n}, ` +
        `this reader supports version ${CURRENT_VERSION}`,
    );
  });

  it("rejects version 0 (never issued)", () => {
    const data = createEntry({ ast: null, state: {} }, 0n);
    expect(() => parseColightData(data)).toThrow(
      "Unsupported .colight file version: found 0",
    );
  });

  it("rejects version 1 (pre-release; no back-compat window)", () => {
    const data = createEntry({ ast: null, state: {} }, 1n);
    expect(() => parseColightData(data)).toThrow(
      `Unsupported .colight file version: found 1, ` +
        `this reader supports version ${CURRENT_VERSION}`,
    );
  });
});
