import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { parseColightData } from "../../src/js/format.js";
import { evaluateNdarray } from "../../src/js/binary";

const __dirname = dirname(fileURLToPath(import.meta.url));
const testFile = join(
  __dirname,
  "..",
  "test-artifacts",
  "test-alignment.colight",
);

// Regression test for format v2 entry padding (spec section 2.2/2.3): a
// multi-entry file whose first entry has an odd unpadded length. In v1 the
// appended entry started at a misaligned absolute offset and the zero-copy
// Float64Array construction in evaluateNdarray threw a RangeError.
// The artifact is written by the Python writer in
// packages/colight/tests/test_format.py::test_alignment_artifact.
describe("Colight format entry alignment", () => {
  beforeAll(() => {
    if (!existsSync(testFile)) {
      throw new Error(
        `Test artifact ${testFile} not found. Run the Python tests first (uv run pytest packages/colight/tests/test_format.py) to create it.`,
      );
    }
  });

  it("zero-copy decodes a float64 array from an appended entry", () => {
    const data = parseColightData(readFileSync(testFile));

    expect(data.updateEntries).toHaveLength(1);
    const { data: update, buffers } = data.updateEntries[0];

    const envelope = update.state.big;
    expect(envelope.__type__).toBe("ndarray");
    expect(envelope.dtype).toBe("float64");

    const bufferView = buffers[envelope.__buffer_index__];
    // The whole point of entry padding: the buffer's absolute offset in the
    // file is 8-aligned, so the typed-array view can be constructed directly.
    expect(bufferView.byteOffset % 8).toBe(0);

    // In v1 this threw: RangeError: start offset of Float64Array should be a
    // multiple of 8.
    const result = evaluateNdarray({ ...envelope, data: bufferView });
    expect(result).toBeInstanceOf(Float64Array);
    expect(Array.from(result)).toEqual([1.5, -2.5, 3.5, 4.5]);
    // Zero-copy: the typed array is a view over the parsed file bytes, not a
    // copy.
    expect(result.buffer).toBe(bufferView.buffer);
  });
});
