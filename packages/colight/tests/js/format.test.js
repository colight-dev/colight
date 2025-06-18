import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import {
  parseColightData,
  loadColightFile,
} from "../../src/colight/js/format.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const testArtifactsDir = join(__dirname, "..", "test-artifacts");
const testColightFile = join(testArtifactsDir, "test-raster.colight");

describe("Colight Format", () => {
  beforeAll(() => {
    // This test relies on Python tests creating the test file first
    if (!existsSync(testColightFile)) {
      throw new Error(
        `Test artifact ${testColightFile} not found. Make sure Python tests run first to create test files.`,
      );
    }
  });

  it("should parse a .colight file created by Python", async () => {
    const fileData = readFileSync(testColightFile);
    const data = parseColightData(fileData);
    const { buffers } = data;

    // Verify structure
    expect(data).toBeDefined();
    expect(data.ast).toBeDefined();
    expect(data.bufferLayout).toBeDefined();
    expect(buffers).toBeDefined();
    expect(Array.isArray(buffers)).toBe(true);

    // Verify buffer layout
    const layout = data.bufferLayout;
    expect(layout.offsets).toBeDefined();
    expect(layout.lengths).toBeDefined();
    expect(layout.count).toBeDefined();
    expect(layout.totalSize).toBeDefined();

    // Verify buffers match layout
    expect(buffers.length).toBe(layout.count);

    // Verify each buffer has correct length
    for (let i = 0; i < buffers.length; i++) {
      // Buffers should be DataView objects
      expect(buffers[i]).toBeInstanceOf(DataView);
      expect(buffers[i].byteLength).toBe(layout.lengths[i]);
    }

    // Verify total size matches
    const totalSize = buffers.reduce((sum, buf) => sum + buf.byteLength, 0);
    expect(totalSize).toBe(layout.totalSize);
  });

  it("should validate file format correctly", () => {
    // Test with invalid magic bytes
    const invalidMagic = new Uint8Array(96);
    invalidMagic.set(new TextEncoder().encode("INVALID\0"), 0);

    expect(() => parseColightData(invalidMagic)).toThrow("Wrong magic bytes");

    // Test with too short data
    const tooShort = new Uint8Array(50);
    expect(() => parseColightData(tooShort)).toThrow("Too short");
  });

  it("should handle buffer references in AST", () => {
    const fileData = readFileSync(testColightFile);
    const data = parseColightData(fileData);
    const { buffers } = data;

    // Function to recursively find buffer references
    function findBufferRefs(obj, refs = []) {
      if (obj && typeof obj === "object") {
        if (obj.__type__ === "buffer" && typeof obj.index === "number") {
          refs.push(obj.index);
        } else if (obj.__buffer_index__ !== undefined) {
          refs.push(obj.__buffer_index__);
        }

        if (Array.isArray(obj)) {
          obj.forEach((item) => findBufferRefs(item, refs));
        } else {
          Object.values(obj).forEach((value) => findBufferRefs(value, refs));
        }
      }
      return refs;
    }

    const bufferRefs = findBufferRefs(data);

    // All buffer references should be valid indices
    for (const ref of bufferRefs) {
      expect(ref).toBeGreaterThanOrEqual(0);
      expect(ref).toBeLessThan(buffers.length);
      expect(buffers[ref]).toBeInstanceOf(DataView);
    }
  });

  it("should create buffer views efficiently", () => {
    const fileData = readFileSync(testColightFile);
    const data = parseColightData(fileData);
    const { buffers } = data;

    // Verify buffers are DataView objects (not copies)
    for (const buffer of buffers) {
      expect(buffer).toBeInstanceOf(DataView);
      expect(buffer.byteLength).toBeGreaterThan(0);
    }
  });

  it("should load file from URL (when used with a server)", async () => {
    // This test would require a test server, so we'll just test the file loading logic
    // by creating a mock fetch
    const originalFetch = global.fetch;

    try {
      const fileData = readFileSync(testColightFile);
      // Convert Node.js Buffer to ArrayBuffer properly
      const arrayBuffer = new ArrayBuffer(fileData.length);
      const view = new Uint8Array(arrayBuffer);
      view.set(fileData);

      global.fetch = async () => ({
        ok: true,
        arrayBuffer: async () => arrayBuffer,
      });

      const data = await loadColightFile("mock://test.colight");
      const { buffers } = data;

      expect(data).toBeDefined();
      expect(buffers).toBeDefined();
      expect(Array.isArray(buffers)).toBe(true);
    } finally {
      global.fetch = originalFetch;
    }
  });

  it("should handle fetch errors gracefully", async () => {
    const originalFetch = global.fetch;

    try {
      global.fetch = async () => ({
        ok: false,
        status: 404,
        statusText: "Not Found",
      });

      await expect(loadColightFile("mock://missing.colight")).rejects.toThrow(
        "Failed to fetch",
      );
    } finally {
      global.fetch = originalFetch;
    }
  });
});
