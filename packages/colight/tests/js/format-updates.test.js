import { describe, it, expect } from "vitest";
import { parseColightData } from "../../src/js/format.js";

describe("Colight Format Updates", () => {
  // Helper to create a .colight entry
  function createEntry(jsonData) {
    const encoder = new TextEncoder();
    const HEADER_SIZE = 96;
    const jsonBytes = encoder.encode(JSON.stringify(jsonData));
    const jsonLength = jsonBytes.length;
    const binaryOffset = HEADER_SIZE + jsonLength;

    const header = new Uint8Array(HEADER_SIZE);
    const view = new DataView(header.buffer);

    // Magic bytes
    header.set(encoder.encode("COLIGHT\0"), 0);

    // Version
    view.setBigUint64(8, 1n, true);

    // JSON offset and length
    view.setBigUint64(16, BigInt(HEADER_SIZE), true);
    view.setBigUint64(24, BigInt(jsonLength), true);

    // Binary offset and length (no buffers)
    view.setBigUint64(32, BigInt(binaryOffset), true);
    view.setBigUint64(40, 0n, true);

    // Number of buffers
    view.setBigUint64(48, 0n, true);

    // Combine header and JSON
    const result = new Uint8Array(header.length + jsonLength);
    result.set(header, 0);
    result.set(jsonBytes, header.length);

    return result;
  }

  it("should parse file with initial state and updates", () => {
    // Create file with initial state and updates
    const initialEntry = createEntry({
      marks: [{ type: "raster" }],
      data: [
        [1, 2, 3],
        [4, 5, 6],
      ],
    });
    const update1 = createEntry({
      updates: { ast: null, state: { count: 0 } },
    });
    const update2 = createEntry({
      updates: { ast: { marks: [{ type: "text" }] }, state: {} },
    });

    const testFileWithUpdates = new Uint8Array(
      initialEntry.length + update1.length + update2.length,
    );
    testFileWithUpdates.set(initialEntry, 0);
    testFileWithUpdates.set(update1, initialEntry.length);
    testFileWithUpdates.set(update2, initialEntry.length + update1.length);

    const result = parseColightData(testFileWithUpdates);

    // Should have initial state
    expect(result).toHaveProperty("marks");
    expect(result).toHaveProperty("buffers");

    // Should have update entries
    expect(result).toHaveProperty("updateEntries");
    expect(Array.isArray(result.updateEntries)).toBe(true);
    expect(result.updateEntries.length).toBe(2);

    // Verify update structure
    const firstUpdate = result.updateEntries[0];
    expect(firstUpdate).toHaveProperty("data");
    expect(firstUpdate).toHaveProperty("buffers");
    expect(firstUpdate.data.state.count).toBe(0);

    const secondUpdate = result.updateEntries[1];
    expect(secondUpdate.data.ast).toHaveProperty("marks");
  });

  it("should parse update-only file", () => {
    // Create update-only file
    const updateOnly1 = createEntry({
      updates: { ast: null, state: { theme: "dark", zoom: 2.0 } },
    });
    const updateOnly2 = createEntry({
      updates: { ast: null, state: { theme: "light" } },
    });

    const testFileUpdatesOnly = new Uint8Array(
      updateOnly1.length + updateOnly2.length,
    );
    testFileUpdatesOnly.set(updateOnly1, 0);
    testFileUpdatesOnly.set(updateOnly2, updateOnly1.length);

    const result = parseColightData(testFileUpdatesOnly);

    // Should not have initial state
    expect(result).not.toHaveProperty("marks");
    expect(result).not.toHaveProperty("buffers");

    // Should only have update entries
    expect(result).toHaveProperty("updateEntries");
    expect(Array.isArray(result.updateEntries)).toBe(true);
    expect(result.updateEntries.length).toBe(2);

    // Verify update contents
    expect(result.updateEntries[0].data.state.theme).toBe("dark");
    expect(result.updateEntries[0].data.state.zoom).toBe(2.0);
    expect(result.updateEntries[1].data.state.theme).toBe("light");
  });

  it("should handle multiple entries correctly", () => {
    // Create file with initial state and two updates
    const entry1 = createEntry({ data: [1, 2, 3] });
    const entry2 = createEntry({ updates: { ast: null, state: { x: 10 } } });
    const entry3 = createEntry({ updates: { ast: null, state: { y: 20 } } });

    const combined = new Uint8Array(
      entry1.length + entry2.length + entry3.length,
    );
    combined.set(entry1, 0);
    combined.set(entry2, entry1.length);
    combined.set(entry3, entry1.length + entry2.length);

    const result = parseColightData(combined);

    // Should have initial data
    expect(result.data).toEqual([1, 2, 3]);

    // Should have both update entries
    expect(result.updateEntries).toHaveLength(2);
    expect(result.updateEntries[0].data.state.x).toBe(10);
    expect(result.updateEntries[1].data.state.y).toBe(20);
  });
});
