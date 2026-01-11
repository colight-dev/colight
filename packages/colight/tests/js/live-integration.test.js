import { describe, it, expect, vi } from "vitest";
// TODO: Update tests to work with refactored code structure
// import { createWebSocketMessageHandler } from "../../src/js/live.jsx";

describe.skip("Live Server Integration", () => {
  it("should preserve unchanged blocks when editing a single block", () => {
    // Setup initial state with multiple blocks
    const latestRunRef = { current: 1 };
    const changedBlocksRef = { current: new Set() };

    // Initial block results - simulating blocks that are already loaded
    const blockResultsRef = {
      current: {
        "block-1": {
          elements: [{ type: "prose", value: "Block 1 content", show: true }],
          ok: true,
          pending: false,
        },
        "block-2": {
          elements: [
            { type: "expression", value: "print('hello')", show: true },
          ],
          ok: true,
          pending: false,
        },
        "block-3": {
          elements: [{ type: "prose", value: "Block 3 content", show: true }],
          ok: true,
          pending: false,
        },
      },
    };

    let currentFile = "test.py";
    let blockResults = { ...blockResultsRef.current };

    const setCurrentFile = vi.fn((file) => {
      currentFile = file;
    });

    const setBlockResults = vi.fn((updater) => {
      if (typeof updater === "function") {
        blockResults = updater(blockResults);
      } else {
        blockResults = updater;
      }
      blockResultsRef.current = blockResults;
    });

    const handleMessage = createWebSocketMessageHandler({
      latestRunRef,
      blockResultsRef,
      changedBlocksRef,
      setCurrentFile,
      setBlockResults,
    });

    // Simulate editing block-2
    // Server sends run-start with all blocks, only block-2 is dirty
    handleMessage({
      run: 2,
      type: "run-start",
      file: "test.py",
      blocks: ["block-1", "block-2", "block-3"],
      dirty: ["block-2"], // Only block-2 changed
    });

    // Check that blockResults was updated
    expect(setBlockResults).toHaveBeenCalled();

    // Get the new blockResults after run-start
    const newBlockResults = setBlockResults.mock.calls[0][0];
    console.log("Block results after run-start:", newBlockResults);

    // Verify that all blocks are still present
    expect(Object.keys(newBlockResults)).toEqual([
      "block-1",
      "block-2",
      "block-3",
    ]);

    // Verify block-1 and block-3 kept their content
    expect(newBlockResults["block-1"].elements).toEqual([
      { type: "prose", value: "Block 1 content", show: true },
    ]);
    expect(newBlockResults["block-3"].elements).toEqual([
      { type: "prose", value: "Block 3 content", show: true },
    ]);

    // Verify block-2 is marked as pending
    expect(newBlockResults["block-2"].pending).toBe(true);

    // Now simulate server sending unchanged for blocks 1 and 3
    handleMessage({
      run: 2,
      type: "block-result",
      block: "block-1",
      unchanged: true,
    });

    handleMessage({
      run: 2,
      type: "block-result",
      block: "block-3",
      unchanged: true,
    });

    // And new content for block-2
    handleMessage({
      run: 2,
      type: "block-result",
      block: "block-2",
      ok: true,
      elements: [{ type: "expression", value: "print('updated')", show: true }],
      content_changed: true,
    });

    // Final state check
    expect(blockResults["block-1"].elements).toEqual([
      { type: "prose", value: "Block 1 content", show: true },
    ]);
    expect(blockResults["block-2"].elements).toEqual([
      { type: "expression", value: "print('updated')", show: true },
    ]);
    expect(blockResults["block-3"].elements).toEqual([
      { type: "prose", value: "Block 3 content", show: true },
    ]);
  });

  it("should handle initial load with all unchanged blocks", () => {
    // Setup fresh client state
    const latestRunRef = { current: 0 };
    const changedBlocksRef = { current: new Set() };
    const blockResultsRef = { current: {} }; // Empty initially

    let currentFile = null;
    let blockResults = {};

    const setCurrentFile = vi.fn((file) => {
      currentFile = file;
    });

    const setBlockResults = vi.fn((updater) => {
      if (typeof updater === "function") {
        blockResults = updater(blockResults);
      } else {
        blockResults = updater;
      }
      blockResultsRef.current = blockResults;
    });

    const handleMessage = createWebSocketMessageHandler({
      latestRunRef,
      blockResultsRef,
      changedBlocksRef,
      setCurrentFile,
      setBlockResults,
    });

    // Simulate server on run 5, client on run 0
    // Server sends run-start with no dirty blocks (all cached)
    handleMessage({
      run: 5,
      type: "run-start",
      file: "test.py",
      blocks: ["block-1", "block-2", "block-3"],
      dirty: [], // All blocks are cached
    });

    // Check the initial state after run-start
    console.log("Initial block results:", blockResults);

    // All blocks should be pending since we have no data
    expect(blockResults["block-1"].pending).toBe(true);
    expect(blockResults["block-2"].pending).toBe(true);
    expect(blockResults["block-3"].pending).toBe(true);

    // Server sends "unchanged" for all blocks
    handleMessage({
      run: 5,
      type: "block-result",
      block: "block-1",
      unchanged: true,
    });

    // Check what happens after unchanged message
    console.log("Block-1 after unchanged:", blockResults["block-1"]);

    // The block should no longer be pending but still have empty elements
    expect(blockResults["block-1"].pending).toBe(false);
    expect(blockResults["block-1"].elements).toEqual([]);
  });

  it("should handle spreading undefined block data", () => {
    const result = {
      ...undefined,
      pending: false,
    };

    console.log("Result of spreading undefined:", result);
    expect(result).toEqual({ pending: false });

    // This is what happens when currentBlockResults[blockId] is undefined
    const blockData = undefined;
    const newBlock = {
      ...blockData,
      pending: false,
    };

    expect(newBlock).toEqual({ pending: false });
    expect(newBlock.elements).toBeUndefined();
  });

  it("should handle ref synchronization issues", () => {
    // This simulates the case where blockResultsRef might be out of sync
    const latestRunRef = { current: 1 };
    const changedBlocksRef = { current: new Set() };

    // Start with empty ref
    const blockResultsRef = { current: {} };

    let blockResults = {
      "block-1": {
        elements: [{ type: "prose", value: "Block 1 content", show: true }],
        ok: true,
        pending: false,
      },
    };

    const setCurrentFile = vi.fn();
    const setBlockResults = vi.fn((updater) => {
      if (typeof updater === "function") {
        blockResults = updater(blockResults);
      } else {
        blockResults = updater;
      }
      // Simulate React not updating the ref immediately
      // blockResultsRef.current = blockResults; // NOT updating ref
    });

    const handleMessage = createWebSocketMessageHandler({
      latestRunRef,
      blockResultsRef, // This ref is stale!
      changedBlocksRef,
      setCurrentFile,
      setBlockResults,
    });

    // Edit the block - server sends run-start
    handleMessage({
      run: 2,
      type: "run-start",
      file: "test.py",
      blocks: ["block-1"],
      dirty: ["block-1"],
    });

    // Check what was passed to setBlockResults
    const newBlockResults = setBlockResults.mock.calls[0][0];
    console.log("New block results with stale ref:", newBlockResults);

    // Since blockResultsRef.current is empty, the block will be created as new
    expect(newBlockResults["block-1"].elements).toEqual([]);
    expect(newBlockResults["block-1"].pending).toBe(true);
  });
});
