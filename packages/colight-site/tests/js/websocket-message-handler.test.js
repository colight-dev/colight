import { describe, it, expect } from "vitest";
import {
  processRunStart,
  processBlockResult,
  processRunEnd,
  processWebSocketMessage,
} from "../../src/js/websocket-message-handler";

describe("WebSocket Message Handler", () => {
  describe("processRunStart", () => {
    it("should ignore old runs", () => {
      const result = processRunStart(
        { run: 1, file: "test.py" },
        2, // latestRun is 2
        {},
      );
      expect(result).toBe(null);
    });

    it("should process new run with manifest", () => {
      const result = processRunStart(
        {
          run: 2,
          file: "test.py",
          blocks: ["block-1", "block-2"],
          dirty: ["block-2"],
        },
        1, // latestRun is 1
        { "block-1": { result: "old" } },
      );

      expect(result).toEqual({
        latestRun: 2,
        currentFile: "test.py",
        changedBlocks: new Set(),
        blockResults: {
          "block-1": { result: "old", pending: false },
          "block-2": { pending: true, elements: [], ok: true },
        },
      });
    });
  });

  describe("processBlockResult", () => {
    it("should handle unchanged blocks", () => {
      const changedBlocks = new Set();
      const result = processBlockResult(
        { run: 1, block: "block-1", unchanged: true },
        1,
        { "block-1": { result: "old", pending: true } },
        changedBlocks,
      );

      expect(result).toEqual({
        blockId: "block-1",
        blockResult: { result: "old", pending: false },
        changed: false,
      });
      expect(changedBlocks.size).toBe(0);
    });

    it("should handle changed blocks", () => {
      const changedBlocks = new Set();
      const result = processBlockResult(
        {
          run: 1,
          block: "block-1",
          content_changed: true,
          ok: true,
          stdout: "output",
        },
        1,
        {},
        changedBlocks,
      );

      expect(result.blockId).toBe("block-1");
      expect(result.blockResult.stdout).toBe("output");
      expect(result.changed).toBe(true);
      expect(changedBlocks.has("block-1")).toBe(true);
    });
  });

  describe("processWebSocketMessage", () => {
    it("should handle run-start messages", () => {
      const state = {
        latestRun: 1,
        blockResults: {},
        changedBlocks: new Set(),
      };

      const action = processWebSocketMessage(
        { type: "run-start", run: 2, file: "test.py", blocks: [], dirty: [] },
        state,
      );

      expect(action.type).toBe("run-start");
      expect(action.latestRun).toBe(2);
    });

    it("should handle reload messages", () => {
      const state = {
        latestRun: 1,
        blockResults: {},
        changedBlocks: new Set(),
      };

      const action = processWebSocketMessage({ type: "reload" }, state);
      expect(action).toEqual({ type: "reload" });
    });

    it("should handle unknown messages", () => {
      const state = {
        latestRun: 1,
        blockResults: {},
        changedBlocks: new Set(),
      };

      const action = processWebSocketMessage({ type: "mystery" }, state);
      expect(action).toEqual({ type: "unknown", messageType: "mystery" });
    });
  });
});
