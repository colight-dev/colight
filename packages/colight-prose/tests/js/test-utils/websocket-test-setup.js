/**
 * WebSocket Test Setup using mock-socket
 *
 * This is our recommended approach for testing WebSocket functionality.
 * The mock-socket library provides a complete WebSocket implementation
 * that works well with websocket-ts.
 */

import { Server } from "mock-socket";
import { vi } from "vitest";

// Re-export everything from the simple mock for convenience
export * from "./websocket-mock-simple.js";

/**
 * Additional test utilities for common WebSocket testing patterns
 */

/**
 * Simulate a complete file load sequence
 * @param {Server} mockServer - The mock WebSocket server
 * @param {string} filePath - The file path to simulate loading
 * @param {Array} blocks - Array of block results to send
 */
export async function simulateFileLoad(mockServer, filePath, blocks = []) {
  const run = Date.now(); // Use timestamp as run ID

  // Send run-start
  mockServer.emit(
    "message",
    JSON.stringify({
      type: "run-start",
      file: filePath,
      run,
      blocks: blocks.map((b) => b.id),
      dirty: blocks.map((b) => b.id),
    }),
  );

  // Send each block result
  for (const block of blocks) {
    await new Promise((resolve) => setTimeout(resolve, 5)); // Small delay
    mockServer.emit(
      "message",
      JSON.stringify({
        type: "block-result",
        run,
        block: block.id,
        ok: true,
        elements: block.elements || [],
        ...block,
      }),
    );
  }

  // Send run-end
  await new Promise((resolve) => setTimeout(resolve, 5));
  mockServer.emit(
    "message",
    JSON.stringify({
      type: "run-end",
      run,
      error: null,
    }),
  );
}

/**
 * Create a mock block with default values
 */
export function createMockBlock(id, content) {
  return {
    id,
    elements: [
      {
        type: "prose",
        value: content,
        show: true,
      },
    ],
    ok: true,
  };
}

/**
 * Wait for element with retry (helps with act() warnings)
 */
export async function waitForElement(getElement, options = {}) {
  const { timeout = 3000, interval = 50 } = options;
  const endTime = Date.now() + timeout;

  while (Date.now() < endTime) {
    try {
      const element = getElement();
      if (element) return element;
    } catch (e) {
      // Element not found yet
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  throw new Error(`Element not found within ${timeout}ms`);
}
