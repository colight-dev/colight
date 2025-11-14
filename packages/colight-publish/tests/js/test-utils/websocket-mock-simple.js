import { Server } from "mock-socket";
import { vi } from "vitest";

/**
 * Simple WebSocket testing setup using mock-socket library
 *
 * This provides a clean way to test WebSocket interactions without
 * complex manual mocking.
 */

let mockServer = null;

/**
 * Setup a mock WebSocket server for testing
 * @param {string} url - The WebSocket URL to mock (default: ws://127.0.0.1:*)
 * @returns {Server} The mock server instance
 */
export function setupMockWebSocketServer(url) {
  // Extract port from window.location.port + 1 logic in the app
  const port =
    typeof window !== "undefined"
      ? parseInt(window.location.port || "5500") + 1
      : 5501;

  const wsUrl = url || `ws://127.0.0.1:${port}`;

  // Clean up any existing server
  if (mockServer) {
    mockServer.stop();
  }

  // Create new mock server
  mockServer = new Server(wsUrl);

  return mockServer;
}

/**
 * Clean up the mock server
 */
export function cleanupMockWebSocketServer() {
  if (mockServer) {
    mockServer.stop();
    mockServer = null;
  }
}

/**
 * Helper to simulate server sending a message to all clients
 */
export function sendServerMessage(message) {
  if (mockServer) {
    mockServer.emit("message", JSON.stringify(message));
  }
}

/**
 * Helper to wait for a client connection
 */
export async function waitForConnection(server) {
  return new Promise((resolve) => {
    server.on("connection", (socket) => {
      resolve(socket);
    });
  });
}

/**
 * Helper to wait for a client message
 */
export async function waitForClientMessage(socket) {
  return new Promise((resolve) => {
    socket.on("message", (data) => {
      resolve(JSON.parse(data));
    });
  });
}

/**
 * Example test setup:
 *
 * import { setupMockWebSocketServer, cleanupMockWebSocketServer, sendServerMessage } from './websocket-mock-simple';
 *
 * describe("WebSocket Tests", () => {
 *   let mockServer;
 *
 *   beforeEach(() => {
 *     mockServer = setupMockWebSocketServer();
 *   });
 *
 *   afterEach(() => {
 *     cleanupMockWebSocketServer();
 *   });
 *
 *   it("should handle messages", async () => {
 *     // Render your component
 *     // The component will connect to the mock server automatically
 *
 *     // Send a message from the "server"
 *     sendServerMessage({
 *       type: "run-start",
 *       file: "test.py",
 *       run: 1
 *     });
 *
 *     // Assert on component behavior
 *   });
 * });
 */
