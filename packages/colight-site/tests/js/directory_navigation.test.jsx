import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LiveServerApp } from "../../src/js/live.jsx";
import {
  setupMockWebSocketServer,
  cleanupMockWebSocketServer,
  sendServerMessage,
  waitForConnection,
  waitForClientMessage,
} from "./test-utils/websocket-test-setup.js";

// Mock fetch
global.fetch = vi.fn();

let mockServer;
let mockSocket;

// Mock directory structure
const mockDirectoryTree = {
  name: "root",
  path: "",
  children: [
    {
      type: "directory",
      name: "src",
      path: "src",
      children: [
        {
          type: "directory",
          name: "components",
          path: "src/components",
          children: [
            {
              type: "directory",
              name: "ui",
              path: "src/components/ui",
              children: [
                {
                  type: "file",
                  name: "button.py",
                  path: "src/components/ui/button.py",
                },
              ],
            },
          ],
        },
        {
          type: "file",
          name: "main.py",
          path: "src/main.py",
        },
      ],
    },
    {
      type: "directory",
      name: "tests",
      path: "tests",
      children: [
        {
          type: "directory",
          name: "unit",
          path: "tests/unit",
          children: [
            {
              type: "file",
              name: "test_main.py",
              path: "tests/unit/test_main.py",
            },
          ],
        },
      ],
    },
  ],
};

describe("Directory Navigation", () => {
  let queryClient;

  beforeEach(async () => {
    // Setup mock WebSocket server
    mockServer = setupMockWebSocketServer();

    // Set up connection handler to capture socket
    mockServer.on("connection", (socket) => {
      mockSocket = socket;
    });

    vi.clearAllMocks();
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });

    // Mock fetch to return our directory structure
    global.fetch.mockImplementation((url) => {
      if (url.includes("/api/index")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockDirectoryTree),
        });
      }
      return Promise.resolve({
        ok: true,
        text: () => Promise.resolve(""),
      });
    });
  });

  afterEach(() => {
    cleanupMockWebSocketServer();
  });

  const renderApp = (initialPath = "/") => {
    return render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[initialPath]}>
          <Routes>
            <Route path="*" element={<LiveServerApp />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
  };

  it("should navigate to collapsed directories correctly", async () => {
    renderApp("/");

    // Wait for directory to load
    await waitFor(() => {
      expect(screen.getByText("src")).toBeTruthy();
    });

    // Click on src
    fireEvent.click(screen.getByText("src"));

    // Should navigate to /src/ and show its contents
    await waitFor(() => {
      expect(screen.getByText("components/ui")).toBeTruthy();
      expect(screen.getByText("main.py")).toBeTruthy();
    });

    // Now navigate to the collapsed path src/components/ui/
    fireEvent.click(screen.getByText("components/ui"));

    await waitFor(() => {
      // Should show the collapsed path
      expect(screen.getByText(/ui/)).toBeTruthy();
    });
  });

  it("should reliably navigate to files from directory browser", async () => {
    renderApp("/src/");

    // Wait for directory to load
    await waitFor(() => {
      expect(screen.getByText("main.py")).toBeTruthy();
    });

    // Wait for WebSocket connection
    await waitForConnection(mockServer);

    // Click on main.py
    fireEvent.click(screen.getByText("main.py"));

    // Should navigate to the file and request it via WebSocket
    const message = await waitForClientMessage(mockSocket);
    expect(message.type).toBe("request-load");
    expect(message.path).toBe("src/main.py");

    // Navigate to another file in a subdirectory
    fireEvent.click(screen.getByText("src")); // Click breadcrumb to go back

    await waitFor(() => {
      expect(screen.getByText("components/ui")).toBeTruthy();
    });

    fireEvent.click(screen.getByText("components/ui"));

    // Should see collapsed ui directory
    await waitFor(() => {
      expect(screen.getByText(/ui/)).toBeTruthy();
    });
  });

  it("should update breadcrumbs when navigating within directory browser", async () => {
    renderApp("/");

    // Wait for root to load
    await waitFor(() => {
      expect(screen.getByText("tests/unit")).toBeTruthy();
    });

    // Navigate to tests directory
    fireEvent.click(screen.getByText("tests/unit"));

    expect(screen.getByText("test_main.py")).toBeTruthy();

    fireEvent.click(screen.getByText("test_main.py"));

    await waitFor(() => {
      const breadcrumbs = screen.getAllByRole("button");
      const breadcrumbTexts = breadcrumbs.map((b) => b.textContent);
      expect(breadcrumbTexts).toContain("test_main.py");
    });
  });

  it("should use route as single source of truth for all navigation", async () => {
    const { rerender } = renderApp("/");

    // Initial load
    await waitFor(() => {
      expect(screen.getByText("src")).toBeTruthy();
    });

    // Wait for WebSocket connection
    await waitForConnection(mockServer);

    // Click on file
    fireEvent.click(screen.getByText("src"));
    expect(screen.getByText("main.py")).toBeTruthy();
    fireEvent.click(screen.getByText("main.py"));

    // Should request the file
    let message = await waitForClientMessage(mockSocket);
    expect(message.type).toBe("request-load");
    expect(message.path).toBe("src/main.py");

    // Open command bar
    fireEvent.keyDown(document, { key: "k", metaKey: true });

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Search files/i)).toBeTruthy();
    });

    // Search and select a file
    const searchInput = screen.getByPlaceholderText(/Search files/i);
    fireEvent.change(searchInput, { target: { value: "button" } });

    await waitFor(() => {
      expect(screen.getByText("button.py")).toBeTruthy();
    });

    // Set up promise to capture next message before clicking
    const messagePromise = new Promise((resolve) => {
      mockSocket.on("message", (data) => {
        resolve(JSON.parse(data));
      });
    });

    fireEvent.click(screen.getByText("button.py"));

    // Wait for the message
    message = await messagePromise;
    expect(message.type).toBe("request-load");
    expect(message.path).toBe("src/components/ui/button.py");
  });
});
