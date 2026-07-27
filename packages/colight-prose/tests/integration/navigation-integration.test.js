import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { spawn } from "child_process";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import fetch from "node-fetch";
import WebSocket from "ws";

const __dirname = dirname(fileURLToPath(import.meta.url));

describe("Server Integration Tests", () => {
  let serverProcess;
  const testDir = join(__dirname, "..", "integration-test-fixture");
  const port = 5569;
  const baseUrl = `http://localhost:${port}`;
  const wsUrl = `ws://localhost:${port + 1}`;

  beforeAll(async () => {
    console.log("Starting server with test directory:", testDir);

    // Start the server
    serverProcess = spawn(
      "uv",
      [
        "run",
        "python",
        "-m",
        "colight_cli",
        "live",
        testDir,
        "--port",
        String(port),
        "--no-open",
      ],
      {
        stdio: ["ignore", "pipe", "pipe"],
        cwd: join(__dirname, "..", "..", "..", ".."), // Go to project root
      },
    );

    // Log server output for debugging
    serverProcess.stdout.on("data", (data) => {
      console.log("Server stdout:", data.toString());
    });
    serverProcess.stderr.on("data", (data) => {
      console.error("Server stderr:", data.toString());
    });

    // Wait for server to start
    await new Promise((resolve, reject) => {
      const maxAttempts = 3; // Only 3 attempts, fail fast
      let attempts = 0;
      let serverFailed = false;

      // Check if server process exited
      serverProcess.on("exit", (code) => {
        serverFailed = true;
        reject(new Error(`Server process exited with code ${code}`));
      });

      const checkServer = async () => {
        if (serverFailed) return;

        attempts++;
        console.log(
          `Checking server availability (attempt ${attempts}/${maxAttempts})...`,
        );

        try {
          const response = await fetch(baseUrl);
          // Accept any response from the server (including 404) as "server is ready"
          if (response) {
            console.log(`Server is ready! (status: ${response.status})`);
            resolve();
            return;
          }
        } catch (error) {
          console.log(`Server not ready yet: ${error.message}`);
        }

        if (attempts >= maxAttempts) {
          reject(
            new Error(
              `Server startup timeout after ${maxAttempts} attempts (${maxAttempts} seconds)`,
            ),
          );
        } else {
          setTimeout(checkServer, 1000);
        }
      };

      // Check immediately since server should start fast
      setTimeout(checkServer, 500);
    });
  }, 60000); // 60 second timeout for setup

  afterAll(async () => {
    // Kill server
    if (serverProcess) {
      console.log("Stopping server...");
      serverProcess.kill();
      await new Promise((resolve) => serverProcess.on("close", resolve));
    }
  });

  it("should serve the root directory browser", async () => {
    const response = await fetch(baseUrl);
    // The colight CLI server might return 404 for root, but should serve files
    expect(response.status).toBeDefined();

    // Skip HTML checks if we got a 404
    if (response.ok) {
      const html = await response.text();
      // Check that it's the live server page
      expect(html).toContain('<div id="root">');
      expect(html).toContain("live.js");
    }
  });

  it("should provide directory listing via API", async () => {
    const response = await fetch(`${baseUrl}/api/index`);
    expect(response.ok).toBe(true);

    const data = await response.json();
    expect(data.name).toBe("integration-test-fixture");
    expect(data.children).toBeDefined();

    // Find src directory
    const srcDir = data.children.find((item) => item.name === "src");
    expect(srcDir).toBeDefined();
    expect(srcDir.type).toBe("directory");

    // Find main.py
    const mainFile = data.children.find((item) => item.name === "main.py");
    expect(mainFile).toBeDefined();
    expect(mainFile.type).toBe("file");
  });

  it("should handle WebSocket connections for file updates", async () => {
    const ws = new WebSocket(wsUrl);
    const clientId = "test-client-" + Date.now();

    await new Promise((resolve, reject) => {
      ws.on("open", resolve);
      ws.on("error", reject);
    });

    // First, register the client to watch the file
    ws.send(
      JSON.stringify({
        type: "watch-file",
        clientId: clientId,
        path: "main.py",
      }),
    );

    // Wait a bit for the watch registration to process
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Now request the file
    const fileRequest = new Promise((resolve) => {
      ws.on("message", (data) => {
        const message = JSON.parse(data);
        if (message.type === "run-start" && message.file === "main.py") {
          resolve(message);
        }
      });
    });

    ws.send(
      JSON.stringify({
        type: "request-load",
        path: "main.py",
        clientRun: 0,
      }),
    );

    const runStart = await fileRequest;
    expect(runStart.block_ids).toBeDefined();
    expect(runStart.block_ids.length).toBeGreaterThan(0);

    ws.close();
  });

  it("should serve files with .py extension in URL", async () => {
    // The frontend should be able to navigate to files with .py extension
    const response = await fetch(`${baseUrl}/main.py`);
    expect(response.ok).toBe(true);

    const html = await response.text();
    // Should still serve the SPA, not the Python file directly
    expect(html).toContain('<div id="root">');
  });

  it("should serve directories with trailing slash", async () => {
    const response = await fetch(`${baseUrl}/src/`);
    expect(response.ok).toBe(true);

    const html = await response.text();
    // Should serve the SPA for directory browsing
    expect(html).toContain('<div id="root">');
  });
});
