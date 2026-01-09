import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./packages/colight/tests/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "html",
  use: {
    baseURL: "http://127.0.0.1:8000",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium-webgpu",
      use: {
        ...devices["Desktop Chrome"],
        // Enable WebGPU in Chrome
        launchOptions: {
          args: [
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--use-angle=vulkan",
          ],
        },
      },
    },
  ],
  // Run local server before starting tests
  webServer: {
    command:
      "uv run python -m colight_cli serve packages/colight/tests/e2e/fixtures --port 8000",
    url: "http://127.0.0.1:8000",
    reuseExistingServer: !process.env.CI,
    timeout: 30000,
  },
});
