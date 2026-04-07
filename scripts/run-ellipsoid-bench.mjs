#!/usr/bin/env node

import { createReadStream } from "node:fs";
import { access, stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import process from "node:process";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const defaultPort = 4173;
const defaultTimeoutMs = 30 * 60 * 1000;

function parseArgs(argv) {
  const options = {
    headed: false,
    serveOnly: false,
    skipBuild: false,
    host: "127.0.0.1",
    port: defaultPort,
    timeoutMs: defaultTimeoutMs,
    counts: null,
    frames: null,
    width: null,
    height: null,
    debug: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const nextValue = () => {
      if (i + 1 >= argv.length) {
        throw new Error(`Missing value for ${arg}`);
      }
      i += 1;
      return argv[i];
    };

    if (arg === "--headed") {
      options.headed = true;
      continue;
    }
    if (arg === "--serve-only") {
      options.serveOnly = true;
      continue;
    }
    if (arg === "--no-build") {
      options.skipBuild = true;
      continue;
    }
    if (arg === "--debug") {
      options.debug = true;
      continue;
    }
    if (arg === "--host") {
      options.host = nextValue();
      continue;
    }
    if (arg === "--port") {
      options.port = Number.parseInt(nextValue(), 10);
      continue;
    }
    if (arg === "--timeout-ms") {
      options.timeoutMs = Number.parseInt(nextValue(), 10);
      continue;
    }
    if (arg === "--counts") {
      options.counts = nextValue();
      continue;
    }
    if (arg === "--frames") {
      options.frames = Number.parseInt(nextValue(), 10);
      continue;
    }
    if (arg === "--width") {
      options.width = Number.parseInt(nextValue(), 10);
      continue;
    }
    if (arg === "--height") {
      options.height = Number.parseInt(nextValue(), 10);
      continue;
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!Number.isFinite(options.port) || options.port < 0) {
    throw new Error(`Invalid --port value: ${options.port}`);
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs <= 0) {
    throw new Error(`Invalid --timeout-ms value: ${options.timeoutMs}`);
  }
  if (options.frames != null && (!Number.isFinite(options.frames) || options.frames < 2)) {
    throw new Error(`Invalid --frames value: ${options.frames}`);
  }
  if (options.width != null && (!Number.isFinite(options.width) || options.width < 1)) {
    throw new Error(`Invalid --width value: ${options.width}`);
  }
  if (options.height != null && (!Number.isFinite(options.height) || options.height < 1)) {
    throw new Error(`Invalid --height value: ${options.height}`);
  }

  return options;
}

function runCommand(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: repoRoot,
      stdio: "inherit",
      shell: process.platform === "win32",
    });

    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(
        new Error(
          `${command} ${args.join(" ")} exited with ${signal ?? code ?? "unknown status"}`,
        ),
      );
    });
  });
}

async function ensureBuilt(skipBuild) {
  const bundlePath = path.join(
    repoRoot,
    "packages/colight/src/colight/js-dist/ellipsoid-bench.js",
  );

  if (!skipBuild) {
    await runCommand("yarn", ["build"]);
  }

  await access(bundlePath);
}

function getContentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  switch (ext) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".js":
    case ".mjs":
      return "application/javascript; charset=utf-8";
    case ".css":
      return "text/css; charset=utf-8";
    case ".json":
    case ".map":
      return "application/json; charset=utf-8";
    case ".svg":
      return "image/svg+xml";
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".wasm":
      return "application/wasm";
    default:
      return "application/octet-stream";
  }
}

async function resolveRequestPath(requestPath) {
  const requestUrl = new URL(requestPath, "http://bench.local");
  const relativePath =
    requestUrl.pathname === "/"
      ? "experiments/ellipsoid-bench/index.html"
      : decodeURIComponent(requestUrl.pathname.slice(1));

  let filePath = path.resolve(repoRoot, relativePath);
  const rootWithSep = `${repoRoot}${path.sep}`;
  if (filePath !== repoRoot && !filePath.startsWith(rootWithSep)) {
    return null;
  }

  let fileStat;
  try {
    fileStat = await stat(filePath);
  } catch (_err) {
    return null;
  }

  if (fileStat.isDirectory()) {
    filePath = path.join(filePath, "index.html");
    try {
      fileStat = await stat(filePath);
    } catch (_err) {
      return null;
    }
  }

  return fileStat.isFile() ? filePath : null;
}

async function startStaticServer(host, requestedPort) {
  const server = createServer(async (req, res) => {
    try {
      const filePath = await resolveRequestPath(req.url || "/");
      if (!filePath) {
        res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
        res.end("Not found");
        return;
      }

      res.writeHead(200, {
        "content-type": getContentType(filePath),
        "cache-control": "no-cache",
      });
      createReadStream(filePath).pipe(res);
    } catch (error) {
      res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
      res.end(error instanceof Error ? error.message : String(error));
    }
  });

  const listen = (port) =>
    new Promise((resolve, reject) => {
      const handleError = (error) => {
        server.off("listening", handleListening);
        reject(error);
      };
      const handleListening = () => {
        server.off("error", handleError);
        resolve();
      };
      server.once("error", handleError);
      server.once("listening", handleListening);
      server.listen(port, host);
    });

  try {
    await listen(requestedPort);
  } catch (error) {
    if (requestedPort !== 0 && error && error.code === "EADDRINUSE") {
      await listen(0);
    } else {
      throw error;
    }
  }

  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Failed to determine benchmark server address");
  }

  return {
    server,
    origin: `http://${host}:${address.port}`,
  };
}

function buildBenchUrl(origin, options) {
  const url = new URL("/experiments/ellipsoid-bench/", origin);

  if (!options.serveOnly) {
    url.searchParams.set("autorun", "1");
  }
  if (options.counts) {
    url.searchParams.set("counts", options.counts);
  }
  if (options.frames != null) {
    url.searchParams.set("frames", String(options.frames));
  }
  if (options.width != null) {
    url.searchParams.set("width", String(options.width));
  }
  if (options.height != null) {
    url.searchParams.set("height", String(options.height));
  }
  if (options.debug) {
    url.searchParams.set("scene3d_debug", "1");
  }

  return url;
}

function formatResultRows(results) {
  return results.map((result) => ({
    count: result.count.toLocaleString(),
    avgFps: result.averageFps,
    medianFps: result.medianFps,
    avgMs: result.averageMs,
    medianMs: result.medianMs,
    minMs: result.minMs,
    maxMs: result.maxMs,
    frames: result.sampleCount,
    precomputeMs: result.precomputeMs,
    datasetBytes: result.datasetBytes,
  }));
}

async function runBenchmark(url, options) {
  const { chromium } = await import("playwright");
  const browser = await chromium.launch({
    headless: !options.headed,
    args: [
      "--enable-unsafe-webgpu",
      "--ignore-gpu-blocklist",
      "--disable-dawn-features=disallow_unsafe_apis",
    ],
  });

  const page = await browser.newPage({
    viewport: {
      width: options.width ?? 1440,
      height: options.height ?? 960,
    },
  });

  page.on("console", (message) => {
    console.log(`[browser:${message.type()}] ${message.text()}`);
  });

  try {
    await page.goto(url.toString(), { waitUntil: "networkidle" });
    const stateHandle = await page.waitForFunction(
      () => {
        const bench = window.__COLIGHT_ELLIPSOID_BENCH__;
        if (!bench) return null;
        const state = bench.getState();
        if (state.status === "completed" || state.status === "error") {
          return state;
        }
        return null;
      },
      undefined,
      { timeout: options.timeoutMs },
    );

    const state = await stateHandle.jsonValue();
    if (!state) {
      throw new Error("Benchmark did not return a final state");
    }

    console.log(`Benchmark status: ${state.status}`);
    if (state.gpuInfo) {
      console.log("GPU info:");
      console.log(JSON.stringify(state.gpuInfo, null, 2));
    }
    if (state.results?.length) {
      console.table(formatResultRows(state.results));
    }
    if (state.status === "error") {
      throw new Error(state.error || "Benchmark failed");
    }
  } finally {
    await browser.close();
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));

  await ensureBuilt(options.skipBuild);
  const { server, origin } = await startStaticServer(options.host, options.port);
  const url = buildBenchUrl(origin, options);

  const shutdown = async () => {
    await new Promise((resolve, reject) => {
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }
        resolve();
      });
    });
  };

  const handleSignal = async (signal) => {
    try {
      await shutdown();
    } finally {
      process.exit(signal === "SIGINT" ? 130 : 143);
    }
  };

  const handleSigint = () => {
    void handleSignal("SIGINT");
  };
  const handleSigterm = () => {
    void handleSignal("SIGTERM");
  };

  process.on("SIGINT", handleSigint);
  process.on("SIGTERM", handleSigterm);

  console.log(`Serving ellipsoid bench at ${url.toString()}`);

  try {
    if (options.serveOnly) {
      await new Promise(() => {});
      return;
    }

    await runBenchmark(url, options);
  } finally {
    process.off("SIGINT", handleSigint);
    process.off("SIGTERM", handleSigterm);
    await shutdown();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : error);
  process.exitCode = 1;
});
