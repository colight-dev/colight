/**
 * Drives `conformance.py` over a subprocess.
 *
 * Python is treated strictly as a black box here: it writes fixtures with
 * Colight's public writer and reads them back with its public parse API, and
 * the JS side is measured against what comes out.
 */

import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import type { Fixture, ExpectedValue } from "./fixtures.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const SCRIPT = resolve(HERE, "conformance.py");
const REPO_ROOT = resolve(HERE, "..", "..", "..");

function callPython(request: unknown): Record<string, unknown> {
  const result = spawnSync("uv", ["run", "python", SCRIPT], {
    cwd: REPO_ROOT,
    input: JSON.stringify(request),
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  if (result.error) {
    throw new Error(`Failed to run ${SCRIPT}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    throw new Error(
      `${SCRIPT} exited with ${result.status}:\n${result.stderr ?? ""}`,
    );
  }
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(result.stdout);
  } catch {
    throw new Error(
      `${SCRIPT} did not produce JSON.\nstdout: ${result.stdout}\nstderr: ${result.stderr}`,
    );
  }
  if (parsed.ok !== true) {
    throw new Error(
      `Python conformance helper failed: ${String(parsed.error)}`,
    );
  }
  return parsed;
}

/** Asks Python to write `fixture` to `path` with Colight's own writer. */
export function pythonWriteFixture(fixture: Fixture, path: string): void {
  callPython({
    command: "write",
    path,
    spec: { initial: fixture.initial, updates: fixture.updates },
  });
}

/** What Python's reader saw in a file. */
export interface PythonRead {
  initial: ExpectedValue | null;
  updates: ExpectedValue[];
}

/** Asks Python to parse `path` with `parse_file_with_updates`. */
export function pythonReadFile(path: string): PythonRead {
  const result = callPython({ command: "read", path });
  return {
    initial: result.initial as ExpectedValue | null,
    updates: result.updates as ExpectedValue[],
  };
}

/**
 * Asks Python to parse `path`, returning the error message instead of throwing.
 * Used to check that Python stops at (rather than chokes on) a torn tail.
 */
export function tryPythonReadFile(
  path: string,
): { ok: true; value: PythonRead } | { ok: false; error: string } {
  try {
    return { ok: true, value: pythonReadFile(path) };
  } catch (error) {
    return { ok: false, error: (error as Error).message };
  }
}
