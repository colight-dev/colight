/**
 * Two-way conformance between the JS `.colight` writer and the Python
 * reference implementation.
 *
 * Four directions, over the shared fixture set in `fixtures.ts`:
 *
 *  1. Python writes -> the existing JS reader reads. Establishes that the
 *     fixture descriptions round-trip at all, and that the portable decoded
 *     shape the other checks compare against is meaningful.
 *  2. JS writes -> Python reads (`parse_file_with_updates`, so update buffers
 *     are covered). This is the new capability.
 *  3. JS writes -> the existing JS reader reads.
 *  4. Byte comparison of the JS writer's output against Python's, for every
 *     fixture whose JSON spelling the spec pins.
 */

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { createFile } from "@colight/format";

import {
  FIXTURES,
  buildPayload,
  expectedValue,
  type Fixture,
} from "./fixtures.js";
import { jsReadFile } from "./jsReader.js";
import { pythonReadFile, pythonWriteFixture } from "./python.js";

let workdir: string;

beforeAll(() => {
  workdir = mkdtempSync(join(tmpdir(), "colight-format-conformance-"));
});

afterAll(() => {
  rmSync(workdir, { recursive: true, force: true });
});

/** Serializes a fixture with the JS writer under test. */
function writeWithJs(fixture: Fixture): Uint8Array {
  return createFile(
    fixture.initial === null ? null : buildPayload(fixture.initial),
    fixture.updates.map(buildPayload),
  );
}

/** The decoded values a conforming reader must report for a fixture. */
function expectedFor(fixture: Fixture) {
  return {
    initial: fixture.initial === null ? null : expectedValue(fixture.initial),
    updates: fixture.updates.map(expectedValue),
  };
}

describe.each(FIXTURES)("fixture $name", (fixture) => {
  it("Python writes -> the existing JS reader reads matching values", () => {
    const path = join(workdir, `${fixture.name}.py.colight`);
    pythonWriteFixture(fixture, path);
    const decoded = jsReadFile(new Uint8Array(readFileSync(path)));
    expect(decoded).toEqual(expectedFor(fixture));
  });

  it("JS writes -> Python reads matching values", () => {
    const path = join(workdir, `${fixture.name}.js.colight`);
    writeFileSync(path, writeWithJs(fixture));
    expect(pythonReadFile(path)).toEqual(expectedFor(fixture));
  });

  it("JS writes -> the existing JS reader reads matching values", () => {
    expect(jsReadFile(writeWithJs(fixture))).toEqual(expectedFor(fixture));
  });

  it("JS writes byte-identical output to Python", () => {
    const path = join(workdir, `${fixture.name}.bytes.colight`);
    pythonWriteFixture(fixture, path);
    const fromPython = new Uint8Array(readFileSync(path));
    const fromJs = writeWithJs(fixture);
    expect(Buffer.from(fromJs).toString("hex")).toBe(
      Buffer.from(fromPython).toString("hex"),
    );
  });
});
