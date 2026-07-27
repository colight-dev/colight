/**
 * The JSON section encoder.
 *
 * Spec §2.2 frees readers from depending on JSON whitespace, so nothing here is
 * required for correctness — but byte-identity with the Python writer is worth
 * keeping, and it hinges entirely on spelling choices JSON itself leaves open.
 * These tests pin the choices this writer makes, and cross-check the numeric
 * ones against Python over a large sample.
 */

import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  encodeJson,
  encodeJsonString,
  encodeNumber,
  pyFloat,
} from "@colight/format";

const REPO_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "..",
  "..",
);

/** Runs `json.dumps` on a list of Python expressions and returns the results. */
function pythonDumps(expressions: string[]): string[] {
  const program = [
    "import json, sys",
    "for line in sys.stdin.read().splitlines():",
    "    sys.stdout.write(json.dumps(eval(line)) + chr(10))",
  ].join("\n");
  const result = spawnSync("uv", ["run", "python", "-c", program], {
    cwd: REPO_ROOT,
    input: expressions.join("\n"),
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(`python failed: ${result.stderr}`);
  }
  return result.stdout.trimEnd().split("\n");
}

describe("compact separators", () => {
  it("emits no whitespace between tokens", () => {
    expect(encodeJson({ a: 1, b: [1, 2], c: { d: null } })).toBe(
      '{"a":1,"b":[1,2],"c":{"d":null}}',
    );
  });

  it("preserves object key insertion order", () => {
    expect(encodeJson({ z: 1, a: 2, m: 3 })).toBe('{"z":1,"a":2,"m":3}');
  });

  it("drops undefined members, like JSON.stringify", () => {
    expect(encodeJson({ a: 1, b: undefined as never, c: 2 })).toBe(
      '{"a":1,"c":2}',
    );
  });
});

describe("string escaping", () => {
  it("escapes every non-ASCII code point as \\uXXXX, like Python's ensure_ascii", () => {
    expect(encodeJsonString("héllo ☃")).toBe('"h\\u00e9llo \\u2603"');
  });

  it("escapes astral characters as surrogate pairs", () => {
    expect(encodeJsonString("🎈")).toBe('"\\ud83c\\udf88"');
  });

  it("uses the short escapes for the usual control characters", () => {
    expect(encodeJsonString('"\\\n\r\t\b\f')).toBe('"\\"\\\\\\n\\r\\t\\b\\f"');
  });

  it("escapes other control characters numerically", () => {
    expect(encodeJsonString("\u0000\u001f")).toBe('"\\u0000\\u001f"');
  });

  it("matches Python for a sample of strings", () => {
    const samples = [
      "",
      "plain",
      "héllo ☃",
      "🎈 balloon",
      '"quoted"',
      "tab\there",
    ];
    const expected = pythonDumps(samples.map((s) => JSON.stringify(s)));
    expect(samples.map(encodeJsonString)).toEqual(expected);
  });
});

describe("number spelling", () => {
  it("writes integral numbers without a decimal point by default", () => {
    expect(encodeNumber(1, false)).toBe("1");
    expect(encodeNumber(-0, false)).toBe("-0.0");
    expect(encodeNumber(1e15, false)).toBe("1000000000000000");
  });

  it("writes pyFloat-marked integral numbers with Python's float spelling", () => {
    expect(encodeJson({ a: pyFloat(1) })).toBe('{"a":1.0}');
    expect(encodeJson({ a: pyFloat(-2) })).toBe('{"a":-2.0}');
  });

  it("uses exponents where Python does, with a padded two-digit exponent", () => {
    // JS would write 1e-7 and 10000000000000000 here.
    expect(encodeNumber(1e-7, true)).toBe("1e-07");
    expect(encodeNumber(1e16, true)).toBe("1e+16");
    expect(encodeNumber(1e300, true)).toBe("1e+300");
  });

  it("refuses NaN and Infinity", () => {
    // Python emits the non-JSON tokens NaN/Infinity; JS emits null. Rather than
    // pick a side, the writer rejects them.
    expect(() => encodeNumber(NaN, false)).toThrow(/non-finite/);
    expect(() => encodeNumber(Infinity, false)).toThrow(/non-finite/);
    expect(() => encodeNumber(-Infinity, false)).toThrow(/non-finite/);
  });

  it("writes BigInt values beyond 2^53 exactly", () => {
    expect(encodeJson({ a: 2n ** 70n })).toBe(`{"a":${2n ** 70n}}`);
  });

  it("matches Python's json.dumps across the exponent switchover points", () => {
    const values: number[] = [];
    for (let exponent = -320; exponent <= 308; exponent++) {
      for (const mantissa of [1, 1.5, 9.999, 1.234567890123]) {
        const value = Number(`${mantissa}e${exponent}`);
        if (Number.isFinite(value) && value !== 0) values.push(value);
      }
    }
    // Round-trip through a hex float literal so Python parses the exact double
    // this test measured, not a decimal re-reading of it.
    const expected = pythonDumps(
      values.map((v) => `float.fromhex(${JSON.stringify(hexOf(v))})`),
    );
    expect(values.map((v) => encodeNumber(v, true))).toEqual(expected);
  });
});

/** A double's exact hex literal, so Python parses precisely this value. */
function hexOf(value: number): string {
  const buffer = new ArrayBuffer(8);
  new DataView(buffer).setFloat64(0, value);
  const bits = new DataView(buffer).getBigUint64(0);
  const sign = bits >> 63n ? "-" : "";
  const exponent = Number((bits >> 52n) & 0x7ffn);
  const mantissa = bits & 0xfffffffffffffn;
  if (exponent === 0) {
    return `${sign}0x0.${mantissa.toString(16).padStart(13, "0")}p-1022`;
  }
  return `${sign}0x1.${mantissa.toString(16).padStart(13, "0")}p${exponent - 1023}`;
}
