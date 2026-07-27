/**
 * JSON encoding for the `.colight` JSON section.
 *
 * Spec §2.2 requires the section to be a single JSON object encoded as UTF-8
 * without a BOM, and explicitly frees readers from depending on any particular
 * whitespace. Everything below the "must parse as the same JSON value" bar is
 * therefore implementation freedom — but byte-for-byte agreement with the
 * Python writer is useful (fixture diffing, content-addressed caching), so this
 * module reproduces Python's `json.dumps(obj, separators=(",", ":"))` output
 * for every value whose spelling is determined by the *value* rather than by
 * the producing language's type system.
 *
 * Two classes of value have no language-neutral canonical spelling:
 *
 *  - **Float-valued integers.** Python prints `1.0` for a float and `1` for an
 *    int; JS has one number type and prints `1` for both. A `.colight` writer
 *    in JS cannot know which was intended. Use {@link pyFloat} to force the
 *    Python float spelling when byte-identity with a Python-authored payload
 *    matters.
 *  - **Non-finite numbers.** Python emits the bare tokens `NaN`, `Infinity`,
 *    `-Infinity`, which are not legal JSON; JS emits `null`. This encoder
 *    rejects them rather than picking a side.
 */

/**
 * Marks a number so it is written with Python's float spelling (`1.0`, not
 * `1`). Only affects numbers whose JS rendering has no decimal point or
 * exponent. Purely cosmetic: the parsed value is identical either way.
 */
export class PyFloat {
  constructor(public readonly value: number) {
    if (!Number.isFinite(value)) {
      throw new Error(`pyFloat requires a finite number, got ${value}.`);
    }
  }
}

/** See {@link PyFloat}. */
export function pyFloat(value: number): PyFloat {
  return new PyFloat(value);
}

/** Values accepted by {@link encodeJson}. */
export type JsonValue =
  | null
  | boolean
  | number
  | bigint
  | string
  | PyFloat
  | readonly JsonValue[]
  | { readonly [key: string]: JsonValue };

const ESCAPES: Record<string, string> = {
  '"': '\\"',
  "\\": "\\\\",
  "\n": "\\n",
  "\r": "\\r",
  "\t": "\\t",
  "\b": "\\b",
  "\f": "\\f",
};

/**
 * Escapes a string exactly as Python's `json.dumps` does with its default
 * `ensure_ascii=True`: every non-ASCII code point becomes a `\uXXXX` escape
 * (surrogate pairs for astral characters), and control characters below 0x20
 * use the short escapes where they exist.
 */
export function encodeJsonString(value: string): string {
  let out = '"';
  for (let i = 0; i < value.length; i++) {
    const char = value[i];
    const escape = ESCAPES[char];
    if (escape !== undefined) {
      out += escape;
      continue;
    }
    const code = value.charCodeAt(i);
    if (code < 0x20 || code > 0x7e) {
      out += "\\u" + code.toString(16).padStart(4, "0");
    } else {
      out += char;
    }
  }
  return out + '"';
}

/**
 * Renders a finite number the way Python's `repr`/`json.dumps` would.
 *
 * Python and JS agree on the shortest-round-trip digits, but differ in where
 * they switch to exponent notation and how they spell the exponent. Python
 * (C's `repr`) uses exponents at |x| >= 1e16 and at |x| < 1e-4, and pads the
 * exponent to two digits; JS uses them at |x| >= 1e21 and |x| < 1e-6, unpadded.
 */
export function encodeNumber(value: number, forceFloat: boolean): string {
  if (!Number.isFinite(value)) {
    throw new Error(
      `Cannot encode non-finite number ${value} in a .colight JSON section: ` +
        `NaN and Infinity are not valid JSON, and Python and JavaScript ` +
        `disagree on how to spell them. Replace it with null or a string.`,
    );
  }
  if (Number.isInteger(value) && !forceFloat && !Object.is(value, -0)) {
    // Integral JS numbers with no float marker are written as integers, which
    // is both what JSON.stringify does and what Python does for an int.
    if (Math.abs(value) < 1e21) return value.toFixed(0);
  }

  if (Number.isInteger(value) && Math.abs(value) < 1e16) {
    // Python's float repr for an integral float: 1.0, -0.0, 1000000000000000.0.
    // toFixed drops the sign of negative zero, so restore it explicitly.
    const sign = Object.is(value, -0) ? "-" : "";
    return `${sign}${value.toFixed(0)}.0`;
  }

  const magnitude = Math.abs(value);
  let text = String(value);
  if (!text.includes("e") && !text.includes("E")) {
    if (magnitude >= 1e16 || (magnitude !== 0 && magnitude < 1e-4)) {
      // JS prints these in positional form where Python uses an exponent.
      text = value.toExponential();
    } else {
      return text;
    }
  }
  // Normalize the exponent to Python's zero-padded two-digit form.
  return text.replace(
    /[eE]([+-]?)(\d+)/,
    (_m, sign: string, digits: string) =>
      `e${sign === "-" ? "-" : "+"}${digits.length < 2 ? "0" + digits : digits}`,
  );
}

function encodeValue(value: JsonValue, path: string): string {
  if (value === null) return "null";
  if (value instanceof PyFloat) return encodeNumber(value.value, true);

  switch (typeof value) {
    case "boolean":
      return value ? "true" : "false";
    case "number":
      return encodeNumber(value, false);
    case "bigint":
      return value.toString(10);
    case "string":
      return encodeJsonString(value);
    case "undefined":
      throw new Error(`Cannot encode undefined at ${path}.`);
  }

  if (Array.isArray(value)) {
    let out = "[";
    for (let i = 0; i < value.length; i++) {
      if (i > 0) out += ",";
      out += encodeValue(value[i], `${path}[${i}]`);
    }
    return out + "]";
  }

  const record = value as { readonly [key: string]: JsonValue };
  let out = "{";
  let first = true;
  for (const key of Object.keys(record)) {
    const entry = record[key];
    // Python's json.dumps has no notion of an absent value; JSON.stringify
    // drops undefined members. Follow JSON.stringify and drop them.
    if (entry === undefined) continue;
    if (!first) out += ",";
    first = false;
    out += encodeJsonString(key) + ":" + encodeValue(entry, `${path}.${key}`);
  }
  return out + "}";
}

/**
 * Encodes a JSON value to a compact UTF-8-safe ASCII string, matching Python's
 * `json.dumps(value, separators=(",", ":"))`.
 *
 * Object keys are emitted in insertion order, which is what both writers do.
 */
export function encodeJson(value: JsonValue): string {
  return encodeValue(value, "$");
}
