/**
 * The conformance fixture set, defined once and consumed by both directions of
 * the suite.
 *
 * Each fixture is described declaratively so the Python side (`conformance.py`,
 * driven over a subprocess) can build the byte-identical counterpart from the
 * same description. Adding a fixture here automatically adds it to every
 * conformance test; keep the descriptions in the small, portable vocabulary
 * that `conformance.py` understands.
 */

import {
  boolArray,
  ndarray,
  pyFloat,
  rawBuffer,
  type Payload,
} from "@colight/format";

/** How a fixture's array/buffer values are described portably. */
export type ValueSpec =
  | { kind: "ndarray"; dtype: string; shape: number[]; values: number[] }
  | { kind: "bool"; shape: number[]; values: boolean[] }
  | { kind: "raw"; bytes: number[] }
  | { kind: "int"; value: number }
  | { kind: "float"; value: number }
  | { kind: "str"; value: string }
  | { kind: "bool_scalar"; value: boolean }
  | { kind: "null" }
  | { kind: "list"; items: ValueSpec[] }
  | { kind: "object"; entries: [string, ValueSpec][] };

/** One conformance fixture: an optional initial entry plus update entries. */
export interface Fixture {
  name: string;
  /** `null` produces an "updates-only" file (spec §3.1). */
  initial: ValueSpec | null;
  updates: ValueSpec[];
  /**
   * Set when the fixture deliberately exercises a value whose JSON spelling is
   * not pinned by the spec, so byte-identity with Python is not asserted.
   * The string explains why, and is surfaced in the test name.
   */
  byteIdenticalCaveat?: string;
}

function obj(...entries: [string, ValueSpec][]): ValueSpec {
  return { kind: "object", entries };
}
function arr(dtype: string, shape: number[], values: number[]): ValueSpec {
  return { kind: "ndarray", dtype, shape, values };
}
function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

/** A state-only update envelope in the shape the readers expect (spec §3.6). */
function update(state: [string, ValueSpec][]): ValueSpec {
  return obj(["ast", { kind: "null" }], ["state", obj(...state)]);
}

export const FIXTURES: Fixture[] = [
  {
    name: "empty-state",
    initial: obj(["ast", { kind: "null" }], ["state", obj()]),
    updates: [],
  },
  {
    name: "scalars-only-no-buffers",
    initial: obj(
      ["ast", { kind: "null" }],
      [
        "state",
        obj(
          ["n", { kind: "int", value: 42 }],
          ["neg", { kind: "int", value: -7 }],
          ["s", { kind: "str", value: "hello" }],
          ["t", { kind: "bool_scalar", value: true }],
          ["f", { kind: "bool_scalar", value: false }],
          ["nil", { kind: "null" }],
          ["nested", obj(["deep", { kind: "list", items: [] }])],
        ),
      ],
    ),
    updates: [],
  },
  {
    name: "all-dtypes-1d",
    initial: obj(
      ["ast", { kind: "null" }],
      [
        "state",
        obj(
          ["i8", arr("int8", [3], [-128, 0, 127])],
          ["i16", arr("int16", [3], [-32768, 0, 32767])],
          ["i32", arr("int32", [3], [-2147483648, 0, 2147483647])],
          ["u8", arr("uint8", [3], [0, 128, 255])],
          ["u16", arr("uint16", [3], [0, 32768, 65535])],
          ["u32", arr("uint32", [3], [0, 2147483648, 4294967295])],
          ["f32", arr("float32", [4], [0, 1.5, -2.25, 1e10])],
          ["f64", arr("float64", [4], [0, 0.1, -1e-9, 1e300])],
          ["i64", arr("int64", [3], [-1, 0, 1])],
          ["u64", arr("uint64", [3], [0, 1, 4294967296])],
          [
            "b",
            { kind: "bool", shape: [4], values: [true, false, true, true] },
          ],
        ),
      ],
    ),
    updates: [],
  },
  {
    name: "shapes-1d-3d-and-empty",
    initial: obj(
      ["ast", { kind: "null" }],
      [
        "state",
        obj(
          ["oneD", arr("float32", [5], range(5))],
          ["threeD", arr("uint8", [2, 3, 4], range(24))],
          ["emptyDim", arr("float64", [0], [])],
          ["emptyInner", arr("int32", [3, 0], [])],
          ["tall", arr("int16", [6, 1], range(6))],
        ),
      ],
    ),
    updates: [],
  },
  {
    name: "raw-buffers-and-odd-lengths",
    initial: obj(
      ["ast", { kind: "null" }],
      [
        "state",
        obj(
          // Odd lengths force inter-buffer padding at every alignment residue.
          ["one", { kind: "raw", bytes: [1] }],
          ["three", { kind: "raw", bytes: [1, 2, 3] }],
          ["five", { kind: "raw", bytes: range(5) }],
          ["seven", { kind: "raw", bytes: range(7) }],
          ["eight", { kind: "raw", bytes: range(8) }],
          ["nine", { kind: "raw", bytes: range(9) }],
          ["mixed", arr("int8", [11], range(11))],
          ["afterOdd", arr("float64", [2], [1.5, 2.5])],
          ["zero", { kind: "raw", bytes: [] }],
          ["trailing", { kind: "raw", bytes: [255, 254, 253] }],
        ),
      ],
    ),
    updates: [],
  },
  {
    name: "several-updates-with-buffers",
    initial: obj(
      ["ast", { kind: "null" }],
      ["state", obj(["base", arr("float32", [3], [1, 2, 3])])],
    ),
    updates: [
      update([["frame", { kind: "int", value: 0 }]]),
      update([
        ["frame", { kind: "int", value: 1 }],
        ["points", arr("float64", [3], [0.5, 1.5, 2.5])],
      ]),
      // Buffer indices restart at 0 in every entry (spec §3.6).
      update([
        ["frame", { kind: "int", value: 2 }],
        ["a", { kind: "raw", bytes: [7, 7, 7] }],
        ["b", arr("uint16", [2, 2], [1, 2, 3, 4])],
      ]),
      update([["frame", { kind: "int", value: 3 }]]),
    ],
  },
  {
    name: "updates-only-file",
    initial: null,
    updates: [
      update([["theme", { kind: "str", value: "dark" }]]),
      update([["theme", { kind: "str", value: "light" }]]),
    ],
  },
  {
    name: "non-ascii-strings",
    initial: obj(
      ["ast", { kind: "null" }],
      [
        "state",
        obj(
          ["greeting", { kind: "str", value: "héllo ☃" }],
          ["kéy", { kind: "str", value: "vàlue" }],
          ["emoji", { kind: "str", value: "🎈 balloon" }],
          ["escapes", { kind: "str", value: 'quote " backslash \\ tab \t' }],
        ),
      ],
    ),
    updates: [],
  },
  {
    name: "float-spellings",
    initial: obj(
      ["ast", { kind: "null" }],
      [
        "state",
        obj(
          ["tenth", { kind: "float", value: 0.1 }],
          ["third", { kind: "float", value: 1 / 3 }],
          ["tiny", { kind: "float", value: 1e-7 }],
          ["small", { kind: "float", value: 1e-5 }],
          ["huge", { kind: "float", value: 1e300 }],
          ["e16", { kind: "float", value: 1e16 }],
          ["whole", { kind: "float", value: 1 }],
          ["denormal", { kind: "float", value: 5e-324 }],
        ),
      ],
    ),
    updates: [],
  },
];

/** Builds the JS-side payload for a {@link ValueSpec}. */
export function buildPayload(spec: ValueSpec): Payload {
  switch (spec.kind) {
    case "ndarray":
      return ndarray({
        dtype: spec.dtype,
        shape: spec.shape,
        data: spec.values,
      });
    case "bool":
      return boolArray(spec.values, spec.shape);
    case "raw":
      return rawBuffer(new Uint8Array(spec.bytes));
    case "int":
      return spec.value;
    case "float":
      // Python writes floats with a decimal point or exponent; mark the value
      // so the JS encoder spells integral floats the same way (see json.ts).
      return pyFloat(spec.value);
    case "str":
      return spec.value;
    case "bool_scalar":
      return spec.value;
    case "null":
      return null;
    case "list":
      return spec.items.map(buildPayload);
    case "object": {
      const out: Record<string, Payload> = {};
      for (const [key, value] of spec.entries) out[key] = buildPayload(value);
      return out;
    }
  }
}

/**
 * The plain JSON value a fixture spec denotes, with buffer-carrying leaves
 * replaced by a portable `{dtype, shape, values}` / `{bytes}` description.
 * Used to assert what a reader produced without re-deriving it from the writer.
 */
export type ExpectedValue =
  | null
  | boolean
  | number
  | string
  | { __array__: { dtype: string; shape: number[]; values: number[] } }
  | { __raw__: number[] }
  | ExpectedValue[]
  | { [key: string]: ExpectedValue };

/** Builds the expected decoded value for a {@link ValueSpec}. */
export function expectedValue(spec: ValueSpec): ExpectedValue {
  switch (spec.kind) {
    case "ndarray":
      return {
        __array__: {
          dtype: spec.dtype,
          shape: spec.shape,
          values: spec.values,
        },
      };
    case "bool":
      return {
        __array__: {
          dtype: "bool",
          shape: spec.shape,
          values: spec.values.map((v) => (v ? 1 : 0)),
        },
      };
    case "raw":
      return { __raw__: spec.bytes };
    case "int":
    case "float":
      return spec.value;
    case "str":
      return spec.value;
    case "bool_scalar":
      return spec.value;
    case "null":
      return null;
    case "list":
      return spec.items.map(expectedValue);
    case "object": {
      const out: Record<string, ExpectedValue> = {};
      for (const [key, value] of spec.entries) out[key] = expectedValue(value);
      return out;
    }
  }
}
