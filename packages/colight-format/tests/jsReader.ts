/**
 * Adapts Colight's existing JavaScript reader (`packages/colight/src/js/format.js`)
 * to the portable decoded shape the conformance fixtures describe.
 *
 * The reader is imported and called as a black box: this file only depends on
 * its documented output — the initial entry's JSON spread onto the result, plus
 * `buffers` and `updateEntries` — never on how it produces that.
 */

import { parseColightData } from "../../colight/src/js/format.js";

import { DTYPE_BYTES, type Dtype, assertDtype } from "@colight/format";
import type { ExpectedValue } from "./fixtures.js";

/** What the JS reader saw in a file, in the fixtures' portable shape. */
export interface JsRead {
  initial: ExpectedValue | null;
  updates: ExpectedValue[];
}

type Reader = (data: Uint8Array) => Record<string, unknown>;

/** Parses `bytes` with the existing JS reader and decodes buffer references. */
export function jsReadFile(bytes: Uint8Array): JsRead {
  const parsed = (parseColightData as Reader)(bytes);
  const updateEntries = (parsed.updateEntries ?? []) as {
    data: unknown;
    buffers: ArrayBufferView[];
  }[];

  const hasInitial = Object.prototype.hasOwnProperty.call(parsed, "buffers");
  let initial: ExpectedValue | null = null;
  if (hasInitial) {
    const {
      buffers,
      updateEntries: _drop,
      // `bufferLayout` is container metadata that the writer derives, not part
      // of the payload the fixtures describe. It is checked structurally in
      // writer.test.ts instead.
      bufferLayout: _layout,
      ...rest
    } = parsed as Record<string, unknown> & { buffers: ArrayBufferView[] };
    initial = decode(rest, buffers);
  }

  return {
    initial,
    updates: updateEntries.map((entry) =>
      stripLayout(decode(entry.data, entry.buffers)),
    ),
  };
}

/** Drops a decoded entry's `bufferLayout`, see above. */
function stripLayout(value: ExpectedValue): ExpectedValue {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return value;
  }
  const { bufferLayout: _drop, ...rest } = value as Record<
    string,
    ExpectedValue
  >;
  return rest;
}

function decode(
  value: unknown,
  buffers: readonly ArrayBufferView[],
): ExpectedValue {
  if (value === null || typeof value !== "object") {
    return value as ExpectedValue;
  }
  if (Array.isArray(value)) {
    return value.map((item) => decode(item, buffers));
  }

  const record = value as Record<string, unknown>;
  if (record.__type__ === "ndarray") {
    const dtype = assertDtype(String(record.dtype));
    const shape = (record.shape as number[]).map(Number);
    const view = buffers[record.__buffer_index__ as number];
    return {
      __array__: { dtype, shape, values: readElements(view, dtype) },
    };
  }
  const keys = Object.keys(record);
  if (keys.length === 1 && keys[0] === "__buffer_index__") {
    const view = buffers[record.__buffer_index__ as number];
    return { __raw__: Array.from(asBytes(view)) };
  }

  const out: Record<string, ExpectedValue> = {};
  for (const key of keys) out[key] = decode(record[key], buffers);
  return out;
}

function asBytes(view: ArrayBufferView): Uint8Array {
  return new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

/** Reads a buffer's elements as plain numbers, little-endian per spec §3.3. */
function readElements(view: ArrayBufferView, dtype: Dtype): number[] {
  const data = new DataView(view.buffer, view.byteOffset, view.byteLength);
  const size = DTYPE_BYTES[dtype];
  const count = view.byteLength / size;
  const values: number[] = [];
  for (let i = 0; i < count; i++) {
    const at = i * size;
    switch (dtype) {
      case "int8":
        values.push(data.getInt8(at));
        break;
      case "uint8":
        values.push(data.getUint8(at));
        break;
      case "bool":
        values.push(data.getUint8(at));
        break;
      case "int16":
        values.push(data.getInt16(at, true));
        break;
      case "uint16":
        values.push(data.getUint16(at, true));
        break;
      case "int32":
        values.push(data.getInt32(at, true));
        break;
      case "uint32":
        values.push(data.getUint32(at, true));
        break;
      case "float32":
        values.push(data.getFloat32(at, true));
        break;
      case "float64":
        values.push(data.getFloat64(at, true));
        break;
      case "int64":
        values.push(Number(data.getBigInt64(at, true)));
        break;
      case "uint64":
        values.push(Number(data.getBigUint64(at, true)));
        break;
    }
  }
  return values;
}
