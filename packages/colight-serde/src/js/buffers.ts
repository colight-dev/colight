import { inferDtype, TypedArray } from "./binary";
import { isNdArray, NdArrayView, TypedArrayLike } from "./ndarray";

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

type SerializableValue =
  | JsonValue
  | ArrayBuffer
  | ArrayBufferView
  | NdArrayView
  | TypedArray
  | SerializableValue[]
  | { [key: string]: SerializableValue };

type BufferLike = ArrayBuffer | ArrayBufferView;

/**
 * Serialized ndarray reference - what gets sent over the wire.
 */
export interface SerializedNdArray {
  __type__: "ndarray";
  __buffer_index__: number;
  dtype: string;
  shape: number[];
  strides: number[];
  order: "C" | "F";
}

/**
 * Serialize an NdArrayView or TypedArray to wire format.
 *
 * Accepts either:
 * - NdArrayView: uses .flat and .shape for full shape info
 * - TypedArray: treated as 1D array
 *
 * Returns [reference, buffer] where reference contains metadata
 * and buffer is the raw ArrayBuffer to be sent out-of-band.
 *
 * Example:
 *   const [ref, buffer] = serializeNdArray(myView);
 *   buffers.push(buffer);
 *   ref.__buffer_index__ = buffers.length - 1;
 */
export function serializeNdArray(
  value: NdArrayView | TypedArray,
): [Omit<SerializedNdArray, "__buffer_index__">, ArrayBuffer] {
  if (isNdArray(value)) {
    // NdArrayView - full shape info available
    const flat = value.flat as TypedArray;
    const dtype = inferDtype(flat);
    const bytesPerElement = (flat as Uint8Array).BYTES_PER_ELEMENT ?? 1;
    return [
      {
        __type__: "ndarray",
        dtype,
        shape: [...value.shape],
        strides: value.strides.map((s) => s * bytesPerElement),
        order: "C",
      },
      flat.buffer as ArrayBuffer,
    ];
  }

  // TypedArray - treat as 1D
  const dtype = inferDtype(value);
  const bytesPerElement = (value as Uint8Array).BYTES_PER_ELEMENT ?? 1;
  return [
    {
      __type__: "ndarray",
      dtype,
      shape: [value.length],
      strides: [bytesPerElement],
      order: "C",
    },
    value.buffer as ArrayBuffer,
  ];
}

/**
 * Recursively traverse data structure, extracting binary buffers.
 *
 * - ArrayBuffer: treated as raw bytes (no __type__)
 * - TypedArray (Float32Array, etc.): treated as ndarray with dtype/shape
 *
 * Returns [serializedData, buffers] where serializedData has buffer references.
 */
export function collectBuffers(
  data: SerializableValue,
): [JsonValue, BufferLike[]] {
  const buffers: BufferLike[] = [];

  function traverse(value: SerializableValue): JsonValue {
    // Raw ArrayBuffer - store as plain buffer reference (no type)
    if (value instanceof ArrayBuffer) {
      const index = buffers.length;
      buffers.push(value);
      return { __buffer_index__: index };
    }

    // TypedArray - store as ndarray with dtype/shape metadata
    if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
      const index = buffers.length;
      buffers.push(value);
      const typedArray = value as TypedArrayLike;
      return {
        __buffer_index__: index,
        __type__: "ndarray",
        dtype: inferDtype(value),
        shape: [typedArray.length],
        strides: [(typedArray as Uint8Array).BYTES_PER_ELEMENT ?? 1],
        order: "C",
      };
    }

    // DataView - treat as raw bytes
    if (value instanceof DataView) {
      const index = buffers.length;
      buffers.push(value);
      return { __buffer_index__: index };
    }

    if (Array.isArray(value)) {
      return value.map(traverse);
    }

    if (value && typeof value === "object") {
      const result: { [key: string]: JsonValue } = {};
      for (const [key, val] of Object.entries(value)) {
        result[key] = traverse(val as SerializableValue);
      }
      return result;
    }

    return value as JsonValue;
  }

  return [traverse(data), buffers];
}

/**
 * Replace buffer references with actual buffer data.
 * Used internally during deserialization before evaluateNdarray.
 */
export function replaceBuffers(
  data: JsonValue,
  buffers: BufferLike[],
): unknown {
  function traverse(value: JsonValue): unknown {
    if (value && typeof value === "object") {
      const obj = value as { [key: string]: JsonValue };
      if (obj.__type__ === "ndarray" && obj.__buffer_index__ !== undefined) {
        const result = {
          ...obj,
          data: buffers[obj.__buffer_index__ as number],
        };
        delete (result as { __buffer_index__?: unknown }).__buffer_index__;
        return result;
      }
      // Raw buffer reference (no __type__)
      if (obj.__buffer_index__ !== undefined && obj.__type__ === undefined) {
        return buffers[obj.__buffer_index__ as number];
      }
      if (Array.isArray(value)) {
        return value.map(traverse);
      }
      const result: { [key: string]: unknown } = {};
      for (const [key, val] of Object.entries(obj)) {
        result[key] = traverse(val);
      }
      return result;
    }
    return value;
  }
  return traverse(data);
}
