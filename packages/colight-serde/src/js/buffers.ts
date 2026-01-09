import { inferDtype, TypedArray } from "./binary";
import { isNdArray, NdArrayView } from "./ndarray";

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
      flat.buffer,
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
    value.buffer,
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
export function collectBuffers(data) {
  const buffers = [];

  function traverse(value) {
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
      return {
        __buffer_index__: index,
        __type__: "ndarray",
        dtype: inferDtype(value),
        shape: [value.length],
        strides: [value.BYTES_PER_ELEMENT],
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
      const result = {};
      for (const [key, val] of Object.entries(value)) {
        result[key] = traverse(val);
      }
      return result;
    }

    return value;
  }

  return [traverse(data), buffers];
}

/**
 * Replace buffer references with actual buffer data.
 * Used internally during deserialization before evaluateNdarray.
 */
export function replaceBuffers(data, buffers) {
  function traverse(value) {
    if (value && typeof value === "object") {
      if (
        value.__type__ === "ndarray" &&
        value.__buffer_index__ !== undefined
      ) {
        value.data = buffers[value.__buffer_index__];
        delete value.__buffer_index__;
        return value;
      }
      // Raw buffer reference (no __type__)
      if (
        value.__buffer_index__ !== undefined &&
        value.__type__ === undefined
      ) {
        return buffers[value.__buffer_index__];
      }
      if (Array.isArray(value)) {
        return value.map(traverse);
      }
      const result = {};
      for (const [key, val] of Object.entries(value)) {
        result[key] = traverse(val);
      }
      return result;
    }
    return value;
  }
  return traverse(data);
}
