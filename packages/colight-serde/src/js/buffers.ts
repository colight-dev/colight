import { inferDtype } from "./binary";

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
