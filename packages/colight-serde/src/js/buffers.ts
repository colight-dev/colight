import { inferDtype } from "./binary";

export function collectBuffers(data) {
  const buffers = [];

  function traverse(value) {
    if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
      const index = buffers.length;
      buffers.push(value);

      const metadata = {
        __buffer_index__: index,
        __type__: "ndarray",
        dtype: inferDtype(value),
        order: "C",
      };

      if (value instanceof ArrayBuffer) {
        metadata.shape = [value.byteLength];
        metadata.strides = [1];
      } else {
        metadata.shape = [value.length];
        metadata.strides = [value.BYTES_PER_ELEMENT || 1];
      }

      return metadata;
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
