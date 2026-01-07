const dtypeMap = {
  float32: Float32Array,
  float64: Float64Array,
  int8: Int8Array,
  int16: Int16Array,
  int32: Int32Array,
  int64: BigInt64Array,
  uint8: Uint8Array,
  uint16: Uint16Array,
  uint32: Uint32Array,
  uint64: BigUint64Array,
  uint8clamped: Uint8ClampedArray,
  bigint64: BigInt64Array,
  biguint64: BigUint64Array,
};

function computeElementStrides(dims, order) {
  const strides = new Array(dims.length);
  if (order === "F") {
    let acc = 1;
    for (let i = 0; i < dims.length; i += 1) {
      strides[i] = acc;
      acc *= dims[i];
    }
    return strides;
  }
  let acc = 1;
  for (let i = dims.length - 1; i >= 0; i -= 1) {
    strides[i] = acc;
    acc *= dims[i];
  }
  return strides;
}

function sliceWithStride(flat, offset, length, stride) {
  if (stride === 1 && typeof flat.slice === "function") {
    return flat.slice(offset, offset + length);
  }
  const out = ArrayBuffer.isView(flat)
    ? new flat.constructor(length)
    : new Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = flat[offset + i * stride];
  }
  return out;
}

function reshapeArray(flat, dims, offset = 0, strides = null) {
  const [dim, ...restDims] = dims;
  const stride = strides ? strides[0] : null;

  if (restDims.length === 0) {
    if (stride === null) {
      return flat.slice(offset, offset + dim);
    }
    return sliceWithStride(flat, offset, dim, stride);
  }

  const nextStride = stride ?? restDims.reduce((a, b) => a * b, 1);
  const nextStrides = strides ? strides.slice(1) : null;
  return Array.from({ length: dim }, (_, i) =>
    reshapeArray(flat, restDims, offset + i * nextStride, nextStrides),
  );
}

function evaluateNdarray(node) {
  const { data, dtype, shape } = node;
  const order = node.order || "C";
  const wireStrides = node.strides || null;
  const ArrayConstructor = dtypeMap[dtype] || Float64Array;
  const bytesPerElement = ArrayConstructor.BYTES_PER_ELEMENT;
  const view =
    data instanceof DataView
      ? data
      : ArrayBuffer.isView(data)
        ? new DataView(data.buffer, data.byteOffset, data.byteLength)
        : new DataView(data);
  const flatArray = new ArrayConstructor(
    view.buffer,
    view.byteOffset,
    view.byteLength / bytesPerElement,
  );
  const elementStrides = wireStrides
    ? wireStrides.map((stride) => stride / bytesPerElement)
    : computeElementStrides(shape, order);
  if (shape.length <= 1) {
    return flatArray;
  }
  return reshapeArray(flatArray, shape, 0, elementStrides);
}

function inferDtype(value) {
  if (!(value instanceof ArrayBuffer || ArrayBuffer.isView(value))) {
    throw new Error("Value must be a TypedArray");
  }
  return value.constructor.name.toLowerCase().replace("array", "");
}

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

export function deserialize(payload, buffers) {
  function traverse(value) {
    if (value && typeof value === "object") {
      if (value.__type__ === "ndarray" && value.__buffer_index__ !== undefined) {
        const data = buffers[value.__buffer_index__];
        return evaluateNdarray({ ...value, data });
      }
      if (value.__buffer_index__ !== undefined) {
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

  return traverse(payload);
}

function nextMessageId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function packMessage(payload, messageId = null) {
  const [serialized, buffers] = collectBuffers(payload);
  const envelope = {
    message_id: messageId ?? nextMessageId(),
    buffer_count: buffers.length,
    payload: serialized,
  };
  return [envelope, buffers];
}

export function unpackMessage(envelope, buffers) {
  if (
    envelope &&
    envelope.buffer_count !== undefined &&
    envelope.buffer_count !== buffers.length
  ) {
    throw new Error(
      `buffer_count mismatch: expected ${envelope.buffer_count}, got ${buffers.length}`,
    );
  }
  return deserialize(envelope.payload, buffers);
}
