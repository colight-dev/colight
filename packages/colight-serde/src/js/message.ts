import { collectBuffers } from "./buffers";
import { evaluateNdarray } from "./binary";

export interface WireMessage<T = unknown> {
  message_id: string | number;
  buffer_count: number;
  payload: T;
}

function nextMessageId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function serialize(payload) {
  return collectBuffers(payload);
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

export function packMessage(payload, messageId = null) {
  const [serialized, buffers] = serialize(payload);
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
