import { collectBuffers } from "./buffers";
import { evaluateNdarray, TypedArray } from "./binary";
import type { NdArrayView } from "./ndarray";

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

export interface WireMessage<T = unknown> {
  message_id: string | number;
  buffer_count: number;
  payload: T;
}

function nextMessageId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function serialize(payload: SerializableValue): [JsonValue, BufferLike[]] {
  return collectBuffers(payload);
}

export function deserialize(payload: JsonValue, buffers: BufferLike[]): unknown {
  function traverse(value: JsonValue): unknown {
    if (value && typeof value === "object") {
      const obj = value as { [key: string]: JsonValue };
      if (obj.__type__ === "ndarray" && obj.__buffer_index__ !== undefined) {
        const data = buffers[obj.__buffer_index__ as number];
        return evaluateNdarray({ ...obj, data } as Parameters<typeof evaluateNdarray>[0]);
      }
      if (obj.__buffer_index__ !== undefined) {
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

  return traverse(payload);
}

export function packMessage(
  payload: SerializableValue,
  messageId: string | number | null = null,
): [WireMessage, BufferLike[]] {
  const [serialized, buffers] = serialize(payload);
  const envelope: WireMessage = {
    message_id: messageId ?? nextMessageId(),
    buffer_count: buffers.length,
    payload: serialized,
  };
  return [envelope, buffers];
}

export function unpackMessage(envelope: WireMessage, buffers: BufferLike[]): unknown {
  if (
    envelope &&
    envelope.buffer_count !== undefined &&
    envelope.buffer_count !== buffers.length
  ) {
    throw new Error(
      `buffer_count mismatch: expected ${envelope.buffer_count}, got ${buffers.length}`,
    );
  }
  return deserialize(envelope.payload as JsonValue, buffers);
}
