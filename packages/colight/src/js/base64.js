/**
 * Base64 encoding and decoding utilities for binary buffers.
 *
 * These functions handle conversion between binary data (ArrayBuffer, TypedArrays)
 * and base64 strings for transport over JSON/WebSocket protocols.
 */

/**
 * Encode a binary buffer to base64 string.
 *
 * Uses chunked encoding to avoid call stack size limits with large buffers.
 *
 * @param {ArrayBuffer | ArrayBufferView} buffer - Binary data to encode
 * @returns {string} Base64 encoded string
 */
export function encodeBufferToBase64(buffer) {
  const view = ArrayBuffer.isView(buffer)
    ? new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
    : new Uint8Array(buffer);

  let binary = "";
  const chunkSize = 0x8000; // 32KB chunks to avoid stack overflow

  for (let i = 0; i < view.length; i += chunkSize) {
    binary += String.fromCharCode(...view.subarray(i, i + chunkSize));
  }

  return btoa(binary);
}

/**
 * Decode a base64 string to Uint8Array.
 *
 * @param {string} base64 - Base64 encoded string
 * @returns {Uint8Array} Decoded binary data
 */
export function decodeBase64ToUint8Array(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  return bytes;
}
