/**
 * Colight file format reader for JavaScript.
 *
 * Parses .colight files with the new binary format and provides
 * buffers as an array indexed by the existing buffer index system.
 */

// File format constants
const MAGIC_BYTES = new TextEncoder().encode("COLIGHT\0");
const HEADER_SIZE = 96;

/**
 * Parse a .colight file from ArrayBuffer or Uint8Array.
 *
 * @param {ArrayBuffer|Uint8Array} data - The .colight file content
 * @returns {{...jsonData, buffers: DataView[]}} - Parsed JSON data spread with buffers array
 * @throws {Error} If file format is invalid
 */
export function parseColightData(data) {
  if (data instanceof ArrayBuffer) {
    data = new Uint8Array(data);
  }

  // Handle Node.js Buffer objects
  if (typeof Buffer !== "undefined" && data instanceof Buffer) {
    data = new Uint8Array(data);
  }

  if (data.length < HEADER_SIZE) {
    throw new Error("Invalid .colight file: Too short");
  }

  // Parse header using DataView directly on data.buffer with offsets
  const dataView = new DataView(data.buffer, data.byteOffset, HEADER_SIZE);

  // Check magic bytes
  for (let i = 0; i < MAGIC_BYTES.length; i++) {
    if (data[i] !== MAGIC_BYTES[i]) {
      throw new Error(`Invalid .colight file: Wrong magic bytes`);
    }
  }

  // Parse header fields (little-endian)
  const version = dataView.getBigUint64(8, true);
  const jsonOffset = Number(dataView.getBigUint64(16, true));
  const jsonLength = Number(dataView.getBigUint64(24, true));
  const binaryOffset = Number(dataView.getBigUint64(32, true));
  const binaryLength = Number(dataView.getBigUint64(40, true));
  const numBuffers = Number(dataView.getBigUint64(48, true));

  if (version > 1n) {
    throw new Error(`Unsupported .colight file version: ${version}`);
  }

  // Robustness checks - binary section should be after JSON section (with possible padding)
  if (binaryOffset < jsonOffset + jsonLength) {
    throw new Error(
      `Invalid .colight file: Binary section overlaps with JSON section`,
    );
  }

  // Check that reserved bytes (56-95) are zero
  for (let i = 56; i < HEADER_SIZE; i++) {
    if (data[i] !== 0) {
      throw new Error(
        `Invalid .colight file: Reserved byte at position ${i} is not zero`,
      );
    }
  }

  // Extract JSON section
  if (jsonOffset + jsonLength > data.length) {
    throw new Error("Invalid .colight file: JSON section extends beyond file");
  }

  const jsonBytes = data.subarray(jsonOffset, jsonOffset + jsonLength);
  const jsonString = new TextDecoder().decode(jsonBytes);
  const jsonData = JSON.parse(jsonString);

  // Extract binary section
  if (binaryOffset + binaryLength > data.length) {
    throw new Error(
      "Invalid .colight file: Binary section extends beyond file",
    );
  }

  const binaryData = data.subarray(binaryOffset, binaryOffset + binaryLength);

  // Extract individual buffers using layout information
  const bufferLayout = jsonData.bufferLayout || {};
  const bufferOffsets = bufferLayout.offsets || [];
  const bufferLengths = bufferLayout.lengths || [];

  if (
    bufferOffsets.length !== numBuffers ||
    bufferLengths.length !== numBuffers
  ) {
    throw new Error("Invalid .colight file: Buffer layout mismatch");
  }

  const buffers = [];
  for (let i = 0; i < numBuffers; i++) {
    const offset = bufferOffsets[i];
    const length = bufferLengths[i];

    if (offset + length > binaryLength) {
      throw new Error(
        `Invalid .colight file: Buffer ${i} extends beyond binary section`,
      );
    }

    // Create a DataView into the binary data without copying
    // This is memory efficient as requested
    const buffer = new DataView(
      binaryData.buffer,
      binaryData.byteOffset + offset,
      length,
    );
    buffers.push(buffer);
  }

  return { ...jsonData, buffers };
}

/**
 * Load and parse a .colight file from a URL.
 *
 * @param {string} url - URL to the .colight file
 * @returns {Promise<{...jsonData, buffers: DataView[]}>} - Parsed data and buffers
 */
export async function loadColightFile(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to fetch ${url}: ${response.status} ${response.statusText}`,
      );
    }

    const arrayBuffer = await response.arrayBuffer();
    return parseColightData(arrayBuffer);
  } catch (error) {
    console.error("Error loading .colight file:", error);
    throw error;
  }
}

/**
 * Parse .colight data from a script tag with type='application/x-colight'.
 *
 * @param {HTMLScriptElement} scriptElement - The script element containing base64-encoded .colight data
 * @returns {{...jsonData, buffers: DataView[]}} - Parsed data and buffers
 */
export function parseColightScript(scriptElement) {
  // Get the base64-encoded content from the script tag
  const base64Data = scriptElement.textContent.trim();

  // Decode base64 to get the raw binary data
  const binaryData = Uint8Array.from(atob(base64Data), (c) => c.charCodeAt(0));

  // Parse the .colight format
  return parseColightData(binaryData);
}
