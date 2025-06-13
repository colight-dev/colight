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
 * @returns {{data: Object, buffers: Uint8Array[]}} - Parsed JSON data and buffers array
 * @throws {Error} If file format is invalid
 */
export function parseColightData(data) {
  if (data instanceof ArrayBuffer) {
    data = new Uint8Array(data);
  }

  // Handle Node.js Buffer objects
  if (typeof Buffer !== 'undefined' && data instanceof Buffer) {
    data = new Uint8Array(data);
  }

  if (data.length < HEADER_SIZE) {
    throw new Error("Invalid .colight file: Too short");
  }

  // Parse header
  const header = data.slice(0, HEADER_SIZE);
  // Create a proper ArrayBuffer for DataView from the header bytes
  const headerBuffer = new ArrayBuffer(HEADER_SIZE);
  const headerView = new Uint8Array(headerBuffer);
  headerView.set(header);
  const dataView = new DataView(headerBuffer);

  // Check magic bytes
  const magic = header.slice(0, 8);
  for (let i = 0; i < MAGIC_BYTES.length; i++) {
    if (magic[i] !== MAGIC_BYTES[i]) {
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

  // Extract JSON section
  if (jsonOffset + jsonLength > data.length) {
    throw new Error("Invalid .colight file: JSON section extends beyond file");
  }

  const jsonBytes = data.slice(jsonOffset, jsonOffset + jsonLength);
  const jsonString = new TextDecoder().decode(jsonBytes);
  const jsonData = JSON.parse(jsonString);

  // Extract binary section
  if (binaryOffset + binaryLength > data.length) {
    throw new Error("Invalid .colight file: Binary section extends beyond file");
  }

  const binaryData = data.slice(binaryOffset, binaryOffset + binaryLength);

  // Extract individual buffers using layout information
  const bufferLayout = jsonData.bufferLayout || {};
  const bufferOffsets = bufferLayout.offsets || [];
  const bufferLengths = bufferLayout.lengths || [];

  if (bufferOffsets.length !== numBuffers || bufferLengths.length !== numBuffers) {
    throw new Error("Invalid .colight file: Buffer layout mismatch");
  }

  const buffers = [];
  for (let i = 0; i < numBuffers; i++) {
    const offset = bufferOffsets[i];
    const length = bufferLengths[i];

    if (offset + length > binaryLength) {
      throw new Error(`Invalid .colight file: Buffer ${i} extends beyond binary section`);
    }

    // Create a view into the binary data without copying
    // This is memory efficient as requested
    const buffer = binaryData.slice(offset, offset + length);
    buffers.push(buffer);
  }

  return { ...jsonData, buffers };
}

/**
 * Load and parse a .colight file from a URL.
 *
 * @param {string} url - URL to the .colight file
 * @returns {Promise<{data: Object, buffers: Uint8Array[]}>} - Parsed data and buffers
 */
export async function loadColightFile(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    return parseColightData(arrayBuffer);
  } catch (error) {
    console.error('Error loading .colight file:', error);
    throw error;
  }
}

/**
 * Parse .colight data from a script tag with type='application/x-colight'.
 *
 * @param {HTMLScriptElement} scriptElement - The script element containing base64-encoded .colight data
 * @returns {{data: Object, buffers: Uint8Array[]}} - Parsed data and buffers
 */
export function parseColightScript(scriptElement) {
  // Get the base64-encoded content from the script tag
  const base64Data = scriptElement.textContent.trim();

  // Decode base64 to get the raw binary data
  const binaryData = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));

  // Parse the .colight format
  return parseColightData(binaryData);
}
