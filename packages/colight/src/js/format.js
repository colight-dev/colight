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
 * Parse a single entry from the .colight data.
 *
 * @param {Uint8Array} data - The .colight file content
 * @param {number} offset - Offset to start parsing from
 * @returns {{jsonData: any, buffers: DataView[], entrySize: number}} - Parsed entry
 * @throws {Error} If file format is invalid
 */
function parseEntry(data, offset) {
  if (offset + HEADER_SIZE > data.length) {
    throw new Error("Invalid .colight file: Entry header extends beyond file");
  }

  // Parse header using DataView
  const headerView = new DataView(
    data.buffer,
    data.byteOffset + offset,
    HEADER_SIZE,
  );

  // Check magic bytes
  for (let i = 0; i < MAGIC_BYTES.length; i++) {
    if (data[offset + i] !== MAGIC_BYTES[i]) {
      throw new Error(`Invalid .colight file: Wrong magic bytes`);
    }
  }

  // Parse header fields (little-endian)
  const version = headerView.getBigUint64(8, true);
  const jsonOffset = Number(headerView.getBigUint64(16, true));
  const jsonLength = Number(headerView.getBigUint64(24, true));
  const binaryOffset = Number(headerView.getBigUint64(32, true));
  const binaryLength = Number(headerView.getBigUint64(40, true));
  const numBuffers = Number(headerView.getBigUint64(48, true));

  if (version > 1n) {
    throw new Error(`Unsupported .colight file version: ${version}`);
  }

  // Extract JSON section
  const absoluteJsonOffset = offset + jsonOffset;
  if (absoluteJsonOffset + jsonLength > data.length) {
    throw new Error("Invalid .colight file: JSON section extends beyond file");
  }

  const jsonBytes = data.subarray(
    absoluteJsonOffset,
    absoluteJsonOffset + jsonLength,
  );
  const jsonString = new TextDecoder().decode(jsonBytes);
  const jsonData = JSON.parse(jsonString);

  // Extract buffers if present
  const buffers = [];
  if (binaryLength > 0) {
    const absoluteBinaryOffset = offset + binaryOffset;
    if (absoluteBinaryOffset + binaryLength > data.length) {
      throw new Error(
        "Invalid .colight file: Binary section extends beyond file",
      );
    }

    const binaryData = data.subarray(
      absoluteBinaryOffset,
      absoluteBinaryOffset + binaryLength,
    );

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

    for (let i = 0; i < numBuffers; i++) {
      const bufOffset = bufferOffsets[i];
      const bufLength = bufferLengths[i];

      if (bufOffset + bufLength > binaryLength) {
        throw new Error(
          `Invalid .colight file: Buffer ${i} extends beyond binary section`,
        );
      }

      // Create a DataView into the binary data without copying
      const buffer = new DataView(
        binaryData.buffer,
        binaryData.byteOffset + bufOffset,
        bufLength,
      );
      buffers.push(buffer);
    }
  }

  // Calculate total entry size
  const entrySize = binaryOffset + binaryLength;

  return {
    jsonData,
    buffers,
    entrySize,
  };
}

/**
 * Parse a .colight file from ArrayBuffer or Uint8Array.
 *
 * @param {ArrayBuffer|Uint8Array} data - The .colight file content
 * @returns {{...initialData, buffers: DataView[], updates?: Array}} - Parsed data with optional updates
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

  let initialData = null;
  let initialBuffers = [];
  const updateEntries = [];

  // Parse all entries until EOF
  let currentOffset = 0;
  let firstEntry = true;

  while (currentOffset < data.length) {
    // Check if there's enough space for another header
    if (currentOffset + HEADER_SIZE > data.length) {
      break;
    }

    try {
      const entry = parseEntry(data, currentOffset);

      if (firstEntry && !entry.jsonData.updates) {
        // First entry without updates is the initial state
        initialData = entry.jsonData;
        initialBuffers = entry.buffers;
      } else if (entry.jsonData.updates) {
        // Entry with updates field is an update entry
        // Store the complete update data including buffers
        updateEntries.push({
          data: entry.jsonData.updates,
          buffers: entry.buffers,
        });
      }

      firstEntry = false;
      currentOffset += entry.entrySize;
    } catch (e) {
      // Re-throw specific errors on first entry
      if (firstEntry && e.message.includes("Wrong magic bytes")) {
        throw e;
      }
      // Otherwise, we've reached the end
      break;
    }
  }

  // Return the appropriate structure
  if (initialData) {
    // Full file with initial state
    const result = { ...initialData, buffers: initialBuffers };
    if (updateEntries.length > 0) {
      result.updateEntries = updateEntries;
    }
    return result;
  } else if (updateEntries.length > 0) {
    // Update-only file
    return { updateEntries };
  } else {
    throw new Error("Invalid .colight file: No entries found");
  }
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
