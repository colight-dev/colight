// WebSocket message processing logic extracted for testability
import createLogger from "./logger.js";
const logger = createLogger("websocket-message-handler");
/**
 * Process a run-start message
 * @param {Object} message - The WebSocket message
 * @param {number} latestRun - The latest run version
 * @param {Object} currentBlockResults - Current block results state
 * @param {string} currentFile - The currently viewed file (optional)
 * @returns {Object|null} Updated state or null if no update needed
 */
export function processRunStart(
  message,
  latestRun,
  currentBlockResults,
  currentFile,
) {
  if (!message.run || message.run < latestRun) {
    return null;
  }

  const file = message.file;

  // Always process run-start messages - navigation decisions are made by the client
  // based on pinning state

  const result = {
    latestRun: message.run,
    currentFile: file,
    changedBlocks: new Set(),
    block_ids: message.block_ids || [], // Store the block IDs list
  };

  // Handle the new cache-key-based system
  if (message.block_ids) {
    const newResults = {};

    // Simple logic:
    // - If we have the block ID already, keep that data
    // - If we don't have it, create empty placeholder

    for (let i = 0; i < message.block_ids.length; i++) {
      const blockId = message.block_ids[i];
      if (currentBlockResults[blockId]) {
        // Same block ID = unchanged content, just update ordinal
        newResults[blockId] = {
          ...currentBlockResults[blockId],
          ordinal: i,
        };
      } else {
        // New block ID = content changed, wait for data
        newResults[blockId] = {
          elements: [],
          ok: true,
          ordinal: i,
        };
      }
    }

    result.blockResults = newResults;
  }

  return result;
}

/**
 * Process a block-result message
 * @param {Object} message - The WebSocket message
 * @param {number} latestRun - The latest run version
 * @param {Object} currentBlockResults - Current block results state
 * @param {Set} changedBlocks - Set of changed block IDs
 * @returns {Object|null} Updated block result or null if no update needed
 */
export function processBlockResult(
  message,
  latestRun,
  currentBlockResults,
  changedBlocks,
) {
  if (!message.run || message.run !== latestRun) {
    return null;
  }

  // The backend sends either:
  // 1. unchanged: true - block had same ID, so cache hit
  // 2. Full data - block had new ID or was forced to send full data

  if (message.unchanged) {
    // Backend says unchanged, which means the block ID hasn't changed
    // We should already have the data from a previous run
    if (!currentBlockResults[message.block]) {
      // This shouldn't happen - unchanged means we've seen this ID before
      console.warn(`Got unchanged for unknown block ${message.block}`);
      return null;
    }

    return {
      blockId: message.block,
      blockResult: currentBlockResults[message.block],
      changed: false,
    };
  }

  // Full data - either new block or forced full data
  const isNewBlock = !currentBlockResults[message.block];
  if (isNewBlock) {
    changedBlocks.add(message.block);
  }

  return {
    blockId: message.block,
    blockResult: {
      ok: message.ok,
      stdout: message.stdout,
      error: message.error,
      showsVisual: message.showsVisual,
      elements: message.elements || [],
      cache_hit: message.cache_hit || false,
      ordinal: message.ordinal,
    },
    changed: isNewBlock,
  };
}

/**
 * Process a run-end message
 * @param {Object} message - The WebSocket message
 * @param {number} latestRun - The latest run version
 * @param {Set} changedBlocks - Set of changed block IDs
 * @returns {Object|null} Information about the completed run
 */
export function processRunEnd(message, latestRun, changedBlocks) {
  if (!message.run || message.run !== latestRun) {
    return null;
  }

  return {
    run: message.run,
    error: message.error,
    changedBlocks: Array.from(changedBlocks),
  };
}

/**
 * Main message processor
 * @param {Object} message - The WebSocket message
 * @param {Object} state - Current state
 * @returns {Object} Action to take based on the message
 */
export function processWebSocketMessage(message, state) {
  const { latestRun, blockResults, changedBlocks, currentFile } = state;
  switch (message.type) {
    case "run-start": {
      const result = processRunStart(
        message,
        latestRun,
        blockResults,
        currentFile,
      );
      if (result) {
        return {
          type: "run-start",
          ...result,
        };
      }
      break;
    }

    case "block-result": {
      const result = processBlockResult(
        message,
        latestRun,
        blockResults,
        changedBlocks,
      );
      if (result) {
        return {
          type: "block-result",
          ...result,
        };
      }
      break;
    }

    case "run-end": {
      const result = processRunEnd(message, latestRun, changedBlocks);
      if (result) {
        return {
          type: "run-end",
          ...result,
        };
      }
      break;
    }

    case "reload":
      return { type: "reload" };

    case "file-changed":
      return {
        type: "file-changed",
        path: message.path,
        watched: message.watched,
      };

    case "directory-changed":
      return { type: "directory-changed" };

    default:
      return { type: "unknown", messageType: message.type };
  }

  return { type: "no-op" };
}
