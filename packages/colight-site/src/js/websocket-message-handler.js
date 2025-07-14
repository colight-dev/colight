// WebSocket message processing logic extracted for testability

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
  };

  // Handle the new manifest-based update
  if (message.blocks && message.dirty) {
    const newResults = {};
    const blockSet = new Set(message.blocks);
    const dirtySet = new Set(message.dirty);

    // Process each block in the manifest
    for (const blockId of message.blocks) {
      if (currentBlockResults[blockId]) {
        // Existing block
        if (dirtySet.has(blockId)) {
          // Mark as pending (will be re-executed)
          newResults[blockId] = {
            ...currentBlockResults[blockId],
            pending: true,
            cache_hit: false,
          };
        } else {
          // Keep unchanged, but clear any pending state
          newResults[blockId] = {
            ...currentBlockResults[blockId],
            pending: false,
          };
        }
      } else {
        // New block - create placeholder
        // Always mark as pending since we don't have data for it yet
        // The server will send either full data or "unchanged"
        newResults[blockId] = {
          pending: true,
          elements: [],
          ok: true,
        };
      }
    }

    result.blockResults = newResults;
  } else {
    // Legacy behavior - don't include blockResults in the result
    // This will prevent clearing existing blocks unnecessarily
    // result.blockResults = {};
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

  // Handle unchanged blocks (lightweight message)
  if (message.unchanged) {
    return {
      blockId: message.block,
      blockResult: {
        ...currentBlockResults[message.block],
        pending: false,
      },
      changed: false,
    };
  }

  // Track changed blocks
  const changed = message.content_changed || false;
  if (changed) {
    changedBlocks.add(message.block);
  }

  // Update block result with new data
  return {
    blockId: message.block,
    blockResult: {
      ok: message.ok,
      stdout: message.stdout,
      error: message.error,
      showsVisual: message.showsVisual,
      elements: message.elements || [],
      cache_hit: message.cache_hit,
      content_changed: message.content_changed,
      pending: false,
    },
    changed,
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

    default:
      return { type: "unknown", messageType: message.type };
  }

  return { type: "no-op" };
}
