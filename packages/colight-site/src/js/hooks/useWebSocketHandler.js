import { useCallback, useRef, useEffect } from "react";
import { processWebSocketMessage } from "../websocket-message-handler.js";
import createLogger from "../logger.js";

const logger = createLogger("ws-handler");

/**
 * Hook for handling WebSocket messages with proper action dispatching
 */
export function useWebSocketHandler({
  blockManager,
  navState,
  navigateTo,
  loadDirectoryTree,
}) {
  const messageHandlerRef = useRef();

  // Update handler when dependencies change
  useEffect(() => {
    messageHandlerRef.current = (message) => {
      // Get current state from block manager
      const state = {
        ...blockManager.getState(),
        currentFile: navState.file,
      };

      // Process message to get action
      const action = processWebSocketMessage(message, state);
      logger.info(action.type, action);

      // Handle the action
      switch (action.type) {
        case "run-start":
          blockManager.handleRunStart(
            action.latestRun,
            action.block_ids,
            action.blockResults,
          );
          break;

        case "block-result":
          blockManager.handleBlockResult(action.blockId, action.blockResult);
          break;

        case "run-end": {
          const changedBlocks = blockManager.handleRunEnd(
            action.run,
            action.error,
          );

          // Auto-scroll to single changed block
          if (changedBlocks.length === 1) {
            scrollToBlock(changedBlocks[0]);
          }
          break;
        }

        case "reload":
          window.location.reload();
          break;

        case "file-changed":
          handleFileChanged(action.path, navState, navigateTo);
          break;

        case "directory-changed":
          loadDirectoryTree();
          break;

        case "unknown":
          logger.warn("Unknown WebSocket message type:", action.messageType);
          break;

        case "no-op":
          // No action needed
          break;
      }
    };
  }, [blockManager, navState, navigateTo, loadDirectoryTree]);

  // Stable callback that won't cause WebSocket to reconnect
  const handleMessage = useCallback((message) => {
    if (messageHandlerRef.current) {
      messageHandlerRef.current(message);
    }
  }, []);

  return handleMessage;
}

/**
 * Scroll to and highlight a block
 */
function scrollToBlock(blockId) {
  setTimeout(() => {
    const element = document.querySelector(`[data-block-id="${blockId}"]`);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "center" });

      // Briefly highlight the block
      element.style.backgroundColor = "#fffbdd";
      setTimeout(() => {
        element.style.backgroundColor = "";
      }, 1000);
    }
  }, 100); // Small delay to ensure DOM is updated
}

/**
 * Handle file-changed WebSocket message
 */
function handleFileChanged(path, navState, navigateTo) {
  // Navigate to changed file only if we're viewing the root directory
  if (navState.type === "directory" && navState.directory === "/") {
    logger.info(`File changed: ${path}, navigating from root...`);
    navigateTo(path);
  } else if (navState.type === "file") {
    logger.info(`File changed: ${path}, but already viewing ${navState.file}`);
  }
}
