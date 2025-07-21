import { useEffect, useRef } from "react";
import {
  useWebSocket,
  useMessageHandler,
} from "./contexts/WebSocketContext.jsx";
import { useStateWithDeps } from "./hooks/useStateWithDeps.js";
import { getClientId } from "./client-id.js";
import bylight from "./bylight.ts";
import { tw } from "../../../colight/src/js/utils";
import createLogger from "./logger.js";

const logger = createLogger("DocumentViewer");

// Import BlockRenderer from live.jsx (we'll export it)
import { BlockRenderer } from "./live.jsx";

/**
 * Smart block replacement algorithm
 * When block IDs change, try to preserve old content while new content loads
 */
function createBlockReplacementMap(prevIds, nextIds, prevResults) {
  const replacementMap = new Map();
  const prevSet = new Set(prevIds);
  const nextSet = new Set(nextIds);

  // For each position, check if we should preserve old content
  nextIds.forEach((nextId, index) => {
    // If this is a new block ID at this position
    if (!prevSet.has(nextId) && index < prevIds.length) {
      const prevId = prevIds[index];
      // If the previous block at this position is no longer present
      if (!nextSet.has(prevId)) {
        // Map: new block ID -> old block content to preserve temporarily
        replacementMap.set(nextId, {
          oldId: prevId,
          oldContent: prevResults[prevId],
          position: index,
        });
      }
    }
  });

  return replacementMap;
}

export const DocumentViewer = ({ file, pragmaOverrides, navigateTo }) => {
  const docRef = useRef();
  const { connected, sendMessage } = useWebSocket();
  const watchedFileRef = useRef(null);
  const clientIdRef = useRef(getClientId());

  // Block management state - resets when file changes
  const [blockResults, setBlockResults] = useStateWithDeps({}, [file]);

  // Refs for tracking state across renders
  const latestRunRef = useRef(0);
  const blockResultsRef = useRef({});
  const prevBlockIdsRef = useRef([]);
  const currentBlockIdsRef = useRef([]);
  const changedBlocksRef = useRef(new Set());
  const replacementMapRef = useRef(new Map());

  // Keep blockResults ref in sync
  blockResultsRef.current = blockResults;

  // Handle WebSocket messages directly
  useMessageHandler({
    types: ["run-start", "block-result", "run-end", "file-changed"],
    handler: (message) => {
      logger.debug("Received message:", message.type);

      switch (message.type) {
        case "run-start": {
          // Only process if it's for our file
          if (message.file !== file) return;

          logger.debug("Run start", {
            runNumber: message.run,
            blockCount: message.block_ids?.length,
          });

          // Update run number
          latestRunRef.current = message.run;

          // Clear changed blocks for new run
          changedBlocksRef.current.clear();

          // Track block ID changes
          prevBlockIdsRef.current = currentBlockIdsRef.current;
          currentBlockIdsRef.current = message.block_ids || [];

          // Create smart replacement map
          replacementMapRef.current = createBlockReplacementMap(
            prevBlockIdsRef.current,
            currentBlockIdsRef.current,
            blockResultsRef.current,
          );

          // Build new block results
          const newResults = {};

          if (message.block_ids) {
            for (let i = 0; i < message.block_ids.length; i++) {
              const blockId = message.block_ids[i];

              if (blockResults[blockId]) {
                // Same block ID = unchanged content, just update ordinal
                newResults[blockId] = {
                  ...blockResults[blockId],
                  ordinal: i,
                  pending: false, // Clear any previous pending state
                };
              } else {
                // New block ID - check if we should preserve old content
                const replacement = replacementMapRef.current.get(blockId);

                if (replacement && replacement.oldContent) {
                  // Preserve old content but mark as pending
                  newResults[blockId] = {
                    ...replacement.oldContent,
                    ordinal: i,
                    pending: true,
                    pendingReplacement: true, // Flag for special handling
                  };
                } else {
                  // Truly new block with no replacement
                  newResults[blockId] = {
                    elements: [],
                    ok: true,
                    ordinal: i,
                    pending: true,
                  };
                }
              }
            }
          }

          setBlockResults(newResults);
          break;
        }

        case "block-result": {
          // Only process if run matches
          if (message.run !== latestRunRef.current) return;

          let blockResult;

          if (message.unchanged) {
            // Backend says unchanged, use existing data
            blockResult = blockResults[message.block];
            if (!blockResult) {
              logger.error(
                `INVARIANT VIOLATION: Got unchanged for unknown block ${message.block}`,
              );
              return;
            }
            // Clear pending state
            blockResult = { ...blockResult, pending: false };
          } else {
            // Full data from backend
            blockResult = {
              ok: message.ok,
              stdout: message.stdout,
              error: message.error,
              showsVisual: message.showsVisual,
              elements: message.elements || [],
              cache_hit: message.cache_hit || false,
              ordinal: message.ordinal,
              pending: false,
              pendingReplacement: false, // Clear replacement flag
            };

            // Track as changed if it's new or was a replacement
            const replacement = replacementMapRef.current.get(message.block);
            if (!blockResults[message.block] || replacement) {
              changedBlocksRef.current.add(message.block);
            }
          }

          setBlockResults((prev) => ({
            ...prev,
            [message.block]: blockResult,
          }));
          break;
        }

        case "run-end": {
          // Only process if run matches
          if (message.run !== latestRunRef.current) return;

          const changedBlocks = Array.from(changedBlocksRef.current);
          logger.info(`Run ${message.run} completed`, { changedBlocks });

          if (message.error) {
            logger.error("Run error:", message.error);
          }

          // Auto-scroll to single changed block
          if (changedBlocks.length === 1) {
            scrollToBlock(changedBlocks[0]);
          }

          // Clear replacement map after run completes
          replacementMapRef.current.clear();
          break;
        }

        case "file-changed": {
          // Navigate to changed file only if we're in the root directory view
          // This logic might need to move up to the parent component
          if (message.path && message.path !== file) {
            logger.info(`File changed: ${message.path}, but viewing ${file}`);
          }
          break;
        }
      }
    },
    priority: 8,
  });

  // Handle file watching
  useEffect(() => {
    if (connected && file) {
      const clientId = clientIdRef.current;

      // Track what we're watching
      watchedFileRef.current = file;

      // Send watch message
      sendMessage({
        type: "watch-file",
        path: file,
        clientId: clientId,
      });

      // Request initial load
      sendMessage({
        type: "request-load",
        path: file,
        clientRun: latestRunRef.current,
      });

      // Cleanup: unwatch on unmount or file change
      return () => {
        if (watchedFileRef.current) {
          sendMessage({
            type: "unwatch-file",
            path: watchedFileRef.current,
            clientId: clientId,
          });
          watchedFileRef.current = null;
        }
      };
    }
  }, [connected, file, sendMessage]);

  // Run syntax highlighting after render
  useEffect(() => {
    if (docRef.current) {
      bylight({ target: docRef.current });
    }
  }, [blockResults]);

  // Show loading state if no blocks yet
  if (!blockResults || Object.keys(blockResults).length === 0) {
    return (
      <div className={tw("max-w-4xl mx-auto px-4 py-8")}>
        <div className={tw("text-center text-gray-500")}>Loading {file}...</div>
        <div className={tw("text-center text-gray-400 text-sm mt-4")}>
          If this takes too long, check that the server is running and the file
          exists.
        </div>
      </div>
    );
  }

  // Sort blocks by their ordinal to maintain document order
  const sortedBlockIds = Object.keys(blockResults).sort((a, b) => {
    const ordinalA = blockResults[a].ordinal ?? 0;
    const ordinalB = blockResults[b].ordinal ?? 0;
    return ordinalA - ordinalB;
  });

  return (
    <div
      ref={docRef}
      className={tw("max-w-4xl mx-auto px-4 py-8  [&_pre]:text-sm")}
    >
      {sortedBlockIds.map((blockId) => {
        const result = blockResults[blockId];
        // Create a simplified block structure from the result
        const block = {
          id: blockId,
          elements: result.elements || [],
          error: result.error,
          stdout: result.stdout,
          showsVisual: result.showsVisual,
          pending: result.pending,
          pendingReplacement: result.pendingReplacement,
        };

        return (
          <BlockRenderer
            key={blockId}
            block={block}
            pragmaOverrides={pragmaOverrides}
          />
        );
      })}
    </div>
  );
};

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
