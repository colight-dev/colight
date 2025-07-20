import { useState, useRef, useCallback } from "react";
import { useStateWithDeps } from "./useStateWithDeps.js";
import createLogger from "../logger.js";

const logger = createLogger("block-manager");

/**
 * Hook for managing block execution results and tracking changes
 */
export function useBlockManager(currentFile) {
  // File-scoped state that resets when file changes
  const [blockResults, setBlockResults] = useStateWithDeps({}, [currentFile]);

  // Refs for tracking state across renders
  const latestRunRef = useRef(0);
  const blockResultsRef = useRef({});
  const prevBlocksRef = useRef(new Set());
  const currentBlocksRef = useRef(new Set());
  const changedBlocksRef = useRef(new Set());

  // Keep blockResults ref in sync
  blockResultsRef.current = blockResults;

  /**
   * Handle run-start message
   */
  const handleRunStart = useCallback(
    (runNumber, blockIds, blockResults) => {
      logger.debug("Run start", { runNumber, blockCount: blockIds?.length });

      // Update run number
      latestRunRef.current = runNumber;

      // Clear changed blocks for new run
      changedBlocksRef.current.clear();

      // Track block changes
      prevBlocksRef.current = currentBlocksRef.current;
      currentBlocksRef.current = new Set(blockIds || []);

      // Update block results if provided
      if (blockResults) {
        setBlockResults(blockResults);
      }
    },
    [setBlockResults],
  );

  /**
   * Handle block-result message
   */
  const handleBlockResult = useCallback(
    (blockId, blockResult) => {
      setBlockResults((prev) => ({
        ...prev,
        [blockId]: blockResult,
      }));
    },
    [setBlockResults],
  );

  /**
   * Handle run-end message
   */
  const handleRunEnd = useCallback((runNumber, error) => {
    logger.info(`Run ${runNumber} completed`, {
      changedBlocks: Array.from(changedBlocksRef.current),
    });

    if (error) {
      logger.error("Run error:", error);
    }

    // Return changed blocks for UI updates
    return Array.from(changedBlocksRef.current);
  }, []);

  /**
   * Check if a block is new/changed in the current run
   */
  const isBlockNew = useCallback((blockId) => {
    return (
      currentBlocksRef.current.has(blockId) &&
      !prevBlocksRef.current.has(blockId)
    );
  }, []);

  /**
   * Get current state snapshot
   */
  const getState = useCallback(
    () => ({
      latestRun: latestRunRef.current,
      blockResults: blockResultsRef.current,
      changedBlocks: changedBlocksRef.current,
      currentBlocks: currentBlocksRef.current,
      prevBlocks: prevBlocksRef.current,
    }),
    [],
  );

  return {
    blockResults,
    setBlockResults,
    handleRunStart,
    handleBlockResult,
    handleRunEnd,
    isBlockNew,
    getState,

    // Expose refs for WebSocket handler
    refs: {
      latestRunRef,
      blockResultsRef,
      prevBlocksRef,
      currentBlocksRef,
      changedBlocksRef,
    },
  };
}
