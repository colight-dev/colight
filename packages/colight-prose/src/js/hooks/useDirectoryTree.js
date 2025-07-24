import { useState, useCallback, useEffect } from "react";
import createLogger from "../logger.js";

const logger = createLogger("directory-tree");

/**
 * Hook for managing directory tree state
 */
export function useDirectoryTree(navState) {
  const [directoryTree, setDirectoryTree] = useState(null);
  const [isLoadingTree, setIsLoadingTree] = useState(false);

  const loadDirectoryTree = useCallback(async () => {
    if (!directoryTree) setIsLoadingTree(true);

    try {
      const response = await fetch("/api/index");
      if (!response.ok) {
        throw new Error("Failed to load directory tree");
      }
      const data = await response.json();
      setDirectoryTree(data);
      logger.debug("Directory tree loaded", {
        entries: Object.keys(data).length,
      });
    } catch (error) {
      logger.error("Failed to load directory tree:", error);
    } finally {
      setIsLoadingTree(false);
    }
  }, [directoryTree]);

  // Load directory tree when viewing a directory
  useEffect(() => {
    if (navState.type === "directory" && !directoryTree && !isLoadingTree) {
      loadDirectoryTree();
    }
  }, [navState.type, directoryTree, isLoadingTree, loadDirectoryTree]);

  return {
    directoryTree,
    isLoadingTree,
    loadDirectoryTree,
  };
}
