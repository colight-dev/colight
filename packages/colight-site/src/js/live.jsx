import React, {
  useEffect,
  useState,
  useCallback,
  useRef,
  useTransition,
} from "react";
import ReactDOM from "react-dom/client";
import { DraggableViewer } from "../../../colight/src/js/widget.jsx";
import {
  parseColightScript,
  parseColightData,
} from "../../../colight/src/js/format.js";
import { tw, md } from "../../../colight/src/js/api.jsx";
import { DirectoryBrowser } from "./DirectoryBrowser.jsx";
import CommandBar from "./CommandBar.jsx";
import TopBar from "./TopBar.jsx";
import { processWebSocketMessage } from "./websocket-message-handler.js";
import "./bylight.js";

// ========== Constants ==========

const WEBSOCKET_RECONNECT_DELAY = 1000; // ms
const HISTORY_DEBOUNCE_DELAY = 300; // ms

const splitPath = (path) => path.split("/").filter(Boolean);

// Get the base path for the application (handles sub-path hosting)
const getBasePath = () => {
  // In production, this might be set via a meta tag or global variable
  return window.COLIGHT_BASE_PATH || "";
};

// Extract the file path from the URL, accounting for base path
const getFilePathFromUrl = () => {
  const basePath = getBasePath();
  const pathname = window.location.pathname;

  if (basePath && pathname.startsWith(basePath)) {
    return pathname.slice(basePath.length + 1); // +1 for the trailing slash
  }

  return pathname.slice(1); // Default behavior
};

// ========== Document Processing ==========
// Removed applyIncrementalUpdate - now using simple state replacement with RunVersion

// ========== Content Rendering Components ==========

const ColightVisual = ({ data, dataRef }) => {
  const containerRef = useRef(null);
  const [[currentKey, currentData, pendingData], setColightData] = useState([
    0,
    null,
    null,
  ]);

  const [isLoading, setIsLoading] = useState(false);
  const [loadedId, setLoadedId] = useState(null);
  const [minHeight, setMinHeight] = useState(0);

  // Load external visual when needed
  useEffect(() => {
    if (data) {
      // We have inline data - parse it directly
      try {
        setMinHeight(containerRef.current?.offsetHeight || 0);
        setColightData(([i, c, p]) => [
          i + 1,
          parseColightScript({ textContent: data }),
          null,
        ]);
      } catch (error) {
        console.error("Error parsing Colight visual:", error);
      }
    } else {
      setIsLoading(true);
      try {
        (async () => {
          const response = await fetch(dataRef.url);
          if (!response.ok) {
            throw new Error(`Failed to load visual: ${response.status}`);
          }
          const blob = await response.blob();
          const pending = parseColightData(await blob.arrayBuffer());
          setColightData(([i, c, p]) => [i, c, pending]);
          setLoadedId(dataRef.id);
        })();
      } catch (error) {
        console.error("Error loading visual:", error);
      } finally {
        setIsLoading(false);
      }
    }
  }, [data, dataRef, loadedId]);

  // Update the displayed visual when loading is complete
  useEffect(() => {
    if (!isLoading && pendingData) {
      setMinHeight(containerRef.current?.offsetHeight || 0);
      setColightData(([i, c, p]) => [i + 1, pendingData, null]);
    }
  }, [isLoading, pendingData]);

  // Show placeholder only if we have nothing to show yet
  if (!currentData && !isLoading) {
    return <div ref={containerRef} className="colight-embed mb-4" />;
  }

  return (
    <div
      ref={containerRef}
      style={{ minHeight }}
      className="colight-embed mb-4 relative"
    >
      {/* Show existing visual if we have one */}
      {currentData && (
        <DraggableViewer
          key={currentKey}
          data={{ ...currentData, onMount: () => setMinHeight(0) }}
        />
      )}

      {/* Show loading overlay when fetching new visual */}
      {isLoading && (
        <div
          className={tw(
            `absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center rounded`,
          )}
        ></div>
      )}
    </div>
  );
};

const ElementRenderer = ({ element, pragmaOverrides }) => {
  // Skip if element shouldn't be shown
  if (!element.show) return null;

  switch (element.type) {
    case "prose":
      return md({ className: "mb-4" }, element.value);

    case "statement":
    case "expression":
      return (
        <>
          <pre
            className={tw("bg-gray-100 p-4 rounded-lg overflow-x-auto mb-4")}
          >
            <code className="language-python">{element.value}</code>
          </pre>
          {/* Render visual if it's an expression with visual data */}
          {(element.visual || element.visual_ref) &&
            !pragmaOverrides.hideVisuals && (
              <ColightVisual
                data={element.visual}
                dataRef={element.visual_ref}
              />
            )}
        </>
      );

    default:
      return null;
  }
};

const BlockRenderer = ({ block, pragmaOverrides }) => {
  // If block is pending but has content, show content with pending indicator
  const isPending = block.pending;

  if (!block.elements || block.elements.length === 0) {
    // Only show placeholder if block truly has no content yet
    if (isPending) {
      return (
        <div
          className={tw(`block-${block.id} opacity-50 animate-pulse`)}
          data-block-id={block.id}
          data-shows-visual={block.showsVisual}
        >
          <div className={tw("bg-gray-100 p-4 rounded-lg mb-4")}>
            <div className={tw("h-4 bg-gray-300 rounded animate-pulse")}></div>
          </div>
        </div>
      );
    }
    return null;
  }

  // Group consecutive statement/expression elements
  const groupedElements = [];
  let currentCodeGroup = [];

  block.elements.forEach((element) => {
    // Apply pragma overrides to element visibility
    let shouldShow = element.show;
    if (pragmaOverrides.hideStatements && element.type === "statement") {
      shouldShow = false;
    }
    if (
      pragmaOverrides.hideCode &&
      (element.type === "expression" || element.type === "statement")
    ) {
      shouldShow = false;
    }
    if (pragmaOverrides.hideProse && element.type === "prose") {
      shouldShow = false;
    }

    // Process all elements to maintain proper grouping boundaries
    if (element.type === "statement" || element.type === "expression") {
      if (shouldShow) {
        currentCodeGroup.push(element);
      } else {
        // Hidden code element - still breaks the group
        if (currentCodeGroup.length > 0) {
          groupedElements.push({
            type: "code-group",
            elements: currentCodeGroup,
          });
          currentCodeGroup = [];
        }
        // If this is a hidden expression with a visual, show just the visual
        if (
          element.type === "expression" &&
          (element.visual || element.visual_ref)
        ) {
          groupedElements.push({
            type: "visual-only",
            element: element,
          });
        }
      }
    } else {
      // Non-code element (prose) - always breaks the code group
      if (currentCodeGroup.length > 0) {
        groupedElements.push({
          type: "code-group",
          elements: currentCodeGroup,
        });
        currentCodeGroup = [];
      }
      // Only add visible non-code elements
      if (shouldShow) {
        groupedElements.push(element);
      }
    }
  });

  // Don't forget the last group if it exists
  if (currentCodeGroup.length > 0) {
    groupedElements.push({ type: "code-group", elements: currentCodeGroup });
  }

  return (
    <div
      className={tw(`block-${block.id} ${isPending ? "relative" : ""}`)}
      data-block-id={block.id}
      data-shows-visual={block.showsVisual}
    >
      {/* Show pending indicator overlay */}
      {isPending && (
        <div
          className={tw(
            "absolute inset-0 bg-yellow-100 bg-opacity-30 rounded-lg pointer-events-none z-10 animate-pulse",
          )}
        />
      )}
      {groupedElements.map((item, idx) => {
        if (item.type === "code-group") {
          return (
            <div key={idx}>
              <pre
                className={tw(
                  "bg-gray-100 p-4 rounded-lg overflow-x-auto mb-4",
                )}
              >
                <code className="language-python">
                  {item.elements.map((el) => el.value).join("\n")}
                </code>
              </pre>
              {/* Render visuals for any expressions in the group */}
              {!pragmaOverrides.hideVisuals &&
                item.elements.map((el, elIdx) =>
                  el.type === "expression" && (el.visual || el.visual_ref) ? (
                    <ColightVisual
                      key={`visual-${idx}-${elIdx}`}
                      data={el.visual}
                      dataRef={el.visual_ref}
                    />
                  ) : null,
                )}
            </div>
          );
        } else if (item.type === "visual-only") {
          return !pragmaOverrides.hideVisuals ? (
            <ColightVisual
              key={`visual-only-${idx}`}
              data={item.element.visual}
              dataRef={item.element.visual_ref}
            />
          ) : null;
        } else {
          return (
            <ElementRenderer
              key={idx}
              element={item}
              pragmaOverrides={pragmaOverrides}
            />
          );
        }
      })}
      {block.error && (
        <div
          className={tw(
            "bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg mb-4",
          )}
        >
          <pre>{block.error}</pre>
        </div>
      )}
    </div>
  );
};

const DocumentRenderer = ({ blocks, currentFile, pragmaOverrides }) => {
  const docRef = useRef();

  useEffect(() => {
    // Run bylight on the entire document after render
    if (docRef.current) {
      window.bylight({ target: docRef.current });
    }
  }, [blocks]); // Re-run when blocks change

  if (!blocks || Object.keys(blocks).length === 0) return null;

  // Sort blocks by their ID to maintain order
  const sortedBlockIds = Object.keys(blocks).sort();

  return (
    <div
      ref={docRef}
      className={tw("max-w-4xl mx-auto px-4 py-8  [&_pre]:text-sm")}
    >
      {sortedBlockIds.map((blockId) => {
        const result = blocks[blockId];
        // Create a simplified block structure from the result
        const block = {
          id: blockId,
          elements: result.elements || [],
          error: result.error,
          stdout: result.stdout,
          showsVisual: result.showsVisual,
          pending: result.pending,
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

// ========== Home Page Component ==========

const HomePage = () => {
  const [watchingPath, setWatchingPath] = useState("");

  useEffect(() => {
    // Get the directory being watched from the API
    const getWatchingPath = async () => {
      try {
        const response = await fetch("/api/index");
        if (response.ok) {
          const data = await response.json();
          setWatchingPath(data.name || "current directory");
        }
      } catch (err) {
        console.error("Failed to get watching path:", err);
      }
    };

    getWatchingPath();
  }, []);

  return (
    <div className={tw("p-10 text-center font-mono text-gray-600")}>
      <div className={tw("text-xl")}>
        Watching{" "}
        <span className={tw("font-bold")}>{watchingPath || "..."}</span>
      </div>
    </div>
  );
};

// ========== Command Bar ==========

// ========== UI Components ==========

// ========== WebSocket Hook ==========

const useWebSocket = (onMessage, wsRefOut) => {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    const connect = () => {
      const wsPort = parseInt(window.location.port) + 1;
      const ws = new WebSocket(`ws://127.0.0.1:${wsPort}`);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("LiveServer connected");
        setConnected(true);
        if (wsRefOut) {
          wsRefOut.current = ws;
        }
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("WebSocket message:", data);
        onMessage(data);
      };

      ws.onclose = () => {
        console.log("LiveServer disconnected");
        setConnected(false);
        setTimeout(connect, WEBSOCKET_RECONNECT_DELAY);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [onMessage]);

  return connected;
};

// ========== Message Handler Factory ==========

// Factory function to create the WebSocket message handler
// This allows us to test the handler logic with mock dependencies
export const createWebSocketMessageHandler = (deps) => {
  const {
    latestRunRef,
    focusedPathRef,
    blockResultsRef,
    changedBlocksRef,
    setCurrentFile,
    setBlockResults,
    startTransition = (fn) => fn(), // Default to synchronous for testing
  } = deps;

  return (message) => {
    // Process the message using the extracted logic
    const state = {
      latestRun: latestRunRef.current,
      focusedPath: focusedPathRef.current,
      blockResults: blockResultsRef.current || {},
      changedBlocks: changedBlocksRef.current,
    };

    const action = processWebSocketMessage(message, state);

    // Handle the action
    switch (action.type) {
      case "run-start":
        // Update latest run version
        latestRunRef.current = action.latestRun;
        changedBlocksRef.current = action.changedBlocks;

        if (action.blockResults) {
          // Update state with new block results
          setCurrentFile(action.currentFile);
          setBlockResults(action.blockResults);
        } else {
          // Legacy behavior - clear blocks
          startTransition(() => {
            setCurrentFile(action.currentFile);
            setBlockResults({});
          });
        }
        break;

      case "block-result":
        setBlockResults((prev) => ({
          ...prev,
          [action.blockId]: action.blockResult,
        }));
        break;

      case "run-end":
        console.log(`Run ${action.run} completed`);
        if (action.error) {
          console.error("Run error:", action.error);
        }

        // Check if we should auto-scroll to a changed block
        if (action.changedBlocks.length === 1) {
          setTimeout(() => {
            const blockId = action.changedBlocks[0];
            const element = document.querySelector(
              `[data-block-id="${blockId}"]`,
            );
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
        break;

      case "reload":
        window.location.reload();
        break;

      case "unknown":
        console.warn("Unknown WebSocket message type:", action.messageType);
        break;

      case "no-op":
        // No action needed
        break;
    }
  };
};

// ========== Main App Component ==========

const LiveServerApp = () => {
  const [currentFile, setCurrentFile] = useState(null);
  const [documentData, setDocumentData] = useState(null);
  const [blockResults, setBlockResults] = useState({}); // Track block results by ID
  const [focusedPath, setFocusedPath] = useState(null); // Single focus state - can be file or directory
  const [browsingDirectory, setBrowsingDirectoryState] = useState(null); // Directory being browsed
  const [isPending, startTransition] = useTransition(); // For smooth transitions
  const [directoryTree, setDirectoryTree] = useState(null); // Cached directory tree
  const [isLoadingTree, setIsLoadingTree] = useState(false);
  const [pragmaOverrides, setPragmaOverrides] = useState({
    hideStatements: false,
    hideCode: false,
    hideProse: false,
    hideVisuals: false,
  });
  const [isCommandBarOpen, setIsCommandBarOpen] = useState(false);

  // Refs for WebSocket callback
  const currentFileRef = useRef(null);
  const focusedPathRef = useRef(null); // Track focused path for WebSocket callback
  const latestRunRef = useRef(0); // Track latest run version
  const blockResultsRef = useRef({}); // Track block results for WebSocket callback

  // Track if we're still initializing
  const [isInitialized, setIsInitialized] = useState(false);

  // Debounce timer for history updates
  const historyUpdateTimerRef = useRef(null);

  // Keep currentFile ref in sync
  useEffect(() => {
    currentFileRef.current = currentFile;
  }, [currentFile]);

  // Keep blockResults ref in sync immediately (not in useEffect)
  blockResultsRef.current = blockResults;

  // Handle URL updates
  useEffect(() => {
    // Don't update URL during initial load
    if (!isInitialized) return;

    // Clear existing timer
    if (historyUpdateTimerRef.current) {
      clearTimeout(historyUpdateTimerRef.current);
    }

    // Debounce history updates
    historyUpdateTimerRef.current = setTimeout(() => {
      // Update URL when current file changes
      if (currentFile) {
        const url = new URL(window.location);
        url.pathname = `/${currentFile}`;
        if (focusedPath) {
          url.searchParams.set("focus", focusedPath);
        } else {
          url.searchParams.delete("focus");
        }
        window.history.pushState({}, "", url.toString());
      } else if (window.location.pathname !== "/") {
        // No file selected, go to root
        const url = new URL(window.location);
        url.pathname = "/";
        if (focusedPath) {
          url.searchParams.set("focus", focusedPath);
        } else {
          url.searchParams.delete("focus");
        }
        window.history.pushState({}, "", url.toString());
      }
    }, HISTORY_DEBOUNCE_DELAY);

    return () => {
      if (historyUpdateTimerRef.current) {
        clearTimeout(historyUpdateTimerRef.current);
      }
    };
  }, [currentFile, focusedPath, isInitialized]);

  useEffect(() => {
    focusedPathRef.current = focusedPath;
  }, [focusedPath]);

  // Load directory tree
  const loadDirectoryTree = async () => {
    if (directoryTree) return; // Already loaded

    setIsLoadingTree(true);
    try {
      const response = await fetch("/api/index");
      if (!response.ok) {
        throw new Error("Failed to load directory tree");
      }
      const data = await response.json();
      startTransition(() => {
        setDirectoryTree(data);
      });
    } catch (error) {
      console.error("Failed to load directory tree:", error);
    } finally {
      setIsLoadingTree(false);
    }
  };

  // Custom setBrowsingDirectory that loads tree first
  const setBrowsingDirectory = async (dir) => {
    // Handle boolean values from TopBar
    if (dir === true) {
      dir = "/";
    } else if (dir === false) {
      dir = null;
    }

    if (dir && !directoryTree) {
      await loadDirectoryTree();
    }
    setBrowsingDirectoryState(dir);
  };

  // Request a file load from the server
  const requestFileLoad = useCallback((path) => {
    // Send a synthetic file-changed event to trigger a build
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "request-load", path }));
    }
  }, []);

  // Track changed blocks for the current run
  const changedBlocksRef = useRef(new Set());

  // Create a stable message handler that doesn't change
  const messageHandlerRef = useRef();

  // Update the handler whenever dependencies change, but don't recreate the callback
  useEffect(() => {
    messageHandlerRef.current = (message) => {
      const handler = createWebSocketMessageHandler({
        latestRunRef,
        focusedPathRef,
        blockResultsRef,
        changedBlocksRef,
        setCurrentFile,
        setBlockResults,
        startTransition,
      });
      return handler(message);
    };
  }, [setCurrentFile, setBlockResults, startTransition]);

  // Stable callback that won't cause WebSocket to reconnect
  const handleWebSocketMessage = useCallback((message) => {
    if (messageHandlerRef.current) {
      messageHandlerRef.current(message);
    }
  }, []);

  // WebSocket ref for sending messages
  const wsRef = useRef(null);

  // Setup WebSocket connection
  const connected = useWebSocket(handleWebSocketMessage, wsRef);

  // Initial load
  useEffect(() => {
    // Handle initial route and focus state
    const path = getFilePathFromUrl();
    const params = new URLSearchParams(window.location.search);
    const focus = params.get("focus");

    if (focus) {
      setFocusedPath(focus);
    }

    if (path) {
      // Set current file - the WebSocket will handle loading when connected
      setCurrentFile(path);
    }

    // Mark as initialized after initial load
    setTimeout(() => setIsInitialized(true), 0);
  }, []);

  // Request file load when connected and current file changes
  useEffect(() => {
    if (connected && currentFile && wsRef.current) {
      // Request the server to load this file
      // Include the current run version so server knows if we need full data
      wsRef.current.send(
        JSON.stringify({
          type: "request-load",
          path: currentFile,
          clientRun: latestRunRef.current,
        }),
      );
    }
  }, [connected, currentFile]);

  // Handle popstate (browser back/forward)
  useEffect(() => {
    const handlePopState = () => {
      const path = getFilePathFromUrl();
      const params = new URLSearchParams(window.location.search);
      const focus = params.get("focus");

      // Restore focus state
      setFocusedPath(focus);

      if (path) {
        setCurrentFile(path);
      } else {
        startTransition(() => {
          setCurrentFile(null);
          setBlockResults({});
        });
      }
    };

    if (typeof window !== "undefined") {
      window.addEventListener("popstate", handlePopState);
      return () => window.removeEventListener("popstate", handlePopState);
    }
  }, []);

  // Handle global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Cmd+K (Mac) or Ctrl+K (Windows/Linux)
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setIsCommandBarOpen((x) => !x);
        if (!directoryTree) loadDirectoryTree();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [directoryTree]);

  return (
    <>
      <CommandBar
        isOpen={isCommandBarOpen}
        onClose={() => setIsCommandBarOpen(false)}
        directoryTree={directoryTree}
        currentFile={currentFile}
        onOpenFile={(path) => {
          setCurrentFile(path);
          setBrowsingDirectoryState(null);
        }}
        pragmaOverrides={pragmaOverrides}
        setPragmaOverrides={setPragmaOverrides}
        focusedPath={focusedPath}
        setFocusedPath={setFocusedPath}
      />

      <TopBar
        currentFile={currentFile}
        connected={connected}
        focusedPath={focusedPath}
        setFocusedPath={setFocusedPath}
        browsingDirectory={browsingDirectory}
        setBrowsingDirectory={setBrowsingDirectory}
        isLoading={false}
        pragmaOverrides={pragmaOverrides}
        setPragmaOverrides={setPragmaOverrides}
      />

      <div className={tw("mt-10 relative")}>
        {browsingDirectory && directoryTree ? (
          <DirectoryBrowser
            directoryPath={browsingDirectory}
            tree={directoryTree}
            onSelectFile={(path) => {
              setCurrentFile(path);
              setBrowsingDirectoryState(null);
            }}
            onClose={() => setBrowsingDirectoryState(null)}
          />
        ) : currentFile && Object.keys(blockResults).length > 0 ? (
          <DocumentRenderer
            key={currentFile}
            blocks={blockResults}
            currentFile={currentFile}
            pragmaOverrides={pragmaOverrides}
          />
        ) : currentFile ? (
          // Loading state for current file
          <div className={tw("max-w-4xl mx-auto px-4 py-8")}>
            <div className={tw("text-center text-gray-500")}>
              Loading {currentFile}...
            </div>
          </div>
        ) : (
          <HomePage />
        )}
      </div>
    </>
  );
};

// ========== Mount the App ==========

if (typeof window !== "undefined") {
  const root = document.getElementById("root");
  if (root) {
    ReactDOM.createRoot(root).render(<LiveServerApp />);
  }
}

// Export components for testing
export { LiveServerApp, BlockRenderer, ColightVisual };
