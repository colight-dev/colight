import { useEffect, useState, useCallback, useRef } from "react";
import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
  useNavigate,
  useParams,
} from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
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
import { useStateWithDeps } from "./hooks/useStateWithDeps.js";
import bylight from "./bylight.js";

// ========== Constants ==========

const WEBSOCKET_RECONNECT_DELAY = 1000; // ms

// ========== Route Parsing - Single Source of Truth ==========

/**
 * Parse the current route into navigation state
 *
 * @param {string} path - The route path from params["*"]
 * @returns {Object} Navigation state
 */
function parseRoute(path) {
  // Normalize path - remove leading/trailing slashes for consistency
  const normalizedPath = path?.replace(/^\/+|\/+$/g, "") || "";

  if (normalizedPath === "") {
    // Root directory
    return {
      type: "directory",
      path: "/",
      displayPath: "/",
      segments: [],
      file: null,
      directory: "/",
    };
  }

  // Check if it's a directory (ends with /)
  if (path.endsWith("/")) {
    const segments = normalizedPath.split("/").filter(Boolean);
    return {
      type: "directory",
      path: normalizedPath + "/",
      displayPath: normalizedPath + "/",
      segments,
      file: null,
      directory: normalizedPath + "/",
    };
  }

  // It's a file
  const segments = normalizedPath.split("/").filter(Boolean);
  const dirSegments = segments.slice(0, -1);

  return {
    type: "file",
    path: normalizedPath,
    displayPath: normalizedPath,
    segments,
    file: normalizedPath,
    directory: dirSegments.length > 0 ? dirSegments.join("/") + "/" : "/",
  };
}

// ========== Navigation Helper ==========

/**
 * Convert any navigation request to a proper route
 * This ensures consistency regardless of how navigation is triggered
 */
function normalizeNavigationPath(path) {
  if (!path || path === "/") {
    return "/"; // Root
  }

  // Remove leading slash for processing
  let normalized = path.startsWith("/") ? path.slice(1) : path;

  // Ensure directories end with /
  if (!normalized.includes(".") && !normalized.endsWith("/")) {
    normalized += "/";
  }

  return "/" + normalized;
}

// ========== Content Components (unchanged) ==========

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
        setColightData(([i]) => [
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
          setColightData(([i, c]) => [i, c, pending]);
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
      setColightData(([i]) => [i + 1, pendingData, null]);
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

const Code = ({ source }) => {
  return (
    <pre
      className={tw(
        "bg-gray-100 p-4 rounded-lg overflow-x-auto mb-4 language-python",
      )}
    >
      {source}
    </pre>
  );
};

const ElementRenderer = ({ element }) => {
  // Skip if element shouldn't be shown
  if (!element.show) return null;

  switch (element.type) {
    case "prose":
      return md({ className: "mb-4" }, element.value);

    case "statement":
    case "expression":
      return <Code source={element.value} />;

    default:
      return null;
  }
};

// Group consecutive code elements and extract visuals
const groupBlockElements = (elements) => {
  const groupedElements = [];
  let currentCodeGroup = [];

  elements.forEach((element) => {
    if (element.type === "statement" || element.type === "expression") {
      currentCodeGroup.push(element);
    } else {
      // Non-code element (prose) - flush the current group
      if (currentCodeGroup.length > 0) {
        groupedElements.push({
          type: "code-group",
          elements: currentCodeGroup,
        });
        currentCodeGroup = [];
      }
      groupedElements.push(element);
    }
  });

  // Don't forget the last group if it exists
  if (currentCodeGroup.length > 0) {
    groupedElements.push({ type: "code-group", elements: currentCodeGroup });
  }

  // Check if the last element is an expression with a visual
  const lastElement = elements[elements.length - 1];
  if (
    lastElement &&
    lastElement.type === "expression" &&
    (lastElement.visual || lastElement.visual_ref)
  ) {
    // Add a separate visual element
    groupedElements.push({
      type: "visual",
      visual: lastElement.visual,
      visual_ref: lastElement.visual_ref,
    });
  }

  return groupedElements;
};

const BlockRenderer = ({ block, pragmaOverrides }) => {
  // If block is pending but has content, show content with pending indicator
  const isPending = block.pending;

  if (!block.elements || block.elements.length === 0) {
    // Only show placeholder if block truly has no content yet
    if (isPending) {
      return (
        <div
          className={tw(`opacity-50 animate-pulse`)}
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

  const groupedElements = groupBlockElements(block.elements);

  return (
    <div
      className={tw(`${isPending ? "relative" : ""}`)}
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
          // Filter elements based on pragma overrides
          const visibleElements = item.elements.filter((el) => {
            if (pragmaOverrides.hideCode) return false;
            if (pragmaOverrides.hideStatements && el.type === "statement")
              return false;
            return el.show;
          });

          // Only render if there are visible elements
          if (visibleElements.length === 0) return null;

          return (
            <Code
              key={idx}
              source={visibleElements.map((el) => el.value).join("\n")}
            />
          );
        } else if (item.type === "visual") {
          return !pragmaOverrides.hideVisuals ? (
            <ColightVisual
              key={`visual-${idx}`}
              data={item.visual}
              dataRef={item.visual_ref}
            />
          ) : null;
        } else {
          // Prose elements - check visibility
          if (pragmaOverrides.hideProse) return null;
          return <ElementRenderer key={idx} element={item} />;
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

const DocumentRenderer = ({ blocks, pragmaOverrides }) => {
  const docRef = useRef();

  useEffect(() => {
    // Run bylight on the entire document after render
    if (docRef.current) {
      console.log("Run bylight on docRef");
      bylight({ target: docRef.current });
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
  }, [onMessage, wsRefOut]);

  return connected;
};

// ========== Message Handler Factory ==========

export const createWebSocketMessageHandler = (deps) => {
  const {
    latestRunRef,
    blockResultsRef,
    changedBlocksRef,
    setCurrentFile,
    setBlockResults,
    currentFile,
  } = deps;

  return (message) => {
    // Process the message using the extracted logic
    const state = {
      latestRun: latestRunRef.current,
      blockResults: blockResultsRef.current || {},
      changedBlocks: changedBlocksRef.current,
      currentFile,
    };

    const action = processWebSocketMessage(message, state);

    // Handle the action
    switch (action.type) {
      case "run-start":
        // Update latest run version
        latestRunRef.current = action.latestRun;
        changedBlocksRef.current = action.changedBlocks;

        if (action.blockResults !== undefined) {
          // Update state with new block results
          setCurrentFile(action.currentFile);
          setBlockResults(action.blockResults);
        } else {
          // Legacy behavior - only update current file, don't clear blocks
          setCurrentFile(action.currentFile);
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
  const navigate = useNavigate();
  const params = useParams();

  // Parse route - SINGLE SOURCE OF TRUTH
  const routePath = params["*"] || "";
  const navState = parseRoute(routePath);

  const [directoryTree, setDirectoryTree] = useState(null); // Cached directory tree
  const [isLoadingTree, setIsLoadingTree] = useState(false);
  const [pragmaOverrides, setPragmaOverrides] = useState({
    hideStatements: false,
    hideCode: false,
    hideProse: false,
    hideVisuals: false,
  });
  const [isCommandBarOpen, setIsCommandBarOpen] = useState(false);
  const [pinnedFile, setPinnedFile] = useState(null); // For pinning current file

  // File-scoped state that resets when file changes
  const [blockResults, setBlockResults] = useStateWithDeps({}, [navState.file]);

  // Refs for WebSocket callback
  const latestRunRef = useRef(0); // Track latest run version
  const blockResultsRef = useRef({}); // Track block results for WebSocket callback
  const loadingFileRef = useRef(null); // Track which file we're loading

  // Keep blockResults ref in sync immediately (not in useEffect)
  blockResultsRef.current = blockResults;

  // Load directory tree
  const loadDirectoryTree = useCallback(async () => {
    if (directoryTree || isLoadingTree) return; // Already loaded or loading

    setIsLoadingTree(true);
    try {
      const response = await fetch("/api/index");
      if (!response.ok) {
        throw new Error("Failed to load directory tree");
      }
      const data = await response.json();
      setDirectoryTree(data);
    } catch (error) {
      console.error("Failed to load directory tree:", error);
    } finally {
      setIsLoadingTree(false);
    }
  }, [directoryTree, isLoadingTree]);

  // Load directory tree when viewing a directory
  useEffect(() => {
    if (navState.type === "directory" && !directoryTree && !isLoadingTree) {
      loadDirectoryTree();
    }
  }, [navState.type, directoryTree, isLoadingTree, loadDirectoryTree]);

  // Single navigation function - always updates the route
  const navigateTo = useCallback(
    (path) => {
      const normalized = normalizeNavigationPath(path);
      // Synchronously update loadingFileRef when navigating
      if (normalized.endsWith(".py")) {
        loadingFileRef.current = normalized.substring(1); // Remove leading /
      } else if (normalized !== "/" && !normalized.endsWith("/")) {
        loadingFileRef.current = normalized.substring(1) + ".py";
      } else {
        loadingFileRef.current = null;
      }
      navigate(normalized);
    },
    [navigate],
  );

  // Track changed blocks for the current run
  const changedBlocksRef = useRef(new Set());

  // Create a stable message handler that doesn't change
  const messageHandlerRef = useRef();

  // Update the handler whenever dependencies change, but don't recreate the callback
  useEffect(() => {
    messageHandlerRef.current = (message) => {
      const handler = createWebSocketMessageHandler({
        latestRunRef,
        blockResultsRef,
        changedBlocksRef,
        setCurrentFile: (file) => {
          // Check if we should navigate based on pinning state
          if (pinnedFile) {
            // If a file is pinned, only navigate if it's the pinned file that changed
            if (file === pinnedFile) {
              navigateTo(file);
            }
            // Otherwise, don't navigate (but still process the update in the background)
          } else {
            // No file is pinned, navigate to any changed file
            if (navState.file !== file) {
              navigateTo(file);
            }
          }
        },
        setBlockResults,
        currentFile: loadingFileRef.current,
      });
      return handler(message);
    };
  }, [navigateTo, setBlockResults, navState.file, pinnedFile]);

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

  // Request file load when connected and we're viewing a file
  useEffect(() => {
    if (connected && navState.type === "file" && wsRef.current) {
      // Update loadingFileRef when we request a file
      loadingFileRef.current = navState.file;
      // Request the server to load this file
      wsRef.current.send(
        JSON.stringify({
          type: "request-load",
          path: navState.file,
          clientRun: latestRunRef.current,
        }),
      );
    }
  }, [connected, navState.type, navState.file]);

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
  }, [directoryTree, loadDirectoryTree]);

  return (
    <>
      <CommandBar
        isOpen={isCommandBarOpen}
        onClose={() => setIsCommandBarOpen(false)}
        directoryTree={directoryTree}
        currentFile={navState.file}
        onOpenFile={navigateTo}
        pragmaOverrides={pragmaOverrides}
        setPragmaOverrides={setPragmaOverrides}
        pinnedFile={pinnedFile}
        setPinnedFile={setPinnedFile}
      />

      <TopBar
        currentFile={navState.file}
        currentPath={navState.path}
        isDirectory={navState.type === "directory"}
        connected={connected}
        onNavigate={navigateTo}
        isLoading={false}
        pragmaOverrides={pragmaOverrides}
        setPragmaOverrides={setPragmaOverrides}
        pinnedFile={pinnedFile}
        setPinnedFile={setPinnedFile}
      />

      <div className={tw("mt-10")}>
        {navState.type === "directory" && directoryTree ? (
          <div className={tw("max-w-4xl mx-auto px-4 py-8")}>
            <DirectoryBrowser
              directoryPath={navState.directory}
              tree={directoryTree}
              onSelectFile={navigateTo}
              onNavigateToDirectory={navigateTo}
            />
          </div>
        ) : navState.type === "directory" && !directoryTree ? (
          // Loading state for directory
          <div className={tw("max-w-4xl mx-auto px-4 py-8")}>
            <div className={tw("text-center text-gray-500")}>
              Loading directory...
            </div>
          </div>
        ) : navState.type === "file" && Object.keys(blockResults).length > 0 ? (
          <DocumentRenderer
            blocks={blockResults}
            pragmaOverrides={pragmaOverrides}
          />
        ) : navState.type === "file" ? (
          // Loading state for file
          <div className={tw("max-w-4xl mx-auto px-4 py-8")}>
            <div className={tw("text-center text-gray-500")}>
              Loading {navState.file}...
            </div>
            {/* Add hint if loading takes too long */}
            <div className={tw("text-center text-gray-400 text-sm mt-4")}>
              If this takes too long, check that the server is running and the
              file exists.
            </div>
          </div>
        ) : null}
      </div>
    </>
  );
};

// ========== Router Setup ==========

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

const router = createBrowserRouter([
  {
    path: "*",
    element: <LiveServerApp />,
  },
]);

// ========== Mount the App ==========

if (typeof window !== "undefined") {
  const root = document.getElementById("root");
  if (root) {
    ReactDOM.createRoot(root).render(
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    );
  }
}

// Export components for testing
export { LiveServerApp, BlockRenderer, ColightVisual };
