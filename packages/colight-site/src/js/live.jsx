import React, {
  useEffect,
  useState,
  useCallback,
  useRef,
  useTransition,
} from "react";
import ReactDOM from "react-dom/client";
import Fuse from "fuse.js";
import { DraggableViewer } from "../../../colight/src/js/widget.jsx";
import {
  parseColightScript,
  parseColightData,
} from "../../../colight/src/js/format.js";
import { tw, md } from "../../../colight/src/js/api.jsx";
import { DirectoryBrowser } from "./DirectoryBrowser.jsx";
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

  const setPendingData = (pending) =>
    setColightData(([i, d, p]) => [i, d, pending]);

  const [isLoading, setIsLoading] = useState(false);
  const [loadedId, setLoadedId] = useState(null);
  const [minHeight, setMinHeight] = useState(0);
  const [isPending, startTransition] = useTransition(); // For smooth transitions

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
      startTransition(() => {
        setColightData(([i, c, p]) => [i + 1, pendingData, null]);
      });
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
      {(isLoading || isPending) && (
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

const CommandBar = ({
  isOpen,
  onClose,
  directoryTree,
  currentFile,
  onOpenFile,
  pragmaOverrides,
  setPragmaOverrides,
  focusedPath,
  setFocusedPath,
}) => {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [commands, setCommands] = useState([]);
  const inputRef = useRef(null);
  const fuse = useRef(null);

  // Initialize Fuse.js for file search
  useEffect(() => {
    if (directoryTree && directoryTree !== null) {
      const allFiles = [];

      // Recursive function to extract all .colight.py files
      const extractFiles = (items, currentPath = "") => {
        if (!items || !Array.isArray(items)) return;

        items.forEach((item) => {
          if (item.type === "file") {
            const fullPath = currentPath
              ? `${currentPath}/${item.name}`
              : item.name;
            allFiles.push({
              name: item.name,
              path: fullPath,
              relativePath: fullPath,
              // Add searchable terms
              searchTerms: [
                item.name.replace(".colight.py", ""),
                item.name,
                fullPath,
                ...fullPath.split("/").filter(Boolean),
              ]
                .join(" ")
                .toLowerCase(),
            });
          } else if (item.type === "directory" && item.children) {
            const newPath = currentPath
              ? `${currentPath}/${item.name}`
              : item.name;
            extractFiles(item.children, newPath);
          }
        });
      };

      // Start extraction - handle both root with children or direct array
      if (directoryTree.children) {
        extractFiles(directoryTree.children);
      } else if (Array.isArray(directoryTree)) {
        extractFiles(directoryTree);
      } else {
        extractFiles([directoryTree]);
      }

      fuse.current = new Fuse(allFiles, {
        keys: ["name", "path", "searchTerms"],
        threshold: 0.4,
        includeScore: true,
        minMatchCharLength: 2,
        // More fuzzy search options
        location: 0,
        distance: 100,
        useExtendedSearch: false,
        ignoreLocation: true,
        findAllMatches: true,
      });
    }
  }, [directoryTree]);

  // Generate commands based on query
  useEffect(() => {
    const newCommands = [];
    const lowerQuery = query.toLowerCase().trim();

    // Always define available commands
    const allCommands = [
      {
        type: "toggle",
        title: `${pragmaOverrides.hideStatements ? "Show" : "Hide"} Statements`,
        subtitle: `${pragmaOverrides.hideStatements ? "○" : "●"} Toggle statement visibility`,
        searchTerms: ["statements", "hide", "show", "toggle"],
        action: () =>
          setPragmaOverrides((prev) => ({
            ...prev,
            hideStatements: !prev.hideStatements,
          })),
      },
      {
        type: "toggle",
        title: `${pragmaOverrides.hideCode ? "Show" : "Hide"} Code`,
        subtitle: `${pragmaOverrides.hideCode ? "○" : "●"} Toggle code visibility`,
        searchTerms: ["code", "hide", "show", "toggle"],
        action: () =>
          setPragmaOverrides((prev) => ({ ...prev, hideCode: !prev.hideCode })),
      },
      {
        type: "toggle",
        title: `${pragmaOverrides.hideProse ? "Show" : "Hide"} Prose`,
        subtitle: `${pragmaOverrides.hideProse ? "○" : "●"} Toggle prose visibility`,
        searchTerms: ["prose", "text", "markdown", "hide", "show", "toggle"],
        action: () =>
          setPragmaOverrides((prev) => ({
            ...prev,
            hideProse: !prev.hideProse,
          })),
      },
      {
        type: "toggle",
        title: `${pragmaOverrides.hideVisuals ? "Show" : "Hide"} Visuals`,
        subtitle: `${pragmaOverrides.hideVisuals ? "○" : "●"} Toggle visual outputs`,
        searchTerms: ["visuals", "visual", "output", "hide", "show", "toggle"],
        action: () =>
          setPragmaOverrides((prev) => ({
            ...prev,
            hideVisuals: !prev.hideVisuals,
          })),
      },
    ];

    if (currentFile) {
      allCommands.push({
        type: "pin",
        title:
          focusedPath === currentFile
            ? "Unpin Current File"
            : "Pin Current File",
        subtitle:
          focusedPath === currentFile
            ? "📌 Remove focus from current file"
            : "📌 Focus on current file only",
        searchTerms: ["pin", "unpin", "focus", "file"],
        action: () =>
          setFocusedPath(focusedPath === currentFile ? null : currentFile),
      });
    }

    if (lowerQuery) {
      // Filter commands that match the query
      const matchingCommands = allCommands.filter(
        (cmd) =>
          cmd.title.toLowerCase().includes(lowerQuery) ||
          cmd.subtitle.toLowerCase().includes(lowerQuery) ||
          cmd.searchTerms.some((term) => term.includes(lowerQuery)),
      );
      newCommands.push(...matchingCommands);

      // File search results
      if (fuse.current) {
        const results = fuse.current.search(query);
        results.slice(0, 10).forEach((result) => {
          newCommands.push({
            type: "file",
            title: result.item.name,
            subtitle: result.item.relativePath,
            score: result.score,
            action: () => onOpenFile(result.item.relativePath),
          });
        });
      }
    } else {
      // Show all commands when no query
      newCommands.push(...allCommands);
    }

    setCommands(newCommands);
    setSelectedIndex(0);
  }, [
    query,
    pragmaOverrides,
    currentFile,
    focusedPath,
    onOpenFile,
    setPragmaOverrides,
    setFocusedPath,
  ]);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isOpen) return;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) => Math.min(prev + 1, commands.length - 1));
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) => Math.max(prev - 1, 0));
          break;
        case "Enter":
          e.preventDefault();
          if (commands[selectedIndex]) {
            commands[selectedIndex].action();
            onClose();
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, commands, selectedIndex, onClose]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div
      className={tw(
        "fixed inset-0 bg-black bg-opacity-50 flex items-start justify-center pt-20 z-[2000]",
      )}
    >
      <div
        className={tw("bg-white rounded-lg shadow-2xl w-full max-w-2xl mx-4")}
      >
        <div className={tw("p-4 border-b border-gray-200")}>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search files or type a command..."
            className={tw("w-full text-lg outline-none")}
          />
        </div>

        <div className={tw("max-h-96 overflow-y-auto")}>
          {commands.map((command, index) => (
            <div
              key={index}
              className={tw(
                `px-4 py-3 cursor-pointer border-b border-gray-100 last:border-b-0 ${
                  index === selectedIndex ? "bg-blue-50" : "hover:bg-gray-50"
                }`,
              )}
              onClick={() => {
                command.action();
                onClose();
              }}
            >
              <div className={tw("font-medium text-gray-900")}>
                {command.title}
              </div>
              <div className={tw("text-sm text-gray-600")}>
                {command.subtitle}
              </div>
            </div>
          ))}

          {commands.length === 0 && (
            <div className={tw("px-4 py-8 text-center text-gray-500")}>
              No results found
            </div>
          )}
        </div>

        <div
          className={tw(
            "px-4 py-2 bg-gray-50 text-xs text-gray-500 border-t border-gray-200",
          )}
        >
          Use ↑↓ to navigate, Enter to select, Esc to close
        </div>
      </div>
    </div>
  );
};

// ========== UI Components ==========

const TopBar = ({
  currentFile,
  connected,
  focusedPath,
  setFocusedPath,
  browsingDirectory,
  setBrowsingDirectory,
  isLoading,
  pragmaOverrides,
  setPragmaOverrides,
}) => {
  // Build breadcrumb
  const buildBreadcrumb = () => {
    const currentPath = browsingDirectory || currentFile;
    if (!currentPath) return null;

    const parts = splitPath(currentPath);
    const isBrowsingDirectory = !!browsingDirectory;

    return (
      <div className={tw("flex items-center text-sm font-mono")}>
        <button
          onClick={() => setBrowsingDirectory("/")}
          className={tw(
            "px-1 py-0.5 rounded transition-colors hover:bg-gray-200",
          )}
          title="Browse root directory"
        >
          root
        </button>

        {parts.map((part, index) => {
          const isLastPart = index === parts.length - 1;
          const isFile = !isBrowsingDirectory && isLastPart;
          const pathSegments = parts.slice(0, index + 1);
          const segmentPath = pathSegments.join("/");
          const segmentPathWithSlash = segmentPath + "/";

          // Check if this item is focused
          const isFocused = isFile
            ? focusedPath === currentFile
            : focusedPath === segmentPathWithSlash;

          return (
            <React.Fragment key={index}>
              <span className={tw("mx-1 text-gray-500")}>/</span>
              <button
                onClick={() => {
                  if (isFile) {
                    // Files toggle focus
                    setFocusedPath(isFocused ? null : segmentPath);
                  } else {
                    // Directories open browser
                    setBrowsingDirectory(segmentPathWithSlash);
                  }
                }}
                className={tw(
                  `px-1 py-0.5 rounded transition-colors hover:bg-gray-200 ${isFocused ? "font-bold" : ""}`,
                )}
                title={
                  isFile
                    ? isFocused
                      ? "Click to unfocus"
                      : "Click to focus"
                    : "Browse this directory"
                }
              >
                {isFocused && <span className={tw("mr-1")}>📌</span>}
                {part}
              </button>
            </React.Fragment>
          );
        })}
        {isBrowsingDirectory && (
          <span className={tw("mx-1 text-gray-500")}>/</span>
        )}
      </div>
    );
  };

  return (
    <div
      className={tw(
        "fixed top-0 left-0 right-0 h-10 bg-white border-b border-gray-300 flex items-center px-5 z-[1000] font-sans text-sm",
      )}
    >
      <div className={tw("flex-1")}>{buildBreadcrumb()}</div>

      {/* Pragma controls */}
      {currentFile && (
        <div className={tw("flex items-center gap-2 mr-4")}>
          <button
            onClick={() =>
              setPragmaOverrides((prev) => ({
                ...prev,
                hideCode: !prev.hideCode,
              }))
            }
            className={tw(
              `px-2 py-1 text-xs rounded border transition-colors ${
                pragmaOverrides.hideCode
                  ? "bg-red-100 border-red-300 text-red-700"
                  : "bg-green-100 border-green-300 text-green-700"
              }`,
            )}
            title={
              pragmaOverrides.hideCode
                ? "Code hidden - click to show"
                : "Code shown - click to hide"
            }
          >
            {pragmaOverrides.hideCode ? "○" : "●"} Code
          </button>
          <button
            onClick={() =>
              setPragmaOverrides((prev) => ({
                ...prev,
                hideProse: !prev.hideProse,
              }))
            }
            className={tw(
              `px-2 py-1 text-xs rounded border transition-colors ${
                pragmaOverrides.hideProse
                  ? "bg-red-100 border-red-300 text-red-700"
                  : "bg-green-100 border-green-300 text-green-700"
              }`,
            )}
            title={
              pragmaOverrides.hideProse
                ? "Prose hidden - click to show"
                : "Prose shown - click to hide"
            }
          >
            {pragmaOverrides.hideProse ? "○" : "●"} Prose
          </button>
        </div>
      )}

      {isLoading ? (
        <div className={tw("ml-2.5")}>
          <div
            className={tw(
              "animate-spin h-4 w-4 border-2 border-gray-300 border-t-gray-600 rounded-full",
            )}
          />
        </div>
      ) : (
        <div
          className={tw(
            `w-2 h-2 rounded-full ml-2.5 ${connected ? "bg-green-400" : "bg-red-400"}`,
          )}
          title={connected ? "Connected" : "Disconnected"}
        />
      )}
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
  }, [onMessage]);

  return connected;
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

  // Track if we're still initializing
  const [isInitialized, setIsInitialized] = useState(false);

  // Debounce timer for history updates
  const historyUpdateTimerRef = useRef(null);

  // Keep refs in sync
  useEffect(() => {
    currentFileRef.current = currentFile;

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

  // Handle WebSocket messages with RunVersion
  const handleWebSocketMessage = useCallback((message) => {
    // Ignore messages from old runs
    if (message.run && message.run < latestRunRef.current) {
      console.log(
        `Ignoring stale message from run ${message.run} (latest: ${latestRunRef.current})`,
      );
      return;
    }

    switch (message.type) {
      case "run-start":
        if (message.run) {
          // Update latest run version
          latestRunRef.current = message.run;

          // Check if this file change is relevant based on focus
          const file = message.file;
          const focus = focusedPathRef.current;

          let shouldProcess = false;

          // If no focus, process all changes
          if (!focus) {
            shouldProcess = true;
          }
          // If focused on a specific file, only process if it's that file
          else if (!focus.endsWith("/")) {
            shouldProcess = file === focus;
          }
          // If focused on a directory, check if file is within it
          else {
            const normalizedFile = file.startsWith("/") ? file.slice(1) : file;
            const normalizedFocus = focus.startsWith("/")
              ? focus.slice(1)
              : focus;
            shouldProcess = normalizedFile.startsWith(normalizedFocus);
          }

          if (shouldProcess) {
            // Clear tracked changed blocks
            changedBlocksRef.current.clear();

            // Handle the new manifest-based update
            if (message.blocks && message.dirty) {
              // We have block manifest - update intelligently
              setCurrentFile(file);
              setBlockResults((prev) => {
                const newResults = {};
                const blockSet = new Set(message.blocks);
                const dirtySet = new Set(message.dirty);

                // Process each block in the manifest
                for (const blockId of message.blocks) {
                  if (prev[blockId]) {
                    // Existing block
                    if (dirtySet.has(blockId)) {
                      // Mark as pending (will be re-executed)
                      newResults[blockId] = {
                        ...prev[blockId],
                        pending: true,
                        cache_hit: false,
                        // Keep all existing properties to maintain layout
                      };
                    } else {
                      // Keep unchanged, but clear any pending state
                      newResults[blockId] = {
                        ...prev[blockId],
                        pending: false,
                      };
                    }
                  } else {
                    // New block - create placeholder
                    newResults[blockId] = {
                      pending: true,
                      elements: [],
                      ok: true,
                    };
                  }
                }

                // Note: blocks not in the manifest are implicitly removed
                // by not including them in newResults

                return newResults;
              });
            } else {
              // Legacy behavior - clear blocks
              startTransition(() => {
                setCurrentFile(file);
                setBlockResults({});
              });
            }
          }
        }
        break;

      case "block-result":
        if (message.run && message.run === latestRunRef.current) {
          // Handle unchanged blocks (lightweight message)
          if (message.unchanged) {
            // Just clear pending state, keep existing results
            setBlockResults((prev) => ({
              ...prev,
              [message.block]: {
                ...prev[message.block],
                pending: false,
              },
            }));
          } else {
            // Track changed blocks
            if (message.content_changed) {
              changedBlocksRef.current.add(message.block);
            }

            // Update block result with new data
            setBlockResults((prev) => {
              return {
                ...prev,
                [message.block]: {
                  ok: message.ok,
                  stdout: message.stdout,
                  error: message.error,
                  showsVisual: message.showsVisual,
                  elements: message.elements || [],
                  cache_hit: message.cache_hit,
                  content_changed: message.content_changed,
                  pending: false, // Clear pending state
                },
              };
            });
          }
        }
        break;

      case "run-end":
        if (message.run && message.run === latestRunRef.current) {
          // Run completed
          console.log(`Run ${message.run} completed`);
          if (message.error) {
            console.error("Run error:", message.error);
          }

          // Check if we should auto-scroll to a changed block
          const changedBlocksList = Array.from(changedBlocksRef.current);

          // If exactly one block changed, scroll to it
          if (true && changedBlocksList.length === 1) {
            setTimeout(() => {
              const blockId = changedBlocksList[0];
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
        }
        break;

      case "reload":
        // General reload - refresh the page
        window.location.reload();
        break;

      default:
        console.warn("Unknown WebSocket message type:", message.type);
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
      wsRef.current.send(
        JSON.stringify({ type: "request-load", path: currentFile }),
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
        isLoading={isPending || isLoadingTree}
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
export { LiveServerApp, BlockRenderer, ColightVisual, CommandBar };
