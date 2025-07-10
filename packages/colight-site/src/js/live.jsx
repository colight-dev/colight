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
import { parseColightScript } from "../../../colight/src/js/format.js";
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

const applyIncrementalUpdate = (currentDoc, newDoc, changes) => {
  // Quick check - if nothing changed, return same reference
  if (
    (!changes.modified || changes.modified.length === 0) &&
    (!changes.removed || changes.removed.length === 0) &&
    (!changes.moved || changes.moved.length === 0)
  ) {
    return currentDoc;
  }

  // Create maps for efficient lookups
  const oldBlocksMap = new Map(currentDoc.blocks.map((b) => [b.id, b]));
  const removedSet = new Set(changes.removed || []);
  const modifiedSet = new Set(changes.modified || []);

  // Build the new blocks array respecting the order from newDoc
  // This handles: new blocks, moved blocks, and maintains correct order
  const updatedBlocks = newDoc.blocks
    .map((newBlock) => {
      const blockId = newBlock.id;

      // Skip removed blocks (shouldn't be in newDoc, but be defensive)
      if (removedSet.has(blockId)) {
        return null;
      }

      // If it's modified or new, use the new block data
      if (modifiedSet.has(blockId) || !oldBlocksMap.has(blockId)) {
        return newBlock;
      }

      // For unmodified blocks, preserve the old reference for React optimization
      return oldBlocksMap.get(blockId);
    })
    .filter(Boolean); // Remove any nulls

  // Check if we actually made any changes
  const hasChanges =
    updatedBlocks.length !== currentDoc.blocks.length ||
    updatedBlocks.some((block, i) => block !== currentDoc.blocks[i]);

  if (!hasChanges) {
    return currentDoc;
  }

  // Return new document with updated blocks
  return {
    ...currentDoc,
    blocks: updatedBlocks,
  };
};

// ========== Content Rendering Components ==========

const ColightVisual = ({ data, dataRef }) => {
  const containerRef = useRef(null);
  const [colightData, setColightData] = useState(null);
  const [pendingColightData, setPendingColightData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadedId, setLoadedId] = useState(null);
  const [minHeight, setMinHeight] = useState(0);
  const keyRef = useRef(0);

  // Helper to convert blob to base64
  const blobToBase64 = async (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(",")[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  // Load visual data
  const loadVisualData = async () => {
    if (!dataRef || !dataRef.url) return;

    setIsLoading(true);
    try {
      const response = await fetch(dataRef.url);
      if (!response.ok) {
        throw new Error(`Failed to load visual: ${response.status}`);
      }
      const blob = await response.blob();
      // Check if parseColightScript supports ArrayBuffer/Uint8Array
      const arrayBuffer = await blob.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);

      // Try to use Uint8Array directly, fall back to base64 if needed
      let parsed;
      try {
        parsed = parseColightScript({ buffer: uint8Array });
      } catch (e) {
        // Fall back to base64 if direct buffer parsing fails
        const base64 = await blobToBase64(blob);
        parsed = parseColightScript({ textContent: base64 });
      }
      setPendingColightData(parsed);
      setLoadedId(dataRef.id);
    } catch (error) {
      console.error("Error loading visual:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (data) {
      // We have inline data - parse it directly
      try {
        const parsed = parseColightScript({ textContent: data });
        setColightData(parsed);
        setPendingColightData(null);
      } catch (error) {
        console.error("Error parsing Colight visual:", error);
      }
    } else if (dataRef && dataRef.id !== loadedId) {
      // Set up intersection observer for lazy loading
      const observer = new IntersectionObserver(
        (entries) => {
          if (entries[0].isIntersecting) {
            loadVisualData();
            observer.disconnect();
          }
        },
        { rootMargin: "100px" },
      );

      const element = containerRef.current;
      if (element) {
        observer.observe(element);
      }

      return () => {
        if (element) {
          observer.unobserve(element);
        }
        observer.disconnect();
      };
    }
  }, [data, dataRef, loadedId]);

  // Update the displayed visual when loading is complete
  useEffect(() => {
    if (!isLoading && pendingColightData) {
      console.log("Swapping visual data");
      setColightData(pendingColightData);
      setMinHeight(containerRef.current?.offsetHeight || 0);
      keyRef.current += 1;
      setPendingColightData(null);
    }
  }, [isLoading, pendingColightData]);

  // Show placeholder only if we have nothing to show yet
  if (!colightData && !isLoading) {
    return <div ref={containerRef} className="colight-embed mb-4" />;
  }

  return (
    <div
      ref={containerRef}
      style={{ minHeight }}
      className="colight-embed mb-4 relative"
    >
      {/* Show existing visual if we have one */}
      {colightData && (
        <DraggableViewer
          key={keyRef.current}
          data={{ ...colightData, onMount: () => setMinHeight(0) }}
        />
      )}

      {/* Show loading overlay when fetching new visual */}
      {isLoading && (
        <div
          className={tw(
            "absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center rounded",
          )}
        >
          <div
            className={tw(
              "animate-spin h-8 w-8 border-4 border-gray-300 border-t-gray-600 rounded-full",
            )}
          />
        </div>
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
          {element.type === "expression" &&
            (element.visual || element.visual_ref) &&
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
  if (!block.elements || block.elements.length === 0) return null;

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
      className={tw(`block-${block.id}`)}
      data-block-id={block.id}
      data-shows-visual={block.showsVisual}
    >
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
              key={idx}
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

const DocumentRenderer = ({ doc, pragmaOverrides }) => {
  const docRef = useRef();

  useEffect(() => {
    // Run bylight on the entire document after render
    if (docRef.current) {
      window.bylight({ target: docRef.current });
    }
  }, [doc]); // Re-run when doc changes

  if (!doc) return null;

  // Debug duplicate keys
  const blockIds = doc.blocks.map((b) => b.id);
  const duplicates = blockIds.filter(
    (id, index) => blockIds.indexOf(id) !== index,
  );
  if (duplicates.length > 0) {
    console.warn("Duplicate block IDs found:", duplicates);
    console.log("All block IDs:", blockIds);
  }

  return (
    <div
      ref={docRef}
      className={tw("max-w-4xl mx-auto px-4 py-8  [&_pre]:text-sm")}
    >
      {doc.blocks.map((block) => (
        <BlockRenderer
          key={block.id}
          block={block}
          pragmaOverrides={pragmaOverrides}
        />
      ))}
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

const useWebSocket = (onFileChange) => {
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
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("WebSocket message:", data);

        if (data.type === "file-changed" && data.path) {
          onFileChange(data.path);
        }
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
  }, [onFileChange]); // Empty deps - no need to recreate on prop changes

  return connected;
};

// ========== Main App Component ==========

const LiveServerApp = () => {
  const [currentFile, setCurrentFile] = useState(null);
  const [documentData, setDocumentData] = useState(null);
  const [fileVersions, setFileVersions] = useState({}); // Track versions for incremental updates
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
  const hasLoadedDataRef = useRef({}); // Track which files have been loaded
  const focusedPathRef = useRef(null); // Track focused path for WebSocket callback
  const pendingVersionRef = useRef({}); // Track pending versions to prevent race conditions

  // Track if we're still initializing
  const [isInitialized, setIsInitialized] = useState(false);

  // Debounce timer for history updates
  const historyUpdateTimerRef = useRef(null);

  // Keep refs in sync
  useEffect(() => {
    currentFileRef.current = currentFile;

    // Clean up pending versions when file changes
    if (!currentFile) {
      // Clear all pending versions when no file is selected
      pendingVersionRef.current = {};
    }

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

  // Fetch document from API
  const fetchDocument = async (path) => {
    const response = await fetch(`/api/document/${path}`);
    const doc = await response.json();

    if (doc.error) {
      throw new Error(doc.error);
    }

    return doc;
  };

  // Load file content (now JSON with incremental update support)
  const loadFile = async (path, isUpdate = false) => {
    try {
      const doc = await fetchDocument(path);

      // Check if this is an incremental update
      const changes = doc._changes;

      if (
        changes &&
        isUpdate &&
        !changes.full &&
        hasLoadedDataRef.current[path]
      ) {
        // Handle incremental update only if we have existing data
        const currentVersion = fileVersions[path] || 0;
        const pendingVersion = pendingVersionRef.current[path] || 0;

        // Ignore out-of-order updates or updates that arrive during pending updates
        if (
          changes.version <= currentVersion ||
          changes.version <= pendingVersion
        ) {
          return;
        }

        // Mark this version as pending
        pendingVersionRef.current[path] = changes.version;

        // Apply incremental updates
        startTransition(() => {
          setDocumentData((current) => {
            return applyIncrementalUpdate(current, doc, changes);
          });
        });

        // Update version and clear pending
        setFileVersions((prev) => ({ ...prev, [path]: changes.version }));
        delete pendingVersionRef.current[path];

        // Clean up old pending versions to prevent memory leak
        const currentPaths = new Set([currentFileRef.current]);
        Object.keys(pendingVersionRef.current).forEach((key) => {
          if (!currentPaths.has(key)) {
            delete pendingVersionRef.current[key];
          }
        });

        // If only one block was modified, scroll to it
        if (changes.modified && changes.modified.length === 1) {
          setTimeout(() => {
            const blockId = changes.modified[0];
            let element = document.querySelector(
              `[data-block-id="${blockId}"]`,
            );
            if (element) {
              element = element.querySelector(".colight-embed") || element;
              element.scrollIntoView({ behavior: "smooth", block: "center" });
            }
          }, 100); // Small delay to ensure DOM is updated
        }
      } else {
        startTransition(() => {
          setDocumentData(doc);
          setCurrentFile(path);
          hasLoadedDataRef.current[path] = true;
        });

        // Update version
        if (changes?.version) {
          setFileVersions((prev) => ({ ...prev, [path]: changes.version }));
        }
      }
    } catch (error) {
      console.error("Failed to load file:", error);
      startTransition(() => {
        setDocumentData({
          error: error.message,
          blocks: [],
        });
      });
    }
  };

  // Handle file changes from WebSocket
  const handleFileChange = useCallback((path) => {
    const focus = focusedPathRef.current;

    // If no focus, navigate to all changes
    if (!focus) {
      loadFile(path, true);
      return;
    }

    // If focused on a specific file, only update if it's that file
    if (!focus.endsWith("/")) {
      if (path === focus) {
        loadFile(path, true);
      }
      return;
    }

    // If focused on a directory, check if file is within it
    const normalizedPath = path.startsWith("/") ? path.slice(1) : path;
    const normalizedFocus = focus.startsWith("/") ? focus.slice(1) : focus;

    if (normalizedPath.startsWith(normalizedFocus)) {
      loadFile(path, true);
    }
  }, []);

  // Setup WebSocket connection
  const connected = useWebSocket(handleFileChange);

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
      loadFile(path);
    }

    // Mark as initialized after initial load
    setTimeout(() => setIsInitialized(true), 0);
  }, []);

  // Handle popstate (browser back/forward)
  useEffect(() => {
    const handlePopState = () => {
      const path = getFilePathFromUrl();
      const params = new URLSearchParams(window.location.search);
      const focus = params.get("focus");

      // Restore focus state
      setFocusedPath(focus);

      if (path) {
        loadFile(path);
      } else {
        startTransition(() => {
          setCurrentFile(null);
          setDocumentData(null);
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
          loadFile(path);
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
              loadFile(path);
              setBrowsingDirectoryState(null);
            }}
            onClose={() => setBrowsingDirectoryState(null)}
          />
        ) : currentFile && documentData ? (
          <DocumentRenderer
            key={currentFile}
            doc={documentData}
            pragmaOverrides={pragmaOverrides}
          />
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
