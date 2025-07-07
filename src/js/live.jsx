import React, { useEffect, useState, useCallback, useRef } from "react";
import ReactDOM from "react-dom/client";
import { render as renderColight } from "../../packages/colight/src/colight/js/widget.jsx";
import { parseColightScript } from "../../packages/colight/src/colight/js/format.js";
import { tw, md } from "../../packages/colight/src/colight/js/api.jsx";
import "./bylight.js";
// ========== Utility Functions ==========

const fuzzySearch = (query, items) => {
  const q = query.toLowerCase();
  return items
    .map((item) => {
      const name = item.toLowerCase();
      let score = 0;
      let lastIndex = -1;

      for (const char of q) {
        const index = name.indexOf(char, lastIndex + 1);
        if (index === -1) return null;
        score += index === lastIndex + 1 ? 2 : 1;
        lastIndex = index;
      }

      return { item, score };
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score)
    .map(({ item }) => item);
};

const getDisplayName = (path) => {
  const parts = path.split("/");
  return parts[parts.length - 1];
};

// ========== Content Rendering Components ==========

const ColightVisual = ({ data }) => {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current && data) {
      try {
        const colightData = parseColightScript({ textContent: data });
        renderColight(containerRef.current, colightData);
      } catch (error) {
        console.error("Error rendering Colight visual:", error);
      }
    }
  }, [data]);

  return <div ref={containerRef} className="colight-embed mb-4" />;
};

const ElementRenderer = ({ element }) => {
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
          {element.type === "expression" && element.visual && (
            <ColightVisual data={element.visual} />
          )}
        </>
      );

    default:
      return null;
  }
};

const BlockRenderer = ({ block }) => {
  if (!block.elements || block.elements.length === 0) return null;

  // Group consecutive statement/expression elements
  const groupedElements = [];
  let currentCodeGroup = [];

  block.elements.forEach((element, idx) => {
    if (!element.show) return;

    if (element.type === "statement" || element.type === "expression") {
      currentCodeGroup.push(element);
    } else {
      // If we have accumulated code elements, add them as a group
      if (currentCodeGroup.length > 0) {
        groupedElements.push({
          type: "code-group",
          elements: currentCodeGroup,
        });
        currentCodeGroup = [];
      }
      // Add the non-code element
      groupedElements.push(element);
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
              {item.elements.map((el, elIdx) =>
                el.type === "expression" && el.visual ? (
                  <ColightVisual
                    key={`visual-${idx}-${elIdx}`}
                    data={el.visual}
                  />
                ) : null,
              )}
            </div>
          );
        } else {
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

const DocumentRenderer = React.memo(
  ({ doc }) => {
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
          <BlockRenderer key={block.id} block={block} />
        ))}
      </div>
    );
  },
  (prevProps, nextProps) => {
    // Custom comparison - only re-render if blocks actually changed
    if (!prevProps.doc || !nextProps.doc) return false;
    return prevProps.doc === nextProps.doc; // Reference equality check
  },
);

// ========== UI Components ==========

const TopBar = ({ currentFile, pinned, setPinned, connected, onHome }) => (
  <div
    className={tw(
      "fixed top-0 left-0 right-0 h-10 bg-gray-100 border-b border-gray-300 flex items-center px-5 z-[1000] font-sans text-sm",
    )}
  >
    <button
      onClick={onHome}
      className={tw(
        "bg-transparent border-none cursor-pointer p-1 mr-3 text-lg flex items-center justify-center rounded transition-colors hover:bg-gray-300",
      )}
      title="Home"
    >
      🏠
    </button>
    <div className={tw("flex-1 font-mono")}>
      {currentFile || "No file selected"}
    </div>
    <button
      onClick={() => setPinned(!pinned)}
      className={tw(
        `border border-gray-500 rounded px-2 py-1 cursor-pointer mr-2.5 text-xs font-sans ${
          pinned ? "bg-gray-800 text-white" : "bg-transparent text-gray-600"
        }`,
      )}
    >
      📌 {pinned ? "" : "Pin"}
    </button>
    <div
      className={tw(
        `w-2 h-2 rounded-full ml-2.5 ${connected ? "bg-green-400" : "bg-red-400"}`,
      )}
    />
  </div>
);

const SearchModal = ({ isOpen, onClose, files, onSelectFile }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef(null);

  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  const filteredFiles = searchQuery ? fuzzySearch(searchQuery, files) : files;

  const handleKeyDown = (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.min(prev + 1, filteredFiles.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === "Enter" && filteredFiles[selectedIndex]) {
      e.preventDefault();
      onSelectFile(filteredFiles[selectedIndex]);
      onClose();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className={tw(
        "fixed inset-0 bg-black bg-opacity-50 flex items-start justify-center pt-[100px] z-[2000]",
      )}
      onClick={onClose}
    >
      <div
        className={tw(
          "bg-white rounded-lg shadow-xl w-[600px] max-h-[400px] flex flex-col font-sans",
        )}
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={searchInputRef}
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Search files..."
          className={tw(
            "p-4 border-none border-b border-gray-200 text-base outline-none rounded-t-lg font-sans",
          )}
        />
        <div className={tw("flex-1 overflow-auto max-h-[350px]")}>
          {filteredFiles.length === 0 ? (
            <div className={tw("p-5 text-center text-gray-600 font-sans")}>
              No files found
            </div>
          ) : (
            filteredFiles.map((file, index) => (
              <div
                key={file}
                onClick={() => {
                  onSelectFile(file);
                  onClose();
                }}
                className={tw(
                  `py-3 px-4 cursor-pointer border-b border-gray-100 flex items-center font-mono text-sm transition-colors ${
                    index === selectedIndex ? "bg-blue-100" : "bg-white"
                  }`,
                )}
                onMouseEnter={() => setSelectedIndex(index)}
              >
                <span className={tw("text-gray-600 mr-2")}>
                  {file.includes("/")
                    ? file.substring(0, file.lastIndexOf("/") + 1)
                    : ""}
                </span>
                <span className={tw("font-bold")}>{getDisplayName(file)}</span>
              </div>
            ))
          )}
        </div>
        <div
          className={tw(
            "py-2 px-4 border-t border-gray-200 text-xs text-gray-600 flex gap-4 font-sans",
          )}
        >
          <span>↑↓ Navigate</span>
          <span>↵ Open</span>
          <span>esc Close</span>
        </div>
      </div>
    </div>
  );
};

const FileList = ({ files, onSelectFile }) => (
  <div className={tw("p-10 max-w-4xl mx-auto font-sans")}>
    <h1 className={tw("text-3xl mb-6 text-gray-800")}>Files</h1>

    {files.length === 0 ? (
      <p className={tw("text-gray-600")}>No files found.</p>
    ) : (
      <>
        <p className={tw("mb-6 text-gray-600")}>
          {files.length} file{files.length !== 1 ? "s" : ""} available. Press{" "}
          <kbd
            className={tw(
              "px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs font-mono",
            )}
          >
            Cmd+K
          </kbd>{" "}
          to search.
        </p>

        <div className={tw("grid gap-2")}>
          {files.map((file) => (
            <a
              key={file}
              href={`/${file}`}
              onClick={(e) => {
                e.preventDefault();
                onSelectFile(file);
              }}
              className={tw(
                "block py-3 px-4 bg-gray-50 rounded-md no-underline text-gray-800 font-mono text-sm transition-colors border border-gray-200 hover:bg-gray-200 hover:border-gray-300",
              )}
            >
              {file}
            </a>
          ))}
        </div>
      </>
    )}
  </div>
);

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
        setTimeout(connect, 1000);
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
  }, []); // Empty deps - no need to recreate on prop changes

  return connected;
};

// ========== CSS for animations ==========

const animationStyles = `
  .block-updated {
    animation: highlight-flash 1s ease-out;
  }

  @keyframes highlight-flash {
    0% { background-color: rgba(255, 235, 59, 0.3); }
    100% { background-color: transparent; }
  }
`;

// ========== Main App Component ==========

const LiveServerApp = () => {
  const [files, setFiles] = useState([]);
  const [currentFile, setCurrentFile] = useState(null);
  const [documentData, setDocumentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [fileVersions, setFileVersions] = useState({}); // Track versions for incremental updates

  // Refs for WebSocket callback
  const currentFileRef = useRef(null);
  const pinnedRef = useRef(false);
  const hasLoadedDataRef = useRef({}); // Track which files have been loaded

  // Keep refs in sync
  useEffect(() => {
    currentFileRef.current = currentFile;
  }, [currentFile]);

  useEffect(() => {
    pinnedRef.current = pinned;
  }, [pinned]);

  // Load file list
  const loadFiles = async () => {
    try {
      const response = await fetch("/api/files");
      const data = await response.json();
      setFiles(data.files || []);
    } catch (error) {
      console.error("Failed to load files:", error);
    }
  };

  // Load file content (now JSON with incremental update support)
  const loadFile = async (path, isUpdate = false) => {
    // Only show loading spinner when we have no document to display yet
    if (!isUpdate || !documentData) {
      setLoading(true);
    }
    try {
      const response = await fetch(`/api/document/${path}`);
      const doc = await response.json();

      if (doc.error) {
        throw new Error(doc.error);
      }

      // Check if this is an incremental update
      const changes = doc._changes;
      console.log("Document loaded:", {
        path,
        isUpdate,
        hasChanges: !!changes,
        changesFull: changes?.full,
        version: changes?.version,
        modified: changes?.modified?.length,
        removed: changes?.removed?.length,
        hasLoadedData: !!hasLoadedDataRef.current[path],
        conditions: {
          hasChanges: !!changes,
          isUpdate,
          hasLoadedData: !!hasLoadedDataRef.current[path],
          notFull: !changes?.full,
        },
      });

      if (
        changes &&
        isUpdate &&
        !changes.full &&
        hasLoadedDataRef.current[path]
      ) {
        // Handle incremental update only if we have existing data
        const currentVersion = fileVersions[path] || 0;

        // Ignore out-of-order updates
        if (changes.version <= currentVersion) {
          console.log("Ignoring out-of-order update", {
            newVersion: changes.version,
            currentVersion,
          });
          return;
        }

        console.log("Performing incremental update");

        // Apply incremental updates
        setDocumentData((current) => {
          console.log(
            "Current blocks:",
            current.blocks.map((b) => ({ id: b.id, line: b.line })),
          );

          // Create a map of modified blocks for quick lookup
          const modifiedBlocksMap = new Map();
          doc.blocks.forEach((block) => {
            if (changes.modified.includes(block.id)) {
              modifiedBlocksMap.set(block.id, block);
            }
          });

          // Check if any blocks actually need updating
          let hasChanges = false;

          // Create new blocks array
          const updatedBlocks = current.blocks
            .filter((block) => {
              if (changes.removed?.includes(block.id)) {
                hasChanges = true;
                return false;
              }
              return true;
            })
            .map((block) => {
              if (modifiedBlocksMap.has(block.id)) {
                console.log("Updating block:", block.id);
                hasChanges = true;
                return modifiedBlocksMap.get(block.id);
              }
              return block;
            });

          console.log(
            "New blocks:",
            updatedBlocks.map((b) => ({ id: b.id, line: b.line })),
          );

          // If nothing changed, return the same object reference
          if (!hasChanges) {
            console.log("No actual changes, returning same document reference");
            return current;
          }

          // Only create new object if there were actual changes
          return {
            ...current,
            blocks: updatedBlocks,
          };
        });

        // Update version
        setFileVersions((prev) => ({ ...prev, [path]: changes.version }));

        // Highlight modified blocks after React renders
        setTimeout(() => {
          if (changes.modified) {
            console.log("Highlighting blocks:", changes.modified);
            changes.modified.forEach((blockId) => {
              const elem = document.querySelector(
                `[data-block-id="${blockId}"]`,
              );
              if (elem) {
                elem.classList.add("block-updated");
                setTimeout(() => elem.classList.remove("block-updated"), 1000);
              } else {
                console.warn("Could not find block to highlight:", blockId);
              }
            });

            // Scroll to the changed block if there's only one
            if (changes.modified.length === 1) {
              const elem = document.querySelector(
                `[data-block-id="${changes.modified[0]}"]`,
              );
              if (elem) {
                elem.scrollIntoView({ behavior: "smooth", block: "center" });
              }
            }
          }
        }, 50); // Small delay to ensure React has rendered
      } else if (
        changes &&
        isUpdate &&
        !changes.full &&
        !hasLoadedDataRef.current[path]
      ) {
        // First update after page load - we have changes but no previous data
        // Treat as full load but try to highlight the modified blocks
        console.log(
          "First update after page load - treating as full load with highlights",
        );

        setDocumentData(doc);
        setCurrentFile(path);
        hasLoadedDataRef.current[path] = true;

        // Update version
        if (changes?.version) {
          setFileVersions((prev) => ({ ...prev, [path]: changes.version }));
        }

        // After render, highlight the modified blocks
        setTimeout(() => {
          if (changes.modified) {
            console.log("Highlighting modified blocks:", changes.modified);
            changes.modified.forEach((blockId) => {
              const elem = document.querySelector(
                `[data-block-id="${blockId}"]`,
              );
              if (elem) {
                elem.classList.add("block-updated");
                setTimeout(() => elem.classList.remove("block-updated"), 1000);

                // Scroll to the first modified block
                if (changes.modified.indexOf(blockId) === 0) {
                  elem.scrollIntoView({ behavior: "smooth", block: "center" });
                }
              }
            });
          }
        }, 100);
      } else {
        // Full document load
        setDocumentData(doc);
        setCurrentFile(path);
        hasLoadedDataRef.current[path] = true;
        window.history.pushState({}, "", `/${path}`);

        // Update version if available
        if (changes?.version) {
          setFileVersions((prev) => ({ ...prev, [path]: changes.version }));
        }
      }
    } catch (error) {
      console.error("Failed to load file:", error);
      setDocumentData({
        error: error.message,
        blocks: [],
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle file changes from WebSocket
  const handleFileChange = useCallback((path) => {
    if (currentFileRef.current === path) {
      loadFile(path, true); // isUpdate = true
    } else if (!pinnedRef.current) {
      loadFile(path, true); // isUpdate = true
    }
  }, []);

  // Setup WebSocket connection
  const connected = useWebSocket(handleFileChange);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen(true);
      }
    };

    if (typeof document !== "undefined") {
      document.addEventListener("keydown", handleKeyDown);
      return () => document.removeEventListener("keydown", handleKeyDown);
    }
  }, []);

  // Inject animation styles
  useEffect(() => {
    const styleEl = document.createElement("style");
    styleEl.textContent = animationStyles;
    document.head.appendChild(styleEl);

    return () => styleEl.remove();
  }, []);

  // Initial load
  useEffect(() => {
    loadFiles();

    // Handle initial route
    const path = window.location.pathname.slice(1);
    if (path) {
      loadFile(path);
    }
  }, []);

  // Handle popstate (browser back/forward)
  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname.slice(1);
      if (path) {
        loadFile(path);
      } else {
        setCurrentFile(null);
        setDocumentData(null);
      }
    };

    if (typeof window !== "undefined") {
      window.addEventListener("popstate", handlePopState);
      return () => window.removeEventListener("popstate", handlePopState);
    }
  }, []);

  const handleHome = () => {
    setCurrentFile(null);
    setDocumentData(null);
    hasLoadedDataRef.current = {}; // Clear all loaded state
    window.history.pushState({}, "", "/");
  };

  return (
    <>
      <TopBar
        currentFile={currentFile}
        pinned={pinned}
        setPinned={setPinned}
        connected={connected}
        onHome={handleHome}
      />

      <SearchModal
        isOpen={searchOpen}
        onClose={() => setSearchOpen(false)}
        files={files}
        onSelectFile={loadFile}
      />

      <div className={tw("mt-10 relative")}>
        {currentFile && documentData ? (
          <>
            {loading && (
              <div
                className={tw(
                  "absolute inset-0 flex items-center justify-center bg-white/70 z-50",
                )}
              >
                <div className={tw("font-mono text-gray-600")}>Loading...</div>
              </div>
            )}
            <DocumentRenderer key={currentFile} doc={documentData} />
          </>
        ) : loading ? (
          <div className={tw("p-10 text-center font-mono text-gray-600")}>
            Loading...
          </div>
        ) : (
          <FileList files={files} onSelectFile={loadFile} />
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
