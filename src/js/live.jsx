import React, { useEffect, useState, useCallback, useRef, useTransition } from "react";
import ReactDOM from "react-dom/client";
import { DraggableViewer } from "../../packages/colight/src/colight/js/widget.jsx";
import { parseColightScript } from "../../packages/colight/src/colight/js/format.js";
import { tw, md } from "../../packages/colight/src/colight/js/api.jsx";
import { DirectoryBrowser } from "./DirectoryBrowser.jsx";
import "./bylight.js";

// ========== Constants ==========

const WEBSOCKET_RECONNECT_DELAY = 1000; // ms
const HISTORY_DEBOUNCE_DELAY = 300; // ms

// ========== Path Utilities ==========

const stripExt = (path) => path.replace(/\.py$/, "");
const splitPath = (path) => path.split("/").filter(Boolean);
const displayName = (path) => {
  const parts = splitPath(path);
  return parts[parts.length - 1] || "";
};

// ========== Document Processing ==========

const applyIncrementalUpdate = (currentDoc, newDoc, changes) => {
  // Create a map of modified blocks for quick lookup
  const modifiedBlocksMap = new Map();
  newDoc.blocks.forEach((block) => {
    if (changes.modified.includes(block.id)) {
      modifiedBlocksMap.set(block.id, block);
    }
  });

  // Check if any blocks actually need updating
  let hasChanges = false;

  // Create new blocks array
  const updatedBlocks = currentDoc.blocks
    .filter((block) => {
      if (changes.removed?.includes(block.id)) {
        hasChanges = true;
        return false;
      }
      return true;
    })
    .map((block) => {
      if (modifiedBlocksMap.has(block.id)) {
        hasChanges = true;
        return modifiedBlocksMap.get(block.id);
      }
      return block;
    });

  // If nothing changed, return the same object reference
  if (!hasChanges) {
    return currentDoc;
  }

  // Only create new object if there were actual changes
  return {
    ...currentDoc,
    blocks: updatedBlocks,
  };
};

// ========== Content Rendering Components ==========

const ColightVisual = ({ data }) => {
  const [colightData, setColightData] = useState(null);

  useEffect(() => {
    if (data) {
      try {
        const parsed = parseColightScript({ textContent: data });
        setColightData(parsed);
      } catch (error) {
        console.error("Error parsing Colight visual:", error);
      }
    }
  }, [data]);

  if (!colightData) {
    return <div className="colight-embed mb-4" />;
  }

  return (
    <div className="colight-embed mb-4">
      <DraggableViewer data={colightData} />
    </div>
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

  block.elements.forEach((element) => {
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

const DocumentRenderer = ({ doc }) => {
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
  }

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
        Watching <span className={tw("font-bold")}>{watchingPath || "..."}</span>
      </div>
    </div>
  );
};

// ========== UI Components ==========

const TopBar = ({ currentFile, connected, focusedPath, setFocusedPath, browsingDirectory, setBrowsingDirectory, isLoading }) => {
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
          className={tw("px-1 py-0.5 rounded transition-colors hover:bg-gray-200")}
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
                className={tw(`px-1 py-0.5 rounded transition-colors hover:bg-gray-200 ${isFocused ? "font-bold" : ""}`)}
                title={isFile ? (isFocused ? "Click to unfocus" : "Click to focus") : "Browse this directory"}
              >
                {isFocused && <span className={tw("mr-1")}>📌</span>}
                {part}
              </button>
            </React.Fragment>
          );
        })}
        {isBrowsingDirectory && <span className={tw("mx-1 text-gray-500")}>/</span>}
      </div>
    );
  };
  
  return (
    <div
      className={tw(
        "fixed top-0 left-0 right-0 h-10 bg-gray-100 border-b border-gray-300 flex items-center px-5 z-[1000] font-sans text-sm",
      )}
    >
      <div className={tw("flex-1")}>
        {buildBreadcrumb()}
      </div>
      {isLoading ? (
        <div className={tw("ml-2.5")}>
          <div className={tw("animate-spin h-4 w-4 border-2 border-gray-300 border-t-gray-600 rounded-full")} />
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
        window.history.replaceState({}, "", url.toString());
      } else if (window.location.pathname !== "/") {
        // No file selected, go to root
        const url = new URL(window.location);
        url.pathname = "/";
        if (focusedPath) {
          url.searchParams.set("focus", focusedPath);
        } else {
          url.searchParams.delete("focus");
        }
        window.history.replaceState({}, "", url.toString());
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
        const pendingVersion = pendingVersionRef.current[path] || 0;

        // Ignore out-of-order updates or updates that arrive during pending updates
        if (changes.version <= currentVersion || changes.version <= pendingVersion) {
          console.log("Ignoring out-of-order or stale update", {
            newVersion: changes.version,
            currentVersion,
            pendingVersion,
          });
          return;
        }

        // Mark this version as pending
        pendingVersionRef.current[path] = changes.version;

        console.log("Performing incremental update");

        // Apply incremental updates
        startTransition(() => {
          setDocumentData((current) => {
            const updated = applyIncrementalUpdate(current, doc, changes);
            if (updated === current) {
              console.log("No actual changes, returning same document reference");
            }
            return updated;
          });
        });

        // Update version and clear pending
        setFileVersions((prev) => ({ ...prev, [path]: changes.version }));
        delete pendingVersionRef.current[path];
        
      } else  {

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
    } else {
      console.log(`Ignoring file change outside focus: ${path} (focus: ${focus})`);
    }
  }, []);

  // Setup WebSocket connection
  const connected = useWebSocket(handleFileChange);


  // Initial load
  useEffect(() => {

    // Handle initial route and focus state
    const path = window.location.pathname.slice(1);
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
      const path = window.location.pathname.slice(1);
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


  return (
    <>
      <TopBar
        currentFile={currentFile}
        connected={connected}
        focusedPath={focusedPath}
        setFocusedPath={setFocusedPath}
        browsingDirectory={browsingDirectory}
        setBrowsingDirectory={setBrowsingDirectory}
        isLoading={isPending || isLoadingTree}
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
          <DocumentRenderer key={currentFile} doc={documentData} />
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
