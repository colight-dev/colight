import React, { useEffect, useState, useCallback, useRef } from "react";
import ReactDOM from "react-dom/client";
import "../../packages/colight/src/colight/js/embed.js"; // Import Colight embed script
import { tw, md } from "../../packages/colight/src/colight/js/api.jsx";
import "./bylight.js"

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
          <pre className={tw("bg-gray-100 p-4 rounded-lg overflow-x-auto mb-4")}>
            <code className="language-python">{element.value}</code>
          </pre>
          {/* Render visual if it's an expression with visual data */}
          {element.type === "expression" && element.visual && element.showVisual && (
            <div className={tw("mb-4")}>
              {element.visual.format === "inline" ? (
                <script type="application/x-colight" data-size={element.visual.size}>
                  {element.visual.data}
                </script>
              ) : (
                <div
                  className="colight-embed"
                  data-src={element.visual.path}
                  data-size={element.visual.size}
                /> 
              )}
            </div>
          )}
        </>
      );

    case "error":
      return (
        <div
          className={tw(
            "bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg mb-4",
          )}
        >
          <pre>{element.value}</pre>
        </div>
      );

    default:
      return null;
  }
};

const FormRenderer = ({ form }) => {
  if (!form.elements || form.elements.length === 0) return null;

  return (
    <div
      className={tw(`form-${form.id} ${form.pragma.join(" ")}`)}
      data-line={form.line}
      data-form-id={form.id}
      data-has-expression={form.hasExpression}
      data-shows-visual={form.showsVisual}
    >
      {form.elements.map((element, idx) => (
        <ElementRenderer key={idx} element={element} />
      ))}
    </div>
  );
};

const DocumentRenderer = ({ doc }) => {

  const docRef = useRef();

  useEffect(() => {
    if (docRef.current) {
      window.bylight({target: docRef.current})
    }
  }, [docRef.current])

  useEffect(() => {
    // Initialize Colight visualizations after render
    setTimeout(() => {
      if (window.colight && window.colight.loadVisuals) {
        console.log("Initializing Colight visualizations");
        window.colight.loadVisuals();
      }
    }, 0);
  }, [doc]);

  if (!doc) return null;

  return (
    <div className={tw("max-w-4xl mx-auto px-4 py-8  [&_pre]:text-sm")} ref={docRef}>
      {doc.forms.map((form) => (
        <FormRenderer key={form.id} form={form} />
      ))}
    </div>
  );
};

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

// ========== Main App Component ==========

const LiveServerApp = () => {
  const [files, setFiles] = useState([]);
  const [currentFile, setCurrentFile] = useState(null);
  const [documentData, setDocumentData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);

  // Refs for WebSocket callback
  const currentFileRef = useRef(null);
  const pinnedRef = useRef(false);

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

  // Load file content (now JSON)
  const loadFile = async (path) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/document/${path}`);
      const doc = await response.json();

      if (doc.error) {
        throw new Error(doc.error);
      }

      setDocumentData(doc);
      setCurrentFile(path);
      window.history.pushState({}, "", `/${path}`);
    } catch (error) {
      console.error("Failed to load file:", error);
      setDocumentData({
        error: error.message,
        forms: [],
      });
    } finally {
      setLoading(false);
    }
  };

  // Handle file changes from WebSocket
  const handleFileChange = useCallback((path) => {
    if (currentFileRef.current === path) {
      loadFile(path);
    } else if (!pinnedRef.current) {
      loadFile(path);
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
        {loading ? (
          <div className={tw("p-10 text-center font-mono text-gray-600")}>
            Loading...
          </div>
        ) : currentFile && documentData ? (
          <DocumentRenderer doc={documentData} />
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
