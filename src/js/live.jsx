import React, { useEffect, useState, useCallback, useRef } from "react";
import ReactDOM from "react-dom/client";
import "../../packages/colight/src/colight/js/embed.js"; // Import Colight embed script

// Simple fuzzy search function
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

// Main SPA component
const LiveServerApp = () => {
  const [files, setFiles] = useState([]);
  const [currentFile, setCurrentFile] = useState(null);
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef(null);
  const wsRef = useRef(null);

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

  // Load file content
  const loadFile = async (path) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/content/${path}`);
      const html = await response.text();
      setContent(html);
      setCurrentFile(path);
      // Update URL without page reload
      window.history.pushState({}, "", `/${path}`);
    } catch (error) {
      console.error("Failed to load file:", error);
      setContent(
        '<div style="color: red; padding: 20px; font-family: monospace;">Failed to load file</div>',
      );
    } finally {
      setLoading(false);
    }
  };

  // WebSocket connection
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
          // If viewing this file, reload its content
          if (currentFile === data.path) {
            loadFile(data.path);
          } else if (!pinned) {
            // Auto-navigate to changed file
            loadFile(data.path);
          }
        }
      };

      ws.onerror = (error) => {
        console.error("LiveServer connection error:", error);
        setConnected(false);
      };

      ws.onclose = () => {
        console.log("LiveServer disconnected");
        setConnected(false);
        wsRef.current = null;
        // Reconnect after 2 seconds
        setTimeout(connect, 2000);
      };
    };

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [currentFile, pinned]);

  // Initial load
  useEffect(() => {
    loadFiles();

    // Check if URL has a file path
    const path = window.location.pathname.slice(1);
    if (path && path !== "/") {
      loadFile(path);
    }
  }, []);

  // Handle popstate for browser back/forward
  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname.slice(1);
      if (path && path !== "/") {
        loadFile(path);
      } else {
        // Navigate to index
        setCurrentFile(null);
        setContent("");
      }
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Cmd/Ctrl + K for search
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen(true);
        setSelectedIndex(0);
        setSearchQuery("");
        setTimeout(() => searchInputRef.current?.focus(), 0);
      }

      // Search modal keyboard handling
      if (searchOpen) {
        const filteredFiles = searchQuery
          ? fuzzySearch(searchQuery, files)
          : files;

        if (e.key === "Escape") {
          e.preventDefault();
          setSearchOpen(false);
          setSearchQuery("");
          setSelectedIndex(0);
        } else if (e.key === "ArrowDown") {
          e.preventDefault();
          setSelectedIndex((prev) =>
            Math.min(prev + 1, filteredFiles.length - 1),
          );
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          setSelectedIndex((prev) => Math.max(prev - 1, 0));
        } else if (e.key === "Enter" && filteredFiles.length > 0) {
          e.preventDefault();
          const selectedFile = filteredFiles[selectedIndex];
          if (selectedFile) {
            loadFile(selectedFile);
            setSearchOpen(false);
            setSearchQuery("");
            setSelectedIndex(0);
          }
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [searchOpen, searchQuery, files, selectedIndex]);

  // Reset selected index when search query changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  // Initialize Colight visualizations when content changes
  useEffect(() => {
    if (content && !loading) {
      // Wait for DOM to update
      setTimeout(() => {
        if (window.colight && window.colight.loadVisuals) {
          console.log("Initializing Colight visualizations");
          window.colight.loadVisuals();
        }
      }, 0);
    }
  }, [content, loading]);

  // Filter files based on search
  const filteredFiles = searchQuery ? fuzzySearch(searchQuery, files) : files;

  // Get display name for file
  const getDisplayName = (path) => {
    // Just show the file name without path for display
    const parts = path.split("/");
    return parts[parts.length - 1];
  };

  return (
    <>
      {/* Top bar */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: "40px",
          background: "#f5f5f5",
          borderBottom: "1px solid #ddd",
          display: "flex",
          alignItems: "center",
          padding: "0 20px",
          zIndex: 1000,
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          fontSize: "13px",
        }}
      >
        <button
          onClick={() => {
            setCurrentFile(null);
            setContent("");
            window.history.pushState({}, "", "/");
          }}
          style={{
            background: "transparent",
            border: "none",
            cursor: "pointer",
            padding: "4px",
            marginRight: "12px",
            fontSize: "18px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: "3px",
            transition: "background 0.2s",
          }}
          onMouseEnter={(e) => (e.target.style.background = "#e0e0e0")}
          onMouseLeave={(e) => (e.target.style.background = "transparent")}
          title="Home"
        >
          🏠
        </button>
        <div style={{ flex: 1, fontFamily: "monospace" }}>
          {currentFile || "No file selected"}
        </div>
        <button
          onClick={() => setPinned(!pinned)}
          style={{
            background: pinned ? "#333" : "transparent",
            color: pinned ? "white" : "#666",
            border: "1px solid #999",
            borderRadius: "3px",
            padding: "4px 8px",
            cursor: "pointer",
            marginRight: "10px",
            fontSize: "12px",
            fontFamily:
              '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          }}
        >
          📌 {pinned ? "" : "Pin"}
        </button>
        <div
          style={{
            width: "8px",
            height: "8px",
            borderRadius: "50%",
            backgroundColor: connected ? "#4ade80" : "#f87171",
            marginLeft: "10px",
          }}
        />
      </div>

      {/* Search modal */}
      {searchOpen && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: "rgba(0, 0, 0, 0.5)",
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "center",
            paddingTop: "100px",
            zIndex: 2000,
          }}
          onClick={() => {
            setSearchOpen(false);
            setSearchQuery("");
            setSelectedIndex(0);
          }}
        >
          <div
            style={{
              background: "white",
              borderRadius: "8px",
              boxShadow: "0 4px 20px rgba(0, 0, 0, 0.2)",
              width: "600px",
              maxHeight: "400px",
              display: "flex",
              flexDirection: "column",
              fontFamily:
                '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <input
              ref={searchInputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search files..."
              style={{
                padding: "16px",
                border: "none",
                borderBottom: "1px solid #eee",
                fontSize: "16px",
                outline: "none",
                borderRadius: "8px 8px 0 0",
                fontFamily:
                  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              }}
            />
            <div
              style={{
                flex: 1,
                overflow: "auto",
                maxHeight: "350px",
              }}
            >
              {filteredFiles.length === 0 ? (
                <div
                  style={{
                    padding: "20px",
                    textAlign: "center",
                    color: "#666",
                    fontFamily:
                      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
                  }}
                >
                  No files found
                </div>
              ) : (
                filteredFiles.map((file, index) => (
                  <div
                    key={file}
                    onClick={() => {
                      loadFile(file);
                      setSearchOpen(false);
                      setSearchQuery("");
                      setSelectedIndex(0);
                    }}
                    style={{
                      padding: "12px 16px",
                      cursor: "pointer",
                      borderBottom: "1px solid #f0f0f0",
                      background: index === selectedIndex ? "#e3f2fd" : "white",
                      display: "flex",
                      alignItems: "center",
                      fontFamily: "monospace",
                      fontSize: "14px",
                      transition: "background 0.1s ease",
                    }}
                    onMouseEnter={() => setSelectedIndex(index)}
                  >
                    <span style={{ color: "#666", marginRight: "8px" }}>
                      {file.includes("/")
                        ? file.substring(0, file.lastIndexOf("/") + 1)
                        : ""}
                    </span>
                    <span style={{ fontWeight: "bold" }}>
                      {getDisplayName(file)}
                    </span>
                  </div>
                ))
              )}
            </div>
            <div
              style={{
                padding: "8px 16px",
                borderTop: "1px solid #eee",
                fontSize: "12px",
                color: "#666",
                display: "flex",
                gap: "16px",
                fontFamily:
                  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              }}
            >
              <span>↑↓ Navigate</span>
              <span>↵ Open</span>
              <span>esc Close</span>
            </div>
          </div>
        </div>
      )}

      {/* Content area */}
      <div
        style={{
          marginTop: "40px",
          position: "relative",
        }}
      >
        {loading ? (
          <div
            style={{
              padding: "40px",
              textAlign: "center",
              fontFamily: "monospace",
              color: "#666",
            }}
          >
            Loading...
          </div>
        ) : content ? (
          <div dangerouslySetInnerHTML={{ __html: content }} />
        ) : (
          <div
            style={{
              padding: "40px",
              maxWidth: "800px",
              margin: "0 auto",
              fontFamily:
                '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            }}
          >
            <h1
              style={{
                fontSize: "32px",
                marginBottom: "24px",
                color: "#333",
              }}
            >
              Files
            </h1>

            {files.length === 0 ? (
              <p style={{ color: "#666" }}>No files found.</p>
            ) : (
              <>
                <p
                  style={{
                    marginBottom: "24px",
                    color: "#666",
                  }}
                >
                  {files.length} file{files.length !== 1 ? "s" : ""} available.
                  Press{" "}
                  <kbd
                    style={{
                      padding: "2px 6px",
                      background: "#f0f0f0",
                      border: "1px solid #ccc",
                      borderRadius: "3px",
                      fontSize: "12px",
                      fontFamily: "monospace",
                    }}
                  >
                    Cmd+K
                  </kbd>{" "}
                  to search.
                </p>

                <div
                  style={{
                    display: "grid",
                    gap: "8px",
                  }}
                >
                  {files.map((file) => (
                    <a
                      key={file}
                      href={`/${file}`}
                      onClick={(e) => {
                        e.preventDefault();
                        loadFile(file);
                      }}
                      style={{
                        display: "block",
                        padding: "12px 16px",
                        background: "#f8f8f8",
                        borderRadius: "6px",
                        textDecoration: "none",
                        color: "#333",
                        fontFamily: "monospace",
                        fontSize: "14px",
                        transition: "background 0.2s",
                        border: "1px solid #e0e0e0",
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.background = "#e8e8e8";
                        e.target.style.borderColor = "#d0d0d0";
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.background = "#f8f8f8";
                        e.target.style.borderColor = "#e0e0e0";
                      }}
                    >
                      {file}
                    </a>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </>
  );
};

// Mount the app
if (typeof window !== "undefined") {
  const root = document.getElementById("root");
  if (root) {
    ReactDOM.createRoot(root).render(<LiveServerApp />);
  }
}
