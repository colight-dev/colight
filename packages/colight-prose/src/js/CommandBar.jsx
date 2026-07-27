import React, { useState, useEffect, useRef, useMemo } from "react";
import Fuse from "fuse.js";
import { tw } from "../../../colight/src/js/api.jsx";
import { getCommands } from "./commands.js";
import createLogger from "./logger.js";

const logger = createLogger("CommandBar");
const RECENT_FILES_KEY = "colight-recent-files";
const MAX_RECENT_FILES = 5;

// Helper to get recent file paths from localStorage
const getRecentPaths = () => {
  try {
    const stored = localStorage.getItem(RECENT_FILES_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (e) {
    logger.error("Failed to get recent paths from localStorage", e);
    return [];
  }
};

// Helper to update recent file paths
const updateRecentPaths = (path) => {
  try {
    const recent = getRecentPaths();
    const updated = [path, ...recent.filter((p) => p !== path)].slice(
      0,
      MAX_RECENT_FILES,
    );
    localStorage.setItem(RECENT_FILES_KEY, JSON.stringify(updated));
    logger.debug("Updated recent paths", updated);
    return updated;
  } catch (e) {
    logger.error("Failed to update recent paths in localStorage", e);
    return [];
  }
};

const CommandBar = ({
  isOpen,
  onClose,
  directoryTree,
  currentFile,
  onOpenFile,
  pragmaOverrides,
  setPragmaOverrides,
}) => {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [commands, setCommands] = useState([]);
  const inputRef = useRef(null);

  const allItems = useMemo(() => {
    const items = [];
    if (directoryTree !== null) {
      // Recursive function to extract all .py files and directories
      const extractItems = (nextItems, currentPath = "") => {
        if (!nextItems || !Array.isArray(nextItems)) return;

        nextItems.forEach((item) => {
          const fullPath =
            item.path ||
            (currentPath ? `${currentPath}/${item.name}` : item.name);

          if (item.type === "file") {
            items.push({
              type: "file",
              name: item.name,
              path: fullPath,
              relativePath: fullPath,
              // Add searchable terms
              searchTerms: [
                item.name.replace(".py", ""),
                item.name,
                fullPath,
                ...fullPath.split("/").filter(Boolean),
              ]
                .join(" ")
                .toLowerCase(),
            });
          } else if (item.type === "directory") {
            // Add the directory itself as a searchable item
            items.push({
              type: "directory",
              name: item.name,
              path: fullPath,
              relativePath: fullPath,
              // Add searchable terms for directories
              searchTerms: [
                item.name,
                fullPath,
                ...fullPath.split("/").filter(Boolean),
              ]
                .join(" ")
                .toLowerCase(),
            });

            // Recursively extract children
            if (item.children) {
              extractItems(item.children, fullPath);
            }
          }
        });
      };

      // Start extraction - handle both root with children or direct array
      if (directoryTree.children) {
        extractItems(directoryTree.children);
      } else if (Array.isArray(directoryTree)) {
        extractItems(directoryTree);
      } else {
        extractItems([directoryTree]);
      }
    }
    return items;
  }, [directoryTree]);

  const fuse = useMemo(() => {
    return new Fuse(allItems, {
      keys: ["searchTerms"],
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
  }, [allItems]);

  // Generate commands based on query
  useEffect(() => {
    if (!isOpen || !fuse) return;
    logger.debug("Generating commands for query:", query);
    const newCommands = [];
    const lowerQuery = query.toLowerCase().trim();
    const allCommands = getCommands({ pragmaOverrides, setPragmaOverrides });

    if (lowerQuery) {
      // Filter commands that match the query
      const matchingCommands = allCommands.filter(
        (cmd) =>
          cmd.title.toLowerCase().includes(lowerQuery) ||
          cmd.subtitle.toLowerCase().includes(lowerQuery) ||
          cmd.searchTerms.some((term) => term.includes(lowerQuery)),
      );
      newCommands.push(...matchingCommands);

      // File and directory search results
      const results = fuse.search(lowerQuery);
      logger.debug("Fuse search results:", results);
      const itemCommands = results.slice(0, 10).map((result) => ({
        type: result.item.type,
        title: result.item.name,
        subtitle:
          result.item.type === "directory"
            ? `📁 ${result.item.relativePath}`
            : result.item.relativePath,
        action: () => {
          if (result.item.type === "file") {
            updateRecentPaths(result.item.path);
          }
          onOpenFile(result.item.path);
        },
      }));
      newCommands.push(...itemCommands);
    } else {
      // Show recent files first when no query

      const recentPaths = getRecentPaths();
      const recentItems = recentPaths
        .map((path) => allItems.find((item) => item.path === path))
        .filter(Boolean)
        .map((item) => ({
          type: item.type,
          title: item.name,
          subtitle:
            item.type === "directory"
              ? `📁 ${item.relativePath} (recent)`
              : `${item.relativePath} (recent)`,
          action: () => {
            if (item.type === "file") {
              updateRecentPaths(item.path);
            }
            onOpenFile(item.path);
          },
        }));

      newCommands.push(...recentItems);

      // Then show all commands
      newCommands.push(...allCommands);
    }

    setCommands(newCommands);
    setSelectedIndex(0);
  }, [
    query,
    pragmaOverrides,
    currentFile,
    onOpenFile,
    setPragmaOverrides,
    isOpen,
    fuse,
  ]);

  // Track current file changes
  useEffect(() => {
    if (currentFile && currentFile.endsWith(".py")) {
      updateRecentPaths(currentFile);
    }
  }, [currentFile]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      logger.info("CommandBar opened");
      inputRef.current?.focus();
      setQuery("");
    }
  }, [isOpen]);

  // Handle keyboard navigation - attached to document when open
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((prev) =>
            commands.length > 0 ? (prev + 1) % commands.length : 0,
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((prev) =>
            commands.length > 0
              ? (prev - 1 + commands.length) % commands.length
              : 0,
          );
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

  if (!isOpen) return null;

  return (
    <div
      className={tw(
        `fixed inset-0 bg-black bg-opacity-50 z-50 flex items-start justify-center pt-20`,
      )}
      onClick={onClose}
    >
      <div
        className={tw(
          `bg-white rounded-lg shadow-2xl max-w-2xl w-full mx-4 max-h-[60vh] flex flex-col`,
        )}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input */}
        <div className={tw(`px-4 py-3 border-b`)}>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search files, directories or type '>' for commands..."
            className={tw(
              `w-full text-lg focus:outline-none placeholder-gray-400`,
            )}
          />
        </div>

        {/* Command List */}
        <div className={tw(`overflow-y-auto flex-1`)}>
          {commands.length > 0 ? (
            commands.map((cmd, index) => (
              <div
                key={index}
                className={tw(
                  `px-4 py-3 cursor-pointer transition-colors ${index === selectedIndex ? `bg-blue-50` : `hover:bg-gray-50`}`,
                )}
                onClick={() => {
                  cmd.action();
                  onClose();
                }}
                onMouseEnter={() => setSelectedIndex(index)}
              >
                <div className={tw(`font-medium text-gray-900`)}>
                  {cmd.title}
                </div>
                {cmd.subtitle && (
                  <div className={tw(`text-sm text-gray-500 mt-0.5`)}>
                    {cmd.subtitle}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className={tw(`px-4 py-8 text-center text-gray-400`)}>
              {query ? "No results found" : "Type to search..."}
            </div>
          )}
        </div>

        {/* Footer */}
        <div
          className={tw(
            `px-4 py-2 border-t text-xs text-gray-400 flex justify-between`,
          )}
        >
          <span>↑↓ Navigate</span>
          <span>⏎ Select</span>
          <span>ESC Close</span>
        </div>
      </div>
    </div>
  );
};

export default CommandBar;
