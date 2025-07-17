import React, { useState, useEffect, useRef } from "react";
import Fuse from "fuse.js";
import { tw } from "../../../colight/src/js/api.jsx";

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
  const fuse = useRef(null);

  // Initialize Fuse.js for file search
  useEffect(() => {
    if (directoryTree && directoryTree !== null) {
      const allFiles = [];

      // Recursive function to extract all .py files
      const extractFiles = (items, currentPath = "") => {
        if (!items || !Array.isArray(items)) return;

        items.forEach((item) => {
          if (item.type === "file") {
            // Use item.path if available, otherwise construct it
            const fullPath =
              item.path ||
              (currentPath ? `${currentPath}/${item.name}` : item.name);
            allFiles.push({
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
          } else if (item.type === "directory" && item.children) {
            // Use item.path if available for collapsed paths
            const newPath =
              item.path ||
              (currentPath ? `${currentPath}/${item.name}` : item.name);
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
        const results = fuse.current.search(lowerQuery);
        const fileCommands = results.slice(0, 10).map((result) => ({
          type: "file",
          title: result.item.name,
          subtitle: result.item.relativePath,
          action: () => onOpenFile(result.item.path),
        }));
        newCommands.push(...fileCommands);
      }
    } else {
      // Show all commands when no query
      newCommands.push(...allCommands);
    }

    setCommands(newCommands);
    setSelectedIndex(0);
  }, [query, pragmaOverrides, currentFile, onOpenFile, setPragmaOverrides]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
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
            placeholder="Search files or type '>' for commands..."
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
