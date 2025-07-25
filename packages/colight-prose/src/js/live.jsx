import { useEffect, useState, useCallback, useRef } from "react";
import ReactDOM from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
  useSearchParams,
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
import bylight from "./bylight.ts";
import { getClientId } from "./client-id.js";
import createLogger from "./logger.js";

// Custom hooks
import { useNavigation } from "./hooks/useNavigation.js";
import { useDirectoryTree } from "./hooks/useDirectoryTree.js";

// Context
import {
  WebSocketProvider,
  useWebSocket,
  useMessageHandler,
} from "./contexts/WebSocketContext.jsx";

// Components
import { DocumentViewer } from "./DocumentViewer.jsx";

const logger = createLogger("live");
window.setColightLogLevel("info");
// ========== Constants ==========

// ========== Constants ==========

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
        logger.error("Error parsing Colight visual:", error);
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
        logger.error("Error loading visual:", error);
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
  const isPendingReplacement = block.pendingReplacement;

  if (!block.elements || block.elements.length === 0) {
    // Only show placeholder if block truly has no content yet
    if (isPending && !isPendingReplacement) {
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
      {/* Show pending indicator overlay - more subtle for replacements */}
      {isPending && (
        <div
          className={tw(
            isPendingReplacement
              ? "absolute inset-0 bg-blue-100 bg-opacity-20 rounded-lg pointer-events-none z-10"
              : "absolute inset-0 bg-yellow-100 bg-opacity-30 rounded-lg pointer-events-none z-10 animate-pulse",
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

// ========== Main App Component ==========

const LiveServerApp = () => {
  // Navigation state
  const { navState, navigateTo, loadingFileRef } = useNavigation();

  // Directory tree management
  const { directoryTree, isLoadingTree, loadDirectoryTree } =
    useDirectoryTree(navState);

  // UI state
  const [pragmaOverrides, setPragmaOverrides] = useState({
    hideStatements: false,
    hideCode: false,
    hideProse: false,
    hideVisuals: false,
  });
  const [isCommandBarOpen, setIsCommandBarOpen] = useState(false);

  // Get WebSocket connection status
  const { connected } = useWebSocket();

  // Handle file-changed messages for navigation
  useMessageHandler({
    types: ["file-changed"],
    handler: (message) => {
      if (message.type === "file-changed") {
        // Navigate to changed file only if we're viewing the root directory
        if (navState.type === "directory" && navState.directory === "/") {
          logger.info(`File changed: ${message.path}, navigating from root...`);
          navigateTo(message.path);
        }
      }
    },
    priority: 10,
  });

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
      />

      <div className={tw("mt-10")}>
        {navState.type === "directory" ? (
          <div className={tw("max-w-4xl mx-auto px-4 py-8")}>
            <DirectoryBrowser
              directoryPath={navState.directory}
              tree={directoryTree}
              onSelectFile={navigateTo}
              onNavigateToDirectory={navigateTo}
              loadDirectoryTree={loadDirectoryTree}
            />
          </div>
        ) : navState.type === "file" ? (
          <DocumentViewer
            file={navState.file}
            pragmaOverrides={pragmaOverrides}
            navigateTo={navigateTo}
          />
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
    element: (
      <WebSocketProvider>
        <LiveServerApp />
      </WebSocketProvider>
    ),
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
