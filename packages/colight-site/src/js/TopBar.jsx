import React from "react";
import { tw } from "../../../colight/src/js/api.jsx";

// Helper function to split path into segments
const splitPath = (path) => path.split("/").filter(Boolean);

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
  // Build breadcrumb data
  const pathSegments = currentFile ? splitPath(currentFile) : [];

  const handleBreadcrumbClick = (index) => {
    if (index === -1) {
      // Root clicked
      setBrowsingDirectory(true);
      setFocusedPath(null);
    } else {
      // Directory segment clicked
      const newPath = pathSegments.slice(0, index + 1).join("/") + "/";
      setFocusedPath(newPath);
      setBrowsingDirectory(false);
    }
  };

  return (
    <div
      className={tw(
        `sticky px-4 py-2 bg-white shadow-sm flex items-center left-0 right-0 top-0 z-[1000] border-b`,
      )}
    >
      {/* Breadcrumb Navigation */}
      <div className={tw(`flex-1`)}>
        <div className={tw(`flex text-sm font-mono items-center`)}>
          <button
            onClick={() => handleBreadcrumbClick(-1)}
            className={tw(
              `px-1 py-0.5 transition-colors rounded hover:bg-gray-200`,
            )}
            title="Browse root directory"
          >
            root
          </button>

          {/* Show path segments if we have a current file */}
          {pathSegments.map((segment, index) => (
            <React.Fragment key={index}>
              <span className={tw(`text-gray-500 mx-1`)}>/</span>
              {index < pathSegments.length - 1 ? (
                <button
                  onClick={() => handleBreadcrumbClick(index)}
                  className={tw(
                    `px-1 py-0.5 transition-colors rounded hover:bg-gray-200`,
                  )}
                  title={`Browse ${segment} directory`}
                >
                  {segment}
                </button>
              ) : (
                <button
                  onClick={() => setFocusedPath(currentFile)}
                  className={tw(
                    `px-1 py-0.5 transition-colors rounded`,
                    focusedPath === currentFile
                      ? `bg-blue-100 text-blue-700`
                      : `hover:bg-gray-200`,
                  )}
                  title={
                    focusedPath === currentFile
                      ? "File is focused (click to unfocus)"
                      : "Click to focus"
                  }
                >
                  {segment}
                </button>
              )}
            </React.Fragment>
          ))}

          {/* Focus indicator when browsing directory */}
          {browsingDirectory && focusedPath && (
            <>
              <span className={tw(`text-gray-500 mx-1`)}>→</span>
              <span className={tw(`text-blue-700 text-sm`)}>
                (browsing {focusedPath})
              </span>
            </>
          )}
        </div>
      </div>

      {/* Pragma Controls - Only show when we have a file */}
      {currentFile && (
        <div className={tw(`flex gap-2 items-center mr-4`)}>
          <button
            onClick={() =>
              setPragmaOverrides((prev) => ({
                ...prev,
                hideCode: !prev.hideCode,
              }))
            }
            className={tw(
              `px-2 py-1 text-xs transition-colors border rounded`,
              pragmaOverrides.hideCode
                ? `bg-gray-100 text-gray-500 border-gray-300`
                : `text-green-700 bg-green-100 border-green-300`,
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
              `px-2 py-1 text-xs transition-colors border rounded`,
              pragmaOverrides.hideProse
                ? `bg-gray-100 text-gray-500 border-gray-300`
                : `text-green-700 bg-green-100 border-green-300`,
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

      {/* Connection Status */}
      <div
        className={tw(
          `h-2 w-2 rounded-full`,
          isLoading
            ? `bg-yellow-400`
            : connected
              ? `bg-green-400`
              : `bg-red-400`,
          `ml-2.5`,
        )}
        title={
          isLoading ? "Loading..." : connected ? "Connected" : "Disconnected"
        }
      />
    </div>
  );
};

export default TopBar;
