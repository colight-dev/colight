import { useCallback, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";

/**
 * Parse the current route into navigation state
 */
export function parseRoute(path) {
  // Normalize path - remove leading/trailing slashes for consistency
  const normalizedPath = path?.replace(/^\/+|\/+$/g, "") || "";

  if (normalizedPath === "") {
    // Root directory
    return {
      type: "directory",
      path: "/",
      displayPath: "/",
      segments: [],
      file: null,
      directory: "/",
    };
  }

  // Check if it's a directory (ends with /)
  if (path.endsWith("/")) {
    const segments = normalizedPath.split("/").filter(Boolean);
    return {
      type: "directory",
      path: normalizedPath + "/",
      displayPath: normalizedPath + "/",
      segments,
      file: null,
      directory: normalizedPath + "/",
    };
  }

  // It's a file
  const segments = normalizedPath.split("/").filter(Boolean);
  const dirSegments = segments.slice(0, -1);

  return {
    type: "file",
    path: normalizedPath,
    displayPath: normalizedPath,
    segments,
    file: normalizedPath,
    directory: dirSegments.length > 0 ? dirSegments.join("/") + "/" : "/",
  };
}

/**
 * Convert any navigation request to a proper route
 */
function normalizeNavigationPath(path) {
  if (!path || path === "/") {
    return "/"; // Root
  }

  // Remove leading slash for processing
  let normalized = path.startsWith("/") ? path.slice(1) : path;

  // Ensure directories end with /
  if (!normalized.includes(".") && !normalized.endsWith("/")) {
    normalized += "/";
  }

  return "/" + normalized;
}

/**
 * Hook for navigation functionality
 */
export function useNavigation() {
  const navigate = useNavigate();
  const params = useParams();
  const loadingFileRef = useRef(null);

  // Parse current route
  const routePath = params["*"] || "";
  const navState = parseRoute(routePath);

  // Navigation function
  const navigateTo = useCallback(
    (path) => {
      const normalized = normalizeNavigationPath(path);

      // Track what we're loading
      if (normalized.endsWith(".py")) {
        loadingFileRef.current = normalized.substring(1); // Remove leading /
      } else if (normalized !== "/" && !normalized.endsWith("/")) {
        loadingFileRef.current = normalized.substring(1) + ".py";
      } else {
        loadingFileRef.current = null;
      }

      navigate(normalized);
    },
    [navigate],
  );

  return {
    navState,
    navigateTo,
    loadingFileRef,
  };
}
