const detectStaticMode = () =>
  typeof window !== "undefined" && Boolean(window.__COLIGHT_STATIC_MODE__);

export const isStaticMode = detectStaticMode();

const detectBase = () => {
  if (typeof window === "undefined") return null;
  if (window.__COLIGHT_STATIC_BASE__) return window.__COLIGHT_STATIC_BASE__;
  if (isStaticMode && window.location.protocol === "file:") return "./api";
  if (isStaticMode) return "/api";
  return null;
};

const normalizeBase = (value) => {
  if (!value) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  return trimmed.endsWith("/") ? trimmed.slice(0, -1) : trimmed;
};

export const staticBase = normalizeBase(detectBase());

const joinPath = (path = "") => {
  const cleanPath = path ? path.replace(/^\//, "") : "";
  if (staticBase) {
    return cleanPath ? `${staticBase}/${cleanPath}` : staticBase;
  }
  return cleanPath ? `/api/${cleanPath}` : "/api";
};

export const getApiUrl = (path = "") => joinPath(path);

export const apiFetch = (path, options) => fetch(getApiUrl(path), options);

export const encodePathSegments = (path) =>
  path
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
