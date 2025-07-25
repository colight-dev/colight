/**
 * Client ID management for LiveServer.
 * Generates and persists a unique client ID for watch/unwatch protocol.
 */

/**
 * Generate a UUID v4
 * @returns {string} A UUID v4 string
 */
function generateUUID() {
  // Use crypto.randomUUID if available (modern browsers)
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }

  // Fallback implementation
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/**
 * Get or create a persistent client ID
 * @returns {string} The client ID
 */
export function getClientId() {
  const STORAGE_KEY = "colight-live-client-id";

  // Try to get existing ID from sessionStorage
  let clientId = sessionStorage.getItem(STORAGE_KEY);

  if (!clientId) {
    // Generate new ID
    clientId = generateUUID();
    sessionStorage.setItem(STORAGE_KEY, clientId);
  }

  return clientId;
}

/**
 * Clear the client ID (useful for testing)
 */
export function clearClientId() {
  const STORAGE_KEY = "colight-live-client-id";
  sessionStorage.removeItem(STORAGE_KEY);
}
