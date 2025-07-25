/**
 * Colight Embed Script
 */

import { render } from "./widget.jsx";
import { loadColightFile, parseColightScript } from "./format.js";
import * as globals from "./globals";

/**
 * Shows an error message in a container element
 *
 * @param {HTMLElement} element - Element to show error in
 * @param {string} message - Error message to display
 */
function showError(element, message) {
  element.innerHTML = `<div class="error" style="color: red; padding: 16px;">
    <h3>Failed to load visual</h3>
    <p>${message}</p>
  </div>`;
}

/**
 * Loads and renders a Colight visual into a container element
 *
 * @param {string|HTMLElement} container - CSS selector or element to render into
 * @param {string} url - URL to the .colight file
 * @param {Object} [options] - Additional options (reserved for future use)
 * @returns {Promise<void>}
 */
export async function loadVisual(container, url) {
  try {
    // Resolve the container element
    const containerElement =
      typeof container === "string"
        ? document.querySelector(container)
        : container;

    if (!containerElement) {
      throw new Error(`Container not found: ${container}`);
    }
    render(containerElement, await loadColightFile(url));
  } catch (error) {
    console.error("Failed to load visual:", error);

    const element =
      typeof container === "string"
        ? document.querySelector(container)
        : container;

    if (element) {
      showError(element, error.message);
    }
  }
}
/**
 * Loads Colight visuals on the page or within a specific container
 *
 * @param {Object} [options] - Configuration options
 * @param {Element} [options.root=document] - Root element to search within
 * @returns {Promise<Array<Element>>} - Array of elements where visuals were loaded
 */
export function loadVisuals(options = {}) {
  const { root = document } = options;

  if (!root) {
    console.error("Root element not found");
    return Promise.resolve([]);
  }

  const loadPromises = [];
  const loadedElements = [];

  // Find all script elements with type="application/x-colight"
  const scripts = root.querySelectorAll('script[type="application/x-colight"]');

  scripts.forEach((element) => {
    // Skip if already processed
    if (element.dataset.colightLoaded === "true") return;

    try {
      // Parse the embedded colight data
      const data = parseColightScript(element);

      // Create a div to render the visual
      const visualDiv = document.createElement("div");
      visualDiv.className = "colight-embed";
      visualDiv.dataset.colightLoaded = "true";
      element.parentNode.insertBefore(visualDiv, element.nextSibling);
      render(visualDiv, data);
      element.dataset.colightLoaded = "true";
      loadedElements.push(element);
    } catch (error) {
      console.error("Error loading embedded visual:", error);
      console.error("Stack trace:", error.stack);
    }
  });

  // Find all elements with class="colight-embed" that haven't been loaded
  const embeds = root.querySelectorAll(
    ".colight-embed:not([data-colight-loaded])",
  );

  embeds.forEach((element) => {
    const src = element.getAttribute("data-src");
    if (src) {
      const promise = loadVisual(element, src)
        .then(() => {
          // Mark as loaded to avoid re-processing
          element.dataset.colightLoaded = "true";
          loadedElements.push(element);
        })
        .catch((error) =>
          console.error(`Error loading visual from ${src}:`, error),
        );

      loadPromises.push(promise);
    }
  });

  return Promise.all(loadPromises).then(() => loadedElements);
}
/**
 * Initialize Colight
 * Auto-discover visuals when the DOM is loaded
 */
export function autoLoad() {
  if (typeof document !== "undefined") {
    globals.colight.loadVisual = loadVisual;
    globals.colight.loadVisuals = loadVisuals;
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => loadVisuals());
    } else {
      loadVisuals();
    }
  }
}

autoLoad();
