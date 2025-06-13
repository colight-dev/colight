/**
 * Colight Embed Script
 */

import { render } from './widget.jsx';
import { loadColightFile } from './format.js';

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
    const containerElement = typeof container === 'string'
      ? document.querySelector(container)
      : container;

    if (!containerElement) {
      throw new Error(`Container not found: ${container}`);
    }
    render(containerElement, await loadColightFile(url));
  } catch (error) {
    console.error('Failed to load visual:', error);

    const element = typeof container === 'string'
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
 * @param {string} [options.selector='.colight-embed'] - CSS selector for visual elements
 * @param {Function} [options.getSrc] - Function to extract source URL from element
 * @returns {Promise<Array<Element>>} - Array of elements where visuals were loaded
 */
export function loadVisuals(options = {}) {
  const {
    root = document,
    selector = '.colight-embed',
    getSrc = (element) => element.getAttribute('data-src') || element.getAttribute('href')
  } = options;

  if (!root) {
    console.error('Root element not found');
    return Promise.resolve([]);
  }

  const loadPromises = [];
  const loadedElements = [];

  // Find all elements with the specified selector
  const elements = root.querySelectorAll(selector);

  elements.forEach(element => {
    // Skip if already processed
    if (element.dataset.colightLoaded === 'true') return;

    // Get the source using the provided getter function
    const src = getSrc(element);
    if (src) {
      const promise = loadVisual(element, src)
        .then(() => {
          // Mark as loaded to avoid re-processing
          element.dataset.colightLoaded = 'true';
          loadedElements.push(element);
        })
        .catch(error => console.error(`Error loading visual from ${src}:`, error));

      loadPromises.push(promise);
    }
  });

  return Promise.all(loadPromises).then(() => loadedElements);
}
/**
 * Initialize Colight
 * Auto-discover visuals when the DOM is loaded
 */
export function initialize() {
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => loadVisuals());
    } else {
      loadVisuals();
    }
  }
}

initialize();
