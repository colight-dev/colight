/**
 * Colight Embed Script
 * This script provides functionality for embedding Colight visualizations in web pages.
 */

import { renderData } from './widget.jsx';
import '../widget.css';

// Binary data delimiter - must match the one in Python code
const BINARY_DELIMITER = '\n---BINARY_DATA---\n';

// CSS is now bundled inline - no need for separate loading

/**
 * Loads a .colight file and parses it into JSON and binary buffers
 *
 * @param {string} url - URL to the .colight file
 * @returns {Promise<{data: Object, buffers: ArrayBuffer[]}>} Parsed data and buffers
 */
export async function loadColightFile(url) {
  try {
    // Fetch the file as ArrayBuffer
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }

    // Get the file content as ArrayBuffer
    const arrayBuffer = await response.arrayBuffer();
    const fileContent = new Uint8Array(arrayBuffer);

    // Convert to string to find the delimiter
    const decoder = new TextDecoder('utf-8');
    const contentStr = decoder.decode(fileContent);

    // Split at the delimiter
    const delimiterPosition = contentStr.indexOf(BINARY_DELIMITER);
    if (delimiterPosition === -1) {
      throw new Error('Invalid .colight file format: Missing delimiter');
    }

    // Extract the JSON header
    const jsonHeader = contentStr.substring(0, delimiterPosition);
    const data = JSON.parse(jsonHeader);

    // Extract binary buffers based on offsets
    const buffers = [];
    const bufferLayout = data.bufferLayout || { offsets: [], count: 0 };
    const binaryStartPos = delimiterPosition + BINARY_DELIMITER.length;

    // Extract each buffer using the specified offsets
    for (let i = 0; i < bufferLayout.count; i++) {
      const offset = bufferLayout.offsets[i];
      const nextOffset = (i < bufferLayout.count - 1)
        ? bufferLayout.offsets[i + 1]
        : bufferLayout.totalSize;

      const bufferSize = nextOffset - offset;
      const bufferStart = binaryStartPos + offset;
      const bufferEnd = bufferStart + bufferSize;

      // Extract the buffer from the file content
      const buffer = fileContent.slice(bufferStart, bufferEnd);
      buffers.push(buffer);
    }

    return { data, buffers };
  } catch (error) {
    console.error('Error loading .colight file:', error);
    throw error;
  }
}

/**
 * Loads and renders a Colight visualization into a container element
 *
 * @param {string|HTMLElement} container - CSS selector or element to render into
 * @param {string} url - URL to the .colight file
 * @param {Object} [options] - Additional options (reserved for future use)
 * @returns {Promise<void>}
 */
export async function loadVisual(container, url, options = {}) {
  try {
    // Resolve the container element
    const containerElement = typeof container === 'string'
      ? document.querySelector(container)
      : container;

    if (!containerElement) {
      throw new Error(`Container not found: ${container}`);
    }

    // CSS is now bundled inline with the script

    // Load the .colight file
    const { data, buffers } = await loadColightFile(url);

    // Render the visualization
    renderData(containerElement, data, buffers);
  } catch (error) {
    console.error('Failed to load visualization:', error);

    // Display error in the container
    if (typeof container === 'string') {
      const element = document.querySelector(container);
      if (element) {
        element.innerHTML = `<div class="error" style="color: red; padding: 16px;">
          <h3>Failed to load visualization</h3>
          <p>${error.message}</p>
        </div>`;
      }
    } else if (container instanceof HTMLElement) {
      container.innerHTML = `<div class="error" style="color: red; padding: 16px;">
        <h3>Failed to load visualization</h3>
        <p>${error.message}</p>
      </div>`;
    }
  }
}

/**
 * Automatically discovers and loads all Colight visualizations on the page or within a specific container
 *
 * @param {string|Element|Document} [root=document] - Root element to search within (useful for SPAs)
 * @returns {Promise<Array<Element>>} - Array of elements where visualizations were loaded
 */
export function loadVisuals(options = {}) {
  // Handle legacy usage where first argument was root
  if (typeof options === 'string' || options instanceof Element || options === document) {
    options = { root: options };
  }

  const {
    root = document,
    selector = '.colight-embed',
    getSrc = (element) => element.getAttribute('data-src') || element.getAttribute('href')
  } = options;

  const rootElement = typeof root === 'string'
    ? document.querySelector(root)
    : root;

  if (!rootElement) {
    console.error('Root element not found');
    return Promise.resolve([]);
  }

  const loadPromises = [];
  const loadedElements = [];

  // Find all elements with the specified selector
  const elements = rootElement.querySelectorAll(selector);

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
        .catch(error => console.error(`Error loading visualization from ${src}:`, error));

      loadPromises.push(promise);
    }
  });

  return Promise.all(loadPromises).then(() => loadedElements);
}

/**
 * Initialize Colight - automatically scan the page and set up listeners for SPA support
 * This is called automatically when the script loads
 *
 * @param {Object} [options] - Initialization options
 * @param {boolean} [options.observeMutations=true] - Whether to observe DOM mutations for SPA support
 */
export function initialize(options = {}) {
  const defaults = {
    observeMutations: true
  };

  const config = { ...defaults, ...options };

  // Auto-discover visualizations when the DOM is loaded
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => loadVisuals());
    } else {
      loadVisuals();
    }

    // Optional: Add MutationObserver to detect new content in SPAs
    if (config.observeMutations && window.MutationObserver) {
      const observer = new MutationObserver((mutations) => {
        // Check if any mutations added nodes that might contain visualizations
        const shouldScan = mutations.some(mutation =>
          mutation.type === 'childList' &&
          mutation.addedNodes.length > 0
        );

        if (shouldScan) {
          // Small delay to ensure DOM is stable
          setTimeout(() => loadVisuals(), 100);
        }
      });

      // Start observing with a slight delay to ensure the page is loaded
      setTimeout(() => {
        observer.observe(document.body, {
          childList: true,
          subtree: true
        });
      }, 500);
    }
  }
}

// Auto-initialize when the script loads
initialize();

// Export the additional API functions that aren't already exported
// (loadVisual and loadColightFile are already exported with their function declarations)
