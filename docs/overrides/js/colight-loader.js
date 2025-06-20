// Colight visualization loader for MkDocs
// This script ensures colight visualizations are loaded on page navigation

(function () {
  // Function to load colight visualizations
  function loadColightVisuals() {
    // Check if colight.loadVisuals function is available
    if (window.colight && typeof window.colight.loadVisuals === "function") {
      console.log("Loading Colight visualizations...");
      window.colight.loadVisuals();
    } else {
      console.warn("Colight loadVisuals function not found, retrying...");
      // Retry after a short delay if the script hasn't loaded yet
      setTimeout(loadColightVisuals, 100);
    }
  }

  // Initial load
  document.addEventListener("DOMContentLoaded", loadColightVisuals);

  // Handle MkDocs Material instant navigation
  // This event fires when navigating between pages without full reload
  document.addEventListener("DOMContentSwitch", loadColightVisuals);

  // Also handle location changes for other navigation methods
  let lastLocation = location.href;
  new MutationObserver(() => {
    const currentLocation = location.href;
    if (currentLocation !== lastLocation) {
      lastLocation = currentLocation;
      // Small delay to ensure DOM is updated
      setTimeout(loadColightVisuals, 50);
    }
  }).observe(document.body, {
    childList: true,
    subtree: true,
  });

  // Fallback: observe for new colight-embed elements
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === 1) {
          // Element node
          // Check if it's a colight embed or contains one
          if (
            (node.classList && node.classList.contains("colight-embed")) ||
            (node.querySelector && node.querySelector(".colight-embed"))
          ) {
            loadColightVisuals();
            break;
          }
        }
      }
    }
  });

  // Start observing when DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      observer.observe(document.body, {
        childList: true,
        subtree: true,
      });
    });
  } else {
    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }
})();
