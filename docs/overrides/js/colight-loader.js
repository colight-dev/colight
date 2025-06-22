// Colight visualization loader for MkDocs
// This script ensures colight visualizations are loaded on page navigation

(function () {
  document.addEventListener("DOMContentSwitch", window.colight.loadVisuals);
})();
