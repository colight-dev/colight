// Colight Output Panel - Webview Script

(function () {
  // @ts-ignore
  const vscode = acquireVsCodeApi();

  let currentMode = "snapshot";
  let widgetModule = null;
  let widgets = new Map(); // evalId -> { container, dispose, widgetId }

  // Ensure window.colight.instances exists for widget state tracking
  window.colight = window.colight || {};
  window.colight.instances = window.colight.instances || {};

  // DOM elements
  const container = document.getElementById("output-container");
  const modeButtons = document.querySelectorAll(".mode-toggle button");
  const clearBtn = document.getElementById("clear-btn");

  // Create experimental interface for a widget to enable bidirectional communication
  function createExperimental(widgetId) {
    return {
      invoke: (command, params, options = {}) => {
        const buffers = options.buffers || [];
        const message = {
          type: "widget-command",
          command,
          widgetId,
          params,
        };

        if (buffers.length) {
          message.buffers = buffers.map((b) => arrayBufferToBase64(b));
        }

        console.log("[Colight] Sending widget-command:", message);
        vscode.postMessage(message);
        return Promise.resolve();
      },
    };
  }

  // Load widget module
  async function loadWidgetModule() {
    if (widgetModule) return widgetModule;

    try {
      // @ts-ignore
      const uri = window.widgetModuleUri;
      widgetModule = await import(uri);
      console.log("Widget module loaded");
      return widgetModule;
    } catch (err) {
      console.error("Failed to load widget module:", err);
      return null;
    }
  }

  // Decode base64 to bytes
  function base64ToBytes(base64) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }

  // Create widget entry element
  function createWidgetEntry(evalId) {
    const entry = document.createElement("div");
    entry.className = "widget-entry";
    entry.dataset.evalId = evalId;

    const closeBtn = document.createElement("button");
    closeBtn.className = "close-btn";
    closeBtn.innerHTML = "&times;";
    closeBtn.title = "Remove";
    closeBtn.addEventListener("click", () => {
      vscode.postMessage({ type: "remove-widget", evalId });
    });

    const widgetContainer = document.createElement("div");
    widgetContainer.className = "widget-container";

    entry.appendChild(closeBtn);
    entry.appendChild(widgetContainer);

    return { entry, widgetContainer };
  }

  // Render widget into container
  // Returns { dispose, widgetId } on success
  async function renderWidget(visualBase64, targetContainer) {
    const module = await loadWidgetModule();
    if (!module) {
      targetContainer.innerHTML =
        '<div class="error-output">Failed to load widget renderer</div>';
      return null;
    }

    try {
      const bytes = base64ToBytes(visualBase64);

      // Parse the colight data - returns object with buffers already merged in
      const data = module.parseColightData(bytes);

      // Get widget ID from the data
      const widgetId = data.id;
      console.log("[Colight] Rendering widget with id:", widgetId);

      // Inject experimental interface for bidirectional communication
      if (widgetId) {
        data.experimental = createExperimental(widgetId);
        console.log("[Colight] Injected experimental interface for widget:", widgetId);
      }

      // Create render target
      const renderTarget = document.createElement("div");
      targetContainer.appendChild(renderTarget);

      // Render the widget - signature is (element, data, id)
      // The data object already contains buffers from parseColightData
      await module.render(renderTarget, data, widgetId);

      // Return dispose function and widget ID
      return {
        widgetId,
        dispose: () => {
          // Clean up widget instance from global registry
          if (widgetId && window.colight?.instances?.[widgetId]) {
            delete window.colight.instances[widgetId];
          }
          if (renderTarget._ReactRoot) {
            renderTarget._ReactRoot.unmount();
          }
        },
      };
    } catch (err) {
      console.error("Failed to render widget:", err);
      targetContainer.innerHTML = `<div class="error-output">Failed to render: ${err.message}</div>`;
      return null;
    }
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  // Add widget to container
  async function addWidget(evalId, visualBase64, mode) {
    // In snapshot mode, clear existing widgets first
    if (mode === "snapshot") {
      clearWidgets();
    }

    // Remove empty state if present
    const emptyState = container.querySelector(".empty-state");
    if (emptyState) {
      emptyState.remove();
    }

    const { entry, widgetContainer } = createWidgetEntry(evalId);

    // Add to DOM
    if (mode === "snapshot" || container.children.length === 0) {
      container.appendChild(entry);
    } else {
      // Prepend in log mode
      container.insertBefore(entry, container.firstChild);
    }

    // Render the widget
    const result = await renderWidget(visualBase64, widgetContainer);

    widgets.set(evalId, {
      container: entry,
      dispose: result?.dispose,
      widgetId: result?.widgetId,
    });
  }

  // Remove widget
  function removeWidget(evalId) {
    const widget = widgets.get(evalId);
    if (widget) {
      if (widget.dispose) {
        try {
          widget.dispose();
        } catch (e) {
          console.error("Error disposing widget:", e);
        }
      }
      widget.container.remove();
      widgets.delete(evalId);
    }

    // Show empty state if no widgets
    if (widgets.size === 0) {
      showEmptyState();
    }
  }

  // Clear all widgets
  function clearWidgets() {
    for (const [evalId, widget] of widgets) {
      if (widget.dispose) {
        try {
          widget.dispose();
        } catch (e) {
          console.error("Error disposing widget:", e);
        }
      }
      widget.container.remove();
    }
    widgets.clear();
    showEmptyState();
  }

  // Show empty state
  function showEmptyState() {
    if (container.querySelector(".empty-state")) return;

    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = `
      <h3>No Output Yet</h3>
      <p>Press <kbd>Cmd+Shift+Enter</kbd> to evaluate a cell to this panel</p>
    `;
    container.appendChild(empty);
  }

  // Set mode
  function setMode(mode) {
    currentMode = mode;
    modeButtons.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });
  }

  // Show error
  function showError(evalId, error) {
    const existingEntry = container.querySelector(`[data-eval-id="${evalId}"]`);
    if (existingEntry) {
      const errorDiv = document.createElement("div");
      errorDiv.className = "error-output";
      errorDiv.textContent = error;
      existingEntry.insertBefore(errorDiv, existingEntry.firstChild);
    } else {
      // Create standalone error entry
      const entry = document.createElement("div");
      entry.className = "widget-entry";
      entry.dataset.evalId = evalId;

      const errorDiv = document.createElement("div");
      errorDiv.className = "error-output";
      errorDiv.textContent = error;
      entry.appendChild(errorDiv);

      if (currentMode === "snapshot") {
        clearWidgets();
        container.appendChild(entry);
      } else {
        container.insertBefore(entry, container.firstChild);
      }
    }
  }

  // Show stdout
  function showStdout(evalId, stdout) {
    const existingEntry = container.querySelector(`[data-eval-id="${evalId}"]`);
    if (existingEntry) {
      const stdoutDiv = document.createElement("div");
      stdoutDiv.className = "stdout-output";
      stdoutDiv.textContent = stdout;
      existingEntry.insertBefore(stdoutDiv, existingEntry.firstChild);
    }
  }

  // Handle messages from extension
  window.addEventListener("message", async (event) => {
    const msg = event.data;

    switch (msg.type) {
      case "add-widget":
        await addWidget(msg.evalId, msg.visual, msg.mode || currentMode);
        break;

      case "remove-widget":
        removeWidget(msg.evalId);
        break;

      case "clear":
        clearWidgets();
        break;

      case "set-mode":
        setMode(msg.mode);
        break;

      case "show-error":
        showError(msg.evalId, msg.error);
        break;

      case "show-stdout":
        showStdout(msg.evalId, msg.stdout);
        break;

      case "update_state":
        // Forward state updates from Python to the widget instance
        console.log("[Colight] Received update_state:", msg);
        if (msg.widgetId && msg.updates) {
          const instance = window.colight?.instances?.[msg.widgetId];
          console.log("[Colight] Widget instance lookup:", msg.widgetId, "found:", !!instance);
          console.log("[Colight] All instances:", Object.keys(window.colight?.instances || {}));
          if (instance && instance.updateWithBuffers) {
            const buffers = (msg.buffers || []).map(base64ToBytes);
            instance.updateWithBuffers(msg.updates, buffers);
            console.log("[Colight] Applied updates to widget");
          } else {
            console.warn("[Colight] Widget instance not found for update:", msg.widgetId);
          }
        }
        break;
    }
  });

  // Mode button handlers
  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.disabled) return;
      const mode = btn.dataset.mode;
      setMode(mode);
      vscode.postMessage({ type: "set-mode", mode });
    });
  });

  // Clear button handler
  clearBtn.addEventListener("click", () => {
    vscode.postMessage({ type: "clear" });
  });

  // Initialize
  showEmptyState();

  // Tell extension we're ready
  vscode.postMessage({ type: "ready" });
})();
