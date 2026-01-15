/**
 * VSCode Output Panel - React-based webview for displaying Colight visuals
 */
import { useState, useEffect, useCallback, useRef } from "react";
import ReactDOM from "react-dom/client";
import { DraggableViewer, parseColightData } from "../widget.jsx";
import { decodeBase64ToUint8Array, encodeBufferToBase64 } from "../base64.js";

// Get VSCode API
const vscode =
  typeof acquireVsCodeApi !== "undefined" ? acquireVsCodeApi() : null;

// Inject styles
const styleSheet = `
:root {
  --vscode-font-family: var(--vscode-editor-font-family, monospace);
  --vscode-font-size: var(--vscode-editor-font-size, 13px);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  font-family: var(--vscode-font-family);
  font-size: var(--vscode-font-size);
  background: var(--vscode-editor-background);
  color: var(--vscode-editor-foreground);
}

#panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-bottom: 1px solid var(--vscode-panel-border);
  background: var(--vscode-sideBar-background);
  position: sticky;
  top: 0;
  z-index: 100;
}

.mode-toggle {
  display: flex;
  gap: 4px;
}

.panel-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.connection-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 6px;
  border: 1px solid var(--vscode-panel-border);
  border-radius: 999px;
  font-size: 11px;
  color: var(--vscode-descriptionForeground);
  background: var(--vscode-editor-background);
}

.connection-status .status-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--vscode-charts-red, #f48771);
}

.connection-status.connected .status-dot {
  background: var(--vscode-charts-green, #89d185);
}

.connection-status.connected .status-label {
  color: var(--vscode-foreground);
}

.mode-toggle button,
.clear-btn {
  padding: 4px 12px;
  border: 1px solid var(--vscode-button-border, transparent);
  border-radius: 4px;
  background: var(--vscode-button-secondaryBackground);
  color: var(--vscode-button-secondaryForeground);
  cursor: pointer;
  font-size: 12px;
}

.mode-toggle button:hover,
.clear-btn:hover {
  background: var(--vscode-button-secondaryHoverBackground);
}

.mode-toggle button.active {
  background: var(--vscode-button-background);
  color: var(--vscode-button-foreground);
}

.mode-toggle button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

#output-container {
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.widget-entry {
  position: relative;
  border: 1px solid var(--vscode-panel-border);
  border-radius: 4px;
  overflow: hidden;
  background: var(--vscode-editor-background);
}

.widget-entry .close-btn {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 20px;
  height: 20px;
  padding: 0;
  border: none;
  border-radius: 4px;
  background: var(--vscode-button-secondaryBackground);
  color: var(--vscode-button-secondaryForeground);
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
  opacity: 0;
  transition: opacity 0.15s;
  z-index: 10;
}

.widget-entry:hover .close-btn {
  opacity: 1;
}

.widget-entry .close-btn:hover {
  background: var(--vscode-button-secondaryHoverBackground);
}

.widget-container {
  min-height: 50px;
}

.stdout-output {
  padding: 8px 12px;
  background: var(--vscode-textBlockQuote-background);
  border-bottom: 1px solid var(--vscode-panel-border);
  font-family: var(--vscode-editor-font-family);
  font-size: var(--vscode-editor-font-size);
  white-space: pre-wrap;
  word-break: break-word;
}

.error-output {
  padding: 8px 12px;
  background: var(--vscode-inputValidation-errorBackground);
  border: 1px solid var(--vscode-inputValidation-errorBorder);
  border-radius: 4px;
  font-family: var(--vscode-editor-font-family);
  font-size: var(--vscode-editor-font-size);
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--vscode-errorForeground);
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px 24px;
  text-align: center;
  color: var(--vscode-descriptionForeground);
}

.empty-state h3 {
  margin: 0 0 8px;
  font-weight: 500;
}

.empty-state p {
  margin: 0;
  font-size: 12px;
}

.empty-state kbd {
  display: inline-block;
  padding: 2px 6px;
  border: 1px solid var(--vscode-panel-border);
  border-radius: 3px;
  background: var(--vscode-button-secondaryBackground);
  font-family: var(--vscode-editor-font-family);
  font-size: 11px;
}
`;

// Inject styles on load
if (
  typeof document !== "undefined" &&
  !document.getElementById("colight-panel-styles")
) {
  const style = document.createElement("style");
  style.id = "colight-panel-styles";
  style.textContent = styleSheet;
  document.head.appendChild(style);
}

/**
 * Create experimental interface for bidirectional widget communication
 */
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
        message.buffers = buffers.map((b) => encodeBufferToBase64(b));
      }

      vscode?.postMessage(message);
      return Promise.resolve();
    },
  };
}

/**
 * Single widget display component
 */
function WidgetDisplay({ evalId, visualBase64, onRemove }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const widgetIdRef = useRef(null);

  useEffect(() => {
    try {
      const bytes = decodeBase64ToUint8Array(visualBase64);
      const parsed = parseColightData(bytes);

      // Get widget ID and inject experimental interface
      const widgetId = parsed.id;
      widgetIdRef.current = widgetId;

      if (widgetId) {
        parsed.experimental = createExperimental(widgetId);
      }

      setData(parsed);

      // Register widget with extension
      if (widgetId) {
        vscode?.postMessage({
          type: "register-widget",
          evalId,
          widgetId,
        });
      }
    } catch (err) {
      console.error("Failed to parse widget data:", err);
      setError(err.message);
    }
  }, [visualBase64, evalId]);

  const handleRemove = useCallback(() => {
    onRemove(evalId, widgetIdRef.current);
  }, [evalId, onRemove]);

  if (error) {
    return <div className="error-output">Failed to render: {error}</div>;
  }

  if (!data) {
    return <div className="widget-container">Loading...</div>;
  }

  return (
    <div className="widget-entry">
      <button className="close-btn" onClick={handleRemove} title="Remove">
        ×
      </button>
      <div className="widget-container">
        <DraggableViewer data={data} />
      </div>
    </div>
  );
}

/**
 * Main Output Panel App
 */
function OutputPanelApp() {
  const [mode, setMode] = useState("snapshot");
  const [widgets, setWidgets] = useState([]);
  const [connected, setConnected] = useState(false);
  const [stdout, setStdout] = useState({});
  const [errors, setErrors] = useState({});

  // Handle messages from extension
  useEffect(() => {
    const handleMessage = (event) => {
      const msg = event.data;

      switch (msg.type) {
        case "add-widget":
          setWidgets((prev) => {
            // In snapshot mode, replace all
            if (msg.mode === "snapshot") {
              return [{ evalId: msg.evalId, visual: msg.visual }];
            }
            // In log mode, prepend
            return [{ evalId: msg.evalId, visual: msg.visual }, ...prev];
          });
          break;

        case "remove-widget":
          setWidgets((prev) => prev.filter((w) => w.evalId !== msg.evalId));
          break;

        case "clear":
          setWidgets([]);
          setStdout({});
          setErrors({});
          break;

        case "set-mode":
          setMode(msg.mode);
          break;

        case "connection-state":
          setConnected(!!msg.connected);
          break;

        case "show-error":
          setErrors((prev) => ({ ...prev, [msg.evalId]: msg.error }));
          break;

        case "show-stdout":
          if (msg.stdout?.trim()) {
            setStdout((prev) => ({ ...prev, [msg.evalId]: msg.stdout }));
          }
          break;

        case "update_state":
          // Forward state updates to the widget instance
          if (msg.widgetId && msg.updates) {
            const instance = window.colight?.instances?.[msg.widgetId];
            if (instance?.updateWithBuffers) {
              const buffers = (msg.buffers || []).map(decodeBase64ToUint8Array);
              instance.updateWithBuffers(msg.updates, buffers);
            }
          }
          break;
      }
    };

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, []);

  // Notify extension that we're ready
  useEffect(() => {
    vscode?.postMessage({ type: "ready" });
  }, []);

  const handleModeChange = useCallback((newMode) => {
    setMode(newMode);
    vscode?.postMessage({ type: "set-mode", mode: newMode });
  }, []);

  const handleClear = useCallback(() => {
    vscode?.postMessage({ type: "clear" });
  }, []);

  const handleRemoveWidget = useCallback((evalId, widgetId) => {
    vscode?.postMessage({ type: "remove-widget", evalId, widgetId });
  }, []);

  return (
    <>
      {/* Header */}
      <div id="panel-header">
        <div className="mode-toggle">
          {["snapshot", "log", "document"].map((m) => (
            <button
              key={m}
              data-mode={m}
              onClick={() => m !== "document" && handleModeChange(m)}
              disabled={m === "document"}
              className={mode === m ? "active" : ""}
              title={m === "document" ? "Coming soon" : undefined}
            >
              {m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>

        <div className="panel-actions">
          {/* Connection status */}
          <div
            className={`connection-status ${connected ? "connected" : ""}`}
            title={
              connected ? "Eval server connected" : "Eval server disconnected"
            }
          >
            <span className="status-dot" />
            <span className="status-label">
              {connected ? "Connected" : "Disconnected"}
            </span>
          </div>

          <button className="clear-btn" onClick={handleClear} title="Clear all">
            Clear
          </button>
        </div>
      </div>

      {/* Content */}
      <div id="output-container">
        {widgets.length === 0 ? (
          <div className="empty-state">
            <h3>No Output Yet</h3>
            <p>
              Press <kbd>Cmd+Shift+Enter</kbd> to evaluate a cell to this panel
            </p>
          </div>
        ) : (
          widgets.map((widget) => (
            <div key={widget.evalId}>
              {/* Stdout */}
              {stdout[widget.evalId] && (
                <div className="stdout-output">{stdout[widget.evalId]}</div>
              )}

              {/* Error */}
              {errors[widget.evalId] && (
                <div className="error-output">{errors[widget.evalId]}</div>
              )}

              {/* Widget */}
              <WidgetDisplay
                evalId={widget.evalId}
                visualBase64={widget.visual}
                onRemove={handleRemoveWidget}
              />
            </div>
          ))
        )}
      </div>
    </>
  );
}

// Mount the app
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<OutputPanelApp />);
