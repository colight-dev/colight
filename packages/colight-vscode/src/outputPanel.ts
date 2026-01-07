import * as vscode from "vscode";
import { EvalServer, WidgetMessage } from "./evalServer";

export type PanelMode = "snapshot" | "log" | "document";

interface WidgetEntry {
  evalId: string;
  visual: string; // base64
  timestamp: number;
}

export class OutputPanel {
  private static instance: OutputPanel | null = null;

  private panel: vscode.WebviewPanel | null = null;
  private mode: PanelMode = "snapshot";
  private widgets: WidgetEntry[] = [];
  private extensionUri: vscode.Uri;
  private evalServer: EvalServer;

  private widgetMessageDisposable: vscode.Disposable | null = null;

  private constructor(extensionUri: vscode.Uri) {
    this.extensionUri = extensionUri;
    this.evalServer = EvalServer.getInstance();
  }

  static getInstance(extensionUri: vscode.Uri): OutputPanel {
    if (!OutputPanel.instance) {
      OutputPanel.instance = new OutputPanel(extensionUri);
    }
    return OutputPanel.instance;
  }

  show(): void {
    if (this.panel) {
      this.panel.reveal(vscode.ViewColumn.Beside);
      return;
    }

    this.panel = vscode.window.createWebviewPanel(
      "colightOutput",
      "Colight Output",
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(this.extensionUri, "media"),
          vscode.Uri.joinPath(this.extensionUri, "dist"),
        ],
      }
    );

    this.panel.webview.html = this._getHtmlContent();

    // Handle messages from webview
    this.panel.webview.onDidReceiveMessage((msg) => {
      this._handleWebviewMessage(msg);
    });

    // Handle panel disposal
    this.panel.onDidDispose(() => {
      this.panel = null;
      this.widgetMessageDisposable?.dispose();
      this.widgetMessageDisposable = null;
    });

    // Forward widget messages from server to webview
    this.widgetMessageDisposable = this.evalServer.onWidgetMessage((msg) => {
      // Forward the message as-is, the webview knows how to handle it
      console.log("[Colight OutputPanel] Forwarding widget message to webview:", msg);
      this._sendToWebview(msg);
    });
  }

  addWidget(evalId: string, visualBase64: string): void {
    const entry: WidgetEntry = {
      evalId,
      visual: visualBase64,
      timestamp: Date.now(),
    };

    if (this.mode === "snapshot") {
      // Replace all widgets with just this one
      this.widgets = [entry];
    } else {
      // Log mode: prepend new widget
      this.widgets.unshift(entry);
    }

    this._sendToWebview({
      type: "add-widget",
      evalId,
      visual: visualBase64,
      mode: this.mode,
    });
  }

  removeWidget(evalId: string): void {
    this.widgets = this.widgets.filter((w) => w.evalId !== evalId);
    this._sendToWebview({
      type: "remove-widget",
      evalId,
    });
  }

  clear(): void {
    this.widgets = [];
    this._sendToWebview({ type: "clear" });
  }

  setMode(mode: PanelMode): void {
    this.mode = mode;
    this._sendToWebview({ type: "set-mode", mode });
  }

  showError(evalId: string, error: string): void {
    this._sendToWebview({
      type: "show-error",
      evalId,
      error,
    });
  }

  showStdout(evalId: string, stdout: string): void {
    if (stdout.trim()) {
      this._sendToWebview({
        type: "show-stdout",
        evalId,
        stdout,
      });
    }
  }

  private _sendToWebview(message: object): void {
    if (this.panel) {
      this.panel.webview.postMessage(message);
    }
  }

  private _handleWebviewMessage(msg: { type: string; [key: string]: unknown }): void {
    switch (msg.type) {
      case "widget-command":
        // Forward widget command to server
        console.log("[Colight OutputPanel] Forwarding widget-command:", msg);
        this.evalServer.sendWidgetCommand(
          msg.widgetId as string,
          msg.command as string,
          (msg.params as Record<string, unknown>) || {},
          msg.buffers as string[] | undefined
        );
        break;

      case "set-mode":
        this.mode = msg.mode as PanelMode;
        break;

      case "remove-widget":
        this.removeWidget(msg.evalId as string);
        break;

      case "clear":
        this.clear();
        break;

      case "ready":
        // Webview is ready, send current state
        this._sendToWebview({ type: "set-mode", mode: this.mode });
        for (const widget of this.widgets) {
          this._sendToWebview({
            type: "add-widget",
            evalId: widget.evalId,
            visual: widget.visual,
            mode: this.mode,
          });
        }
        break;
    }
  }

  private _getHtmlContent(): string {
    const webview = this.panel!.webview;

    // Get URIs for resources
    const styleUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, "media", "output.css")
    );

    const scriptUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, "media", "output.js")
    );

    const widgetUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this.extensionUri, "media", "widget.mjs")
    );

    const nonce = this._getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}' 'unsafe-eval' ${webview.cspSource}; img-src ${webview.cspSource} data: blob:; font-src ${webview.cspSource};">
  <link rel="stylesheet" href="${styleUri}">
  <title>Colight Output</title>
</head>
<body>
  <div id="panel-header">
    <div class="mode-toggle">
      <button data-mode="snapshot" class="active">Snapshot</button>
      <button data-mode="log">Log</button>
      <button data-mode="document" disabled title="Coming soon">Document</button>
    </div>
    <button id="clear-btn" title="Clear all">Clear</button>
  </div>
  <div id="output-container"></div>

  <script nonce="${nonce}">
    window.widgetModuleUri = "${widgetUri}";
  </script>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }

  private _getNonce(): string {
    let text = "";
    const possible =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (let i = 0; i < 32; i++) {
      text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
  }
}
