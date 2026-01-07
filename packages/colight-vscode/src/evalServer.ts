import * as vscode from "vscode";
import * as cp from "child_process";
import WebSocket from "ws";

// Create output channel for debugging - visible in Output panel as "Colight"
const outputChannel = vscode.window.createOutputChannel("Colight");

function log(msg: string): void {
  outputChannel.appendLine(`[${new Date().toISOString()}] ${msg}`);
}

export interface EvalResult {
  evalId: string;
  visual: string | null; // base64 encoded colight data
  stdout: string;
  error: string | null;
}

export interface WidgetMessage {
  type: string;
  widgetId: string;
  [key: string]: unknown;
}

type WidgetMessageHandler = (msg: WidgetMessage) => void;
type EvalResultHandler = (result: EvalResult) => void;

export type ServerState = "stopped" | "starting" | "running";

export class EvalServer {
  private static instance: EvalServer | null = null;

  private process: cp.ChildProcess | null = null;
  private ws: WebSocket | null = null;
  private port: number = 5510;
  private wsPort: number = 5511;
  private isStarting: boolean = false;
  private startPromise: Promise<void> | null = null;

  private widgetMessageHandlers: Set<WidgetMessageHandler> = new Set();
  private pendingEvals: Map<string, EvalResultHandler> = new Map();

  private _state: ServerState = "stopped";
  private statusBarItem: vscode.StatusBarItem | null = null;
  private _connected: boolean = false;
  private connectionEmitter = new vscode.EventEmitter<boolean>();
  readonly onConnectionStateChange = this.connectionEmitter.event;

  private constructor() {
    this._createStatusBarItem();
  }

  private _createStatusBarItem(): void {
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.statusBarItem.command = "extension.evalServerMenu";
    this._updateStatusBar();
    this.statusBarItem.show();
  }

  private _updateStatusBar(): void {
    if (!this.statusBarItem) return;

    switch (this._state) {
      case "stopped":
        this.statusBarItem.text = "$(circle-outline) Colight";
        this.statusBarItem.tooltip = "Colight eval server: Stopped\nClick for options";
        this.statusBarItem.backgroundColor = undefined;
        break;
      case "starting":
        this.statusBarItem.text = "$(sync~spin) Colight";
        this.statusBarItem.tooltip = "Colight eval server: Starting...";
        this.statusBarItem.backgroundColor = new vscode.ThemeColor(
          "statusBarItem.warningBackground"
        );
        break;
      case "running":
        this.statusBarItem.text = "$(circle-filled) Colight";
        this.statusBarItem.tooltip = "Colight eval server: Running\nClick for options";
        this.statusBarItem.backgroundColor = undefined;
        break;
    }
  }

  get state(): ServerState {
    return this._state;
  }

  get connected(): boolean {
    return this._connected;
  }

  private _setState(state: ServerState): void {
    this._state = state;
    this._updateStatusBar();
  }

  private _setConnected(connected: boolean): void {
    if (this._connected === connected) {
      return;
    }
    this._connected = connected;
    this.connectionEmitter.fire(connected);
  }

  static getInstance(): EvalServer {
    if (!EvalServer.instance) {
      EvalServer.instance = new EvalServer();
    }
    return EvalServer.instance;
  }

  async start(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    if (this.isStarting && this.startPromise) {
      return this.startPromise; // Already starting
    }

    this.isStarting = true;
    this._setState("starting");
    this.startPromise = this._doStart();

    try {
      await this.startPromise;
      this._setState("running");
    } catch (err) {
      this._setState("stopped");
      throw err;
    } finally {
      this.isStarting = false;
      this.startPromise = null;
    }
  }

  private async _doStart(): Promise<void> {
    // Try to connect to existing server first
    try {
      await this._connectWebSocket();
      log("Connected to existing eval server (was already running)");
      return;
    } catch {
      // Server not running, start it
      log("No existing server found, will start a new one");
    }

    // Start the server process
    log(`Starting colight eval server on port ${this.port}...`);
    outputChannel.show(true);

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;

    // Try different ways to start the server
    // 1. Try 'uv run colight eval' (for development)
    // 2. Try 'colight eval' directly (if installed globally)
    // 3. Try 'python -m colight eval' (if colight is installed as package)

    const commands = [
      { cmd: "uv", args: ["run", "colight", "eval", "--port", String(this.port)] },
      { cmd: "colight", args: ["eval", "--port", String(this.port)] },
      { cmd: "python", args: ["-m", "colight", "eval", "--port", String(this.port)] },
      { cmd: "python3", args: ["-m", "colight", "eval", "--port", String(this.port)] },
    ];

    let lastError: Error | null = null;

    for (const { cmd, args } of commands) {
      try {
        log(`Trying: ${cmd} ${args.join(" ")}`);

        this.process = cp.spawn(cmd, args, {
          cwd: workspaceFolder,
          stdio: ["ignore", "pipe", "pipe"],
          // Inherit environment to get PATH
          env: { ...process.env },
        });

        // Set up event handlers - log to output channel
        this.process.stdout?.on("data", (data) => {
          const text = data.toString().trim();
          if (text) {
            log(`[server] ${text}`);
          }
        });

        this.process.stderr?.on("data", (data) => {
          const text = data.toString().trim();
          if (text) {
            log(`[server stderr] ${text}`);
          }
        });

        let processErrored = false;
        this.process.on("error", (err) => {
          processErrored = true;
          lastError = err;
          log(`Failed to start with ${cmd}: ${err.message}`);
        });

        this.process.on("exit", (code) => {
          log(`Eval server (${cmd}) exited with code ${code}`);
          this.process = null;
          this.ws = null;
        });

        // Show the output channel so user can see server logs
        outputChannel.show(true);

        // Wait a moment to see if process errors immediately
        await new Promise((resolve) => setTimeout(resolve, 500));

        if (!processErrored && this.process && !this.process.killed) {
          log(`Successfully started with: ${cmd}`);
          break;
        }
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
        log(`Failed to spawn ${cmd}: ${lastError.message}`);
      }
    }

    if (!this.process || this.process.killed) {
      const errorMsg = lastError?.message || "Unknown error";
      vscode.window.showErrorMessage(
        `Failed to start Colight eval server. Make sure 'colight' is installed. Error: ${errorMsg}`
      );
      throw new Error(`Failed to start eval server: ${errorMsg}`);
    }

    // Wait for server to be ready
    await this._waitForServer();
    await this._connectWebSocket();
  }

  private async _waitForServer(maxAttempts: number = 30): Promise<void> {
    for (let i = 0; i < maxAttempts; i++) {
      try {
        // Try a quick WebSocket connection
        await new Promise<void>((resolve, reject) => {
          const testWs = new WebSocket(`ws://127.0.0.1:${this.wsPort}`);
          const timeout = setTimeout(() => {
            testWs.close();
            reject(new Error("Timeout"));
          }, 500);

          testWs.on("open", () => {
            clearTimeout(timeout);
            testWs.close();
            resolve();
          });

          testWs.on("error", () => {
            clearTimeout(timeout);
            reject(new Error("Connection failed"));
          });
        });
        return; // Success
      } catch {
        await new Promise((r) => setTimeout(r, 200));
      }
    }
    throw new Error("Eval server failed to start");
  }

  private async _connectWebSocket(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`ws://127.0.0.1:${this.wsPort}`);

      const timeout = setTimeout(() => {
        this.ws?.close();
        reject(new Error("WebSocket connection timeout"));
      }, 5000);

      this.ws.on("open", () => {
        clearTimeout(timeout);
        this._setConnected(true);
        console.log("WebSocket connected to eval server");
        resolve();
      });

      this.ws.on("error", (err) => {
        clearTimeout(timeout);
        this._setConnected(false);
        reject(err);
      });

      this.ws.on("message", (data) => {
        this._handleMessage(data.toString());
      });

      this.ws.on("close", () => {
        console.log("WebSocket disconnected from eval server");
        this.ws = null;
        this._setConnected(false);
      });
    });
  }

  private _handleMessage(data: string): void {
    try {
      const msg = JSON.parse(data);
      log(`Received message from server: type=${msg.type}, widgetId=${msg.widgetId || 'none'}`);

      if (msg.type === "eval-result") {
        const handler = this.pendingEvals.get(msg.evalId);
        if (handler) {
          this.pendingEvals.delete(msg.evalId);
          handler({
            evalId: msg.evalId,
            visual: msg.visual,
            stdout: msg.stdout || "",
            error: msg.error,
          });
        }
      } else if (msg.widgetId) {
        // Widget message - forward to handlers
        log(`Forwarding widget message to ${this.widgetMessageHandlers.size} handlers`);
        for (const handler of this.widgetMessageHandlers) {
          handler(msg);
        }
      }
    } catch (e) {
      log(`Failed to parse eval server message: ${e}`);
    }
  }

  async eval(code: string, filePath?: string): Promise<EvalResult> {
    await this.start();

    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("Not connected to eval server");
    }

    const evalId = `eval-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pendingEvals.delete(evalId);
        reject(new Error("Eval timeout"));
      }, 30000);

      this.pendingEvals.set(evalId, (result) => {
        clearTimeout(timeout);
        resolve(result);
      });

      const msg = {
        type: "eval-code",
        code,
        evalId,
        filePath: filePath || "<eval>",
      };

      this.ws!.send(JSON.stringify(msg));
    });
  }

  onWidgetMessage(handler: WidgetMessageHandler): vscode.Disposable {
    this.widgetMessageHandlers.add(handler);
    return {
      dispose: () => {
        this.widgetMessageHandlers.delete(handler);
      },
    };
  }

  sendWidgetCommand(
    widgetId: string,
    command: string,
    params: Record<string, unknown>,
    buffers?: string[]
  ): void {
    log(`sendWidgetCommand: widgetId=${widgetId}, command=${command}, wsState=${this.ws?.readyState}`);
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      log(`Cannot send widget command: not connected`);
      return;
    }

    const msg: Record<string, unknown> = {
      type: "widget-command",
      widgetId,
      command,
      params,
    };

    if (buffers && buffers.length > 0) {
      msg.buffers = buffers;
    }

    log(`Sending widget-command to server: ${JSON.stringify(msg)}`);
    this.ws.send(JSON.stringify(msg));
  }

  disposeWidget(widgetId: string): void {
    log(`disposeWidget: widgetId=${widgetId}, wsState=${this.ws?.readyState}`);
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      log(`Cannot dispose widget: not connected`);
      return;
    }

    const msg = {
      type: "widget-dispose",
      widgetId,
    };

    log(`Sending widget-dispose to server: ${JSON.stringify(msg)}`);
    this.ws.send(JSON.stringify(msg));
  }

  async stop(): Promise<void> {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.process) {
      this.process.kill();
      this.process = null;
    }

    this._setConnected(false);
    this._setState("stopped");
  }

  async restart(): Promise<void> {
    await this.stop();
    await this.start();
  }

  dispose(): void {
    this.stop();
    this.statusBarItem?.dispose();
  }
}
