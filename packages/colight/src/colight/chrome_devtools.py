# chrome_devtools.py
"""Chrome DevTools Protocol client for HTML content manipulation and screenshots.

Every launched Chrome is fully isolated: it gets an ephemeral debug port
(``--remote-debugging-port=0``, read back from the ``DevToolsActivePort``
file) and a fresh temporary ``--user-data-dir``, so concurrent colight
processes — and anything else automating Chrome on this machine — never
collide, and a killed Chrome can never leave a stale profile lock behind.

Within one process, contexts share a single launched Chrome (one tab per
context) to amortize startup cost; that instance is torn down when the last
context closes (after ``keep_alive`` seconds) and unconditionally at exit.
Attaching to an externally managed Chrome is opt-in only, via the ``port``
parameter or the ``COLIGHT_CHROME_PORT`` environment variable.
"""

import atexit
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Set, Union

from websockets.sync.client import connect

# Import ColightHTTPServer
from colight.http_server import ColightHTTPServer

DEBUG_WINDOW = False

# Opt-in: attach to an already-running Chrome on this debug port instead of
# launching an isolated instance.
CHROME_PORT_ENV = "COLIGHT_CHROME_PORT"

LAUNCH_TIMEOUT = 20.0

_lock = threading.RLock()
_shared_instance: Optional["ChromeInstance"] = None
_active_count = 0
_shutdown_timer: Optional[threading.Timer] = None
_live_instances: Set["ChromeInstance"] = set()


def format_bytes(bytes):
    if bytes >= 1024 * 1024:
        return f"{bytes / (1024 * 1024):.2f}MB"
    return f"{bytes / 1024:.2f}KB"


def find_chrome():
    """Find Chrome executable on the system"""
    possible_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
        "/usr/bin/google-chrome",  # Linux
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",  # Windows
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]

    # Check PATH first
    for cmd in ["google-chrome", "chromium", "chromium-browser", "chrome"]:
        chrome_path = shutil.which(cmd)
        if chrome_path:
            return chrome_path

    # Check common installation paths
    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("Could not find Chrome. Please install Chrome.")


def check_chrome_version(chrome_path):
    """Check if Chrome version supports the new headless mode

    Args:
        chrome_path: Path to Chrome executable

    Returns:
        tuple: (version_number, is_supported)

    Raises:
        RuntimeError: If Chrome version cannot be determined
    """
    try:
        # Run Chrome with --version flag
        output = subprocess.check_output(
            [chrome_path, "--version"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        # Parse version string (format like "Google Chrome 112.0.5615.49")
        version_str = output.strip()
        # Extract version number
        import re

        match = re.search(r"(\d+)\.", version_str)
        if not match:
            raise RuntimeError(f"Could not parse Chrome version from: {version_str}")

        major_version = int(match.group(1))
        # New headless mode (--headless=new) was introduced in Chrome 109
        return major_version, major_version >= 109
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Failed to determine Chrome version: {e}")


class ChromeInstance:
    """One launched Chrome process with its own debug port and profile dir.

    Instances are created via :meth:`launch`, which starts Chrome with
    ``--remote-debugging-port=0`` plus a fresh temporary ``--user-data-dir``
    and reads the actual ephemeral port from the ``DevToolsActivePort`` file
    Chrome writes into that directory.
    """

    def __init__(
        self, process: subprocess.Popen, port: int, user_data_dir: Path
    ) -> None:
        self.process = process
        self.port = port
        self.user_data_dir = user_data_dir

    @classmethod
    def launch(
        cls,
        width: int = 400,
        height: Optional[int] = None,
        debug: bool = False,
    ) -> "ChromeInstance":
        """Launch an isolated Chrome and wait for its DevTools endpoint.

        Args:
            width: Initial window width.
            height: Initial window height (defaults to ``width``).
            debug: Print launch diagnostics.

        Returns:
            A running ChromeInstance.

        Raises:
            RuntimeError: If Chrome exits early or the DevTools endpoint does
                not come up within ``LAUNCH_TIMEOUT`` seconds.
        """
        chrome_path = find_chrome()
        version, supports_new_headless = check_chrome_version(chrome_path)
        if debug:
            print(f"[chrome_devtools.py] Starting Chrome {version}: {chrome_path}")
            if not supports_new_headless:
                print(
                    f"[chrome_devtools.py] Warning: Chrome {version} lacks "
                    "--headless=new; using legacy headless mode"
                )

        user_data_dir = Path(tempfile.mkdtemp(prefix="colight-chrome-"))

        chrome_cmd = [chrome_path]
        if not DEBUG_WINDOW:
            chrome_cmd.append(
                "--headless=new" if supports_new_headless else "--headless"
            )
        chrome_cmd.extend(
            [
                "--remote-debugging-port=0",
                f"--user-data-dir={user_data_dir}",
                "--remote-allow-origins=*",
                "--disable-search-engine-choice-screen",
                "--ash-no-nudges",
                "--no-first-run",
                "--disable-features=Translate",
                "--no-default-browser-check",
                "--hide-scrollbars",
                f"--window-size={width},{height or width}",
                "--app=data:,",
            ]
        )
        if sys.platform.startswith("linux"):
            chrome_cmd.extend(
                [
                    "--no-sandbox",
                    "--use-angle=vulkan",
                    "--enable-features=Vulkan",
                    "--enable-unsafe-webgpu",
                    "--disable-vulkan-surface",
                ]
            )

        process = subprocess.Popen(
            chrome_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        try:
            port = cls._wait_for_port(process, user_data_dir)
        except Exception:
            _terminate_process(process)
            shutil.rmtree(user_data_dir, ignore_errors=True)
            raise

        instance = cls(process, port, user_data_dir)
        with _lock:
            _live_instances.add(instance)
        if debug:
            print(
                f"[chrome_devtools.py] Chrome ready on port {port} "
                f"(profile {user_data_dir})"
            )
        return instance

    @staticmethod
    def _wait_for_port(process: subprocess.Popen, user_data_dir: Path) -> int:
        """Poll ``DevToolsActivePort`` until Chrome publishes its port."""
        port_file = user_data_dir / "DevToolsActivePort"
        deadline = time.time() + LAUNCH_TIMEOUT
        while time.time() < deadline:
            if process.poll() is not None:
                stderr = b""
                if process.stderr:
                    stderr = process.stderr.read() or b""
                raise RuntimeError(
                    "Chrome exited during startup "
                    f"(code {process.returncode}): "
                    f"{stderr.decode(errors='replace')[-500:]}"
                )
            try:
                first_line = port_file.read_text().splitlines()[0].strip()
                port = int(first_line)
            except (OSError, ValueError, IndexError):
                time.sleep(0.05)
                continue
            # Confirm the endpoint answers before handing it out.
            try:
                urllib.request.urlopen(
                    f"http://localhost:{port}/json/version", timeout=1
                )
            except Exception:
                time.sleep(0.05)
                continue
            return port
        raise RuntimeError(
            f"Chrome did not publish a DevTools port within {LAUNCH_TIMEOUT}s "
            f"(user data dir: {user_data_dir})"
        )

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def close(self, debug: bool = False) -> None:
        """Terminate Chrome and remove its temporary profile directory."""
        if debug:
            print(
                f"[chrome_devtools.py] Closing Chrome on port {self.port} "
                f"(profile {self.user_data_dir})"
            )
        _terminate_process(self.process)
        shutil.rmtree(self.user_data_dir, ignore_errors=True)
        with _lock:
            _live_instances.discard(self)


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def _acquire_shared_instance(
    width: int, height: Optional[int], debug: bool
) -> "ChromeInstance":
    """Return the live shared Chrome, replacing a dead one (``_lock`` held).

    A dead shared instance is properly closed first — reaping the process
    and removing its temporary profile directory — before a replacement is
    launched.
    """
    global _shared_instance
    if _shared_instance is not None and not _shared_instance.is_alive():
        if debug:
            print(
                "[chrome_devtools.py] Shared Chrome died "
                f"(port {_shared_instance.port}); cleaning up and relaunching"
            )
        _shared_instance.close(debug=debug)
        _shared_instance = None
    if _shared_instance is None:
        _shared_instance = ChromeInstance.launch(
            width=width, height=height, debug=debug
        )
    elif debug:
        print(
            "[chrome_devtools.py] Reusing in-process Chrome "
            f"on port {_shared_instance.port}"
        )
    return _shared_instance


def _cancel_shutdown_timer() -> None:
    global _shutdown_timer
    if _shutdown_timer:
        _shutdown_timer.cancel()
        _shutdown_timer = None


def _close_shared(debug: bool = False) -> None:
    """Close the process-shared instance (assumes ``_lock`` is held)."""
    global _shared_instance
    _cancel_shutdown_timer()
    if _shared_instance is not None:
        _shared_instance.close(debug=debug)
        _shared_instance = None


def shutdown_chrome(debug=False):
    """Immediately shut down every Chrome this process launched.

    Closes the shared instance (cancelling any keep-alive timer) and any
    private instances still alive, removing their temporary profile
    directories. Useful in tests or to guarantee nothing lingers.

    Args:
        debug: Whether to print debug messages during shutdown.
    """
    global _active_count
    with _lock:
        if debug:
            print("[chrome_devtools.py] Explicitly shutting down Chrome")
        _close_shared(debug=debug)
        for instance in list(_live_instances):
            instance.close(debug=debug)
        _active_count = 0


def _cleanup_on_exit():
    """Cleanup function called on program exit"""
    with _lock:
        _cancel_shutdown_timer()
        global _shared_instance
        _shared_instance = None
        for instance in list(_live_instances):
            instance.close()


atexit.register(_cleanup_on_exit)


class ChromeContext:
    """A DevTools session (one tab) against an isolated Chrome instance.

    By default contexts in the same process share one launched Chrome (a tab
    each); pass ``reuse=False`` for a private Chrome instance that is
    terminated as soon as the context closes. To attach to an externally
    managed Chrome instead of launching one, pass ``port=`` or set the
    ``COLIGHT_CHROME_PORT`` environment variable — that browser is never
    terminated by colight.
    """

    _id_counter = 0

    def __init__(
        self,
        width=400,
        height=None,
        scale=1.0,
        debug=False,
        reuse=True,
        keep_alive: float = 1.0,
        window_vars=None,
        port: Optional[int] = None,
    ):
        ChromeContext._id_counter += 1
        self.id = f"chrome_{int(time.time() * 1000)}_{ChromeContext._id_counter}"
        self.width = width
        self.height = height
        self.scale = scale
        self.debug = debug
        self.reuse = reuse
        self.keep_alive = keep_alive
        self.window_vars = window_vars or {}
        self._explicit_port = port
        self.port: Optional[int] = None
        self._instance: Optional[ChromeInstance] = None
        self._uses_shared = False
        self._attached = False
        self.ws = None
        self.cmd_id = 0
        self.target_id = None  # Store target ID for tab cleanup
        # Use ColightHTTPServer for serving files
        self.server = ColightHTTPServer(
            host="localhost", port=0, debug=debug, serve_cwd=True
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def set_size(self, width=None, height=None, scale=None):
        self.width = width or self.width
        self.height = height or self.height or self.width
        if scale:
            self.scale = scale
        if DEBUG_WINDOW:
            self._send_command(
                "Browser.setWindowBounds",
                {
                    "windowId": self._send_command("Browser.getWindowForTarget")[
                        "windowId"
                    ],
                    "bounds": {"width": self.width, "height": self.height},
                },
            )
        self._send_command(
            "Page.setDeviceMetricsOverride",
            {
                "width": self.width,
                "height": self.height,
                "deviceScaleFactor": self.scale,
                "mobile": False,
            },
        )

    def _resolve_attach_port(self) -> Optional[int]:
        if self._explicit_port is not None:
            return self._explicit_port
        env_port = os.environ.get(CHROME_PORT_ENV)
        if env_port:
            try:
                return int(env_port)
            except ValueError:
                raise ValueError(
                    f"{CHROME_PORT_ENV} must be an integer, got {env_port!r}"
                )
        return None

    def start(self):
        """Connect to Chrome (launching an isolated instance if needed)."""
        global _shared_instance, _active_count

        if self.ws is not None:
            if self.debug:
                print(
                    "[chrome_devtools.py] Chrome already started, adjusting size only"
                )
            self.set_size()
            return  # Already started

        # Start ColightHTTPServer
        self.server.start()
        self.server_port = self.server.actual_port

        if self.debug:
            print(
                f"[chrome_devtools.py] Starting HTTP server on port {self.server_port}"
            )

        attach_port = self._resolve_attach_port()
        if attach_port is not None:
            # Explicitly requested attach to an externally managed Chrome.
            try:
                urllib.request.urlopen(
                    f"http://localhost:{attach_port}/json/version", timeout=2
                )
            except Exception as e:
                raise RuntimeError(
                    f"No Chrome DevTools endpoint on port {attach_port} "
                    f"(requested via {'port=' if self._explicit_port else CHROME_PORT_ENV}): {e}"
                )
            self.port = attach_port
            self._attached = True
            if self.debug:
                print(
                    f"[chrome_devtools.py] Attached to external Chrome on port {attach_port}"
                )
        else:
            with _lock:
                if self.reuse:
                    _cancel_shutdown_timer()
                    self._instance = _acquire_shared_instance(
                        self.width, self.height, self.debug
                    )
                    self._uses_shared = True
                    _active_count += 1
            if self._instance is None:
                self._instance = ChromeInstance.launch(
                    width=self.width, height=self.height, debug=self.debug
                )
            self.port = self._instance.port

        # Always open a fresh page for this context
        try:
            req = urllib.request.Request(
                f"http://localhost:{self.port}/json/new",
                method="PUT",
            )
            target = json.loads(urllib.request.urlopen(req, timeout=5).read())
            self.target_id = target["id"]  # Store target ID for cleanup
            ws_url = target["webSocketDebuggerUrl"]
        except Exception as e:
            raise RuntimeError(f"Failed to open Chrome page: {e}")

        self.ws = connect(ws_url, max_size=100 * 1024 * 1024)  # 100MB max message size
        # Enable required domains
        self._send_command("Page.enable")
        self._send_command("Runtime.enable")
        self._send_command("Console.enable")  # Enable console events

    def stop(self):
        """Close this context's tab and release its Chrome instance."""
        global _shared_instance, _active_count, _shutdown_timer

        if self.debug:
            print("[chrome_devtools.py] Stopping Chrome context")

        # Close the tab if we have a target ID
        if self.target_id and self.ws:
            try:
                if self.debug:
                    print(f"[chrome_devtools.py] Closing tab {self.target_id}")
                # Close the tab using the target ID
                req = urllib.request.Request(
                    f"http://localhost:{self.port}/json/close/{self.target_id}",
                    method="PUT",
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                if self.debug:
                    print(
                        f"[chrome_devtools.py] Failed to close tab {self.target_id}: {e}"
                    )

        if self.ws:
            self.ws.close()
            self.ws = None

        self.target_id = None  # Clear target ID

        if self._attached:
            # Externally managed Chrome: never terminate it.
            pass
        elif self._uses_shared:
            with _lock:
                _active_count -= 1
                if _active_count <= 0 and not DEBUG_WINDOW:
                    if self.keep_alive > 0:
                        if self.debug:
                            print(
                                "[chrome_devtools.py] Scheduling shared Chrome "
                                f"shutdown in {self.keep_alive}s (keep_alive)"
                            )
                        _cancel_shutdown_timer()
                        _shutdown_timer = threading.Timer(
                            self.keep_alive, _shutdown_shared_if_idle
                        )
                        _shutdown_timer.daemon = True
                        _shutdown_timer.start()
                    else:
                        if self.debug:
                            print(
                                "[chrome_devtools.py] Shutting down shared Chrome "
                                "immediately (keep_alive=0)"
                            )
                        _close_shared(debug=self.debug)
        elif self._instance is not None and not DEBUG_WINDOW:
            # Private instance: terminate and clean up right away.
            self._instance.close(debug=self.debug)

        self._instance = None
        self._uses_shared = False
        self._attached = False

        # Stop ColightHTTPServer
        if self.server:
            if self.debug:
                print("[chrome_devtools.py] Shutting down HTTP server")
            self.server.stop()

    def _send_command(self, method, params=None):
        """Send a command to Chrome and wait for the response"""
        if not self.ws:
            raise RuntimeError("Not connected to Chrome")

        self.cmd_id += 1
        message = {"id": self.cmd_id, "method": method, "params": params or {}}

        message_str = json.dumps(message)

        self.ws.send(message_str)

        # Wait for response with matching id
        while True:
            response = json.loads(self.ws.recv())

            # Print console messages if debug is enabled
            if self.debug and response.get("method") == "Console.messageAdded":
                message = response["params"]["message"]
                level = message.get("level", "log")
                text = message.get("text", "")
                print(f"[chrome.{level}]: {text}")

            # Handle command response
            if "id" in response and response["id"] == self.cmd_id:
                if "error" in response:
                    raise RuntimeError(
                        f"Chrome DevTools command failed: {response['error']}"
                    )
                return response.get("result", {})

    def load_html(self, html, files=None):
        """Serve HTML content and optional files over localhost and load it in the page"""
        self.set_size()

        # Inject window variables if provided
        if self.window_vars:
            # Create a script that sets window variables before anything else
            window_vars_script = "<script>\n"
            for key, value in self.window_vars.items():
                window_vars_script += f"window.{key} = {json.dumps(value)};\n"
            window_vars_script += "</script>\n"

            # Insert the script at the beginning of the head tag, or right after <html> if no head
            if "<head>" in html:
                html = html.replace("<head>", f"<head>\n{window_vars_script}", 1)
            elif "<html>" in html:
                html = html.replace("<html>", f"<html>\n{window_vars_script}", 1)
            else:
                # If no html tag, prepend the script
                html = window_vars_script + html

            if self.debug:
                print(
                    f"[chrome_devtools.py] Injected window variables: {list(self.window_vars.keys())}"
                )

        # Serve files using ColightHTTPServer
        if files:
            for k, v in files.items():
                self.server.add_served_file(k, v)
        self.server.add_served_file("index.html", html)

        # Navigate to page
        url = self.server.get_url("index.html")
        self._send_command("Page.navigate", {"url": url})

        while True:
            if not self.ws:
                raise RuntimeError("[chrome_devtools.py] WebSocket connection lost")
            response = json.loads(self.ws.recv())
            if response.get("method") == "Page.loadEventFired":
                if self.debug:
                    print("[chrome_devtools.py] Page load complete")
                break

    def evaluate(self, expression, return_by_value=True, await_promise=False):
        """Evaluate JavaScript code in the page context

        Args:
            expression: JavaScript expression to evaluate
            return_by_value: Whether to return the result by value
            await_promise: Whether to wait for promise resolution
        """
        result = self._send_command(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": return_by_value,
                "awaitPromise": await_promise,
            },
        )

        return result.get("result", {}).get("value")

    def capture_image(self, format: str = "png", quality: int = 90) -> bytes:
        """Capture a screenshot of the page as PNG or WebP bytes.

        Args:
            format: Image format ("png" or "webp")
            quality: Image quality for WebP (0-100, ignored for PNG)

        Returns:
            Image bytes in the specified format
        """
        if self.debug:
            print(f"[chrome_devtools.py] Capturing image in {format.upper()} format")

        if format not in ["png", "webp"]:
            raise ValueError(f"Unsupported format: {format}. Use 'png' or 'webp'.")

        params = {
            "format": format,
            "captureBeyondViewport": True,
            "clip": {
                "x": 0,
                "y": 0,
                "width": self.width,
                "height": self.height,
                # Device scale factor (set via Page.setDeviceMetricsOverride)
                # already multiplies the output size; a clip scale other than
                # 1 would apply self.scale twice.
                "scale": 1,
            },
        }

        # Add quality parameter for WebP
        if format == "webp":
            params["quality"] = quality

        result = self._send_command("Page.captureScreenshot", params)

        if not result or "data" not in result:
            raise RuntimeError("Failed to capture image")

        return base64.b64decode(result["data"])

    def capture_pdf(self) -> bytes:
        """Capture the current page as a PDF and return PDF bytes."""
        if self.debug:
            print("[chrome_devtools.py] Capturing PDF")

        # Convert pixel width to inches at 96 DPI
        paper_width = self.width / 96
        paper_height = paper_width * ((self.height or self.width) / self.width)

        # Request PDF with stream transfer mode
        result = self._send_command(
            "Page.printToPDF",
            {
                "landscape": False,
                "printBackground": True,
                "preferCSSPageSize": True,
                "paperWidth": paper_width,
                "paperHeight": paper_height,
                "marginTop": 0,
                "marginBottom": 0,
                "marginLeft": 0,
                "marginRight": 0,
                "transferMode": "ReturnAsStream",
            },
        )
        if not result or "stream" not in result:
            raise RuntimeError("Failed to capture PDF - no stream handle returned")

        # Read the PDF data in chunks
        stream_handle = result["stream"]
        pdf_chunks = []

        while True:
            chunk_result = self._send_command(
                "IO.read", {"handle": stream_handle, "size": 500000}
            )

            if not chunk_result:
                raise RuntimeError("Failed to read PDF stream")

            if "data" in chunk_result:
                pdf_chunks.append(base64.b64decode(chunk_result["data"]))

            if chunk_result.get("eof", False):
                break

        # Close the stream
        self._send_command("IO.close", {"handle": stream_handle})

        # Combine all chunks
        return b"".join(pdf_chunks)

    def check_webgpu_support(self):
        """Check if WebGPU is available and functional in the browser

        Returns:
            dict: Detailed WebGPU support information including:
                - supported: bool indicating if WebGPU is available
                - adapter: information about the GPU adapter if available
                - reason: explanation if WebGPU is not supported
                - features: list of supported features if available
        """
        # First load a blank page to ensure we have a proper context
        self.load_html("<html><body></body></html>")

        result = self.evaluate(
            """
            (async function() {
                if (!navigator.gpu) {
                    return {
                        supported: false,
                        reason: 'navigator.gpu is not available'
                    };
                }

                try {

                    let adapter;
                    const startTime = performance.now();
                    for (let i = 0; i < 10; i++) {
                        adapter = await navigator.gpu.requestAdapter({
                            powerPreference: 'high-performance'
                        });
                        if (adapter) {
                            console.log(`GPU adapter ready after ${((performance.now() - startTime)/1000).toFixed(2)}s (attempt ${i + 1})`);
                            break;
                        }
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                    if (!adapter) {
                        console.log(`Failed to get GPU adapter after ${((performance.now() - startTime)/1000).toFixed(2)}s`);
                    }

                    if (!adapter) {
                        return {
                            supported: false,
                            reason: 'No WebGPU adapter found'
                        };
                    }
                    // note that adapter.requestAdapterInfo doesn't always exist so we don't use it

                    // Request device with basic features
                    const device = await adapter.requestDevice({
                        requiredFeatures: []
                    });

                    if (!device) {
                        return {
                            supported: false,
                            reason: 'Failed to create WebGPU device'
                        };
                    }

                    // Try to create a simple buffer to verify device works
                    try {
                        const testBuffer = device.createBuffer({
                            size: 4,
                            usage: GPUBufferUsage.COPY_DST
                        });
                        testBuffer.destroy();
                    } catch (e) {
                        return {
                            supported: false,
                            reason: 'Device creation succeeded but buffer operations failed'
                        };
                    }

                    return {
                        supported: true,
                        adapter: {
                            name: 'WebGPU Device'
                        },
                        features: Array.from(adapter.features).map(f => f.toString())
                    };
                } catch (e) {
                    return {
                        supported: false,
                        reason: e.toString()
                    };
                }
            })()
        """,
            await_promise=True,
        )

        if self.debug:
            if result.get("supported"):
                print(
                    f"[chrome_devtools.py] WebGPU Adapter: '{result.get('adapter', {}).get('name')}'"
                )
                print(
                    f"[chrome_devtools.py]   Features: {', '.join(result.get('features', []))}"
                )
            else:
                print(
                    f"[chrome_devtools.py] WebGPU not supported: {result.get('reason')}"
                )

        return result

    def save_gpu_info(self, output_path: Union[str, Path]):
        """Save Chrome's GPU diagnostics page (chrome://gpu) to a PDF file

        Args:
            output_path: Path where to save the PDF file

        Returns:
            Path to the saved PDF file
        """
        output_path = Path(output_path)
        if self.debug:
            print(f"[chrome_devtools.py] Capturing GPU diagnostics to: {output_path}")

        # Navigate to GPU info page
        self._send_command("Page.navigate", {"url": "chrome://gpu"})

        # Wait for page load
        while True and self.ws:
            response = json.loads(self.ws.recv())
            if response.get("method") == "Page.loadEventFired":
                break

        # Print to PDF
        result = self._send_command(
            "Page.printToPDF",
            {
                "landscape": False,
                "printBackground": True,
                "preferCSSPageSize": True,
            },
        )

        if not result or "data" not in result:
            raise RuntimeError("Failed to generate PDF")

        # Save PDF
        pdf_data = base64.b64decode(result["data"])
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with open(output_path, "wb") as f:
            f.write(pdf_data)

        if self.debug:
            print(f"[chrome_devtools.py] GPU diagnostics saved to: {output_path}")

        return output_path


def _shutdown_shared_if_idle() -> None:
    """Timer callback: close the shared instance if still idle."""
    global _shutdown_timer
    with _lock:
        _shutdown_timer = None
        if _active_count <= 0:
            _close_shared()


def main():
    """Example usage"""
    html = """
    <html>
    <head></head>
    <body style="background:red; width:100vw; height:100vh;"><div></div></body>
    </html>
    """

    with ChromeContext(width=400, height=600, debug=True) as chrome:
        # Check WebGPU support
        chrome.check_webgpu_support()

        # Load content served via localhost
        chrome.load_html(html)

        # Capture and save red background image
        image_data = chrome.capture_image()
        Path("./scratch/screenshots").mkdir(exist_ok=True, parents=True)
        with open("./scratch/screenshots/webgpu_test_red.png", "wb") as f:
            f.write(image_data)

        # Change to green and capture again
        chrome.evaluate('document.body.style.background = "green"; "changed!"')
        image_data = chrome.capture_image()
        with open("./scratch/screenshots/webgpu_test_green.png", "wb") as f:
            f.write(image_data)


if __name__ == "__main__":
    main()
