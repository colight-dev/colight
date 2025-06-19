# chrome_devtools.py
"""
Simple Chrome DevTools Protocol client for HTML content manipulation and screenshots
"""

import os
import time
import json
import base64
import shutil
import subprocess
import threading
import urllib.request
from websockets.sync.client import connect
import sys
import atexit
from pathlib import Path
from typing import Union

# Import ColightHTTPServer
from colight.server import ColightHTTPServer

DEBUG_WINDOW = False

PORT_FILE = Path(".colight-port")

_shared_process = None
_shared_port: int | None = None
_shared_owned = False
_active_count = 0
_shutdown_timer: threading.Timer | None = None


def _cleanup_on_exit():
    """Cleanup function called on program exit"""
    global _shared_process, _shutdown_timer

    # Cancel any pending shutdown timer
    if _shutdown_timer:
        _shutdown_timer.cancel()
        _shutdown_timer = None

    # Terminate Chrome if still running
    if _shared_process and _shared_process.poll() is None:
        try:
            _shared_process.terminate()
            _shared_process.wait(timeout=2)
        except Exception:
            try:
                _shared_process.kill()
            except Exception:
                pass

    # Remove port file
    if PORT_FILE.exists():
        try:
            PORT_FILE.unlink()
        except Exception:
            pass


# Register cleanup function to run on exit
atexit.register(_cleanup_on_exit)


def format_bytes(bytes):
    if bytes >= 1024 * 1024:
        return f"{bytes/(1024*1024):.2f}MB"
    return f"{bytes/1024:.2f}KB"


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


def shutdown_chrome(debug=False):
    """Explicitly shut down Chrome and clean up global state

    This function can be called to immediately terminate Chrome without waiting
    for contexts to close or keep-alive timers to expire. Useful for tests
    or when you need to ensure Chrome is completely stopped.

    Args:
        debug: Whether to print debug messages during shutdown
    """
    global _shared_process, _shared_port, _shared_owned, _active_count, _shutdown_timer

    if debug:
        print("[chrome_devtools.py] Explicitly shutting down Chrome")

    # Cancel any pending shutdown timer
    if _shutdown_timer:
        _shutdown_timer.cancel()
        _shutdown_timer = None
        if debug:
            print("[chrome_devtools.py] Cancelled pending shutdown timer")

    # Terminate Chrome process if it exists and is running
    if _shared_process and _shared_process.poll() is None:
        if debug:
            print("[chrome_devtools.py] Terminating Chrome process")
        _shared_process.terminate()
        try:
            _shared_process.wait(timeout=5)
            if debug:
                print("[chrome_devtools.py] Chrome process terminated successfully")
        except subprocess.TimeoutExpired:
            if debug:
                print(
                    "[chrome_devtools.py] Chrome process did not terminate, forcing kill"
                )
            _shared_process.kill()
            _shared_process.wait()

    # Clean up port file
    if PORT_FILE.exists():
        PORT_FILE.unlink()
        if debug:
            print("[chrome_devtools.py] Removed port file")

    # Reset global state
    _shared_process = None
    _shared_port = None
    _shared_owned = False
    _active_count = 0

    if debug:
        print("[chrome_devtools.py] Chrome shutdown complete")


class ChromeContext:
    """Manages a Chrome instance and provides methods for content manipulation and screenshots"""

    def __init__(
        self,
        port=9222,
        width=400,
        height=None,
        scale=1.0,
        debug=False,
        reuse=True,
        keep_alive: float = 1.0,
    ):
        self.id = f"chrome_{int(time.time() * 1000)}_{hash(str(port))}"  # Unique ID for this context
        self.port = port
        self.width = width
        self.height = height
        self.scale = scale
        self.debug = debug
        self.reuse = reuse
        self.keep_alive = keep_alive
        self.chrome_process = None
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

    def start(self):
        """Start Chrome and connect to DevTools Protocol"""
        global \
            _shared_process, \
            _shared_port, \
            _shared_owned, \
            _active_count, \
            _shutdown_timer

        if self.chrome_process:
            if self.debug:
                print(
                    "[chrome_devtools.py] Chrome already started, adjusting size only"
                )
            self.set_size()
            return  # Already started

        # Cancel pending shutdown if a new context starts
        if _shutdown_timer:
            _shutdown_timer.cancel()
            _shutdown_timer = None

        # Start ColightHTTPServer
        self.server.start()
        self.server_port = self.server.actual_port

        if self.debug:
            print(
                f"[chrome_devtools.py] Starting HTTP server on port {self.server_port}"
            )

        # Attempt to reuse existing Chrome process
        reused = False
        if self.reuse:
            if _shared_process and _shared_process.poll() is None:
                self.chrome_process = _shared_process
                self.port = _shared_port
                reused = True
                if self.debug:
                    print(
                        f"[chrome_devtools.py] Reusing in-memory Chrome on port {self.port}"
                    )
            elif PORT_FILE.exists():
                try:
                    port = int(PORT_FILE.read_text().strip())
                    urllib.request.urlopen(f"http://localhost:{port}/json", timeout=1)
                    self.port = port
                    _shared_port = port
                    _shared_process = None
                    _shared_owned = False
                    reused = True
                    if self.debug:
                        print(
                            f"[chrome_devtools.py] Reusing Chrome from port file on port {port}"
                        )
                except Exception:
                    if self.debug:
                        print(
                            "[chrome_devtools.py] Existing port file invalid, starting new Chrome"
                        )
                    PORT_FILE.unlink(missing_ok=True)

        if not reused:
            chrome_path = find_chrome()
            if self.debug:
                print(f"[chrome_devtools.py] Starting Chrome from: {chrome_path}")

            version, supports_new_headless = check_chrome_version(chrome_path)
            if self.debug:
                print(f"[chrome_devtools.py] Chrome version: {version}")
                if not supports_new_headless:
                    print(
                        f"[chrome_devtools.py] Warning: Chrome version {version} does not support the new headless mode (--headless=new). Using legacy headless mode instead"
                    )

            headless_flag = ""
            if not DEBUG_WINDOW:
                headless_flag = (
                    "--headless=new" if supports_new_headless else "--headless"
                )

            chrome_cmd = [
                chrome_path,
                headless_flag,
                f"--remote-debugging-port={self.port}",
                "--remote-allow-origins=*",
                "--disable-search-engine-choice-screen",
                "--ash-no-nudges",
                "--no-first-run",
                "--disable-features=Translate",
                "--no-default-browser-check",
                "--hide-scrollbars",
                f"--window-size={self.width},{self.height or self.width}",
                "--app=data:,",
            ]

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

            self.chrome_process = subprocess.Popen(
                chrome_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            _shared_process = self.chrome_process
            _shared_port = self.port
            _shared_owned = True
            PORT_FILE.write_text(str(self.port))

        # Wait for Chrome to become responsive
        start_time = time.time()
        if self.debug:
            print(
                f"[chrome_devtools.py] Attempting to connect to Chrome on port {self.port}"
            )

        while True:
            try:
                urllib.request.urlopen(
                    f"http://localhost:{self.port}/json/version", timeout=1
                )
                break
            except Exception:
                if time.time() - start_time > 10:
                    raise RuntimeError("Chrome did not start in time")
                time.sleep(0.1)

        chrome_startup_time = time.time() - start_time
        if self.debug:
            print(
                f"[chrome_devtools.py] Chrome became responsive in {chrome_startup_time:.3f}s"
            )

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

        self.ws = connect(ws_url)
        _active_count += 1
        # Enable required domains
        self._send_command("Page.enable")
        self._send_command("Runtime.enable")
        self._send_command("Console.enable")  # Enable console events

    def stop(self):
        """Stop Chrome and clean up"""
        global \
            _shared_process, \
            _shared_port, \
            _shared_owned, \
            _active_count, \
            _shutdown_timer

        if self.debug:
            print("[chrome_devtools.py] Stopping Chrome process")

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

        _active_count -= 1
        if self.chrome_process and not DEBUG_WINDOW:
            if _active_count <= 0 and _shared_owned:

                def _term():
                    global _shared_process, _shared_port, _shared_owned, _shutdown_timer
                    if _shared_process:
                        _shared_process.terminate()
                        try:
                            _shared_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if self.debug:
                                print(
                                    "[chrome_devtools.py] Chrome process did not terminate, forcing kill"
                                )
                            _shared_process.kill()
                    if PORT_FILE.exists():
                        PORT_FILE.unlink()
                    _shutdown_timer = None
                    _shared_owned = False
                    _shared_process = None
                    _shared_port = None

                if self.keep_alive > 0:
                    if self.debug:
                        print(
                            f"[chrome_devtools.py] Scheduling Chrome shutdown in {self.keep_alive}s (keep_alive timeout)"
                        )
                    _shutdown_timer = threading.Timer(self.keep_alive, _term)
                    _shutdown_timer.start()
                else:
                    if self.debug:
                        print(
                            "[chrome_devtools.py] Shutting down Chrome immediately (keep_alive=0)"
                        )
                    _term()

            self.chrome_process = None
        else:
            if _active_count <= 0 and _shared_owned and _shutdown_timer is None:
                # ensure timer when contexts closed without chrome_process reference
                def _term():
                    global _shared_process, _shared_port, _shared_owned, _shutdown_timer
                    if _shared_process:
                        _shared_process.terminate()
                        try:
                            _shared_process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            _shared_process.kill()
                    if PORT_FILE.exists():
                        PORT_FILE.unlink()
                    _shutdown_timer = None
                    _shared_owned = False
                    _shared_process = None
                    _shared_port = None

                if self.keep_alive > 0:
                    if self.debug:
                        print(
                            f"[chrome_devtools.py] Scheduling Chrome shutdown in {self.keep_alive}s (keep_alive timeout, no process reference)"
                        )
                    _shutdown_timer = threading.Timer(self.keep_alive, _term)
                    _shutdown_timer.start()
                else:
                    if self.debug:
                        print(
                            "[chrome_devtools.py] Shutting down Chrome immediately (keep_alive=0, no process reference)"
                        )
                    _term()

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
        # if self.debug:
        #     size_bytes = len(message_str.encode("utf-8"))
        #     print(
        #         f"[chrome_devtools.py] Sending message of size: {format_bytes(size_bytes)} via WebSocket"
        #     )

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
                "scale": self.scale,
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
