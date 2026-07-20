"""The colight daemon: a warm render service behind the agent CLI.

``colight daemon`` keeps headless Chrome (and recently loaded scenes) warm
so tight agent loops — screenshot, pick-at, pick-where, verify — stop
paying a browser launch per invocation. The daemon is a thin HTTP dispatch
layer over the very same :mod:`colight.cli_tools` functions the CLI runs
directly; it holds no command logic of its own.

Design invariants:

* **No user code runs here.** ``.py`` targets are evaluated in the CLI
  client process (identical to direct mode); the daemon receives the
  serialized visual plus a parse-level content fingerprint. Working
  directory, environment and ``sys.modules`` state therefore never leak
  between requests.
* **No shared Chrome port.** The daemon owns a pool of up to N fully
  isolated :class:`~colight.chrome_devtools.ChromeInstance` processes
  (ephemeral debug port + temp profile each), checked out per request so
  parallel CLI queries parallelize.
* **Warm scenes are content-addressed.** A small LRU maps scene keys to
  loaded StudioContext tabs; camera-framing/highlight requests mark their
  scene mutated so it is reloaded (from cached bytes — still no
  re-evaluation) before the next reuse.

Lifecycle: the discovery file is written on start and removed via atexit
and SIGTERM/SIGINT handlers; the daemon shuts itself down after
``idle_timeout`` seconds without a tool request.
"""

import atexit
import base64
import contextlib
import hmac
import json
import os
import pathlib
import secrets
import signal
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from colight.chrome_devtools import ChromeInstance
from colight.screenshots import StudioContext

from . import daemon_client, scene_pick, screenshot_tools, summaries

DEFAULT_IDLE_TIMEOUT = 30 * 60.0
DEFAULT_POOL_SIZE = 2
DEFAULT_SCENE_CACHE = 4


class NeedVisualError(Exception):
    """No warm scene for this key and no visual supplied (HTTP 409)."""


# ========== Chrome pool ==========


class ChromePool:
    """Up to N isolated Chrome instances, checked out one request at a time.

    Instances are launched lazily (the daemon starts Chrome-free) and each
    keeps the full isolation of :meth:`ChromeInstance.launch` — ephemeral
    debug port, temporary profile. ``acquire`` blocks when all N instances
    are busy.
    """

    def __init__(self, max_instances: int = DEFAULT_POOL_SIZE) -> None:
        self._max = max(1, max_instances)
        self._cond = threading.Condition()
        self._idle: List[ChromeInstance] = []
        self._all: List[ChromeInstance] = []
        self._count = 0  # live + currently-launching instances
        self._busy = 0
        self._launches = 0
        self._closed = False

    def acquire(self) -> ChromeInstance:
        """Check out an instance, launching one lazily if under the cap."""
        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("Chrome pool is closed")
                self._reap_dead_locked()
                if self._idle:
                    instance = self._idle.pop()
                    self._busy += 1
                    return instance
                if self._count < self._max:
                    self._count += 1  # reserve the slot before launching
                    break
                self._cond.wait()
        try:
            instance = ChromeInstance.launch()
        except Exception:
            with self._cond:
                self._count -= 1
                self._cond.notify()
            raise
        with self._cond:
            if self._closed:
                self._count -= 1
                instance.close()
                raise RuntimeError("Chrome pool is closed")
            self._launches += 1
            self._busy += 1
            self._all.append(instance)
            return instance

    def release(self, instance: ChromeInstance) -> None:
        """Return a checked-out instance to the pool."""
        with self._cond:
            self._busy -= 1
            if self._closed or not instance.is_alive():
                self._count -= 1
                if instance in self._all:
                    self._all.remove(instance)
                instance.close()
            else:
                self._idle.append(instance)
            self._cond.notify()

    def _reap_dead_locked(self) -> None:
        alive = [i for i in self._idle if i.is_alive()]
        for dead in self._idle:
            if dead not in alive:
                self._count -= 1
                if dead in self._all:
                    self._all.remove(dead)
                dead.close()
        self._idle = alive

    def close(self) -> None:
        """Terminate every pooled Chrome (busy ones included)."""
        with self._cond:
            self._closed = True
            instances = list(self._all)
            self._idle = []
            self._all = []
            self._cond.notify_all()
        for instance in instances:
            instance.close()

    def stats(self) -> Dict[str, int]:
        with self._cond:
            return {
                "max": self._max,
                "instances": self._count,
                "busy": self._busy,
                "launches": self._launches,
            }


# ========== Warm scene cache ==========


class WarmScene:
    """One loaded scene: a Chrome tab plus the bytes to reload it from."""

    def __init__(
        self,
        key: str,
        studio: StudioContext,
        data: Dict[str, Any],
        buffers: List[bytes],
        width: int,
        height: Optional[int],
        block_id: Optional[str],
    ) -> None:
        self.key = key
        self.studio = studio
        self.data = data
        self.buffers = buffers
        self.width = width
        self.height = height
        self.block_id = block_id
        self.lock = threading.Lock()
        self.dirty = False
        self.closed = False

    def reload(self) -> None:
        """Reload the cached visual (camera/decorations were mutated)."""
        screenshot_tools.load_visual(
            self.studio, self.data, self.buffers, self.width, self.height
        )
        self.dirty = False

    def close(self) -> None:
        """Close the tab (callers hold ``lock``). Idempotent."""
        if not self.closed:
            self.closed = True
            self.studio.stop()


class _SceneHandle:
    """SceneLike over a studio; ``mark_mutated`` flags the warm entry."""

    def __init__(
        self, studio: StudioContext, entry: Optional[WarmScene] = None
    ) -> None:
        self.studio = studio
        self._entry = entry

    def capture(self) -> Tuple[bytes, int, int]:
        studio = self.studio
        png = studio.capture_bytes(format="png")
        pixel_width = int(round(studio.width * studio.scale))
        pixel_height = int(round((studio.height or studio.width) * studio.scale))
        return png, pixel_width, pixel_height

    def mark_mutated(self) -> None:
        if self._entry is not None:
            self._entry.dirty = True


class SceneService:
    """Warm-scene LRU over a Chrome pool (the daemon's render heart)."""

    def __init__(self, pool: ChromePool, capacity: int = DEFAULT_SCENE_CACHE) -> None:
        self._pool = pool
        self._capacity = max(1, capacity)
        self._cache: "OrderedDict[str, WarmScene]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "capacity": self._capacity,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def _get(self, key: str) -> Optional[WarmScene]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
            return entry

    def _insert(self, entry: WarmScene) -> None:
        evicted: List[WarmScene] = []
        with self._lock:
            previous = self._cache.pop(entry.key, None)
            if previous is not None and previous is not entry:
                evicted.append(previous)
            self._cache[entry.key] = entry
            while len(self._cache) > self._capacity:
                _key, victim = self._cache.popitem(last=False)
                self.evictions += 1
                evicted.append(victim)
        for victim in evicted:
            with victim.lock:
                victim.close()

    def _drop(self, entry: WarmScene) -> None:
        """Remove and close a broken entry (caller holds ``entry.lock``)."""
        with self._lock:
            if self._cache.get(entry.key) is entry:
                self._cache.pop(entry.key)
        entry.close()

    def close_all(self) -> None:
        with self._lock:
            entries = list(self._cache.values())
            self._cache.clear()
        for entry in entries:
            with entry.lock:
                entry.close()

    @contextlib.contextmanager
    def scene(
        self,
        key: str,
        visual_b64: Optional[str],
        block_id: Optional[str],
        width: int,
        height: Optional[int],
        dpr: float,
        debug: bool,
        ready_timeout: Optional[float],
        fresh: bool = False,
    ) -> Iterator[_SceneHandle]:
        """Yield a loaded scene for ``key`` (warm when possible).

        ``fresh=True`` bypasses the cache in both directions: the scene is
        loaded into a new tab and closed afterwards (used by
        ``screenshot --check`` for a genuinely independent render).

        Raises:
            NeedVisualError: Warm miss and ``visual_b64`` was not supplied.
        """
        data: Optional[Dict[str, Any]] = None
        buffers: Optional[List[bytes]] = None
        if not fresh:
            entry = self._get(key)
            if entry is not None:
                with entry.lock:
                    if not entry.closed:
                        self.hits += 1
                        try:
                            if entry.dirty:
                                entry.reload()
                            yield _SceneHandle(entry.studio, entry)
                            return
                        except Exception as e:
                            # A broken warm scene (dead tab, crashed
                            # Chrome) must never be served again; genuine
                            # tool errors (ValueError) leave it intact.
                            if not isinstance(e, ValueError):
                                self._drop(entry)
                            raise
            self.misses += 1
        elif visual_b64 is None:
            # A fresh render of a warm key can reuse the cached bytes
            # (fresh means "independent render", not "re-shipped visual").
            warm = self._get(key)
            if warm is not None and not warm.closed:
                data, buffers = warm.data, warm.buffers
        if data is None or buffers is None:
            if visual_b64 is None:
                raise NeedVisualError(key)
            data, buffers = summaries.parse_colight_bytes(base64.b64decode(visual_b64))

        instance = self._pool.acquire()
        try:
            studio = StudioContext(
                width=width,
                height=height,
                scale=dpr,
                debug=debug,
                ready_timeout=ready_timeout,
                port=instance.port,
            )
            try:
                studio.start()
                screenshot_tools.load_visual(studio, data, buffers, width, height)
            except Exception:
                studio.stop()
                raise
            if fresh:
                try:
                    yield _SceneHandle(studio)
                finally:
                    studio.stop()
                return
            entry = WarmScene(key, studio, data, buffers, width, height, block_id)
            with entry.lock:
                self._insert(entry)
                try:
                    yield _SceneHandle(studio, entry)
                except Exception as e:
                    if not isinstance(e, ValueError):
                        self._drop(entry)
                    raise
        finally:
            self._pool.release(instance)


class _ServiceSceneSource:
    """SceneSource adapter: request body -> SceneService scenes."""

    def __init__(self, service: SceneService, body: Dict[str, Any]) -> None:
        self._service = service
        self._body = body
        self.block_id: Optional[str] = body.get("block_id")

    @contextlib.contextmanager
    def scene(self, fresh: bool = False) -> Iterator[_SceneHandle]:
        body = self._body
        with self._service.scene(
            key=body["key"],
            visual_b64=body.get("visual"),
            block_id=self.block_id,
            width=int(body.get("width", screenshot_tools.DEFAULT_WIDTH)),
            height=body.get("height"),
            dpr=float(body.get("dpr", screenshot_tools.DEFAULT_DPR)),
            debug=bool(body.get("debug", False)),
            ready_timeout=body.get("ready_timeout"),
            fresh=fresh,
        ) as handle:
            entry = handle._entry
            if entry is not None:
                # Warm hits know the block id from when the scene was
                # loaded; requests served warm do not carry one.
                self.block_id = entry.block_id
            yield handle


# ========== Request handlers (thin dispatch over cli_tools) ==========


def _handle_screenshot(server: "DaemonServer", body: Dict[str, Any]) -> Dict[str, Any]:
    source = _ServiceSceneSource(server.scenes, body)
    raw_views = body.get("views")
    return screenshot_tools.screenshot_source(
        source,
        body["target"],
        pathlib.Path(body["out"]),
        dpr=float(body.get("dpr", screenshot_tools.DEFAULT_DPR)),
        check=bool(body.get("check", False)),
        frame=body.get("frame"),
        out_label=body.get("out_label"),
        rulers=bool(body.get("rulers", False)),
        views=[str(name) for name in raw_views] if raw_views else None,
    )


def _handle_pick_at(server: "DaemonServer", body: Dict[str, Any]) -> Dict[str, Any]:
    source = _ServiceSceneSource(server.scenes, body)
    raw_min_alpha = body.get("min_alpha")
    return scene_pick.pick_at_source(
        source,
        body["target"],
        float(body["x"]),
        float(body["y"]),
        radius=float(body.get("radius", scene_pick.DEFAULT_RADIUS)),
        min_alpha=float(raw_min_alpha) if raw_min_alpha is not None else None,
    )


def _handle_pick_where(server: "DaemonServer", body: Dict[str, Any]) -> Dict[str, Any]:
    source = _ServiceSceneSource(server.scenes, body)
    raw_ranges = body.get("instances")
    ranges = [(int(a), int(b)) for a, b in raw_ranges] if raw_ranges else None
    out = pathlib.Path(body["out"]) if body.get("out") else None
    return scene_pick.pick_where_source(
        source,
        body["target"],
        str(body["component"]),
        instances=ranges,
        out=out,
        out_label=body.get("out_label"),
    )


def _handle_render(server: "DaemonServer", body: Dict[str, Any]) -> Dict[str, Any]:
    data, buffers = summaries.parse_colight_bytes(base64.b64decode(body["visual"]))
    instance = server.pool.acquire()
    try:
        with screenshot_tools.RenderSession(
            width=int(body.get("width", screenshot_tools.DEFAULT_WIDTH)),
            dpr=float(body.get("dpr", screenshot_tools.DEFAULT_DPR)),
            ready_timeout=body.get(
                "ready_timeout", screenshot_tools.DEFAULT_READY_TIMEOUT
            ),
            chrome_port=instance.port,
        ) as session:
            png, width, height = session.render(
                data, buffers, height=body.get("height")
            )
    finally:
        server.pool.release(instance)
    return {
        "png": base64.b64encode(png).decode("ascii"),
        "width": width,
        "height": height,
    }


_POST_HANDLERS: Dict[
    str, Callable[["DaemonServer", Dict[str, Any]], Dict[str, Any]]
] = {
    "/screenshot": _handle_screenshot,
    "/pick-at": _handle_pick_at,
    "/pick-where": _handle_pick_where,
    "/render": _handle_render,
}


# ========== HTTP server ==========


class DaemonServer:
    """The daemon process: HTTP dispatch + Chrome pool + warm scenes.

    Usable programmatically (``start()`` / ``shutdown()``, as tests do) or
    as a blocking foreground process (``serve_forever()``, which also
    installs SIGTERM/SIGINT handlers).
    """

    def __init__(
        self,
        project_root: pathlib.Path,
        host: str = "127.0.0.1",
        port: int = 0,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        pool_size: int = DEFAULT_POOL_SIZE,
        scene_cache: int = DEFAULT_SCENE_CACHE,
        token: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.project_root = project_root.resolve()
        self.host = host
        self.requested_port = port
        self.idle_timeout = idle_timeout
        self.verbose = verbose
        self.token = token or secrets.token_hex(16)
        self.pool = ChromePool(pool_size)
        self.scenes = SceneService(self.pool, scene_cache)
        self.started = time.time()
        self.last_activity = time.time()
        self.request_counts: Dict[str, int] = {}
        self._counts_lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._serve_thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("daemon not started")
        return self._httpd.server_address[1]

    @property
    def file_path(self) -> pathlib.Path:
        return daemon_client.daemon_file_path(self.project_root)

    def start(self) -> None:
        """Bind the HTTP server, write the discovery file, start workers."""
        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self.host, self.requested_port), handler)
        self._httpd.daemon_threads = True
        self._serve_thread = threading.Thread(
            target=self._httpd.serve_forever, name="colight-daemon-http", daemon=True
        )
        self._serve_thread.start()
        self._write_file()
        atexit.register(self.shutdown)
        threading.Thread(
            target=self._idle_monitor, name="colight-daemon-idle", daemon=True
        ).start()
        if self.verbose:
            print(
                f"[colight daemon] listening on {self.host}:{self.port} "
                f"(root {self.project_root})",
                flush=True,
            )

    def run_until_shutdown(self) -> None:
        """Block (with signal handlers) until shutdown; requires ``start()``."""

        def _on_signal(signum: int, _frame: Any) -> None:
            self.shutdown()

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)
        self._stopped.wait()

    def serve_forever(self) -> None:
        """Run in the foreground until shutdown (signal, idle, or /shutdown)."""
        self.start()
        self.run_until_shutdown()

    def shutdown(self) -> None:
        """Tear everything down and remove the discovery file. Idempotent."""
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
        self._remove_file()
        self.scenes.close_all()
        self.pool.close()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        self._stopped.set()

    @property
    def stopped(self) -> bool:
        return self._stopped.is_set()

    def _idle_monitor(self) -> None:
        interval = max(0.1, min(5.0, self.idle_timeout / 4))
        while not self._stopped.wait(interval):
            if time.time() - self.last_activity > self.idle_timeout:
                if self.verbose:
                    print(
                        f"[colight daemon] idle for {self.idle_timeout:g}s; "
                        "shutting down",
                        flush=True,
                    )
                self.shutdown()
                return

    # -- discovery file ----------------------------------------------------

    def _write_file(self) -> None:
        directory = self.file_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        # .colight_cache is gitignored at the repo root too, but a local
        # catch-all keeps it ignored in projects without that entry (same
        # convention colight run's record store uses).
        gitignore = directory / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n", encoding="utf-8")
        payload = {
            "port": self.port,
            "pid": os.getpid(),
            "version": daemon_client.colight_version(),
            "token": self.token,
            "started": self.started,
        }
        tmp = self.file_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.chmod(tmp, 0o600)  # the token is a local credential
        os.replace(tmp, self.file_path)

    def _remove_file(self) -> None:
        """Remove the discovery file iff it still belongs to this process."""
        try:
            info = daemon_client.read_daemon_file(self.file_path)
            if info is not None and info.pid == os.getpid():
                self.file_path.unlink()
        except OSError:
            pass

    # -- request accounting ------------------------------------------------

    def touch(self) -> None:
        self.last_activity = time.time()

    def count_request(self, endpoint: str) -> None:
        with self._counts_lock:
            self.request_counts[endpoint] = self.request_counts.get(endpoint, 0) + 1

    def status_payload(self) -> Dict[str, Any]:
        with self._counts_lock:
            counts = dict(self.request_counts)
        return {
            "ok": True,
            "version": daemon_client.colight_version(),
            "pid": os.getpid(),
            "port": self.port,
            "root": str(self.project_root),
            "started": self.started,
            "uptime": round(time.time() - self.started, 1),
            "idle_timeout": self.idle_timeout,
            "requests": {"total": sum(counts.values()), **counts},
            "pool": self.pool.stats(),
            "warm": self.scenes.stats(),
        }


def _make_handler(server: DaemonServer) -> type:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            if server.verbose:
                sys.stderr.write(
                    f"[colight daemon] {self.command} {self.path} " f"{format % args}\n"
                )

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization", "")
            expected = f"Bearer {server.token}"
            return hmac.compare_digest(header.encode(), expected.encode())

        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if not self._authorized():
                self._send_json(
                    401, {"error": {"type": "auth", "message": "bad token"}}
                )
                return
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "version": daemon_client.colight_version(),
                        "pid": os.getpid(),
                    },
                )
            elif self.path == "/status":
                self._send_json(200, server.status_payload())
            else:
                self._send_json(
                    404, {"error": {"type": "http", "message": "not found"}}
                )

        def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            if not self._authorized():
                self._send_json(
                    401, {"error": {"type": "auth", "message": "bad token"}}
                )
                return
            if self.path == "/shutdown":
                self._send_json(200, {"ok": True})
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            handler = _POST_HANDLERS.get(self.path)
            if handler is None:
                self._send_json(
                    404, {"error": {"type": "http", "message": "not found"}}
                )
                return
            server.touch()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode() or "{}")
                payload = handler(server, body)
            except NeedVisualError:
                self._send_json(409, {"need_visual": True})
                return
            except (ValueError, FileNotFoundError, TimeoutError) as e:
                self._send_json(
                    400,
                    {"error": {"type": type(e).__name__, "message": str(e)}},
                )
                return
            except Exception as e:  # infrastructure failure -> client falls back
                self._send_json(
                    500,
                    {"error": {"type": type(e).__name__, "message": str(e)}},
                )
                return
            server.count_request(self.path)
            server.touch()
            self._send_json(200, payload)

    return Handler


__all__ = [
    "DEFAULT_IDLE_TIMEOUT",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_SCENE_CACHE",
    "ChromePool",
    "DaemonServer",
    "NeedVisualError",
    "SceneService",
    "WarmScene",
]
