"""Client side of the colight daemon: discovery, transport, command routing.

Discovery is nREPL-style: ``colight daemon`` writes
``<project_root>/.colight_cache/daemon.json`` ({port, pid, version, token,
started}) and CLI commands look for that file by walking up from the
target's directory (then from the cwd). The daemon is used iff the file
exists AND its pid is alive AND its version matches this process's colight
version AND an authenticated health check succeeds; otherwise commands fall
back silently to direct mode (and remove the file when the pid is dead).

The daemon never executes user code: ``.py`` targets are evaluated in the
short-lived CLI process (exactly like direct mode) and the serialized
visual is shipped to the daemon, which owns the warm Chrome pool and a
small LRU of loaded scenes keyed by a client-computed content fingerprint.
A repeated query against an unchanged target therefore skips both
re-evaluation (the fingerprint is computed by parsing, not executing) and
scene re-loading (the daemon still has it warm).
"""

import base64
import hashlib
import json
import os
import pathlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import colight.format as colight_format
from colight.runtime.parser import find_project_root, parse_colight_file

DAEMON_DIRNAME = ".colight_cache"
DAEMON_FILENAME = "daemon.json"

DEFAULT_TIMEOUT = 120.0
HEALTH_TIMEOUT = 2.0

# Direct mode honors COLIGHT_CHROME_PORT by attaching to an external
# Chrome; routing through the daemon would silently ignore that request,
# so its presence disables daemon use.
CHROME_PORT_ENV = "COLIGHT_CHROME_PORT"

_ERROR_TYPES = {
    "ValueError": ValueError,
    "FileNotFoundError": FileNotFoundError,
    "TimeoutError": TimeoutError,
}


class DaemonUnavailable(Exception):
    """The daemon cannot serve this request; fall back silently to direct."""


class NeedVisual(Exception):
    """The daemon has no warm scene for this key; re-send with the visual."""


def colight_version() -> str:
    """The installed colight version (daemon/client must match exactly)."""
    try:
        from importlib.metadata import version

        return version("colight")
    except Exception:
        return "unknown"


def daemon_file_path(project_root: pathlib.Path) -> pathlib.Path:
    """Discovery-file location for a project root."""
    return project_root / DAEMON_DIRNAME / DAEMON_FILENAME


@dataclass
class DaemonInfo:
    """A validated, reachable daemon."""

    path: pathlib.Path
    port: int
    pid: int
    version: str
    token: str
    started: float

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def read_daemon_file(path: pathlib.Path) -> Optional[DaemonInfo]:
    """Parse a discovery file, tolerating absence and corruption."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return DaemonInfo(
            path=path,
            port=int(raw["port"]),
            pid=int(raw["pid"]),
            version=str(raw["version"]),
            token=str(raw["token"]),
            started=float(raw.get("started", 0.0)),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def pid_alive(pid: int) -> bool:
    """Whether a pid refers to a live process (no signal is delivered)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def request(
    info: DaemonInfo,
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Send one authenticated request; map errors to client exceptions.

    Raises:
        NeedVisual: 409 — the daemon wants the serialized visual.
        ValueError / FileNotFoundError / TimeoutError: tool errors the
            daemon reported (HTTP 400), re-raised as their original types.
        DaemonUnavailable: connection problems, auth mismatch or daemon-side
            infrastructure failures (HTTP 401/5xx) — callers fall back to
            direct mode silently.
    """
    payload = json.dumps(body or {}).encode() if method == "POST" else None
    req = urllib.request.Request(
        f"{info.base_url}{path}",
        data=payload,
        method=method,
        headers={
            "Authorization": f"Bearer {info.token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
        except Exception:
            detail = {}
        if e.code == 409 and detail.get("need_visual"):
            raise NeedVisual() from None
        if e.code == 400:
            error = detail.get("error") or {}
            exc_type = _ERROR_TYPES.get(error.get("type", ""), ValueError)
            raise exc_type(error.get("message", "daemon request failed")) from None
        raise DaemonUnavailable(f"daemon returned HTTP {e.code}") from None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise DaemonUnavailable(str(e)) from None


def _healthy(info: DaemonInfo) -> bool:
    try:
        payload = request(info, "GET", "/health", timeout=HEALTH_TIMEOUT)
    except (DaemonUnavailable, ValueError, FileNotFoundError, TimeoutError):
        return False
    return payload.get("ok") is True


def validate_info(info: DaemonInfo) -> bool:
    """Whether a parsed discovery file points at a usable daemon."""
    return pid_alive(info.pid) and info.version == colight_version() and _healthy(info)


def discover(*start_dirs: pathlib.Path) -> Optional[DaemonInfo]:
    """Find a usable daemon by walking up from each start directory.

    The first discovery file whose daemon validates (live pid, matching
    version, healthy endpoint) wins. Files pointing at dead pids are
    removed; every other failure is silent (direct mode takes over).
    """
    if os.environ.get(CHROME_PORT_ENV):
        return None
    seen: set = set()
    for start in start_dirs:
        start = start.resolve()
        if start.is_file():
            start = start.parent
        for candidate_dir in [start, *start.parents]:
            path = candidate_dir / DAEMON_DIRNAME / DAEMON_FILENAME
            if path in seen:
                continue
            seen.add(path)
            if not path.is_file():
                continue
            info = read_daemon_file(path)
            if info is None:
                continue
            if not pid_alive(info.pid):
                # Stale file from a crashed daemon: clean it up.
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            if info.version != colight_version():
                continue
            if not _healthy(info):
                continue
            return info
    return None


def discover_for_target(target: pathlib.Path) -> Optional[DaemonInfo]:
    """Discover a daemon for a CLI target (target dir first, then cwd)."""
    return discover(target, pathlib.Path.cwd())


# ========== Scene fingerprints ==========


def scene_key(
    target: pathlib.Path,
    block: Optional[str],
    width: int,
    height: Optional[int],
    dpr: float,
) -> str:
    """Content fingerprint of a target's rendered scene, without executing.

    ``.colight`` targets hash the artifact bytes. ``.py`` targets reuse the
    runtime's transitive block cache keys (own source + upstream block
    sources + local import mtimes — the same machinery ``colight run`` uses
    to skip blocks), so edits anywhere in the dependency chain change the
    key. Render-shaping parameters (block selection, viewport, dpr) are
    folded in because they change the loaded scene.
    """
    resolved = target.resolve()
    hasher = hashlib.sha256()
    hasher.update(f"{resolved}|{block}|{width}|{height}|{dpr}|".encode())
    if resolved.suffix == ".colight":
        hasher.update(resolved.read_bytes())
    else:
        document = parse_colight_file(
            resolved, project_root=find_project_root(resolved)
        )
        from . import blocks as blocks_mod

        for blk, sid in blocks_mod.assign_stable_ids(document):
            hasher.update(f"{sid}:{blk.id};".encode())
    return hasher.hexdigest()[:32]


def _encode_visual(data: Dict[str, Any], buffers: List[Any]) -> str:
    return base64.b64encode(colight_format.create_bytes(data, buffers)).decode("ascii")


# ========== Command routing ==========


def _scene_request(
    info: DaemonInfo,
    endpoint: str,
    target: pathlib.Path,
    block: Optional[str],
    width: int,
    height: Optional[int],
    dpr: float,
    debug: bool,
    ready_timeout: Optional[float],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """POST a scene-backed command, evaluating the target only on warm miss."""
    body: Dict[str, Any] = {
        "key": scene_key(target, block, width, height, dpr),
        "target": str(target),
        "width": width,
        "height": height,
        "dpr": dpr,
        "debug": debug,
        "ready_timeout": ready_timeout,
        **extra,
    }
    try:
        return request(info, "POST", endpoint, body)
    except NeedVisual:
        from . import screenshot_tools

        data, buffers, block_id = screenshot_tools.resolve_visual(target, block)
        body["visual"] = _encode_visual(data, buffers)
        body["block_id"] = block_id
        return request(info, "POST", endpoint, body)


def try_screenshot(
    target: pathlib.Path,
    out: pathlib.Path,
    block: Optional[str],
    width: int,
    height: Optional[int],
    dpr: float,
    check: bool,
    frame: Optional[str],
    debug: bool,
    ready_timeout: Optional[float],
    rulers: bool = False,
    views: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Route ``colight screenshot`` through a discovered daemon.

    Composition (rulers/views) happens in the daemon's
    ``screenshot_source`` call — the identical code direct mode runs (the
    version match discovery enforces guarantees identical bytes).

    Returns:
        The CLI payload, or None when no usable daemon exists (or it became
        unavailable mid-request) — the caller then runs direct mode.
    """
    info = discover_for_target(target)
    if info is None:
        return None
    try:
        return _scene_request(
            info,
            "/screenshot",
            target,
            block,
            width,
            height,
            dpr,
            debug,
            ready_timeout,
            {
                "out": str(out.resolve()),
                "out_label": str(out),
                "check": check,
                "frame": frame,
                "rulers": rulers,
                "views": views,
            },
        )
    except DaemonUnavailable:
        return None


def try_pick_at(
    target: pathlib.Path,
    x: float,
    y: float,
    radius: float,
    block: Optional[str],
    width: int,
    height: Optional[int],
    dpr: float,
    debug: bool,
    ready_timeout: Optional[float],
    min_alpha: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Route ``colight pick-at`` through a discovered daemon (None = direct)."""
    info = discover_for_target(target)
    if info is None:
        return None
    extra: Dict[str, Any] = {"x": x, "y": y, "radius": radius}
    if min_alpha is not None:
        extra["min_alpha"] = min_alpha
    try:
        return _scene_request(
            info,
            "/pick-at",
            target,
            block,
            width,
            height,
            dpr,
            debug,
            ready_timeout,
            extra,
        )
    except DaemonUnavailable:
        return None


def try_pick_where(
    target: pathlib.Path,
    component_selector: str,
    instances: Optional[List[Tuple[int, int]]],
    out: Optional[pathlib.Path],
    block: Optional[str],
    width: int,
    height: Optional[int],
    dpr: float,
    debug: bool,
    ready_timeout: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Route ``colight pick-where`` through a discovered daemon (None = direct)."""
    info = discover_for_target(target)
    if info is None:
        return None
    extra: Dict[str, Any] = {
        "component": component_selector,
        "instances": [list(pair) for pair in instances] if instances else None,
    }
    if out is not None:
        extra["out"] = str(out.resolve())
        extra["out_label"] = str(out)
    try:
        return _scene_request(
            info,
            "/pick-where",
            target,
            block,
            width,
            height,
            dpr,
            debug,
            ready_timeout,
            extra,
        )
    except DaemonUnavailable:
        return None


class RemoteRenderSession:
    """Render-session stand-in that renders on the daemon's Chrome pool.

    Implements the subset of :class:`screenshot_tools.RenderSession` the
    verify screenshot layer uses (context manager + ``render``).
    """

    def __init__(self, info: DaemonInfo, width: int, dpr: float) -> None:
        self._info = info
        self._width = width
        self._dpr = dpr

    def __enter__(self) -> "RemoteRenderSession":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    def render(
        self,
        data: Dict[str, Any],
        buffers: List[bytes],
        height: Optional[int] = None,
    ) -> Tuple[bytes, int, int]:
        payload = request(
            self._info,
            "POST",
            "/render",
            {
                "visual": _encode_visual(data, buffers),
                "width": self._width,
                "height": height,
                "dpr": self._dpr,
            },
        )
        return (
            base64.b64decode(payload["png"]),
            int(payload["width"]),
            int(payload["height"]),
        )


__all__ = [
    "DAEMON_DIRNAME",
    "DAEMON_FILENAME",
    "DaemonInfo",
    "DaemonUnavailable",
    "NeedVisual",
    "RemoteRenderSession",
    "colight_version",
    "daemon_file_path",
    "discover",
    "discover_for_target",
    "pid_alive",
    "read_daemon_file",
    "request",
    "scene_key",
    "try_pick_at",
    "try_pick_where",
    "try_screenshot",
    "validate_info",
]
