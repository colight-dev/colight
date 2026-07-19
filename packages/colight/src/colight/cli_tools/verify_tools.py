"""Golden verification for colight visuals: ``colight verify``.

Pins each visual a target produces against a stored golden so agents and CI
share one regression workflow. A golden stores three layers per visual:

* the actual ``.colight`` artifact bytes (compact), so a failure can produce
  a full semantic diff instead of just a hash mismatch;
* the canonicalized-structure hash (``summaries.visual_fingerprint`` — the
  canonical JSON with per-run widget/uuid ids normalized, plus buffer
  content hashes);
* the deterministic screenshot's sha256 and pixel dimensions (optional —
  skipped with a warning when Chrome or the JS bundle is unavailable, or
  explicitly with ``--no-pixels``).

Goldens live under ``<project_root>/tests/goldens/<relpath-of-target>/``
(one directory per target: ``manifest.json`` plus one ``.colight`` per
visual-producing block). The ``tests/`` root mirrors the repo's existing
convention of keeping committed test fixtures/baselines under ``tests``
(cf. ``tests/visual/baselines``); ``--goldens DIR`` overrides it.
"""

import hashlib
import json
import pathlib
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import colight.env as env
import colight.format as colight_format
from colight.chrome_devtools import find_chrome
from colight.runtime.parser import find_project_root

from . import blocks as blocks_mod
from . import diff_tools, screenshot_tools, summaries, targets

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
GOLDENS_DIRNAME = pathlib.Path("tests") / "goldens"
MAX_CHANGED_PATHS = 10

DEFAULT_SCREENSHOT_WIDTH = 800
DEFAULT_SCREENSHOT_DPR = 1.0


class RenderSessionLike(Protocol):
    """What the screenshot layer needs from a render session.

    Satisfied by :class:`screenshot_tools.RenderSession` (local Chrome) and
    by the daemon client's remote session (renders on the daemon's pooled
    Chrome).
    """

    def __enter__(self) -> "RenderSessionLike": ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

    def render(
        self,
        data: Dict[str, Any],
        buffers: List[bytes],
        height: Optional[int] = None,
    ) -> Tuple[bytes, int, int]: ...


SessionFactory = Callable[[int, float], RenderSessionLike]
"""Callable of (width, dpr) producing a render session.

When a factory is supplied by the caller, the local Chrome/JS-bundle
availability probe (:func:`pixels_unavailable_reason`) is skipped — the
caller vouches that the factory can render (e.g. via the colight daemon).
"""


def _local_session_factory(width: int, dpr: float) -> RenderSessionLike:
    return screenshot_tools.RenderSession(width=width, dpr=dpr)


class VerifyError(ValueError):
    """Raised for unrecoverable verify problems (exit code 2)."""


def default_goldens_root(target: pathlib.Path) -> pathlib.Path:
    """Default goldens root for a target: ``<project_root>/tests/goldens``."""
    return find_project_root(target.resolve()) / GOLDENS_DIRNAME


def project_relpath(target: pathlib.Path) -> pathlib.Path:
    """A target's path relative to its project root.

    ``find_project_root`` always returns an ancestor of the target, so the
    relative path always exists.
    """
    resolved = target.resolve()
    return resolved.relative_to(find_project_root(resolved))


def golden_dir(target: pathlib.Path, goldens_root: pathlib.Path) -> pathlib.Path:
    """Directory holding a target's goldens (mirrors its project relpath)."""
    return goldens_root / project_relpath(target)


def pixels_unavailable_reason() -> Optional[str]:
    """Why deterministic screenshots cannot be produced here (None if ok)."""
    widget_path = env.WIDGET_PATH
    if not (isinstance(widget_path, pathlib.Path) and widget_path.exists()):
        return "colight JS bundle not built (js-dist missing)"
    try:
        find_chrome()
    except FileNotFoundError:
        return "Chrome not found"
    return None


def _load_current_visuals(
    target: pathlib.Path,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load a target's current visuals.

    Returns:
        Tuple of (kind, visuals, errors); each visual has ``id`` (block id,
        or ``"artifact"`` for ``.colight`` targets), optional ``lines``,
        ``data`` and ``buffers``.

    Raises:
        VerifyError: Unsupported target or empty ``.colight`` file.
    """
    try:
        loaded = targets.load_target(target)
    except ValueError as e:
        raise VerifyError(str(e)) from e
    if loaded.kind == "colight":
        visual = loaded.visuals[0]
        return "colight", [{"id": "artifact", **visual}], []
    visuals = [
        {
            "id": item["block"],
            "lines": item["lines"],
            "data": item["data"],
            "buffers": item["buffers"],
        }
        for item in loaded.visuals
    ]
    return "py", visuals, loaded.errors


def _artifact_bytes(visual: Dict[str, Any]) -> bytes:
    return colight_format.create_bytes(visual["data"], visual["buffers"])


def _structure_hash(artifact: bytes) -> str:
    return summaries.visual_fingerprint(artifact)


def _render_screenshot(
    session: RenderSessionLike, visual: Dict[str, Any]
) -> Dict[str, Any]:
    """Deterministic screenshot layer for a visual (sha256 + dimensions)."""
    png, pixel_width, pixel_height = session.render(
        visual["data"], visual["buffers"], height=None
    )
    return {
        "sha256": hashlib.sha256(png).hexdigest(),
        "width": pixel_width,
        "height": pixel_height,
    }


def _pair_with_goldens(
    visuals: List[Dict[str, Any]], golden_blocks: List[Dict[str, Any]]
) -> Tuple[List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]], List[Dict[str, Any]]]:
    """Pair current visuals to golden entries: by id first, then positionally.

    Positional pairing of the leftovers (see
    :func:`blocks.pair_by_stable_id`) means an *edited* block (whose stable
    id changed with its source) still gets a semantic diff against its old
    golden instead of degrading to "new" + "removed".

    Returns:
        Tuple of (pairs, orphaned golden entries).
    """
    matches, orphaned = blocks_mod.pair_by_stable_id(
        [visual["id"] for visual in visuals], golden_blocks
    )
    return list(zip(visuals, matches)), orphaned


def _diff_summary(
    golden_artifact: bytes, visual: Dict[str, Any], epsilon: float
) -> Dict[str, Any]:
    """Compact semantic-diff summary between a golden and the current visual."""
    golden_data, golden_buffers = summaries.parse_colight_bytes(golden_artifact)
    pair = diff_tools.diff_visual_pair(
        {"data": golden_data, "buffers": golden_buffers},
        {"data": visual["data"], "buffers": visual["buffers"]},
        epsilon,
    )
    arrays = pair["arrays"]["changed"]
    summary: Dict[str, Any] = {
        "arrays_changed": len(arrays),
        "components": {
            k: len(pair["components"][k]) for k in ("added", "removed", "changed")
        },
        "values_changed": len(pair["values"]["changed"])
        + pair["values"].get("truncated", {}).get("changed", 0),
        "state": pair["state"],
    }
    max_abs: Optional[float] = None
    max_path: Optional[str] = None
    mean_values: List[float] = []
    for entry in arrays:
        delta = entry.get("max_abs_delta")
        if delta is not None and (max_abs is None or delta > max_abs):
            max_abs = delta
            max_path = entry["path"]
        if "mean_abs_delta" in entry:
            mean_values.append(entry["mean_abs_delta"])
    if max_abs is not None:
        summary["max_abs_delta"] = max_abs
        summary["max_abs_delta_path"] = max_path
    if mean_values:
        summary["mean_abs_delta"] = sum(mean_values) / len(mean_values)
    changed_paths = [entry["path"] for entry in arrays]
    changed_paths += [item["path"] for item in pair["values"]["changed"]]
    changed_paths += [item["path"] for item in pair["components"]["changed"]]
    if changed_paths:
        summary["changed_paths"] = changed_paths[:MAX_CHANGED_PATHS]
        if len(changed_paths) > MAX_CHANGED_PATHS:
            summary["changed_paths_truncated"] = len(changed_paths) - MAX_CHANGED_PATHS
    return summary


def _load_manifest(directory: pathlib.Path) -> Optional[Dict[str, Any]]:
    try:
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("version") != MANIFEST_VERSION:
        return None
    return manifest


def _compute_entries(
    visuals: List[Dict[str, Any]],
    session: Optional[RenderSessionLike],
) -> List[Dict[str, Any]]:
    """Current-state golden entries (artifact bytes and payloads in-memory).

    Args:
        visuals: Current visuals (``id``, ``data``, ``buffers``, ...).
        session: Shared render session for the screenshot layer, or None to
            skip pixels.
    """
    entries: List[Dict[str, Any]] = []
    for visual in visuals:
        artifact = _artifact_bytes(visual)
        entry: Dict[str, Any] = {
            "id": visual["id"],
            "artifact_bytes": artifact,
            "structure_hash": _structure_hash(artifact),
            "data": visual["data"],
            "buffers": visual["buffers"],
        }
        if "lines" in visual:
            entry["lines"] = visual["lines"]
        if session is not None:
            entry["screenshot"] = _render_screenshot(session, visual)
        entries.append(entry)
    return entries


def verify_target(
    target: pathlib.Path,
    goldens_root: Optional[pathlib.Path] = None,
    pixels: bool = True,
    epsilon: float = diff_tools.DEFAULT_EPSILON,
    session_factory: Optional[SessionFactory] = None,
) -> Dict[str, Any]:
    """Verify a target against its stored goldens.

    Args:
        target: A ``.py`` document or ``.colight`` artifact.
        goldens_root: Golden storage root (default:
            ``<project_root>/tests/goldens``).
        pixels: Also verify the screenshot layer (auto-skipped with a
            warning when Chrome or the JS bundle is unavailable).
        epsilon: Numeric threshold for the semantic diff on mismatch.
        session_factory: Render-session provider for the screenshot layer
            (default: local headless Chrome; the daemon client passes a
            remote session and thereby skips the local availability probe).

    Returns:
        Per-target payload with ``status`` (``match`` | ``mismatch`` |
        ``no-goldens`` | ``error``) and per-block layer results.
    """
    root = goldens_root or default_goldens_root(target)
    directory = golden_dir(target, root)
    payload: Dict[str, Any] = {"target": str(target), "goldens": str(directory)}
    warnings: List[str] = []

    kind, visuals, errors = _load_current_visuals(target)
    payload["kind"] = kind
    if errors:
        payload["status"] = "error"
        payload["errors"] = errors
        return payload

    manifest = _load_manifest(directory)
    if manifest is None:
        payload["status"] = "no-goldens"
        payload["hint"] = "run `colight verify --update` to create goldens"
        return payload

    stored_shot = manifest.get("screenshot") or {}
    width = stored_shot.get("width", DEFAULT_SCREENSHOT_WIDTH)
    dpr = stored_shot.get("dpr", DEFAULT_SCREENSHOT_DPR)

    reason = (
        pixels_unavailable_reason() if (pixels and session_factory is None) else None
    )
    check_pixels = pixels and reason is None
    if pixels and reason is not None:
        warnings.append(f"pixels skipped: {reason}")

    pairs, orphaned = _pair_with_goldens(visuals, manifest.get("blocks", []))

    blocks_out: List[Dict[str, Any]] = []
    mismatched = False
    factory = session_factory or _local_session_factory
    # One shared render session per target: the Chrome tab, HTTP server and
    # JS bundle load are paid once, not per visual. Opened lazily on the
    # first screenshot; a no-op to close if pixels were never rendered.
    with factory(width, dpr) as session:
        for visual, golden in pairs:
            entry: Dict[str, Any] = {"id": visual["id"]}
            if "lines" in visual:
                entry["lines"] = visual["lines"]
            if golden is None:
                entry["status"] = "new"
                entry["hint"] = "no golden for this visual (run --update)"
                mismatched = True
                blocks_out.append(entry)
                continue
            if golden["id"] != visual["id"]:
                entry["golden_id"] = golden["id"]

            artifact = _artifact_bytes(visual)
            structure_hash = _structure_hash(artifact)
            structure_match = structure_hash == golden["structure_hash"]
            entry["structure"] = {
                "hash": structure_hash,
                "golden_hash": golden["structure_hash"],
                "match": structure_match,
            }
            if not structure_match:
                golden_path = directory / golden["artifact"]
                try:
                    entry["diff"] = _diff_summary(
                        golden_path.read_bytes(), visual, epsilon
                    )
                except OSError:
                    warnings.append(f"golden artifact missing: {golden_path}")

            pixel_match: Optional[bool] = None
            golden_shot = golden.get("screenshot")
            if check_pixels and golden_shot:
                shot = _render_screenshot(session, visual)
                pixel_match = shot["sha256"] == golden_shot["sha256"]
                entry["pixels"] = {
                    "sha256": shot["sha256"],
                    "golden_sha256": golden_shot["sha256"],
                    "width": shot["width"],
                    "height": shot["height"],
                    "match": pixel_match,
                }
                if (shot["width"], shot["height"]) != (
                    golden_shot.get("width"),
                    golden_shot.get("height"),
                ):
                    entry["pixels"]["golden_size"] = [
                        golden_shot.get("width"),
                        golden_shot.get("height"),
                    ]
            elif check_pixels and not golden_shot:
                entry["pixels"] = {"match": None, "note": "golden has no screenshot"}

            if not structure_match:
                entry["status"] = "structure-changed"
                mismatched = True
            elif pixel_match is False:
                entry["status"] = "pixels-changed"
                mismatched = True
            else:
                entry["status"] = "match"
            blocks_out.append(entry)

    for golden in orphaned:
        blocks_out.append({"id": golden["id"], "status": "removed"})
        mismatched = True

    payload["blocks"] = blocks_out
    payload["status"] = "mismatch" if mismatched else "match"
    if warnings:
        payload["warnings"] = warnings
    return payload


def update_target(
    target: pathlib.Path,
    goldens_root: Optional[pathlib.Path] = None,
    pixels: bool = True,
    epsilon: float = diff_tools.DEFAULT_EPSILON,
    session_factory: Optional[SessionFactory] = None,
) -> Dict[str, Any]:
    """Write/refresh a target's goldens, reporting changes vs the previous set.

    Args:
        target: A ``.py`` document or ``.colight`` artifact.
        goldens_root: Golden storage root override.
        pixels: Also store the screenshot layer (auto-skipped with a warning
            when Chrome or the JS bundle is unavailable).
        epsilon: Numeric threshold for the reported diff summaries.
        session_factory: Render-session provider for the screenshot layer
            (default: local headless Chrome).

    Returns:
        Per-target payload; block statuses are ``added`` | ``updated`` |
        ``unchanged`` | ``removed``, with the same semantic diff summary a
        failing verify would show, so updating is an informed act.
    """
    root = goldens_root or default_goldens_root(target)
    directory = golden_dir(target, root)
    payload: Dict[str, Any] = {"target": str(target), "goldens": str(directory)}
    warnings: List[str] = []

    kind, visuals, errors = _load_current_visuals(target)
    payload["kind"] = kind
    if errors:
        payload["status"] = "error"
        payload["errors"] = errors
        return payload
    if not visuals:
        payload["status"] = "error"
        payload["errors"] = [
            {"error": {"type": "ValueError", "message": "target produces no visuals"}}
        ]
        return payload

    reason = (
        pixels_unavailable_reason() if (pixels and session_factory is None) else None
    )
    check_pixels = pixels and reason is None
    if pixels and reason is not None:
        warnings.append(f"pixels skipped: {reason}")

    previous = _load_manifest(directory)
    prev_shot = (previous or {}).get("screenshot") or {}
    width = prev_shot.get("width", DEFAULT_SCREENSHOT_WIDTH)
    dpr = prev_shot.get("dpr", DEFAULT_SCREENSHOT_DPR)

    factory = session_factory or _local_session_factory
    with factory(width, dpr) as session:
        entries = _compute_entries(visuals, session if check_pixels else None)
    pairs, orphaned = _pair_with_goldens(entries, (previous or {}).get("blocks", []))

    blocks_out: List[Dict[str, Any]] = []
    for entry, golden in pairs:
        report: Dict[str, Any] = {"id": entry["id"]}
        if golden is None:
            report["status"] = "added"
        else:
            if golden["id"] != entry["id"]:
                report["golden_id"] = golden["id"]
            structure_changed = entry["structure_hash"] != golden["structure_hash"]
            golden_shot = golden.get("screenshot")
            new_shot = entry.get("screenshot")
            pixels_changed = (
                golden_shot is not None
                and new_shot is not None
                and new_shot["sha256"] != golden_shot["sha256"]
            )
            if structure_changed:
                report["status"] = "updated"
                report["layer"] = "structure"
                golden_path = directory / golden["artifact"]
                try:
                    report["diff"] = _diff_summary(
                        golden_path.read_bytes(),
                        {"data": entry["data"], "buffers": entry["buffers"]},
                        epsilon,
                    )
                except OSError:
                    warnings.append(f"previous golden artifact missing: {golden_path}")
            elif pixels_changed:
                report["status"] = "updated"
                report["layer"] = "pixels"
            else:
                report["status"] = "unchanged"
        blocks_out.append(report)
    for golden in orphaned:
        blocks_out.append({"id": golden["id"], "status": "removed"})

    # Write the new goldens: manifest + one artifact per visual.
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*.colight"):
        stale.unlink()
    manifest_blocks: List[Dict[str, Any]] = []
    for entry in entries:
        artifact_name = f"{entry['id']}.colight"
        (directory / artifact_name).write_bytes(entry["artifact_bytes"])
        manifest_entry: Dict[str, Any] = {
            "id": entry["id"],
            "artifact": artifact_name,
            "structure_hash": entry["structure_hash"],
        }
        if "lines" in entry:
            manifest_entry["lines"] = entry["lines"]
        if "screenshot" in entry:
            manifest_entry["screenshot"] = entry["screenshot"]
        manifest_blocks.append(manifest_entry)
    manifest: Dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "target": str(project_relpath(target)),
        "kind": kind,
        "blocks": manifest_blocks,
    }
    if check_pixels:
        manifest["screenshot"] = {"width": width, "dpr": dpr}
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )

    payload["blocks"] = blocks_out
    payload["status"] = "updated"
    if warnings:
        payload["warnings"] = warnings
    return payload


def run_verify(
    targets: List[pathlib.Path],
    goldens_root: Optional[pathlib.Path] = None,
    update: bool = False,
    pixels: bool = True,
    epsilon: float = diff_tools.DEFAULT_EPSILON,
    session_factory: Optional[SessionFactory] = None,
) -> Tuple[Dict[str, Any], int]:
    """Verify (or update goldens for) a list of targets.

    Returns:
        Tuple of (payload, exit code). Exit codes: 0 all match (or update
        succeeded), 1 mismatches, 2 error, 3 no goldens found.
    """
    # Two distinct targets must never share a goldens directory (possible
    # when one --goldens root is applied to targets from different project
    # roots whose project-relative paths coincide): verifying would compare
    # against the wrong goldens and updating would silently clobber them.
    directories: Dict[pathlib.Path, pathlib.Path] = {}
    for target in targets:
        root = goldens_root or default_goldens_root(target)
        directory = golden_dir(target, root).resolve()
        other = directories.get(directory)
        if other is not None and other != target.resolve():
            message = (
                f"goldens collision: {other} and {target.resolve()} both map to "
                f"{directory}; use per-target --goldens roots"
            )
            error_result = {
                "target": str(target),
                "status": "error",
                "errors": [{"error": {"type": "ValueError", "message": message}}],
            }
            return {"targets": [error_result], "ok": False}, 2
        directories[directory] = target.resolve()

    results: List[Dict[str, Any]] = []
    for target in targets:
        try:
            if update:
                results.append(
                    update_target(
                        target,
                        goldens_root=goldens_root,
                        pixels=pixels,
                        epsilon=epsilon,
                        session_factory=session_factory,
                    )
                )
            else:
                results.append(
                    verify_target(
                        target,
                        goldens_root=goldens_root,
                        pixels=pixels,
                        epsilon=epsilon,
                        session_factory=session_factory,
                    )
                )
        except VerifyError as e:
            results.append(
                {
                    "target": str(target),
                    "status": "error",
                    "errors": [{"error": {"type": "ValueError", "message": str(e)}}],
                }
            )

    statuses = {result["status"] for result in results}
    if "error" in statuses:
        exit_code = 2
    elif "mismatch" in statuses:
        exit_code = 1
    elif "no-goldens" in statuses:
        exit_code = 3
    else:
        exit_code = 0
    payload = {
        "targets": results,
        "ok": exit_code == 0,
    }
    return payload, exit_code


__all__ = [
    "DEFAULT_SCREENSHOT_DPR",
    "DEFAULT_SCREENSHOT_WIDTH",
    "RenderSessionLike",
    "SessionFactory",
    "VerifyError",
    "default_goldens_root",
    "golden_dir",
    "pixels_unavailable_reason",
    "run_verify",
    "update_target",
    "verify_target",
]
