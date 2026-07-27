"""Frame-time probe: sweep a `$state` parameter and report per-stage timings.

Every `$state` change re-evaluates the serialized AST, re-runs ``compileScene``,
and walks the whole components array through the render effect's equality gate.
This module drives that path from the CLI — set the parameter, wait for the
render, repeat — and reads back the client-side probe's samples so the cost of
each stage can be reported as a distribution rather than guessed at.

The client half lives in ``packages/colight/src/js/probe.ts`` and is inert until
``window.__colightProbe`` is set, which :func:`enable_probe` does before the
sweep starts.

**Reading the numbers.** The per-stage figures (``evaluate``, ``compile``,
``equality.deep``, ``equality.filter``, ``render``) are pure in-page CPU spans
and are directly comparable to a frame budget. The ``total`` stage and
``frame.interval_*`` are *not*: the sweep drives one state change per CDP
round-trip and waits for a frame between steps, so both include scheduling and
CLI latency that a user dragging a slider would not pay. Use ``total`` as an
upper bound and the sum of the CPU stages as the load a real interaction puts on
the main thread.

Daemon mode is out of scope: a probe run wants a cold, exclusively-owned tab so
no other request's renders land in the sample buffer. ``colight screenshot
--probe-param`` therefore always takes the direct path.
"""

import statistics
from typing import Any, Dict, List, Sequence

# Stage roster, mirroring PROBE_STAGES in packages/colight/src/js/probe.ts.
STAGE_ORDER = (
    "evaluate",
    "compile",
    "equality.deep",
    "equality.filter",
    "render",
    "total",
)


def parse_range(text: str) -> tuple[float, float]:
    """Parse a ``"lo,hi"`` probe range.

    Args:
        text: Range spec, two comma-separated numbers.

    Returns:
        Tuple of (lo, hi).

    Raises:
        ValueError: If the text is not two parseable numbers.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f'--probe-range must be "lo,hi" (got "{text}")')
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f'--probe-range must be two numbers (got "{text}")')
    return lo, hi


def sweep_values(lo: float, hi: float, frames: int) -> List[float]:
    """Evenly spaced sweep values, inclusive of both endpoints.

    Args:
        lo: Range start.
        hi: Range end.
        frames: Number of steps (must be >= 1).

    Returns:
        List of ``frames`` values from lo to hi.
    """
    if frames < 1:
        raise ValueError("--probe-frames must be >= 1")
    if frames == 1:
        return [lo]
    step = (hi - lo) / (frames - 1)
    return [lo + step * i for i in range(frames)]


def enable_probe(studio: Any) -> None:
    """Turn the client probe on and clear any samples from earlier renders.

    Args:
        studio: A live ``StudioContext``.
    """
    studio.evaluate(
        "(function(){"
        "window.__colightProbe = true;"
        "if (window.__colightProbeApi) {"
        "window.__colightProbeApi.refresh();"
        "window.__colightProbeApi.reset();"
        "return true;"
        "} return false;"
        "})()"
    )


def reset_probe(studio: Any) -> None:
    """Discard samples collected so far, keeping the probe enabled."""
    studio.evaluate(
        "(function(){"
        "if (window.__colightProbeApi) {"
        "window.__colightProbeApi.reset(); return true;"
        "} return false;"
        "})()"
    )


def read_probe(studio: Any) -> Dict[str, Any]:
    """Read the client probe's accumulated samples.

    Args:
        studio: A live ``StudioContext``.

    Returns:
        The raw snapshot dict, or an empty snapshot if the probe is absent
        (e.g. a bundle built before the probe landed).
    """
    snapshot = studio.evaluate(
        "(function(){"
        "if (!window.__colightProbeApi) return null;"
        "return window.__colightProbeApi.snapshot();"
        "})()"
    )
    if not isinstance(snapshot, dict):
        return {
            "enabled": False,
            "stages": {},
            "writes": {"calls": [], "bytes": []},
            "frameIntervals": [],
            "frames": 0,
        }
    return snapshot


def await_frame(studio: Any, timeout_ms: int = 2000) -> bool:
    """Block until the client probe records at least one more frame.

    ``update_state`` only awaits the readiness manager, and the light render
    path (``requestRender``) schedules a bare ``requestAnimationFrame`` without
    registering a pending update — so readiness can resolve before the frame
    actually renders. Left alone, a fast sweep outruns the renderer and its
    rAFs coalesce, measuring far fewer frames than it set values. Waiting on
    the probe's own frame counter makes every swept value cost exactly the
    frame a user dragging the slider would see.

    Args:
        studio: A live ``StudioContext``.
        timeout_ms: Give up after this long (a frame that never renders is a
            real result — it means the state change produced no render).

    Returns:
        True if a new frame was observed, False on timeout.
    """
    js = f"""
    (async function() {{
        const api = window.__colightProbeApi;
        if (!api) return false;
        const start = api.snapshot().frames;
        const deadline = performance.now() + {timeout_ms};
        while (performance.now() < deadline) {{
            await new Promise(r => requestAnimationFrame(r));
            if (api.snapshot().frames > start) return true;
        }}
        return false;
    }})()
    """
    return bool(studio.evaluate(js, await_promise=True))


def _percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of ``values`` (q in 0..1)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _stats(values: Sequence[float]) -> Dict[str, float]:
    """Median / p95 / max / total / mean over a sample list."""
    if not values:
        return {
            "count": 0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
            "total_ms": 0.0,
        }
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
        "total_ms": sum(values),
    }


def summarize(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a raw client snapshot to per-stage aggregate statistics.

    Args:
        snapshot: Raw dict from :func:`read_probe`.

    Returns:
        A ``stats`` section: per-stage median/p95/max/mean/total in ms, plus
        per-frame ``writeBuffer`` call/byte stats and frame-interval timing.
    """
    raw_stages = snapshot.get("stages") or {}
    stages: Dict[str, Dict[str, float]] = {}
    # Report the known roster in a stable order, then anything unexpected.
    names = [n for n in STAGE_ORDER if n in raw_stages]
    names += [n for n in sorted(raw_stages) if n not in STAGE_ORDER]
    for name in names:
        durations = raw_stages[name].get("durations") or []
        entry = _stats(durations)
        entry["occurrences"] = raw_stages[name].get("count", len(durations))
        stages[name] = entry

    writes = snapshot.get("writes") or {}
    call_samples = writes.get("calls") or []
    byte_samples = writes.get("bytes") or []
    intervals = snapshot.get("frameIntervals") or []

    frame_stats = _stats(intervals)
    return {
        "stages": stages,
        "writes": {
            "frames": len(call_samples),
            "calls_median": statistics.median(call_samples) if call_samples else 0,
            "calls_total": sum(call_samples),
            "bytes_median": statistics.median(byte_samples) if byte_samples else 0,
            "bytes_p95": _percentile(byte_samples, 0.95),
            "bytes_total": sum(byte_samples),
        },
        "frame": {
            "count": snapshot.get("frames", 0),
            "interval_median_ms": frame_stats["median_ms"],
            "interval_p95_ms": frame_stats["p95_ms"],
            "interval_max_ms": frame_stats["max_ms"],
            # Effective sustained rate over the swept frames.
            "fps_median": (
                1000.0 / frame_stats["median_ms"] if frame_stats["median_ms"] else 0.0
            ),
            "budget_60fps_ms": 16.67,
            "budget_120fps_ms": 8.33,
        },
    }


def run_sweep(
    scene: Any,
    param: str,
    lo: float,
    hi: float,
    frames: int,
    warmup: int = 2,
) -> Dict[str, Any]:
    """Sweep ``param`` across ``[lo, hi]`` and collect per-stage timings.

    Each step writes the parameter through the same ``updateWithBuffers`` path
    a slider uses and waits for render readiness, so the measured frames are
    exactly the frames a user dragging that slider would produce.

    Args:
        scene: A loaded scene (``RenderSession``) exposing ``.studio``.
        param: `$state` key to drive.
        lo: Sweep start value.
        hi: Sweep end value.
        frames: Number of swept frames to measure.
        warmup: Frames to run and discard before measuring, so first-frame
            pipeline creation and lazy allocation do not pollute the samples.

    Returns:
        A ``probe`` payload: the sweep description and the ``stats`` section.
    """
    studio = scene.studio
    enable_probe(studio)

    values = sweep_values(lo, hi, frames)

    # Warmup: drive a few frames, then discard everything recorded so far.
    for value in sweep_values(lo, hi, max(1, warmup)):
        studio.update_state([{param: value}])
        await_frame(studio)
    reset_probe(studio)

    missed = 0
    for value in values:
        studio.update_state([{param: value}])
        # One frame per swept value: without this the rAF-coalesced light path
        # would merge many state changes into a single render.
        if not await_frame(studio):
            missed += 1

    snapshot = read_probe(studio)
    stats = summarize(snapshot)
    return {
        "param": param,
        "range": [lo, hi],
        "frames": frames,
        "warmup": warmup,
        "measured_frames": snapshot.get("frames", 0),
        # Swept values that produced no render within the timeout. Non-zero
        # means the state change was a genuine no-op for the renderer, not
        # that a frame was lost.
        "frames_without_render": missed,
        **stats,
    }


def format_summary(probe: Dict[str, Any]) -> List[str]:
    """Human-readable lines for a probe payload (the non-``--json`` output)."""
    lines = [
        f"probe: {probe['param']} {probe['range'][0]:g}..{probe['range'][1]:g} "
        f"over {probe['frames']} frames "
        f"({probe.get('measured_frames', 0)} rendered)"
    ]
    stages = probe.get("stages", {})
    if stages:
        width = max(len(n) for n in stages)
        lines.append(f"  {'stage'.ljust(width)}  median      p95    total")
        for name, s in stages.items():
            lines.append(
                f"  {name.ljust(width)}  "
                f"{s['median_ms']:6.2f}ms {s['p95_ms']:6.2f}ms "
                f"{s['total_ms']:7.1f}ms"
            )
    frame = probe.get("frame", {})
    if frame.get("interval_median_ms"):
        lines.append(
            f"  frame interval: {frame['interval_median_ms']:.2f}ms median "
            f"({frame['fps_median']:.0f} fps), p95 {frame['interval_p95_ms']:.2f}ms"
        )
    writes = probe.get("writes", {})
    if writes.get("frames"):
        lines.append(
            f"  writeBuffer: {writes['calls_median']:.0f} calls/frame median, "
            f"{writes['bytes_median'] / 1024:.1f} KiB/frame median, "
            f"{writes['bytes_total'] / 1048576:.1f} MiB total"
        )
    return lines


__all__ = [
    "STAGE_ORDER",
    "parse_range",
    "sweep_values",
    "enable_probe",
    "reset_probe",
    "read_probe",
    "summarize",
    "run_sweep",
    "format_summary",
]
