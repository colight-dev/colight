/**
 * Frame-time probe: opt-in per-stage timing for the `$state` -> render path.
 *
 * Every `$state` change re-evaluates the serialized AST, re-runs `compileScene`,
 * and walks the whole components array through the render effect's equality
 * gate. This module measures how much CPU each of those stages costs, so the
 * question "which stage dominates at which scale" can be answered with numbers
 * instead of reasoning.
 *
 * **Off by default, and free when off.** Nothing here allocates or records
 * until `window.__colightProbe` is truthy. The hot-path helpers (`probeStage`,
 * `probeCountWrite`, `probeFrame`) each begin with a single monomorphic boolean
 * read of the module-level `enabled` flag; when it is false they return the
 * callback's value directly (or return immediately), adding one predictable
 * branch and no allocation, no `performance.now()` call, and no closure. The
 * flag is re-read from `window` by `probeRefresh()`, which the scene calls once
 * per mount, so toggling the global mid-session is possible without leaving a
 * `window` lookup in the per-frame path.
 *
 * Stages are recorded as plain accumulating samples rather than
 * `performance.mark`/`measure` entries: the buffer is bounded, the CLI reads it
 * back in one `evaluate` round-trip, and there is no dependency on the page's
 * performance-entry buffer size.
 */

/** One stage's timing samples, in milliseconds. */
export interface ProbeStageSamples {
  /** Per-occurrence durations in ms, in occurrence order. */
  durations: number[];
  /** Number of occurrences (may exceed `durations.length` if capped). */
  count: number;
}

/** Per-frame buffer-write totals. */
export interface ProbeWriteSamples {
  /** `writeBuffer` call count within each frame. */
  calls: number[];
  /** `writeBuffer` bytes within each frame. */
  bytes: number[];
}

export interface ProbeSnapshot {
  enabled: boolean;
  /** Timing samples keyed by stage name. */
  stages: Record<string, ProbeStageSamples>;
  /** Buffer writes accumulated per completed frame. */
  writes: ProbeWriteSamples;
  /** rAF-to-rAF wall time in ms, one entry per observed frame boundary. */
  frameIntervals: number[];
  /** Frames counted since the last reset. */
  frames: number;
}

/**
 * Stage names recorded by the probe. Kept as a const roster so the CLI and the
 * tests agree on the vocabulary without stringly-typed drift.
 */
export const PROBE_STAGES = {
  /** `$state.evaluate` over the serialized AST (widget.jsx -> eval.js). */
  evaluate: "evaluate",
  /** `compileScene` over the raw components. */
  compile: "compile",
  /** `deepEqualModuloTypedArrays(components, prevComponents)`. */
  equalityDeep: "equality.deep",
  /** `componentsEqualIgnoringFilter` — the second walk on the non-equal path. */
  equalityFilter: "equality.filter",
  /** `renderFrame` from entry through `onSubmittedWorkDone`. */
  render: "render",
  /** State change -> render submitted, the end-to-end span. */
  total: "total",
} as const;

export type ProbeStage = (typeof PROBE_STAGES)[keyof typeof PROBE_STAGES];

/** Cap on retained samples per stage, so a long session cannot grow unbounded. */
const MAX_SAMPLES = 4096;

let enabled = false;
let stages: Record<string, ProbeStageSamples> = Object.create(null);
let writeCalls = 0;
let writeBytes = 0;
const writes: ProbeWriteSamples = { calls: [], bytes: [] };
const frameIntervals: number[] = [];
let lastFrameAt = 0;
let stateChangeAt = 0;
let frames = 0;

function readGlobalFlag(): boolean {
  return (
    typeof window !== "undefined" && Boolean((window as any).__colightProbe)
  );
}

/**
 * Re-read `window.__colightProbe` and latch it into the module flag.
 *
 * Called once per scene mount (and by the CLI before a sweep) so the per-frame
 * helpers never touch `window`.
 *
 * @returns Whether the probe is now enabled.
 */
export function probeRefresh(): boolean {
  enabled = readGlobalFlag();
  return enabled;
}

/** Whether the probe is currently recording. */
export function probeEnabled(): boolean {
  return enabled;
}

/** Clear all accumulated samples (keeps the enabled flag). */
export function probeReset(): void {
  stages = Object.create(null);
  writes.calls.length = 0;
  writes.bytes.length = 0;
  frameIntervals.length = 0;
  writeCalls = 0;
  writeBytes = 0;
  lastFrameAt = 0;
  stateChangeAt = 0;
  frames = 0;
}

function record(stage: string, ms: number): void {
  let entry = stages[stage];
  if (entry === undefined) {
    entry = stages[stage] = { durations: [], count: 0 };
  }
  entry.count += 1;
  if (entry.durations.length < MAX_SAMPLES) entry.durations.push(ms);
}

/**
 * Time `fn` under `stage` and return its result.
 *
 * When the probe is off this is `return fn()` behind one boolean test — no
 * clock read, no allocation — so it is safe on the hot path.
 *
 * @param stage Stage name, normally a {@link PROBE_STAGES} member.
 * @param fn Work to time.
 * @returns Whatever `fn` returns.
 */
export function probeStage<T>(stage: string, fn: () => T): T {
  if (!enabled) return fn();
  const t0 = performance.now();
  try {
    return fn();
  } finally {
    record(stage, performance.now() - t0);
  }
}

/**
 * Time an async `fn` under `stage` and return its result.
 *
 * @param stage Stage name, normally a {@link PROBE_STAGES} member.
 * @param fn Async work to time.
 * @returns A promise for whatever `fn` resolves to.
 */
export async function probeStageAsync<T>(
  stage: string,
  fn: () => Promise<T>,
): Promise<T> {
  if (!enabled) return fn();
  const t0 = performance.now();
  try {
    return await fn();
  } finally {
    record(stage, performance.now() - t0);
  }
}

/** Record a pre-measured duration for `stage`. */
export function probeRecord(stage: string, ms: number): void {
  if (!enabled) return;
  record(stage, ms);
}

/**
 * Count one `writeBuffer` of `bytes` against the frame in progress.
 *
 * Totals are flushed into a per-frame sample by {@link probeFrame}.
 */
export function probeCountWrite(bytes: number): void {
  if (!enabled) return;
  writeCalls += 1;
  writeBytes += bytes;
}

/**
 * Open the end-to-end span: a `$state` write has been applied.
 *
 * The matching close is the next {@link probeFrame}, which attributes the
 * elapsed wall time to the `total` stage. Because React/MobX render
 * asynchronously after the state write returns, this span — not the sum of the
 * CPU stages — is what "state change to render submitted" actually costs.
 */
export function probeStateChange(): void {
  if (!enabled) return;
  stateChangeAt = performance.now();
}

/**
 * Close out the frame in progress: flush the buffer-write totals, close the
 * end-to-end span opened by {@link probeStateChange}, and record the wall-clock
 * interval since the previous frame.
 *
 * Called from the renderer once the frame's GPU work has been submitted.
 */
export function probeFrame(): void {
  if (!enabled) return;
  frames += 1;
  if (stateChangeAt !== 0) {
    record(PROBE_STAGES.total, performance.now() - stateChangeAt);
    stateChangeAt = 0;
  }
  writes.calls.push(writeCalls);
  writes.bytes.push(writeBytes);
  writeCalls = 0;
  writeBytes = 0;
  const now = performance.now();
  if (lastFrameAt !== 0) frameIntervals.push(now - lastFrameAt);
  lastFrameAt = now;
}

/**
 * Wrap `device.queue.writeBuffer` so every call is counted.
 *
 * Wrapping once at device creation is what keeps the ~15 `writeBuffer` call
 * sites in `impl3d.tsx`/`pipelineUtils.ts` untouched.
 *
 * The wrapper is installed unconditionally — the probe is usually enabled
 * *after* the device exists, so deferring the wrap would miss every write.
 * Its body is `probeCountWrite`, a no-op boolean test when the probe is off,
 * and it copies the original's own properties across so a spied
 * `writeBuffer` (`vi.fn`) keeps its mock surface and assertions still see it.
 *
 * @param device The freshly created WebGPU device.
 */
export function probeInstrumentDevice(device: GPUDevice): void {
  const queue = device.queue as any;
  if (queue.__colightProbeWrapped) return;
  const originalFn = queue.writeBuffer;
  if (typeof originalFn !== "function") return;
  const original = originalFn.bind(queue);
  queue.writeBuffer = function (
    buffer: GPUBuffer,
    offset: number,
    data: any,
    dataOffset?: number,
    size?: number,
  ) {
    if (enabled) {
      // Byte count mirrors the WebGPU overload rules: an explicit `size` is in
      // elements for typed arrays and bytes for ArrayBuffers; without it the
      // whole (possibly offset) view is written.
      let bytes: number;
      const bytesPerElement = data?.BYTES_PER_ELEMENT ?? 1;
      if (size !== undefined) {
        bytes = size * bytesPerElement;
      } else if (data?.byteLength !== undefined) {
        bytes = data.byteLength - (dataOffset ?? 0) * bytesPerElement;
      } else {
        bytes = 0;
      }
      probeCountWrite(bytes);
    }
    return original(buffer, offset, data, dataOffset, size);
  };
  // Carry the original's own properties over (`mock`, `mockClear`, … for a
  // test spy) so wrapping stays invisible to whoever installed it.
  for (const key of Reflect.ownKeys(originalFn)) {
    if (key === "length" || key === "name" || key === "prototype") continue;
    const descriptor = Object.getOwnPropertyDescriptor(originalFn, key);
    if (descriptor) {
      Object.defineProperty(queue.writeBuffer, key, descriptor);
    }
  }
  queue.__colightProbeWrapped = true;
}

/** Read out everything recorded so far. */
export function probeSnapshot(): ProbeSnapshot {
  const out: Record<string, ProbeStageSamples> = {};
  for (const key of Object.keys(stages)) {
    out[key] = {
      durations: stages[key].durations.slice(),
      count: stages[key].count,
    };
  }
  return {
    enabled,
    stages: out,
    writes: { calls: writes.calls.slice(), bytes: writes.bytes.slice() },
    frameIntervals: frameIntervals.slice(),
    frames,
  };
}

if (typeof window !== "undefined") {
  // Agent-facing surface: the CLI drives the sweep entirely through these.
  (window as any).__colightProbeApi = {
    refresh: probeRefresh,
    reset: probeReset,
    snapshot: probeSnapshot,
    enabled: probeEnabled,
    stages: PROBE_STAGES,
  };
  // Latch whatever the flag is at load time, so a page that sets
  // `window.__colightProbe = true` before the bundle loads is recording from
  // the first frame.
  probeRefresh();
}
