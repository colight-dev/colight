import React, {
  startTransition,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactDOM from "react-dom/client";
import { $StateContext } from "../context";
import { SceneInner } from "../scene3d/impl3d";
import {
  formatInstanceLimit,
  parseScene3DDebugOptions,
  type Scene3DDebugRegistry,
  type Scene3DDebugState,
} from "../scene3d/debug";
import { Ellipsoid, deco } from "../scene3d/scene3d";
import type { ComponentConfig } from "../scene3d/components";
import {
  DEFAULT_SCENE3D_GEOMETRY_OPTIONS,
  type PrimitiveImplementationMode,
  type Scene3DGeometryOptions,
} from "../scene3d/types";
import { createStateStore } from "../widget";

type RunStatus =
  | "idle"
  | "precomputing"
  | "warming"
  | "running"
  | "completed"
  | "error";

interface BenchResult {
  renderMode: PrimitiveImplementationMode;
  count: number;
  sampleCount: number;
  precomputeMs: number;
  averageMs: number;
  medianMs: number;
  minMs: number;
  maxMs: number;
  averageFps: number;
  medianFps: number;
  datasetBytes: number;
}

interface BenchRunState {
  status: RunStatus;
  counts: number[];
  frameCount: number;
  renderMode: PrimitiveImplementationMode;
  currentCountIndex: number;
  currentCount: number | null;
  progressLabel: string;
  results: BenchResult[];
  error?: string;
  gpuInfo?: Record<string, unknown>;
}

type PreviewStatus = "idle" | "preparing" | "ready" | "error";

interface Dataset {
  count: number;
  frameCount: number;
  centersByFrame: Float32Array[];
  colors: Float32Array;
  halfSize: [number, number, number];
  datasetBytes: number;
}

interface BenchApi {
  run: (options?: Partial<BenchOptions>) => Promise<BenchRunState>;
  getState: () => BenchRunState;
}

interface BenchOptions {
  counts: number[];
  frameCount: number;
  width: number;
  height: number;
  autorun: boolean;
  renderMode: PrimitiveImplementationMode;
  ellipsoidStacks: number;
  ellipsoidSlices: number;
}

interface WindowWithBench extends Window {
  __COLIGHT_ELLIPSOID_BENCH__?: BenchApi;
  __COLIGHT_ELLIPSOID_BENCH_DEBUG__?: {
    getState: () => {
      bench: BenchRunState;
      scene3d: Scene3DDebugRegistry | null;
    };
  };
  __COLIGHT_SCENE3D_DEBUG__?: Scene3DDebugRegistry;
}

const DEFAULT_COUNTS = [
  250_000, 500_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000,
];
const DEFAULT_FRAME_COUNT = 60;
const DEFAULT_WIDTH = 1280;
const DEFAULT_HEIGHT = 900;
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

const DEFAULT_HALF_SIZE: [number, number, number] = [0.018, 0.03, 0.018];
const HOVER_ELLIPSOID_COLOR: [number, number, number] = [1, 0.24, 0.08];
const PROXY_TRIANGLES_PER_IMPOSTOR = 2;

function parseInteger(value: string | null, fallback: number) {
  if (!value) return fallback;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseInputInteger(value: string, fallback: number) {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseSegmentCount(value: string | null, fallback: number) {
  return Math.max(3, parseInteger(value, fallback));
}

function parseCounts(value: string | null) {
  if (!value) return DEFAULT_COUNTS;
  const parsed = value
    .split(",")
    .map((part) => Number.parseInt(part.trim().replace(/_/g, ""), 10))
    .filter((item) => Number.isFinite(item) && item > 0);
  return parsed.length ? parsed : DEFAULT_COUNTS;
}

function getInitialOptions(): BenchOptions {
  const params = new URLSearchParams(window.location.search);
  const renderMode = params.get("render_mode");
  return {
    counts: parseCounts(params.get("counts")),
    frameCount: parseInteger(params.get("frames"), DEFAULT_FRAME_COUNT),
    width: parseInteger(params.get("width"), DEFAULT_WIDTH),
    height: parseInteger(params.get("height"), DEFAULT_HEIGHT),
    autorun: params.get("autorun") === "1",
    renderMode: renderMode === "impostor" ? "impostor" : "mesh",
    ellipsoidStacks: parseSegmentCount(
      params.get("ellipsoid_stacks"),
      DEFAULT_SCENE3D_GEOMETRY_OPTIONS.ellipsoidStacks,
    ),
    ellipsoidSlices: parseSegmentCount(
      params.get("ellipsoid_slices"),
      DEFAULT_SCENE3D_GEOMETRY_OPTIONS.ellipsoidSlices,
    ),
  };
}

function round(value: number) {
  return Math.round(value * 100) / 100;
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function formatCount(count: number) {
  if (count >= 1_000_000) {
    return `${round(count / 1_000_000)}m`;
  }
  if (count >= 1_000) {
    return `${round(count / 1_000)}k`;
  }
  return `${count}`;
}

function formatBytes(bytes: number) {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${round(value)} ${units[unitIndex]}`;
}

function countSphereTriangles(stacks: number, slices: number) {
  return stacks * slices * 2;
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);

  switch (i % 6) {
    case 0:
      return [v, t, p];
    case 1:
      return [q, v, p];
    case 2:
      return [p, v, t];
    case 3:
      return [p, q, v];
    case 4:
      return [t, p, v];
    default:
      return [v, p, q];
  }
}

function pingPong(frame: number, frameCount: number) {
  if (frameCount <= 1) return 0;
  const normalized = frame / (frameCount - 1);
  return normalized <= 0.5 ? normalized * 2 : (1 - normalized) * 2;
}

function nextPaint() {
  return new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

async function describeGPU() {
  if (!navigator.gpu) {
    return {
      supported: false,
      reason: "navigator.gpu is not available",
    };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });

    if (!adapter) {
      return {
        supported: false,
        reason: "No WebGPU adapter was returned",
      };
    }

    let info: Record<string, unknown> = {};
    if ("info" in adapter && adapter.info) {
      info = { ...(adapter.info as Record<string, unknown>) };
    } else if ("requestAdapterInfo" in adapter) {
      try {
        info = { ...((await (adapter as any).requestAdapterInfo()) || {}) };
      } catch (_err) {
        info = {};
      }
    }

    return {
      supported: true,
      ...info,
    };
  } catch (error) {
    return {
      supported: false,
      reason: error instanceof Error ? error.message : String(error),
    };
  }
}

async function buildDataset(
  count: number,
  frameCount: number,
  onProgress: (label: string) => void,
): Promise<Dataset> {
  const startedAt = performance.now();
  const centersByFrame: Float32Array[] = [];
  const colors = new Float32Array(count * 3);
  const baseX = new Float32Array(count);
  const baseY = new Float32Array(count);
  const baseZ = new Float32Array(count);
  const travelX = new Float32Array(count);
  const travelY = new Float32Array(count);
  const travelZ = new Float32Array(count);
  const curlX = new Float32Array(count);
  const curlY = new Float32Array(count);
  const curlZ = new Float32Array(count);
  const phaseOffset = new Float32Array(count);
  const chunkSize = 65_536;

  onProgress(
    `Generating static ellipsoid data for ${formatCount(count)} ellipsoids`,
  );
  for (let start = 0; start < count; start += chunkSize) {
    const end = Math.min(start + chunkSize, count);

    for (let i = start; i < end; i++) {
      const u = i / count;
      const ring = i % 2048;
      const layer = Math.floor(i / 2048);
      const angle = ring * GOLDEN_ANGLE + layer * 0.011;
      const radialWave = Math.sin(u * 29) * 0.55 + Math.cos(u * 83) * 0.35;
      const radius = 1.3 + radialWave;
      const y = 5.0 * (u - 0.5) + 0.35 * Math.sin(angle * 3);
      const phase = angle * 0.35 + u * 18;
      const amplitude =
        0.05 + 0.12 * (0.5 + 0.5 * Math.sin(u * 13 + angle * 0.2));

      baseX[i] = radius * Math.cos(angle);
      baseY[i] = y;
      baseZ[i] = radius * Math.sin(angle) + 0.25 * Math.cos(u * 41);

      travelX[i] = amplitude * Math.cos(phase);
      travelY[i] = amplitude * 0.55 * Math.sin(phase * 0.5);
      travelZ[i] = amplitude * Math.sin(phase);

      curlX[i] = amplitude * 0.35 * Math.cos(phase + Math.PI / 2);
      curlY[i] = amplitude * 0.6 * Math.cos(phase * 0.75);
      curlZ[i] = amplitude * 0.35 * Math.sin(phase + Math.PI / 2);
      phaseOffset[i] = phase;

      const hue = (u * 0.73 + (ring % 19) / 19) % 1;
      const saturation =
        0.5 + 0.2 * (0.5 + 0.5 * Math.sin(u * 17 + angle * 0.1));
      const value = 0.55 + 0.35 * (0.5 + 0.5 * Math.cos(u * 11));
      const rgb = hsvToRgb(hue, saturation, value);
      colors[i * 3 + 0] = rgb[0];
      colors[i * 3 + 1] = rgb[1];
      colors[i * 3 + 2] = rgb[2];
    }

    onProgress(
      `Static data ${Math.round((end / count) * 100)}% for ${formatCount(count)} ellipsoids`,
    );
    await nextPaint();
  }

  onProgress(
    `Precomputing ${frameCount} motion frames for ${formatCount(count)} ellipsoids`,
  );
  for (let frame = 0; frame < frameCount; frame++) {
    const centersFrame = new Float32Array(count * 3);
    const frameT = frameCount <= 1 ? 0 : frame / (frameCount - 1);
    const tween = pingPong(frame, frameCount);
    const swirlPhase = frameT * Math.PI * 2;

    for (let start = 0; start < count; start += chunkSize) {
      const end = Math.min(start + chunkSize, count);

      for (let i = start; i < end; i++) {
        const curl = tween * Math.sin(phaseOffset[i] + swirlPhase);
        const outOffset = i * 3;
        centersFrame[outOffset + 0] =
          baseX[i] + travelX[i] * tween + curlX[i] * curl;
        centersFrame[outOffset + 1] =
          baseY[i] + travelY[i] * tween + curlY[i] * curl;
        centersFrame[outOffset + 2] =
          baseZ[i] + travelZ[i] * tween + curlZ[i] * curl;
      }
    }

    centersByFrame.push(centersFrame);
    onProgress(
      `Motion frames ${frame + 1}/${frameCount} for ${formatCount(count)} ellipsoids`,
    );
    await nextPaint();
  }

  const datasetBytes =
    colors.byteLength +
    centersByFrame.reduce((total, frame) => total + frame.byteLength, 0);

  console.log(
    `[ellipsoid-bench] Precomputed ${frameCount} frames for ${formatCount(count)} in ${round(performance.now() - startedAt)} ms (${formatBytes(datasetBytes)})`,
  );

  return {
    count,
    frameCount,
    centersByFrame,
    colors,
    halfSize: DEFAULT_HALF_SIZE,
    datasetBytes,
  };
}

function SceneBench({
  dataset,
  frameIndex,
  width,
  height,
  renderMode,
  geometryOptions,
  hoveredIndex,
  onHoverChange,
  onFrameRendered,
}: {
  dataset: Dataset | null;
  frameIndex: number;
  width: number;
  height: number;
  renderMode: PrimitiveImplementationMode;
  geometryOptions: Pick<
    Scene3DGeometryOptions,
    "ellipsoidStacks" | "ellipsoidSlices"
  >;
  hoveredIndex: number | null;
  onHoverChange: (index: number | null) => void;
  onFrameRendered: (timestamp: number) => void;
}) {
  const components = useMemo<ComponentConfig[]>(() => {
    if (!dataset) return [];

    return [
      Ellipsoid({
        centers: dataset.centersByFrame[frameIndex],
        colors: dataset.colors,
        alpha: 1,
        render_mode: renderMode,
        half_size: dataset.halfSize,
        quaternion: [0, 0, 0, 1],
        decorations:
          hoveredIndex == null
            ? undefined
            : [
                deco(hoveredIndex, {
                  color: HOVER_ELLIPSOID_COLOR,
                }),
              ],
        onHover: onHoverChange,
      }),
    ];
  }, [dataset, frameIndex, hoveredIndex, onHoverChange, renderMode]);

  if (!dataset) {
    return (
      <div className="scene-placeholder">
        Precompute a dataset to start the benchmark.
      </div>
    );
  }

  return (
    <SceneInner
      key={`ellipsoid-${renderMode}-${geometryOptions.ellipsoidStacks}x${geometryOptions.ellipsoidSlices}`}
      components={components}
      geometryOptions={geometryOptions}
      containerWidth={width}
      containerHeight={height}
      onFrameRendered={onFrameRendered}
      onReady={() => {}}
    />
  );
}

function App() {
  const initialOptionsRef = useRef(getInitialOptions());
  const debugOptions = useMemo(() => parseScene3DDebugOptions(), []);
  const [countsText, setCountsText] = useState(
    initialOptionsRef.current.counts.join(","),
  );
  const [frameCount, setFrameCount] = useState(
    initialOptionsRef.current.frameCount,
  );
  const [width, setWidth] = useState(initialOptionsRef.current.width);
  const [height, setHeight] = useState(initialOptionsRef.current.height);
  const [renderMode, setRenderMode] = useState<PrimitiveImplementationMode>(
    initialOptionsRef.current.renderMode,
  );
  const [ellipsoidStacks, setEllipsoidStacks] = useState(
    initialOptionsRef.current.ellipsoidStacks,
  );
  const [ellipsoidSlices, setEllipsoidSlices] = useState(
    initialOptionsRef.current.ellipsoidSlices,
  );
  const [$state, set$State] = useState<any>(null);
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playFps, setPlayFps] = useState(30);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [previewStatus, setPreviewStatus] = useState<PreviewStatus>("idle");
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewCount, setPreviewCount] = useState<number | null>(null);
  const [sceneDebugState, setSceneDebugState] =
    useState<Scene3DDebugState | null>(null);
  const [runState, setRunState] = useState<BenchRunState>({
    status: "idle",
    counts: initialOptionsRef.current.counts,
    frameCount: initialOptionsRef.current.frameCount,
    renderMode: initialOptionsRef.current.renderMode,
    currentCountIndex: -1,
    currentCount: null,
    progressLabel: "Waiting to start",
    results: [],
  });

  const runStateRef = useRef(runState);
  const activeRunIdRef = useRef(0);
  const previewRequestIdRef = useRef(0);
  const sessionRef = useRef<{
    runId: number;
    order: number[];
    orderIndex: number;
    requestStartedAt: number | null;
    samples: number[];
    precomputeMs: number;
    count: number;
    datasetBytes: number;
  } | null>(null);

  useEffect(() => {
    createStateStore({ state: {} }).then(set$State);
  }, []);

  useEffect(() => {
    runStateRef.current = runState;
  }, [runState]);

  const setProgress = useCallback((label: string) => {
    setRunState((current) => ({
      ...current,
      progressLabel: label,
    }));
  }, []);

  const finishRun = useCallback((nextState: BenchRunState) => {
    setRunState(nextState);
  }, []);

  const benchmarkActive =
    runState.status === "precomputing" ||
    runState.status === "warming" ||
    runState.status === "running";

  const preparePreview = useCallback(
    async ({ autoplay = false }: { autoplay?: boolean } = {}) => {
      const counts = parseCounts(countsText);
      const count = counts[0];
      const requestId = ++previewRequestIdRef.current;

      setIsPlaying(false);
      setHoveredIndex(null);
      setPreviewStatus("preparing");
      setPreviewError(null);
      setPreviewCount(count);

      try {
        const nextDataset = await buildDataset(count, frameCount, (label) => {
          if (previewRequestIdRef.current !== requestId) return;
          setPreviewStatus("preparing");
          setRunState((current) => ({
            ...current,
            renderMode,
            progressLabel: `Preview (${renderMode}): ${label}`,
          }));
        });

        if (previewRequestIdRef.current !== requestId) {
          return;
        }

        setDataset(nextDataset);
        setFrameIndex(0);
        setPreviewStatus("ready");
        setRunState((current) => ({
          ...current,
          renderMode,
          progressLabel: `Preview ready for ${formatCount(count)} (${renderMode})`,
        }));
        if (autoplay) {
          setIsPlaying(true);
        }
      } catch (error) {
        if (previewRequestIdRef.current !== requestId) {
          return;
        }
        setPreviewStatus("error");
        setPreviewError(error instanceof Error ? error.message : String(error));
      }
    },
    [countsText, frameCount, renderMode],
  );

  const queueMeasuredFrame = useCallback(
    (session: NonNullable<typeof sessionRef.current>) => {
      const nextFrame = session.order[session.orderIndex];
      session.orderIndex += 1;

      requestAnimationFrame(() => {
        if (
          sessionRef.current !== session ||
          session.runId !== activeRunIdRef.current
        ) {
          return;
        }

        session.requestStartedAt = performance.now();
        startTransition(() => {
          setFrameIndex(nextFrame);
        });
        setRunState((current) => ({
          ...current,
          status: "running",
          progressLabel: `Running ${formatCount(session.count)} (${runStateRef.current.renderMode}): ${session.orderIndex}/${session.order.length} frames`,
        }));
      });
    },
    [],
  );

  const handleFrameRendered = useCallback(
    (timestamp: number) => {
      const session = sessionRef.current;
      if (!session || session.runId !== activeRunIdRef.current) return;

      if (session.requestStartedAt == null) {
        queueMeasuredFrame(session);
        return;
      }

      session.samples.push(timestamp - session.requestStartedAt);
      session.requestStartedAt = null;

      if (session.orderIndex >= session.order.length) {
        const averageMs =
          session.samples.reduce((total, sample) => total + sample, 0) /
          session.samples.length;
        const medianMs = median(session.samples);
        const minMs = Math.min(...session.samples);
        const maxMs = Math.max(...session.samples);

        setRunState((current) => ({
          ...current,
          status: "idle",
          currentCount: null,
          progressLabel: `Completed ${formatCount(session.count)}`,
          results: [
            ...current.results,
            {
              renderMode: current.renderMode,
              count: session.count,
              sampleCount: session.samples.length,
              precomputeMs: round(session.precomputeMs),
              averageMs: round(averageMs),
              medianMs: round(medianMs),
              minMs: round(minMs),
              maxMs: round(maxMs),
              averageFps: round(1000 / averageMs),
              medianFps: round(1000 / medianMs),
              datasetBytes: session.datasetBytes,
            },
          ],
        }));
        sessionRef.current = null;
        return;
      }

      queueMeasuredFrame(session);
    },
    [queueMeasuredFrame],
  );

  const handleHoverChange = useCallback((index: number | null) => {
    setHoveredIndex(index);
  }, []);

  const run = useCallback(
    async (partialOptions?: Partial<BenchOptions>) => {
      const options: BenchOptions = {
        counts: partialOptions?.counts || parseCounts(countsText),
        frameCount: partialOptions?.frameCount || frameCount,
        width: partialOptions?.width || width,
        height: partialOptions?.height || height,
        autorun: partialOptions?.autorun ?? false,
        renderMode: partialOptions?.renderMode || renderMode,
        ellipsoidStacks: partialOptions?.ellipsoidStacks || ellipsoidStacks,
        ellipsoidSlices: partialOptions?.ellipsoidSlices || ellipsoidSlices,
      };

      activeRunIdRef.current += 1;
      previewRequestIdRef.current += 1;
      const runId = activeRunIdRef.current;
      setIsPlaying(false);
      setHoveredIndex(null);
      setPreviewStatus("idle");
      setPreviewError(null);
      setDataset(null);
      setFrameIndex(0);

      const gpuInfo = await describeGPU();
      setRunState({
        status: "precomputing",
        counts: options.counts,
        frameCount: options.frameCount,
        renderMode: options.renderMode,
        currentCountIndex: -1,
        currentCount: null,
        progressLabel: "Preparing benchmark",
        results: [],
        gpuInfo,
      });

      try {
        for (let index = 0; index < options.counts.length; index++) {
          if (runId !== activeRunIdRef.current) {
            throw new Error("Superseded by a newer benchmark run");
          }

          const count = options.counts[index];
          setRunState((current) => ({
            ...current,
            status: "precomputing",
            renderMode: options.renderMode,
            currentCountIndex: index,
            currentCount: count,
            progressLabel: `Preparing ${formatCount(count)} ellipsoids (${options.renderMode})`,
          }));

          const precomputeStartedAt = performance.now();
          const nextDataset = await buildDataset(
            count,
            options.frameCount,
            setProgress,
          );
          const precomputeMs = performance.now() - precomputeStartedAt;

          if (runId !== activeRunIdRef.current) {
            throw new Error("Superseded by a newer benchmark run");
          }

          await new Promise<void>((resolve) => {
            const order = Array.from(
              { length: options.frameCount - 1 },
              (_, i) => i + 1,
            );
            order.push(0);
            sessionRef.current = {
              runId,
              order,
              orderIndex: 0,
              requestStartedAt: null,
              samples: [],
              precomputeMs,
              count,
              datasetBytes: nextDataset.datasetBytes,
            };

            setDataset(nextDataset);
            setFrameIndex(0);
            setRunState((current) => ({
              ...current,
              status: "warming",
              renderMode: options.renderMode,
              currentCountIndex: index,
              currentCount: count,
              progressLabel: `Warmup render for ${formatCount(count)} (${options.renderMode})`,
            }));

            const poll = () => {
              if (!sessionRef.current) {
                resolve();
                return;
              }
              window.setTimeout(poll, 25);
            };
            poll();
          });

          await nextPaint();
        }

        const finalState = {
          ...runStateRef.current,
          status: "completed" as const,
          currentCount: null,
          currentCountIndex: options.counts.length - 1,
          progressLabel: "Benchmark completed",
        };
        finishRun(finalState);
        return finalState;
      } catch (error) {
        const nextState = {
          ...runStateRef.current,
          status: "error" as const,
          error: error instanceof Error ? error.message : String(error),
          progressLabel: "Benchmark failed",
        };
        finishRun(nextState);
        return nextState;
      }
    },
    [
      countsText,
      ellipsoidSlices,
      ellipsoidStacks,
      finishRun,
      frameCount,
      height,
      renderMode,
      setProgress,
      width,
    ],
  );

  useEffect(() => {
    (window as WindowWithBench).__COLIGHT_ELLIPSOID_BENCH__ = {
      run,
      getState: () => runStateRef.current,
    };
  }, [run]);

  useEffect(() => {
    (window as WindowWithBench).__COLIGHT_ELLIPSOID_BENCH_DEBUG__ = {
      getState: () => ({
        bench: runStateRef.current,
        scene3d: (window as WindowWithBench).__COLIGHT_SCENE3D_DEBUG__ || null,
      }),
    };
  }, []);

  useEffect(() => {
    if (!debugOptions.verbose) return;

    const updateDebugState = () => {
      const scenes =
        (window as WindowWithBench).__COLIGHT_SCENE3D_DEBUG__?.scenes || {};
      const sceneStates = Object.values(scenes);
      const latestScene =
        [...sceneStates]
          .reverse()
          .find(
            (scene) =>
              Object.keys(scene.snapshots).length > 0 ||
              scene.events.length > 0,
          ) ||
        sceneStates[sceneStates.length - 1] ||
        null;

      if (!latestScene) {
        setSceneDebugState(null);
        return;
      }

      setSceneDebugState({
        ...latestScene,
        events: [...latestScene.events],
        snapshots: { ...latestScene.snapshots },
      });
    };

    updateDebugState();
    const intervalId = window.setInterval(updateDebugState, 250);
    return () => window.clearInterval(intervalId);
  }, [debugOptions.verbose]);

  useEffect(() => {
    if (!$state || !initialOptionsRef.current.autorun) return;
    void run(initialOptionsRef.current);
  }, [$state, run]);

  useEffect(() => {
    setHoveredIndex(null);
  }, [dataset?.count]);

  useEffect(() => {
    if (!isPlaying || !dataset || benchmarkActive) {
      return;
    }

    let rafId = 0;
    let lastTimestamp = performance.now();
    let accumulator = 0;
    const frameDuration = 1000 / Math.max(1, playFps);

    const tick = (now: number) => {
      accumulator += now - lastTimestamp;
      lastTimestamp = now;

      if (accumulator >= frameDuration) {
        const frameSteps = Math.floor(accumulator / frameDuration);
        accumulator -= frameSteps * frameDuration;
        startTransition(() => {
          setFrameIndex(
            (current) => (current + frameSteps) % dataset.frameCount,
          );
        });
      }

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [benchmarkActive, dataset, isPlaying, playFps]);

  const scene = useMemo(() => {
    if (!$state) return null;
    return (
      <$StateContext.Provider value={$state}>
        <SceneBench
          dataset={dataset}
          frameIndex={frameIndex}
          width={width}
          height={height}
          renderMode={renderMode}
          geometryOptions={{
            ellipsoidStacks,
            ellipsoidSlices,
          }}
          hoveredIndex={hoveredIndex}
          onHoverChange={handleHoverChange}
          onFrameRendered={handleFrameRendered}
        />
      </$StateContext.Provider>
    );
  }, [
    $state,
    dataset,
    ellipsoidSlices,
    ellipsoidStacks,
    frameIndex,
    handleHoverChange,
    handleFrameRendered,
    height,
    hoveredIndex,
    renderMode,
    width,
  ]);

  const counts = useMemo(() => parseCounts(countsText), [countsText]);
  const trianglesPerEllipsoid = useMemo(
    () => countSphereTriangles(ellipsoidStacks, ellipsoidSlices),
    [ellipsoidSlices, ellipsoidStacks],
  );
  const activeTriangleCount = useMemo(() => {
    const activeCount = previewCount ?? runState.currentCount;
    if (activeCount == null) return null;
    return (
      activeCount *
      (renderMode === "mesh"
        ? trianglesPerEllipsoid
        : PROXY_TRIANGLES_PER_IMPOSTOR)
    );
  }, [previewCount, renderMode, runState.currentCount, trianglesPerEllipsoid]);
  const sceneDebugReport = useMemo(() => {
    if (!sceneDebugState) return null;
    return {
      options: sceneDebugState.options,
      snapshots: sceneDebugState.snapshots,
      recentEvents: sceneDebugState.events.slice(-10),
    };
  }, [sceneDebugState]);

  return (
    <div className="app-shell">
      <div className="controls-panel">
        <div className="panel-header">
          <h1>Colight Ellipsoid FPS Bench</h1>
          <p>
            Precomputes <code>Float32Array</code> frame data and measures the
            cost of rendering fresh per-frame center buffers through{" "}
            <code>scene3d</code>.
          </p>
        </div>

        <div className="control-cluster">
          <label>
            Renderer
            <select
              value={renderMode}
              onChange={(event) =>
                setRenderMode(
                  event.target.value === "impostor" ? "impostor" : "mesh",
                )
              }
            >
              <option value="mesh">Mesh</option>
              <option value="impostor">Impostor</option>
            </select>
          </label>
          <label>
            Counts
            <input
              type="text"
              value={countsText}
              onChange={(event) => setCountsText(event.target.value)}
            />
          </label>
          <div className="size-grid compact-grid">
            <label>
              Frames
              <input
                type="number"
                min={2}
                max={240}
                value={frameCount}
                onChange={(event) =>
                  setFrameCount(
                    parseInputInteger(event.target.value, frameCount),
                  )
                }
              />
            </label>
            <label>
              Play FPS
              <input
                type="number"
                min={1}
                max={120}
                value={playFps}
                onChange={(event) =>
                  setPlayFps(parseInputInteger(event.target.value, playFps))
                }
              />
            </label>
          </div>

          <div className="size-grid compact-grid">
            <label>
              Width
              <input
                type="number"
                min={320}
                step={10}
                value={width}
                onChange={(event) =>
                  setWidth(parseInputInteger(event.target.value, width))
                }
              />
            </label>
            <label>
              Height
              <input
                type="number"
                min={240}
                step={10}
                value={height}
                onChange={(event) =>
                  setHeight(parseInputInteger(event.target.value, height))
                }
              />
            </label>
          </div>

          <div className="size-grid compact-grid">
            <label>
              Stacks
              <input
                type="number"
                min={3}
                step={1}
                value={ellipsoidStacks}
                onChange={(event) =>
                  setEllipsoidStacks(
                    Math.max(
                      3,
                      parseInputInteger(event.target.value, ellipsoidStacks),
                    ),
                  )
                }
              />
            </label>
            <label>
              Slices
              <input
                type="number"
                min={3}
                step={1}
                value={ellipsoidSlices}
                onChange={(event) =>
                  setEllipsoidSlices(
                    Math.max(
                      3,
                      parseInputInteger(event.target.value, ellipsoidSlices),
                    ),
                  )
                }
              />
            </label>
          </div>
        </div>

        <div className="size-grid button-grid">
          <button
            className="run-button"
            onClick={() => void preparePreview()}
            disabled={
              !$state || benchmarkActive || previewStatus === "preparing"
            }
          >
            Load Preview
          </button>
          <button
            className="run-button"
            onClick={() => {
              if (!dataset) {
                void preparePreview({ autoplay: true });
                return;
              }
              setIsPlaying((current) => !current);
            }}
            disabled={
              !$state || benchmarkActive || previewStatus === "preparing"
            }
          >
            {isPlaying ? "Pause Loop" : "Play Loop"}
          </button>
        </div>

        <button
          className="run-button"
          onClick={() =>
            void run({
              counts,
              frameCount,
              width,
              height,
              renderMode,
              ellipsoidStacks,
              ellipsoidSlices,
            })
          }
          disabled={!$state || benchmarkActive || previewStatus === "preparing"}
        >
          Run Benchmark
        </button>

        <div className="status-card">
          <div>Status: {runState.status}</div>
          <div>{runState.progressLabel}</div>
          <div>Preview: {previewStatus}</div>
          <div>
            Scene debug: {debugOptions.verbose ? "enabled" : "disabled"}
          </div>
          <div>
            Max instances per draw:{" "}
            {formatInstanceLimit(debugOptions.maxInstancesPerDraw)}
          </div>
          <div>Renderer: {renderMode}</div>
          <div>
            {renderMode === "mesh"
              ? `Mesh detail: ${ellipsoidStacks} x ${ellipsoidSlices} (${formatCount(trianglesPerEllipsoid)} tris / ellipsoid)`
              : `Impostor proxy: ${PROXY_TRIANGLES_PER_IMPOSTOR} tris / ellipsoid`}
          </div>
          {previewCount != null && (
            <div>Preview count: {previewCount.toLocaleString()}</div>
          )}
          {hoveredIndex != null && (
            <div>Hovered ellipsoid: {hoveredIndex.toLocaleString()}</div>
          )}
          {activeTriangleCount != null && (
            <div>
              Active triangle budget: {formatCount(activeTriangleCount)}
            </div>
          )}
          {isPlaying && dataset && <div>Looping at {playFps} FPS target</div>}
          {runState.currentCount != null && (
            <div>Current count: {runState.currentCount.toLocaleString()}</div>
          )}
          {runState.gpuInfo && (
            <pre>{JSON.stringify(runState.gpuInfo, null, 2)}</pre>
          )}
          {previewError && <div className="error-text">{previewError}</div>}
          {runState.error && <div className="error-text">{runState.error}</div>}
          {sceneDebugReport && (
            <details>
              <summary>Scene3D debug report</summary>
              <pre>{JSON.stringify(sceneDebugReport, null, 2)}</pre>
            </details>
          )}
        </div>

        <div className="results-wrap">
          <table className="results-table">
            <thead>
              <tr>
                <th>Renderer</th>
                <th>Count</th>
                <th>Avg FPS</th>
                <th>Median FPS</th>
                <th>Avg ms</th>
                <th>Median ms</th>
                <th>Min ms</th>
                <th>Max ms</th>
                <th>Frames</th>
                <th>Precompute</th>
                <th>Dataset</th>
              </tr>
            </thead>
            <tbody>
              {runState.results.map((result) => (
                <tr key={`${result.renderMode}-${result.count}`}>
                  <td>{result.renderMode}</td>
                  <td>{result.count.toLocaleString()}</td>
                  <td>{result.averageFps}</td>
                  <td>{result.medianFps}</td>
                  <td>{result.averageMs}</td>
                  <td>{result.medianMs}</td>
                  <td>{result.minMs}</td>
                  <td>{result.maxMs}</td>
                  <td>{result.sampleCount}</td>
                  <td>{result.precomputeMs} ms</td>
                  <td>{formatBytes(result.datasetBytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="viewer-panel">
        <div className="viewer-meta">
          <span>
            Viewport: {width} x {height}
          </span>
          {dataset && (
            <span>
              Frame: {frameIndex + 1}/{dataset.frameCount}
            </span>
          )}
          {dataset && <span>Dataset: {formatBytes(dataset.datasetBytes)}</span>}
        </div>
        <div className="scene-shell" style={{ width, height }}>
          {scene}
        </div>
      </div>
    </div>
  );
}

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Missing #root element for ellipsoid benchmark");
}

ReactDOM.createRoot(rootElement).render(<App />);
