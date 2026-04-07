const DEFAULT_EVENT_LIMIT = 200;
export const DEFAULT_SAFARI_MAX_INSTANCES_PER_DRAW = 1_000_000;

const GPU_LIMIT_KEYS = [
  "maxTextureDimension1D",
  "maxTextureDimension2D",
  "maxTextureDimension3D",
  "maxTextureArrayLayers",
  "maxBindGroups",
  "maxBindGroupsPlusVertexBuffers",
  "maxBindingsPerBindGroup",
  "maxDynamicUniformBuffersPerPipelineLayout",
  "maxDynamicStorageBuffersPerPipelineLayout",
  "maxSampledTexturesPerShaderStage",
  "maxSamplersPerShaderStage",
  "maxStorageBuffersPerShaderStage",
  "maxStorageTexturesPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxUniformBufferBindingSize",
  "maxStorageBufferBindingSize",
  "minUniformBufferOffsetAlignment",
  "minStorageBufferOffsetAlignment",
  "maxVertexBuffers",
  "maxBufferSize",
  "maxVertexAttributes",
  "maxVertexBufferArrayStride",
  "maxInterStageShaderVariables",
  "maxColorAttachments",
  "maxColorAttachmentBytesPerSample",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
] as const;

export interface Scene3DDebugOptions {
  verbose: boolean;
  maxInstancesPerDraw: number;
  userAgent: string;
}

export interface Scene3DDebugEvent {
  at: number;
  type: string;
  label: string;
  data?: Record<string, unknown>;
}

export interface Scene3DDebugState {
  id: string;
  createdAt: number;
  options: {
    verbose: boolean;
    maxInstancesPerDraw: number | null;
    userAgent: string;
  };
  snapshots: Record<string, unknown>;
  events: Scene3DDebugEvent[];
}

export interface Scene3DDebugRegistry {
  scenes: Record<string, Scene3DDebugState>;
}

export interface Scene3DDebugProbe {
  options: Scene3DDebugOptions;
  state: Scene3DDebugState;
  snapshot(name: string, value: unknown): void;
  record(type: string, label: string, data?: Record<string, unknown>): void;
  error(label: string, error: unknown, data?: Record<string, unknown>): void;
  dispose(): void;
}

export interface DrawSlice {
  firstInstance: number;
  instanceCount: number;
}

declare global {
  interface Window {
    __COLIGHT_SCENE3D_DEBUG__?: Scene3DDebugRegistry;
  }
}

function nowMs() {
  if (
    typeof performance !== "undefined" &&
    typeof performance.now === "function"
  ) {
    return performance.now();
  }
  return Date.now();
}

function normalizeSearchParams(search?: string | URLSearchParams) {
  if (search instanceof URLSearchParams) {
    return search;
  }

  if (typeof search === "string") {
    const normalized = search.startsWith("?") ? search.slice(1) : search;
    return new URLSearchParams(normalized);
  }

  if (typeof window !== "undefined") {
    return new URLSearchParams(window.location.search);
  }

  return new URLSearchParams();
}

function parsePositiveInteger(value: string | null) {
  if (!value) return null;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) return null;
  return parsed;
}

export function isSafariWebKitUserAgent(userAgent: string) {
  return (
    /Safari\//.test(userAgent) &&
    !/(Chrome|Chromium|CriOS|Edg|OPR|FxiOS)/.test(userAgent)
  );
}

export function parseScene3DDebugOptions(
  search?: string | URLSearchParams,
): Scene3DDebugOptions {
  const params = normalizeSearchParams(search);
  const userAgent =
    typeof navigator !== "undefined" ? navigator.userAgent || "" : "";
  const verbose =
    params.get("scene3d_debug") === "1" || params.get("debug") === "1";
  const overrideMaxInstances = parsePositiveInteger(
    params.get("scene3d_max_instances_per_draw"),
  );
  const maxInstancesPerDraw =
    overrideMaxInstances ??
    (isSafariWebKitUserAgent(userAgent)
      ? DEFAULT_SAFARI_MAX_INSTANCES_PER_DRAW
      : Number.POSITIVE_INFINITY);

  return {
    verbose,
    maxInstancesPerDraw,
    userAgent,
  };
}

export function formatInstanceLimit(maxInstancesPerDraw: number) {
  return Number.isFinite(maxInstancesPerDraw)
    ? maxInstancesPerDraw.toLocaleString()
    : "unlimited";
}

export function serializeGpuLimits(
  limits?: GPUSupportedLimits | null,
): Record<string, number> {
  if (!limits) return {};

  const serialized: Record<string, number> = {};
  for (const key of GPU_LIMIT_KEYS) {
    const value = limits[key];
    if (typeof value === "number") {
      serialized[key] = value;
    }
  }
  return serialized;
}

function syncRegistry(id: string, state: Scene3DDebugState) {
  if (typeof window === "undefined") return;
  if (!window.__COLIGHT_SCENE3D_DEBUG__) {
    window.__COLIGHT_SCENE3D_DEBUG__ = { scenes: {} };
  }
  window.__COLIGHT_SCENE3D_DEBUG__.scenes[id] = state;
}

function normalizeError(error: unknown) {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
    };
  }

  if (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof error.message === "string"
  ) {
    return {
      name:
        "name" in error && typeof error.name === "string"
          ? error.name
          : "Error",
      message: error.message,
      stack:
        "stack" in error && typeof error.stack === "string"
          ? error.stack
          : undefined,
    };
  }

  return {
    name: "Error",
    message: String(error),
  };
}

function pushEvent(
  state: Scene3DDebugState,
  type: string,
  label: string,
  data?: Record<string, unknown>,
) {
  state.events.push({
    at: nowMs(),
    type,
    label,
    data,
  });

  if (state.events.length > DEFAULT_EVENT_LIMIT) {
    state.events.splice(0, state.events.length - DEFAULT_EVENT_LIMIT);
  }
}

export function createScene3DDebugProbe(
  id: string,
  options = parseScene3DDebugOptions(),
): Scene3DDebugProbe {
  const state: Scene3DDebugState = {
    id,
    createdAt: nowMs(),
    options: {
      verbose: options.verbose,
      maxInstancesPerDraw: Number.isFinite(options.maxInstancesPerDraw)
        ? options.maxInstancesPerDraw
        : null,
      userAgent: options.userAgent,
    },
    snapshots: {},
    events: [],
  };

  syncRegistry(id, state);

  return {
    options,
    state,
    snapshot(name: string, value: unknown) {
      state.snapshots[name] = value;
      syncRegistry(id, state);
    },
    record(type: string, label: string, data?: Record<string, unknown>) {
      if (!options.verbose) return;
      pushEvent(state, type, label, data);
      syncRegistry(id, state);
    },
    error(label: string, error: unknown, data?: Record<string, unknown>) {
      const normalized = normalizeError(error);
      pushEvent(state, "error", label, {
        ...data,
        error: normalized,
      });
      state.snapshots.lastError = {
        label,
        at: nowMs(),
        ...data,
        error: normalized,
      };
      syncRegistry(id, state);
    },
    dispose() {
      if (typeof window === "undefined") return;
      delete window.__COLIGHT_SCENE3D_DEBUG__?.scenes[id];
    },
  };
}

export async function withGpuErrorScopes<T>(
  device: GPUDevice,
  probe: Scene3DDebugProbe | null | undefined,
  label: string,
  data: Record<string, unknown>,
  run: () => Promise<T> | T,
) {
  if (
    !probe?.options.verbose ||
    typeof device.pushErrorScope !== "function" ||
    typeof device.popErrorScope !== "function"
  ) {
    return await run();
  }

  const filters: GPUErrorFilter[] = ["out-of-memory", "internal", "validation"];
  filters.forEach((filter) => device.pushErrorScope(filter));
  let result: T | undefined;
  let runError: unknown;
  let scopedFailure: GPUError | null = null;

  try {
    result = await run();
  } catch (error) {
    runError = error;
  } finally {
    for (let i = filters.length - 1; i >= 0; i--) {
      const filter = filters[i];
      const scopedError = await device.popErrorScope();
      if (scopedError) {
        probe.error(`${label}/${filter}`, scopedError, data);
        if (!scopedFailure) {
          scopedFailure = scopedError;
        }
      }
    }
  }

  if (runError) {
    throw runError;
  }

  if (scopedFailure) {
    throw scopedFailure;
  }

  return result as T;
}

export function buildDrawSlices(
  instanceCount: number,
  maxInstancesPerDraw: number,
): DrawSlice[] {
  if (instanceCount <= 0) return [];

  const safeMaxInstancesPerDraw =
    Number.isFinite(maxInstancesPerDraw) && maxInstancesPerDraw > 0
      ? Math.floor(maxInstancesPerDraw)
      : instanceCount;

  if (safeMaxInstancesPerDraw >= instanceCount) {
    return [{ firstInstance: 0, instanceCount }];
  }

  const slices: DrawSlice[] = [];
  for (
    let firstInstance = 0;
    firstInstance < instanceCount;
    firstInstance += safeMaxInstancesPerDraw
  ) {
    slices.push({
      firstInstance,
      instanceCount: Math.min(
        safeMaxInstancesPerDraw,
        instanceCount - firstInstance,
      ),
    });
  }

  return slices;
}
