/// <reference types="@webgpu/types" />
/**
 * D1 regression harness: an inline `Mesh` in a scene driven by live state must
 * not leak GPU resources.
 *
 * The measured defect (D0, 2026-07-26): every `$state` change re-evaluates the
 * serialized AST, rebuilding the props tree, so a Mesh's `geometry` is a fresh
 * JS object every frame even when its typed arrays are identical. The old
 * object-identity WeakMap in inlineMesh.ts therefore never hit, minting a new
 * `__InlineMesh_N` type name per frame: 2 render pipelines + 4 shader modules +
 * 2 GPUBuffers per mesh per frame, and zero destroys.
 *
 * The load-bearing assertion here is BYTES (allocated minus destroyed), not
 * pipeline count - a pipelines-only test would pass on a fix that still leaked.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, act } from "@testing-library/react";
import React from "react";
import { SceneImpl } from "../../../src/js/scene3d/impl3d";
import { compileScene } from "../../../src/js/scene3d/compiler";
import { clearInlineMeshCache } from "../../../src/js/scene3d/inlineMesh";
import { setupWebGPU, cleanupWebGPU } from "../webgpu-setup";
import { withBlankState } from "../test-utils";

/** Counters at the device-wrapper level: creates and destroys, with byte sizes. */
interface GpuCounters {
  pipelines: number;
  shaderModules: number;
  buffersCreated: number;
  buffersDestroyed: number;
  bytesAllocated: number;
  bytesDestroyed: number;
  /** Vertex/index buffers only - the geometry buffers this workstream owns. */
  geometryBytesAllocated: number;
  geometryBytesDestroyed: number;
  writeBufferCalls: number;
  /** writeBuffer calls targeting a vertex/index buffer only. */
  geometryWriteCalls: number;
}

function makeCountingDevice() {
  const counters: GpuCounters = {
    pipelines: 0,
    shaderModules: 0,
    buffersCreated: 0,
    buffersDestroyed: 0,
    bytesAllocated: 0,
    bytesDestroyed: 0,
    geometryBytesAllocated: 0,
    geometryBytesDestroyed: 0,
    writeBufferCalls: 0,
    geometryWriteCalls: 0,
  };

  const GEOMETRY_USAGE =
    (globalThis as any).GPUBufferUsage.VERTEX |
    (globalThis as any).GPUBufferUsage.INDEX;

  const createBuffer = vi.fn((desc: GPUBufferDescriptor) => {
    counters.buffersCreated++;
    counters.bytesAllocated += desc.size;
    const isGeometry = (desc.usage & GEOMETRY_USAGE) !== 0;
    if (isGeometry) counters.geometryBytesAllocated += desc.size;
    let destroyed = false;
    return {
      size: desc.size,
      usage: desc.usage,
      destroy: vi.fn(() => {
        if (destroyed) return;
        destroyed = true;
        counters.buffersDestroyed++;
        counters.bytesDestroyed += desc.size;
        if (isGeometry) counters.geometryBytesDestroyed += desc.size;
      }),
      mapAsync: vi.fn().mockResolvedValue(undefined),
      getMappedRange: vi.fn(() => new ArrayBuffer(desc.size)),
      unmap: vi.fn(),
    };
  });

  const queue = {
    writeBuffer: vi.fn((buffer: any) => {
      counters.writeBufferCalls++;
      // Classify by the target's usage so geometry rewrites can be counted
      // independently of uniform/instance traffic, whose volume depends on how
      // many render passes the harness happens to schedule.
      if (buffer && (buffer.usage & GEOMETRY_USAGE) !== 0) {
        counters.geometryWriteCalls++;
      }
    }),
    submit: vi.fn(),
    onSubmittedWorkDone: vi.fn().mockResolvedValue(undefined),
  } as unknown as GPUQueue;

  const device = {
    createBuffer,
    createBindGroup: vi.fn(() => ({ label: "bg" })),
    createBindGroupLayout: vi.fn(() => ({ label: "bgl" })),
    createPipelineLayout: vi.fn(() => ({ label: "pl" })),
    createRenderPipeline: vi.fn(() => {
      counters.pipelines++;
      return { label: "pipeline" };
    }),
    createShaderModule: vi.fn(() => {
      counters.shaderModules++;
      return { label: "shader" };
    }),
    createSampler: vi.fn(() => ({ label: "sampler" })),
    createCommandEncoder: vi.fn(() => ({
      beginRenderPass: vi.fn(() => ({
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        setVertexBuffer: vi.fn(),
        setIndexBuffer: vi.fn(),
        setViewport: vi.fn(),
        setScissorRect: vi.fn(),
        draw: vi.fn(),
        drawIndexed: vi.fn(),
        end: vi.fn(),
      })),
      copyTextureToBuffer: vi.fn(),
      copyBufferToBuffer: vi.fn(),
      finish: vi.fn(() => ({ label: "cmd" })),
    })),
    createTexture: vi.fn(() => ({
      createView: vi.fn(() => ({ label: "view" })),
      destroy: vi.fn(),
    })),
    queue,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  } as unknown as GPUDevice;

  return { device, counters };
}

/** A tapered tube: the tentacle fixture's shape, small enough for a unit test. */
const N_RINGS = 6;
const N_SIDES = 8;

function tubePositions(bend: number): Float32Array {
  const out = new Float32Array(N_RINGS * N_SIDES * 3);
  let o = 0;
  for (let i = 0; i < N_RINGS; i++) {
    const t = i / (N_RINGS - 1);
    const r = 0.5 - 0.4 * t;
    const yShift = bend * t * t;
    for (let j = 0; j < N_SIDES; j++) {
      const a = (j / N_SIDES) * Math.PI * 2;
      out[o++] = r * Math.cos(a);
      out[o++] = r * Math.sin(a) + yShift;
      out[o++] = t * 4;
    }
  }
  return out;
}

function tubeIndices(nRings: number): Uint32Array {
  const idx: number[] = [];
  for (let i = 0; i < nRings - 1; i++) {
    for (let j = 0; j < N_SIDES; j++) {
      const a = i * N_SIDES + j;
      const b = i * N_SIDES + ((j + 1) % N_SIDES);
      const c = (i + 1) * N_SIDES + j;
      const d = (i + 1) * N_SIDES + ((j + 1) % N_SIDES);
      idx.push(a, c, b, b, c, d);
    }
  }
  return new Uint32Array(idx);
}

const BASE_INDICES = tubeIndices(N_RINGS);
const BASE_POSITIONS = tubePositions(0);

/**
 * One "frame" of the AST re-evaluation the widget performs on any `$state`
 * change: a brand new component object wrapping a brand new geometry object.
 * `sameArrays: true` reproduces the case that matters most - the typed arrays
 * are the identical buffers, only the wrapping objects are new.
 */
function meshFrame(opts: {
  positions?: Float32Array;
  indices?: Uint32Array;
  scale?: number;
  geometryKey?: string;
}) {
  return {
    type: "Mesh" as const,
    geometry: {
      positions: opts.positions ?? BASE_POSITIONS,
      indices: opts.indices ?? BASE_INDICES,
    },
    ...(opts.geometryKey !== undefined
      ? { geometryKey: opts.geometryKey }
      : {}),
    centers: new Float32Array([0, 0, 0]),
    scale: opts.scale ?? 1,
    color: [0.4, 0.6, 0.9] as [number, number, number],
    shading: "lit" as const,
    cullMode: "none" as GPUCullMode,
  };
}

describe("Scene3D inline mesh resource lifecycle (D1)", () => {
  let container: HTMLDivElement;
  let device: GPUDevice;
  let counters: GpuCounters;
  let Wrapped: React.ComponentType<React.ComponentProps<typeof SceneImpl>>;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    setupWebGPU();
    clearInlineMeshCache();
    Wrapped = withBlankState(SceneImpl);

    const made = makeCountingDevice();
    device = made.device;
    counters = made.counters;

    Object.defineProperty(navigator, "gpu", {
      value: {
        requestAdapter: vi.fn().mockResolvedValue({
          requestDevice: vi.fn().mockResolvedValue(device),
        }),
        getPreferredCanvasFormat: vi.fn().mockReturnValue("rgba8unorm"),
      },
      configurable: true,
    });

    Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
      value: vi.fn((type: string) =>
        type === "webgpu"
          ? {
              configure: vi.fn(),
              getCurrentTexture: vi.fn(() => ({
                createView: vi.fn(() => ({ label: "view" })),
              })),
            }
          : null,
      ),
      configurable: true,
    });
  });

  afterEach(() => {
    document.body.removeChild(container);
    vi.clearAllMocks();
    cleanupWebGPU();
  });

  /**
   * Renders a sequence of frames, each compiled from scratch (as the widget
   * does), and snapshots the counters after a warm-up frame so steady-state
   * growth is what is measured.
   */
  async function sweep(
    frames: any[][],
    snapshotAfter = 1,
  ): Promise<{ baseline: GpuCounters; final: GpuCounters }> {
    let result: ReturnType<typeof render> | undefined;
    let baseline: GpuCounters | undefined;

    for (let i = 0; i < frames.length; i++) {
      const compiled = compileScene(frames[i] as any);
      const element = (
        <Wrapped
          components={compiled.components}
          primitiveSpecs={compiled.primitiveSpecs}
          containerWidth={400}
          containerHeight={300}
          onReady={vi.fn()}
        />
      );
      // eslint-disable-next-line no-await-in-loop
      await act(async () => {
        if (!result) {
          result = render(element, { container });
        } else {
          result.rerender(element);
        }
      });
      if (i + 1 === snapshotAfter) baseline = { ...counters };
    }

    return { baseline: baseline!, final: { ...counters } };
  }

  it("state-driven prop change with unchanged geometry allocates nothing new", async () => {
    // The exact shape of D0's "control 4": positions are literally the same
    // typed array every frame; only a scalar instance prop is state-driven.
    const N = 12;
    const frames = Array.from({ length: N }, (_, i) => [
      meshFrame({ scale: 1 + i * 0.01 }),
    ]);

    const { baseline, final } = await sweep(frames, 2);

    expect(final.pipelines).toBe(baseline.pipelines);
    expect(final.shaderModules).toBe(baseline.shaderModules);
    expect(final.geometryBytesAllocated - final.geometryBytesDestroyed).toBe(
      baseline.geometryBytesAllocated - baseline.geometryBytesDestroyed,
    );
    expect(final.bytesAllocated - final.bytesDestroyed).toBe(
      baseline.bytesAllocated - baseline.bytesDestroyed,
    );
  });

  it("fresh positions per frame with stable topology allocates nothing new", async () => {
    // Every frame ships a brand-new Float32Array of the same length - the shape
    // every deforming-mesh workload takes (cloth, FEM, an evolving surface).
    const N = 12;
    const frames = Array.from({ length: N }, (_, i) => [
      meshFrame({ positions: tubePositions(i * 0.05) }),
    ]);

    const { baseline, final } = await sweep(frames, 2);

    expect(final.pipelines).toBe(baseline.pipelines);
    expect(final.shaderModules).toBe(baseline.shaderModules);
    // The load-bearing assertion: no net GPU bytes accumulate across the sweep.
    expect(final.geometryBytesAllocated - final.geometryBytesDestroyed).toBe(
      baseline.geometryBytesAllocated - baseline.geometryBytesDestroyed,
    );
    expect(final.bytesAllocated - final.bytesDestroyed).toBe(
      baseline.bytesAllocated - baseline.bytesDestroyed,
    );
    // ...and the new contents DID reach the GPU: with a reused identity the
    // only way for changed vertices to render is a write into the buffer that
    // already exists (D1b's write path). Exactly 2 geometry writes per changed
    // frame - the interleaved vertex buffer and the index buffer - and zero
    // allocations. Counted on vertex/index buffers only, so unrelated uniform
    // traffic cannot make this flaky.
    expect(final.geometryWriteCalls).toBe(
      baseline.geometryWriteCalls + 2 * (N - 2),
    );
  });

  it("a channel-driven positions sweep allocates nothing new", async () => {
    // The declared-resampling shape: a pose table ships once and a $state
    // scalar picks the frame. `resampleChannel` returns a fresh Float32Array
    // per parameter value, so this must land on the same contents-write path
    // as a hand-computed fresh array - no pipelines, no leaked bytes.
    const { resampleChannel } = await import("../../../src/js/channels");

    const POSE_ANGLES = [-80, -40, 0, 40, 80];
    const POSES = POSE_ANGLES.map((a) => tubePositions(a * 0.01));

    const N = 12;
    const frames = Array.from({ length: N }, (_, i) => {
      // A continuous parameter finer than the pose spacing, so most frames
      // are genuinely interpolated rather than landing on a sample.
      const bend = -80 + (i * 160) / (N - 1);
      const positions = resampleChannel({
        parameter: "bend",
        value: bend,
        at: POSE_ANGLES,
        values: POSES,
        rule: "linear",
      }) as Float32Array;
      expect(positions).toBeInstanceOf(Float32Array);
      expect(positions.length).toBe(POSES[0].length);
      return [meshFrame({ positions })];
    });

    const { baseline, final } = await sweep(frames, 2);

    expect(final.pipelines).toBe(baseline.pipelines);
    expect(final.shaderModules).toBe(baseline.shaderModules);
    expect(final.geometryBytesAllocated - final.geometryBytesDestroyed).toBe(
      baseline.geometryBytesAllocated - baseline.geometryBytesDestroyed,
    );
    expect(final.bytesAllocated - final.bytesDestroyed).toBe(
      baseline.bytesAllocated - baseline.bytesDestroyed,
    );
    // The resampled vertices reach the GPU by writing into the buffers that
    // already exist: 2 geometry writes (interleaved vertex + index) per
    // changed frame, and zero allocations.
    expect(final.geometryWriteCalls).toBe(
      baseline.geometryWriteCalls + 2 * (N - 2),
    );
  });

  it("a stable user geometry_key keeps identity across fresh arrays", async () => {
    const N = 10;
    const frames = Array.from({ length: N }, (_, i) => [
      meshFrame({ positions: tubePositions(i * 0.05), geometryKey: "tube" }),
    ]);

    const { baseline, final } = await sweep(frames, 2);

    expect(final.pipelines).toBe(baseline.pipelines);
    expect(final.shaderModules).toBe(baseline.shaderModules);
    expect(final.bytesAllocated - final.bytesDestroyed).toBe(
      baseline.bytesAllocated - baseline.bytesDestroyed,
    );
  });

  it("a topology change rebuilds, and does not leak the buffers it replaces", async () => {
    const bigPositions = new Float32Array((N_RINGS + 4) * N_SIDES * 3);
    for (let i = 0; i < bigPositions.length; i++) bigPositions[i] = i * 0.001;
    const bigIndices = tubeIndices(N_RINGS + 4);

    const frames = [
      [meshFrame({})],
      [meshFrame({})],
      // vertex count AND index buffer change: a genuine topology change
      [meshFrame({ positions: bigPositions, indices: bigIndices })],
      [meshFrame({ positions: bigPositions, indices: bigIndices })],
    ];

    const { baseline, final } = await sweep(frames, 2);

    // A different topology is a different identity: it legitimately builds.
    expect(final.pipelines).toBeGreaterThan(baseline.pipelines);
    // The geometry buffers of the identity it replaced are released, not
    // orphaned in the resources map under a dead `__InlineMesh_N` key.
    expect(final.geometryBytesDestroyed).toBeGreaterThan(
      baseline.geometryBytesDestroyed,
    );
    // But once at the new topology, it settles - frames 3 and 4 are identical.
    const afterGrowth = { ...counters };
    await act(async () => {});
    expect(afterGrowth.pipelines).toBe(final.pipelines);
  });

  it("growing geometry under a stable geometry_key grows buffers and destroys the ones it replaces", async () => {
    const small = tubePositions(0);
    const smallIdx = tubeIndices(N_RINGS);
    const big = new Float32Array((N_RINGS + 6) * N_SIDES * 3);
    for (let i = 0; i < big.length; i++) big[i] = Math.sin(i) * 0.5;
    const bigIdx = tubeIndices(N_RINGS + 6);

    const frames = [
      [meshFrame({ positions: small, indices: smallIdx, geometryKey: "g" })],
      [meshFrame({ positions: small, indices: smallIdx, geometryKey: "g" })],
      [meshFrame({ positions: big, indices: bigIdx, geometryKey: "g" })],
      [meshFrame({ positions: big, indices: bigIdx, geometryKey: "g" })],
      [meshFrame({ positions: big, indices: bigIdx, geometryKey: "g" })],
    ];

    const { baseline, final } = await sweep(frames, 2);

    // The user named the identity, so no pipeline churn even though the
    // buffers had to grow past their allocation.
    expect(final.pipelines).toBe(baseline.pipelines);
    expect(final.shaderModules).toBe(baseline.shaderModules);
    // Every outgrown buffer is destroyed rather than orphaned.
    expect(final.buffersDestroyed).toBeGreaterThan(baseline.buffersDestroyed);
    expect(final.geometryBytesDestroyed).toBeGreaterThan(
      baseline.geometryBytesDestroyed,
    );
  });

  it("shrinking geometry under a stable geometry_key reuses the buffers in place", async () => {
    const big = new Float32Array((N_RINGS + 6) * N_SIDES * 3);
    for (let i = 0; i < big.length; i++) big[i] = Math.sin(i) * 0.5;
    const bigIdx = tubeIndices(N_RINGS + 6);
    const small = tubePositions(0);
    const smallIdx = tubeIndices(N_RINGS);

    const frames = [
      [meshFrame({ positions: big, indices: bigIdx, geometryKey: "g" })],
      [meshFrame({ positions: big, indices: bigIdx, geometryKey: "g" })],
      [meshFrame({ positions: small, indices: smallIdx, geometryKey: "g" })],
      [meshFrame({ positions: small, indices: smallIdx, geometryKey: "g" })],
    ];

    const { baseline, final } = await sweep(frames, 2);

    // Fewer bytes fit in the existing allocation, so nothing is allocated or
    // destroyed - the grow-only reuse pattern, applied to geometry.
    expect(final.pipelines).toBe(baseline.pipelines);
    expect(final.geometryBytesAllocated).toBe(baseline.geometryBytesAllocated);
    expect(final.geometryBytesDestroyed).toBe(baseline.geometryBytesDestroyed);
  });

  it("auto-computed flat normals track changing positions under lit shading", async () => {
    // The interleaved vertex buffer carries normals; nothing supplies them, so
    // `shading: "lit"` computes them. If a contents change reused a stale
    // normal array the surface would shade as its rest pose forever.
    const { resolveInlineMeshes } = await import(
      "../../../src/js/scene3d/inlineMesh"
    );

    const readNormals = (positions: Float32Array) => {
      const { inlineSpecs } = resolveInlineMeshes([
        meshFrame({ positions, geometryKey: "n" }) as any,
      ]);
      const spec = Object.values(inlineSpecs!)[0] as any;
      const { vertexData } = spec.buildGeometryData();
      // Layout is [px,py,pz, nx,ny,nz] - no colors, no uvs.
      const normals: number[] = [];
      for (let i = 0; i < vertexData.length; i += 6) {
        normals.push(vertexData[i + 3], vertexData[i + 4], vertexData[i + 5]);
      }
      return normals;
    };

    const rest = readNormals(tubePositions(0));
    const bent = readNormals(tubePositions(1.2));

    expect(bent.length).toBe(rest.length);
    expect(bent.some((n, i) => Math.abs(n - rest[i]) > 1e-4)).toBe(true);
  });

  // ---------------------------------------------------------------------------
  // Identity collision between distinct meshes of identical structure.
  //
  // Two Mesh components in one scene can share a vertex count, format and index
  // shape while holding entirely different vertex data (two same-resolution
  // spheres, two quads, two copies of a template). They must not collapse onto
  // one cache entry: doing so gives them one type name, one spec and one
  // geometry holder, so both render whichever geometry was resolved last.
  // ---------------------------------------------------------------------------

  /** Two triangles over four vertices - the minimal same-structure pair. */
  function quad(x: number): Float32Array {
    return new Float32Array([x, 0, 0, x + 1, 0, 0, x, 1, 0, x + 1, 1, 0]);
  }
  const QUAD_INDICES = new Uint32Array([0, 1, 2, 1, 3, 2]);

  function quadFrame(positions: Float32Array) {
    return {
      type: "Mesh" as const,
      geometry: { positions, indices: QUAD_INDICES },
      centers: new Float32Array([0, 0, 0]),
      color: [0.4, 0.6, 0.9] as [number, number, number],
      shading: "lit" as const,
      cullMode: "none" as GPUCullMode,
    };
  }

  it("two structurally identical meshes keep separate geometry", async () => {
    const { resolveInlineMeshes } = await import(
      "../../../src/js/scene3d/inlineMesh"
    );

    const a = quad(0);
    const b = quad(100);
    const { components, inlineSpecs } = resolveInlineMeshes([
      quadFrame(a) as any,
      quadFrame(b) as any,
    ]);

    // Distinct components must resolve to distinct primitive types, or impl3d
    // (one geometry resource per type name) cannot hold both geometries.
    expect(components[0].type).not.toBe(components[1].type);
    expect(Object.keys(inlineSpecs!)).toHaveLength(2);

    // Stronger: each spec's interleaved buffer must carry its own positions.
    const readX = (type: string) => {
      const { vertexData } = (inlineSpecs as any)[type].buildGeometryData();
      // Layout is [px,py,pz, nx,ny,nz] - no colors, no uvs.
      const xs: number[] = [];
      for (let i = 0; i < vertexData.length; i += 6) xs.push(vertexData[i]);
      return xs;
    };
    expect(readX(components[0].type)).toEqual([0, 1, 0, 1]);
    expect(readX(components[1].type)).toEqual([100, 101, 100, 101]);
  });

  it("re-compiling two structurally identical meshes bumps no contents versions", async () => {
    const { resolveInlineMeshes } = await import(
      "../../../src/js/scene3d/inlineMesh"
    );

    const a = quad(0);
    const b = quad(100);

    // The same arrays, recompiled - exactly what a $state change produces.
    const first = resolveInlineMeshes([
      quadFrame(a) as any,
      quadFrame(b) as any,
    ]);
    const keysAfterFirst = Object.values(first.inlineSpecs!).map(
      (s: any) => s.geometryKey,
    );
    const second = resolveInlineMeshes([
      quadFrame(a) as any,
      quadFrame(b) as any,
    ]);
    const keysAfterSecond = Object.values(second.inlineSpecs!).map(
      (s: any) => s.geometryKey,
    );

    // Each component keeps hitting its own entry, so nothing is a contents
    // change. Under a collision the two would thrash one holder and bump twice
    // per compile even though no array changed.
    expect(second.components.map((c) => c.type)).toEqual(
      first.components.map((c) => c.type),
    );
    expect(keysAfterSecond).toEqual(keysAfterFirst);
  });

  it("two structurally identical meshes each get their own geometry resource", async () => {
    // The user-visible symptom of the collision was not merely "both render the
    // last geometry" - the shared type name plus a contents version thrashed
    // twice per compile left the scene blank on real hardware. Assert the
    // renderer builds a separate geometry resource per component.
    const a = quad(0);
    const b = quad(100);
    const compiled = compileScene([quadFrame(a), quadFrame(b)] as any);

    // Two inline specs, two distinct component types...
    expect(Object.keys(compiled.primitiveSpecs!)).toHaveLength(2);
    // ...and each spec reports its own vertex bytes.
    const specs = Object.values(compiled.primitiveSpecs!) as any[];
    const firstX = specs[0].buildGeometryData().vertexData[0];
    const secondX = specs[1].buildGeometryData().vertexData[0];
    expect(new Set([firstX, secondX]).size).toBe(2);
  });

  it("two structurally identical meshes do not churn or leak across a sweep", async () => {
    const a = quad(0);
    const b = quad(100);
    const N = 10;
    const frames = Array.from({ length: N }, () => [
      quadFrame(a),
      quadFrame(b),
    ]);

    const { baseline, final } = await sweep(frames, 2);

    expect(final.pipelines).toBe(baseline.pipelines);
    expect(final.shaderModules).toBe(baseline.shaderModules);
    expect(final.geometryBytesAllocated - final.geometryBytesDestroyed).toBe(
      baseline.geometryBytesAllocated - baseline.geometryBytesDestroyed,
    );
    // Nothing changed, so no geometry contents are rewritten either. Counted
    // on vertex/index buffers only: uniform and instance traffic varies with
    // the number of render passes scheduled, which is not part of the contract.
    expect(final.geometryWriteCalls).toBe(baseline.geometryWriteCalls);
  });

  it("the drawn index format follows the coerced buffer, not the identity bucket", async () => {
    const { resolveInlineMeshes } = await import(
      "../../../src/js/scene3d/inlineMesh"
    );

    // Same element count, plain JS arrays (so both land in one identity
    // bucket), but the second crosses 65535 and must be coerced to uint32.
    const narrow = [0, 1, 2, 1, 3, 2];
    const wide = [0, 1, 2, 1, 3, 70000];
    const positionsFor = (n: number) => new Float32Array(n * 3);

    const build = (indices: number[], vertexCount: number) => {
      const { inlineSpecs } = resolveInlineMeshes([
        {
          ...quadFrame(positionsFor(vertexCount)),
          geometry: { positions: positionsFor(vertexCount), indices },
        } as any,
      ]);
      const spec = Object.values(inlineSpecs!)[0] as any;
      return spec.buildGeometryData().indexData;
    };

    expect(build(narrow, 4)).toBeInstanceOf(Uint16Array);
    expect(build(wide, 70001)).toBeInstanceOf(Uint32Array);
  });
});
