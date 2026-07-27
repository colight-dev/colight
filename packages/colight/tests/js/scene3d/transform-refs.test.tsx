/// <reference types="@webgpu/types" />
/**
 * D2: per-vertex weighted transform references on an inline `Mesh`.
 *
 * A vertex is positioned by a weighted combination of named Group transforms
 * instead of rigidly by one. What is tested here:
 *
 *   1. Name -> palette index resolution at compile, and its loud failure.
 *   2. The identity/variant key separating blended from rigid meshes, so they
 *      cannot share a spec, shaders or a pipeline-cache entry.
 *   3. The write separation that motivated the storage-buffer choice: a
 *      weights-only change writes the reference buffer and leaves the
 *      interleaved vertex buffer alone; a palette-only sweep (a Group
 *      quaternion driven by $state) writes NEITHER and allocates nothing.
 *   4. Blend correctness at the data level - the resolved buffer against the
 *      real flattened palette.
 *
 * The counting device is the one from inline-mesh-lifecycle.test.tsx, extended
 * to count writes to storage buffers separately from vertex/index buffers.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, act } from "@testing-library/react";
import React from "react";
import { SceneImpl } from "../../../src/js/scene3d/impl3d";
import { compileScene } from "../../../src/js/scene3d/compiler";
import {
  clearInlineMeshCache,
  resolveInlineMeshes,
  resolveTransformRefs,
} from "../../../src/js/scene3d/inlineMesh";
import {
  flattenGroups,
  quatRotate,
  type Quat,
} from "../../../src/js/scene3d/groups";
import { getFormatKey } from "../../../src/js/scene3d/primitives/mesh";
import { setupWebGPU, cleanupWebGPU } from "../webgpu-setup";
import { withBlankState } from "../test-utils";

// =============================================================================
// Fixture: a strip of vertices spanning two Groups
// =============================================================================

const N_VERTS = 6;

/** Vertices along +Z, all in the root group's frame. */
const POSITIONS = new Float32Array(
  Array.from({ length: N_VERTS }, (_, i) => [0, 0, i * 0.5]).flat(),
);
const INDICES = new Uint32Array([0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4]);

/** Slot 0 = "root", slot 1 = "elbow", for every vertex. */
const SLOTS = new Float32Array(
  Array.from({ length: N_VERTS }, () => [0, 1]).flat(),
);

/** Weights ramping from all-root to all-elbow across the strip. */
function ramp(shift = 0): Float32Array {
  const out = new Float32Array(N_VERTS * 2);
  for (let i = 0; i < N_VERTS; i++) {
    const t = Math.min(1, Math.max(0, i / (N_VERTS - 1) + shift));
    out[i * 2] = 1 - t;
    out[i * 2 + 1] = t;
  }
  return out;
}

/** Rotation about +X by `deg`, as the [x, y, z, w] quaternion Group takes. */
function bendQuat(deg: number): Quat {
  const half = (deg * Math.PI) / 180 / 2;
  return [Math.sin(half), 0, 0, Math.cos(half)];
}

function blendedMesh(
  opts: {
    weights?: Float32Array;
    refs?: string[];
    positions?: Float32Array;
  } = {},
) {
  return {
    type: "Mesh" as const,
    geometry: {
      positions: opts.positions ?? POSITIONS,
      indices: INDICES,
      transformRefs: opts.refs ?? ["root", "elbow"],
      transformIndices: SLOTS,
      transformWeights: opts.weights ?? ramp(),
    },
    centers: new Float32Array([0, 0, 0]),
    color: [0.4, 0.6, 0.9] as [number, number, number],
    shading: "lit" as const,
    cullMode: "none" as GPUCullMode,
  };
}

function rigidMesh() {
  return {
    type: "Mesh" as const,
    geometry: { positions: POSITIONS, indices: INDICES },
    centers: new Float32Array([0, 0, 0]),
    color: [0.4, 0.6, 0.9] as [number, number, number],
    shading: "lit" as const,
    cullMode: "none" as GPUCullMode,
  };
}

/** The scene: a named root group holding the mesh and a named elbow group. */
function scene(
  bendDeg: number,
  meshComponent: ReturnType<typeof blendedMesh> | ReturnType<typeof rigidMesh>,
) {
  return [
    {
      type: "Group" as const,
      name: "root",
      children: [
        meshComponent,
        {
          type: "Group" as const,
          name: "elbow",
          position: [0, 0, 1.5] as [number, number, number],
          quaternion: bendQuat(bendDeg),
          children: [
            // A tiny rigid child so the elbow group is not childless; the
            // blended mesh references it by NAME, not by containment.
            {
              type: "PointCloud" as const,
              centers: new Float32Array([0, 0, 0]),
            },
          ],
        },
      ],
    },
  ];
}

// =============================================================================
// 1. Name -> palette index resolution
// =============================================================================

describe("transform reference resolution", () => {
  beforeEach(() => clearInlineMeshCache());

  it("gives every named Group its own palette slot", () => {
    const result = flattenGroups(scene(30, blendedMesh()) as any);
    expect(result.namedTransforms.get("root")).toBeDefined();
    expect(result.namedTransforms.get("elbow")).toBeDefined();
    // Distinct slots, and neither is the shared identity at index 0.
    expect(result.namedTransforms.get("root")).not.toBe(
      result.namedTransforms.get("elbow"),
    );
    expect(result.namedTransforms.get("root")).toBeGreaterThan(0);
  });

  it("a named Group gets a slot even when its transform is identity", () => {
    // "root" has no position/quaternion/scale, so its world transform is
    // identity - but a reference to it must still resolve to a real slot,
    // because $state can make it non-identity on the very next frame.
    const result = flattenGroups(scene(0, blendedMesh()) as any);
    const rootIdx = result.namedTransforms.get("root")!;
    expect(rootIdx).toBeGreaterThan(0);
    expect(result.transforms[rootIdx]).toBeDefined();
  });

  it("resolves slots to ABSOLUTE palette indices", () => {
    const flattened = flattenGroups(scene(30, blendedMesh()) as any);
    const resolved = resolveTransformRefs(
      blendedMesh().geometry,
      flattened.namedTransforms,
    )!;

    expect(resolved.count).toBe(2);
    expect(resolved.data.length).toBe(N_VERTS * 2 * 2);

    const rootIdx = flattened.namedTransforms.get("root")!;
    const elbowIdx = flattened.namedTransforms.get("elbow")!;
    // Layout is [paletteIndex, weight] per reference, K per vertex.
    for (let v = 0; v < N_VERTS; v++) {
      expect(resolved.data[v * 4 + 0]).toBe(rootIdx);
      expect(resolved.data[v * 4 + 2]).toBe(elbowIdx);
    }
  });

  it("carries the weights through unchanged", () => {
    const flattened = flattenGroups(scene(0, blendedMesh()) as any);
    const weights = ramp();
    const resolved = resolveTransformRefs(
      blendedMesh({ weights }).geometry,
      flattened.namedTransforms,
    )!;
    for (let v = 0; v < N_VERTS; v++) {
      expect(resolved.data[v * 4 + 1]).toBeCloseTo(weights[v * 2], 6);
      expect(resolved.data[v * 4 + 3]).toBeCloseTo(weights[v * 2 + 1], 6);
    }
  });

  it("a missing Group name is a LOUD error naming what does exist", () => {
    const flattened = flattenGroups(scene(0, blendedMesh()) as any);
    expect(() =>
      resolveTransformRefs(
        blendedMesh({ refs: ["root", "shoulder"] }).geometry,
        flattened.namedTransforms,
      ),
    ).toThrow(/"shoulder"/);
    // The message must list the names that DO resolve, or the author has to
    // guess what they mistyped.
    expect(() =>
      resolveTransformRefs(
        blendedMesh({ refs: ["root", "shoulder"] }).geometry,
        flattened.namedTransforms,
      ),
    ).toThrow(/"elbow"/);
  });

  it("a scene with no named groups still fails loudly rather than silently", () => {
    expect(() =>
      resolveTransformRefs(blendedMesh().geometry, undefined),
    ).toThrow(/does not define/);
  });

  it("a partial declaration is an error", () => {
    const geometry: any = {
      positions: POSITIONS,
      transformRefs: ["root"],
    };
    expect(() =>
      resolveTransformRefs(geometry, new Map([["root", 1]])),
    ).toThrow(/must be supplied together/);
  });

  it("mismatched index/weight lengths are an error", () => {
    const geometry: any = {
      positions: POSITIONS,
      transformRefs: ["root", "elbow"],
      transformIndices: SLOTS,
      transformWeights: new Float32Array(N_VERTS),
    };
    expect(() =>
      resolveTransformRefs(
        geometry,
        new Map([
          ["root", 1],
          ["elbow", 2],
        ]),
      ),
    ).toThrow(/same shape/);
  });

  it("a slot outside transform_refs is an error", () => {
    const geometry: any = {
      positions: POSITIONS,
      transformRefs: ["root"],
      transformIndices: SLOTS, // references slot 1, which does not exist
      transformWeights: ramp(),
    };
    expect(() =>
      resolveTransformRefs(geometry, new Map([["root", 1]])),
    ).toThrow(/outside the 1 slot/);
  });

  it("no declaration resolves to nothing", () => {
    expect(
      resolveTransformRefs(rigidMesh().geometry, new Map()),
    ).toBeUndefined();
  });
});

// =============================================================================
// 2. Identity / variant key
// =============================================================================

describe("blended meshes are a distinct pipeline variant", () => {
  beforeEach(() => {
    setupWebGPU();
    clearInlineMeshCache();
  });
  afterEach(() => cleanupWebGPU());

  it("K is part of the format key", () => {
    const format = {
      hasNormals: true,
      hasColors: false,
      colorComponents: 3 as const,
      hasUVs: false,
    };
    const rigid = getFormatKey(format, "lit", false, 0);
    const k2 = getFormatKey(format, "lit", false, 2);
    const k3 = getFormatKey(format, "lit", false, 3);
    expect(new Set([rigid, k2, k3]).size).toBe(3);
  });

  it("a blended and a rigid mesh of identical geometry get different specs", () => {
    const flattened = flattenGroups(scene(0, blendedMesh()) as any);
    const { components, inlineSpecs } = resolveInlineMeshes(
      [blendedMesh(), rigidMesh()] as any,
      flattened.namedTransforms,
    );
    // Same positions, same indices, same format - but one blends. They must
    // not collapse onto one entry: their shaders and pipeline layouts differ.
    expect(components[0].type).not.toBe(components[1].type);
    expect(Object.keys(inlineSpecs!)).toHaveLength(2);
    expect((inlineSpecs as any)[components[0].type].transformRefCount).toBe(2);
    expect(
      (inlineSpecs as any)[components[1].type].transformRefCount,
    ).toBeUndefined();
  });

  it("a blended mesh keeps its identity across recompiles", () => {
    const flattened = flattenGroups(scene(0, blendedMesh()) as any);
    const first = resolveInlineMeshes(
      [blendedMesh()] as any,
      flattened.namedTransforms,
    );
    const second = resolveInlineMeshes(
      [blendedMesh()] as any,
      flattened.namedTransforms,
    );
    expect(second.components[0].type).toBe(first.components[0].type);
  });

  it("a weights change keeps the identity but bumps only the references", () => {
    const flattened = flattenGroups(scene(0, blendedMesh()) as any);
    const first = resolveInlineMeshes(
      [blendedMesh({ weights: ramp(0) })] as any,
      flattened.namedTransforms,
    );
    const specA = Object.values(first.inlineSpecs!)[0] as any;
    const geoKeyA = specA.geometryKey;
    const refKeyA = specA.transformRefsKey;

    const second = resolveInlineMeshes(
      [blendedMesh({ weights: ramp(0.2) })] as any,
      flattened.namedTransforms,
    );
    const specB = Object.values(second.inlineSpecs!)[0] as any;

    expect(second.components[0].type).toBe(first.components[0].type);
    // Geometry contents did NOT change - the vertex buffer must not be
    // rewritten for a weights edit.
    expect(specB.geometryKey).toBe(geoKeyA);
    // The references did.
    expect(specB.transformRefsKey).not.toBe(refKeyA);
  });

  it("the blended shader applies the blend in the picking pass too", () => {
    // pick-at on a deformed region must hit the deformed surface, not the
    // rest pose - so the picking vertex shader runs the identical blend.
    const flattened = flattenGroups(scene(0, blendedMesh()) as any);
    const { inlineSpecs } = resolveInlineMeshes(
      [blendedMesh()] as any,
      flattened.namedTransforms,
    );
    const spec = Object.values(inlineSpecs!)[0] as any;
    // Both pipelines are generated from the same spec; reach the shader source
    // through the pipeline-creation path by inspecting what it would compile.
    const shaders: string[] = [];
    const device = {
      createBindGroupLayout: vi.fn(() => ({})),
      createPipelineLayout: vi.fn(() => ({})),
      createShaderModule: vi.fn((d: any) => {
        shaders.push(d.code);
        return {};
      }),
      createRenderPipeline: vi.fn(() => ({})),
    } as unknown as GPUDevice;
    const cache = new Map();
    Object.defineProperty(navigator, "gpu", {
      value: { getPreferredCanvasFormat: () => "rgba8unorm" },
      configurable: true,
    });
    spec.getRenderPipeline(device, {} as any, cache);
    spec.getPickingPipeline(device, {} as any, cache);

    const vertexShaders = shaders.filter((s) => s.includes("@vertex"));
    expect(vertexShaders).toHaveLength(2);
    for (const src of vertexShaders) {
      expect(src).toContain("blendedFrame(localPos, vertexIndex)");
      expect(src).toContain("vertexTransformRefs");
      // ...and it must NOT also apply the component's own group transform,
      // which the referenced entries already include.
      expect(src).not.toContain("rigidGroupFrame");
    }
  });
});

// =============================================================================
// 3. Blend correctness at the data level
// =============================================================================

describe("blend math against the real palette", () => {
  beforeEach(() => clearInlineMeshCache());

  /**
   * The shader's blend, evaluated against the REAL resolved buffer and the
   * REAL flattened palette. The arithmetic here is the test's expectation of
   * what `blendedGroupTransformFn` computes (WGSL cannot run in vitest); the
   * inputs it reads are entirely produced by shipped code.
   */
  function blend(
    localPos: [number, number, number],
    vertex: number,
    resolved: { count: number; data: Float32Array },
    palette: ReturnType<typeof flattenGroups>["transforms"],
  ): [number, number, number] {
    let out: [number, number, number] = [0, 0, 0];
    for (let k = 0; k < resolved.count; k++) {
      const base = (vertex * resolved.count + k) * 2;
      const t = palette[resolved.data[base]];
      const w = resolved.data[base + 1];
      const scaled: [number, number, number] = [
        localPos[0] * t.scale[0],
        localPos[1] * t.scale[1],
        localPos[2] * t.scale[2],
      ];
      const r = quatRotate(t.quaternion, scaled);
      out = [
        out[0] + w * (r[0] + t.position[0]),
        out[1] + w * (r[1] + t.position[1]),
        out[2] + w * (r[2] + t.position[2]),
      ];
    }
    return out;
  }

  it("w = [1, 0] lands exactly where the first transform puts it", () => {
    const flattened = flattenGroups(scene(45, blendedMesh()) as any);
    const weights = new Float32Array(N_VERTS * 2);
    for (let i = 0; i < N_VERTS; i++) {
      weights[i * 2] = 1;
      weights[i * 2 + 1] = 0;
    }
    const resolved = resolveTransformRefs(
      blendedMesh({ weights }).geometry,
      flattened.namedTransforms,
    )!;

    const local: [number, number, number] = [0, 0, 2];
    const got = blend(local, 3, resolved, flattened.transforms);
    // "root" is identity here, so the vertex must land on its own local
    // position - untouched by the 45-degree elbow it also references.
    expect(got[0]).toBeCloseTo(0, 6);
    expect(got[1]).toBeCloseTo(0, 6);
    expect(got[2]).toBeCloseTo(2, 6);
  });

  it("w = [0.5, 0.5] between two translations lands at the midpoint", () => {
    // Two named groups, pure translations, so the blend has an exact answer.
    const components = [
      {
        type: "Group",
        name: "a",
        position: [0, 0, 0],
        children: [
          {
            ...blendedMesh({ refs: ["a", "b"] }),
            geometry: {
              positions: POSITIONS,
              indices: INDICES,
              transformRefs: ["a", "b"],
              transformIndices: SLOTS,
              transformWeights: new Float32Array(
                Array.from({ length: N_VERTS }, () => [0.5, 0.5]).flat(),
              ),
            },
          },
          {
            type: "Group",
            name: "b",
            position: [10, 0, 0],
            children: [
              { type: "PointCloud", centers: new Float32Array([0, 0, 0]) },
            ],
          },
        ],
      },
    ];
    const flattened = flattenGroups(components as any);
    const resolved = resolveTransformRefs(
      (components[0].children[0] as any).geometry,
      flattened.namedTransforms,
    )!;

    const got = blend([0, 0, 1], 2, resolved, flattened.transforms);
    // a puts it at (0,0,1); b puts it at (10,0,1); half-and-half is x = 5.
    expect(got[0]).toBeCloseTo(5, 6);
    expect(got[1]).toBeCloseTo(0, 6);
    expect(got[2]).toBeCloseTo(1, 6);
  });

  it("a rotated reference moves the vertex through that rotation", () => {
    const flattened = flattenGroups(scene(90, blendedMesh()) as any);
    const weights = new Float32Array(N_VERTS * 2);
    for (let i = 0; i < N_VERTS; i++) {
      weights[i * 2] = 0;
      weights[i * 2 + 1] = 1; // all elbow
    }
    const resolved = resolveTransformRefs(
      blendedMesh({ weights }).geometry,
      flattened.namedTransforms,
    )!;

    // The elbow sits at z = 1.5 and rotates 90 degrees about +X, which maps
    // local +Z onto world -Y.
    const got = blend([0, 0, 1], 0, resolved, flattened.transforms);
    expect(got[0]).toBeCloseTo(0, 5);
    expect(got[1]).toBeCloseTo(-1, 5);
    expect(got[2]).toBeCloseTo(1.5, 5);
  });

  it("weights that sum to 1 keep the blend inside the span of its references", () => {
    const flattened = flattenGroups(scene(60, blendedMesh()) as any);
    const resolved = resolveTransformRefs(
      blendedMesh().geometry,
      flattened.namedTransforms,
    )!;
    const local: [number, number, number] = [0, 0, 1];
    const a = blend(local, 0, resolved, flattened.transforms); // w = [1, 0]
    const b = blend(local, N_VERTS - 1, resolved, flattened.transforms); // [0, 1]
    for (let v = 1; v < N_VERTS - 1; v++) {
      const mid = blend(local, v, resolved, flattened.transforms);
      for (let axis = 0; axis < 3; axis++) {
        const lo = Math.min(a[axis], b[axis]) - 1e-5;
        const hi = Math.max(a[axis], b[axis]) + 1e-5;
        expect(mid[axis]).toBeGreaterThanOrEqual(lo);
        expect(mid[axis]).toBeLessThanOrEqual(hi);
      }
    }
  });
});

// =============================================================================
// 4. The write separation (counting harness)
// =============================================================================

interface Counters {
  pipelines: number;
  shaderModules: number;
  buffersCreated: number;
  buffersDestroyed: number;
  bytesAllocated: number;
  bytesDestroyed: number;
  /** writeBuffer calls targeting a VERTEX/INDEX buffer. */
  geometryWrites: number;
  /** writeBuffer calls targeting a STORAGE buffer labelled as references. */
  transformRefWrites: number;
}

function makeCountingDevice() {
  const counters: Counters = {
    pipelines: 0,
    shaderModules: 0,
    buffersCreated: 0,
    buffersDestroyed: 0,
    bytesAllocated: 0,
    bytesDestroyed: 0,
    geometryWrites: 0,
    transformRefWrites: 0,
  };

  const GEOMETRY_USAGE =
    (globalThis as any).GPUBufferUsage.VERTEX |
    (globalThis as any).GPUBufferUsage.INDEX;

  const createBuffer = vi.fn((desc: GPUBufferDescriptor) => {
    counters.buffersCreated++;
    counters.bytesAllocated += desc.size;
    let destroyed = false;
    return {
      size: desc.size,
      usage: desc.usage,
      label: desc.label,
      destroy: vi.fn(() => {
        if (destroyed) return;
        destroyed = true;
        counters.buffersDestroyed++;
        counters.bytesDestroyed += desc.size;
      }),
      mapAsync: vi.fn().mockResolvedValue(undefined),
      getMappedRange: vi.fn(() => new ArrayBuffer(desc.size)),
      unmap: vi.fn(),
    };
  });

  const queue = {
    writeBuffer: vi.fn((buffer: any) => {
      if (!buffer) return;
      if ((buffer.usage & GEOMETRY_USAGE) !== 0) counters.geometryWrites++;
      if (String(buffer.label ?? "").startsWith("Transform references")) {
        counters.transformRefWrites++;
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

describe("blended mesh resource lifecycle", () => {
  let container: HTMLDivElement;
  let device: GPUDevice;
  let counters: Counters;
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

  async function sweep(
    frames: any[][],
    snapshotAfter = 2,
  ): Promise<{ baseline: Counters; final: Counters }> {
    let result: ReturnType<typeof render> | undefined;
    let baseline: Counters | undefined;

    for (let i = 0; i < frames.length; i++) {
      const compiled = compileScene(frames[i] as any);
      const element = (
        <Wrapped
          components={compiled.components}
          primitiveSpecs={compiled.primitiveSpecs}
          transforms={compiled.transforms}
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

  it("a weights-only change writes the reference buffer and NOT the vertex buffer", async () => {
    // The whole reason the references are a storage buffer rather than vertex
    // attributes: editing weights must not touch the interleaved vertices.
    const N = 8;
    const frames = Array.from({ length: N }, (_, i) => [
      scene(30, blendedMesh({ weights: ramp(i * 0.02) }))[0],
    ]);

    const { baseline, final } = await sweep(frames);

    // Vertex/index buffer traffic is FLAT across the sweep.
    expect(final.geometryWrites).toBe(baseline.geometryWrites);
    // The changed weights DID reach the GPU: exactly one reference-buffer
    // write per changed frame, and nothing else.
    expect(final.transformRefWrites).toBe(
      baseline.transformRefWrites + (N - 2),
    );
    // And nothing churned or leaked.
    expect(final.pipelines).toBe(baseline.pipelines);
    expect(final.shaderModules).toBe(baseline.shaderModules);
    expect(final.bytesAllocated - final.bytesDestroyed).toBe(
      baseline.bytesAllocated - baseline.bytesDestroyed,
    );
  });

  it("a palette-only sweep writes NEITHER buffer and allocates nothing", async () => {
    // The headline: a Group quaternion driven by $state animates the blend
    // with zero geometry AND zero reference traffic - only the tiny
    // transforms palette is repacked, which every rigid scene already does.
    const N = 10;
    const weights = ramp();
    const frames = Array.from({ length: N }, (_, i) => [
      scene(-60 + (i * 120) / (N - 1), blendedMesh({ weights }))[0],
    ]);

    const { baseline, final } = await sweep(frames);

    expect(final.geometryWrites).toBe(baseline.geometryWrites);
    expect(final.transformRefWrites).toBe(baseline.transformRefWrites);
    expect(final.pipelines).toBe(baseline.pipelines);
    expect(final.shaderModules).toBe(baseline.shaderModules);
    expect(final.bytesAllocated - final.bytesDestroyed).toBe(
      baseline.bytesAllocated - baseline.bytesDestroyed,
    );
  });

  it("a positions change still rides the geometry write path", async () => {
    // The two paths stay independent in both directions: moving vertices
    // writes geometry and leaves the references alone.
    const N = 8;
    const weights = ramp();
    const frames = Array.from({ length: N }, (_, i) => {
      const positions = new Float32Array(POSITIONS);
      for (let v = 0; v < N_VERTS; v++) positions[v * 3] = i * 0.01 * v;
      return [scene(30, blendedMesh({ weights, positions }))[0]];
    });

    const { baseline, final } = await sweep(frames);

    // 2 geometry writes per changed frame (interleaved vertex + index)...
    expect(final.geometryWrites).toBe(baseline.geometryWrites + 2 * (N - 2));
    // ...and the references, unchanged, are not rewritten.
    expect(final.transformRefWrites).toBe(baseline.transformRefWrites);
    expect(final.bytesAllocated - final.bytesDestroyed).toBe(
      baseline.bytesAllocated - baseline.bytesDestroyed,
    );
  });

  it("the blended mesh renders (its pipelines and buffers are built)", async () => {
    const { final } = await sweep([[scene(30, blendedMesh())[0]]], 1);
    expect(final.pipelines).toBeGreaterThan(0);
    expect(final.transformRefWrites).toBeGreaterThan(0);
  });
});
