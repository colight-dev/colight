/**
 * D2-pre: the four primitives that hand-write their shaders (BoundingBox,
 * EllipsoidAxes, ImagePlane, Mesh) must agree with the generated path on the
 * three things the schema decides — instance stride, shader locations, and the
 * auto-injected framework attributes (transformIndex / filterValue /
 * filterIndex).
 *
 * Regression tests for three live defects:
 *   1. BoundingBox never read the transforms buffer, so a BoundingBox inside a
 *      transformed Group rendered untransformed.
 *   2. All four skipped the filterValue/filterIndex collapse, so `filter_by`
 *      silently no-opped on them.
 *   3. Mesh's hand-written instance layouts declared 15-float (render) and
 *      12-float (picking) strides while the schema-driven fill wrote 17 and 14.
 *      The picking mismatch put pickID where filterValue lands, so a Mesh was
 *      entirely invisible to the pick pass — even a single instance.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { setupWebGPU, cleanupWebGPU } from "../webgpu-setup";
import {
  buildRenderData,
  buildPickingData,
  boundingBoxSpec,
  ellipsoidAxesSpec,
  imagePlaneSpec,
} from "../../../src/js/scene3d/components";
import { defineMesh } from "../../../src/js/scene3d/primitives/mesh";
import { unpackID } from "../../../src/js/scene3d/picking";
import { compileScene } from "../../../src/js/scene3d/compiler";
import type {
  PrimitiveSpec,
  VertexBufferLayout,
} from "../../../src/js/scene3d/types";

// The pipeline factories reference WebGPU enum globals; the shader-source
// checks below drive them, so the jsdom shims must be in place.
beforeAll(() => setupWebGPU());
afterAll(() => cleanupWebGPU());

// A unit triangle is enough geometry: these tests are about the instance
// record, not the vertices.
const TRIANGLE = {
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
  indices: new Uint16Array([0, 1, 2]),
};

/** Fill a component's render buffer exactly as impl3d does. */
function renderInstances(
  spec: PrimitiveSpec<any>,
  elem: unknown,
): Float32Array {
  const count = spec.getElementCount(elem as any) * spec.instancesPerElement;
  const out = new Float32Array(count * spec.floatsPerInstance);
  buildRenderData(elem as any, spec as any, out, 0);
  return out;
}

function pickInstances(
  spec: PrimitiveSpec<any>,
  elem: unknown,
  baseID = 0,
): Float32Array {
  const count = spec.getElementCount(elem as any) * spec.instancesPerElement;
  const out = new Float32Array(count * spec.floatsPerPicking);
  buildPickingData(elem as any, spec as any, out, baseID, 0);
  return out;
}

/** Names of the attributes a layout declares, by shader location. */
function locations(layout: VertexBufferLayout): number[] {
  return layout.attributes.map((a) => a.shaderLocation);
}

const ALL_SPECS: Array<[string, PrimitiveSpec<any>]> = [
  ["BoundingBox", boundingBoxSpec as any],
  ["EllipsoidAxes", ellipsoidAxesSpec as any],
  ["ImagePlane", imagePlaneSpec as any],
  ["Mesh", defineMesh("__test_mesh_layout", TRIANGLE) as any],
];

describe("declared GPU stride matches the CPU fill stride", () => {
  // This is THE invariant defect 3 violated. The CPU fill writes at
  // `floatsPerInstance` floats per instance; the GPU steps the instance
  // buffer by `arrayStride` bytes. When they disagree, instance 0 still
  // looks right and every later instance reads garbage — and in the picking
  // pass pickID lands on the wrong float, killing picking outright.
  for (const [name, spec] of ALL_SPECS) {
    it(`${name} render layout`, () => {
      expect(spec.renderInstanceLayout.arrayStride).toBe(
        spec.floatsPerInstance * 4,
      );
    });

    it(`${name} picking layout`, () => {
      expect(spec.pickingInstanceLayout.arrayStride).toBe(
        spec.floatsPerPicking * 4,
      );
    });

    it(`${name} declares contiguous shader locations`, () => {
      for (const layout of [
        spec.renderInstanceLayout,
        spec.pickingInstanceLayout,
      ]) {
        const locs = locations(layout);
        for (let i = 1; i < locs.length; i++) {
          expect(locs[i]).toBe(locs[i - 1] + 1);
        }
      }
      // Picking carries one extra slot (pickID) beyond render's non-color
      // attributes, and both start at the same location.
      expect(locations(spec.pickingInstanceLayout)[0]).toBe(
        locations(spec.renderInstanceLayout)[0],
      );
    });
  }
});

describe("framework attributes are present on every hand-written primitive", () => {
  // transformIndex + filterValue + filterIndex are auto-injected by
  // processSchema; skipping them is what made filter_by a no-op and (for
  // BoundingBox) made group transforms unreachable.
  const FRAMEWORK_FLOATS = 3;

  const cases: Array<[string, PrimitiveSpec<any>, number, number]> = [
    // name, spec, own render floats, own picking floats (before framework)
    // BoundingBox: center(3)+halfSize(3)+edgeSize(1)+rotation(4)+color(3)+alpha(1)
    ["BoundingBox", boundingBoxSpec as any, 15, 11],
    // EllipsoidAxes: position(3)+size(3)+rotation(4)+color(3)+alpha(1)
    ["EllipsoidAxes", ellipsoidAxesSpec as any, 14, 10],
    // ImagePlane: position(3)+rotation(4)+size(2)+color(3)+alpha(1)
    ["ImagePlane", imagePlaneSpec as any, 13, 9],
    // Mesh: position(3)+size(3)+rotation(4)+color(3)+alpha(1)
    ["Mesh", defineMesh("__test_mesh_framework", TRIANGLE) as any, 14, 10],
  ];

  for (const [name, spec, ownRender, ownPicking] of cases) {
    it(`${name} reserves transformIndex/filterValue/filterIndex slots`, () => {
      expect(spec.floatsPerInstance).toBe(ownRender + FRAMEWORK_FLOATS);
      // picking = own geometry attrs + framework + pickID
      expect(spec.floatsPerPicking).toBe(ownPicking + FRAMEWORK_FLOATS + 1);
    });
  }
});

describe("defect 3: Mesh instance records", () => {
  it("makes a single instance pickable (pickID at the declared offset)", () => {
    // The picking regression that mattered most: with a 12-float declared
    // stride and a 14-float fill, pickID landed on filterValue and every
    // Mesh was invisible to the pick pass.
    const spec = defineMesh("__test_mesh_pick_one", TRIANGLE);
    const elem = {
      type: "__test_mesh_pick_one",
      centers: new Float32Array([0, 0, 0]),
    };
    const data = pickInstances(spec, elem, 7);

    const pickIDOffset =
      spec.pickingInstanceLayout.attributes.at(-1)!.offset / 4;
    expect(pickIDOffset).toBe(spec.floatsPerPicking - 1);
    // The pickID slot must decode back to the element's global ID, not NaN
    // (which is what the filterValue default put there before the fix).
    expect(unpackID(data[pickIDOffset])).toBe(7);
  });

  it("writes instance 1's centre at instance 1's stride", () => {
    const spec = defineMesh("__test_mesh_two", TRIANGLE);
    const elem = {
      type: "__test_mesh_two",
      centers: new Float32Array([-3, 0, 0, 3, 0, 0]),
    };
    expect(spec.getElementCount(elem as any)).toBe(2);

    const data = renderInstances(spec, elem);
    const stride = spec.floatsPerInstance;
    expect(data.length).toBe(2 * stride);
    expect(stride).toBe(spec.renderInstanceLayout.arrayStride / 4);

    expect(Array.from(data.slice(0, 3))).toEqual([-3, 0, 0]);
    // The whole point: with the old 15-float declared stride the GPU read
    // these floats from the middle of instance 0.
    expect(Array.from(data.slice(stride, stride + 3))).toEqual([3, 0, 0]);

    // Every instance fully initialised — default scale and identity rotation,
    // not zeros left over from a short write.
    for (let i = 0; i < 2; i++) {
      const o = i * stride;
      expect(Array.from(data.slice(o + 3, o + 6))).toEqual([1, 1, 1]);
      expect(Array.from(data.slice(o + 6, o + 10))).toEqual([0, 0, 0, 1]);
    }
  });

  it("gives each instance a distinct pickID at the declared picking stride", () => {
    const spec = defineMesh("__test_mesh_two_pick", TRIANGLE);
    const elem = {
      type: "__test_mesh_two_pick",
      centers: new Float32Array([-3, 0, 0, 3, 0, 0]),
    };
    const data = pickInstances(spec, elem);
    const stride = spec.floatsPerPicking;
    expect(stride).toBe(spec.pickingInstanceLayout.arrayStride / 4);
    expect(unpackID(data[stride - 1])).toBe(0);
    expect(unpackID(data[2 * stride - 1])).toBe(1);
    expect(Array.from(data.slice(stride, stride + 3))).toEqual([3, 0, 0]);
  });
});

/**
 * Offset of a named schema attribute in the render instance record, read off
 * the declared vertex layout rather than hardcoded — the layout is what the
 * schema produced, so this stays true as attributes are added.
 */
function attrIndex(
  spec: PrimitiveSpec<any>,
  picking: boolean,
  i: number,
): number {
  const layout = picking
    ? spec.pickingInstanceLayout
    : spec.renderInstanceLayout;
  return layout.attributes[i].offset / 4;
}

describe("defect 1: BoundingBox honours group transforms", () => {
  it("carries a non-zero _transformIndex into the instance record", () => {
    // A BoundingBox inside a translated Group must reach the shader with the
    // group's transform slot. Before the fix its instance layout had no
    // transformIndex slot at all and its shader never read the transforms
    // buffer, so the box rendered untransformed.
    const { components, transforms } = compileScene([
      {
        type: "Group",
        position: [6, 0, 0],
        children: [
          {
            type: "BoundingBox",
            centers: new Float32Array([-3, 0, 0]),
            half_size: 0.5,
          },
        ],
      } as any,
    ]);

    const comp = components[0] as any;
    expect(comp.type).toBe("BoundingBox");
    const transformIndex = comp._transformIndex;
    expect(transformIndex).toBeGreaterThan(0);
    // The group's translation lives in that palette slot.
    expect(Array.from(transforms[transformIndex].position)).toEqual([6, 0, 0]);

    // ...and the fill writes it into the slot the layout declares. It is the
    // 7th attribute (center, halfSize, edgeSize, rotation, color, alpha, then
    // transformIndex).
    const data = renderInstances(boundingBoxSpec as any, comp);
    expect(data[attrIndex(boundingBoxSpec as any, false, 6)]).toBe(
      transformIndex,
    );
  });

  it("reads the transforms buffer in both its shaders", () => {
    // Structural check on the WGSL: the group transform must be composed via
    // the shared helper in the render AND picking passes, or a transformed
    // BoundingBox is right in one and wrong in the other.
    for (const wgsl of shaderSources(boundingBoxSpec as any)) {
      expect(wgsl).toContain("rigidGroupFrame(");
      expect(wgsl).toContain("transforms[");
    }
  });
});

/**
 * The vertex shader sources a spec hands to its render and picking pipelines,
 * captured through a stub device.
 */
function shaderSources(spec: PrimitiveSpec<any>): string[] {
  const sources: string[] = [];
  const device = {
    createShaderModule: ({ code }: { code: string }) => {
      sources.push(code);
      return {};
    },
    createPipelineLayout: () => ({}),
    createRenderPipeline: () => ({}),
    createBindGroupLayout: () => ({}),
  } as unknown as GPUDevice;
  (globalThis as any).navigator = {
    gpu: { getPreferredCanvasFormat: () => "bgra8unorm" },
  };
  spec.getRenderPipeline(device, {} as GPUBindGroupLayout, new Map());
  spec.getPickingPipeline(device, {} as GPUBindGroupLayout, new Map());
  const vertexShaders = sources.filter((s) => s.includes("@vertex"));
  expect(vertexShaders.length).toBe(2); // render + picking
  return vertexShaders;
}

describe("defect 2: filter_by reaches all four hand-written primitives", () => {
  for (const [name, spec] of ALL_SPECS) {
    it(`${name} vertex shaders apply the filter collapse`, () => {
      // Both passes must test AND collapse: a filtered-out instance has to be
      // invisible and unpickable. Before the fix none of these four emitted
      // either, so filter_by silently no-opped.
      for (const wgsl of shaderSources(spec)) {
        expect(wgsl).toContain("filterParams[u32(filterIndex)]");
        expect(wgsl).toContain("if (!_filterPass)");
      }
    });
  }

  it("writes a filtered component's per-instance values and slot index", () => {
    // End of the CPU half: compileScene assigns a filterParams slot and the
    // schema fill puts the per-instance value + slot into the instance record.
    const { components, filterParams } = compileScene([
      {
        type: "BoundingBox",
        centers: new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]),
        half_size: 0.5,
        filter_by: { values: [0.1, 0.5, 0.9], min: 0.3, label: "grade" },
      } as any,
    ]);
    const comp = components[0] as any;
    expect(comp._filterIndex).toBe(1);
    expect(filterParams[4 + 2]).toBe(1); // slot 1 isActive

    const data = renderInstances(boundingBoxSpec as any, comp);
    const stride = (boundingBoxSpec as any).floatsPerInstance;
    const fv = attrIndex(boundingBoxSpec as any, false, 7); // filterValue
    const fi = attrIndex(boundingBoxSpec as any, false, 8); // filterIndex
    expect(data[fv]).toBeCloseTo(0.1, 6);
    expect(data[stride + fv]).toBeCloseTo(0.5, 6);
    expect(data[2 * stride + fv]).toBeCloseTo(0.9, 6);
    for (let i = 0; i < 3; i++) {
      expect(data[i * stride + fi]).toBe(1);
    }
  });
});
