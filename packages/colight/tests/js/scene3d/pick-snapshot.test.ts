/**
 * Tests for the GPU-free half of the agent-facing pick snapshot API:
 * legend building/decoding, id-buffer base64 encoding, selection bounds
 * and camera fitting. The GPU readback itself is exercised end-to-end by
 * the Python CLI tests (test_cli_pick.py) against real Chrome.
 */

import { describe, it, expect } from "vitest";
import {
  buildPickLegend,
  decodePickId,
  u32ToBase64,
  base64ToU32,
  inRanges,
  elementRadius,
  computeSelectionBounds,
  unionBounds,
  describeInstance,
} from "../../../src/js/scene3d/pick-snapshot";
import { fitCameraToBounds } from "../../../src/js/scene3d/camera3d";
import { packID } from "../../../src/js/scene3d/picking";
import {
  pointCloudSpec,
  ellipsoidSpec,
  cuboidSpec,
  ComponentConfig,
} from "../../../src/js/scene3d/components";
import { IDENTITY_GPU_TRANSFORM } from "../../../src/js/scene3d/gpu-transforms";
import type { ComponentOffset } from "../../../src/js/scene3d/types";

const CUBOID = (extra: Record<string, unknown> = {}): ComponentConfig =>
  ({
    type: "Cuboid",
    centers: new Float32Array([0, 0, 0, 2, 0, 0]),
    ...extra,
  }) as unknown as ComponentConfig;

describe("buildPickLegend", () => {
  it("serializes the interactive componentOffsets registry", () => {
    const components = [
      CUBOID(),
      {
        type: "PointCloud",
        centers: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      } as unknown as ComponentConfig,
    ];
    // Two render objects (one per primitive type), exactly as impl3d
    // builds them: pickingStart is the global element index.
    const offsets: ComponentOffset[][] = [
      [{ componentIdx: 1, elementStart: 0, pickingStart: 2, elementCount: 3 }],
      [{ componentIdx: 0, elementStart: 0, pickingStart: 0, elementCount: 2 }],
    ];

    const legend = buildPickLegend(components, offsets);
    expect(legend).toEqual([
      { component: 0, type: "Cuboid", count: 2, idBase: 0 },
      { component: 1, type: "PointCloud", count: 3, idBase: 2 },
    ]);
  });

  it("carries group paths", () => {
    const component = CUBOID();
    (component as any)._groupPath = ["robot", "arm"];
    const legend = buildPickLegend(
      [component],
      [
        [
          {
            componentIdx: 0,
            elementStart: 0,
            pickingStart: 0,
            elementCount: 2,
          },
        ],
      ],
    );
    expect(legend[0].groupPath).toEqual(["robot", "arm"]);
  });
});

describe("decodePickId", () => {
  const legend = [
    { component: 0, type: "Cuboid", count: 2, idBase: 0 },
    { component: 1, type: "PointCloud", count: 3, idBase: 2 },
  ];

  it("decodes background and instances with packID semantics", () => {
    expect(decodePickId(0, legend)).toBeNull();
    expect(decodePickId(packID(0), legend)).toEqual({
      component: 0,
      instance: 0,
    });
    expect(decodePickId(packID(1), legend)).toEqual({
      component: 0,
      instance: 1,
    });
    expect(decodePickId(packID(2), legend)).toEqual({
      component: 1,
      instance: 0,
    });
    expect(decodePickId(packID(4), legend)).toEqual({
      component: 1,
      instance: 2,
    });
    expect(decodePickId(packID(5), legend)).toBeNull();
  });
});

describe("u32 base64 encoding", () => {
  it("matches Node's base64 for little-endian bytes", () => {
    const ids = new Uint32Array([0, 1, 0xfffffe, 0x00badca7]);
    const expected = Buffer.from(
      ids.buffer,
      ids.byteOffset,
      ids.byteLength,
    ).toString("base64");
    expect(u32ToBase64(ids)).toBe(expected);
  });

  it("round-trips", () => {
    const ids = new Uint32Array([7, 0, 42, 0xffffff, 123456]);
    expect(Array.from(base64ToU32(u32ToBase64(ids)))).toEqual(Array.from(ids));
  });
});

describe("inRanges", () => {
  it("treats missing ranges as select-all and bounds as inclusive", () => {
    expect(inRanges(5)).toBe(true);
    expect(inRanges(5, [])).toBe(true);
    expect(inRanges(5, [[0, 4]])).toBe(false);
    expect(inRanges(4, [[0, 4]])).toBe(true);
    expect(
      inRanges(7, [
        [0, 2],
        [7, 7],
      ]),
    ).toBe(true);
  });
});

describe("elementRadius", () => {
  it("prefers per-instance half_sizes, then half_size, then spec defaults", () => {
    const perInstance = CUBOID({
      half_sizes: new Float32Array([1, 1, 1, 2, 2, 2]),
    });
    expect(elementRadius(perInstance, cuboidSpec, 1)).toBeCloseTo(
      Math.sqrt(12),
    );

    const constant = CUBOID({ half_size: [3, 0, 4] });
    expect(elementRadius(constant, cuboidSpec, 0)).toBeCloseTo(5);

    // No sizes at all: the spec default (cuboid half_size 0.1) applies.
    expect(elementRadius(CUBOID(), cuboidSpec, 0)).toBeCloseTo(
      0.1 * Math.sqrt(3),
    );
  });

  it("uses point sizes and scale multipliers", () => {
    const cloud = {
      type: "PointCloud",
      centers: new Float32Array([0, 0, 0, 1, 1, 1]),
      sizes: new Float32Array([0.5, 2]),
      scales: new Float32Array([1, 3]),
    } as unknown as ComponentConfig;
    expect(elementRadius(cloud, pointCloudSpec, 1)).toBeCloseTo(6);
  });
});

describe("computeSelectionBounds", () => {
  const transforms = [IDENTITY_GPU_TRANSFORM];

  it("expands centers by the per-element radius", () => {
    const component = CUBOID({ half_size: [0.5, 0.5, 0.5] });
    const bounds = computeSelectionBounds(component, cuboidSpec, transforms);
    const radius = Math.sqrt(0.75);
    expect(bounds).not.toBeNull();
    expect(bounds!.min[0]).toBeCloseTo(-radius);
    expect(bounds!.max[0]).toBeCloseTo(2 + radius);
    expect(bounds!.min[1]).toBeCloseTo(-radius);
  });

  it("restricts to instance ranges", () => {
    const component = CUBOID({ half_size: [0.5, 0.5, 0.5] });
    const bounds = computeSelectionBounds(component, cuboidSpec, transforms, [
      [1, 1],
    ]);
    expect(bounds!.min[0]).toBeCloseTo(2 - Math.sqrt(0.75));
    expect(bounds!.max[0]).toBeCloseTo(2 + Math.sqrt(0.75));
  });

  it("returns null for empty selections", () => {
    const component = CUBOID();
    expect(
      computeSelectionBounds(component, cuboidSpec, transforms, [[5, 9]]),
    ).toBeNull();
  });

  it("applies the component's group transform", () => {
    const component = CUBOID({ half_size: [1, 1, 1] });
    (component as any)._transformIndex = 1;
    const withTransform = [
      IDENTITY_GPU_TRANSFORM,
      {
        position: [10, 0, 0] as [number, number, number],
        quaternion: [0, 0, 0, 1] as [number, number, number, number],
        scale: [2, 2, 2] as [number, number, number],
      },
    ];
    const bounds = computeSelectionBounds(
      component,
      cuboidSpec,
      withTransform,
      [[0, 0]],
    );
    // Center [0,0,0] -> scaled+translated to [10,0,0]; radius sqrt(3)*2.
    expect(bounds!.min[0]).toBeCloseTo(10 - Math.sqrt(3) * 2);
    expect(bounds!.max[0]).toBeCloseTo(10 + Math.sqrt(3) * 2);
  });

  it("unions bounds", () => {
    const a = { min: [0, 0, 0] as any, max: [1, 1, 1] as any };
    const b = { min: [-2, 0.5, 0] as any, max: [0.5, 3, 1] as any };
    expect(unionBounds(a, b)).toEqual({ min: [-2, 0, 0], max: [1, 3, 1] });
    expect(unionBounds(null, a)).toBe(a);
    expect(unionBounds(a, null)).toBe(a);
  });
});

describe("describeInstance", () => {
  it("dereferences per-instance arrays", () => {
    const component = {
      type: "Ellipsoid",
      centers: new Float32Array([0, 0, 0, 1, 2, 3]),
      colors: new Float32Array([1, 0, 0, 0, 0, 1]),
      alphas: new Float32Array([0.5, 0.75]),
      half_sizes: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    } as unknown as ComponentConfig;
    const values = describeInstance(component, ellipsoidSpec, 1);
    expect(values.center).toEqual([1, 2, 3]);
    expect(values.color).toEqual([0, 0, 1]);
    expect(values.alpha).toBeCloseTo(0.75);
    expect(values.half_size).toEqual([
      expect.closeTo(0.4),
      expect.closeTo(0.5),
      expect.closeTo(0.6),
    ]);
  });

  it("falls back to constants and spec defaults", () => {
    const component = {
      type: "Ellipsoid",
      centers: new Float32Array([5, 5, 5]),
      color: [0.2, 0.4, 0.6],
    } as unknown as ComponentConfig;
    const values = describeInstance(component, ellipsoidSpec, 0);
    expect(values.center).toEqual([5, 5, 5]);
    expect(values.color).toEqual([0.2, 0.4, 0.6]);
    expect(values.alpha).toBe(1);
    // Ellipsoid spec default half_size
    expect(values.half_size).toEqual([0.5, 0.5, 0.5]);
  });
});

describe("fitCameraToBounds", () => {
  const camera = {
    position: [0, 0, 10] as [number, number, number],
    target: [0, 0, 0] as [number, number, number],
    up: [0, 1, 0] as [number, number, number],
    fov: 45,
    near: 0.001,
    far: 100,
  };

  it("targets the bounds center and preserves direction/up/fov", () => {
    const fitted = fitCameraToBounds(
      camera,
      { min: [1, 1, 1], max: [3, 3, 3] },
      1,
    );
    expect(fitted.target).toEqual([2, 2, 2]);
    expect(fitted.up).toEqual([0, 1, 0]);
    expect(fitted.fov).toBe(45);
    // Direction preserved: position = target + dir * distance with dir = +z.
    expect(fitted.position[0]).toBeCloseTo(2);
    expect(fitted.position[1]).toBeCloseTo(2);
    expect(fitted.position[2]).toBeGreaterThan(2);
  });

  it("places the bounding sphere inside the frustum", () => {
    const bounds = { min: [-1, -1, -1] as any, max: [1, 1, 1] as any };
    const fitted = fitCameraToBounds(camera, bounds, 1);
    const radius = Math.sqrt(3);
    const distance = Math.hypot(
      fitted.position[0] - fitted.target[0],
      fitted.position[1] - fitted.target[1],
      fitted.position[2] - fitted.target[2],
    );
    const halfFov = (45 / 2) * (Math.PI / 180);
    expect(distance).toBeGreaterThanOrEqual(radius / Math.sin(halfFov));
    // Near/far bracket the sphere.
    expect(fitted.near).toBeLessThan(distance - radius);
    expect(fitted.far).toBeGreaterThan(distance + radius);
  });

  it("handles degenerate (single point) bounds without NaN", () => {
    const fitted = fitCameraToBounds(
      camera,
      { min: [1, 1, 1], max: [1, 1, 1] },
      1,
    );
    expect(Number.isFinite(fitted.position[2])).toBe(true);
    expect(fitted.target).toEqual([1, 1, 1]);
  });

  it("accounts for narrow aspect ratios", () => {
    const wide = fitCameraToBounds(
      camera,
      { min: [-1, -1, -1], max: [1, 1, 1] },
      1,
    );
    const narrow = fitCameraToBounds(
      camera,
      { min: [-1, -1, -1], max: [1, 1, 1] },
      0.5,
    );
    const dist = (c: {
      position: ArrayLike<number>;
      target: ArrayLike<number>;
    }) =>
      Math.hypot(
        c.position[0] - c.target[0],
        c.position[1] - c.target[1],
        c.position[2] - c.target[2],
      );
    // Narrower viewport needs more distance to fit the same sphere.
    expect(dist(narrow)).toBeGreaterThan(dist(wide));
  });
});
