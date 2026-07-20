/**
 * Tests for named annotation callouts — anchor resolution + camera projection.
 *
 * Exercises the real annotations module against the real compileScene output
 * and the real camera transforms (createCameraState / projectToScreen), the
 * same path the overlay and the snapshot API use, so the tests can never drift
 * from what is rendered.
 */

import { describe, it, expect } from "vitest";
import { compileScene } from "../../../src/js/scene3d/compiler";
import { cuboidSpec } from "../../../src/js/scene3d/components";
import { IDENTITY_GPU_TRANSFORM } from "../../../src/js/scene3d/gpu-transforms";
import { createCameraState } from "../../../src/js/scene3d/camera3d";
import {
  resolveAnchorWorld,
  resolveAnnotations,
  annotationsForInstance,
  isInstanceAnchor,
  DEFAULT_ANNOTATION_COLOR,
  Annotations,
} from "../../../src/js/scene3d/annotations";

const SPECS = { Cuboid: cuboidSpec } as Record<string, typeof cuboidSpec>;
const specFor = (component: { type: string }) => SPECS[component.type];

function cuboidScene() {
  return compileScene([
    {
      type: "Cuboid",
      centers: new Float32Array([-3, 0, 0, -1, 0, 0, 1, 0, 0, 3, 0, 0]),
      half_size: 0.5,
    } as any,
  ]);
}

// A camera looking down -Z at the origin from z=12 (matches the docs scene).
const CAMERA = createCameraState({
  position: [0, 0, 12],
  target: [0, 0, 0],
  up: [0, 1, 0],
  fov: 45,
  near: 0.01,
  far: 100,
});
const RECT = { width: 400, height: 400 };

describe("isInstanceAnchor", () => {
  it("distinguishes the two anchor forms", () => {
    expect(isInstanceAnchor({ component: 0, instance: 2 })).toBe(true);
    expect(isInstanceAnchor({ position: [1, 2, 3] })).toBe(false);
  });
});

describe("resolveAnchorWorld", () => {
  it("resolves an instance anchor to that instance's center", () => {
    const { components, transforms } = cuboidScene();
    const world = resolveAnchorWorld(
      { component: 0, instance: 2 },
      components,
      specFor,
      transforms,
    );
    // Instance 2's center is [1, 0, 0].
    expect(world).not.toBeNull();
    expect(world![0]).toBeCloseTo(1);
    expect(world![1]).toBeCloseTo(0);
    expect(world![2]).toBeCloseTo(0);
  });

  it("returns null for an out-of-range instance anchor", () => {
    const { components, transforms } = cuboidScene();
    expect(
      resolveAnchorWorld(
        { component: 0, instance: 99 },
        components,
        specFor,
        transforms,
      ),
    ).toBeNull();
    expect(
      resolveAnchorWorld(
        { component: 5, instance: 0 },
        components,
        specFor,
        transforms,
      ),
    ).toBeNull();
  });

  it("passes a position anchor through unchanged when there is no origin", () => {
    const world = resolveAnchorWorld({ position: [10, 20, 30] }, [], specFor, [
      IDENTITY_GPU_TRANSFORM,
    ]);
    expect(world).toEqual([10, 20, 30]);
  });

  it("subtracts the scene origin from a position anchor (origin-aware)", () => {
    // Geometry is pre-shifted by -origin in Python; the world-coord anchor
    // must land in the same shifted space to line up with the geometry.
    const world = resolveAnchorWorld(
      { position: [445000, 10, 5] },
      [],
      specFor,
      [IDENTITY_GPU_TRANSFORM],
      [445000, 0, 0],
    );
    expect(world).toEqual([0, 10, 5]);
  });
});

describe("resolveAnnotations (project with the live camera)", () => {
  const annotations: Annotations = {
    "on-inst": {
      text: "instance 1",
      anchor: { component: 0, instance: 1 },
    },
    "on-pos": {
      text: "at origin",
      anchor: { position: [0, 0, 0] },
      style: { color: [0, 1, 0] },
    },
  };

  it("projects visible anchors to in-viewport screen positions", () => {
    const { components, transforms } = cuboidScene();
    const resolved = resolveAnnotations(
      annotations,
      components,
      specFor,
      transforms,
      CAMERA,
      RECT,
    );
    // Sorted by name: on-inst, on-pos.
    expect(resolved.map((a) => a.name)).toEqual(["on-inst", "on-pos"]);
    for (const a of resolved) {
      expect(a.visible).toBe(true);
      expect(a.screen).not.toBeNull();
      expect(a.screen!.x).toBeGreaterThanOrEqual(0);
      expect(a.screen!.x).toBeLessThanOrEqual(RECT.width);
      expect(a.screen!.y).toBeGreaterThanOrEqual(0);
      expect(a.screen!.y).toBeLessThanOrEqual(RECT.height);
    }
    // The origin anchor projects to the center of the viewport.
    const pos = resolved.find((a) => a.name === "on-pos")!;
    expect(pos.screen!.x).toBeCloseTo(200, 0);
    expect(pos.screen!.y).toBeCloseTo(200, 0);
    // Instance 1 (center x=-1) projects left of center.
    const inst = resolved.find((a) => a.name === "on-inst")!;
    expect(inst.screen!.x).toBeLessThan(200);
  });

  it("carries the style color, defaulting when absent", () => {
    const { components, transforms } = cuboidScene();
    const resolved = resolveAnnotations(
      annotations,
      components,
      specFor,
      transforms,
      CAMERA,
      RECT,
    );
    expect(resolved.find((a) => a.name === "on-pos")!.color).toEqual([0, 1, 0]);
    expect(resolved.find((a) => a.name === "on-inst")!.color).toEqual(
      DEFAULT_ANNOTATION_COLOR,
    );
  });

  it("reports visible:false for an anchor behind the camera", () => {
    // Anchor far behind the camera (camera at z=12 looking toward -z).
    const behind: Annotations = {
      back: { text: "behind", anchor: { position: [0, 0, 100] } },
    };
    const resolved = resolveAnnotations(
      behind,
      [],
      specFor,
      [IDENTITY_GPU_TRANSFORM],
      CAMERA,
      RECT,
    );
    expect(resolved[0].visible).toBe(false);
    expect(resolved[0].screen).toBeNull();
    // The world position is still resolved (for reporting).
    expect(resolved[0].world).toEqual([0, 0, 100]);
  });

  it("reports visible:false for an anchor outside the viewport", () => {
    // Far off to the side: in front of the camera but well outside the frustum.
    const offscreen: Annotations = {
      side: { text: "off", anchor: { position: [1000, 0, 0] } },
    };
    const resolved = resolveAnnotations(
      offscreen,
      [],
      specFor,
      [IDENTITY_GPU_TRANSFORM],
      CAMERA,
      RECT,
    );
    expect(resolved[0].visible).toBe(false);
    expect(resolved[0].screen).not.toBeNull();
    expect(resolved[0].screen!.x).toBeGreaterThan(RECT.width);
  });

  it("returns [] for undefined annotations", () => {
    expect(
      resolveAnnotations(undefined, [], specFor, [], CAMERA, RECT),
    ).toEqual([]);
  });
});

describe("annotationsForInstance (pick-at membership)", () => {
  const annotations: Annotations = {
    a: { text: "a", anchor: { component: 0, instance: 1 } },
    b: { text: "b", anchor: { component: 0, instance: 1 } },
    c: { text: "c", anchor: { component: 0, instance: 2 } },
    d: { text: "d", anchor: { position: [0, 0, 0] } },
  };

  it("names instance anchors on the hit instance (sorted, position anchors excluded)", () => {
    expect(annotationsForInstance(annotations, 0, 1)).toEqual(["a", "b"]);
    expect(annotationsForInstance(annotations, 0, 2)).toEqual(["c"]);
    expect(annotationsForInstance(annotations, 1, 1)).toEqual([]);
  });

  it("returns [] for undefined annotations", () => {
    expect(annotationsForInstance(undefined, 0, 1)).toEqual([]);
  });
});
