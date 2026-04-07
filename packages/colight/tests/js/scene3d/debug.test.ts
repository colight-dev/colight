import { describe, it, expect, beforeEach, afterEach } from "vitest";
import {
  DEFAULT_SAFARI_MAX_INSTANCES_PER_DRAW,
  buildDrawSlices,
  createScene3DDebugProbe,
  parseScene3DDebugOptions,
} from "../../../src/js/scene3d/debug";

describe("scene3d debug helpers", () => {
  const originalUserAgent = navigator.userAgent;

  beforeEach(() => {
    Object.defineProperty(navigator, "userAgent", {
      value: originalUserAgent,
      configurable: true,
    });
    delete (window as Window & { __COLIGHT_SCENE3D_DEBUG__?: unknown })
      .__COLIGHT_SCENE3D_DEBUG__;
  });

  afterEach(() => {
    Object.defineProperty(navigator, "userAgent", {
      value: originalUserAgent,
      configurable: true,
    });
    delete (window as Window & { __COLIGHT_SCENE3D_DEBUG__?: unknown })
      .__COLIGHT_SCENE3D_DEBUG__;
  });

  it("splits large instance draws into stable slices", () => {
    expect(buildDrawSlices(0, 1_000_000)).toEqual([]);
    expect(buildDrawSlices(250_000, 1_000_000)).toEqual([
      { firstInstance: 0, instanceCount: 250_000 },
    ]);
    expect(buildDrawSlices(2_500_000, 1_000_000)).toEqual([
      { firstInstance: 0, instanceCount: 1_000_000 },
      { firstInstance: 1_000_000, instanceCount: 1_000_000 },
      { firstInstance: 2_000_000, instanceCount: 500_000 },
    ]);
  });

  it("parses debug flags and safari draw limits from the query string", () => {
    Object.defineProperty(navigator, "userAgent", {
      value:
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 15_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
      configurable: true,
    });

    expect(parseScene3DDebugOptions("?scene3d_debug=1")).toEqual({
      verbose: true,
      maxInstancesPerDraw: DEFAULT_SAFARI_MAX_INSTANCES_PER_DRAW,
      userAgent: navigator.userAgent,
    });

    expect(
      parseScene3DDebugOptions(
        "?scene3d_debug=1&scene3d_max_instances_per_draw=250000",
      ),
    ).toEqual({
      verbose: true,
      maxInstancesPerDraw: 250_000,
      userAgent: navigator.userAgent,
    });
  });

  it("registers debug state on window and keeps the latest error snapshot", () => {
    const probe = createScene3DDebugProbe("scene-test", {
      verbose: true,
      maxInstancesPerDraw: 100,
      userAgent: "test-agent",
    });

    probe.snapshot("canvas", { width: 100, height: 80 });
    probe.record("render", "frame-start", { frame: 1 });
    probe.error("render-failed", new Error("boom"), { frame: 1 });

    const registry = (
      window as Window & {
        __COLIGHT_SCENE3D_DEBUG__?: {
          scenes: Record<string, { snapshots: Record<string, unknown> }>;
        };
      }
    ).__COLIGHT_SCENE3D_DEBUG__;

    expect(registry?.scenes["scene-test"]).toBeDefined();
    expect(registry?.scenes["scene-test"].snapshots.canvas).toEqual({
      width: 100,
      height: 80,
    });
    expect(registry?.scenes["scene-test"].snapshots.lastError).toMatchObject({
      label: "render-failed",
      frame: 1,
    });

    probe.dispose();
    expect(registry?.scenes["scene-test"]).toBeUndefined();
  });
});
