/**
 * Unit tests for per-instance line radii (sizes).
 *
 * LineSegments expand user data per-segment, so a per-segment `sizes` array
 * must land distinct radii in both the render and picking buffers, indexed the
 * same way as per-segment `colors`. LineBeams `sizes` are per-line: every
 * segment of a line shares its line's radius.
 *
 * The size attribute layout for both beam primitives is:
 *   start(3) end(3) size(1) color(3) alpha(1)  -> size at float offset 6.
 * The picking buffer keeps geometry only (no color/alpha):
 *   start(3) end(3) size(1) ... transform/filter ... pickID
 * so size is likewise at picking offset 6.
 */

import { describe, it, expect } from "vitest";
import {
  buildRenderData,
  buildPickingData,
  lineSegmentsSpec,
  lineBeamsSpec,
  LineSegmentsComponentConfig,
  LineBeamsComponentConfig,
} from "../../../src/js/scene3d/components";

const SIZE_OFFSET = 6;

describe("LineSegments per-segment sizes", () => {
  const twoSegments = (
    sizes?: Float32Array,
    extra: Record<string, unknown> = {},
  ): LineSegmentsComponentConfig =>
    ({
      type: "LineSegments",
      starts: new Float32Array([0, 0, 0, 1, 0, 0]),
      ends: new Float32Array([1, 0, 0, 2, 0, 0]),
      sizes,
      ...extra,
    }) as unknown as LineSegmentsComponentConfig;

  it("writes a distinct radius per segment into the render buffer", () => {
    const cfg = twoSegments(new Float32Array([0.05, 0.5]));
    const buf = new Float32Array(lineSegmentsSpec.floatsPerInstance * 2);
    buildRenderData(cfg, lineSegmentsSpec, buf, 0);

    const thin = buf[SIZE_OFFSET];
    const thick = buf[lineSegmentsSpec.floatsPerInstance + SIZE_OFFSET];
    expect(thin).toBeCloseTo(0.05);
    expect(thick).toBeCloseTo(0.5);
    // Thick vs thin segments differ (the coverage/pick distinction geologists
    // rely on for structural-intensity thickness).
    expect(thick).toBeGreaterThan(thin * 5);
  });

  it("writes the same distinct radii into the picking buffer", () => {
    const cfg = twoSegments(new Float32Array([0.05, 0.5]));
    const buf = new Float32Array(lineSegmentsSpec.floatsPerPicking * 2);
    buildPickingData(cfg, lineSegmentsSpec, buf, 0, 0);

    const thin = buf[SIZE_OFFSET];
    const thick = buf[lineSegmentsSpec.floatsPerPicking + SIZE_OFFSET];
    expect(thin).toBeCloseTo(0.05);
    expect(thick).toBeCloseTo(0.5);
  });

  it("expands sizes identically to per-segment colors (same index)", () => {
    // Segment 1 is both the thick one and the green one: sizes and colors must
    // agree on which element index they style.
    const cfg = twoSegments(new Float32Array([0.05, 0.5]), {
      colors: new Float32Array([1, 0, 0, 0, 1, 0]),
    });
    const buf = new Float32Array(lineSegmentsSpec.floatsPerInstance * 2);
    buildRenderData(cfg, lineSegmentsSpec, buf, 0);

    const stride = lineSegmentsSpec.floatsPerInstance;
    const colorOffset = lineSegmentsSpec.colorOffset;
    // Segment 0: thin + red
    expect(buf[SIZE_OFFSET]).toBeCloseTo(0.05);
    expect([
      buf[colorOffset],
      buf[colorOffset + 1],
      buf[colorOffset + 2],
    ]).toEqual([1, 0, 0]);
    // Segment 1: thick + green
    expect(buf[stride + SIZE_OFFSET]).toBeCloseTo(0.5);
    expect([
      buf[stride + colorOffset],
      buf[stride + colorOffset + 1],
      buf[stride + colorOffset + 2],
    ]).toEqual([0, 1, 0]);
  });

  it("falls back to the scalar size for every segment", () => {
    const cfg = twoSegments(undefined, { size: 0.3 });
    const buf = new Float32Array(lineSegmentsSpec.floatsPerInstance * 2);
    buildRenderData(cfg, lineSegmentsSpec, buf, 0);
    expect(buf[SIZE_OFFSET]).toBeCloseTo(0.3);
    expect(buf[lineSegmentsSpec.floatsPerInstance + SIZE_OFFSET]).toBeCloseTo(
      0.3,
    );
  });
});

describe("LineBeams per-line sizes", () => {
  it("shares one radius across every segment of a line", () => {
    // Two lines (index 0 and 1), one segment each; sizes are per line index.
    const cfg: LineBeamsComponentConfig = {
      type: "LineBeams",
      points: new Float32Array([
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0, // line 0, one segment
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1, // line 1, one segment
      ]),
      sizes: new Float32Array([0.05, 0.5]),
    } as unknown as LineBeamsComponentConfig;

    const buf = new Float32Array(lineBeamsSpec.floatsPerInstance * 2);
    buildRenderData(cfg, lineBeamsSpec, buf, 0);
    expect(buf[SIZE_OFFSET]).toBeCloseTo(0.05);
    expect(buf[lineBeamsSpec.floatsPerInstance + SIZE_OFFSET]).toBeCloseTo(0.5);
  });
});
