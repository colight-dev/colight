/**
 * Tests for the per-instance filter mask resolution — the shared mechanism
 * that both filtering (P1) and named selections (P2) build on.
 *
 * Verifies compileScene's resolveFilters: filterParams slot packing, per-
 * component _filterIndex / _filterValues, active-filter reporting, and that a
 * threshold-only change is detected (componentsEqualIgnoringFilter) so it can
 * take the light render path without re-uploading instance data.
 */

import { describe, it, expect } from "vitest";
import {
  compileScene,
  FLOATS_PER_FILTER,
} from "../../../src/js/scene3d/compiler";
import { componentsEqualIgnoringFilter } from "../../../src/js/scene3d/utils";
import { cuboidSpec } from "../../../src/js/scene3d/components";

function cuboid(extra: Record<string, unknown> = {}) {
  return {
    type: "Cuboid" as const,
    centers: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
    half_size: 0.4,
    ...extra,
  };
}

describe("resolveFilters (mask resolution core)", () => {
  it("packs slot 0 as inactive when no component is filtered", () => {
    const { filterParams, filters, components } = compileScene([cuboid()]);
    // Slot 0 only: [min, max, isActive, pad] all zero.
    expect(Array.from(filterParams)).toEqual([0, 0, 0, 0]);
    expect(filters).toEqual([]);
    expect((components[0] as any)._filterIndex).toBe(0);
  });

  it("assigns a unique slot with thresholds and _filterValues to a filtered component", () => {
    const values = [0.1, 0.5, 0.9];
    const { filterParams, filters, components } = compileScene([
      cuboid({ filter_by: { values, min: 0.3, max: 0.8, label: "grade" } }),
    ]);
    const comp = components[0] as any;
    // Slot 0 inactive, slot 1 carries this component's thresholds.
    expect(comp._filterIndex).toBe(1);
    expect(filterParams.length).toBe(2 * FLOATS_PER_FILTER);
    const slot = Array.from(filterParams.slice(FLOATS_PER_FILTER));
    expect(slot[0]).toBeCloseTo(0.3, 6); // min
    expect(slot[1]).toBeCloseTo(0.8, 6); // max
    expect(slot[2]).toBe(1); // isActive
    expect(slot[3]).toBe(0); // pad
    // values become the per-instance scalar attribute, coerced to float32.
    expect(comp._filterValues).toBeInstanceOf(Float32Array);
    expect(Array.from(comp._filterValues)).toEqual([
      expect.closeTo(0.1, 6),
      expect.closeTo(0.5, 6),
      expect.closeTo(0.9, 6),
    ]);
    // Reporting for inspect / screenshot --json.
    expect(filters).toEqual([
      { component: 0, type: "Cuboid", label: "grade", min: 0.3, max: 0.8 },
    ]);
  });

  it("treats missing min/max as unbounded (-Inf / +Inf)", () => {
    const { filterParams } = compileScene([
      cuboid({ filter_by: { values: [0, 1, 2], min: 0.5 } }),
    ]);
    const slot = Array.from(filterParams.slice(FLOATS_PER_FILTER));
    expect(slot[0]).toBe(0.5); // min
    expect(slot[1]).toBe(Infinity); // max unbounded
    expect(slot[2]).toBe(1); // active
  });

  it("gives each filtered component its own slot", () => {
    const { filterParams, components } = compileScene([
      cuboid({ filter_by: { values: [0, 1, 2], min: 0.1 } }),
      cuboid({ filter_by: { values: [0, 1, 2], min: 0.2 } }),
    ]);
    expect((components[0] as any)._filterIndex).toBe(1);
    expect((components[1] as any)._filterIndex).toBe(2);
    expect(filterParams.length).toBe(3 * FLOATS_PER_FILTER);
  });

  it("keeps filterValue/filterIndex in the render + picking instance layouts (collapse reaches both passes)", () => {
    // The injected framework attributes must be part of every instanced
    // primitive so the vertex-shader collapse applies uniformly to render and
    // picking. floatsPerInstance/Picking grew to hold them.
    // Cuboid: pos(3)+size(3)+quat(4)+color(3)+alpha(1)+transformIndex(1)
    //   +filterValue(1)+filterIndex(1) = 17 render floats.
    expect(cuboidSpec.floatsPerInstance).toBe(17);
    // Picking excludes color+alpha but keeps geometry + framework attrs + pickID.
    // pos(3)+size(3)+quat(4)+transformIndex(1)+filterValue(1)+filterIndex(1)+pickID(1)=14
    expect(cuboidSpec.floatsPerPicking).toBe(14);
  });
});

describe("componentsEqualIgnoringFilter (light render-path detection)", () => {
  it("returns true when only filter thresholds differ (no instance re-upload needed)", () => {
    const a = compileScene([
      cuboid({ filter_by: { values: [0, 1, 2], min: 0.2 } }),
    ]).components;
    const b = compileScene([
      cuboid({ filter_by: { values: [0, 1, 2], min: 0.7 } }),
    ]).components;
    expect(componentsEqualIgnoringFilter(a, b)).toBe(true);
  });

  it("returns false when the filter values (instance data) differ", () => {
    const a = compileScene([
      cuboid({ filter_by: { values: [0, 1, 2], min: 0.2 } }),
    ]).components;
    const b = compileScene([
      cuboid({ filter_by: { values: [9, 9, 9], min: 0.2 } }),
    ]).components;
    expect(componentsEqualIgnoringFilter(a, b)).toBe(false);
  });

  it("returns false when non-filter data differs", () => {
    const a = compileScene([cuboid({ color: [1, 0, 0] })]).components;
    const b = compileScene([cuboid({ color: [0, 1, 0] })]).components;
    expect(componentsEqualIgnoringFilter(a, b)).toBe(false);
  });
});
