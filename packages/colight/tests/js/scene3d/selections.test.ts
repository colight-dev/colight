/**
 * Tests for named selections — the second consumer of the shared per-instance
 * mask abstraction. Selections resolve (via the SAME mask logic as filters) to
 * instance indices, become decorations, and carry membership metadata for
 * addressability. Verified through the real compileScene + selections helpers.
 */

import { describe, it, expect } from "vitest";
import { compileScene } from "../../../src/js/scene3d/compiler";
import {
  resolveSelectionInstances,
  selectionsForInstance,
  DEFAULT_SELECTION_STYLE,
  Selections,
} from "../../../src/js/scene3d/selections";

function cuboid(extra: Record<string, unknown> = {}) {
  return {
    type: "Cuboid" as const,
    centers: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
    half_size: 0.4,
    ...extra,
  };
}

describe("resolveSelectionInstances (shared mask logic)", () => {
  it("resolves an explicit instance list (in-range, unique, sorted)", () => {
    const idx = resolveSelectionInstances(
      { instances: [3, 1, 1, 99, -1] },
      undefined,
      4,
    );
    expect(idx).toEqual([1, 3]);
  });

  it("resolves a threshold predicate over inline values (NaN excluded)", () => {
    const idx = resolveSelectionInstances(
      { values: [0.1, 0.9, NaN, 0.5], min: 0.4 },
      undefined,
      4,
    );
    expect(idx).toEqual([1, 3]);
  });

  it("resolves a predicate over a component attribute via values_ref", () => {
    const comp = { CU_pct: [0.1, 0.9, 0.5] } as any;
    const idx = resolveSelectionInstances(
      { values_ref: "CU_pct", min: 0.5 },
      comp,
      3,
    );
    expect(idx).toEqual([1, 2]);
  });
});

describe("compileScene selection -> decoration (existing render machinery)", () => {
  const selections: Selections = {
    "sel-hi": { component: 0, source: { instances: [1, 3] }, style: "default" },
    "grade-hi": {
      component: 0,
      source: { values: [0.1, 0.9, 0.5, 0.2], min: 0.4 },
      style: { color: [0, 1, 0] },
    },
  };

  it("appends decorations to the target component and reports membership", () => {
    const { components, selections: reports } = compileScene(
      [cuboid()],
      undefined,
      selections,
    );
    const decos = (components[0] as any).decorations;
    expect(decos).toHaveLength(2);
    // Explicit selection uses the default highlight style.
    expect(decos[0].indexes).toEqual([1, 3]);
    expect(decos[0].color).toEqual(DEFAULT_SELECTION_STYLE.color);
    // Predicate selection uses its custom style and resolves values >= 0.4.
    expect(decos[1].indexes).toEqual([1, 2]);
    expect(decos[1].color).toEqual([0, 1, 0]);

    expect(reports).toEqual([
      {
        name: "sel-hi",
        component: 0,
        type: "Cuboid",
        count: 2,
        predicate: false,
        indexes: [1, 3],
      },
      {
        name: "grade-hi",
        component: 0,
        type: "Cuboid",
        count: 2,
        predicate: true,
        indexes: [1, 2],
      },
    ]);
  });

  it("reports selection membership for a picked instance", () => {
    const { selections: reports } = compileScene(
      [cuboid()],
      undefined,
      selections,
    );
    // Instance 1 is in both selections; instance 3 only in sel-hi; 0 in neither.
    expect(selectionsForInstance(reports, 0, 1)).toEqual([
      "sel-hi",
      "grade-hi",
    ]);
    expect(selectionsForInstance(reports, 0, 3)).toEqual(["sel-hi"]);
    expect(selectionsForInstance(reports, 0, 0)).toEqual([]);
  });

  it("produces no selections and no decorations when none are declared", () => {
    const { components, selections: reports } = compileScene([cuboid()]);
    expect(reports).toEqual([]);
    expect((components[0] as any).decorations).toBeUndefined();
  });
});
