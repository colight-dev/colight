/**
 * The client-side resampler behind `Plot.channel(...)`.
 *
 * Sample rows arrive exactly as the AST evaluator decodes them: a 1-D
 * declaration is a flat typed array, a 2-D one an array of per-row typed
 * arrays (see `evaluateNdarray`). The tests feed both shapes.
 */
import { describe, it, expect } from "vitest";
import { resampleChannel } from "../../src/js/channels";
import type { ChannelConfig } from "../../src/js/channels";
import * as api from "../../src/js/api";
import { evaluate } from "../../src/js/eval";

/** A (N, k) declaration as it reaches JS: rows as separate typed arrays. */
function rows(values: number[][]): Float32Array[] {
  return values.map((r) => new Float32Array(r));
}

function make(
  overrides: Partial<ChannelConfig> & Pick<ChannelConfig, "at" | "values">,
): ChannelConfig {
  return {
    parameter: "t",
    value: 0,
    rule: "linear",
    ...overrides,
  } as ChannelConfig;
}

describe("resampleChannel", () => {
  const AT = new Float64Array([0, 1, 2]);
  const SCALARS = new Float32Array([10, 20, 40]);
  const VEC3 = rows([
    [0, 0, 0],
    [1, 2, 3],
    [3, 6, 9],
  ]);

  describe("linear", () => {
    it("lerps scalars between the bracketing samples", () => {
      const at = { at: AT, values: SCALARS, rule: "linear" as const };
      expect(resampleChannel(make({ ...at, value: 0 }))).toBe(10);
      expect(resampleChannel(make({ ...at, value: 0.5 }))).toBe(15);
      expect(resampleChannel(make({ ...at, value: 1 }))).toBe(20);
      expect(resampleChannel(make({ ...at, value: 1.25 }))).toBe(25);
      expect(resampleChannel(make({ ...at, value: 2 }))).toBe(40);
    });

    it("lerps (N, 3) rows elementwise", () => {
      const out = resampleChannel(
        make({ at: AT, values: VEC3, rule: "linear", value: 1.5 }),
      );
      expect(Array.from(out as number[])).toEqual([2, 4, 6]);
    });
  });

  describe("nearest", () => {
    it("picks the closer sample, ties going to the upper one", () => {
      const base = { at: AT, values: SCALARS, rule: "nearest" as const };
      expect(resampleChannel(make({ ...base, value: 0.4 }))).toBe(10);
      expect(resampleChannel(make({ ...base, value: 0.5 }))).toBe(20);
      expect(resampleChannel(make({ ...base, value: 0.6 }))).toBe(20);
      expect(resampleChannel(make({ ...base, value: 1.9 }))).toBe(40);
    });

    it("returns whole (N, 3) rows without blending them", () => {
      const out = resampleChannel(
        make({ at: AT, values: VEC3, rule: "nearest", value: 1.6 }),
      );
      expect(Array.from(out as number[])).toEqual([3, 6, 9]);
    });
  });

  describe("step", () => {
    it("holds the sample at or below the parameter", () => {
      const base = { at: AT, values: SCALARS, rule: "step" as const };
      expect(resampleChannel(make({ ...base, value: 0.0 }))).toBe(10);
      expect(resampleChannel(make({ ...base, value: 0.99 }))).toBe(10);
      expect(resampleChannel(make({ ...base, value: 1.0 }))).toBe(20);
      expect(resampleChannel(make({ ...base, value: 1.99 }))).toBe(20);
      expect(resampleChannel(make({ ...base, value: 2.0 }))).toBe(40);
    });

    it("holds whole (N, 3) rows", () => {
      const out = resampleChannel(
        make({ at: AT, values: VEC3, rule: "step", value: 1.9 }),
      );
      expect(Array.from(out as number[])).toEqual([1, 2, 3]);
    });
  });

  describe("qlerp", () => {
    /** Rotation about +X by `deg`, as the xyzw quaternion Group takes. */
    function aboutX(deg: number): number[] {
      const half = (deg * Math.PI) / 360;
      return [Math.sin(half), 0, 0, Math.cos(half)];
    }

    it("takes the short arc through an antipodal pair", () => {
      // q and -q are the same rotation; a naive lerp between them collapses
      // to zero at the midpoint instead of holding the rotation.
      const q = aboutX(60);
      const antipodal = q.map((c) => -c);
      const out = resampleChannel(
        make({
          at: new Float64Array([0, 1]),
          values: rows([q, antipodal]),
          rule: "qlerp",
          value: 0.5,
        }),
      ) as number[];

      // Same rotation at both ends, so every intermediate value is it too
      // (up to sign, which is the same rotation).
      const sign = out[3] < 0 ? -1 : 1;
      expect(out.map((c) => c * sign)).toEqual(
        q.map((c) => expect.closeTo(c, 6) as unknown as number),
      );
    });

    it("normalizes the result", () => {
      const out = resampleChannel(
        make({
          at: new Float64Array([-80, 80]),
          values: rows([aboutX(-80), aboutX(80)]),
          rule: "qlerp",
          value: 17,
        }),
      ) as number[];
      expect(Math.hypot(...out)).toBeCloseTo(1, 12);
      expect(out).toHaveLength(4);
    });

    it("interpolates monotonically across a signed sweep", () => {
      // 9 samples across [-80, 80] is the tentacle fixture's declaration.
      const at = new Float64Array(
        Array.from({ length: 9 }, (_, i) => -80 + (i * 160) / 8),
      );
      const values = rows(
        Array.from({ length: 9 }, (_, i) => aboutX(-80 + (i * 160) / 8)),
      );
      const angleAt = (v: number) => {
        const q = resampleChannel(
          make({ at, values, rule: "qlerp", value: v }),
        ) as number[];
        return (2 * Math.atan2(q[0], q[3]) * 180) / Math.PI;
      };
      let previous = -Infinity;
      for (let v = -80; v <= 80; v += 5) {
        const a = angleAt(v);
        expect(a).toBeGreaterThan(previous);
        // Samples 20 deg apart: normalized lerp tracks true slerp to within
        // 0.01 deg across the whole sweep, so the declared arc is faithful.
        // (At 2 samples — the endpoints alone — the error is 5.6 deg.)
        expect(Math.abs(a - v)).toBeLessThan(0.02);
        previous = a;
      }
    });
  });

  describe("domain edges", () => {
    it("clamps below the first and above the last sample", () => {
      const base = { at: AT, values: SCALARS, rule: "linear" as const };
      expect(resampleChannel(make({ ...base, value: -100 }))).toBe(10);
      expect(resampleChannel(make({ ...base, value: 1000 }))).toBe(40);
    });

    it("returns the single row for a one-sample channel", () => {
      const single = {
        at: new Float64Array([5]),
        values: rows([[7, 8, 9]]),
      };
      expect(
        Array.from(
          resampleChannel(
            make({ ...single, rule: "linear", value: -3 }),
          ) as number[],
        ),
      ).toEqual([7, 8, 9]);
      expect(
        Array.from(
          resampleChannel(
            make({ ...single, rule: "qlerp", value: 99 }),
          ) as number[],
        ),
      ).toEqual([7, 8, 9]);
      expect(
        resampleChannel(
          make({
            at: new Float64Array([5]),
            values: new Float32Array([42]),
            rule: "linear",
            value: 0,
          }),
        ),
      ).toBe(42);
    });

    it("reads a coincident pair as a jump rather than dividing by zero", () => {
      // A duplicate coordinate declares a discontinuity: the zero-width
      // interval must yield the value in force after the jump, finite.
      const at = new Float64Array([0, 1, 1, 2]);
      const values = new Float32Array([0, 10, 20, 30]);
      const at1 = resampleChannel(
        make({ at, values, rule: "linear", value: 1 }),
      );
      expect(Number.isFinite(at1 as number)).toBe(true);
      expect(at1).toBe(20);
      // Either side of the jump still interpolates normally.
      expect(
        resampleChannel(make({ at, values, rule: "linear", value: 0.5 })),
      ).toBe(5);
      expect(
        resampleChannel(make({ at, values, rule: "linear", value: 1.5 })),
      ).toBe(25);
    });
  });

  describe("as the AST evaluator invokes it", () => {
    it("is reachable at the path Plot.channel serializes", () => {
      const fn = "colight.resampleChannel"
        .split(".")
        .reduce((acc: any, key) => acc[key], api as any);
      expect(fn).toBe(resampleChannel);
    });

    it("returns a scalar rather than a constructed object", async () => {
      // `evaluate` calls a resolved reference with `new` when it looks
      // constructor-shaped (that is how MarkSpec/PlotSpec are built), and
      // `new` on a function returning a number discards the number. A scalar
      // channel driving e.g. a mark's `r` must survive that path.
      const { createStateStore } = await import("../../src/js/widget");
      const $state = await createStateStore({ state: { size: 0.5 } });

      const node = {
        __type__: "function",
        path: "colight.resampleChannel",
        args: [
          {
            parameter: "size",
            value: {
              __type__: "js_source",
              value: '$state["size"]',
              params: [],
              expression: true,
              scope: {},
            },
            at: new Float64Array([0, 0.5, 1]),
            values: new Float64Array([2, 8, 30]),
            rule: "linear",
          },
        ],
      };

      expect(evaluate(node, $state, undefined, [])).toBe(8);
    });
  });

  describe("wide rows", () => {
    // A pose table: each row is one flattened set of vertex positions.
    const WIDE = rows([
      Array.from({ length: 12 }, (_, i) => i),
      Array.from({ length: 12 }, (_, i) => i + 100),
    ]);

    it("returns a fresh Float32Array each call", () => {
      const config = make({
        at: new Float64Array([0, 1]),
        values: WIDE,
        rule: "linear",
        value: 0.25,
      });
      const first = resampleChannel(config);
      const second = resampleChannel(config);

      expect(first).toBeInstanceOf(Float32Array);
      expect(second).toBeInstanceOf(Float32Array);
      // Fresh identity is what the geometry contents-change contract keys on.
      expect(first).not.toBe(second);
      expect(first).not.toBe(WIDE[0]);
      expect(Array.from(first as Float32Array)).toEqual(
        Array.from(second as Float32Array),
      );
      expect(Array.from(first as Float32Array)).toEqual(
        Array.from({ length: 12 }, (_, i) => i + 25),
      );
    });

    it("returns a fresh row for step and nearest too, never the sample", () => {
      for (const rule of ["step", "nearest"] as const) {
        const out = resampleChannel(
          make({
            at: new Float64Array([0, 1]),
            values: WIDE,
            rule,
            value: 0.1,
          }),
        );
        expect(out).not.toBe(WIDE[0]);
        expect(Array.from(out as Float32Array)).toEqual(Array.from(WIDE[0]));
      }
    });
  });
});
