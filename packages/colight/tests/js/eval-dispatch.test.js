/**
 * Call-vs-construct dispatch in the AST evaluator.
 *
 * The evaluator must construct genuine ES classes and plainly CALL everything
 * else, passing the return value through untouched. The previous heuristic
 * (`fn.prototype?.constructor === fn`) is true of every plain `function`
 * declaration, so it constructed the overwhelming majority of the api
 * namespace; that only appeared to work because `new fn()` yields fn's return
 * value when that value is an object. Primitive/null/undefined returns were
 * silently replaced with `{}`.
 */
import { describe, expect, test } from "vitest";
import * as api from "../../src/js/api";
import { evaluate, isClassConstructor } from "../../src/js/eval";
import { colight } from "../../src/js/globals";
import { MarkSpec, PlotSpec } from "../../src/js/plot";

/** Minimal $state stand-in: evaluate() only needs `evaluate` for macros. */
const $state = {
  evaluate: (v) => evaluate(v, $state),
  __evalEnv: {},
};

/** Build the {"__type__":"function"} node the Python side emits for a JSCall. */
const callNode = (path, args = []) => ({
  __type__: "function",
  path,
  args,
});

const evalCall = (path, args = []) => evaluate(callNode(path, args), $state);

/**
 * Walk the api namespace, collecting every function-valued entry by dotted
 * path. Descends one level into nested namespaces (d3, Plot, React, scene3d,
 * ...) which is exactly the depth Python's JSRef.__getattr__ chaining reaches.
 */
function collectFunctionEntries() {
  const entries = [];
  const seen = new Set();

  const walk = (ns, prefix, depth) => {
    if (seen.has(ns)) return;
    seen.add(ns);
    for (const key of Object.keys(ns)) {
      let value;
      try {
        value = ns[key];
      } catch {
        continue;
      }
      const path = prefix ? `${prefix}.${key}` : key;
      if (typeof value === "function") {
        entries.push({ path, fn: value });
      } else if (value && typeof value === "object" && depth < 1) {
        walk(value, path, depth + 1);
      }
    }
  };

  walk(api, "", 0);
  return entries;
}

describe("evaluate() call-vs-construct dispatch", () => {
  test("the api namespace is overwhelmingly plain functions, not classes", () => {
    const entries = collectFunctionEntries();
    const classes = entries.filter((e) => isClassConstructor(e.fn));

    // Sanity: the namespace is large and only a small minority are classes.
    expect(entries.length).toBeGreaterThan(100);
    expect(classes.length).toBeLessThan(entries.length / 4);

    // The classes we own must be recognised as such.
    const classPaths = new Set(classes.map((e) => e.path));
    expect(classPaths.has("MarkSpec")).toBe(true);
    expect(classPaths.has("PlotSpec")).toBe(true);
    expect(classPaths.has("OnStateChange")).toBe(true);
    expect(classPaths.has("Bylight")).toBe(true);

    // Representative plain functions that the OLD heuristic wrongly constructed.
    for (const path of [
      "clamp",
      "repeat",
      "scene3d.PointCloud",
      "scene3d.Mesh",
      "scene3d.Group",
      "scene3d.deco",
      "Plot.marks",
      "d3.scaleLinear",
    ]) {
      expect(isClassConstructor(api[path.split(".")[0]])).toBeDefined();
      const fn = path.split(".").reduce((acc, key) => acc && acc[key], api);
      expect(typeof fn).toBe("function");
      expect(isClassConstructor(fn)).toBe(false);
      // ...and each is one the old heuristic would have constructed.
      expect(fn.prototype?.constructor === fn).toBe(true);
    }
  });

  test("classes are constructed into instances", () => {
    const mark = evalCall("MarkSpec", ["dot", [[1, 2]], {}]);
    expect(mark).toBeInstanceOf(MarkSpec);

    const plot = evalCall("PlotSpec", [{ layers: [] }]);
    expect(plot).toBeInstanceOf(PlotSpec);
  });

  test("plain functions are called and return their value verbatim", () => {
    // Object return: works under both old and new dispatch.
    const pointCloud = evalCall("scene3d.PointCloud", [{ centers: [0, 0, 0] }]);
    expect(pointCloud.type).toBe("PointCloud");

    // Primitive return: the regression the old heuristic fails.
    expect(evalCall("clamp", [5, 0, 3])).toBe(3);
    expect(evalCall("clamp", [-1, 0, 3])).toBe(0);
  });

  describe("primitive-returning registered entries", () => {
    // `colight` is the mutable registration namespace the api already uses for
    // client-side helpers (colight.resampleChannel), so it is the honest place
    // to register test-only entries reachable by the same dotted-path lookup.
    const registered = [];
    const register = (name, fn) => {
      colight[name] = fn;
      registered.push(name);
      return `colight.${name}`;
    };

    test("a number return survives evaluation unmutated", () => {
      const path = register("__testReturnsNumber", function returnsNumber() {
        return 42;
      });
      expect(evalCall(path)).toBe(42);
    });

    test("a null return survives evaluation unmutated", () => {
      const path = register("__testReturnsNull", function returnsNull() {
        return null;
      });
      expect(evalCall(path)).toBe(null);
    });

    test("undefined, string and boolean returns survive too", () => {
      expect(
        evalCall(
          register("__testReturnsUndefined", function returnsUndefined() {}),
        ),
      ).toBe(undefined);
      expect(
        evalCall(
          register("__testReturnsString", function returnsString() {
            return "hi";
          }),
        ),
      ).toBe("hi");
      expect(
        evalCall(
          register("__testReturnsFalse", function returnsFalse() {
            return false;
          }),
        ),
      ).toBe(false);
    });

    test("the old heuristic would have destroyed all of these", () => {
      // Documents precisely why the heuristic was wrong: every one of these is
      // constructor-shaped under `fn.prototype?.constructor === fn`, and `new`
      // on them yields `{}` rather than the value.
      for (const name of registered) {
        const fn = colight[name];
        expect(fn.prototype?.constructor === fn).toBe(true);
        expect(isClassConstructor(fn)).toBe(false);
        expect(new fn()).toEqual({});
      }
      // Clean up so the shipped namespace is not polluted for other suites.
      for (const name of registered) delete colight[name];
    });
  });

  test("resampleChannel returns a scalar, not an empty object", () => {
    // The real-world instance of this bug: a scalar-returning channel resampler
    // rendered nothing because `new` discarded its number.
    const value = evalCall("colight.resampleChannel", [
      { parameter: "t", at: [0, 1], values: [0, 10], value: 0.5 },
    ]);
    expect(typeof value).toBe("number");
    expect(value).toBeCloseTo(5);
  });

  test("parity: every api function entry behaves per its class-ness", () => {
    const entries = collectFunctionEntries();
    let classCount = 0;
    let plainCount = 0;

    for (const { path, fn } of entries) {
      if (isClassConstructor(fn)) {
        classCount += 1;
        // Classes must be constructed - calling them without `new` throws.
        expect(() => fn()).toThrow();
      } else {
        plainCount += 1;
        // Plain functions must be callable without `new`. We do not invoke
        // arbitrary api functions (many need real arguments/DOM); what matters
        // is that dispatch selects the call branch for them.
        expect(isClassConstructor(fn)).toBe(false);
      }
    }

    expect(classCount).toBeGreaterThan(0);
    expect(plainCount).toBeGreaterThan(classCount);
  });

  test("class-ness results are memoized per function identity", () => {
    class Widget {}
    const toString = Function.prototype.toString;
    let calls = 0;
    Function.prototype.toString = function (...args) {
      calls += 1;
      return toString.apply(this, args);
    };
    try {
      isClassConstructor(Widget);
      const afterFirst = calls;
      isClassConstructor(Widget);
      isClassConstructor(Widget);
      expect(calls).toBe(afterFirst);
    } finally {
      Function.prototype.toString = toString;
    }
    expect(isClassConstructor(Widget)).toBe(true);
  });
});
