import { describe, it, expect, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useStateWithDeps } from "../../src/js/hooks/useStateWithDeps.js";

describe("useStateWithDeps", () => {
  it("should initialize with initial state", () => {
    const { result } = renderHook(() => useStateWithDeps("initial", ["dep1"]));
    expect(result.current[0]).toBe("initial");
  });

  it("should reset state when dependencies change", () => {
    const { result, rerender } = renderHook(
      ({ deps }) => useStateWithDeps("initial", deps),
      { initialProps: { deps: ["dep1"] } },
    );

    // Initial state
    expect(result.current[0]).toBe("initial");

    // Manually set state to something different
    act(() => {
      result.current[1]("modified");
    });
    expect(result.current[0]).toBe("modified");

    // Change dependencies - should reset to initial state
    rerender({ deps: ["dep2"] });
    expect(result.current[0]).toBe("initial");
  });

  it("should not reset state when dependencies stay the same", () => {
    const { result, rerender } = renderHook(
      ({ deps }) => useStateWithDeps("initial", deps),
      { initialProps: { deps: ["dep1"] } },
    );

    // Set state to something different
    act(() => {
      result.current[1]("modified");
    });
    expect(result.current[0]).toBe("modified");

    // Re-render with same dependencies - should NOT reset
    rerender({ deps: ["dep1"] });
    expect(result.current[0]).toBe("modified");
  });

  it("should handle function as initial state", () => {
    const initialFn = vi.fn(() => ({ data: "test" }));
    const { result, rerender } = renderHook(
      ({ deps }) => useStateWithDeps(initialFn, deps),
      { initialProps: { deps: ["dep1"] } },
    );

    // Should call function and return correct result
    expect(result.current[0]).toEqual({ data: "test" });
    expect(initialFn).toHaveBeenCalled();

    const initialCallCount = initialFn.mock.calls.length;

    // Change dependencies - should call function again
    rerender({ deps: ["dep2"] });
    expect(result.current[0]).toEqual({ data: "test" });
    expect(initialFn).toHaveBeenCalledTimes(initialCallCount + 1);
  });

  it("should handle object dependencies correctly", () => {
    const obj1 = { id: 1 };
    const obj2 = { id: 2 };

    const { result, rerender } = renderHook(
      ({ deps }) => useStateWithDeps("initial", deps),
      { initialProps: { deps: [obj1] } },
    );

    // Set state to something different
    act(() => {
      result.current[1]("modified");
    });
    expect(result.current[0]).toBe("modified");

    // Change object reference - should reset
    rerender({ deps: [obj2] });
    expect(result.current[0]).toBe("initial");
  });

  it("should reset from complex state to simple state", () => {
    const { result, rerender } = renderHook(
      ({ deps }) => useStateWithDeps({}, deps),
      { initialProps: { deps: ["file1.py"] } },
    );

    // Set complex state
    act(() => {
      result.current[1]({
        block1: { content: "File 1 data" },
        block2: { content: "More data" },
      });
    });

    expect(Object.keys(result.current[0])).toHaveLength(2);

    // Change file - should reset to empty object
    rerender({ deps: ["file2.py"] });
    expect(result.current[0]).toEqual({});
    expect(Object.keys(result.current[0])).toHaveLength(0);
  });
});
