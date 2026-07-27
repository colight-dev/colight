import { describe, expect, it } from "vitest";
import { ReadyStateManager } from "../../src/js/ready";

const tick = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe("ReadyStateManager quiescence (primary contract)", () => {
  it("resolves settled immediately when nothing is pending", async () => {
    const ready = new ReadyStateManager();
    await expect(ready.whenReady()).resolves.toEqual({ settled: true });
  });

  it("resolves settled once pending work drains", async () => {
    const ready = new ReadyStateManager();
    const done = ready.beginUpdate("render");
    const waiting = ready.whenReady();
    done();
    await expect(waiting).resolves.toEqual({ settled: true });
  });

  it("waits indefinitely for quiescence when no settle window is given", async () => {
    const ready = new ReadyStateManager();
    ready.beginUpdate("render");
    let resolved = false;
    void ready.whenReady().then(() => {
      resolved = true;
    });
    await tick(50);
    expect(resolved).toBe(false);
  });

  it("prefers quiescence over the window when the scene settles in time", async () => {
    const ready = new ReadyStateManager();
    const done = ready.beginUpdate("render");
    const waiting = ready.whenReady(1000);
    done();
    await expect(waiting).resolves.toEqual({ settled: true });
  });
});

describe("ReadyStateManager settle-window fallback", () => {
  it("resolves unsettled when a busy scene never drains", async () => {
    const ready = new ReadyStateManager();
    // A frame completes (page is alive) but new work keeps arriving faster
    // than it drains, so the counter never reaches zero.
    ready.beginUpdate("frame 1")();
    ready.beginUpdate("frame 2");
    ready.beginUpdate("frame 3");

    const result = await ready.whenReady(20);
    expect(result.settled).toBe(false);
    expect(result.reason).toMatch(/still animating after 20ms/);
    expect(result.reason).toMatch(/2 render updates still pending/);
  });

  it("never resolves when no frame has completed (a broken page)", async () => {
    const ready = new ReadyStateManager();
    ready.beginUpdate("render that never finishes");

    let settled: unknown = null;
    void ready.whenReady(10).then((r) => {
      settled = r;
    });
    await tick(60);
    // Left pending on purpose: the caller's hard timeout must surface this
    // as a failure rather than it being mislabeled as merely busy.
    expect(settled).toBeNull();
  });

  it("reports a completed frame only after work actually finishes", async () => {
    const ready = new ReadyStateManager();
    const done = ready.beginUpdate("render");
    expect(ready.hasCompletedFrame()).toBe(false);
    done();
    expect(ready.hasCompletedFrame()).toBe(true);
  });

  it("reset clears the completed-frame signal", () => {
    const ready = new ReadyStateManager();
    ready.beginUpdate("render")();
    ready.reset();
    expect(ready.hasCompletedFrame()).toBe(false);
    expect(ready.isReady()).toBe(true);
  });
});
