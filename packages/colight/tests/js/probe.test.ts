import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  PROBE_STAGES,
  probeRefresh,
  probeReset,
  probeEnabled,
  probeStage,
  probeStageAsync,
  probeRecord,
  probeCountWrite,
  probeFrame,
  probeStateChange,
  probeSnapshot,
  probeInstrumentDevice,
} from "../../src/js/probe";

function enable() {
  (window as any).__colightProbe = true;
  probeRefresh();
  probeReset();
}

function disable() {
  (window as any).__colightProbe = false;
  probeRefresh();
  probeReset();
}

describe("probe", () => {
  beforeEach(() => {
    disable();
  });

  describe("when off", () => {
    it("is disabled by default and records nothing", () => {
      expect(probeEnabled()).toBe(false);
      probeStage(PROBE_STAGES.evaluate, () => 1);
      probeRecord(PROBE_STAGES.render, 5);
      probeCountWrite(1024);
      probeFrame();
      const snapshot = probeSnapshot();
      expect(snapshot.stages).toEqual({});
      expect(snapshot.frames).toBe(0);
      expect(snapshot.writes.bytes).toEqual([]);
    });

    it("still returns the wrapped work's value", () => {
      expect(probeStage("x", () => 42)).toBe(42);
    });

    it("does not read the clock on the hot path", () => {
      // The off-path must not even call performance.now(): that is the whole
      // argument for leaving these calls in the render path.
      const now = vi.spyOn(performance, "now");
      probeStage("x", () => 1);
      probeCountWrite(10);
      probeFrame();
      expect(now).not.toHaveBeenCalled();
      now.mockRestore();
    });
  });

  describe("when on", () => {
    beforeEach(enable);

    it("records a duration per occurrence", () => {
      probeStage(PROBE_STAGES.compile, () => 1);
      probeStage(PROBE_STAGES.compile, () => 2);
      const entry = probeSnapshot().stages[PROBE_STAGES.compile];
      expect(entry.count).toBe(2);
      expect(entry.durations).toHaveLength(2);
      for (const d of entry.durations) expect(d).toBeGreaterThanOrEqual(0);
    });

    it("records even when the wrapped work throws", () => {
      expect(() =>
        probeStage(PROBE_STAGES.evaluate, () => {
          throw new Error("boom");
        }),
      ).toThrow("boom");
      expect(probeSnapshot().stages[PROBE_STAGES.evaluate].count).toBe(1);
    });

    it("times async stages", async () => {
      await probeStageAsync(PROBE_STAGES.render, async () => {
        await new Promise((r) => setTimeout(r, 5));
      });
      const entry = probeSnapshot().stages[PROBE_STAGES.render];
      expect(entry.count).toBe(1);
      expect(entry.durations[0]).toBeGreaterThan(0);
    });

    it("accumulates writes into per-frame samples", () => {
      probeCountWrite(100);
      probeCountWrite(200);
      probeFrame();
      probeCountWrite(50);
      probeFrame();
      const { writes } = probeSnapshot();
      expect(writes.calls).toEqual([2, 1]);
      expect(writes.bytes).toEqual([300, 50]);
    });

    it("records rAF-to-rAF intervals between frames, not before the first", () => {
      probeFrame();
      expect(probeSnapshot().frameIntervals).toHaveLength(0);
      probeFrame();
      probeFrame();
      const snapshot = probeSnapshot();
      expect(snapshot.frames).toBe(3);
      expect(snapshot.frameIntervals).toHaveLength(2);
    });

    it("closes the state-change span on the next frame", () => {
      probeStateChange();
      probeFrame();
      const total = probeSnapshot().stages[PROBE_STAGES.total];
      expect(total.count).toBe(1);
      expect(total.durations[0]).toBeGreaterThanOrEqual(0);
      // A frame with no preceding state change adds no `total` sample.
      probeFrame();
      expect(probeSnapshot().stages[PROBE_STAGES.total].count).toBe(1);
    });

    it("reset clears samples but keeps the probe enabled", () => {
      probeStage("x", () => 1);
      probeFrame();
      probeReset();
      const snapshot = probeSnapshot();
      expect(snapshot.enabled).toBe(true);
      expect(snapshot.stages).toEqual({});
      expect(snapshot.frames).toBe(0);
    });
  });

  describe("device instrumentation", () => {
    function fakeDevice() {
      const writeBuffer = vi.fn();
      return {
        device: { queue: { writeBuffer } } as unknown as GPUDevice,
        writeBuffer,
      };
    }

    it("counts bytes from a typed array's byteLength", () => {
      enable();
      const { device, writeBuffer } = fakeDevice();
      probeInstrumentDevice(device);
      device.queue.writeBuffer({} as GPUBuffer, 0, new Float32Array(10));
      probeFrame();
      expect(writeBuffer).toHaveBeenCalledOnce();
      expect(probeSnapshot().writes.bytes).toEqual([40]);
    });

    it("honours an explicit element size", () => {
      enable();
      const { device } = fakeDevice();
      probeInstrumentDevice(device);
      // size is in ELEMENTS for a typed array: 4 floats = 16 bytes.
      device.queue.writeBuffer({} as GPUBuffer, 0, new Float32Array(10), 0, 4);
      probeFrame();
      expect(probeSnapshot().writes.bytes).toEqual([16]);
    });

    it("forwards every argument to the real queue", () => {
      enable();
      const { device, writeBuffer } = fakeDevice();
      probeInstrumentDevice(device);
      const buffer = {} as GPUBuffer;
      const data = new Uint8Array(8);
      device.queue.writeBuffer(buffer, 16, data, 2, 4);
      expect(writeBuffer).toHaveBeenCalledWith(buffer, 16, data, 2, 4);
    });

    it("wraps a device only once", () => {
      enable();
      const { device } = fakeDevice();
      probeInstrumentDevice(device);
      const wrapped = device.queue.writeBuffer;
      probeInstrumentDevice(device);
      expect(device.queue.writeBuffer).toBe(wrapped);
    });

    it("forwards but records nothing while the probe is off", () => {
      disable();
      const { device, writeBuffer } = fakeDevice();
      probeInstrumentDevice(device);
      device.queue.writeBuffer({} as GPUBuffer, 0, new Float32Array(4));
      expect(writeBuffer).toHaveBeenCalledOnce();
      expect(probeSnapshot().writes.bytes).toEqual([]);
    });

    it("keeps a spy's mock surface intact through the wrap", () => {
      // The device is wrapped before the probe is switched on, so the wrap
      // must be invisible to a caller holding a spy — this is what keeps the
      // existing writeBuffer-counting tests passing.
      disable();
      const { device, writeBuffer } = fakeDevice();
      probeInstrumentDevice(device);
      const wrapped = device.queue.writeBuffer as unknown as typeof writeBuffer;
      expect(wrapped.mock).toBeDefined();
      device.queue.writeBuffer({} as GPUBuffer, 0, new Float32Array(4));
      expect(wrapped).toHaveBeenCalledOnce();
      expect(wrapped.mock.calls).toHaveLength(1);
    });
  });
});
