/**
 * @module gpu-lifecycle.test
 * @description Tests for GPU resource lifecycle management.
 */

import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  GPULifecycleManager,
  createVersionedResource,
  replaceVersionedResource,
  FrameResources,
  validateFrameResources,
  resetDefaultLifecycleManager,
} from "../../src/js/scene3d/gpu-lifecycle";

// =============================================================================
// Mock Resources
// =============================================================================

function createMockBuffer(label = "test-buffer"): {
  destroy: ReturnType<typeof vi.fn>;
  label: string;
} {
  return {
    destroy: vi.fn(),
    label,
  };
}

// =============================================================================
// GPULifecycleManager Tests
// =============================================================================

describe("GPULifecycleManager", () => {
  let lifecycle: GPULifecycleManager;

  beforeEach(() => {
    lifecycle = new GPULifecycleManager();
    resetDefaultLifecycleManager();
  });

  describe("beginFrame", () => {
    it("increments frame version", () => {
      expect(lifecycle.currentFrame).toBe(0);
      lifecycle.beginFrame();
      expect(lifecycle.currentFrame).toBe(1);
      lifecycle.beginFrame();
      expect(lifecycle.currentFrame).toBe(2);
    });

    it("returns the new frame version", () => {
      expect(lifecycle.beginFrame()).toBe(1);
      expect(lifecycle.beginFrame()).toBe(2);
    });
  });

  describe("scheduleDestroy", () => {
    it("tracks resources for destruction", () => {
      const buffer = createMockBuffer();

      lifecycle.scheduleDestroy(buffer);

      expect(lifecycle.pendingCount).toBe(1);
      expect(buffer.destroy).not.toHaveBeenCalled();
    });

    it("can queue multiple resources", () => {
      const buffer1 = createMockBuffer("buffer1");
      const buffer2 = createMockBuffer("buffer2");

      lifecycle.scheduleDestroy(buffer1);
      lifecycle.scheduleDestroy(buffer2);

      expect(lifecycle.pendingCount).toBe(2);
    });
  });

  describe("processDestructions", () => {
    it("destroys resources scheduled in previous frames", () => {
      const buffer = createMockBuffer();

      lifecycle.beginFrame(); // frame 1
      lifecycle.scheduleDestroy(buffer);
      lifecycle.beginFrame(); // frame 2

      lifecycle.processDestructions();

      expect(buffer.destroy).toHaveBeenCalledTimes(1);
      expect(lifecycle.pendingCount).toBe(0);
    });

    it("does not destroy resources scheduled in current frame", () => {
      const buffer = createMockBuffer();

      lifecycle.beginFrame(); // frame 1
      lifecycle.scheduleDestroy(buffer);
      // Note: no beginFrame() before processDestructions

      lifecycle.processDestructions();

      expect(buffer.destroy).not.toHaveBeenCalled();
      expect(lifecycle.pendingCount).toBe(1);
    });

    it("destroys resources scheduled at frame 0 after first frame", () => {
      const buffer = createMockBuffer();

      // Schedule at frame 0
      lifecycle.scheduleDestroy(buffer);
      lifecycle.beginFrame(); // frame 1

      lifecycle.processDestructions();

      expect(buffer.destroy).toHaveBeenCalledTimes(1);
    });

    it("handles empty pending list gracefully", () => {
      expect(() => lifecycle.processDestructions()).not.toThrow();
    });

    it("handles destruction errors gracefully", () => {
      const buffer = createMockBuffer();
      buffer.destroy.mockImplementation(() => {
        throw new Error("Already destroyed");
      });

      lifecycle.scheduleDestroy(buffer);
      lifecycle.beginFrame();

      // Should not throw
      expect(() => lifecycle.processDestructions()).not.toThrow();
    });
  });

  describe("destroyAll", () => {
    it("destroys all pending resources immediately", () => {
      const buffer1 = createMockBuffer("buffer1");
      const buffer2 = createMockBuffer("buffer2");

      lifecycle.scheduleDestroy(buffer1);
      lifecycle.scheduleDestroy(buffer2);

      lifecycle.destroyAll();

      expect(buffer1.destroy).toHaveBeenCalledTimes(1);
      expect(buffer2.destroy).toHaveBeenCalledTimes(1);
      expect(lifecycle.pendingCount).toBe(0);
    });

    it("ignores destruction errors", () => {
      const buffer = createMockBuffer();
      buffer.destroy.mockImplementation(() => {
        throw new Error("Already destroyed");
      });

      lifecycle.scheduleDestroy(buffer);

      expect(() => lifecycle.destroyAll()).not.toThrow();
    });
  });
});

// =============================================================================
// Versioned Resource Tests
// =============================================================================

describe("Versioned Resources", () => {
  let lifecycle: GPULifecycleManager;

  beforeEach(() => {
    lifecycle = new GPULifecycleManager();
  });

  describe("createVersionedResource", () => {
    it("creates a versioned wrapper with default version 1", () => {
      const buffer = createMockBuffer();
      const versioned = createVersionedResource(buffer);

      expect(versioned.resource).toBe(buffer);
      expect(versioned.version).toBe(1);
    });

    it("accepts custom initial version", () => {
      const buffer = createMockBuffer();
      const versioned = createVersionedResource(buffer, 5);

      expect(versioned.version).toBe(5);
    });
  });

  describe("replaceVersionedResource", () => {
    it("schedules old resource for destruction", () => {
      const oldBuffer = createMockBuffer("old");
      const newBuffer = createMockBuffer("new");
      const current = createVersionedResource(oldBuffer);

      replaceVersionedResource(current, newBuffer, lifecycle);

      expect(lifecycle.pendingCount).toBe(1);
    });

    it("returns new versioned resource with incremented version", () => {
      const oldBuffer = createMockBuffer("old");
      const newBuffer = createMockBuffer("new");
      const current = createVersionedResource(oldBuffer, 3);

      const updated = replaceVersionedResource(current, newBuffer, lifecycle);

      expect(updated.resource).toBe(newBuffer);
      expect(updated.version).toBe(4);
    });

    it("does not destroy old resource immediately", () => {
      const oldBuffer = createMockBuffer("old");
      const newBuffer = createMockBuffer("new");
      const current = createVersionedResource(oldBuffer);

      replaceVersionedResource(current, newBuffer, lifecycle);

      expect(oldBuffer.destroy).not.toHaveBeenCalled();
    });
  });
});

// =============================================================================
// Frame Resources Validation Tests
// =============================================================================

describe("Frame Resources Validation", () => {
  it("does not throw for current frame resources", () => {
    const resources: FrameResources = {
      frameVersion: 5,
      uniformBindGroup: {} as GPUBindGroup,
      transformsVersion: 1,
    };

    expect(() => validateFrameResources(resources, 5)).not.toThrow();
  });

  it("logs error for stale frame resources in development", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    const resources: FrameResources = {
      frameVersion: 3,
      uniformBindGroup: {} as GPUBindGroup,
      transformsVersion: 1,
    };

    validateFrameResources(resources, 5);

    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining("Stale frame resources"),
    );

    consoleSpy.mockRestore();
  });
});

// =============================================================================
// Integration-style Tests
// =============================================================================

describe("Lifecycle Integration Patterns", () => {
  it("supports typical render loop pattern", async () => {
    const lifecycle = new GPULifecycleManager();
    const destroyedBuffers: string[] = [];

    // Frame 1: Create initial buffer
    lifecycle.beginFrame();
    const buffer1 = {
      destroy: () => destroyedBuffers.push("buffer1"),
      label: "buffer1",
    };

    // Frame 2: Replace buffer (buffer1 gets scheduled for destruction)
    lifecycle.beginFrame();
    lifecycle.scheduleDestroy(buffer1);
    const buffer2 = {
      destroy: () => destroyedBuffers.push("buffer2"),
      label: "buffer2",
    };

    // End of frame 2: GPU work completes, but buffer1 was scheduled this frame
    // so it's not destroyed yet (GPU might still be using it)
    lifecycle.processDestructions();
    expect(destroyedBuffers).toEqual([]);

    // Frame 3: Now buffer1 can be safely destroyed
    lifecycle.beginFrame();
    lifecycle.processDestructions();
    expect(destroyedBuffers).toEqual(["buffer1"]);

    // Replace buffer2
    lifecycle.scheduleDestroy(buffer2);

    // Frame 4: buffer2 can be destroyed
    lifecycle.beginFrame();
    lifecycle.processDestructions();
    expect(destroyedBuffers).toEqual(["buffer1", "buffer2"]);
  });

  it("handles rapid buffer resizing correctly", () => {
    const lifecycle = new GPULifecycleManager();
    const buffers = [
      createMockBuffer("b1"),
      createMockBuffer("b2"),
      createMockBuffer("b3"),
    ];

    lifecycle.beginFrame();

    // Multiple resizes in same frame
    lifecycle.scheduleDestroy(buffers[0]);
    lifecycle.scheduleDestroy(buffers[1]);
    lifecycle.scheduleDestroy(buffers[2]);

    // All scheduled in same frame - none destroyed yet
    lifecycle.processDestructions();
    expect(buffers[0].destroy).not.toHaveBeenCalled();
    expect(buffers[1].destroy).not.toHaveBeenCalled();
    expect(buffers[2].destroy).not.toHaveBeenCalled();

    // Next frame - all get destroyed
    lifecycle.beginFrame();
    lifecycle.processDestructions();
    expect(buffers[0].destroy).toHaveBeenCalled();
    expect(buffers[1].destroy).toHaveBeenCalled();
    expect(buffers[2].destroy).toHaveBeenCalled();
  });
});
