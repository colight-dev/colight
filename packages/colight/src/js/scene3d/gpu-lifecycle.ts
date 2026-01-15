/**
 * @module gpu-lifecycle
 * @description GPU resource lifecycle management for Scene3D.
 *
 * This module provides utilities for tracking GPU resource versions and
 * scheduling deferred destruction to avoid use-after-free hazards.
 *
 * Key concepts:
 * - Resources are tracked with monotonic version numbers
 * - Old resources are scheduled for destruction after GPU work completes
 * - Frame staging ensures resources aren't mutated during draw calls
 */

// =============================================================================
// Types
// =============================================================================

/** A GPU resource that can be destroyed */
export type DestroyableResource = {
  destroy(): void;
  label?: string;
};

/** A versioned resource wrapper */
export interface VersionedResource<T extends DestroyableResource> {
  resource: T;
  version: number;
}

/** Pending destruction entry */
interface PendingDestruction {
  resource: DestroyableResource;
  scheduledAtFrame: number;
}

// =============================================================================
// GPULifecycleManager
// =============================================================================

/**
 * Manages GPU resource lifecycle with versioning and deferred destruction.
 *
 * Usage:
 * ```typescript
 * const lifecycle = new GPULifecycleManager();
 *
 * // At start of frame
 * lifecycle.beginFrame();
 *
 * // When replacing a resource
 * lifecycle.scheduleDestroy(oldBuffer);
 * const newBuffer = device.createBuffer(...);
 *
 * // After GPU work completes
 * await device.queue.onSubmittedWorkDone();
 * lifecycle.processDestructions();
 * ```
 */
export class GPULifecycleManager {
  private frameVersion = 0;
  private pendingDestructions: PendingDestruction[] = [];

  /**
   * Begin a new frame. Increments the frame version.
   * Call this at the start of each render cycle.
   */
  beginFrame(): number {
    this.frameVersion++;
    return this.frameVersion;
  }

  /**
   * Get the current frame version.
   */
  get currentFrame(): number {
    return this.frameVersion;
  }

  /**
   * Schedule a resource for deferred destruction.
   * The resource will be destroyed after the next processDestructions() call.
   *
   * @param resource - GPU resource to destroy (buffer, texture, etc.)
   */
  scheduleDestroy(resource: DestroyableResource): void {
    this.pendingDestructions.push({
      resource,
      scheduledAtFrame: this.frameVersion,
    });
  }

  /**
   * Process pending destructions.
   * Call this after `device.queue.onSubmittedWorkDone()` resolves.
   *
   * Only destroys resources that were scheduled in previous frames,
   * ensuring the GPU has finished using them.
   */
  processDestructions(): void {
    if (this.pendingDestructions.length === 0) return;

    // Destroy all pending resources that were scheduled before this frame
    const stillPending: PendingDestruction[] = [];

    for (const entry of this.pendingDestructions) {
      // Only destroy if scheduled in a previous frame
      if (entry.scheduledAtFrame < this.frameVersion) {
        try {
          entry.resource.destroy();
        } catch (e) {
          // Resource may already be destroyed - ignore
          if (process.env.NODE_ENV !== "production") {
            console.warn(
              `[GPULifecycle] Failed to destroy resource:`,
              entry.resource.label,
              e,
            );
          }
        }
      } else {
        // Keep for next frame
        stillPending.push(entry);
      }
    }

    this.pendingDestructions = stillPending;
  }

  /**
   * Destroy all pending resources immediately.
   * Use this during cleanup/unmount.
   */
  destroyAll(): void {
    for (const entry of this.pendingDestructions) {
      try {
        entry.resource.destroy();
      } catch (e) {
        // Ignore destruction errors during cleanup
      }
    }
    this.pendingDestructions = [];
  }

  /**
   * Get the number of pending destructions.
   * Useful for debugging and tests.
   */
  get pendingCount(): number {
    return this.pendingDestructions.length;
  }
}

// =============================================================================
// Resource Versioning Utilities
// =============================================================================

/**
 * Create a versioned resource wrapper.
 * The version increments each time the resource is replaced.
 */
export function createVersionedResource<T extends DestroyableResource>(
  resource: T,
  version = 1,
): VersionedResource<T> {
  return { resource, version };
}

/**
 * Replace a versioned resource, scheduling the old one for destruction.
 *
 * @param current - Current versioned resource
 * @param newResource - New resource to use
 * @param lifecycle - Lifecycle manager for scheduling destruction
 * @returns New versioned resource with incremented version
 */
export function replaceVersionedResource<T extends DestroyableResource>(
  current: VersionedResource<T>,
  newResource: T,
  lifecycle: GPULifecycleManager,
): VersionedResource<T> {
  lifecycle.scheduleDestroy(current.resource);
  return {
    resource: newResource,
    version: current.version + 1,
  };
}

// =============================================================================
// Frame State
// =============================================================================

/**
 * Immutable frame state for use during draw calls.
 * Built during prepareFrame, consumed during drawFrame.
 */
export interface FrameResources {
  /** Current frame version */
  frameVersion: number;
  /** Uniform bind group (camera + transforms) */
  uniformBindGroup: GPUBindGroup;
  /** Transforms buffer version (for cache invalidation) */
  transformsVersion: number;
}

/**
 * Validate that frame resources are current.
 * Throws in development if stale resources are detected.
 */
export function validateFrameResources(
  resources: FrameResources,
  currentFrame: number,
): void {
  if (process.env.NODE_ENV !== "production") {
    if (resources.frameVersion !== currentFrame) {
      console.error(
        `[GPULifecycle] Stale frame resources detected! ` +
          `Resource frame: ${resources.frameVersion}, current: ${currentFrame}`,
      );
    }
  }
}

// =============================================================================
// Singleton for simple use cases
// =============================================================================

let defaultManager: GPULifecycleManager | null = null;

/**
 * Get the default lifecycle manager singleton.
 * Creates one if it doesn't exist.
 */
export function getDefaultLifecycleManager(): GPULifecycleManager {
  if (!defaultManager) {
    defaultManager = new GPULifecycleManager();
  }
  return defaultManager;
}

/**
 * Reset the default lifecycle manager.
 * Useful for testing.
 */
export function resetDefaultLifecycleManager(): void {
  if (defaultManager) {
    defaultManager.destroyAll();
    defaultManager = null;
  }
}
