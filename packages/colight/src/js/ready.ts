/**
 * Ready state tracking system for Colight components
 *
 * This module provides utilities for tracking readiness of asynchronous components
 * like WebGPU rendering in Scene3d. Components can register themselves as "loading"
 * and signal when they're "ready", and other parts of the application can wait for
 * all components to be ready.
 */

const DEBUG = false;

const log = (...body: any[]) => {
  if (!DEBUG) return;
  console.log(...body);
};

/**
 * Global ready state manager that tracks pending async operations
 */
export class ReadyStateManager {
  private pendingCount = 0;
  private readyPromise: Promise<void> | null = null;
  private resolveReady: (() => void) | null = null;

  /**
   * Increment the pending counter, indicating an async operation has started
   * @returns A function to call when the operation completes
   */
  public beginUpdate(label: string): () => void {
    let valid = true;
    this.pendingCount++;
    log(
      `[ReadyState]${" ".repeat(this.pendingCount * 2)} 🟡 ${label}`,
      `pending: ${this.pendingCount}`,
    );
    this.ensurePromise();

    return () => {
      if (!valid) return;
      valid = false;
      this.pendingCount--;
      log(
        `[ReadyState]${" ".repeat((this.pendingCount + 1) * 2)} 🟢 ${label}`,
        `pending: ${this.pendingCount}`,
      );
      if (this.pendingCount === 0 && this.resolveReady) {
        log(`[ReadyState]  🔥 All updates complete`);
        this.resolveReady();
        this.readyPromise = null;
        this.resolveReady = null;
      }
    };
  }

  /**
   * Returns a promise that resolves when all pending operations are complete
   */
  public async whenReady(): Promise<void> {
    if (this.pendingCount === 0) {
      return Promise.resolve();
    }

    log(
      `[ReadyState] whenReady called, waiting for ${this.pendingCount} pending updates`,
    );
    this.ensurePromise();
    return this.readyPromise!;
  }

  /**
   * Returns true if there are no pending operations
   */
  public isReady(): boolean {
    return this.pendingCount === 0;
  }

  /**
   * Reset the ready state for testing purposes
   */
  public reset(): void {
    this.pendingCount = 0;
    this.readyPromise = null;
    this.resolveReady = null;
  }

  private ensurePromise(): void {
    if (!this.readyPromise) {
      this.readyPromise = new Promise<void>((resolve) => {
        this.resolveReady = resolve;
      });
    }
  }
}
