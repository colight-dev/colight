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
 * Outcome of waiting for readiness.
 *
 * `settled: true` is the primary contract: every unit of pending work
 * registered via `beginUpdate` drained, so the scene has stopped changing and
 * a capture of it is deterministic.
 *
 * `settled: false` is the labeled fallback: the pending counter never reached
 * zero within the settle window, but at least one frame of work ran to
 * completion, so the page is rendering fine — it is simply still animating.
 * A capture taken here is a valid picture of a moving scene, not a
 * reproducible hash.
 */
export interface ReadyResult {
  settled: boolean;
  /** Short human-readable explanation, present only when `settled` is false. */
  reason?: string;
}

const SETTLED: ReadyResult = { settled: true };

/**
 * Global ready state manager that tracks pending async operations
 */
export class ReadyStateManager {
  private pendingCount = 0;
  private readyPromise: Promise<void> | null = null;
  private resolveReady: (() => void) | null = null;
  private completedUpdates = 0;

  /**
   * Whether at least one unit of registered work has run to completion.
   *
   * This is the "the page is alive" signal. It is recorded at exactly the
   * point the pending counter is decremented — i.e. from the same callback
   * the renderer invokes after `device.queue.onSubmittedWorkDone()` resolves
   * (see impl3d's `renderFrame`) — so it means real GPU work finished, not
   * that a timer elapsed.
   */
  public hasCompletedFrame(): boolean {
    return this.completedUpdates > 0;
  }

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
      this.completedUpdates++;
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
   * Returns a promise that resolves when all pending operations are complete.
   *
   * Quiescence is the default and primary contract: with no `settleWindowMs`,
   * this waits indefinitely for the pending counter to drain, exactly as it
   * always has.
   *
   * With a `settleWindowMs`, a second tier applies. A continuously-animating
   * scene whose per-tick work outlasts the tick never drains the counter, so
   * waiting for quiescence alone would hang until the caller's load timeout
   * and report an opaque failure for a page that is rendering perfectly well.
   * When the window elapses without quiescence, this resolves
   * `{settled: false}` **provided at least one unit of work completed** — the
   * scene is busy, not broken. If nothing ever completed, the promise keeps
   * waiting so a genuinely broken page still fails loudly at the caller's
   * timeout rather than being mislabeled as merely busy.
   *
   * @param settleWindowMs Milliseconds to wait for quiescence before falling
   *   back to the unsettled result. Omit to wait indefinitely.
   */
  public async whenReady(settleWindowMs?: number): Promise<ReadyResult> {
    if (this.pendingCount === 0) {
      return SETTLED;
    }

    log(
      `[ReadyState] whenReady called, waiting for ${this.pendingCount} pending updates`,
    );
    this.ensurePromise();
    const quiesced = this.readyPromise!.then(() => SETTLED);
    if (settleWindowMs === undefined) {
      return quiesced;
    }
    return Promise.race([quiesced, this.unsettledAfter(settleWindowMs)]);
  }

  /**
   * Resolves `{settled: false}` once `windowMs` has elapsed AND a frame has
   * completed. If no frame has completed by then, it never resolves, leaving
   * the caller's timeout to surface a genuinely broken page.
   */
  private unsettledAfter(windowMs: number): Promise<ReadyResult> {
    return new Promise<ReadyResult>((resolve) => {
      setTimeout(() => {
        if (!this.hasCompletedFrame()) return;
        const pending = this.pendingCount;
        log(`[ReadyState]  ⏱️ settle window elapsed, ${pending} still pending`);
        resolve({
          settled: false,
          reason:
            `scene still animating after ${windowMs}ms ` +
            `(${pending} render update${pending === 1 ? "" : "s"} still pending)`,
        });
      }, windowMs);
    });
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
    this.completedUpdates = 0;
  }

  private ensurePromise(): void {
    if (!this.readyPromise) {
      this.readyPromise = new Promise<void>((resolve) => {
        this.resolveReady = resolve;
      });
    }
  }
}
