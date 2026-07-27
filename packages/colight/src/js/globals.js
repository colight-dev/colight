import { parseColightData, loadColightFile } from "./format.js";

export const colight = {
  // Registry of all component instances
  instances: {},

  // Format parsing functions
  parseColightData,
  loadColightFile,
};

/**
 * Wait for a rendered instance to be ready.
 *
 * @param {string} id Instance id.
 * @param {number} [settleWindowMs] Milliseconds to wait for the scene to
 *   settle before resolving as unsettled instead (see ReadyStateManager).
 *   Omit to wait indefinitely for quiescence.
 * @returns {Promise<{settled: boolean, reason?: string}>}
 */
colight.whenReady = async function (id, settleWindowMs) {
  while (!colight.instances[id]) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  return await colight.instances[id].whenReady(settleWindowMs);
};

window.colight = colight;
