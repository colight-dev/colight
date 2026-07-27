import { parseColightData, loadColightFile } from "./format.js";

export const colight = {
  // Registry of all component instances
  instances: {},

  // Format parsing functions
  parseColightData,
  loadColightFile,
};

colight.whenReady = async function (id) {
  while (!colight.instances[id]) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  await colight.instances[id].whenReady();
};

window.colight = colight;
