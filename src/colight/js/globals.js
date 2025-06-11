export const colight = {
  // Registry of all component instances
  instances: {}
}

colight.whenReady = async function(id) {
  while (!colight.instances[id]) {
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  await colight.instances[id].whenReady();
};

window.colight = colight
