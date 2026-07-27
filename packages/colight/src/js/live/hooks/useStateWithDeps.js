import { useState, useRef, useEffect } from "react";

/**
 * Hook that manages state that resets when dependencies change.
 * Uses a simple useEffect pattern that's reliable and well-understood.
 *
 * @param {*} initialState - Initial state value or function
 * @param {Array} deps - Dependencies array
 * @returns {[*, Function]} - State and setState tuple
 */
export function useStateWithDeps(initialState, deps) {
  const [state, setState] = useState(initialState);
  const prevDepsRef = useRef();

  useEffect(() => {
    // Check if this is the first render or if deps changed
    const depsChanged =
      !prevDepsRef.current ||
      prevDepsRef.current.length !== deps.length ||
      prevDepsRef.current.some((dep, i) => !Object.is(dep, deps[i]));

    if (depsChanged) {
      prevDepsRef.current = [...deps]; // Store a copy
      const newState =
        typeof initialState === "function" ? initialState() : initialState;
      setState(newState);
    }
  }, deps);

  return [state, setState];
}

// Re-export with original name for compatibility
export const useFileState = (file, initialState) =>
  useStateWithDeps(initialState, [file]);
