import React from "react";
import { $StateContext } from "../../src/colight/js/context";
import { createStateStore } from "../../src/colight/js/widget";

/**
 * Wraps a component with a blank state context for testing.
 * @param Component The component to wrap
 * @param initialState Optional initial state to provide to the store
 * @returns A wrapped component with blank state context
 */
export function withBlankState<P extends object>(
  Component: React.ComponentType<P>,
  initialState: Record<string, any> = {},
) {
  return function WrappedComponent(props: P) {
    const $state = createStateStore({
      initialState,
      syncedKeys: new Set(),
    });

    return (
      <$StateContext.Provider value={$state}>
        <Component {...props} />
      </$StateContext.Provider>
    );
  };
}
