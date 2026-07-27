import React, { useState, useEffect } from "react";
import { $StateContext } from "../../src/js/context";
import { createStateStore } from "../../src/js/widget";

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
    const [$state, set$State] = useState<any>(null);

    useEffect(() => {
      createStateStore({
        state: initialState,
      }).then(set$State);
    }, []);

    if (!$state) {
      return null; // or a loading indicator
    }

    return (
      <$StateContext.Provider value={$state}>
        <Component {...props} />
      </$StateContext.Provider>
    );
  };
}
