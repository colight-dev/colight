import * as React from "react";
const { createContext } = React;

declare global {
  interface Window {
    $StateContext: React.Context<any>;
  }
}

export const $StateContext: React.Context<any> = (window.$StateContext =
  window.$StateContext || createContext<any>(undefined));
export const AUTOGRID_MIN: number = 165;
