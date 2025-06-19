import * as React from "react";
const { useState, useEffect, useContext, useRef, useCallback, useMemo } = React;

import { $StateContext } from "./context";
import Katex from "katex";
import markdownItKatex from "./markdown-it-katex";
import { tw, joinClasses } from "./utils";

import MarkdownIt from "markdown-it";
const KATEX_DIST_URL = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist";

const KATEX_CSS_URL = `${KATEX_DIST_URL}/katex.min.css`;

// KaTeX font URLs for preloading
const KATEX_FONTS = [
  `${KATEX_DIST_URL}/fonts/KaTeX_Main-Regular.woff2`,
  `${KATEX_DIST_URL}/fonts/KaTeX_Math-Italic.woff2`,
  `${KATEX_DIST_URL}/fonts/KaTeX_Main-Bold.woff2`,
  `${KATEX_DIST_URL}/fonts/KaTeX_Main-Italic.woff2`,
  `${KATEX_DIST_URL}/fonts/KaTeX_AMS-Regular.woff2`,
  `${KATEX_DIST_URL}/fonts/KaTeX_Size1-Regular.woff2`,
  `${KATEX_DIST_URL}/fonts/KaTeX_Size2-Regular.woff2`,
  `${KATEX_DIST_URL}/fonts/KaTeX_Size3-Regular.woff2`,
  `${KATEX_DIST_URL}/fonts/KaTeX_Size4-Regular.woff2`,
];

let katexResourcesPromise = null;
let finishKatexUpdate = null;

function preloadKatexFonts() {
  // Add font preload links
  KATEX_FONTS.forEach((fontUrl) => {
    const existingLink = document.querySelector(`link[href="${fontUrl}"]`);
    if (!existingLink) {
      const link = document.createElement("link");
      link.rel = "preload";
      link.href = fontUrl;
      link.as = "font";
      link.type = "font/woff2";
      link.crossOrigin = "anonymous";
      document.head.appendChild(link);
    }
  });
}

function loadKatexResources($state) {
  if (katexResourcesPromise) return katexResourcesPromise;

  finishKatexUpdate = $state.beginUpdate("api/katex-resources");

  // Preload fonts first
  preloadKatexFonts();

  katexResourcesPromise = new Promise((resolve, reject) => {
    const existingLink = document.querySelector(
      `link[href="${KATEX_CSS_URL}"]`,
    );
    if (!existingLink) {
      const link = document.createElement("link");
      link.rel = "stylesheet";
      link.href = KATEX_CSS_URL;
      link.onload = () => {
        // Wait a bit for fonts to be available
        setTimeout(() => {
          finishKatexUpdate();
          resolve();
        }, 100);
      };
      link.onerror = () => {
        finishKatexUpdate();
        reject(new Error("Failed to load KaTeX CSS"));
      };
      document.head.appendChild(link);
    } else {
      // CSS already loaded, just wait for fonts
      setTimeout(() => {
        finishKatexUpdate();
        resolve();
      }, 100);
    }
  });

  return katexResourcesPromise;
}

let katexResourcesLoaded = false;

function useKatexResources() {
  const [katexLoaded, setKatexLoaded] = useState(katexResourcesLoaded);
  const $state = useContext($StateContext);

  // Effect to load KaTeX resources (CSS + fonts)
  useEffect(() => {
    if (katexLoaded) return;

    loadKatexResources($state)
      .then(() => {
        setKatexLoaded(true);
        katexResourcesLoaded = true;
      })
      .catch((error) => {
        console.error("Error loading KaTeX resources:", error);
      });
  }, []);

  return katexLoaded;
}

export function katex(tex) {
  const containerRef = useRef(null);
  const katexReady = useKatexResources();

  // Effect to render KaTeX when resources are loaded and tex changes
  useEffect(() => {
    if (!katexReady || !containerRef.current) return;

    try {
      Katex.render(tex, containerRef.current, {
        throwOnError: false,
      });
    } catch (error) {
      console.error("Error rendering KaTeX:", error);
    }
  }, [tex, katexReady]);

  return <div ref={containerRef} />;
}

const MarkdownItInstance = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  quotes: `""''`,
});

MarkdownItInstance.use(markdownItKatex);

export function md(options, text) {
  useKatexResources();

  if (typeof options === "string" && !text) {
    text = options;
    options = {};
  }

  return (
    <div
      className={tw(joinClasses("prose", options.className))}
      dangerouslySetInnerHTML={{ __html: MarkdownItInstance.render(text) }}
    />
  );
}
