import * as d3 from "d3";
import * as mobx from "mobx";
import * as React from "react";
import * as ReactDOM from "react-dom/client";
import * as api from "./api";
import widgetCSS from "../widget.css";
import { evaluate, loadImports, collectBuffers, replaceBuffers } from "./eval";
import { $StateContext } from "./context";
import { useCellUnmounted, tw } from "./utils";
import { ReadyStateManager } from "./ready";
import * as globals from "./globals";
import { parseColightData, parseColightScript } from "./format";

const { useState, useMemo, useCallback, useEffect } = React;

function resolveRef(node, $state) {
  if (node && typeof node === "object" && node["__type__"] === "ref") {
    return resolveRef($state[node.state_key], $state);
  }
  return node;
}

window.moduleCache = window.moduleCache || new Map();

// Inject CSS if not already present
function injectCSS() {
  if (!document.querySelector("#colight-widget-styles")) {
    const style = document.createElement("style");
    style.id = "colight-widget-styles";
    style.textContent = widgetCSS;
    document.head.appendChild(style);
  }
}

function applyUpdate($state, init, op, payload) {
  const evaluatedPayload = $state.evaluate(payload);
  switch (op) {
    case "append":
      return [...init, evaluatedPayload];
    case "concat":
      return [...init, ...evaluatedPayload];
    case "reset":
      return evaluatedPayload;
    case "setAt":
      const [i, v] = evaluatedPayload;
      const newArray = init.slice();
      newArray[i] = v;
      return newArray;
    default:
      throw new Error(`Unknown operation: ${op}`);
  }
}

// normalize updates to handle both dict and array formats
function normalizeUpdates(updates) {
  return updates.flatMap((entry) => {
    if (entry.constructor === Object) {
      return Object.entries(entry).map(([key, value]) => [key, "reset", value]);
    }
    // handle array format [key, operation, payload]
    const [key, operation, payload] = entry;
    return [[typeof key === "string" ? key : key.id, operation, payload]];
  });
}

/**
 * Reduce a .colight update entry to the update operations it applies, mirroring
 * the render CLI / applyUpdateEntries semantics (see docs format.md §3.6):
 *
 *   - a dict `ast` ({key: value}) or list of [key, op, payload] ops is applied
 *     as-is,
 *   - otherwise a non-empty `state` map is applied as per-key resets.
 *
 * Returns `[[key, op, payload], ...]` (buffers NOT yet resolved) or [] when the
 * entry carries a fresh AST (a re-render, which the scrubber cannot recompute
 * incrementally — such entries fall back to the built-in forward apply).
 *
 * @param {{data: object}} entry - one parsed update entry
 * @returns {Array<[string, string, any]>}
 */
export function updateEntryOps(entry) {
  const data = entry?.data ?? {};
  const ast = data.ast;
  if (ast != null) {
    // A fresh AST (dict/list that is not an update payload) is not a state
    // patch — the caller handles that case; here we only reduce patch forms.
    if (Array.isArray(ast) || ast.constructor === Object) {
      return normalizeUpdates(Array.isArray(ast) ? ast : [ast]);
    }
    return [];
  }
  const state = data.state;
  if (state && Object.keys(state).length > 0) {
    return normalizeUpdates([state]);
  }
  return [];
}

/**
 * Compute the flat list of update operations that move state from its initial
 * value to the value after applying the first `k` update entries.
 *
 * Buffers are resolved per-entry (each entry's refs index into its own
 * buffers). Later entries override earlier ones for the same key, so the
 * scrubber can recompute any position from the initial state in one pass — no
 * snapshot infrastructure, correct for hundreds of entries.
 *
 * @param {Array<{data: object, buffers: any[]}>} entries
 * @param {number} k - number of entries to apply (0 = initial state)
 * @returns {Array<[string, string, any]>} normalized ops with buffers resolved
 */
export function computeStepUpdates(entries, k) {
  const ops = [];
  const count = Math.max(0, Math.min(k, entries.length));
  for (let i = 0; i < count; i++) {
    const entry = entries[i];
    const entryOps = updateEntryOps(entry);
    if (!entryOps.length) continue;
    ops.push(...replaceBuffers(entryOps, entry.buffers || []));
  }
  return ops;
}

/**
 * Gets a deeply nested value from the state store using dot notation.
 *
 * Traverses nested objects/arrays by splitting the property path on dots.
 * For array access, converts string indices to integers.
 * Falls back to stateHandler.get() for non-nested properties.
 *
 * @param {Object} target - The state store target object
 * @param {string} prop - The property path using dot notation (e.g. "a.b.c" or "points.0.x")
 * @returns {any} The value at the specified path
 * @throws {Error} If the path cannot be resolved
 */
function getDeep(stateHandler, target, prop) {
  if (prop.includes(".")) {
    const parts = prop.split(".");
    const topKey = parts[0];
    const rest = parts.slice(1);

    // Then traverse the remaining path
    return rest.reduce(
      (obj, key) => {
        if (Array.isArray(obj) || ArrayBuffer.isView(obj)) {
          return obj[parseInt(key)];
        }
        return obj[key];
      },
      stateHandler.get(target, topKey),
    );
  }
  return stateHandler.get(target, prop);
}

/**
 * Sets a deeply nested value in the state store using dot notation.
 * Creates new objects/arrays along the path to maintain proper reactivity.
 *
 * @param {Object} target - The state store target object
 * @param {string} prop - The property path using dot notation (e.g. "a.b.c" or "points.0.x")
 * @param {any} value - The value to set
 * @returns {boolean} True if the set operation succeeded
 */
function setDeep(stateHandler, target, prop, value) {
  if (!prop.includes(".")) {
    return stateHandler.set(target, prop, value);
  }

  const parts = prop.split(".");
  const first = parts[0];
  const current = stateHandler.get(target, first);

  // Build up the new object/array structure from bottom up
  let result = value;
  for (let i = parts.length - 1; i > 0; i--) {
    const key = parts[i];
    const parentKey = parts[i - 1];
    const index = parseInt(key);
    const isArray = !isNaN(index);
    const parentIndex = parseInt(parentKey);
    const isParentArray = !isNaN(parentIndex);

    // Get the parent container we'll be modifying
    const parent =
      i === 1
        ? current
        : isParentArray
          ? parentKey in current
            ? current[parentIndex]
            : []
          : parentKey in current
            ? current[parentKey]
            : {};

    // Create the new container with our value
    if (isArray) {
      if (ArrayBuffer.isView(parent)) {
        const newArray = new parent.constructor(parent);
        newArray[index] = result;
        result = newArray;
      } else {
        result = Object.assign([...parent], { [index]: result });
      }
    } else {
      result = { ...parent, [key]: result };
    }
  }

  return stateHandler.set(target, first, result);
}

/**
 * Creates a reactive state store with optional sync capabilities
 * @param {Object} data - The data object containing state, experimental interface, and buffers
 * @returns {Promise<Proxy>} A promise that resolves to a proxied state store with reactive capabilities
 */
export async function createStateStore(data) {
  data = { syncedKeys: [], imports: [], listeners: {}, ...data };
  const experimental = data.experimental;
  const buffers = data.buffers;
  const stateMap = mobx.observable.map(data.state, { deep: false });
  const computeds = {};
  const reactions = {};
  const readyState = new ReadyStateManager();
  const evalEnv = { colight: { api } };

  const stateHandler = {
    get(target, key) {
      if (key in target) return target[key];
      return target.__computed(key);
    },
    set: (_target, key, value) => {
      const newValue =
        typeof value === "function" ? value(stateMap.get(key)) : value;
      const updates = applyUpdates([[key, "reset", newValue]]);
      notifyPython(updates);
      return true;
    },
    ownKeys(_target) {
      return Array.from(stateMap.keys());
    },
    getOwnPropertyDescriptor(_target, key) {
      return {
        enumerable: true,
        configurable: true,
        value: this.get(_target, key),
      };
    },
  };

  // Track the current transaction depth and accumulated updates
  let updateDepth = 0;
  let transactionUpdates = null;

  // Timeline-scrubber state (see TimelineScrubber). The baseline is the
  // initial value of every key any update entry touches, captured once so the
  // scrubber can rewind to any step by resetting + replaying entries 0..k.
  // These live in the closure, not on the proxy (whose `set` trap would treat
  // them as state keys).
  let timelineBaseline = null;
  let timelinePrepared = false;
  let initialStateSnapshot = null;

  function notifyPython(updates) {
    if (!experimental || !updates) return;
    updates = updates.filter(([key]) => $state.__syncedKeys.has(key));
    if (!updates.length) return;

    // if we're already in a transaction, just add to it.
    if (transactionUpdates) {
      transactionUpdates.push(...updates);
      return;
    }
    const [processedUpdates, buffers] = collectBuffers(updates);
    experimental.invoke(
      "handle_updates",
      {
        updates: processedUpdates,
      },
      { buffers },
    );
  }

  // notify python when computed state changes.
  // these are dependent reactions which will run within applyUpdates.
  const listenToComputed = (key, value) => {
    reactions[key]?.(); // clean up existing reaction, if it exists.
    const isComputed =
      value?.constructor === Object && value.__type__ === "js_source";
    if ($state.__syncedKeys.has(key) && isComputed) {
      reactions[key] = mobx.reaction(
        () => $state[key],
        (value) => notifyPython([[key, "reset", value]]),
        { fireImmediately: true },
      );
    }
  };

  // notify js listeners when updates occur
  const notifyJs = (updates) => {
    updates.forEach(([key]) => {
      const keyListeners = $state.__listeners[key];
      if (keyListeners) {
        const value = $state[key];
        keyListeners.forEach((callback) => callback({ value }));
      }
    });
  };

  const applyUpdates = (updates) => {
    // Track update depth and initialize accumulator at root level
    updateDepth++;
    const isRoot = updateDepth === 1;
    if (isRoot) {
      transactionUpdates = [];
    }

    // Add initial updates to accumulated list
    transactionUpdates.push(...updates);

    // Run updates within a mobx action to batch reactions
    mobx.action(() => {
      for (const update of updates) {
        const [key, operation, payload] = update;
        const init = $state[key];
        stateMap.set(key, applyUpdate($state, init, operation, payload));
      }
    })();

    // Notify JS listeners which may trigger more updates
    notifyJs(updates);

    updateDepth--;

    // Only return accumulated updates at root level
    if (isRoot) {
      const rootUpdates = transactionUpdates;
      transactionUpdates = null;
      return rootUpdates;
    }

    return null;
  };

  const $state = new Proxy(
    {
      evaluate: (ast) => evaluate(ast, $state, experimental, buffers),
      whenReady() {
        return readyState.whenReady();
      },
      beginUpdate(label) {
        return readyState.beginUpdate(label);
      },
      __evalEnv: evalEnv,
      __listeners: {},
      __syncedKeys: new Set(),
      __backfill: function (state, syncedKeys) {
        syncedKeys.forEach((key) => $state.__syncedKeys.add(key));
        for (const [key, value] of Object.entries(state)) {
          if (!stateMap.has(key)) {
            stateMap.set(key, value);
          }
          listenToComputed(key, value);
        }
      },
      __mergeListeners: function (listeners) {
        for (const [key, fns] of Object.entries(listeners)) {
          if (!$state.__listeners[key]) {
            $state.__listeners[key] = [];
          }
          $state.__listeners[key].push(...fns);
        }
      },

      __updateEnvironment: async function ({
        imports,
        listeners,
        state,
        syncedKeys,
        updateEntries,
      }) {
        // load imports
        Object.assign(evalEnv, await loadImports(imports));
        $state.__backfill(state, syncedKeys);
        $state.__mergeListeners($state.evaluate(listeners));
        await $state.applyUpdateEntries(updateEntries);
      },

      __resolveRef: function (node) {
        return resolveRef(node, $state);
      },

      __computed: function (key) {
        if (!(key in computeds)) {
          computeds[key] = mobx.computed(() => {
            return $state.evaluate(stateMap.get(key));
          });
        }
        return computeds[key].get();
      },

      updateWithBuffers: mobx.action((updates, buffers) => {
        updates = replaceBuffers(updates, buffers);
        applyUpdates(normalizeUpdates(updates));
      }),

      applyUpdateEntries: async (entries) => {
        if (!entries) return;
        for (const { data, buffers } of entries) {
          // Resolve buffer refs in the entry's `state` against THIS entry's
          // buffers before backfill — otherwise __backfill would store raw
          // {__buffer_index__} envelopes that later evaluate against the wrong
          // (store) buffers.
          await $state.__updateEnvironment({
            ...data,
            state: data.state
              ? replaceBuffers(data.state, buffers || [])
              : data.state,
          });
        }
        mobx.runInAction(() => {
          entries.forEach((entry) => {
            // updateEntryOps handles all patch forms uniformly: a dict `ast`
            // ({key: value}, produced by state-update artifacts like ARC
            // replays), a list of [key, op, payload] ops, or a state-only
            // (ast: null) entry — each applied as resets that overwrite
            // existing keys (matching the render CLI).
            const ops = updateEntryOps(entry);
            if (ops.length) {
              $state.updateWithBuffers(ops, entry.buffers || []);
            }
          });
        });
      },

      update: (...updates) => {
        updates = applyUpdates(normalizeUpdates(updates));
        notifyPython(updates);
      },

      // Run the async environment setup (imports/backfill/listeners) for a
      // list of update entries ONCE, then snapshot the initial value of every
      // key they touch (using the pre-update snapshot when available, else the
      // current value). After this, applyUpdateEntriesUpTo is synchronous.
      prepareUpdateEntries: async (entries) => {
        if (!entries || timelinePrepared) return;
        timelinePrepared = true;
        for (const { data, buffers } of entries) {
          // Backfill new keys / load imports without applying patches: the
          // envelope carries no nested updateEntries, so this only sets up env.
          await $state.__updateEnvironment({
            ...data,
            state: data.state
              ? replaceBuffers(data.state, buffers || [])
              : data.state,
            updateEntries: undefined,
          });
        }
        const snapshot = initialStateSnapshot || {};
        const baseline = {};
        for (const entry of entries) {
          for (const [key] of updateEntryOps(entry)) {
            if (!(key in baseline)) {
              baseline[key] =
                key in snapshot ? snapshot[key] : stateMap.get(key);
            }
          }
        }
        timelineBaseline = baseline;
      },

      // Set state to initial + first `k` update entries. Rewinds by resetting
      // the touched keys to their captured baseline, then replays entries
      // 0..k in one action (recompute-from-scratch; correct and cheap for
      // hundreds of entries).
      applyUpdateEntriesUpTo: (entries, k) => {
        const baseline = timelineBaseline || {};
        mobx.runInAction(() => {
          for (const [key, value] of Object.entries(baseline)) {
            stateMap.set(key, value);
          }
          const ops = computeStepUpdates(entries, k);
          if (ops.length) applyUpdates(ops);
        });
      },
    },
    {
      ...stateHandler,
      get: getDeep.bind(null, stateHandler),
      set: setDeep.bind(null, stateHandler),
    },
  );

  // Snapshot the initial state values BEFORE any update entries are applied,
  // so a TimelineScrubber can rewind to step 0. __updateEnvironment below then
  // applies the entries as usual (final frame on load); the scrubber, once
  // mounted, resets to whichever step the user selects.
  initialStateSnapshot = {};
  for (const key of stateMap.keys()) {
    initialStateSnapshot[key] = stateMap.get(key);
  }

  await $state.__updateEnvironment(data);

  return $state;
}

/**
 * Timeline scrubber for a .colight artifact that carries update entries
 * (a replay/episode). Renders a step slider (0..N), play/pause and a step
 * count; each position applies update entries 0..k on top of the initial
 * state. Recomputes from the initial state on every scrub — no snapshots.
 *
 * This is chrome around the visual driving which UPDATE ENTRIES are applied.
 * It is independent of any in-AST Plot.Slider (animateBy autoplay), which
 * drives its own state key within whatever entry is currently applied.
 */
function TimelineScrubber({ $state, updateEntries }) {
  const total = updateEntries.length;
  const [step, setStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      await $state.prepareUpdateEntries(updateEntries);
      if (!cancelled) setReady(true);
    })();
    return () => {
      cancelled = true;
    };
  }, [$state, updateEntries]);

  useEffect(() => {
    if (ready) $state.applyUpdateEntriesUpTo(updateEntries, step);
  }, [ready, step, $state, updateEntries]);

  useEffect(() => {
    if (!isPlaying) return;
    const id = setInterval(() => {
      setStep((prev) => {
        if (prev >= total) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1000 / 8);
    return () => clearInterval(id);
  }, [isPlaying, total]);

  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => {
      // Restart from the beginning if we're at the end.
      if (!prev && step >= total) setStep(0);
      return !prev;
    });
  }, [step, total]);

  return (
    <div
      className={tw(
        "flex items-center gap-2 my-2 text-xs text-gray-700 select-none",
      )}
      data-testid="timeline-scrubber"
    >
      <button
        type="button"
        onClick={togglePlay}
        className={tw(
          "px-2 py-0.5 rounded border border-gray-300 hover:bg-gray-100 cursor-pointer",
        )}
        aria-label={isPlaying ? "Pause" : "Play"}
      >
        {isPlaying ? "❚❚" : "▶"}
      </button>
      <input
        type="range"
        min={0}
        max={total}
        step={1}
        value={step}
        onChange={(e) => {
          setIsPlaying(false);
          setStep(Number(e.target.value));
        }}
        className={tw("flex-1 outline-none cursor-pointer")}
        aria-label="Timeline step"
      />
      <span className={tw("font-mono tabular-nums whitespace-nowrap")}>
        {step} / {total}
      </span>
    </div>
  );
}

export function StateProvider(data) {
  const id = data.id;
  const [$state, setCurrentState] = useState(null);
  const [currentAst, setCurrentAst] = useState(null);

  useEffect(() => {
    (async () => {
      setCurrentState(await createStateStore(data));
    })();
    return;
  }, [id]);

  useEffect(() => {
    if ($state) {
      setCurrentAst(data.ast);
    }
    return;
  }, [$state, data.ast]);

  useEffect(() => {
    // if we have an AnyWidget model (ie. we are in widget model),
    // listen for `update_state` events.
    if ($state && data.model) {
      const cb = (msg, buffers) => {
        if (msg.type === "update_state") {
          $state.updateWithBuffers(msg.updates, buffers);
        }
      };
      data.model.on("msg:custom", cb);
      return () => data.model.off("msg:custom", cb);
    }
  }, [$state, data.model]);

  useEffect(() => {
    if (currentAst) {
      globals.colight.instances[id] = $state;
      if (data.onMount) {
        setTimeout(async () => {
          await $state.whenReady();
          data.onMount();
        }, 10);
      }
      return () => delete globals.colight.instances[id];
    }
  }, [id, currentAst]);

  if (!currentAst) return;

  return (
    <$StateContext.Provider value={$state}>
      <api.Node value={currentAst} />
      {data.updateEntries && data.updateEntries.length > 0 && (
        <TimelineScrubber $state={$state} updateEntries={data.updateEntries} />
      )}
    </$StateContext.Provider>
  );
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className={tw("p-4 border border-red-500 rounded bg-red-50")}>
          <h2 className={tw("text-red-700 font-bold mb-2")}>
            Something went wrong
          </h2>
          <pre className={tw("text-sm text-red-600 whitespace-pre-wrap")}>
            {this.state.error?.message}
          </pre>
        </div>
      );
    }

    return this.props.children;
  }
}

function DraggableViewer({ data: initialData }) {
  const [currentData, setCurrentData] = useState(initialData);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      // Check if we're leaving the component entirely
      if (!e.currentTarget.contains(e.relatedTarget)) {
        setDragActive(false);
      }
    }
  }, []);

  const handleDrop = useCallback(
    async (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0];

        // Check if it's a .colight file
        if (file.name.endsWith(".colight")) {
          try {
            const arrayBuffer = await file.arrayBuffer();
            const parsedData = parseColightData(arrayBuffer);

            // Check if this is an update-only file
            if (!parsedData.ast && parsedData.updateEntries) {
              const $state = globals.colight.instances[currentData.id];
              if (currentData && currentData.id && $state) {
                $state.applyUpdateEntries(parsedData.updateEntries);
              } else {
                alert("No visualization loaded to apply updates to");
              }
            } else {
              setCurrentData(parsedData);
            }
          } catch (error) {
            console.error("Error parsing .colight file:", error);
            alert(`Error loading .colight file: ${error.message}`);
          }
        } else {
          alert("Please drop a .colight file");
        }
      }
    },
    [currentData],
  );

  return (
    <div
      className={tw(
        `relative ${dragActive ? "ring-2 ring-blue-500 ring-opacity-50" : ""}`,
      )}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <Viewer {...currentData} />
      {dragActive && (
        <div
          className={tw(
            "absolute inset-0 bg-blue-500 bg-opacity-10 pointer-events-none flex items-center justify-center",
          )}
        >
          <div className={tw("bg-white px-4 py-2 rounded shadow-lg")}>
            <p className={tw("text-sm font-medium")}>Drop .colight file here</p>
          </div>
        </div>
      )}
    </div>
  );
}

export function Viewer(data) {
  injectCSS();

  const [el, setEl] = useState();
  const elRef = useCallback((element) => element && setEl(element), [setEl]);
  const isUnmounted = useCellUnmounted(el?.parentNode);

  if (isUnmounted || !data) {
    return null;
  }

  return (
    <div className="colight-container" ref={elRef}>
      {el && (
        <ErrorBoundary>
          <StateProvider {...data} />
        </ErrorBoundary>
      )}
      {data.size && data.dev && (
        <div className={tw("text-xl p-3")}>{data.size}</div>
      )}
    </div>
  );
}

/**
 * Renders the Colight widget into a specified DOM element.
 * Handles both inline base64 encoded buffers and URLs for large buffers.
 *
 * @param {string|HTMLElement} element - The target DOM element or its ID.
 * @param {object} data - The widget data containing placeholders like {__buffer__: i}.
 * @param {Array<string|object>} buffers_payload - Mixed list containing either:
 *   - Base64 encoded strings for inline buffers.
 *   - Objects like { type: 'url', url: string } for large buffers.
 * @param {string} id - A unique identifier for the widget instance.
 */
export const render = async (element, data, id) => {
  id = id || data.id;

  // If element is a string, treat it as an ID and find/create the element
  const el =
    typeof element === "string"
      ? document.getElementById(element) ||
        (() => {
          const div = document.createElement("div");
          div.id = element;
          document.body.appendChild(div);
          return div;
        })()
      : element;

  if (el._ReactRoot) {
    el._ReactRoot.unmount();
  }
  // Assert that data is an object
  if (typeof data !== "object" || data === null) {
    console.error("data must be an object, got:", typeof data);
    return;
  }

  const { buffers } = data;
  if (buffers !== undefined && buffers !== null && !Array.isArray(buffers)) {
    console.error("buffers_payload must be an array, got:", typeof buffers);
  }

  const root = ReactDOM.createRoot(el);
  el._ReactRoot = root;
  // Pass the original data (with placeholders) and the fully resolved buffers array
  root.render(<DraggableViewer data={{ ...data, id }} />);
};

export {
  parseColightData,
  parseColightScript,
  DraggableViewer,
  TimelineScrubber,
};

globals.colight.render = render;
