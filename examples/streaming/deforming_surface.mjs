/**
 * A streaming producer written in JavaScript, with no Python involved.
 *
 * It generates a traveling wave over a grid mesh and appends one state update
 * per tick to a `.colight` file, using `@colight/format` alone. Colight's CLI
 * then measures the artifact — `inspect`, `diff`, `render` — without caring
 * what wrote it. That is the whole point: any language writes, colight
 * measures.
 *
 * Usage:
 *
 *     node examples/streaming/deforming_surface.mjs [--out PATH] [--ticks N]
 *                                                   [--grid N] [--delay MS]
 *
 * `--delay` inserts a pause between ticks so a reader can be pointed at the
 * file while it is still growing; with the default of 0 the producer runs flat
 * out.
 */

import { mkdirSync } from "node:fs";
import { dirname } from "node:path";

import { ndarray } from "@colight/format";
import { ColightFileWriter } from "@colight/format/node";

// ---------------------------------------------------------------------------
// Arguments

function parseArgs(argv) {
  const opts = {
    out: "streaming-surface.colight",
    ticks: 60,
    grid: 32,
    delay: 0,
    // Shifts the wave's phase. Two runs that differ only in `--phase` are what
    // `colight diff` is for: same shape, same tick count, different motion.
    phase: 0,
  };
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, "");
    const value = argv[i + 1];
    if (!(key in opts)) {
      throw new Error(`Unknown option --${key}`);
    }
    opts[key] = key === "out" ? value : Number(value);
  }
  return opts;
}

const { out, ticks, grid, delay, phase } = parseArgs(process.argv.slice(2));

// ---------------------------------------------------------------------------
// Geometry: a grid mesh whose vertex positions we rewrite every tick.

/** Triangle indices for a `grid x grid` lattice of vertices. */
function gridIndices(n) {
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let k = 0;
  for (let row = 0; row < n - 1; row++) {
    for (let col = 0; col < n - 1; col++) {
      const a = row * n + col;
      const b = a + 1;
      const c = a + n;
      const d = c + 1;
      indices[k++] = a;
      indices[k++] = c;
      indices[k++] = b;
      indices[k++] = b;
      indices[k++] = c;
      indices[k++] = d;
    }
  }
  return indices;
}

/**
 * Vertex positions at time `t`: a unit-square lattice in x/y whose z is a
 * traveling wave. Returns a fresh Float32Array — the mesh's geometry update
 * path writes new contents into the existing GPU buffers, so a new array per
 * tick is the cheap, correct thing to send.
 */
function wavePositions(n, t) {
  const positions = new Float32Array(n * n * 3);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const x = col / (n - 1) - 0.5;
      const y = row / (n - 1) - 0.5;
      const radius = Math.hypot(x, y);
      const z = 0.18 * Math.sin(12 * radius - 3 * t) * Math.exp(-2 * radius);
      const at = (row * n + col) * 3;
      positions[at] = x;
      positions[at + 1] = y;
      positions[at + 2] = z;
    }
  }
  return positions;
}

// ---------------------------------------------------------------------------
// The artifact: an initial entry carrying the scene, then one update per tick.

/** `Plot.js("$state.positions")` — a JS expression evaluated by the viewer. */
const stateRef = (expression) => ({
  __type__: "js_source",
  value: expression,
  params: [],
  expression: true,
  scope: {},
});

const indices = gridIndices(grid);
const vertexCount = grid * grid;

/**
 * The initial-state entry: a scene3d Mesh whose positions come from
 * `$state.positions`, so every subsequent state-only update redeforms it.
 */
const initialEntry = {
  ast: [
    { __type__: "js_ref", path: "Column" },
    {},
    [
      { __type__: "js_ref", path: "scene3d.Scene" },
      {
        // Pinned rather than auto-fitted, so screenshots taken at different
        // update indices are comparable: only the surface moves.
        defaultCamera: {
          position: [1.1, -1.1, 0.8],
          target: [0, 0, 0],
          up: [0, 0, 1],
          fov: 45,
        },
        layers: [
          {
            __type__: "function",
            path: "scene3d.Mesh",
            args: [
              {
                geometry: {
                  positions: stateRef("$state.positions"),
                  indices: ndarray(indices, [indices.length]),
                },
                centers: ndarray(new Float32Array([0, 0, 0]), [3]),
                color: [0.29, 0.56, 0.89],
                shading: "lit",
                // The surface is open, so show both faces.
                cullMode: "none",
              },
            ],
          },
        ],
      },
    ],
    null,
  ],
  // `tick` is declared here so update entries that carry it are updating a
  // known key rather than introducing one.
  state: {
    tick: 0,
    positions: ndarray(wavePositions(grid, phase), [vertexCount, 3]),
  },
  syncedKeys: [],
  listeners: {},
  imports: [],
  animateBy: [],
};

/** One tick: a state-only update replacing the whole positions array. */
const updateEntry = (tick) => ({
  ast: null,
  state: {
    tick,
    positions: ndarray(wavePositions(grid, phase + tick / 10), [
      vertexCount,
      3,
    ]),
  },
  syncedKeys: [],
  listeners: {},
  imports: [],
  animateBy: [],
});

// ---------------------------------------------------------------------------
// Produce.

mkdirSync(dirname(out) || ".", { recursive: true });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Hold the file open across every append: roughly 3x the throughput of
// reopening per entry, and every append still leaves the file readable.
const writer = ColightFileWriter.create(out, initialEntry);
try {
  for (let tick = 1; tick <= ticks; tick++) {
    writer.append(updateEntry(tick));
    if (delay > 0) await sleep(delay);
  }
} finally {
  writer.close();
}

console.log(
  `wrote ${out}: ${vertexCount} vertices, ${ticks} update entries` +
    (delay > 0 ? ` at ${delay}ms/tick` : ""),
);
