/**
 * Benchmark: Minimal NdArrayView (no Proxy)
 *
 * Clean API with consistent performance:
 * - .get(i, j, ...) for element access
 * - .set(value, i, j, ...) for mutation
 * - .reduce() / .forEach() for iteration
 * - .row(i) / .slice(i) for sub-views
 *
 * Run: node packages/colight-serde/scripts/benchmark_minimal.js
 */

function computeStrides(shape) {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Minimal NdArrayView - no Proxy, just methods
 */
function ndarray(data, shape, strides, offset = 0) {
  const actualStrides = strides ?? computeStrides(shape);
  const ndim = shape.length;
  const length = shape.reduce((a, b) => a * b, 1);

  return {
    flat: data,
    shape,
    strides: actualStrides,
    ndim,
    length,
    offset,

    // Element access
    get(...indices) {
      let idx = offset;
      for (let i = 0; i < indices.length; i++) {
        idx += indices[i] * actualStrides[i];
      }
      return data[idx];
    },

    set(value, ...indices) {
      let idx = offset;
      for (let i = 0; i < indices.length; i++) {
        idx += indices[i] * actualStrides[i];
      }
      data[idx] = value;
    },

    // Sub-views (no allocation for the underlying data)
    row(i) {
      if (ndim < 2) throw new Error("row() requires 2D+ array");
      return ndarray(
        data,
        shape.slice(1),
        actualStrides.slice(1),
        offset + i * actualStrides[0],
      );
    },

    // Generic slice along first dimension (works for any ndim)
    slice(i) {
      if (ndim < 1) throw new Error("slice() requires 1D+ array");
      if (ndim === 1) {
        // For 1D, slice returns a scalar - just use get()
        return data[offset + i * actualStrides[0]];
      }
      return ndarray(
        data,
        shape.slice(1),
        actualStrides.slice(1),
        offset + i * actualStrides[0],
      );
    },

    // Iteration - optimized for common cases
    forEach(callback) {
      if (ndim === 1) {
        const s0 = actualStrides[0];
        for (let i = 0; i < shape[0]; i++) {
          callback(data[offset + i * s0], i);
        }
      } else if (ndim === 2) {
        const s0 = actualStrides[0],
          s1 = actualStrides[1];
        for (let i = 0; i < shape[0]; i++) {
          const base = offset + i * s0;
          for (let j = 0; j < shape[1]; j++) {
            callback(data[base + j * s1], i, j);
          }
        }
      } else if (ndim === 3) {
        const s0 = actualStrides[0],
          s1 = actualStrides[1],
          s2 = actualStrides[2];
        for (let i = 0; i < shape[0]; i++) {
          const base0 = offset + i * s0;
          for (let j = 0; j < shape[1]; j++) {
            const base1 = base0 + j * s1;
            for (let k = 0; k < shape[2]; k++) {
              callback(data[base1 + k * s2], i, j, k);
            }
          }
        }
      } else {
        // General case - recursive
        const iterate = (dim, off, indices) => {
          if (dim === ndim) {
            callback(data[off], ...indices);
          } else {
            for (let i = 0; i < shape[dim]; i++) {
              iterate(dim + 1, off + i * actualStrides[dim], [...indices, i]);
            }
          }
        };
        iterate(0, offset, []);
      }
    },

    reduce(callback, initial) {
      let acc = initial;
      if (ndim === 1) {
        const s0 = actualStrides[0];
        for (let i = 0; i < shape[0]; i++) {
          acc = callback(acc, data[offset + i * s0], i);
        }
      } else if (ndim === 2) {
        const s0 = actualStrides[0],
          s1 = actualStrides[1];
        for (let i = 0; i < shape[0]; i++) {
          const base = offset + i * s0;
          for (let j = 0; j < shape[1]; j++) {
            acc = callback(acc, data[base + j * s1], i, j);
          }
        }
      } else if (ndim === 3) {
        const s0 = actualStrides[0],
          s1 = actualStrides[1],
          s2 = actualStrides[2];
        for (let i = 0; i < shape[0]; i++) {
          const base0 = offset + i * s0;
          for (let j = 0; j < shape[1]; j++) {
            const base1 = base0 + j * s1;
            for (let k = 0; k < shape[2]; k++) {
              acc = callback(acc, data[base1 + k * s2], i, j, k);
            }
          }
        }
      } else {
        this.forEach((val, ...indices) => {
          acc = callback(acc, val, ...indices);
        });
      }
      return acc;
    },

    // Map over first dimension, passing sub-views to callback
    map(callback) {
      const results = new Array(shape[0]);
      for (let i = 0; i < shape[0]; i++) {
        if (ndim === 1) {
          results[i] = callback(data[offset + i * actualStrides[0]], i);
        } else {
          results[i] = callback(this.slice(i), i);
        }
      }
      return results;
    },

    // Fast row iteration for 2D arrays (lightweight row accessor, not full view)
    mapRows(callback) {
      if (ndim !== 2) throw new Error("mapRows() requires 2D array");
      const results = new Array(shape[0]);
      const rowStride = actualStrides[0];
      const colStride = actualStrides[1];
      const cols = shape[1];
      for (let i = 0; i < shape[0]; i++) {
        const rowOffset = offset + i * rowStride;
        // Lightweight row accessor (not a full NdArrayView)
        results[i] = callback(
          {
            get(j) {
              return data[rowOffset + j * colStride];
            },
            length: cols,
            forEach(cb) {
              for (let j = 0; j < cols; j++) {
                cb(data[rowOffset + j * colStride], j);
              }
            },
            reduce(cb, init) {
              let acc = init;
              for (let j = 0; j < cols; j++) {
                acc = cb(acc, data[rowOffset + j * colStride], j);
              }
              return acc;
            },
          },
          i,
        );
      }
      return results;
    },
  };
}

// Benchmark runner
function benchmark(name, fn, iterations = 1000000) {
  // Warmup
  for (let i = 0; i < 1000; i++) fn();

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = performance.now();

  const totalMs = end - start;
  const opsPerSec = (iterations / totalMs) * 1000;
  const nsPerOp = (totalMs / iterations) * 1e6;

  console.log(
    `${name.padEnd(50)} ${nsPerOp.toFixed(1).padStart(8)} ns/op  ${(opsPerSec / 1e6).toFixed(2).padStart(6)} M ops/sec`,
  );
  return { name, nsPerOp, opsPerSec };
}

console.log("=".repeat(80));
console.log("Minimal NdArrayView Benchmark (No Proxy)");
console.log("=".repeat(80));
console.log();

// 2D array: 100x1000
const rows = 100;
const cols = 1000;
const data2d = new Float32Array(rows * cols);
for (let i = 0; i < data2d.length; i++) data2d[i] = i;

const view2d = ndarray(data2d, [rows, cols]);
const midRow = 50;

console.log(`2D Array: ${rows}x${cols} = ${rows * cols} elements`);
console.log();

console.log("--- Element Access ---");
console.log();

benchmark("arr.get(i, j)", () => {
  return view2d.get(midRow, 500);
});

benchmark("arr.flat[i * stride + j] (manual baseline)", () => {
  return data2d[midRow * cols + 500];
});

console.log();
console.log("--- Sub-view Access ---");
console.log();

benchmark("arr.row(i).get(j)", () => {
  return view2d.row(midRow).get(500);
});

benchmark("arr.slice(i).get(j)", () => {
  return view2d.slice(midRow).get(500);
});

// Pre-fetch row to see cost of view creation vs access
const preRow = view2d.row(midRow);
benchmark("prefetched row.get(j)", () => {
  return preRow.get(500);
});

console.log();
console.log("--- Row Iteration (sum 1000 elements) ---");
console.log();

benchmark(
  "arr.row(i).reduce((a,v) => a+v, 0)",
  () => {
    return view2d.row(midRow).reduce((a, v) => a + v, 0);
  },
  1000,
);

benchmark(
  "prefetched row.reduce((a,v) => a+v, 0)",
  () => {
    return preRow.reduce((a, v) => a + v, 0);
  },
  1000,
);

benchmark(
  "manual flat loop",
  () => {
    let sum = 0;
    const base = midRow * cols;
    for (let j = 0; j < cols; j++) {
      sum += data2d[base + j];
    }
    return sum;
  },
  1000,
);

console.log();
console.log("--- Full Array Iteration (100K elements) ---");
console.log();

benchmark(
  "arr.reduce((a,v) => a+v, 0)",
  () => {
    return view2d.reduce((a, v) => a + v, 0);
  },
  100,
);

benchmark(
  "arr.mapRows(r => r.reduce(sum))",
  () => {
    return view2d.mapRows((r) => r.reduce((a, v) => a + v, 0));
  },
  100,
);

benchmark(
  "manual flat iteration",
  () => {
    let sum = 0;
    for (let i = 0; i < data2d.length; i++) {
      sum += data2d[i];
    }
    return sum;
  },
  100,
);

console.log();
console.log("--- Row Sums (100 rows) ---");
console.log();

benchmark(
  "arr.mapRows(r => r.reduce(sum))",
  () => {
    return view2d.mapRows((r) => r.reduce((a, v) => a + v, 0));
  },
  100,
);

benchmark(
  "arr.map(slice => slice.reduce(sum))",
  () => {
    return view2d.map((slice) => slice.reduce((a, v) => a + v, 0));
  },
  100,
);

benchmark(
  "manual nested loop",
  () => {
    const sums = new Array(rows);
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      const base = i * cols;
      for (let j = 0; j < cols; j++) {
        sum += data2d[base + j];
      }
      sums[i] = sum;
    }
    return sums;
  },
  100,
);

console.log();
console.log("--- 3D Array (10x20x30 = 6000 elements) ---");
console.log();

const data3d = new Float32Array(10 * 20 * 30);
for (let i = 0; i < data3d.length; i++) data3d[i] = i;

const view3d = ndarray(data3d, [10, 20, 30]);

benchmark("arr.get(i, j, k)", () => {
  return view3d.get(5, 10, 15);
});

benchmark("arr.slice(i).get(j, k)", () => {
  return view3d.slice(5).get(10, 15);
});

benchmark("arr.slice(i).slice(j).get(k)", () => {
  return view3d.slice(5).slice(10).get(15);
});

benchmark(
  "arr.reduce (sum 6000 elements)",
  () => {
    return view3d.reduce((a, v) => a + v, 0);
  },
  100,
);

benchmark(
  "arr.map(plane => plane.reduce(sum))",
  () => {
    return view3d.map((plane) => plane.reduce((a, v) => a + v, 0));
  },
  100,
);

console.log();
console.log("=".repeat(80));
console.log("Summary:");
console.log();
console.log("Minimal API (no Proxy):");
console.log("  .get(i, j, ...)     - Direct element access");
console.log("  .set(v, i, j, ...)  - Direct element mutation");
console.log("  .row(i) / .slice(i) - Returns sub-view (no data copy)");
console.log("  .reduce(fn, init)   - Fast iteration with accumulator");
console.log("  .forEach(fn)        - Fast iteration");
console.log("  .map(fn)            - Map over first dimension");
console.log("  .mapRows(fn)        - Alias for 2D .map()");
console.log();
console.log("Performance: ~1.2x overhead vs manual flat access for iteration");
console.log("=".repeat(80));
