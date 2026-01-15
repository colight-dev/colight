/**
 * Benchmark: NdArrayView access patterns
 *
 * Compares:
 * 1. arr[i][j] - Proxy-based indexed access (creates sub-views)
 * 2. arr.get(i, j) - Direct offset calculation
 * 3. arr[i][j] with cached sub-views - Proxy with Map caching
 * 4. arr.flat[i * stride + j] - Manual flat array access
 *
 * Run: node packages/colight-serde/scripts/benchmark_ndarray.js
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

// Uncached ndarray (matches current implementation) with iteration methods
function ndarray(data, shape, strides, offset = 0) {
  const actualStrides = strides ?? computeStrides(shape);
  const ndim = shape.length;
  const length = shape.reduce((a, b) => a * b, 1);

  const base = {
    flat: data,
    shape,
    strides: actualStrides,
    ndim,
    length,

    get(...indices) {
      if (indices.length !== ndim) {
        throw new Error(`Expected ${ndim} indices, got ${indices.length}`);
      }
      let idx = offset;
      for (let i = 0; i < ndim; i++) {
        idx += indices[i] * actualStrides[i];
      }
      return data[idx];
    },

    // Iteration methods that avoid proxy overhead
    forEach(callback) {
      if (ndim === 1) {
        for (let i = 0; i < shape[0]; i++) {
          callback(data[offset + i * actualStrides[0]], i);
        }
      } else if (ndim === 2) {
        for (let i = 0; i < shape[0]; i++) {
          for (let j = 0; j < shape[1]; j++) {
            callback(
              data[offset + i * actualStrides[0] + j * actualStrides[1]],
              i,
              j,
            );
          }
        }
      }
    },

    reduce(callback, initial) {
      let acc = initial;
      if (ndim === 1) {
        for (let i = 0; i < shape[0]; i++) {
          acc = callback(acc, data[offset + i * actualStrides[0]], i);
        }
      } else if (ndim === 2) {
        for (let i = 0; i < shape[0]; i++) {
          for (let j = 0; j < shape[1]; j++) {
            acc = callback(
              acc,
              data[offset + i * actualStrides[0] + j * actualStrides[1]],
              i,
              j,
            );
          }
        }
      }
      return acc;
    },

    // Map over rows (returns array of callback results)
    mapRows(callback) {
      if (ndim < 2) throw new Error("mapRows requires 2D+ array");
      const results = new Array(shape[0]);
      const rowStride = actualStrides[0];
      const colStride = actualStrides[1];
      const cols = shape[1];
      for (let i = 0; i < shape[0]; i++) {
        const rowOffset = offset + i * rowStride;
        // Pass a lightweight row accessor
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

    // Sum helper (very common operation)
    sum() {
      let total = 0;
      const len = shape.reduce((a, b) => a * b, 1);
      // For C-contiguous arrays, just iterate flat
      if (actualStrides[ndim - 1] === 1) {
        for (let i = 0; i < len; i++) {
          total += data[offset + i];
        }
      } else {
        // General case
        this.forEach((val) => {
          total += val;
        });
      }
      return total;
    },
  };

  if (ndim <= 1) {
    return new Proxy(base, {
      get(target, prop) {
        if (typeof prop === "string" && !isNaN(Number(prop))) {
          const i = Number(prop);
          return data[offset + i * actualStrides[0]];
        }
        return target[prop];
      },
    });
  }

  return new Proxy(base, {
    get(target, prop) {
      if (typeof prop === "string" && !isNaN(Number(prop))) {
        const i = Number(prop);
        const newOffset = offset + i * actualStrides[0];
        const newShape = shape.slice(1);
        const newStrides = actualStrides.slice(1);

        if (newShape.length === 0) {
          return data[newOffset];
        }

        // Creates new proxy on every access
        return ndarray(data, newShape, newStrides, newOffset);
      }
      return target[prop];
    },
  });
}

// Cached ndarray (caches sub-views in a Map)
function ndarrayCached(data, shape, strides, offset = 0) {
  const actualStrides = strides ?? computeStrides(shape);
  const ndim = shape.length;
  const length = shape.reduce((a, b) => a * b, 1);

  // Cache for sub-views (keyed by index)
  const subViewCache = new Map();

  const base = {
    flat: data,
    shape,
    strides: actualStrides,
    ndim,
    length,

    get(...indices) {
      if (indices.length !== ndim) {
        throw new Error(`Expected ${ndim} indices, got ${indices.length}`);
      }
      let idx = offset;
      for (let i = 0; i < ndim; i++) {
        idx += indices[i] * actualStrides[i];
      }
      return data[idx];
    },
  };

  if (ndim <= 1) {
    return new Proxy(base, {
      get(target, prop) {
        if (typeof prop === "string" && !isNaN(Number(prop))) {
          const i = Number(prop);
          return data[offset + i * actualStrides[0]];
        }
        return target[prop];
      },
    });
  }

  return new Proxy(base, {
    get(target, prop) {
      if (typeof prop === "string" && !isNaN(Number(prop))) {
        const i = Number(prop);

        // Check cache first
        if (subViewCache.has(i)) {
          return subViewCache.get(i);
        }

        const newOffset = offset + i * actualStrides[0];
        const newShape = shape.slice(1);
        const newStrides = actualStrides.slice(1);

        if (newShape.length === 0) {
          return data[newOffset];
        }

        const subView = ndarrayCached(data, newShape, newStrides, newOffset);
        subViewCache.set(i, subView);
        return subView;
      }
      return target[prop];
    },
  });
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
    `${name.padEnd(45)} ${nsPerOp.toFixed(1).padStart(8)} ns/op  ${(opsPerSec / 1e6).toFixed(2).padStart(6)} M ops/sec`,
  );
  return { name, nsPerOp, opsPerSec };
}

console.log("=".repeat(75));
console.log("NdArrayView Performance Benchmark");
console.log("=".repeat(75));
console.log();

// Test data: 100x100 matrix (10,000 elements)
const size = 100;
const data = new Float32Array(size * size);
for (let i = 0; i < data.length; i++) data[i] = i;

const shape = [size, size];
const strides = [size, 1];

const viewUncached = ndarray(data, shape);
const viewCached = ndarrayCached(data, shape, strides);

console.log(`Array size: ${size}x${size} = ${size * size} elements`);
console.log(`Iterations: 1,000,000 per test (unless noted)`);
console.log();

// Test 1: Single element access patterns
console.log("--- Single Element Access (middle of array) ---");
console.log();

const midI = Math.floor(size / 2);
const midJ = Math.floor(size / 2);

benchmark("arr[i][j] (uncached proxy)", () => {
  return viewUncached[midI][midJ];
});

benchmark("arr[i][j] (cached proxy)", () => {
  return viewCached[midI][midJ];
});

benchmark("arr.get(i, j)", () => {
  return viewUncached.get(midI, midJ);
});

benchmark("arr.flat[i * stride + j] (manual)", () => {
  return data[midI * size + midJ];
});

console.log();

// Test 2: Row iteration (common pattern)
console.log("--- Row Iteration (sum one row of 100 elements) [10K iters] ---");
console.log();

benchmark(
  "arr[row][j] loop (uncached)",
  () => {
    let sum = 0;
    for (let j = 0; j < size; j++) {
      sum += viewUncached[midI][j];
    }
    return sum;
  },
  10000,
);

benchmark(
  "arr[row][j] loop (cached)",
  () => {
    let sum = 0;
    for (let j = 0; j < size; j++) {
      sum += viewCached[midI][j];
    }
    return sum;
  },
  10000,
);

benchmark(
  "row = arr[i]; row[j] loop (hoist row access)",
  () => {
    let sum = 0;
    const row = viewUncached[midI];
    for (let j = 0; j < size; j++) {
      sum += row[j];
    }
    return sum;
  },
  10000,
);

benchmark(
  "arr.get(row, j) loop",
  () => {
    let sum = 0;
    for (let j = 0; j < size; j++) {
      sum += viewUncached.get(midI, j);
    }
    return sum;
  },
  10000,
);

benchmark(
  "arr.flat[row * stride + j] loop (manual)",
  () => {
    let sum = 0;
    const base = midI * size;
    for (let j = 0; j < size; j++) {
      sum += data[base + j];
    }
    return sum;
  },
  10000,
);

console.log();

// Test 3: Full matrix iteration
console.log(
  "--- Full Matrix Iteration (sum all 10,000 elements) [100 iters] ---",
);
console.log();

benchmark(
  "arr[i][j] nested loop (uncached)",
  () => {
    let sum = 0;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        sum += viewUncached[i][j];
      }
    }
    return sum;
  },
  100,
);

benchmark(
  "arr[i][j] nested loop (cached)",
  () => {
    let sum = 0;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        sum += viewCached[i][j];
      }
    }
    return sum;
  },
  100,
);

benchmark(
  "row = arr[i]; row[j] nested (hoist row)",
  () => {
    let sum = 0;
    for (let i = 0; i < size; i++) {
      const row = viewUncached[i];
      for (let j = 0; j < size; j++) {
        sum += row[j];
      }
    }
    return sum;
  },
  100,
);

benchmark(
  "arr.get(i, j) nested loop",
  () => {
    let sum = 0;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        sum += viewUncached.get(i, j);
      }
    }
    return sum;
  },
  100,
);

benchmark(
  "arr.flat direct iteration",
  () => {
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      sum += data[i];
    }
    return sum;
  },
  100,
);

console.log();

// Test 4: 3D array access
console.log("--- 3D Array Access (10x10x10 = 1000 elements) ---");
console.log();

const size3d = 10;
const data3d = new Float32Array(size3d * size3d * size3d);
for (let i = 0; i < data3d.length; i++) data3d[i] = i;

const view3d = ndarray(data3d, [size3d, size3d, size3d]);
const view3dCached = ndarrayCached(
  data3d,
  [size3d, size3d, size3d],
  [size3d * size3d, size3d, 1],
);

benchmark("arr[i][j][k] (uncached)", () => {
  return view3d[5][5][5];
});

benchmark("arr[i][j][k] (cached)", () => {
  return view3dCached[5][5][5];
});

benchmark("arr.get(i, j, k)", () => {
  return view3d.get(5, 5, 5);
});

console.log();

// Test 5: Random access pattern
console.log("--- Random Access Pattern (1000 random lookups) [1K iters] ---");
console.log();

// Pre-generate random indices
const randomIndices = [];
for (let n = 0; n < 1000; n++) {
  randomIndices.push([
    Math.floor(Math.random() * size),
    Math.floor(Math.random() * size),
  ]);
}

benchmark(
  "arr[i][j] random (uncached)",
  () => {
    let sum = 0;
    for (const [i, j] of randomIndices) {
      sum += viewUncached[i][j];
    }
    return sum;
  },
  1000,
);

benchmark(
  "arr[i][j] random (cached)",
  () => {
    let sum = 0;
    for (const [i, j] of randomIndices) {
      sum += viewCached[i][j];
    }
    return sum;
  },
  1000,
);

benchmark(
  "arr.get(i, j) random",
  () => {
    let sum = 0;
    for (const [i, j] of randomIndices) {
      sum += viewUncached.get(i, j);
    }
    return sum;
  },
  1000,
);

console.log();

// Test 6: Wide array (100 rows x 1000 columns)
console.log("--- Wide Array (100x1000 = 100K elements) ---");
console.log();

const rows = 100;
const cols = 1000;
const dataWide = new Float32Array(rows * cols);
for (let i = 0; i < dataWide.length; i++) dataWide[i] = i;

const viewWide = ndarray(dataWide, [rows, cols]);
const viewWideCached = ndarrayCached(dataWide, [rows, cols], [cols, 1]);

const midRow = Math.floor(rows / 2);

benchmark("arr[i][j] single access (uncached)", () => {
  return viewWide[midRow][500];
});

benchmark("arr[i][j] single access (cached)", () => {
  return viewWideCached[midRow][500];
});

benchmark("arr.get(i, j) single access", () => {
  return viewWide.get(midRow, 500);
});

console.log();

benchmark(
  "Sum row (1000 elems) arr[row][j] (uncached)",
  () => {
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      sum += viewWide[midRow][j];
    }
    return sum;
  },
  1000,
);

benchmark(
  "Sum row (1000 elems) arr[row][j] (cached)",
  () => {
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      sum += viewWideCached[midRow][j];
    }
    return sum;
  },
  1000,
);

benchmark(
  "Sum row (1000 elems) hoisted: row[j]",
  () => {
    let sum = 0;
    const row = viewWide[midRow];
    for (let j = 0; j < cols; j++) {
      sum += row[j];
    }
    return sum;
  },
  1000,
);

benchmark(
  "Sum row (1000 elems) arr.get(row, j)",
  () => {
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      sum += viewWide.get(midRow, j);
    }
    return sum;
  },
  1000,
);

benchmark(
  "Sum row (1000 elems) flat[base + j]",
  () => {
    let sum = 0;
    const base = midRow * cols;
    for (let j = 0; j < cols; j++) {
      sum += dataWide[base + j];
    }
    return sum;
  },
  1000,
);

console.log();

benchmark(
  "Sum all 100K elems arr[i][j] (uncached)",
  () => {
    let sum = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        sum += viewWide[i][j];
      }
    }
    return sum;
  },
  10,
);

benchmark(
  "Sum all 100K elems arr[i][j] (cached)",
  () => {
    let sum = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        sum += viewWideCached[i][j];
      }
    }
    return sum;
  },
  10,
);

benchmark(
  "Sum all 100K elems hoisted row[j]",
  () => {
    let sum = 0;
    for (let i = 0; i < rows; i++) {
      const row = viewWide[i];
      for (let j = 0; j < cols; j++) {
        sum += row[j];
      }
    }
    return sum;
  },
  10,
);

benchmark(
  "Sum all 100K elems arr.get(i, j)",
  () => {
    let sum = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        sum += viewWide.get(i, j);
      }
    }
    return sum;
  },
  10,
);

benchmark(
  "Sum all 100K elems flat iteration",
  () => {
    let sum = 0;
    for (let i = 0; i < dataWide.length; i++) {
      sum += dataWide[i];
    }
    return sum;
  },
  10,
);

console.log();

// Test 7: Callback-based iteration patterns
console.log("--- Callback-based Iteration (100x1000 = 100K elements) ---");
console.log();

benchmark(
  "arr.sum() (built-in)",
  () => {
    return viewWide.sum();
  },
  100,
);

benchmark(
  "arr.reduce((acc, v) => acc + v, 0)",
  () => {
    return viewWide.reduce((acc, v) => acc + v, 0);
  },
  100,
);

benchmark(
  "arr.forEach with external sum",
  () => {
    let sum = 0;
    viewWide.forEach((v) => {
      sum += v;
    });
    return sum;
  },
  100,
);

benchmark(
  "arr.mapRows(row => row.reduce(sum))",
  () => {
    return viewWide.mapRows((row) => row.reduce((a, v) => a + v, 0));
  },
  100,
);

console.log();

// Test 8: Row sums comparison
console.log("--- Row Sums (100 rows of 1000 elements each) ---");
console.log();

benchmark(
  "arr[i].reduce manual loop",
  () => {
    const sums = new Array(rows);
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      for (let j = 0; j < cols; j++) {
        sum += viewWide[i][j];
      }
      sums[i] = sum;
    }
    return sums;
  },
  10,
);

benchmark(
  "arr.mapRows(row => row.reduce(...))",
  () => {
    return viewWide.mapRows((row) => row.reduce((a, v) => a + v, 0));
  },
  10,
);

benchmark(
  "Manual flat with stride",
  () => {
    const sums = new Array(rows);
    for (let i = 0; i < rows; i++) {
      let sum = 0;
      const base = i * cols;
      for (let j = 0; j < cols; j++) {
        sum += dataWide[base + j];
      }
      sums[i] = sum;
    }
    return sums;
  },
  10,
);

console.log();
console.log("=".repeat(75));
console.log("Analysis:");
console.log();
console.log("1. SINGLE ACCESS: Uncached proxy ~2-3x slower than .get()");
console.log(
  "   Cached proxy brings it close to .get() for repeated same-index access",
);
console.log();
console.log(
  "2. ROW ITERATION: Hoisting row access (row = arr[i]) is key optimization",
);
console.log("   This creates proxy once, then 1D access is fast");
console.log();
console.log("3. FULL MATRIX: .get(i,j) is ~2-3x faster than arr[i][j]");
console.log("   Caching helps significantly for repeated row access");
console.log();
console.log("4. 3D ARRAYS: Cost compounds - each [] creates new proxy");
console.log("   .get(i,j,k) stays O(1) regardless of dimensionality");
console.log();
console.log("Recommendations:");
console.log("- For one-off access: arr[i][j] is fine (ergonomic)");
console.log("- For row iteration: hoist row access: `const row = arr[i]`");
console.log("- For hot inner loops: use .get(i, j) or direct .flat access");
console.log("- Caching adds complexity, recommend hoisting instead");
console.log("=".repeat(75));
