/**
 * Tests for scene3d coercion functionality.
 *
 * These tests verify that:
 * 1. Flat typed arrays (from Python numpy arrays) are coerced to Float32Array
 * 2. Regular arrays are coerced to Float32Array
 * 3. Singular props are resolved to plural form (center → centers)
 * 4. Scalar values are expanded (half_size: 0.5 → [0.5, 0.5, 0.5])
 */

import { describe, it, expect, vi, afterEach } from "vitest";
import { coerceToFloat32 } from "../../../src/js/scene3d/arrayUtils";

// Import coercion helpers and specs using dynamic imports to avoid circular dependencies
const { coerceFloat32Fields, resolveSingular, expandScalar } = await import(
  "../../../src/js/scene3d/primitives/define"
);

const { pointCloudSpec, ellipsoidSpec, cuboidSpec, boundingBoxSpec } =
  await import("../../../src/js/scene3d/components");

const { compileScene } = await import("../../../src/js/scene3d/compiler");

describe("coerceToFloat32", () => {
  it("returns a flat Float32Array unchanged", () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);

    const result = coerceToFloat32(data);

    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it("converts an Int32Array to Float32Array", () => {
    const data = new Int32Array([1, 2, 3, 4, 5, 6]);

    const result = coerceToFloat32(data);

    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it("converts regular array to Float32Array", () => {
    const arr = [1, 2, 3, 4, 5, 6];

    const result = coerceToFloat32(arr);

    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it("flattens nested array to Float32Array", () => {
    const arr = [
      [1, 2, 3],
      [4, 5, 6],
    ];

    const result = coerceToFloat32(arr);

    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
  });

  it("converts other TypedArray to Float32Array", () => {
    const data = new Int16Array([1, 2, 3]);

    const result = coerceToFloat32(data);

    expect(result).toBeInstanceOf(Float32Array);
  });

  it("returns Float32Array unchanged", () => {
    const data = new Float32Array([1, 2, 3]);

    const result = coerceToFloat32(data);

    expect(result).toBe(data); // Same reference
  });

  it("returns non-array values unchanged", () => {
    expect(coerceToFloat32(42)).toBe(42);
    expect(coerceToFloat32("hello")).toBe("hello");
    expect(coerceToFloat32(null)).toBe(null);
    expect(coerceToFloat32(undefined)).toBe(undefined);
  });
});

describe("coerceFloat32Fields", () => {
  it("coerces multiple fields", () => {
    const obj = {
      centers: [0, 0, 0, 1, 1, 1],
      colors: [1, 0, 0, 0, 1, 0],
      name: "test",
    };

    const result = coerceFloat32Fields(obj, ["centers", "colors"]);

    expect(result.centers).toBeInstanceOf(Float32Array);
    expect(result.colors).toBeInstanceOf(Float32Array);
    expect(result.name).toBe("test"); // unchanged
  });

  it("handles flat typed-array fields", () => {
    const centersData = new Float32Array([0, 0, 0, 1, 1, 1]);
    const obj = {
      centers: centersData,
    };

    const result = coerceFloat32Fields(obj, ["centers"]);

    expect(result.centers).toBeInstanceOf(Float32Array);
    expect(result.centers).toEqual(new Float32Array([0, 0, 0, 1, 1, 1]));
  });

  it("skips undefined fields", () => {
    const obj = {
      centers: [0, 0, 0],
      colors: undefined,
    };

    const result = coerceFloat32Fields(obj, ["centers", "colors"]);

    expect(result.centers).toBeInstanceOf(Float32Array);
    expect(result.colors).toBeUndefined();
  });
});

describe("resolveSingular", () => {
  it("wraps singular value in array as plural", () => {
    const props = { center: [1, 2, 3] };

    const result = resolveSingular(props, "center", "centers");

    expect(result.centers).toEqual([[1, 2, 3]]);
    expect(result.center).toBeUndefined();
  });

  it("removes singular if plural already exists", () => {
    const props = {
      center: [1, 2, 3],
      centers: [
        [4, 5, 6],
        [7, 8, 9],
      ],
    };

    const result = resolveSingular(props, "center", "centers");

    expect(result.centers).toEqual([
      [4, 5, 6],
      [7, 8, 9],
    ]);
    expect(result.center).toBeUndefined();
  });

  it("returns unchanged if neither singular nor plural exists", () => {
    const props = { color: [1, 0, 0] };

    const result = resolveSingular(props, "center", "centers");

    expect(result).toEqual(props);
  });
});

describe("expandScalar", () => {
  it("expands scalar to vec3", () => {
    const props = { half_size: 0.5 };

    const result = expandScalar(props, "half_size");

    expect(result.half_size).toEqual([0.5, 0.5, 0.5]);
  });

  it("leaves array unchanged", () => {
    const props = { half_size: [0.1, 0.2, 0.3] };

    const result = expandScalar(props, "half_size");

    expect(result.half_size).toEqual([0.1, 0.2, 0.3]);
  });

  it("leaves undefined unchanged", () => {
    const props = { color: [1, 0, 0] };

    const result = expandScalar(props, "half_size");

    expect(result.half_size).toBeUndefined();
  });
});

describe("primitive coerce functions", () => {
  describe("PointCloud", () => {
    it("coerces flat typed-array centers", () => {
      const centersData = new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]);
      const props = {
        centers: centersData,
      };

      const result = pointCloudSpec.coerce!(props);

      expect(result.centers).toBeInstanceOf(Float32Array);
      expect(result.type).toBe("PointCloud");
    });

    it("resolves singular center to centers", () => {
      const props = {
        center: [1, 2, 3],
      };

      const result = pointCloudSpec.coerce!(props);

      expect(result.centers).toBeInstanceOf(Float32Array);
      expect(result.centers).toEqual(new Float32Array([1, 2, 3]));
    });
  });

  describe("Ellipsoid", () => {
    it("coerces flat typed-array centers and half_sizes", () => {
      const centersData = new Float32Array([0, 0, 0]);
      const halfSizesData = new Float32Array([0.5, 0.5, 0.5]);
      const props = {
        centers: centersData,
        half_sizes: halfSizesData,
      };

      const result = ellipsoidSpec.coerce!(props);

      expect(result.centers).toBeInstanceOf(Float32Array);
      expect(result.half_sizes).toBeInstanceOf(Float32Array);
      expect(result.type).toBe("Ellipsoid");
    });

    it("expands scalar half_size", () => {
      const props = {
        centers: [[0, 0, 0]],
        half_size: 0.5,
      };

      const result = ellipsoidSpec.coerce!(props);

      expect(result.half_size).toEqual([0.5, 0.5, 0.5]);
    });
  });

  describe("Cuboid", () => {
    it("coerces flat typed-array centers", () => {
      const centersData = new Float32Array([0, 0, 0, 1, 1, 1]);
      const props = {
        centers: centersData,
        half_size: 0.1,
      };

      const result = cuboidSpec.coerce!(props);

      expect(result.centers).toBeInstanceOf(Float32Array);
      expect(result.half_size).toEqual([0.1, 0.1, 0.1]);
      expect(result.type).toBe("Cuboid");
    });

    it("resolves singular center and expands scalar half_size", () => {
      const props = {
        center: [0, 0, 0],
        half_size: 0.1,
      };

      const result = cuboidSpec.coerce!(props);

      expect(result.centers).toBeInstanceOf(Float32Array);
      expect(result.centers).toEqual(new Float32Array([0, 0, 0]));
      expect(result.half_size).toEqual([0.1, 0.1, 0.1]);
    });
  });

  describe("BoundingBox", () => {
    it("coerces flat typed-array centers", () => {
      const centersData = new Float32Array([0, 0, 0]);
      const props = {
        centers: centersData,
        half_size: [1, 1, 1],
      };

      const result = boundingBoxSpec.coerce!(props);

      expect(result.centers).toBeInstanceOf(Float32Array);
      expect(result.type).toBe("BoundingBox");
    });
  });
});

// =============================================================================
// Python boundary convention: multi-word data props are snake_case
// =============================================================================

describe("snake_case data props at the Python boundary", () => {
  it("Ellipsoid consumes half_sizes and quaternions", () => {
    const result = ellipsoidSpec.coerce!({
      centers: [[0, 0, 0]],
      half_sizes: [0.1, 0.2, 0.3],
      quaternions: [1, 0, 0, 0],
    });

    expect(result.half_sizes).toEqual(new Float32Array([0.1, 0.2, 0.3]));
    expect(result.quaternions).toEqual(new Float32Array([1, 0, 0, 0]));
  });

  it("Ellipsoid consumes fill_mode and switches to EllipsoidAxes", () => {
    const result = ellipsoidSpec.coerce!({
      centers: [[0, 0, 0]],
      fill_mode: "MajorWireframe",
    });

    expect(result.type).toBe("EllipsoidAxes");
  });

  it("Cuboid consumes scalar half_size", () => {
    const result = cuboidSpec.coerce!({
      centers: [[0, 0, 0]],
      half_size: 0.25,
    });

    expect(result.half_size).toEqual([0.25, 0.25, 0.25]);
  });

  it("boundingBoxSpec consumes half_size arrays", () => {
    const result = boundingBoxSpec.coerce!({
      centers: [[0, 0, 0]],
      half_sizes: [1, 2, 3],
    });

    expect(result.half_sizes).toEqual(new Float32Array([1, 2, 3]));
  });
});

// =============================================================================
// Unknown props are LOUD (warn instead of silently dropping)
// =============================================================================

describe("unknown prop warnings", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("warns when a camelCased data prop reaches the compiler", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    compileScene([
      {
        type: "Ellipsoid",
        centers: new Float32Array([0, 0, 0]),
        halfSize: 0.1, // wrong spelling — data props are snake_case
      } as any,
    ]);

    const messages = warn.mock.calls.map((call) => String(call[0]));
    expect(
      messages.some((m) => m.includes("Ellipsoid") && m.includes("halfSize")),
    ).toBe(true);
  });

  it("does not warn for valid snake_case data and camelCase framework props", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    compileScene([
      {
        type: "Cuboid",
        centers: new Float32Array([0, 0, 0]),
        half_size: 0.25,
        quaternions: new Float32Array([1, 0, 0, 0]),
        hoverProps: { outline: true, outlineColor: [1, 0, 0] },
        pickingScale: 2.0,
        onHover: () => {},
        decorations: [{ indexes: [0], outlineWidth: 3 }],
      } as any,
    ]);

    expect(warn).not.toHaveBeenCalled();
  });

  it("warns only once for the same unknown prop", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const component = {
      type: "PointCloud",
      centers: new Float32Array([0, 0, 0]),
      pointSizes: [1, 2, 3],
    } as any;
    compileScene([component]);
    compileScene([component]);

    const matching = warn.mock.calls.filter((call) =>
      String(call[0]).includes("pointSizes"),
    );
    expect(matching.length).toBe(1);
  });
});
