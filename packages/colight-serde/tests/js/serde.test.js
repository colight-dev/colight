import { describe, expect, it, vi } from "vitest";
import {
  packMessage,
  unpackMessage,
  collectBuffers,
  evaluateNdarray,
  ndarray,
  isNdArray,
  toNestedArray,
  inferDtype,
  asNestedArray,
  serializeNdArray,
} from "../../src/js/index";

describe("@colight/serde", () => {
  describe("packMessage / unpackMessage roundtrip", () => {
    it("should roundtrip TypedArray", () => {
      const original = { points: new Float32Array([1, 2, 3, 4, 5, 6]) };
      const [envelope, buffers] = packMessage(original);

      expect(envelope.buffer_count).toBe(1);
      expect(buffers.length).toBe(1);

      const restored = unpackMessage(envelope, buffers);
      expect(restored.points).toBeInstanceOf(Float32Array);
      expect(Array.from(restored.points)).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("should roundtrip nested structure with mixed types", () => {
      const original = {
        vertices: new Float32Array([0, 1, 2, 3, 4, 5]),
        indices: new Uint32Array([0, 1, 2]),
        name: "mesh",
        count: 42,
      };
      const [envelope, buffers] = packMessage(original);

      expect(envelope.buffer_count).toBe(2);

      const restored = unpackMessage(envelope, buffers);
      expect(restored.name).toBe("mesh");
      expect(restored.count).toBe(42);
      expect(Array.from(restored.vertices)).toEqual([0, 1, 2, 3, 4, 5]);
      expect(Array.from(restored.indices)).toEqual([0, 1, 2]);
    });

    it("should roundtrip arrays nested in arrays", () => {
      const original = {
        frames: [
          new Float32Array([1, 2]),
          new Float32Array([3, 4]),
        ],
      };
      const [envelope, buffers] = packMessage(original);

      expect(buffers.length).toBe(2);

      const restored = unpackMessage(envelope, buffers);
      expect(Array.from(restored.frames[0])).toEqual([1, 2]);
      expect(Array.from(restored.frames[1])).toEqual([3, 4]);
    });
  });

  describe("collectBuffers", () => {
    it("should distinguish ArrayBuffer (raw bytes) from TypedArray", () => {
      const buffer = new ArrayBuffer(8);
      const [serialized, buffers] = collectBuffers({ raw: buffer });

      expect(serialized.raw.__buffer_index__).toBe(0);
      expect(serialized.raw.__type__).toBeUndefined(); // Raw bytes, no type
      expect(buffers[0]).toBe(buffer);
    });

    it("should mark TypedArray as ndarray with metadata", () => {
      const arr = new Float32Array([1, 2, 3]);
      const [serialized, buffers] = collectBuffers({ data: arr });

      expect(serialized.data.__type__).toBe("ndarray");
      expect(serialized.data.dtype).toBe("float32");
      expect(serialized.data.shape).toEqual([3]);
    });

    it("should handle DataView as raw bytes", () => {
      const buffer = new ArrayBuffer(8);
      const view = new DataView(buffer);
      const [serialized] = collectBuffers({ view });

      expect(serialized.view.__buffer_index__).toBe(0);
      expect(serialized.view.__type__).toBeUndefined();
    });
  });

  describe("inferDtype", () => {
    it("should infer dtype from TypedArray", () => {
      expect(inferDtype(new Float32Array(1))).toBe("float32");
      expect(inferDtype(new Float64Array(1))).toBe("float64");
      expect(inferDtype(new Int32Array(1))).toBe("int32");
      expect(inferDtype(new Uint8Array(1))).toBe("uint8");
    });

    it("should handle ArrayBuffer as uint8", () => {
      expect(inferDtype(new ArrayBuffer(8))).toBe("uint8");
    });
  });

  describe("ndarray view", () => {
    it("should provide .get() access to 2D array", () => {
      const flat = new Float32Array([1, 2, 3, 4, 5, 6]);
      const view = ndarray(flat, [2, 3]);

      expect(view.shape).toEqual([2, 3]);
      expect(view.ndim).toBe(2);
      expect(view.flat).toBe(flat);

      // Access elements via .get()
      expect(view.get(0, 0)).toBe(1);
      expect(view.get(0, 1)).toBe(2);
      expect(view.get(0, 2)).toBe(3);
      expect(view.get(1, 0)).toBe(4);
      expect(view.get(1, 1)).toBe(5);
      expect(view.get(1, 2)).toBe(6);
    });

    it("should provide .slice() for sub-views", () => {
      const view = ndarray([1, 2, 3, 4, 5, 6], [2, 3]);
      const row0 = view.slice(0);
      expect(row0.get(0)).toBe(1);
      expect(row0.get(2)).toBe(3);

      const row1 = view.slice(1);
      expect(row1.get(0)).toBe(4);
      expect(row1.get(2)).toBe(6);
    });

    it("should handle 3D arrays", () => {
      const flat = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
      const view = ndarray(flat, [2, 2, 2]);

      expect(view.get(0, 0, 0)).toBe(1);
      expect(view.get(0, 0, 1)).toBe(2);
      expect(view.get(0, 1, 0)).toBe(3);
      expect(view.get(1, 1, 1)).toBe(8);

      // Also via nested slice
      expect(view.slice(0).slice(0).get(0)).toBe(1);
      expect(view.slice(1).slice(1).get(1)).toBe(8);
    });

    it("should handle Fortran order strides", () => {
      // 2x3 array in Fortran order: columns are contiguous
      // Logical: [[1,2,3], [4,5,6]]
      // F-order storage: [1,4,2,5,3,6]
      const flat = new Float32Array([1, 4, 2, 5, 3, 6]);
      const view = ndarray(flat, [2, 3], [1, 2]); // F-order strides

      expect(view.get(0, 0)).toBe(1);
      expect(view.get(0, 1)).toBe(2);
      expect(view.get(0, 2)).toBe(3);
      expect(view.get(1, 0)).toBe(4);
      expect(view.get(1, 1)).toBe(5);
      expect(view.get(1, 2)).toBe(6);
    });

    it("should be recognized by isNdArray", () => {
      const view = ndarray([1, 2, 3, 4], [2, 2]);
      expect(isNdArray(view)).toBe(true);
      expect(isNdArray([1, 2, 3])).toBe(false);
      expect(isNdArray(null)).toBe(false);
    });

    it("should convert to nested array via toNestedArray", () => {
      const view = ndarray([1, 2, 3, 4, 5, 6], [2, 3]);
      const nested = toNestedArray(view);

      expect(nested).toEqual([[1, 2, 3], [4, 5, 6]]);
    });
  });

  describe("evaluateNdarray options", () => {
    const makeNode = (shape) => ({
      data: new Float32Array([1, 2, 3, 4, 5, 6]).buffer,
      dtype: "float32",
      shape,
    });

    it("should return NdArrayView by default for 2D", () => {
      const result = evaluateNdarray(makeNode([2, 3]));
      expect(isNdArray(result)).toBe(true);
      expect(result.flat).toBeInstanceOf(Float32Array);
      expect(result.shape).toEqual([2, 3]);
      expect(result.strides).toEqual([3, 1]);
      // Also supports .get() access
      expect(result.get(0, 0)).toBe(1);
      expect(result.get(1, 2)).toBe(6);
    });

    it("should return nested arrays with multidimensional: 'nested'", () => {
      const result = evaluateNdarray(makeNode([2, 3]), {
        multidimensional: "nested",
      });
      expect(Array.isArray(result)).toBe(true);
      expect(result[0]).toBeInstanceOf(Float32Array);
    });

    it("should always return TypedArray for 1D", () => {
      const result = evaluateNdarray(makeNode([6]));
      expect(result).toBeInstanceOf(Float32Array);
    });
  });

  describe("asNestedArray utility", () => {
    const makeNode = (shape) => ({
      data: new Float32Array([1, 2, 3, 4, 5, 6]).buffer,
      dtype: "float32",
      shape,
    });

    it("should convert NdArrayView to nested arrays", () => {
      const view = evaluateNdarray(makeNode([2, 3]));
      const nested = asNestedArray(view);
      expect(nested).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });
  });

  describe("BigInt handling", () => {
    it("should warn on BigInt overflow", () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const bigValue = BigInt(Number.MAX_SAFE_INTEGER) + 10n;
      const data = new BigInt64Array([bigValue]).buffer;
      const node = { data, dtype: "int64", shape: [1] };

      evaluateNdarray(node);

      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("exceeds Number.MAX_SAFE_INTEGER"),
      );

      warnSpy.mockRestore();
    });

    it("should not warn for safe BigInt values", () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const safeValue = 42n;
      const data = new BigInt64Array([safeValue]).buffer;
      const node = { data, dtype: "int64", shape: [1] };

      const result = evaluateNdarray(node);
      expect(result[0]).toBe(42);
      expect(warnSpy).not.toHaveBeenCalled();

      warnSpy.mockRestore();
    });
  });

  describe("buffer_count validation", () => {
    it("should throw on buffer count mismatch", () => {
      const [envelope, buffers] = packMessage({ x: new Float32Array([1, 2]) });
      envelope.buffer_count = 5; // Wrong!

      expect(() => unpackMessage(envelope, buffers)).toThrow(
        "buffer_count mismatch",
      );
    });
  });

  describe("serializeNdArray", () => {
    it("should serialize TypedArray to wire format", () => {
      const arr = new Float32Array([1, 2, 3, 4]);
      const [ref, buffer] = serializeNdArray(arr);

      expect(ref.__type__).toBe("ndarray");
      expect(ref.dtype).toBe("float32");
      expect(ref.shape).toEqual([4]);
      expect(ref.order).toBe("C");
      expect(ref.strides).toEqual([4]); // 4 bytes per float32
      expect(buffer).toBe(arr.buffer);
    });

    it("should serialize NdArrayView with full shape info", () => {
      const flat = new Float32Array([1, 2, 3, 4, 5, 6]);
      const view = ndarray(flat, [2, 3]);
      const [ref, buffer] = serializeNdArray(view);

      expect(ref.__type__).toBe("ndarray");
      expect(ref.dtype).toBe("float32");
      expect(ref.shape).toEqual([2, 3]);
      expect(ref.order).toBe("C");
      // strides in bytes: [3*4, 1*4] = [12, 4]
      expect(ref.strides).toEqual([12, 4]);
      expect(buffer).toBe(flat.buffer);
    });

    it("should serialize different dtypes correctly", () => {
      const tests = [
        { arr: new Int32Array([1, 2]), dtype: "int32", stride: 4 },
        { arr: new Uint8Array([1, 2]), dtype: "uint8", stride: 1 },
        { arr: new Float64Array([1, 2]), dtype: "float64", stride: 8 },
      ];

      for (const { arr, dtype, stride } of tests) {
        const [ref] = serializeNdArray(arr);
        expect(ref.dtype).toBe(dtype);
        expect(ref.strides).toEqual([stride]);
      }
    });

    it("should produce wire format compatible with Python deserialize", () => {
      // This test verifies the format matches what Python expects
      const flat = new Float32Array([1, 2, 3, 4, 5, 6]);
      const view = ndarray(flat, [2, 3]);
      const [ref, buffer] = serializeNdArray(view);

      // Add buffer index as would happen during collection
      const wireRef = { ...ref, __buffer_index__: 0 };

      // Should match Python's expected format exactly
      expect(wireRef).toEqual({
        __type__: "ndarray",
        __buffer_index__: 0,
        dtype: "float32",
        shape: [2, 3],
        order: "C",
        strides: [12, 4],
      });
    });
  });
});
