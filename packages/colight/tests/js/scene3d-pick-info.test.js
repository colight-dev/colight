/**
 * Unit tests for scene3d picking logic.
 *
 * These tests verify the CPU-side picking logic without requiring a GPU.
 * They test:
 * - Normal encoding/decoding between GPU and CPU
 * - Face detection for cuboids
 * - Local/world coordinate transforms
 * - Pick info construction
 */

import { describe, it, expect } from "vitest";
import {
  decodeNormalFromBytes,
  encodeNormalToBytes,
  detectCuboidFace,
  worldNormalToLocal,
  worldPositionToLocal,
  readQuat,
  FACE_NAMES,
} from "../../src/js/scene3d/pick-info.ts";
import { readVec3, sub as subVec3, dot as dotVec3 } from "../../src/js/scene3d/vec3.ts";
import { createCubeGeometry } from "../../src/js/scene3d/geometry.ts";

describe("scene3d pick-info", () => {
  describe("normal encoding/decoding", () => {
    it("should round-trip encode and decode normals", () => {
      const testNormals = [
        [1, 0, 0], // +X
        [-1, 0, 0], // -X
        [0, 1, 0], // +Y
        [0, -1, 0], // -Y
        [0, 0, 1], // +Z
        [0, 0, -1], // -Z
      ];

      for (const normal of testNormals) {
        const encoded = encodeNormalToBytes(normal);
        const decoded = decodeNormalFromBytes(new Uint8Array(encoded));

        // Allow small epsilon due to quantization
        expect(decoded[0]).toBeCloseTo(normal[0], 1);
        expect(decoded[1]).toBeCloseTo(normal[1], 1);
        expect(decoded[2]).toBeCloseTo(normal[2], 1);
      }
    });

    it("should encode axis-aligned normals to expected byte values", () => {
      // +X normal [1, 0, 0] should encode to [255, 128, 128] (approximately)
      const plusX = encodeNormalToBytes([1, 0, 0]);
      expect(plusX[0]).toBe(255); // 1.0 * 0.5 + 0.5 = 1.0 -> 255
      expect(plusX[1]).toBe(128); // 0.0 * 0.5 + 0.5 = 0.5 -> 128
      expect(plusX[2]).toBe(128);

      // -X normal [-1, 0, 0] should encode to [0, 128, 128]
      const minusX = encodeNormalToBytes([-1, 0, 0]);
      expect(minusX[0]).toBe(0); // -1.0 * 0.5 + 0.5 = 0.0 -> 0
      expect(minusX[1]).toBe(128);
      expect(minusX[2]).toBe(128);
    });

    it("should decode GPU output bytes correctly", () => {
      // Simulate GPU output for +Z normal
      const gpuBytes = new Uint8Array([128, 128, 255]); // encoded [0, 0, 1]
      const decoded = decodeNormalFromBytes(gpuBytes);

      expect(decoded[0]).toBeCloseTo(0, 1);
      expect(decoded[1]).toBeCloseTo(0, 1);
      expect(decoded[2]).toBeCloseTo(1, 1);
    });
  });

  describe("face detection", () => {
    it("should detect +X face from positive X normal", () => {
      const result = detectCuboidFace([0.99, 0.01, 0.0]);
      expect(result.index).toBe(0);
      expect(result.name).toBe("+x");
    });

    it("should detect -X face from negative X normal", () => {
      const result = detectCuboidFace([-0.99, 0.01, 0.0]);
      expect(result.index).toBe(1);
      expect(result.name).toBe("-x");
    });

    it("should detect +Y face from positive Y normal", () => {
      const result = detectCuboidFace([0.0, 0.99, 0.01]);
      expect(result.index).toBe(2);
      expect(result.name).toBe("+y");
    });

    it("should detect -Y face from negative Y normal", () => {
      const result = detectCuboidFace([0.0, -0.99, 0.01]);
      expect(result.index).toBe(3);
      expect(result.name).toBe("-y");
    });

    it("should detect +Z face from positive Z normal", () => {
      const result = detectCuboidFace([0.0, 0.01, 0.99]);
      expect(result.index).toBe(4);
      expect(result.name).toBe("+z");
    });

    it("should detect -Z face from negative Z normal", () => {
      const result = detectCuboidFace([0.0, 0.01, -0.99]);
      expect(result.index).toBe(5);
      expect(result.name).toBe("-z");
    });

    it("should handle slightly off-axis normals", () => {
      // A normal that's mostly +Z but slightly tilted
      const result = detectCuboidFace([0.1, 0.2, 0.97]);
      expect(result.name).toBe("+z");
    });
  });

  describe("coordinate transforms", () => {
    it("should transform world normal to local space with identity quaternion", () => {
      const worldNormal = [1, 0, 0];
      const identityQuat = [0, 0, 0, 1];

      const localNormal = worldNormalToLocal(worldNormal, identityQuat);

      expect(localNormal[0]).toBeCloseTo(1, 5);
      expect(localNormal[1]).toBeCloseTo(0, 5);
      expect(localNormal[2]).toBeCloseTo(0, 5);
    });

    it("should transform world normal with 90 degree Z rotation", () => {
      // Quaternion for 90 degree rotation around Z axis
      const angle = Math.PI / 2;
      const quat = [0, 0, Math.sin(angle / 2), Math.cos(angle / 2)];

      // When object is rotated +90 deg around Z, its local +X now points to world +Y
      // So a world +X normal, when transformed to local space via inverse rotation,
      // becomes local -Y (rotating -90 deg around Z)
      const worldNormal = [1, 0, 0];
      const localNormal = worldNormalToLocal(worldNormal, quat);

      expect(localNormal[0]).toBeCloseTo(0, 5);
      expect(localNormal[1]).toBeCloseTo(-1, 5);
      expect(localNormal[2]).toBeCloseTo(0, 5);
    });

    it("should transform world position to local space", () => {
      const worldPos = [2, 3, 4];
      const center = [1, 1, 1];
      const identityQuat = [0, 0, 0, 1];

      const localPos = worldPositionToLocal(worldPos, center, identityQuat);

      expect(localPos[0]).toBeCloseTo(1, 5);
      expect(localPos[1]).toBeCloseTo(2, 5);
      expect(localPos[2]).toBeCloseTo(3, 5);
    });
  });

  describe("vector utilities", () => {
    it("readVec3 should extract vector at index", () => {
      const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);

      expect(readVec3(data, 0)).toEqual([1, 2, 3]);
      expect(readVec3(data, 1)).toEqual([4, 5, 6]);
      expect(readVec3(data, 2)).toEqual([7, 8, 9]);
    });

    it("readQuat should extract quaternion at index", () => {
      const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

      expect(readQuat(data, 0)).toEqual([1, 2, 3, 4]);
      expect(readQuat(data, 1)).toEqual([5, 6, 7, 8]);
    });

    it("subVec3 should subtract vectors", () => {
      expect(subVec3([3, 4, 5], [1, 1, 1])).toEqual([2, 3, 4]);
    });

    it("dotVec3 should compute dot product", () => {
      expect(dotVec3([1, 0, 0], [1, 0, 0])).toBe(1);
      expect(dotVec3([1, 0, 0], [0, 1, 0])).toBe(0);
      expect(dotVec3([1, 2, 3], [4, 5, 6])).toBe(32);
    });
  });

  describe("face detection integration", () => {
    it("should correctly identify all 6 faces of axis-aligned cube", () => {
      const identityQuat = [0, 0, 0, 1];

      // Test each face with its expected normal
      const testCases = [
        { worldNormal: [1, 0, 0], expectedFace: "+x" },
        { worldNormal: [-1, 0, 0], expectedFace: "-x" },
        { worldNormal: [0, 1, 0], expectedFace: "+y" },
        { worldNormal: [0, -1, 0], expectedFace: "-y" },
        { worldNormal: [0, 0, 1], expectedFace: "+z" },
        { worldNormal: [0, 0, -1], expectedFace: "-z" },
      ];

      for (const { worldNormal, expectedFace } of testCases) {
        const localNormal = worldNormalToLocal(worldNormal, identityQuat);
        const face = detectCuboidFace(localNormal);
        expect(face.name).toBe(expectedFace);
      }
    });

    it("should correctly identify faces of rotated cube", () => {
      // 90 degree rotation around Z axis
      const angle = Math.PI / 2;
      const zRotQuat = [0, 0, Math.sin(angle / 2), Math.cos(angle / 2)];

      // When cube is rotated +90 degrees around Z:
      // - Local +X now points to world +Y
      // - So a world +Y normal, in local space, is the local +X direction
      // This means hitting the face that faces +Y in world = the local +X face

      const worldNormal = [0, 1, 0]; // +Y in world
      const localNormal = worldNormalToLocal(worldNormal, zRotQuat);
      const face = detectCuboidFace(localNormal);

      // World +Y transforms to local +X (inverse of the rotation)
      expect(face.name).toBe("+x");
    });
  });

  describe("GPU output simulation", () => {
    it("should handle typical GPU picking output", () => {
      // Simulate what the GPU would output for a +Z face hit
      const gpuNormalBytes = new Uint8Array([128, 128, 255]); // [0, 0, 1] encoded

      const normal = decodeNormalFromBytes(gpuNormalBytes);
      const face = detectCuboidFace(normal);

      expect(face.name).toBe("+z");
    });

    it("should handle GPU output with quantization noise", () => {
      // Real GPU output might have slight variations due to interpolation
      // Simulate a +X normal with some noise: [254, 129, 127]
      const gpuNormalBytes = new Uint8Array([254, 129, 127]);

      const normal = decodeNormalFromBytes(gpuNormalBytes);
      const face = detectCuboidFace(normal);

      expect(face.name).toBe("+x");
    });
  });

  describe("cube geometry", () => {
    it("should have correct normals for each face", () => {
      const cube = createCubeGeometry();
      const { vertexData } = cube;

      // Cube has 6 faces * 4 vertices = 24 vertices
      // Each vertex has 6 floats: x, y, z, nx, ny, nz
      expect(vertexData.length).toBe(24 * 6);

      // Expected normals for each face (4 vertices per face)
      const expectedFaceNormals = [
        [1, 0, 0], // +X (vertices 0-3)
        [-1, 0, 0], // -X (vertices 4-7)
        [0, 1, 0], // +Y (vertices 8-11)
        [0, -1, 0], // -Y (vertices 12-15)
        [0, 0, 1], // +Z (vertices 16-19)
        [0, 0, -1], // -Z (vertices 20-23)
      ];

      for (let face = 0; face < 6; face++) {
        const expected = expectedFaceNormals[face];
        for (let v = 0; v < 4; v++) {
          const vertexIdx = face * 4 + v;
          const nx = vertexData[vertexIdx * 6 + 3];
          const ny = vertexData[vertexIdx * 6 + 4];
          const nz = vertexData[vertexIdx * 6 + 5];

          expect(nx).toBeCloseTo(expected[0], 5);
          expect(ny).toBeCloseTo(expected[1], 5);
          expect(nz).toBeCloseTo(expected[2], 5);
        }
      }
    });

    it("should have unit-length normals", () => {
      const cube = createCubeGeometry();
      const { vertexData } = cube;

      for (let i = 0; i < 24; i++) {
        const nx = vertexData[i * 6 + 3];
        const ny = vertexData[i * 6 + 4];
        const nz = vertexData[i * 6 + 5];
        const length = Math.sqrt(nx * nx + ny * ny + nz * nz);

        expect(length).toBeCloseTo(1, 5);
      }
    });
  });
});
