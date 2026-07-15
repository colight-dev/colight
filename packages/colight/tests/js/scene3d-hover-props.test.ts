/**
 * Unit tests for hoverProps and pickingScale functionality.
 *
 * Tests that hoverProps correctly modifies render data when an instance is hovered,
 * and that pickingScale correctly inflates picking geometry.
 */

import { describe, it, expect } from "vitest";
import {
  buildRenderData,
  buildPickingData,
  pointCloudSpec,
  ellipsoidSpec,
  lineBeamsSpec,
  PointCloudComponentConfig,
  EllipsoidComponentConfig,
  LineBeamsComponentConfig,
} from "../../src/js/scene3d/components";

describe("scene3d hoverProps", () => {
  describe("PointCloud with hoverProps", () => {
    const createPointCloudConfig = (
      hoverProps?: PointCloudComponentConfig["hoverProps"],
    ): PointCloudComponentConfig => ({
      type: "PointCloud",
      centers: new Float32Array([0, 0, 0, 1, 1, 1]), // 2 points
      color: [1, 0, 0] as [number, number, number], // Red
      size: 10,
      alpha: 1.0,
      hoverProps,
    });

    it("should apply hoverProps color when instance is hovered", () => {
      const config = createPointCloudConfig({
        color: [0, 1, 0], // Green on hover
      });

      const buffer = new Float32Array(pointCloudSpec.floatsPerInstance * 2);

      // PointCloud layout: position(3) + size(1) + color(3) + alpha(1)
      // So color is at indices 4, 5, 6

      // Build without hover
      buildRenderData(config, pointCloudSpec, buffer, 0, undefined, undefined);
      const normalColor = [buffer[4], buffer[5], buffer[6]];

      // Build with hover on instance 0
      buildRenderData(config, pointCloudSpec, buffer, 0, undefined, 0);
      const hoveredColor = [buffer[4], buffer[5], buffer[6]];

      // Normal color should be red
      expect(normalColor).toEqual([1, 0, 0]);
      // Hovered color should be green
      expect(hoveredColor).toEqual([0, 1, 0]);
    });

    it("should apply hoverProps scale when instance is hovered", () => {
      const config = createPointCloudConfig({
        scale: 2.0, // Double size on hover
      });

      const buffer = new Float32Array(pointCloudSpec.floatsPerInstance * 2);

      // PointCloud layout: position(3) + size(1) + color(3) + alpha(1)
      // Size is at index 3

      // Build without hover
      buildRenderData(config, pointCloudSpec, buffer, 0, undefined, undefined);
      const normalSize = buffer[3];

      // Build with hover on instance 0
      buildRenderData(config, pointCloudSpec, buffer, 0, undefined, 0);
      const hoveredSize = buffer[3];

      expect(hoveredSize).toBe(normalSize * 2);
    });

    it("should apply hoverProps alpha when instance is hovered", () => {
      const config = createPointCloudConfig({
        alpha: 0.5, // Half opacity on hover
      });

      const buffer = new Float32Array(pointCloudSpec.floatsPerInstance * 2);

      // PointCloud layout: position(3) + size(1) + color(3) + alpha(1)
      // Alpha is at index 7

      // Build without hover
      buildRenderData(config, pointCloudSpec, buffer, 0, undefined, undefined);
      const normalAlpha = buffer[7];

      // Build with hover on instance 0
      buildRenderData(config, pointCloudSpec, buffer, 0, undefined, 0);
      const hoveredAlpha = buffer[7];

      expect(normalAlpha).toBe(1.0);
      expect(hoveredAlpha).toBe(0.5);
    });

    it("should only affect the hovered instance", () => {
      const config = createPointCloudConfig({
        color: [0, 1, 0], // Green on hover
      });

      const buffer = new Float32Array(pointCloudSpec.floatsPerInstance * 2);

      // PointCloud layout: position(3) + size(1) + color(3) + alpha(1)
      // Color is at indices 4, 5, 6

      // Build with hover on instance 0
      buildRenderData(config, pointCloudSpec, buffer, 0, undefined, 0);

      // Instance 0 should be green
      const instance0Color = [buffer[4], buffer[5], buffer[6]];
      // Instance 1 should remain red
      const instance1Color = [
        buffer[pointCloudSpec.floatsPerInstance + 4],
        buffer[pointCloudSpec.floatsPerInstance + 5],
        buffer[pointCloudSpec.floatsPerInstance + 6],
      ];

      expect(instance0Color).toEqual([0, 1, 0]);
      expect(instance1Color).toEqual([1, 0, 0]);
    });

    it("should not modify data when no hoverProps defined", () => {
      const config = createPointCloudConfig(undefined);

      const buffer1 = new Float32Array(pointCloudSpec.floatsPerInstance * 2);
      const buffer2 = new Float32Array(pointCloudSpec.floatsPerInstance * 2);

      // Build without hover
      buildRenderData(config, pointCloudSpec, buffer1, 0, undefined, undefined);
      // Build with hover (but no hoverProps)
      buildRenderData(config, pointCloudSpec, buffer2, 0, undefined, 0);

      expect(Array.from(buffer1)).toEqual(Array.from(buffer2));
    });
  });

  describe("Ellipsoid with hoverProps", () => {
    const createEllipsoidConfig = (
      hoverProps?: EllipsoidComponentConfig["hoverProps"],
    ): EllipsoidComponentConfig => ({
      type: "Ellipsoid",
      centers: new Float32Array([0, 0, 0]),
      half_size: [1, 1, 1] as [number, number, number],
      color: [1, 0, 0] as [number, number, number],
      alpha: 1.0,
      hoverProps,
    });

    it("should apply hoverProps to ellipsoid", () => {
      const config = createEllipsoidConfig({
        color: [0, 0, 1], // Blue on hover
        scale: 1.5,
      });

      const buffer = new Float32Array(ellipsoidSpec.floatsPerInstance);

      // Build with hover on instance 0
      buildRenderData(config, ellipsoidSpec, buffer, 0, undefined, 0);

      // Color is at indices 7, 8, 9 for Ellipsoid
      const hoveredColor = [buffer[7], buffer[8], buffer[9]];
      expect(hoveredColor).toEqual([0, 0, 1]);

      // Half sizes (3, 4, 5) should be scaled by 1.5
      const hoveredHalfSizes = [buffer[3], buffer[4], buffer[5]];
      expect(hoveredHalfSizes).toEqual([1.5, 1.5, 1.5]);
    });
  });

  describe("hoverProps with decorations", () => {
    it("should apply hoverProps on top of decorations", () => {
      const config: PointCloudComponentConfig = {
        type: "PointCloud",
        centers: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]), // 3 points
        color: [1, 0, 0] as [number, number, number], // Red base
        size: 10,
        decorations: [
          { indexes: [1], color: [0, 0, 1] }, // Instance 1 is blue via decoration
        ],
        hoverProps: {
          color: [0, 1, 0], // Green on hover
        },
      };

      const buffer = new Float32Array(pointCloudSpec.floatsPerInstance * 3);

      // PointCloud layout: position(3) + size(1) + color(3) + alpha(1)
      // Color is at indices 4, 5, 6

      // Build with hover on instance 1 (which also has a decoration)
      buildRenderData(config, pointCloudSpec, buffer, 0, undefined, 1);

      // Instance 0: base red
      const inst0Color = [buffer[4], buffer[5], buffer[6]];
      expect(inst0Color).toEqual([1, 0, 0]);

      // Instance 1: hoverProps should override decoration
      const floatsPerInst = pointCloudSpec.floatsPerInstance;
      const inst1Color = [
        buffer[floatsPerInst + 4],
        buffer[floatsPerInst + 5],
        buffer[floatsPerInst + 6],
      ];
      expect(inst1Color).toEqual([0, 1, 0]);

      // Instance 2: base red
      const inst2Color = [
        buffer[2 * floatsPerInst + 4],
        buffer[2 * floatsPerInst + 5],
        buffer[2 * floatsPerInst + 6],
      ];
      expect(inst2Color).toEqual([1, 0, 0]);
    });
  });
});

describe("scene3d pickingScale", () => {
  describe("PointCloud with pickingScale", () => {
    it("should scale picking geometry when pickingScale is set", () => {
      const config: PointCloudComponentConfig = {
        type: "PointCloud",
        centers: new Float32Array([0, 0, 0, 1, 1, 1]), // 2 points
        size: 10,
        pickingScale: 2.0, // Double the picking size
      };

      const buffer = new Float32Array(pointCloudSpec.floatsPerPicking * 2);

      // Build picking data
      buildPickingData(config, pointCloudSpec, buffer, 0, 0, undefined);

      // PointCloud picking layout: position(3) + size(1) + pickID(1)
      // Size is at index 3
      const pickingSize = buffer[3];

      // Should be base size * pickingScale = 10 * 2 = 20
      expect(pickingSize).toBe(20);
    });

    it("should not affect picking geometry when pickingScale is not set", () => {
      const config: PointCloudComponentConfig = {
        type: "PointCloud",
        centers: new Float32Array([0, 0, 0]),
        size: 10,
        // No pickingScale
      };

      const buffer = new Float32Array(pointCloudSpec.floatsPerPicking);

      buildPickingData(config, pointCloudSpec, buffer, 0, 0, undefined);

      // Size should be unchanged
      expect(buffer[3]).toBe(10);
    });
  });

  describe("Ellipsoid with pickingScale", () => {
    it("should scale ellipsoid half_sizes in picking geometry", () => {
      const config: EllipsoidComponentConfig = {
        type: "Ellipsoid",
        centers: new Float32Array([0, 0, 0]),
        half_size: [1, 2, 3] as [number, number, number],
        pickingScale: 3.0, // Triple the picking size
      };

      const buffer = new Float32Array(ellipsoidSpec.floatsPerPicking);

      buildPickingData(config, ellipsoidSpec, buffer, 0, 0, undefined);

      // Ellipsoid picking layout: position(3) + half_size(3) + quaternion(4) + pickID(1)
      // Half sizes are at indices 3, 4, 5
      const pickingHalfSizes = [buffer[3], buffer[4], buffer[5]];

      // Should be base half_size * pickingScale
      expect(pickingHalfSizes).toEqual([3, 6, 9]);
    });
  });

  describe("LineBeams with pickingScale", () => {
    it("should scale line beam size in picking geometry", () => {
      const config: LineBeamsComponentConfig = {
        type: "LineBeams",
        // Two points on same line (index 0)
        points: new Float32Array([0, 0, 0, 0, 1, 1, 1, 0]),
        size: 0.02, // Thin line
        pickingScale: 5.0, // 5x larger for easier clicking
      };

      const buffer = new Float32Array(lineBeamsSpec.floatsPerPicking);

      buildPickingData(config, lineBeamsSpec, buffer, 0, 0, undefined);

      // LineBeams picking layout: start(3) + end(3) + size(1) + pickID(1)
      // Size is at index 6
      const pickingSize = buffer[6];

      // Should be base size * pickingScale = 0.02 * 5 = 0.1
      expect(pickingSize).toBeCloseTo(0.1);
    });
  });
});
