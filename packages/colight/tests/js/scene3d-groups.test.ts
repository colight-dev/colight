/**
 * Unit tests for the Groups (hierarchical transforms) functionality.
 *
 * Tests that groups correctly transform child components and compose transforms.
 */

import { describe, it, expect } from "vitest";
import {
  flattenGroups,
  hasGroups,
  isGroup,
  composeTransforms,
  identityTransform,
  quatMultiply,
  quatRotate,
  quatFromAxisAngle,
  GroupConfig,
  Transform,
  Quat,
} from "../../src/js/scene3d/groups";
import { PointCloudComponentConfig, EllipsoidComponentConfig } from "../../src/js/scene3d/components";

describe("scene3d groups", () => {
  describe("quaternion math", () => {
    it("should rotate vector by identity quaternion", () => {
      const identity: Quat = [0, 0, 0, 1];
      const v = quatRotate(identity, [1, 0, 0]);
      expect(v[0]).toBeCloseTo(1);
      expect(v[1]).toBeCloseTo(0);
      expect(v[2]).toBeCloseTo(0);
    });

    it("should rotate vector by 90 degrees around Y axis", () => {
      const q = quatFromAxisAngle([0, 1, 0], Math.PI / 2);
      const v = quatRotate(q, [1, 0, 0]);
      // 90 degrees around Y: [1,0,0] -> [0,0,-1]
      expect(v[0]).toBeCloseTo(0);
      expect(v[1]).toBeCloseTo(0);
      expect(v[2]).toBeCloseTo(-1);
    });

    it("should compose quaternions correctly", () => {
      const q1 = quatFromAxisAngle([0, 1, 0], Math.PI / 2); // 90 deg Y
      const q2 = quatFromAxisAngle([1, 0, 0], Math.PI / 2); // 90 deg X
      const composed = quatMultiply(q1, q2);

      // Apply composed to [0, 0, 1]
      // First q2 rotates [0,0,1] -> [0,-1,0]
      // Then q1 rotates [0,-1,0] -> [0,-1,0] (Y rotation doesn't affect Y-aligned vector)
      const v = quatRotate(composed, [0, 0, 1]);
      expect(v[0]).toBeCloseTo(0);
      expect(v[1]).toBeCloseTo(-1);
      expect(v[2]).toBeCloseTo(0);
    });
  });

  describe("transform composition", () => {
    it("should compose identity transforms", () => {
      const t1 = identityTransform();
      const t2 = identityTransform();
      const composed = composeTransforms(t1, t2);

      expect(composed.position).toEqual([0, 0, 0]);
      expect(composed.scale).toEqual([1, 1, 1]);
      expect(composed.quaternion[3]).toBeCloseTo(1); // w component
    });

    it("should compose position transforms", () => {
      const parent: Transform = {
        position: [1, 0, 0],
        quaternion: [0, 0, 0, 1],
        scale: [1, 1, 1],
      };
      const child: Transform = {
        position: [0, 1, 0],
        quaternion: [0, 0, 0, 1],
        scale: [1, 1, 1],
      };
      const composed = composeTransforms(parent, child);

      expect(composed.position[0]).toBeCloseTo(1);
      expect(composed.position[1]).toBeCloseTo(1);
      expect(composed.position[2]).toBeCloseTo(0);
    });

    it("should compose scale transforms", () => {
      const parent: Transform = {
        position: [0, 0, 0],
        quaternion: [0, 0, 0, 1],
        scale: [2, 2, 2],
      };
      const child: Transform = {
        position: [1, 0, 0], // Will be scaled by parent
        quaternion: [0, 0, 0, 1],
        scale: [0.5, 0.5, 0.5],
      };
      const composed = composeTransforms(parent, child);

      // Child position is scaled by parent scale
      expect(composed.position[0]).toBeCloseTo(2);
      // Scales multiply
      expect(composed.scale).toEqual([1, 1, 1]);
    });

    it("should compose rotation and position", () => {
      // Parent rotated 90 degrees around Y
      const parent: Transform = {
        position: [0, 0, 0],
        quaternion: quatFromAxisAngle([0, 1, 0], Math.PI / 2),
        scale: [1, 1, 1],
      };
      const child: Transform = {
        position: [1, 0, 0], // Will be rotated by parent
        quaternion: [0, 0, 0, 1],
        scale: [1, 1, 1],
      };
      const composed = composeTransforms(parent, child);

      // [1,0,0] rotated 90 degrees around Y becomes [0,0,-1]
      expect(composed.position[0]).toBeCloseTo(0);
      expect(composed.position[1]).toBeCloseTo(0);
      expect(composed.position[2]).toBeCloseTo(-1);
    });
  });

  describe("isGroup and hasGroups", () => {
    it("should identify group configs", () => {
      const group: GroupConfig = {
        type: "Group",
        children: [],
      };
      const pointCloud: PointCloudComponentConfig = {
        type: "PointCloud",
        centers: new Float32Array([0, 0, 0]),
      };

      expect(isGroup(group)).toBe(true);
      expect(isGroup(pointCloud)).toBe(false);
    });

    it("should detect groups in component array", () => {
      const components1 = [
        { type: "PointCloud", centers: new Float32Array([0, 0, 0]) },
      ];
      const components2 = [
        { type: "Group", children: [] },
        { type: "PointCloud", centers: new Float32Array([0, 0, 0]) },
      ];

      expect(hasGroups(components1 as any)).toBe(false);
      expect(hasGroups(components2 as any)).toBe(true);
    });
  });

  describe("flattenGroups", () => {
    it("should pass through non-group components unchanged", () => {
      const components: PointCloudComponentConfig[] = [
        {
          type: "PointCloud",
          centers: new Float32Array([0, 0, 0]),
          color: [1, 0, 0],
        },
      ];

      const flattened = flattenGroups(components);

      expect(flattened.length).toBe(1);
      expect(flattened[0].type).toBe("PointCloud");
      expect((flattened[0] as PointCloudComponentConfig).centers).toEqual(
        new Float32Array([0, 0, 0])
      );
    });

    it("should transform child centers by group position", () => {
      const group: GroupConfig = {
        type: "Group",
        position: [10, 0, 0],
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0, 1, 0, 0]), // Two points
          } as PointCloudComponentConfig,
        ],
      };

      const flattened = flattenGroups([group]);

      expect(flattened.length).toBe(1);
      const pc = flattened[0] as PointCloudComponentConfig;
      // Points should be offset by [10, 0, 0]
      expect(pc.centers[0]).toBeCloseTo(10); // First point x
      expect(pc.centers[1]).toBeCloseTo(0);
      expect(pc.centers[2]).toBeCloseTo(0);
      expect(pc.centers[3]).toBeCloseTo(11); // Second point x
    });

    it("should transform child centers by group rotation", () => {
      const group: GroupConfig = {
        type: "Group",
        quaternion: quatFromAxisAngle([0, 1, 0], Math.PI / 2), // 90 deg Y
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([1, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const flattened = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      // [1,0,0] rotated 90 degrees around Y becomes [0,0,-1]
      expect(pc.centers[0]).toBeCloseTo(0);
      expect(pc.centers[1]).toBeCloseTo(0);
      expect(pc.centers[2]).toBeCloseTo(-1);
    });

    it("should transform child centers by group scale", () => {
      const group: GroupConfig = {
        type: "Group",
        scale: 2,
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([1, 2, 3]),
          } as PointCloudComponentConfig,
        ],
      };

      const flattened = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      expect(pc.centers[0]).toBeCloseTo(2);
      expect(pc.centers[1]).toBeCloseTo(4);
      expect(pc.centers[2]).toBeCloseTo(6);
    });

    it("should scale half_sizes by group scale", () => {
      const group: GroupConfig = {
        type: "Group",
        scale: [2, 3, 4],
        children: [
          {
            type: "Ellipsoid",
            centers: new Float32Array([0, 0, 0]),
            half_size: [1, 1, 1],
          } as EllipsoidComponentConfig,
        ],
      };

      const flattened = flattenGroups([group]);
      const ellipsoid = flattened[0] as EllipsoidComponentConfig;

      expect(ellipsoid.half_size).toEqual([2, 3, 4]);
    });

    it("should add groupPath for named groups", () => {
      const group: GroupConfig = {
        type: "Group",
        name: "myGroup",
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const flattened = flattenGroups([group]);

      expect((flattened[0] as any)._groupPath).toEqual(["myGroup"]);
    });

    it("should handle nested groups", () => {
      const innerGroup: GroupConfig = {
        type: "Group",
        name: "inner",
        position: [1, 0, 0],
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const outerGroup: GroupConfig = {
        type: "Group",
        name: "outer",
        position: [0, 1, 0],
        children: [innerGroup],
      };

      const flattened = flattenGroups([outerGroup]);

      expect(flattened.length).toBe(1);
      const pc = flattened[0] as PointCloudComponentConfig;

      // Position should be outer [0,1,0] + inner [1,0,0] = [1,1,0]
      expect(pc.centers[0]).toBeCloseTo(1);
      expect(pc.centers[1]).toBeCloseTo(1);
      expect(pc.centers[2]).toBeCloseTo(0);

      // Group path should include both names
      expect((pc as any)._groupPath).toEqual(["outer", "inner"]);
    });

    it("should compose quaternions from child components", () => {
      const childQuat = quatFromAxisAngle([1, 0, 0], Math.PI / 2); // 90 deg X

      const group: GroupConfig = {
        type: "Group",
        quaternion: quatFromAxisAngle([0, 1, 0], Math.PI / 2), // 90 deg Y
        children: [
          {
            type: "Ellipsoid",
            centers: new Float32Array([0, 0, 0]),
            quaternion: childQuat,
          } as EllipsoidComponentConfig,
        ],
      };

      const flattened = flattenGroups([group]);
      const ellipsoid = flattened[0] as EllipsoidComponentConfig;

      // The composed quaternion should rotate vectors by both rotations
      // First child's X rotation, then parent's Y rotation
      const resultQuat = ellipsoid.quaternion as [number, number, number, number];

      // Apply to [0, 0, 1]:
      // Child (90 X): [0, 0, 1] -> [0, -1, 0]
      // Parent (90 Y): [0, -1, 0] -> [0, -1, 0] (Y rotation doesn't affect Y-aligned)
      const v = quatRotate(resultQuat, [0, 0, 1]);
      expect(v[0]).toBeCloseTo(0);
      expect(v[1]).toBeCloseTo(-1);
      expect(v[2]).toBeCloseTo(0);
    });
  });
});
