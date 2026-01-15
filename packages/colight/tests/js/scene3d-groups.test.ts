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
import {
  PointCloudComponentConfig,
  EllipsoidComponentConfig,
} from "../../src/js/scene3d/components";
import { EllipsoidAxesComponentConfig } from "../../src/js/scene3d/primitives/ellipsoidAxes";
import { ImagePlaneComponentConfig } from "../../src/js/scene3d/primitives/imagePlane";
import { MeshComponentConfig } from "../../src/js/scene3d/primitives/mesh";

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

    it("should detect groups with transforms in component array", () => {
      const noGroups = [
        { type: "PointCloud", centers: new Float32Array([0, 0, 0]) },
      ];
      const compositionOnlyGroup = [
        { type: "Group", children: [] },
        { type: "PointCloud", centers: new Float32Array([0, 0, 0]) },
      ];
      const groupWithChildProps = [
        {
          type: "Group",
          childDefaults: { outline: true },
          children: [],
        },
        { type: "PointCloud", centers: new Float32Array([0, 0, 0]) },
      ];
      const groupWithTransform = [
        { type: "Group", children: [], position: [1, 0, 0] },
        { type: "PointCloud", centers: new Float32Array([0, 0, 0]) },
      ];

      expect(hasGroups(noGroups as any)).toBe(false);
      // Composition-only groups (no transform) return false for perf optimization
      expect(hasGroups(compositionOnlyGroup as any)).toBe(false);
      expect(hasGroups(groupWithChildProps as any)).toBe(true);
      expect(hasGroups(groupWithTransform as any)).toBe(true);
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

      const { components: flattened } = flattenGroups(components);

      expect(flattened.length).toBe(1);
      expect(flattened[0].type).toBe("PointCloud");
      expect((flattened[0] as PointCloudComponentConfig).centers).toEqual(
        new Float32Array([0, 0, 0]),
      );
    });

    it("should set transform index for group with position (GPU transforms)", () => {
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

      const { components: flattened, transforms } = flattenGroups([group]);

      expect(flattened.length).toBe(1);
      const pc = flattened[0] as PointCloudComponentConfig;
      // Centers stay in local space (GPU transforms them)
      expect(pc.centers[0]).toBeCloseTo(0);
      expect(pc.centers[3]).toBeCloseTo(1);
      // Transform index should point to the group transform
      expect((flattened[0] as any)._transformIndex).toBe(1);
      // Transforms array should have identity + this transform
      expect(transforms.length).toBe(2);
      expect(transforms[1].position).toEqual([10, 0, 0]);
    });

    it("should set transform index for group with rotation (GPU transforms)", () => {
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

      const { components: flattened, transforms } = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      // Centers stay in local space (GPU transforms them)
      expect(pc.centers[0]).toBeCloseTo(1);
      expect(pc.centers[1]).toBeCloseTo(0);
      expect(pc.centers[2]).toBeCloseTo(0);
      // Transform should be set
      expect((flattened[0] as any)._transformIndex).toBe(1);
      // Quaternion should be in transforms array
      expect(transforms[1].quaternion[3]).toBeCloseTo(Math.cos(Math.PI / 4)); // cos(45deg)
    });

    it("should set transform index for group with scale (GPU transforms)", () => {
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

      const { components: flattened, transforms } = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      // Centers stay in local space
      expect(pc.centers[0]).toBeCloseTo(1);
      expect(pc.centers[1]).toBeCloseTo(2);
      expect(pc.centers[2]).toBeCloseTo(3);
      // Scale should be in transforms array
      expect(transforms[1].scale).toEqual([2, 2, 2]);
    });

    it("should keep half_sizes in local space (GPU applies scale)", () => {
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

      const { components: flattened, transforms } = flattenGroups([group]);
      const ellipsoid = flattened[0] as EllipsoidComponentConfig;

      // half_size stays in local space (GPU applies scale)
      expect(ellipsoid.half_size).toEqual([1, 1, 1]);
      // Scale is in transforms array
      expect(transforms[1].scale).toEqual([2, 3, 4]);
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

      const { components: flattened } = flattenGroups([group]);

      expect((flattened[0] as any)._groupPath).toEqual(["myGroup"]);
    });

    it("should apply childDefaults defaults to children", () => {
      const group: GroupConfig = {
        type: "Group",
        childDefaults: {
          color: [0.1, 0.2, 0.3],
          outline: true,
          outlineWidth: 4,
        },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const { components: flattened } = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      expect(pc.color).toEqual([0.1, 0.2, 0.3]);
      expect(pc.outline).toBe(true);
      expect(pc.outlineWidth).toBe(4);
    });

    it("should not override child props when provided", () => {
      const group: GroupConfig = {
        type: "Group",
        childDefaults: {
          color: [1, 0, 0],
          outlineColor: [0, 1, 0],
        },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
            color: [0, 0, 1],
            outlineColor: [1, 1, 0],
          } as PointCloudComponentConfig,
        ],
      };

      const { components: flattened } = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      expect(pc.color).toEqual([0, 0, 1]);
      expect(pc.outlineColor).toEqual([1, 1, 0]);
    });

    it("should apply hoverProps from childDefaults", () => {
      const group: GroupConfig = {
        type: "Group",
        childDefaults: {
          hoverProps: {
            color: [1, 1, 0],
            scale: 1.2,
            outline: true,
          },
        },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const { components: flattened } = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      expect(pc.hoverProps).toEqual({
        color: [1, 1, 0],
        scale: 1.2,
        outline: true,
      });
    });

    it("should not override child hoverProps when provided", () => {
      const group: GroupConfig = {
        type: "Group",
        childDefaults: {
          hoverProps: { color: [1, 0, 0], scale: 1.5 },
        },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
            hoverProps: { color: [0, 1, 0], outline: true },
          } as PointCloudComponentConfig,
        ],
      };

      const { components: flattened } = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      // Child's hoverProps should win over group's
      expect(pc.hoverProps).toEqual({ color: [0, 1, 0], outline: true });
    });

    it("should apply childOverrides and override child values", () => {
      const group: GroupConfig = {
        type: "Group",
        childOverrides: {
          color: [1, 0, 0], // Group wants all children red
          outlineWidth: 5,
        },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
            color: [0, 0, 1], // Child says blue
            outlineColor: [1, 1, 0], // Child sets outline color
          } as PointCloudComponentConfig,
        ],
      };

      const { components: flattened } = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      // Group overrides should win
      expect(pc.color).toEqual([1, 0, 0]); // Red from override, not blue from child
      expect(pc.outlineWidth).toBe(5); // From override
      // Child values not in overrides should be preserved
      expect(pc.outlineColor).toEqual([1, 1, 0]); // From child
    });

    it("should apply both childDefaults and childOverrides", () => {
      const group: GroupConfig = {
        type: "Group",
        childDefaults: {
          alpha: 0.5, // Default alpha (child can override)
          outlineColor: [0, 1, 0], // Default outline color (child can override)
        },
        childOverrides: {
          color: [1, 0, 0], // Force red (group wins)
        },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
            alpha: 0.8, // Child overrides alpha default
            outlineColor: [1, 1, 0], // Child overrides outline default
          } as PointCloudComponentConfig,
        ],
      };

      const { components: flattened } = flattenGroups([group]);
      const pc = flattened[0] as PointCloudComponentConfig;

      // Override wins over child
      expect(pc.color).toEqual([1, 0, 0]);
      // Child wins over default
      expect(pc.alpha).toBe(0.8);
      expect(pc.outlineColor).toEqual([1, 1, 0]);
    });

    it("should handle nested groups with childOverrides", () => {
      const innerGroup: GroupConfig = {
        type: "Group",
        childOverrides: {
          outlineWidth: 3, // Inner group forces outline width
        },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
            color: [0, 0, 1], // Child color
          } as PointCloudComponentConfig,
        ],
      };

      const outerGroup: GroupConfig = {
        type: "Group",
        childOverrides: {
          color: [1, 0, 0], // Outer group forces color
        },
        children: [innerGroup],
      };

      const { components: flattened } = flattenGroups([outerGroup]);
      const pc = flattened[0] as PointCloudComponentConfig;

      // Both overrides should be applied (inner can override outer for its own children)
      expect(pc.color).toEqual([1, 0, 0]); // From outer override
      expect(pc.outlineWidth).toBe(3); // From inner override
    });

    it("should handle nested groups (GPU transforms with composed transform)", () => {
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

      const { components: flattened, transforms } = flattenGroups([outerGroup]);

      expect(flattened.length).toBe(1);
      const pc = flattened[0] as PointCloudComponentConfig;

      // Centers stay in local space (GPU applies transforms)
      expect(pc.centers[0]).toBeCloseTo(0);
      expect(pc.centers[1]).toBeCloseTo(0);
      expect(pc.centers[2]).toBeCloseTo(0);

      // Transform should be composed: outer [0,1,0] + inner [1,0,0] = [1,1,0]
      expect((flattened[0] as any)._transformIndex).toBe(1);
      expect(transforms[1].position[0]).toBeCloseTo(1);
      expect(transforms[1].position[1]).toBeCloseTo(1);
      expect(transforms[1].position[2]).toBeCloseTo(0);

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

      const { components: flattened } = flattenGroups([group]);
      const ellipsoid = flattened[0] as EllipsoidComponentConfig;

      // The composed quaternion should rotate vectors by both rotations
      // First child's X rotation, then parent's Y rotation
      const resultQuat = ellipsoid.quaternion as [
        number,
        number,
        number,
        number,
      ];

      // Apply to [0, 0, 1]:
      // Child (90 X): [0, 0, 1] -> [0, -1, 0]
      // Parent (90 Y): [0, -1, 0] -> [0, -1, 0] (Y rotation doesn't affect Y-aligned)
      const v = quatRotate(resultQuat, [0, 0, 1]);
      expect(v[0]).toBeCloseTo(0);
      expect(v[1]).toBeCloseTo(-1);
      expect(v[2]).toBeCloseTo(0);
    });

    it("should register group hoverProps in groupRegistry", () => {
      const group: GroupConfig = {
        type: "Group",
        name: "myGroup",
        hoverProps: {
          color: [1, 0, 0],
          scale: 1.5,
        },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const { components: flattened, groupRegistry } = flattenGroups([group]);

      expect(flattened.length).toBe(1);
      expect(groupRegistry.size).toBe(1);
      expect(groupRegistry.has("myGroup")).toBe(true);

      const handlers = groupRegistry.get("myGroup")!;
      expect(handlers.hoverProps).toEqual({ color: [1, 0, 0], scale: 1.5 });
    });

    it("should register nested group hoverProps in groupRegistry", () => {
      const innerGroup: GroupConfig = {
        type: "Group",
        name: "inner",
        hoverProps: { color: [0, 1, 0] },
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
        hoverProps: { scale: 1.2 },
        children: [innerGroup],
      };

      const { groupRegistry } = flattenGroups([outerGroup]);

      expect(groupRegistry.size).toBe(2);
      expect(groupRegistry.has("outer")).toBe(true);
      expect(groupRegistry.has("outer/inner")).toBe(true);

      expect(groupRegistry.get("outer")!.hoverProps).toEqual({ scale: 1.2 });
      expect(groupRegistry.get("outer/inner")!.hoverProps).toEqual({
        color: [0, 1, 0],
      });
    });

    it("should auto-generate names for anonymous groups with handlers", () => {
      const group: GroupConfig = {
        type: "Group",
        // No name - should get auto-generated name
        hoverProps: { color: [1, 0, 0] },
        onClick: () => {},
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const { components: flattened, groupRegistry } = flattenGroups([group]);

      expect(flattened.length).toBe(1);
      // Should have auto-generated name _group_0
      expect(groupRegistry.size).toBe(1);
      expect(groupRegistry.has("_group_0")).toBe(true);

      const handlers = groupRegistry.get("_group_0")!;
      expect(handlers.hoverProps).toEqual({ color: [1, 0, 0] });
      expect(handlers.onClick).toBeDefined();

      // Component should have the auto-generated path
      expect((flattened[0] as any)._groupPath).toEqual(["_group_0"]);
    });

    it("should assign unique auto-generated names to multiple anonymous groups", () => {
      const group1: GroupConfig = {
        type: "Group",
        hoverProps: { color: [1, 0, 0] },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const group2: GroupConfig = {
        type: "Group",
        hoverProps: { color: [0, 1, 0] },
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([1, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const { components: flattened, groupRegistry } = flattenGroups([
        group1,
        group2,
      ]);

      expect(flattened.length).toBe(2);
      expect(groupRegistry.size).toBe(2);
      expect(groupRegistry.has("_group_0")).toBe(true);
      expect(groupRegistry.has("_group_1")).toBe(true);

      expect((flattened[0] as any)._groupPath).toEqual(["_group_0"]);
      expect((flattened[1] as any)._groupPath).toEqual(["_group_1"]);
    });

    it("should set transform index for EllipsoidAxes in group", () => {
      const group: GroupConfig = {
        type: "Group",
        position: [5, 0, 0],
        children: [
          {
            type: "EllipsoidAxes",
            centers: new Float32Array([0, 0, 0]),
            half_size: [1, 2, 3],
          } as EllipsoidAxesComponentConfig,
        ],
      };

      const { components: flattened, transforms } = flattenGroups([group]);

      expect(flattened.length).toBe(1);
      const axes = flattened[0] as EllipsoidAxesComponentConfig;

      // Centers stay in local space (GPU transforms them)
      expect(axes.centers[0]).toBeCloseTo(0);
      // Transform index should point to the group transform
      expect((flattened[0] as any)._transformIndex).toBe(1);
      // Transforms array should have identity + this transform
      expect(transforms.length).toBe(2);
      expect(transforms[1].position).toEqual([5, 0, 0]);
    });

    it("should set transform index for ImagePlane in group", () => {
      // Create a simple test image source
      const testImage = {
        data: new Uint8Array([255, 0, 0, 255]),
        width: 1,
        height: 1,
      };

      const group: GroupConfig = {
        type: "Group",
        position: [10, 10, 0],
        scale: 2,
        children: [
          {
            type: "ImagePlane",
            image: testImage,
            centers: new Float32Array([0, 0, 0]),
            size: [1, 1],
          } as ImagePlaneComponentConfig,
        ],
      };

      const { components: flattened, transforms } = flattenGroups([group]);

      expect(flattened.length).toBe(1);
      const imagePlane = flattened[0] as ImagePlaneComponentConfig;

      // Centers stay in local space
      expect(imagePlane.centers[0]).toBeCloseTo(0);
      // Transform index should be set
      expect((flattened[0] as any)._transformIndex).toBe(1);
      // Transform should include position and scale
      expect(transforms[1].position).toEqual([10, 10, 0]);
      expect(transforms[1].scale).toEqual([2, 2, 2]);
    });

    it("should set transform index for Mesh (custom primitive) in group", () => {
      const group: GroupConfig = {
        type: "Group",
        position: [1, 2, 3],
        quaternion: quatFromAxisAngle([0, 0, 1], Math.PI / 2),
        children: [
          {
            type: "CustomMesh", // defineMesh creates custom type names
            centers: new Float32Array([0, 0, 0]),
            scale: [1, 1, 1],
          } as unknown as MeshComponentConfig,
        ],
      };

      const { components: flattened, transforms } = flattenGroups([group]);

      expect(flattened.length).toBe(1);

      // Transform index should be set
      expect((flattened[0] as any)._transformIndex).toBe(1);
      // Transform should include position and rotation
      expect(transforms[1].position).toEqual([1, 2, 3]);
      // Quaternion should match (90 deg around Z)
      expect(transforms[1].quaternion[2]).toBeCloseTo(Math.sin(Math.PI / 4));
      expect(transforms[1].quaternion[3]).toBeCloseTo(Math.cos(Math.PI / 4));
    });

    it("should compose transforms correctly for nested groups with mixed primitive types", () => {
      const testImage = {
        data: new Uint8Array([255, 0, 0, 255]),
        width: 1,
        height: 1,
      };

      const innerGroup: GroupConfig = {
        type: "Group",
        position: [1, 0, 0],
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as PointCloudComponentConfig,
          {
            type: "EllipsoidAxes",
            centers: new Float32Array([0, 0, 0]),
          } as EllipsoidAxesComponentConfig,
        ],
      };

      const outerGroup: GroupConfig = {
        type: "Group",
        position: [0, 10, 0],
        scale: 2,
        children: [
          innerGroup,
          {
            type: "ImagePlane",
            image: testImage,
            centers: new Float32Array([0, 0, 0]),
          } as ImagePlaneComponentConfig,
        ],
      };

      const { components: flattened, transforms } = flattenGroups([outerGroup]);

      expect(flattened.length).toBe(3);

      // Inner group components should have composed transform
      // Outer position [0,10,0] + (inner position [1,0,0] * outer scale 2) = [2,10,0]
      const innerTransformIdx0 = (flattened[0] as any)._transformIndex;
      const innerTransformIdx1 = (flattened[1] as any)._transformIndex;
      expect(innerTransformIdx0).toBeGreaterThan(0);
      expect(innerTransformIdx1).toBeGreaterThan(0);

      // Both inner components should have equivalent transforms (composed from same parents)
      expect(transforms[innerTransformIdx0].position[0]).toBeCloseTo(2);
      expect(transforms[innerTransformIdx0].position[1]).toBeCloseTo(10);
      expect(transforms[innerTransformIdx0].scale).toEqual([2, 2, 2]);

      expect(transforms[innerTransformIdx1].position[0]).toBeCloseTo(2);
      expect(transforms[innerTransformIdx1].position[1]).toBeCloseTo(10);
      expect(transforms[innerTransformIdx1].scale).toEqual([2, 2, 2]);

      // ImagePlane in outer group should have outer transform only
      const outerTransformIdx = (flattened[2] as any)._transformIndex;
      expect(outerTransformIdx).toBeGreaterThan(0);
      expect(transforms[outerTransformIdx].position).toEqual([0, 10, 0]);
    });
  });

  describe("module loading", () => {
    it("should load gpu-transforms without circular dependency errors", async () => {
      // This test verifies that the circular dependency fix works
      // If there's a TDZ error, this import would fail
      const gpuTransforms = await import("../../src/js/scene3d/gpu-transforms");

      // Verify IDENTITY_GPU_TRANSFORM has correct values
      expect(gpuTransforms.IDENTITY_GPU_TRANSFORM).toBeDefined();
      expect(gpuTransforms.IDENTITY_GPU_TRANSFORM.position).toEqual([0, 0, 0]);
      expect(gpuTransforms.IDENTITY_GPU_TRANSFORM.quaternion).toEqual([
        0, 0, 0, 1,
      ]);
      expect(gpuTransforms.IDENTITY_GPU_TRANSFORM.scale).toEqual([1, 1, 1]);
    });

    it("should load groups module and use gpu-transforms correctly", async () => {
      // Verify the groups module can create transforms array with identity
      const group: GroupConfig = {
        type: "Group",
        // No transform - should use identity
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as PointCloudComponentConfig,
        ],
      };

      const { transforms } = flattenGroups([group]);

      // Should have identity transform at index 0
      expect(transforms.length).toBe(1);
      expect(transforms[0].position).toEqual([0, 0, 0]);
      expect(transforms[0].quaternion).toEqual([0, 0, 0, 1]);
      expect(transforms[0].scale).toEqual([1, 1, 1]);
    });
  });
});
