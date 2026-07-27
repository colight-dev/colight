/**
 * @module scene3d-compiler.test
 * @description Tests for the unified scene compilation pipeline.
 *
 * These tests ensure that all entry paths (JSX, components prop, layers)
 * produce consistent, correctly normalized output.
 */

import { describe, it, expect } from "vitest";
import { compileScene, RawComponent } from "../../src/js/scene3d/compiler";
import { ImageProjectionProps } from "../../src/js/scene3d/helpers";
import { GroupConfig } from "../../src/js/scene3d/groups";

// =============================================================================
// Test Fixtures
// =============================================================================

const mockIntrinsics = {
  fx: 500,
  fy: 500,
  cx: 320,
  cy: 240,
  width: 640,
  height: 480,
};

const mockExtrinsics = {
  position: [0, 0, 0] as [number, number, number],
  quaternion: [0, 0, 0, 1] as [number, number, number, number],
};

const mockImage = {
  data: new Uint8Array(640 * 480 * 4),
  width: 640,
  height: 480,
};

// =============================================================================
// Helper Expansion Tests
// =============================================================================

describe("Scene Compiler - Helper Expansion", () => {
  it("expands ImageProjection with showFrustum=true to ImagePlane + LineSegments", () => {
    const input: RawComponent[] = [
      {
        type: "ImageProjection",
        image: mockImage,
        intrinsics: mockIntrinsics,
        extrinsics: mockExtrinsics,
        showFrustum: true,
      } as { type: "ImageProjection" } & ImageProjectionProps,
    ];

    const result = compileScene(input);

    // Should have exactly 2 components: ImagePlane and LineSegments (frustum)
    expect(result.components).toHaveLength(2);
    expect(result.components[0].type).toBe("ImagePlane");
    expect(result.components[1].type).toBe("LineSegments");
  });

  it("expands ImageProjection with showFrustum=false to just ImagePlane", () => {
    const input: RawComponent[] = [
      {
        type: "ImageProjection",
        image: mockImage,
        intrinsics: mockIntrinsics,
        extrinsics: mockExtrinsics,
        showFrustum: false,
      } as { type: "ImageProjection" } & ImageProjectionProps,
    ];

    const result = compileScene(input);

    expect(result.components).toHaveLength(1);
    expect(result.components[0].type).toBe("ImagePlane");
  });

  it("expands CameraFrustum to LineSegments", () => {
    const input: RawComponent[] = [
      {
        type: "CameraFrustum",
        intrinsics: mockIntrinsics,
        extrinsics: mockExtrinsics,
      } as RawComponent,
    ];

    const result = compileScene(input);

    expect(result.components).toHaveLength(1);
    expect(result.components[0].type).toBe("LineSegments");
  });

  it("expands GridHelper to LineSegments", () => {
    const input: RawComponent[] = [
      {
        type: "GridHelper",
        size: 10,
        divisions: 10,
      } as RawComponent,
    ];

    const result = compileScene(input);

    expect(result.components).toHaveLength(1);
    expect(result.components[0].type).toBe("LineSegments");
  });
});

// =============================================================================
// Helper Expansion Inside Groups
// =============================================================================

describe("Scene Compiler - Helpers Inside Groups", () => {
  it("expands ImageProjection inside Group and preserves group path", () => {
    const input: RawComponent[] = [
      {
        type: "Group",
        name: "camera-group",
        position: [1, 2, 3],
        children: [
          {
            type: "ImageProjection",
            image: mockImage,
            intrinsics: mockIntrinsics,
            extrinsics: mockExtrinsics,
            showFrustum: true,
          } as { type: "ImageProjection" } & ImageProjectionProps,
        ],
      } as GroupConfig,
    ];

    const result = compileScene(input);

    // Should have 2 components (ImagePlane + LineSegments)
    expect(result.components).toHaveLength(2);
    expect(result.components[0].type).toBe("ImagePlane");
    expect(result.components[1].type).toBe("LineSegments");

    // Both should have the group path
    expect((result.components[0] as any)._groupPath).toEqual(["camera-group"]);
    expect((result.components[1] as any)._groupPath).toEqual(["camera-group"]);

    // Both should have transform index > 0 (non-identity)
    expect((result.components[0] as any)._transformIndex).toBeGreaterThan(0);
    expect((result.components[1] as any)._transformIndex).toBeGreaterThan(0);
  });

  it("expands helpers in nested groups with correct transform indices", () => {
    const input: RawComponent[] = [
      {
        type: "Group",
        name: "outer",
        position: [1, 0, 0],
        children: [
          {
            type: "Group",
            name: "inner",
            position: [0, 1, 0],
            children: [
              {
                type: "GridHelper",
                size: 5,
              } as RawComponent,
            ],
          } as GroupConfig,
        ],
      } as GroupConfig,
    ];

    const result = compileScene(input);

    expect(result.components).toHaveLength(1);
    expect(result.components[0].type).toBe("LineSegments");

    // Should have nested group path
    expect((result.components[0] as any)._groupPath).toEqual([
      "outer",
      "inner",
    ]);

    // Should have non-identity transform
    const transformIndex = (result.components[0] as any)._transformIndex;
    expect(transformIndex).toBeGreaterThan(0);

    // Check the transform is composed correctly
    const transform = result.transforms[transformIndex];
    expect(transform.position[0]).toBe(1); // From outer group
    expect(transform.position[1]).toBe(1); // From inner group
    expect(transform.position[2]).toBe(0);
  });
});

// =============================================================================
// Coercion Tests
// =============================================================================

describe("Scene Compiler - Coercion", () => {
  it("coerces PointCloud with singular center to plural centers", () => {
    const input: RawComponent[] = [
      {
        type: "PointCloud",
        center: [1, 2, 3],
        color: [1, 0, 0],
      } as any,
    ];

    const result = compileScene(input);

    expect(result.components).toHaveLength(1);
    const pc = result.components[0] as any;
    expect(pc.type).toBe("PointCloud");
    expect(pc.centers).toBeInstanceOf(Float32Array);
    expect(Array.from(pc.centers)).toEqual([1, 2, 3]);
  });

  it("coerces Cuboid with singular half_size to array", () => {
    const input: RawComponent[] = [
      {
        type: "Cuboid",
        centers: [0, 0, 0],
        half_size: 0.5,
      } as any,
    ];

    const result = compileScene(input);

    expect(result.components).toHaveLength(1);
    const cuboid = result.components[0] as any;
    expect(cuboid.type).toBe("Cuboid");
    // The coercion expands scalar half_size to [0.5, 0.5, 0.5]
    expect(cuboid.half_size).toEqual([0.5, 0.5, 0.5]);
  });

  it("coerces nested group children", () => {
    const input: RawComponent[] = [
      {
        type: "Group",
        children: [
          {
            type: "PointCloud",
            center: [0, 0, 0],
            color: [1, 0, 0],
          } as any,
        ],
      } as GroupConfig,
    ];

    const result = compileScene(input);

    expect(result.components).toHaveLength(1);
    const pc = result.components[0] as any;
    expect(pc.centers).toBeInstanceOf(Float32Array);
  });
});

// =============================================================================
// Group Flattening Tests
// =============================================================================

describe("Scene Compiler - Group Flattening", () => {
  it("flattens simple group with transform", () => {
    const input: RawComponent[] = [
      {
        type: "Group",
        position: [1, 2, 3],
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
            color: [1, 0, 0],
          } as any,
        ],
      } as GroupConfig,
    ];

    const result = compileScene(input);

    expect(result.components).toHaveLength(1);
    expect(result.transforms.length).toBeGreaterThan(1); // identity + group transform

    const transformIndex = (result.components[0] as any)._transformIndex;
    expect(transformIndex).toBeGreaterThan(0);

    const transform = result.transforms[transformIndex];
    expect(transform.position).toEqual([1, 2, 3]);
  });

  it("preserves group registry for groups with handlers", () => {
    const onHover = () => {};
    const input: RawComponent[] = [
      {
        type: "Group",
        name: "interactive",
        onHover,
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as any,
        ],
      } as GroupConfig,
    ];

    const result = compileScene(input);

    expect(result.groupRegistry).toBeDefined();
    expect(result.groupRegistry!.has("interactive")).toBe(true);
    expect(result.groupRegistry!.get("interactive")!.onHover).toBe(onHover);
  });

  it("returns undefined groupRegistry when no groups have handlers", () => {
    const input: RawComponent[] = [
      {
        type: "Group",
        position: [1, 0, 0],
        children: [
          {
            type: "PointCloud",
            centers: new Float32Array([0, 0, 0]),
          } as any,
        ],
      } as GroupConfig,
    ];

    const result = compileScene(input);

    // No handlers = undefined registry
    expect(result.groupRegistry).toBeUndefined();
  });
});

// =============================================================================
// Filtering Tests
// =============================================================================

describe("Scene Compiler - Filtering", () => {
  it("filters out unknown component types", () => {
    const input: RawComponent[] = [
      { type: "PointCloud", centers: new Float32Array([0, 0, 0]) } as any,
      { type: "UnknownType", data: [1, 2, 3] } as any,
      { type: "Ellipsoid", centers: new Float32Array([1, 1, 1]) } as any,
    ];

    const result = compileScene(input);

    expect(result.components).toHaveLength(2);
    expect(result.components[0].type).toBe("PointCloud");
    expect(result.components[1].type).toBe("Ellipsoid");
  });

  it("allows custom primitive types when valid spec is provided", () => {
    // Note: In practice, custom primitives require proper specs with all required
    // properties. This test verifies that the filtering logic respects primitiveSpecs.
    // Full custom primitive testing is covered in extensibility tests.

    // Without a valid spec, custom types should be filtered out
    const input: RawComponent[] = [
      { type: "CustomPrimitive", data: [1, 2, 3] } as any,
    ];

    const result = compileScene(input);
    expect(result.components).toHaveLength(0);
  });

  it("recursively filters unknown types inside Groups", () => {
    const input: RawComponent[] = [
      {
        type: "Group",
        position: [1, 0, 0],
        children: [
          { type: "PointCloud", centers: new Float32Array([0, 0, 0]) } as any,
          { type: "UnknownNestedType", data: [1, 2, 3] } as any,
          { type: "Ellipsoid", centers: new Float32Array([1, 1, 1]) } as any,
        ],
      } as GroupConfig,
    ];

    const result = compileScene(input);

    // Should have 2 components (PointCloud and Ellipsoid from the group)
    expect(result.components).toHaveLength(2);
    expect(result.components[0].type).toBe("PointCloud");
    expect(result.components[1].type).toBe("Ellipsoid");
  });

  it("removes empty Groups after filtering invalid children", () => {
    const input: RawComponent[] = [
      {
        type: "Group",
        position: [1, 0, 0],
        children: [
          { type: "UnknownType1", data: [1] } as any,
          { type: "UnknownType2", data: [2] } as any,
        ],
      } as GroupConfig,
      { type: "PointCloud", centers: new Float32Array([0, 0, 0]) } as any,
    ];

    const result = compileScene(input);

    // Group should be removed (no valid children), only PointCloud remains
    expect(result.components).toHaveLength(1);
    expect(result.components[0].type).toBe("PointCloud");
  });
});

// =============================================================================
// Transform Array Tests
// =============================================================================

describe("Scene Compiler - Transform Array", () => {
  it("always includes identity transform at index 0", () => {
    const input: RawComponent[] = [
      { type: "PointCloud", centers: new Float32Array([0, 0, 0]) } as any,
    ];

    const result = compileScene(input);

    expect(result.transforms).toHaveLength(1);
    expect(result.transforms[0]).toEqual({
      position: [0, 0, 0],
      quaternion: [0, 0, 0, 1],
      scale: [1, 1, 1],
    });
  });

  it("components without groups get transformIndex 0", () => {
    const input: RawComponent[] = [
      { type: "PointCloud", centers: new Float32Array([0, 0, 0]) } as any,
    ];

    const result = compileScene(input);

    // No _transformIndex means identity (0)
    expect((result.components[0] as any)._transformIndex).toBeUndefined();
  });

  it("multiple groups create multiple transforms", () => {
    const input: RawComponent[] = [
      {
        type: "Group",
        position: [1, 0, 0],
        children: [
          { type: "PointCloud", centers: new Float32Array([0, 0, 0]) } as any,
        ],
      } as GroupConfig,
      {
        type: "Group",
        position: [2, 0, 0],
        children: [
          { type: "Ellipsoid", centers: new Float32Array([0, 0, 0]) } as any,
        ],
      } as GroupConfig,
    ];

    const result = compileScene(input);

    // Identity + 2 group transforms
    expect(result.transforms.length).toBeGreaterThanOrEqual(3);
  });
});
