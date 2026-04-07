import { describe, expect, it } from "vitest";

import {
  billboardShaderProgram,
  ellipsoidImpostorShaderProgram,
  lineBeamShaderProgram,
  rigidLitShaderProgram,
} from "../../../src/js/scene3d/shaders";
import {
  cuboidSpec,
  ellipsoidSpec,
  lineBeamsSpec,
  pointCloudSpec,
} from "../../../src/js/scene3d/components";

describe("scene3d shader generation", () => {
  it("builds billboard shaders and layouts from declarative fields", () => {
    expect(billboardShaderProgram.renderFloatsPerInstance).toBe(8);
    expect(billboardShaderProgram.pickIDFloatsPerInstance).toBe(1);
    expect(billboardShaderProgram.colorOffset).toBe(4);
    expect(billboardShaderProgram.alphaOffset).toBe(7);
    expect(
      billboardShaderProgram.renderLayout.attributes.map((attribute) => ({
        shaderLocation: attribute.shaderLocation,
        format: attribute.format,
      })),
    ).toEqual([
      { shaderLocation: 2, format: "float32x3" },
      { shaderLocation: 3, format: "float32" },
      { shaderLocation: 4, format: "float32x3" },
      { shaderLocation: 5, format: "float32" },
    ]);
    expect(
      billboardShaderProgram.pickIDLayout.attributes.map((attribute) => ({
        shaderLocation: attribute.shaderLocation,
        format: attribute.format,
      })),
    ).toEqual([{ shaderLocation: 6, format: "float32" }]);
    expect(billboardShaderProgram.renderVertex).toContain("camera.cameraRight");
    expect(billboardShaderProgram.fragment).not.toContain("calculateLighting");
    expect(billboardShaderProgram.fragment).toContain("encodePickID");
  });

  it("builds lit rigid-body shaders with quaternion transforms", () => {
    expect(rigidLitShaderProgram.renderFloatsPerInstance).toBe(14);
    expect(rigidLitShaderProgram.pickIDFloatsPerInstance).toBe(1);
    expect(rigidLitShaderProgram.colorOffset).toBe(10);
    expect(rigidLitShaderProgram.alphaOffset).toBe(13);
    expect(rigidLitShaderProgram.renderVertex).toContain(
      "quat_rotate(quaternion, scaledLocal)",
    );
    expect(rigidLitShaderProgram.renderVertex).toContain(
      "let worldNormal = quat_rotate(quaternion, invScaledNorm);",
    );
    expect(rigidLitShaderProgram.fragment).toContain("calculateLighting");
    expect(rigidLitShaderProgram.renderVertex).toContain(
      "out.pickID = pickID;",
    );
  });

  it("builds beam shaders without runtime branching", () => {
    expect(lineBeamShaderProgram.renderFloatsPerInstance).toBe(11);
    expect(lineBeamShaderProgram.pickIDFloatsPerInstance).toBe(1);
    expect(lineBeamShaderProgram.colorOffset).toBe(7);
    expect(lineBeamShaderProgram.alphaOffset).toBe(10);
    expect(lineBeamShaderProgram.renderVertex).toContain("segmentLength");
    expect(lineBeamShaderProgram.renderVertex).toContain(
      "out.pickID = pickID;",
    );
  });

  it("builds derived-pick shader variants for opaque render objects", () => {
    expect(billboardShaderProgram.renderVertexDerivedPick).toContain(
      "@builtin(instance_index) instanceIndex: u32",
    );
    expect(billboardShaderProgram.renderVertexDerivedPick).toContain(
      "objectPick.pickBase",
    );
    expect(rigidLitShaderProgram.renderVertexDerivedPick).toContain(
      "objectPick.instancesPerElement",
    );
    expect(ellipsoidImpostorShaderProgram.renderVertexDerivedPick).toContain(
      "objectPick.pickBase",
    );
  });

  it("propagates generated buffer metadata into primitive specs", () => {
    const pointCloudMeshSpec = pointCloudSpec.resolveSpec({
      type: "PointCloud",
      centers: new Float32Array(),
    });
    const ellipsoidMeshSpec = ellipsoidSpec.resolveSpec({
      type: "Ellipsoid",
      centers: new Float32Array(),
    });
    const ellipsoidImpostorSpec = ellipsoidSpec.resolveSpec({
      type: "Ellipsoid",
      centers: new Float32Array(),
      render_mode: "impostor",
    });
    const cuboidMeshSpec = cuboidSpec.resolveSpec({
      type: "Cuboid",
      centers: new Float32Array(),
    });
    const lineBeamsMeshSpec = lineBeamsSpec.resolveSpec({
      type: "LineBeams",
      points: new Float32Array(),
    });

    expect(pointCloudMeshSpec.floatsPerInstance).toBe(
      billboardShaderProgram.renderFloatsPerInstance,
    );
    expect(pointCloudMeshSpec.colorOffset).toBe(
      billboardShaderProgram.colorOffset,
    );
    expect(ellipsoidMeshSpec.floatsPerInstance).toBe(
      rigidLitShaderProgram.renderFloatsPerInstance,
    );
    expect(ellipsoidImpostorSpec.floatsPerInstance).toBe(
      ellipsoidImpostorShaderProgram.renderFloatsPerInstance,
    );
    expect(ellipsoidImpostorSpec.resourceKey).toBe("Ellipsoid:impostor");
    expect(cuboidMeshSpec.alphaOffset).toBe(rigidLitShaderProgram.alphaOffset);
    expect(lineBeamsMeshSpec.floatsPerPicking).toBe(
      lineBeamShaderProgram.pickIDFloatsPerInstance,
    );
  });
});
