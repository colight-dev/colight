/// <reference types="@webgpu/types" />
import { describe, it, expect, vi, beforeEach, afterEach, Mock } from "vitest";
import { render, act } from "@testing-library/react";
import React from "react";
import { SceneImpl } from "../../../src/js/scene3d/impl3d";
import { defineMesh } from "../../../src/js/scene3d/primitives/mesh";
import { definePrimitive, attr } from "../../../src/js/scene3d/primitives/define";
import { setupWebGPU, cleanupWebGPU } from "../webgpu-setup";

describe("Scene3D Extensibility", () => {
  let mockDevice: GPUDevice;
  let mockQueue: GPUQueue;
  let mockContext: GPUCanvasContext;

  beforeEach(() => {
    setupWebGPU();

    mockQueue = {
      writeBuffer: vi.fn(),
      submit: vi.fn(),
      onSubmittedWorkDone: vi.fn().mockResolvedValue(undefined),
    } as unknown as GPUQueue;

    mockContext = {
      configure: vi.fn(),
      getCurrentTexture: vi.fn(() => ({
        createView: vi.fn(),
      })),
    } as unknown as GPUCanvasContext;

    const createBuffer = vi.fn((desc: GPUBufferDescriptor) => ({
      destroy: vi.fn(),
      size: desc.size,
      usage: desc.usage,
      mapAsync: vi.fn().mockResolvedValue(undefined),
      getMappedRange: vi.fn(() => new ArrayBuffer(desc.size)),
      unmap: vi.fn(),
    }));

    mockDevice = {
      createBuffer,
      createBindGroup: vi.fn(),
      createBindGroupLayout: vi.fn(),
      createPipelineLayout: vi.fn(() => ({})),
      createRenderPipeline: vi.fn(),
      createShaderModule: vi.fn(() => ({})),
      createCommandEncoder: vi.fn(() => ({
        beginRenderPass: vi.fn(() => ({
          setPipeline: vi.fn(),
          setBindGroup: vi.fn(),
          setVertexBuffer: vi.fn(),
          setIndexBuffer: vi.fn(),
          draw: vi.fn(),
          drawIndexed: vi.fn(),
          end: vi.fn(),
        })),
        finish: vi.fn(),
      })),
      createTexture: vi.fn(() => ({
        createView: vi.fn(),
        destroy: vi.fn(),
      })),
      queue: mockQueue,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    } as unknown as GPUDevice;

    Object.defineProperty(navigator, "gpu", {
      value: {
        requestAdapter: vi.fn().mockResolvedValue({
          requestDevice: vi.fn().mockResolvedValue(mockDevice),
        }),
        getPreferredCanvasFormat: vi.fn().mockReturnValue("rgba8unorm"),
      },
      configurable: true,
    });

    Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
      value: (type: string) => (type === "webgpu" ? mockContext : null),
      configurable: true,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
    cleanupWebGPU();
  });

  it("should render a custom mesh defined via defineMesh", async () => {
    const customGeometry = {
      vertexData: new Float32Array([0, 0, 0, 0, 0, 1]), // 1 vertex: pos + norm
      indexData: new Uint16Array([0]),
    };
    
    const CustomMeshSpec = defineMesh("CustomMesh", customGeometry);
    
    const components = [
      {
        type: "CustomMesh",
        centers: new Float32Array([1, 2, 3]),
        scales: new Float32Array([1, 1, 1]),
      } as any,
    ];

    await act(async () => {
      render(
        <SceneImpl
          components={components}
          containerWidth={800}
          containerHeight={600}
          primitiveSpecs={{ CustomMesh: CustomMeshSpec }}
        />,
      );
    });

    // Verify buffer creation for the custom geometry
    const createBuffer = mockDevice.createBuffer as Mock;
    expect(createBuffer).toHaveBeenCalled();
    
    // Verify that the custom spec's geometry resource was created
    // We can't easily check internal cache but we can check if it attempted to render
    expect(mockQueue.writeBuffer).toHaveBeenCalled();
  });

  it("should support screenspace_offset transform", async () => {
    const ScreenspaceSpec = definePrimitive({
      name: "ScreenspaceObj",
      attributes: {
        anchor: attr.vec3("anchors"),
        offset: attr.vec3("offsets"),
        size: attr.f32("sizes", 1.0),
        color: attr.vec3("colors", [1, 1, 1]),
        alpha: attr.f32("alphas", 1.0),
      },
      geometry: { type: "sphere" },
      transform: "screenspace_offset",
    });

    const components = [
      {
        type: "ScreenspaceObj",
        anchors: new Float32Array([0, 0, 0]),
        offsets: new Float32Array([1, 0, 0]),
        sizes: new Float32Array([10]),
      } as any,
    ];

    await act(async () => {
      render(
        <SceneImpl
          components={components}
          containerWidth={800}
          containerHeight={600}
          primitiveSpecs={{ ScreenspaceObj: ScreenspaceSpec }}
        />,
      );
    });

    // Verify that the vertex shader generated contains screenspaceScale logic
    const createShaderModule = mockDevice.createShaderModule as Mock;
    const shaderCalls = createShaderModule.mock.calls;
    const vsCall = shaderCalls.find(call => call[0].code.includes("screenspaceScale"));
    expect(vsCall).toBeDefined();
    expect(vsCall[0].code).toContain("clipAnchor.w * 0.001 * size");
  });
});
