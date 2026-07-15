/// <reference types="@webgpu/types" />
import { describe, it, expect, vi, beforeEach, afterEach, Mock } from "vitest";
import { render, act } from "@testing-library/react";
import React from "react";
import { Scene, PointCloud } from "../../../src/js/scene3d";
import { setupWebGPU, cleanupWebGPU } from "../webgpu-setup";

describe("Scene3D Standalone", () => {
  let container: HTMLDivElement;
  let mockDevice: GPUDevice;
  let mockQueue: GPUQueue;
  let mockContext: GPUCanvasContext;
  let originalResizeObserver: typeof ResizeObserver | undefined;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);

    setupWebGPU();

    originalResizeObserver = globalThis.ResizeObserver;
    globalThis.ResizeObserver = class {
      observe() {}
      disconnect() {}
    } as typeof ResizeObserver;

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
      createPipelineLayout: vi.fn((desc: GPUPipelineLayoutDescriptor) => ({
        label: "Mock Pipeline Layout",
      })),
      createRenderPipeline: vi.fn(),
      createShaderModule: vi.fn((desc: GPUShaderModuleDescriptor) => ({
        label: "Mock Shader Module",
      })),
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
      createTexture: vi.fn((desc: GPUTextureDescriptor) => ({
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

    const mockGetContext = vi.fn((contextType: string) => {
      if (contextType === "webgpu") {
        return mockContext;
      }
      return null;
    });

    Object.defineProperty(HTMLCanvasElement.prototype, "getContext", {
      value: mockGetContext,
      configurable: true,
    });
  });

  afterEach(() => {
    document.body.removeChild(container);
    vi.clearAllMocks();
    cleanupWebGPU();

    if (originalResizeObserver) {
      globalThis.ResizeObserver = originalResizeObserver;
    } else {
      delete (globalThis as any).ResizeObserver;
    }
  });

  it("renders without a Colight state provider", async () => {
    const components = [
      PointCloud({
        centers: [0, 0, 0, 1, 1, 1],
        colors: [1, 0, 0, 0, 1, 0],
      }),
    ];

    let result;
    await act(async () => {
      result = render(
        <Scene components={components} width={400} height={300} />,
      );
    });

    const canvas = result!.container.querySelector("canvas");
    expect(canvas).toBeDefined();

    const createRenderPipeline = mockDevice.createRenderPipeline as Mock;
    expect(createRenderPipeline).toHaveBeenCalled();
  });
});
