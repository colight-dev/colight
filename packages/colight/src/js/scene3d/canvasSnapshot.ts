/**
 * Canvas snapshot utilities for scene3d.
 *
 * Provides functionality to capture WebGPU canvas content as images,
 * useful for PDF export and screenshot features.
 */

import { useEffect, useRef, useMemo } from "react";

interface CanvasEntry {
  canvas: HTMLCanvasElement;
  overlay?: HTMLCanvasElement;
  device?: GPUDevice;
  context?: GPUCanvasContext;
  renderCallback?: (
    texture: GPUTexture,
    depthTexture: GPUTexture | null,
  ) => Promise<void>;
}

interface CanvasRegistry {
  [key: string]: CanvasEntry;
}

// Global registry of active canvases
const activeCanvases: CanvasRegistry = {};

/**
 * Hook to register a canvas element for snapshot functionality.
 *
 * @param device - Optional WebGPU device to use for texture copying
 * @param context - Optional WebGPU context for rendering
 * @param renderCallback - Optional callback to render the scene to a texture
 * @returns Object containing the ref to attach to canvas
 */
export function useCanvasSnapshot(
  device?: GPUDevice,
  context?: GPUCanvasContext,
  renderCallback?: (
    texture: GPUTexture,
    depthTexture: GPUTexture | null,
  ) => Promise<void>,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const id = useMemo(
    () => `scene3d_${Math.random().toString(36).slice(2)}`,
    [],
  );

  useEffect(() => {
    if (canvasRef.current) {
      // Register canvas when mounted
      activeCanvases[id] = {
        canvas: canvasRef.current,
        device,
        context,
        renderCallback,
      };

      // Cleanup on unmount
      return () => {
        delete activeCanvases[id];
      };
    }
  }, [id, device, context, renderCallback]);

  return {
    canvasRef,
    getActiveCanvases: () => Object.keys(activeCanvases),
  };
}

/**
 * Creates image overlays for all registered WebGPU canvases.
 * Used before PDF export to capture 3D content as static images.
 *
 * @returns Promise that resolves when all overlays are created
 */
export async function createCanvasOverlays(): Promise<void> {
  const canvasEntries = Object.entries(activeCanvases);

  for (const [idx, [id, entry]] of canvasEntries.entries()) {
    // Stagger to allow GPU work to complete
    if (idx % 2 === 0) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    const { canvas, device, context, renderCallback } = entry;
    if (!device || !context || !renderCallback) {
      console.warn(
        `[scene3d] Missing required WebGPU resources for canvas ${id}`,
      );
      continue;
    }

    const width = canvas.width;
    const height = canvas.height;

    // Calculate aligned bytes per row (must be multiple of 256)
    const bytesPerPixel = 4; // RGBA8
    const bytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
    const alignedBufferSize = bytesPerRow * height;

    // Create a texture to render the scene to
    const texture = device.createTexture({
      size: [width, height],
      format: navigator.gpu.getPreferredCanvasFormat(),
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Create a depth texture for the render pass
    const depthTexture = device.createTexture({
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Render the scene to our texture
    await renderCallback(texture, depthTexture);

    // Create a buffer to read back the pixel data
    const readbackBuffer = device.createBuffer({
      size: alignedBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Copy from our texture to the buffer
    const commandEncoder = device.createCommandEncoder();
    commandEncoder.copyTextureToBuffer(
      { texture },
      {
        buffer: readbackBuffer,
        bytesPerRow,
        rowsPerImage: height,
      },
      [width, height, 1],
    );
    device.queue.submit([commandEncoder.finish()]);

    // Wait for the copy to complete and map the buffer
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const mappedData = new Uint8Array(readbackBuffer.getMappedRange());

    // Create a properly sized array for the actual pixel data
    const pixelData = new Uint8Array(width * height * bytesPerPixel);

    // Create Uint32Array views for faster processing
    const mappedU32 = new Uint32Array(mappedData.buffer);
    const pixelU32 = new Uint32Array(pixelData.buffer);

    // Copy rows accounting for alignment and convert BGRA to RGBA
    for (let row = 0; row < height; row++) {
      const sourceRowStart = (row * bytesPerRow) / 4;
      const targetRowStart = row * width;

      for (let x = 0; x < width; x++) {
        const pixel = mappedU32[sourceRowStart + x];
        const b = pixel & 0x000000ff;
        const g = pixel & 0x0000ff00;
        const r = pixel & 0x00ff0000;
        const a = pixel & 0xff000000;
        pixelU32[targetRowStart + x] = a | (b << 16) | g | (r >> 16);
      }
    }

    // Create canvas and draw the pixel data
    const overlayCanvas = document.createElement("canvas");
    overlayCanvas.width = width;
    overlayCanvas.height = height;
    const ctx = overlayCanvas.getContext("2d")!;
    const imageData = ctx.createImageData(width, height);
    imageData.data.set(pixelData);
    ctx.putImageData(imageData, 0, 0);

    // Position the overlay canvas absolutely within the parent container
    overlayCanvas.style.position = "absolute";
    overlayCanvas.style.left = "0";
    overlayCanvas.style.top = "0";
    overlayCanvas.style.width = "100%";
    overlayCanvas.style.height = "100%";
    overlayCanvas.style.objectFit = "cover";
    overlayCanvas.style.opacity = "100%";

    // Add to parent container
    const parentContainer = canvas.parentElement;
    if (!parentContainer) {
      console.warn("[scene3d] Canvas has no parent element");
      continue;
    }
    parentContainer.appendChild(overlayCanvas);
    entry.overlay = overlayCanvas;

    // Cleanup resources
    try {
      readbackBuffer.unmap();
      readbackBuffer.destroy();
      texture.destroy();
      depthTexture.destroy();
    } catch (err) {
      console.warn("[scene3d] Error during resource cleanup:", err);
    }
  }
}

/**
 * Removes all canvas overlays created by createCanvasOverlays.
 * Call after PDF generation is complete to restore interactive WebGPU canvases.
 */
export function removeCanvasOverlays(): void {
  Object.values(activeCanvases).forEach((entry) => {
    if (entry.overlay) {
      entry.overlay.remove();
      delete entry.overlay;
    }
  });
}

/**
 * Get the current count of registered canvases.
 * Useful for debugging or status reporting.
 */
export function getCanvasCount(): number {
  return Object.keys(activeCanvases).length;
}
