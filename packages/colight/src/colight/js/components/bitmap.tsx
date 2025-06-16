import * as React from "react";
import { useContext, useEffect, useMemo, useRef } from "react";
import { $StateContext } from "../context";
import { useContainerWidth } from "../utils";

/**
 * Props interface for the bitmap component
 */
interface BitmapProps {
  /** Raw pixel data as Uint8Array/Uint8ClampedArray */
  pixels: Uint8Array | Uint8ClampedArray;
  /** Width of the bitmap in pixels */
  width: number;
  /** Height of the bitmap in pixels */
  height: number;
  /** CSS styles to apply to the canvas element */
  style?: React.CSSProperties;
  /** How to interpolate pixels when scaling */
  interpolation?: "nearest" | "bilinear";
}

/**
 * Renders raw pixel data as a bitmap image in a canvas element.
 *
 * Supports both RGB (3 bytes per pixel) and RGBA (4 bytes per pixel) formats.
 * RGB data is automatically converted to RGBA by setting alpha to 255.
 * The canvas is scaled to fit the container width while maintaining aspect ratio.
 *
 * @param pixels - Raw pixel data as Uint8Array/Uint8ClampedArray
 * @param width - Width of the bitmap in pixels
 * @param height - Height of the bitmap in pixels
 * @param interpolation - How to interpolate pixels when scaling
 * @returns React component rendering the bitmap
 */
export function Bitmap({
  pixels,
  width,
  height,
  interpolation = "nearest",
  style,
  ...props
}: BitmapProps) {
  const [ref, containerWidth] = useContainerWidth();
  const $state = useContext($StateContext);
  const done = useMemo(() => $state.beginUpdate("bitmap"), [containerWidth]);

  useEffect(() => {
    const container = ref.current;
    if (!container) {
      console.warn("Container ref not available");
      return;
    }

    // Remove any existing canvas elements
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    // Create new canvas
    const canvas = document.createElement("canvas");
    const displayWidth = containerWidth;
    const displayHeight = Math.round(displayWidth * (height / width));

    if (interpolation === "nearest" && containerWidth > width * 2) {
      // For nearest-neighbor, use high resolution canvas with temp canvas approach
      const scale = 2; // Use 4x resolution for crisp pixels
      canvas.width = displayWidth * scale;
      canvas.height = displayHeight * scale;
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;

      // Create temporary canvas for initial pixel data
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = width;
      tempCanvas.height = height;
      const tempCtx = tempCanvas.getContext("2d");
      if (!tempCtx) {
        console.warn("Could not get temp 2d context");
        return;
      }

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        console.warn("Could not get 2d context");
        return;
      }

      // Put pixels in temp canvas first
      const imageData = createImageData(pixels, width, height);
      tempCtx.putImageData(imageData, 0, 0);

      // Scale to final size with nearest neighbor interpolation
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
    } else {
      // For bilinear, render at original pixel dimensions and let browser handle scaling
      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${displayWidth}px`;
      canvas.style.height = `${displayHeight}px`;
      canvas.style.imageRendering =
        interpolation === "nearest" ? "pixelated" : "auto";

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        console.warn("Could not get 2d context");
        return;
      }

      // Draw pixels directly at original resolution
      const imageData = createImageData(pixels, width, height);
      ctx.putImageData(imageData, 0, 0);
    }

    Object.assign(canvas.style, style);
    container.appendChild(canvas);
    done();
  }, [pixels, width, height, interpolation, containerWidth, ref]);

  // Helper function to create appropriate ImageData from pixels
  function createImageData(
    pixels: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
  ): ImageData {
    const bytesPerPixel = pixels.length / (width * height);
    if (bytesPerPixel === 3) {
      // Convert RGB to RGBA
      const rgba = new Uint8ClampedArray(width * height * 4);
      for (let i = 0; i < pixels.length; i += 3) {
        const j = (i / 3) * 4;
        rgba[j] = pixels[i];
        rgba[j + 1] = pixels[i + 1];
        rgba[j + 2] = pixels[i + 2];
        rgba[j + 3] = 255;
      }
      return new ImageData(rgba, width, height);
    } else {
      // Assume RGBA
      return new ImageData(new Uint8ClampedArray(pixels), width, height);
    }
  }

  return <div ref={ref} {...props} />;
}
