/**
 * @module pointer
 * @description Pointer context for tracking mouse/touch state over the scene.
 */

import { Vec3 } from "./vec3";
import { Ray } from "./ray";
import { CameraState } from "./camera3d";
import { PickInfo } from "./types";

/**
 * Current pointer state over the scene canvas.
 * Updated on every mouse move / pick update.
 */
export interface PointerContext {
  /** Current pointer position in canvas-relative pixels (null if not over canvas) */
  screen: { x: number; y: number } | null;

  /** Computed ray from current position (null if no position or invalid camera) */
  ray: Ray | null;

  /** Current pick info from GPU picking (null if nothing hovered) */
  pick: PickInfo | null;

  /** Canvas rect dimensions for coordinate transforms */
  rect: { width: number; height: number };

  /** Current camera state */
  camera: CameraState | null;
}

export type CursorType =
  | "auto"
  | "grab"
  | "grabbing"
  | "pointer"
  | "crosshair"
  | "move"
  | "default";

/**
 * Hint about what cursor to display.
 */
export interface CursorHint {
  /** Suggested cursor style */
  cursor: CursorType;

  /** Reason for the cursor hint */
  reason: "draggable" | "clickable" | "dragging" | "camera" | null;

  /** Component being hovered (if applicable) */
  component?: { index: number; type: string };
}

/**
 * Create an initial empty pointer context.
 */
export function createPointerContext(
  rect: { width: number; height: number } = { width: 0, height: 0 },
  camera: CameraState | null = null,
): PointerContext {
  return {
    screen: null,
    ray: null,
    pick: null,
    rect,
    camera,
  };
}

/**
 * Determine cursor hint from component props.
 */
export function getCursorHint(
  pick: PickInfo | null,
  isDragging: boolean,
  isCameraDragging: boolean,
): CursorHint {
  if (isCameraDragging) {
    return { cursor: "move", reason: "camera" };
  }

  if (isDragging) {
    return { cursor: "grabbing", reason: "dragging" };
  }

  if (!pick) {
    return { cursor: "default", reason: null };
  }

  // Check if the hovered component is draggable or clickable
  // We don't have direct access to component props here, so this will be
  // determined at the Scene level based on component configuration
  return { cursor: "default", reason: null };
}
