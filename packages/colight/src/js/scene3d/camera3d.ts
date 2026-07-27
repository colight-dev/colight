import * as glMatrix from "gl-matrix";
import type { TypedArray } from "../binary";

export interface CameraParams {
  position: [number, number, number] | TypedArray;
  target: [number, number, number] | TypedArray;
  up: [number, number, number] | TypedArray;
  fov: number;
  near: number;
  far: number;
}

export interface CameraState {
  position: glMatrix.vec3;
  target: glMatrix.vec3;
  up: glMatrix.vec3;
  fov: number;
  near: number;
  far: number;
}

/**
 * Tracks the state of an active mouse drag interaction.
 * Used for camera control operations.
 */
export interface DraggingState {
  /** Which mouse button initiated the drag (0=left, 1=middle, 2=right) */
  button: number;

  /** Initial X coordinate when drag started */
  startX: number;

  /** Initial Y coordinate when drag started */
  startY: number;

  /** Current X coordinate */
  x: number;

  /** Current Y coordinate */
  y: number;

  /** Keyboard modifiers active when drag started */
  modifiers: string[];

  /** Initial camera state when drag started */
  startCam: CameraState;

  /** Canvas bounding rect when drag started */
  rect: DOMRect;
}

export const DEFAULT_CAMERA: CameraParams = {
  position: [2, 2, 2], // Simple diagonal view
  target: [0, 0, 0],
  up: [0, 1, 0],
  fov: 45, // Slightly narrower FOV for better perspective
  near: 0.001,
  far: 100.0,
};

export function createCameraState(
  params: CameraParams | null | undefined,
): CameraState {
  const p = {
    position: params?.position ?? DEFAULT_CAMERA.position,
    target: params?.target ?? DEFAULT_CAMERA.target,
    up: params?.up ?? DEFAULT_CAMERA.up,
    fov: params?.fov ?? DEFAULT_CAMERA.fov,
    near: params?.near ?? DEFAULT_CAMERA.near,
    far: params?.far ?? DEFAULT_CAMERA.far,
  };

  const position = glMatrix.vec3.fromValues(
    p.position[0],
    p.position[1],
    p.position[2],
  );
  const target = glMatrix.vec3.fromValues(
    p.target[0],
    p.target[1],
    p.target[2],
  );
  const up = glMatrix.vec3.fromValues(p.up[0], p.up[1], p.up[2]);
  glMatrix.vec3.normalize(up, up);

  return {
    position,
    target,
    up,
    fov: p.fov,
    near: p.near,
    far: p.far,
  };
}

export function createCameraParams(state: CameraState): CameraParams {
  return {
    position: Array.from(state.position) as [number, number, number],
    target: Array.from(state.target) as [number, number, number],
    up: Array.from(state.up) as [number, number, number],
    fov: state.fov,
    near: state.near,
    far: state.far,
  };
}

/**
 * Gets the current radius (distance from camera position to target)
 */
function getRadius(camera: CameraState): number {
  const dir = glMatrix.vec3.sub(
    glMatrix.vec3.create(),
    camera.position,
    camera.target,
  );
  return glMatrix.vec3.length(dir);
}

/**
 * Orbit the camera around the target, using the camera's 'up' as the vertical axis.
 * Takes the current camera state and drag state to calculate the orbit.
 */
export function orbit(dragState: DraggingState): CameraState {
  const deltaX = dragState.x - dragState.startX;
  const deltaY = dragState.y - dragState.startY;

  const { target, up } = dragState.startCam;
  const radius = getRadius(dragState.startCam);
  const useQuaternions = false; // Flag to switch between rotation methods

  if (useQuaternions) {
    // Get current direction from camera to target
    const dir = glMatrix.vec3.sub(
      glMatrix.vec3.create(),
      dragState.startCam.position,
      target,
    );
    glMatrix.vec3.normalize(dir, dir);

    // Calculate rotation axis perpendicular to drag direction and view direction
    const dragVector = glMatrix.vec3.fromValues(-deltaX, deltaY, 0);
    const dragLength = glMatrix.vec3.length(dragVector);

    if (dragLength > 0) {
      // Project view direction to screen plane (z=0)
      const screenDir = glMatrix.vec3.fromValues(dir[0], dir[1], 0);
      glMatrix.vec3.normalize(screenDir, screenDir);

      // Get rotation axis perpendicular to drag and view
      const rotationAxis = glMatrix.vec3.cross(
        glMatrix.vec3.create(),
        dragVector,
        glMatrix.vec3.fromValues(0, 0, -1), // Into screen
      );
      glMatrix.vec3.normalize(rotationAxis, rotationAxis);

      // Create single rotation quaternion
      const angle = dragLength * 0.01;
      const rotation = glMatrix.quat.setAxisAngle(
        glMatrix.quat.create(),
        rotationAxis,
        angle,
      );

      // Apply rotation to direction vector
      const rotatedDir = glMatrix.vec3.transformQuat(
        glMatrix.vec3.create(),
        dir,
        rotation,
      );
      glMatrix.vec3.scale(rotatedDir, rotatedDir, radius);

      // Calculate new position
      const newPosition = glMatrix.vec3.add(
        glMatrix.vec3.create(),
        target,
        rotatedDir,
      );

      return {
        ...dragState.startCam,
        position: newPosition,
      };
    }
    return dragState.startCam;
  } else {
    // Get current direction from target to camera
    const dir = glMatrix.vec3.sub(
      glMatrix.vec3.create(),
      dragState.startCam.position,
      target,
    );
    glMatrix.vec3.normalize(dir, dir);

    // Get current phi (angle from up to dir)
    const upDot = glMatrix.vec3.dot(up, dir);
    let phi = Math.acos(clamp(upDot, -1.0, 1.0));

    // Get current theta (angle around up axis)
    const { refForward, refRight } = getReferenceFrame(up);
    const x = glMatrix.vec3.dot(dir, refRight);
    const z = glMatrix.vec3.dot(dir, refForward);
    let theta = Math.atan2(x, z);

    // Adjust angles based on mouse movement
    theta -= deltaX * 0.01;
    phi -= deltaY * 0.01;

    // Clamp phi to avoid gimbal lock at poles
    phi = Math.max(0.001, Math.min(Math.PI - 0.001, phi));

    // Compute new position in spherical coordinates
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    const newPosition = glMatrix.vec3.create();
    glMatrix.vec3.scaleAndAdd(newPosition, newPosition, up, cosPhi * radius);
    glMatrix.vec3.scaleAndAdd(
      newPosition,
      newPosition,
      refForward,
      sinPhi * cosTheta * radius,
    );
    glMatrix.vec3.scaleAndAdd(
      newPosition,
      newPosition,
      refRight,
      sinPhi * sinTheta * radius,
    );
    glMatrix.vec3.add(newPosition, target, newPosition);

    return {
      ...dragState.startCam,
      position: newPosition,
    };
  }
}

/**
 * Pan the camera in the plane perpendicular to the view direction,
 * using the camera's 'up' as the orientation reference for 'right'.
 */
export function pan(dragState: DraggingState): CameraState {
  const deltaX = dragState.x - dragState.startX;
  const deltaY = dragState.y - dragState.startY;

  // forward = (target - position)
  const forward = glMatrix.vec3.sub(
    glMatrix.vec3.create(),
    dragState.startCam.target,
    dragState.startCam.position,
  );
  // right = forward x up
  const right = glMatrix.vec3.cross(
    glMatrix.vec3.create(),
    forward,
    dragState.startCam.up,
  );
  glMatrix.vec3.normalize(right, right);

  // actualUp = right x forward
  const actualUp = glMatrix.vec3.cross(glMatrix.vec3.create(), right, forward);
  glMatrix.vec3.normalize(actualUp, actualUp);

  // Scale movement by distance from target
  const scale = getRadius(dragState.startCam) * 0.002;
  const movement = glMatrix.vec3.create();

  // Move along the local right/actualUp vectors
  glMatrix.vec3.scaleAndAdd(movement, movement, right, -deltaX * scale);
  glMatrix.vec3.scaleAndAdd(movement, movement, actualUp, deltaY * scale);

  // Update position and target
  const newPosition = glMatrix.vec3.add(
    glMatrix.vec3.create(),
    dragState.startCam.position,
    movement,
  );
  const newTarget = glMatrix.vec3.add(
    glMatrix.vec3.create(),
    dragState.startCam.target,
    movement,
  );

  return {
    ...dragState.startCam,
    position: newPosition,
    target: newTarget,
  };
}

/**
 * Roll the camera around the view direction axis.
 * Computes the angle between the initial and current mouse positions relative to the target.
 */
export function roll(dragState: DraggingState): CameraState {
  const { target, position, up } = dragState.startCam;
  const rect = dragState.rect;

  // Convert mouse coordinates to be relative to target's screen position
  // First get view and projection matrices
  const view = getViewMatrix(dragState.startCam);
  const aspect = rect.width / rect.height;
  const proj = getProjectionMatrix(dragState.startCam, aspect);

  // Project target point to screen space
  const targetScreen = glMatrix.vec4.fromValues(
    target[0],
    target[1],
    target[2],
    1.0,
  );
  glMatrix.vec4.transformMat4(targetScreen, targetScreen, view);
  glMatrix.vec4.transformMat4(targetScreen, targetScreen, proj);

  // Convert to NDC and then to screen coordinates
  // Note: For WebGPU Y is inverted compared to WebGL (+Y is down)
  const targetX = (targetScreen[0] / targetScreen[3] + 1) * 0.5 * rect.width;
  const targetY = (targetScreen[1] / targetScreen[3] + 1) * 0.5 * rect.height;

  // Get vectors from target to start and current mouse positions
  const startVec = [dragState.startX - targetX, dragState.startY - targetY];
  const currentVec = [dragState.x - targetX, dragState.y - targetY];

  // Compute angle between vectors
  const startLen = Math.sqrt(
    startVec[0] * startVec[0] + startVec[1] * startVec[1],
  );
  const currentLen = Math.sqrt(
    currentVec[0] * currentVec[0] + currentVec[1] * currentVec[1],
  );

  if (startLen < 1e-6 || currentLen < 1e-6) return dragState.startCam;

  const cosAngle =
    (startVec[0] * currentVec[0] + startVec[1] * currentVec[1]) /
    (startLen * currentLen);
  const crossProduct =
    startVec[0] * currentVec[1] - startVec[1] * currentVec[0];
  // Negate angle since WebGPU Y coordinates are flipped compared to WebGL
  const angle = -Math.acos(clamp(cosAngle, -1, 1)) * Math.sign(crossProduct);

  // Get view direction
  const viewDir = glMatrix.vec3.sub(glMatrix.vec3.create(), target, position);
  glMatrix.vec3.normalize(viewDir, viewDir);

  // Create rotation matrix around view direction
  const rotationMatrix = glMatrix.mat4.create();
  glMatrix.mat4.rotate(rotationMatrix, rotationMatrix, angle, viewDir);

  // Rotate up vector
  const newUp = glMatrix.vec3.create();
  glMatrix.vec3.transformMat4(newUp, up, rotationMatrix);

  return {
    ...dragState.startCam,
    up: newUp,
  };
}

/**
 * Zoom the camera in/out by moving along the view direction while keeping the target fixed.
 */
export function zoom(camera: CameraState, deltaY: number): CameraState {
  // Get current distance from target
  const direction = glMatrix.vec3.sub(
    glMatrix.vec3.create(),
    camera.position,
    camera.target,
  );
  const distance = glMatrix.vec3.length(direction);

  // Exponential zoom factor
  const newDistance = Math.max(0.01, distance * Math.exp(deltaY * 0.001));

  // Move the camera position accordingly
  glMatrix.vec3.normalize(direction, direction);
  const newPosition = glMatrix.vec3.scaleAndAdd(
    glMatrix.vec3.create(),
    camera.target,
    direction,
    newDistance,
  );

  return {
    ...camera,
    position: newPosition,
  };
}

/**
 * Move both camera and target along the view direction, creating a "fly through" effect.
 */
export function dolly(camera: CameraState, deltaY: number): CameraState {
  // Get normalized view direction
  const direction = glMatrix.vec3.sub(
    glMatrix.vec3.create(),
    camera.position,
    camera.target,
  );
  glMatrix.vec3.normalize(direction, direction);

  // Calculate movement distance
  const moveAmount = deltaY * 0.001;

  // Move both position and target
  const newPosition = glMatrix.vec3.scaleAndAdd(
    glMatrix.vec3.create(),
    camera.position,
    direction,
    moveAmount,
  );

  const newTarget = glMatrix.vec3.scaleAndAdd(
    glMatrix.vec3.create(),
    camera.target,
    direction,
    moveAmount,
  );

  return {
    ...camera,
    position: newPosition,
    target: newTarget,
  };
}

/**
 * Adjust the camera's field of view (FOV), creating a zoom-like effect that preserves perspective.
 */
export function adjustFov(camera: CameraState, deltaY: number): CameraState {
  // Calculate new FOV with exponential scaling
  const newFov = clamp(
    camera.fov * Math.exp(deltaY * 0.001),
    10, // Min FOV in degrees
    120, // Max FOV in degrees
  );

  return {
    ...camera,
    fov: newFov,
  };
}

export function getViewMatrix(camera: CameraState): Float32Array {
  return glMatrix.mat4.lookAt(
    glMatrix.mat4.create(),
    camera.position,
    camera.target,
    camera.up,
  ) as Float32Array;
}

// Convert degrees to radians
function degreesToRadians(degrees: number): number {
  return degrees * (Math.PI / 180);
}

// Create a perspective projection matrix, converting FOV from degrees
export function getProjectionMatrix(
  camera: CameraState,
  aspect: number,
): Float32Array {
  return glMatrix.mat4.perspective(
    glMatrix.mat4.create(),
    degreesToRadians(camera.fov), // Convert FOV to radians
    aspect,
    camera.near,
    camera.far,
  ) as Float32Array;
}

/**
 * Build a local reference frame around the 'up' vector so we can
 * measure angles (phi/theta) consistently in an "any up" scenario.
 */
function getReferenceFrame(up: glMatrix.vec3): {
  refForward: glMatrix.vec3;
  refRight: glMatrix.vec3;
} {
  // Try worldForward = (0, 0, 1). If that's collinear with 'up', fallback to (1, 0, 0)
  const EPS = 1e-8;
  const worldForward = glMatrix.vec3.fromValues(0, 0, 1);
  let crossVal = glMatrix.vec3.cross(glMatrix.vec3.create(), up, worldForward);

  if (glMatrix.vec3.length(crossVal) < EPS) {
    // up is nearly parallel with (0,0,1)
    crossVal = glMatrix.vec3.cross(
      glMatrix.vec3.create(),
      up,
      glMatrix.vec3.fromValues(1, 0, 0),
    );
  }
  glMatrix.vec3.normalize(crossVal, crossVal);
  const refRight = crossVal; // X-axis
  const refForward = glMatrix.vec3.cross(glMatrix.vec3.create(), refRight, up); // Z-axis
  glMatrix.vec3.normalize(refForward, refForward);

  return { refForward, refRight };
}

/** Clamps a value x to the [minVal, maxVal] range. */
function clamp(x: number, minVal: number, maxVal: number): number {
  return Math.max(minVal, Math.min(x, maxVal));
}

export function hasCameraMoved(
  current: glMatrix.vec3,
  last: glMatrix.vec3 | undefined,
  threshold: number,
): boolean {
  if (!last) return true;
  const dx = current[0] - last[0];
  const dy = current[1] - last[1];
  const dz = current[2] - last[2];
  return dx * dx + dy * dy + dz * dz > threshold;
}
