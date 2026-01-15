/**
 * @module gpu-transforms
 * @description GPU transform buffer utilities for Scene3D.
 *
 * This module handles building and managing the transform storage buffer
 * that allows group transforms to be applied on the GPU rather than CPU.
 */

import { Transform } from "./groups";
import { Vec3 } from "./vec3";

// Define identity constants locally to avoid circular dependency with groups.ts
// These must match the values in groups.ts
const IDENTITY_POS: Vec3 = [0, 0, 0];
const IDENTITY_QUAT: [number, number, number, number] = [0, 0, 0, 1];

// =============================================================================
// Types
// =============================================================================

/**
 * GPU-side transform layout (48 bytes, 12 floats).
 * Matches the WGSL GroupTransform struct with proper alignment.
 */
export interface GPUTransform {
  position: Vec3; // 3 floats + 1 pad
  quaternion: [number, number, number, number]; // 4 floats (xyzw)
  scale: Vec3; // 3 floats + 1 pad
}

/** Number of floats per transform in the GPU buffer */
export const FLOATS_PER_TRANSFORM = 12;

/** Byte size of each transform (48 bytes) */
export const BYTES_PER_TRANSFORM = FLOATS_PER_TRANSFORM * 4;

// =============================================================================
// Identity Transform
// =============================================================================

/**
 * Identity transform: no translation, no rotation, uniform scale of 1.
 * This is always at index 0 in the transforms buffer.
 */
export const IDENTITY_GPU_TRANSFORM: GPUTransform = {
  position: IDENTITY_POS,
  quaternion: IDENTITY_QUAT,
  scale: [1, 1, 1],
};

// =============================================================================
// Transform Buffer Building
// =============================================================================

/**
 * Convert a Transform to GPUTransform format.
 * The quaternion is already in xyzw order in groups.ts, matching GPU expectations.
 */
export function toGPUTransform(transform: Transform): GPUTransform {
  return {
    position: transform.position,
    quaternion: transform.quaternion,
    scale: transform.scale,
  };
}

/**
 * Check if a transform is effectively identity (no-op).
 */
export function isIdentityTransform(transform: GPUTransform): boolean {
  const { position, quaternion, scale } = transform;
  return (
    position[0] === 0 &&
    position[1] === 0 &&
    position[2] === 0 &&
    quaternion[0] === 0 &&
    quaternion[1] === 0 &&
    quaternion[2] === 0 &&
    quaternion[3] === 1 &&
    scale[0] === 1 &&
    scale[1] === 1 &&
    scale[2] === 1
  );
}

/**
 * Pack an array of GPUTransforms into a Float32Array for GPU upload.
 *
 * Layout per transform (48 bytes / 12 floats):
 * - position.x, position.y, position.z, _pad0
 * - quaternion.x, quaternion.y, quaternion.z, quaternion.w
 * - scale.x, scale.y, scale.z, _pad1
 *
 * @param transforms Array of transforms (index 0 should be identity)
 * @returns Float32Array ready for GPU buffer upload
 */
export function packTransformsToBuffer(
  transforms: GPUTransform[],
): Float32Array {
  const buffer = new Float32Array(transforms.length * FLOATS_PER_TRANSFORM);

  for (let i = 0; i < transforms.length; i++) {
    const t = transforms[i];
    const offset = i * FLOATS_PER_TRANSFORM;

    // Position + pad
    buffer[offset + 0] = t.position[0];
    buffer[offset + 1] = t.position[1];
    buffer[offset + 2] = t.position[2];
    buffer[offset + 3] = 0; // padding

    // Quaternion (xyzw)
    buffer[offset + 4] = t.quaternion[0];
    buffer[offset + 5] = t.quaternion[1];
    buffer[offset + 6] = t.quaternion[2];
    buffer[offset + 7] = t.quaternion[3];

    // Scale + pad
    buffer[offset + 8] = t.scale[0];
    buffer[offset + 9] = t.scale[1];
    buffer[offset + 10] = t.scale[2];
    buffer[offset + 11] = 0; // padding
  }

  return buffer;
}

// =============================================================================
// Shader Declarations
// =============================================================================

/**
 * WGSL struct declaration for GroupTransform.
 * This should be included in vertex shaders that use group transforms.
 */
export const groupTransformStruct = `
struct GroupTransform {
  position: vec3<f32>,
  _pad0: f32,
  quaternion: vec4<f32>,
  scale: vec3<f32>,
  _pad1: f32,
};
`;

/**
 * WGSL storage buffer binding declaration for transforms.
 * Uses @group(0) @binding(1) - binding 0 is camera uniforms.
 */
export const transformsBufferBinding = `
@group(0) @binding(1) var<storage, read> transforms: array<GroupTransform>;
`;

/**
 * WGSL helper function to apply a group transform to a position.
 * Applies scale, then rotation, then translation.
 */
export const applyGroupTransformFn = `
fn applyGroupTransform(pos: vec3<f32>, idx: u32) -> vec3<f32> {
  let t = transforms[idx];
  let scaled = pos * t.scale;
  let rotated = quat_rotate(t.quaternion, scaled);
  return rotated + t.position;
}
`;

/**
 * Combined shader preamble for group transforms.
 * Include this in vertex shaders that need group transform support.
 */
export const groupTransformShaderPreamble =
  groupTransformStruct + transformsBufferBinding + applyGroupTransformFn;
