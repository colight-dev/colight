/**
 * @module quaternion
 * @description Utilities for quaternion-based rotations in 3D
 */

/**
 * Creates a quaternion from axis-angle representation
 *
 * @param axis - Rotation axis (must be normalized)
 * @param angle - Rotation angle in radians
 * @returns Quaternion as [x, y, z, w]
 */
export function fromAxisAngle(
  axis: [number, number, number],
  angle: number,
): [number, number, number, number] {
  const halfAngle = angle * 0.5;
  const s = Math.sin(halfAngle);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(halfAngle)];
}

/**
 * Creates a quaternion from Euler angles (in radians)
 * Uses the ZYX convention (yaw, pitch, roll)
 *
 * @param yaw - Rotation around Z axis
 * @param pitch - Rotation around Y axis
 * @param roll - Rotation around X axis
 * @returns Quaternion as [x, y, z, w]
 */
export function fromEuler(
  yaw: number,
  pitch: number,
  roll: number,
): [number, number, number, number] {
  const cy = Math.cos(yaw * 0.5);
  const sy = Math.sin(yaw * 0.5);
  const cp = Math.cos(pitch * 0.5);
  const sp = Math.sin(pitch * 0.5);
  const cr = Math.cos(roll * 0.5);
  const sr = Math.sin(roll * 0.5);

  return [
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
    cr * cp * cy + sr * sp * sy,
  ];
}

/**
 * Normalizes a quaternion
 *
 * @param q - Quaternion to normalize
 * @returns Normalized quaternion
 */
export function normalize(
  q: [number, number, number, number],
): [number, number, number, number] {
  const len = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (len < 1e-10) {
    return [0, 0, 0, 1]; // Identity quaternion
  }
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

/**
 * Gets the rotation matrix (3x3) from a quaternion
 *
 * @param q - Quaternion [x, y, z, w]
 * @returns 3x3 rotation matrix as an array of 9 numbers (column-major)
 */
export function toMatrix3(q: [number, number, number, number]): number[] {
  const x = q[0],
    y = q[1],
    z = q[2],
    w = q[3];
  const xx = x * x,
    yy = y * y,
    zz = z * z;
  const xy = x * y,
    xz = x * z,
    yz = y * z;
  const wx = w * x,
    wy = w * y,
    wz = w * z;

  // Column-major order (WebGPU convention)
  return [
    1 - 2 * (yy + zz),
    2 * (xy + wz),
    2 * (xz - wy),
    2 * (xy - wz),
    1 - 2 * (xx + zz),
    2 * (yz + wx),
    2 * (xz + wy),
    2 * (yz - wx),
    1 - 2 * (xx + yy),
  ];
}

/**
 * Multiplies two quaternions
 *
 * @param a - First quaternion [x, y, z, w]
 * @param b - Second quaternion [x, y, z, w]
 * @returns a * b
 */
export function multiply(
  a: [number, number, number, number],
  b: [number, number, number, number],
): [number, number, number, number] {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

/**
 * Creates an identity quaternion (no rotation)
 *
 * @returns Identity quaternion [0, 0, 0, 1]
 */
export function identity(): [number, number, number, number] {
  return [0, 0, 0, 1];
}

/**
 * Rotates a vector by a quaternion
 *
 * @param v - Vector to rotate [x, y, z]
 * @param q - Quaternion to rotate by [x, y, z, w]
 * @returns Rotated vector [x', y', z']
 */
export function rotateVector(
  v: [number, number, number],
  q: [number, number, number, number],
): [number, number, number] {
  // Convert vector to quaternion with w=0
  const vq: [number, number, number, number] = [v[0], v[1], v[2], 0];

  // q * v * q^-1
  // For unit quaternions, q^-1 = [-x, -y, -z, w]
  const qInv: [number, number, number, number] = [-q[0], -q[1], -q[2], q[3]];

  // First multiply q * v
  const temp = multiply(q, vq);

  // Then multiply (q * v) * q^-1
  const result = multiply(temp, qInv);

  // The result's vector part is the rotated vector
  return [result[0], result[1], result[2]];
}

/**
 * WGSL shader code for quaternion operations
 * Provides functions for:
 * - Quaternion multiplication
 * - Quaternion rotation of a vector
 */
export const quaternionShaderFunctions = /*wgsl*/ `
// Multiply two quaternions (a * b)
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
  );
}

// Rotate a vector by a quaternion
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  // Calculate vector part of quaternion product
  let qv = vec3<f32>(q.x, q.y, q.z);
  let uv = cross(qv, v);
  let uuv = cross(qv, uv);
  // v' = v + 2(q.w * uv + uuv)
  return v + 2.0 * (q.w * uv + uuv);
}
`;
