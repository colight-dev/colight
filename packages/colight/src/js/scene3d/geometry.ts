export function createSphereGeometry(stacks = 16, slices = 24) {
  const verts: number[] = [];
  const idxs: number[] = [];
  for (let i = 0; i <= stacks; i++) {
    const phi = (i / stacks) * Math.PI;
    const sp = Math.sin(phi),
      cp = Math.cos(phi);
    for (let j = 0; j <= slices; j++) {
      const theta = (j / slices) * 2 * Math.PI;
      const st = Math.sin(theta),
        ct = Math.cos(theta);
      const x = sp * ct,
        y = cp,
        z = sp * st;
      verts.push(x, y, z, x, y, z); // pos + normal
    }
  }
  for (let i = 0; i < stacks; i++) {
    for (let j = 0; j < slices; j++) {
      const row1 = i * (slices + 1) + j;
      const row2 = (i + 1) * (slices + 1) + j;
      // Reverse winding order by swapping vertices
      idxs.push(row1, row1 + 1, row2, row1 + 1, row2 + 1, row2); // Changed from (row1,row2,row1+1, row1+1,row2,row2+1)
    }
  }
  return {
    vertexData: new Float32Array(verts),
    indexData: new Uint16Array(idxs),
  };
}

/**
 * Generate tube geometry for the ellipsoid axes.
 *
 * @param majorRadius - The radius of the ring's centerline.
 * @param tubeRadius - The thickness of the tube.
 * @param majorSegments - Number of segments along the ring (u direction).
 * @param minorSegments - Number of segments around the tube (v direction).
 */
export function createEllipsoidAxes(
  majorRadius: number,
  tubeRadius: number,
  majorSegments: number,
  minorSegments: number,
) {
  const verts: number[] = [];
  const idxs: number[] = [];

  // Generate three rings: one for each principal plane (XY, XZ, YZ)
  for (let ring = 0; ring < 3; ring++) {
    const ringBaseIndex = verts.length / 9; // now 9 floats per vertex: center (3), offset (3), normal (3)

    // Loop along the ring’s centerline (u parameter)
    for (let i = 0; i <= majorSegments; i++) {
      const u = i / majorSegments;
      const theta = u * 2 * Math.PI;

      // Compute centerline position for the ring based on its plane.
      let cx = 0,
        cy = 0,
        cz = 0;
      let tangent: number[] = [0, 0, 0];
      let refNormal: number[] = [0, 0, 0];

      if (ring === 0) {
        // XY plane
        cx = majorRadius * Math.cos(theta);
        cy = majorRadius * Math.sin(theta);
        cz = 0;
        tangent = [-Math.sin(theta), Math.cos(theta), 0];
        refNormal = [0, 0, 1];
      } else if (ring === 1) {
        // XZ plane
        cx = majorRadius * Math.cos(theta);
        cy = 0;
        cz = majorRadius * Math.sin(theta);
        tangent = [-Math.sin(theta), 0, Math.cos(theta)];
        refNormal = [0, 1, 0];
      } else {
        // YZ plane
        cx = 0;
        cy = majorRadius * Math.cos(theta);
        cz = majorRadius * Math.sin(theta);
        tangent = [0, -Math.sin(theta), Math.cos(theta)];
        refNormal = [1, 0, 0];
      }

      // Compute Frenet frame for the tube’s cross-section.
      const binormal = normalize(cross(tangent, refNormal));
      const N = normalize(cross(binormal, tangent));

      // Loop around the tube's cross-section (v parameter)
      for (let j = 0; j <= minorSegments; j++) {
        const v = j / minorSegments;
        const phi = v * 2 * Math.PI;

        // Compute tube offset using the Frenet frame.
        const offX =
          tubeRadius * (Math.cos(phi) * N[0] + Math.sin(phi) * binormal[0]);
        const offY =
          tubeRadius * (Math.cos(phi) * N[1] + Math.sin(phi) * binormal[1]);
        const offZ =
          tubeRadius * (Math.cos(phi) * N[2] + Math.sin(phi) * binormal[2]);

        // The vertex normal is the normalized tube offset.
        const n = normalize([offX, offY, offZ]);

        // Store centerline coordinate and offset as separate attributes.
        verts.push(
          cx,
          cy,
          cz, // centerline position (3 floats)
          offX,
          offY,
          offZ, // tube offset (3 floats)
          n[0],
          n[1],
          n[2],
        ); // normal (3 floats)
      }
    }

    // Build indices for a grid of (majorSegments+1) x (minorSegments+1) vertices.
    for (let i = 0; i < majorSegments; i++) {
      for (let j = 0; j < minorSegments; j++) {
        const current = ringBaseIndex + i * (minorSegments + 1) + j;
        const next = current + (minorSegments + 1);
        // Two triangles per quad.
        idxs.push(current, next, current + 1);
        idxs.push(current + 1, next, next + 1);
      }
    }
  }

  return {
    vertexData: new Float32Array(verts),
    indexData: new Uint16Array(idxs),
  };
}

function cross(a: number[], b: number[]): number[] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(v: number[]): number[] {
  const len = Math.hypot(v[0], v[1], v[2]);
  return len > 0 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 0];
}

export function createCubeGeometry() {
  // 6 faces => 24 verts, 36 indices
  const positions: number[] = [
    // +X face (right) - when looking at it from right side
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0, // reordered: BL,BR,TL,TR
    // -X face (left) - when looking at it from left side
    -1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    -1.0, // reordered: BL,BR,TL,TR
    // +Y face (top) - when looking down at it
    -1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0, // reordered: BL,BR,TL,TR
    // -Y face (bottom) - when looking up at it
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0, // reordered: BL,BR,TL,TR
    // +Z face (front) - when looking at front
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0, // reordered: BL,BR,TL,TR
    // -Z face (back) - when looking at it from behind
    1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0, // reordered: BL,BR,TL,TR
  ];

  // Normals stay the same as they define face orientation
  const normals: number[] = [
    // +X
    1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
    // -X
    -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
    // +Y
    0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
    // -Y
    0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
    // +Z
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    // -Z
    0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
  ];

  // For each face, define triangles in CCW order when viewed from outside
  const indices: number[] = [];
  for (let face = 0; face < 6; face++) {
    const base = face * 4;
    // All faces use same pattern: BL->BR->TL, BR->TR->TL
    indices.push(
      base + 0,
      base + 1,
      base + 2, // first triangle: BL->BR->TL
      base + 1,
      base + 3,
      base + 2, // second triangle: BR->TR->TL
    );
  }

  // Interleave positions and normals
  const vertexData = new Float32Array(positions.length * 2);
  for (let i = 0; i < positions.length / 3; i++) {
    vertexData[i * 6 + 0] = positions[i * 3 + 0];
    vertexData[i * 6 + 1] = positions[i * 3 + 1];
    vertexData[i * 6 + 2] = positions[i * 3 + 2];
    vertexData[i * 6 + 3] = normals[i * 3 + 0];
    vertexData[i * 6 + 4] = normals[i * 3 + 1];
    vertexData[i * 6 + 5] = normals[i * 3 + 2];
  }
  return {
    vertexData,
    indexData: new Uint16Array(indices),
  };
}

/******************************************************
 * createBeamGeometry
 * Returns a "unit beam" from z=0..1, with rectangular cross-section of width=1.
 * Reuses cube geometry with transformation to match original beam positions.
 ******************************************************/
export function createBeamGeometry() {
  // Get base cube geometry
  const cube = createCubeGeometry();
  const vertexData = new Float32Array(cube.vertexData);

  // Transform vertices:
  // Scale z by 0.5 and translate by 0.5 to make beam start at origin
  // and extend one unit in +z direction
  for (let i = 0; i < vertexData.length; i += 6) {
    // Only transform position z coordinate (first 3 components), not normals
    vertexData[i + 2] = vertexData[i + 2] * 0.5 + 0.5;
  }

  return {
    vertexData,
    indexData: cube.indexData,
  };
}
