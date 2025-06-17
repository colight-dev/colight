import { quaternionShaderFunctions } from "../quaternion";
import {
  createVertexBufferLayout,
  cameraStruct,
  lightingConstants,
  lightingCalc,
  pickingVSOut,
  pickingFragCode,
} from "../shaders";

import { PrimitiveSpec } from "../types";

import {
  fillAlpha,
  fillColor,
  applyDecoration,
  EllipsoidComponentConfig,
  getOrCreatePipeline,
  createRenderPipeline,
  createBuffers,
  createTranslucentGeometryPipeline,
} from "../components";

import { packID } from "../picking";
import { acopy } from "../../utils";

import { createEllipsoidAxes } from "../geometry";

export const RING_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [3, "float32x3"], // instance center position
    [4, "float32x3"], // instance size
    [5, "float32x4"], // instance quaternion
    [6, "float32x3"], // instance color
    [7, "float32"], // instance alpha
  ],
  "instance",
);

export const RING_PICKING_INSTANCE_LAYOUT = createVertexBufferLayout(
  [
    [3, "float32x3"], // position
    [4, "float32x3"], // size
    [5, "float32x4"], // quaternion
    [6, "float32"], // pickID (now shared across rings)
  ],
  "instance",
);

export const RING_GEOMETRY_LAYOUT = createVertexBufferLayout(
  [
    [0, "float32x3"], // centerline position
    [1, "float32x3"], // tube offset
    [2, "float32x3"], // normal
  ],
  "vertex",
);

export const ringShaders = /*wgsl*/ `
  ${cameraStruct}
  ${quaternionShaderFunctions}
  ${pickingVSOut}

  struct RenderVSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    @location(2) worldPos: vec3<f32>,
    @location(3) normal: vec3<f32>
  };

  fn computeRingPosition(
    center: vec3<f32>,
    offset: vec3<f32>,
    position: vec3<f32>,
    size: vec3<f32>,
    quaternion: vec4<f32>
  ) -> vec3<f32> {
    // Apply non-uniform scaling to the centerline.
    let scaledCenter = quat_rotate(quaternion, center * size);

    // Compute a uniform scale for the tube offset (e.g. average of nonuniform scales).
    let uniformScale = (size.x + size.y + size.z) / 3.0;
    let scaledOffset = quat_rotate(quaternion, offset * uniformScale);

    // Final world position: instance position plus transformed center and offset.
    return position + scaledCenter + scaledOffset;
  }

  @vertex
  fn vs_render(
    @location(0) center: vec3<f32>,  // Centerline attribute (first 3 floats)
    @location(1) offset: vec3<f32>,  // Tube offset attribute (next 3 floats)
    @location(2) inNormal: vec3<f32>, // Precomputed normal (last 3 floats)
    @location(3) position: vec3<f32>,  // Instance center
    @location(4) size: vec3<f32>,      // Instance non-uniform scaling for ellipsoid
    @location(5) quaternion: vec4<f32>,// Instance rotation
    @location(6) inColor: vec3<f32>,   // Color attribute
    @location(7) alpha: f32            // Alpha attribute
  ) -> RenderVSOut {
    let worldPos = computeRingPosition(center, offset, position, size, quaternion);

    // For normals, we want the tube's offset direction unperturbed by nonuniform scaling.
    let worldNormal = quat_rotate(quaternion, normalize(offset));

    var out: RenderVSOut;
    out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
    out.color = inColor;
    out.alpha = alpha;
    out.worldPos = worldPos;
    out.normal = worldNormal;
    return out;
  }

  @vertex
  fn vs_pick(
    @location(0) center: vec3<f32>,  // Centerline attribute (first 3 floats)
    @location(1) offset: vec3<f32>,  // Tube offset attribute (next 3 floats)
    @location(2) inNormal: vec3<f32>, // Precomputed normal (last 3 floats)
    @location(3) position: vec3<f32>,  // Instance center
    @location(4) size: vec3<f32>,      // Instance non-uniform scaling for ellipsoid
    @location(5) quaternion: vec4<f32>,// Instance rotation
    @location(6) pickID: f32           // Picking ID
  ) -> VSOut {
    let worldPos = computeRingPosition(center, offset, position, size, quaternion);

    var out: VSOut;
    out.position = camera.mvp * vec4<f32>(worldPos, 1.0);
    out.pickID = pickID;
    return out;
  }`;

export const ringFragCode = /*wgsl*/ `
  ${cameraStruct}
  ${lightingConstants}
  ${lightingCalc}

  @fragment
  fn fs_main(
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    @location(2) worldPos: vec3<f32>,
    @location(3) normal: vec3<f32>
  )-> @location(0) vec4<f32> {
    let litColor = calculateLighting(color, normal, worldPos);
    return vec4<f32>(litColor, alpha);
  }`;

export const ellipsoidAxesSpec: PrimitiveSpec<EllipsoidComponentConfig> = {
  type: "EllipsoidAxes",

  defaults: {
    half_size: [0.5, 0.5, 0.5],
    quaternion: [0, 0, 0, 1],
  },

  // Return the number of ellipsoids (elements), not instances
  getElementCount(elem) {
    return elem.centers.length / 3;
  },

  floatsPerInstance: 14, // position(3) + size(3) + quat(4) + color(3) + alpha(1) = 14 per ring

  floatsPerPicking: 11, // same layout as Ellipsoid: 11 per ring

  // This tells the system we have 3 instances per element
  instancesPerElement: 3,

  // Return centers for the elements (not instances)
  getCenters(elem) {
    // Just return the original centers - one per element
    return elem.centers;
  },

  // Fill render geometry for a single instance
  fillRenderGeometry(constants, elem, elemIndex, out, outIndex) {
    const outOffset = outIndex * this.floatsPerInstance;

    // Position
    acopy(elem.centers, elemIndex * 3, out, outOffset, 3);

    // Half sizes
    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, outOffset + 3, 3);
    } else {
      acopy(elem.half_sizes!, elemIndex * 3, out, outOffset + 3, 3);
    }

    // Quaternion
    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, outOffset + 6, 4);
    } else {
      acopy(elem.quaternions!, elemIndex * 4, out, outOffset + 6, 4);
    }

    // The shader will handle the ring orientation based on instance_index

    fillAlpha(this, constants, elem, elemIndex, out, outOffset);
    fillColor(this, constants, elem, elemIndex, out, outOffset);
  },

  colorOffset: 10,
  alphaOffset: 13,

  applyDecoration(dec, out, outIndex, floatsPerInstance) {
    // outIndex is already the instance index, so we can just apply the decoration directly
    applyDecoration(this, dec, out, outIndex * floatsPerInstance);
  },

  applyDecorationScale(out, offset, scaleFactor) {
    out[offset + 3] *= scaleFactor;
    out[offset + 4] *= scaleFactor;
    out[offset + 5] *= scaleFactor;
  },

  fillPickingGeometry(constants, elem, elemIndex, out, outIndex, baseID) {
    const outOffset = outIndex * this.floatsPerPicking;

    // Position
    acopy(elem.centers, elemIndex * 3, out, outOffset, 3);

    // Half sizes
    if (constants.half_size) {
      acopy(constants.half_size as ArrayLike<number>, 0, out, outOffset + 3, 3);
    } else {
      acopy(elem.half_sizes!, elemIndex * 3, out, outOffset + 3, 3);
    }

    // Quaternion
    if (constants.quaternion) {
      acopy(constants.quaternion, 0, out, outOffset + 6, 4);
    } else {
      acopy(elem.quaternions!, elemIndex * 4, out, outOffset + 6, 4);
    }

    // The shader will handle the ring orientation based on instance_index

    // Use the ellipsoid index for picking
    out[outOffset + 10] = packID(baseID + elemIndex);
  },
  renderConfig: {
    cullMode: "back",
    topology: "triangle-list",
  },

  getRenderPipeline(device, bindGroupLayout, cache) {
    const format = navigator.gpu.getPreferredCanvasFormat();
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesShading",
      () => {
        return createTranslucentGeometryPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: ringShaders,
            fragmentShader: ringFragCode,
            vertexEntryPoint: "vs_render",
            fragmentEntryPoint: "fs_main",
            bufferLayouts: [RING_GEOMETRY_LAYOUT, RING_INSTANCE_LAYOUT],
            primitive: this.renderConfig,
          },
          format,
          ellipsoidAxesSpec,
        );
      },
      cache,
    );
  },

  getPickingPipeline(device, bindGroupLayout, cache) {
    return getOrCreatePipeline(
      device,
      "EllipsoidAxesPicking",
      () => {
        return createRenderPipeline(
          device,
          bindGroupLayout,
          {
            vertexShader: ringShaders,
            fragmentShader: pickingFragCode,
            vertexEntryPoint: "vs_pick",
            fragmentEntryPoint: "fs_pick",
            bufferLayouts: [RING_GEOMETRY_LAYOUT, RING_PICKING_INSTANCE_LAYOUT],
            primitive: this.renderConfig,
          },
          "rgba8unorm",
        );
      },
      cache,
    );
  },

  createGeometryResource(device) {
    return createBuffers(device, createEllipsoidAxes(1.0, 0.05, 32, 16));
  },
};
