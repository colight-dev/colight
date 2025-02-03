/******************************************************
 * 2) Constants and Camera Functions
 ******************************************************/

/**
 * Global lighting configuration for the 3D scene.
 * Uses a simple Blinn-Phong lighting model with ambient, diffuse, and specular components.
 */
export const LIGHTING = {
    /** Ambient light intensity, affects overall scene brightness */
    AMBIENT_INTENSITY: 0.4,

    /** Diffuse light intensity, affects surface shading based on light direction */
    DIFFUSE_INTENSITY: 0.6,

    /** Specular highlight intensity */
    SPECULAR_INTENSITY: 0.2,

    /** Specular power/shininess, higher values create sharper highlights */
    SPECULAR_POWER: 20.0,

    /** Light direction components relative to camera */
    DIRECTION: {
        /** Right component of light direction */
        RIGHT: 0.2,
        /** Up component of light direction */
        UP: 0.5,
        /** Forward component of light direction */
        FORWARD: 0,
    }
} as const;

// Common shader code templates
export const cameraStruct = /*wgsl*/`
struct Camera {
  mvp: mat4x4<f32>,
  cameraRight: vec3<f32>,
  _pad1: f32,
  cameraUp: vec3<f32>,
  _pad2: f32,
  lightDir: vec3<f32>,
  _pad3: f32,
  cameraPos: vec3<f32>,
  _pad4: f32,
};
@group(0) @binding(0) var<uniform> camera : Camera;`;

export const lightingConstants = /*wgsl*/`
const AMBIENT_INTENSITY = ${LIGHTING.AMBIENT_INTENSITY}f;
const DIFFUSE_INTENSITY = ${LIGHTING.DIFFUSE_INTENSITY}f;
const SPECULAR_INTENSITY = ${LIGHTING.SPECULAR_INTENSITY}f;
const SPECULAR_POWER = ${LIGHTING.SPECULAR_POWER}f;`;

export const lightingCalc = /*wgsl*/`
fn calculateLighting(baseColor: vec3<f32>, normal: vec3<f32>, worldPos: vec3<f32>) -> vec3<f32> {
  let N = normalize(normal);
  let L = normalize(camera.lightDir);
  let V = normalize(camera.cameraPos - worldPos);

  let lambert = max(dot(N, L), 0.0);
  let ambient = AMBIENT_INTENSITY;
  var color = baseColor * (ambient + lambert * DIFFUSE_INTENSITY);

  let H = normalize(L + V);
  let spec = pow(max(dot(N, H), 0.0), SPECULAR_POWER);
  color += vec3<f32>(1.0) * spec * SPECULAR_INTENSITY;

  return color;
}`;



export const billboardVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) Position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alpha: f32
};

@vertex
fn vs_main(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) instancePos: vec3<f32>,
  @location(3) col: vec3<f32>,
  @location(4) alpha: f32,
  @location(5) size: f32
)-> VSOut {
  // Create camera-facing orientation
  let right = camera.cameraRight;
  let up = camera.cameraUp;

  // Transform quad vertices to world space
  let scaledRight = right * (localPos.x * size);
  let scaledUp = up * (localPos.y * size);
  let worldPos = instancePos + scaledRight + scaledUp;

  var out: VSOut;
  out.Position = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.color = col;
  out.alpha = alpha;
  return out;
}`;

export const billboardPickingVertCode = /*wgsl*/`
@vertex
fn vs_pointcloud(
  @location(0) localPos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) instancePos: vec3<f32>,
  @location(3) pickID: f32,
  @location(4) size: f32
)-> VSOut {
  // Create camera-facing orientation
  let right = camera.cameraRight;
  let up = camera.cameraUp;

  // Transform quad vertices to world space
  let scaledRight = right * (localPos.x * size);
  let scaledUp = up * (localPos.y * size);
  let worldPos = instancePos + scaledRight + scaledUp;

  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

export const billboardFragCode = /*wgsl*/`
@fragment
fn fs_main(@location(0) color: vec3<f32>, @location(1) alpha: f32)-> @location(0) vec4<f32> {
  return vec4<f32>(color, alpha);
}`;


export const ellipsoidVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>,
  @location(5) instancePos: vec3<f32>
};

@vertex
fn vs_main(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) iPos: vec3<f32>,
  @location(3) iScale: vec3<f32>,
  @location(4) iColor: vec3<f32>,
  @location(5) iAlpha: f32
)-> VSOut {
  let worldPos = iPos + (inPos * iScale);
  let scaledNorm = normalize(inNorm / iScale);

  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos,1.0);
  out.normal = scaledNorm;
  out.baseColor = iColor;
  out.alpha = iAlpha;
  out.worldPos = worldPos;
  out.instancePos = iPos;
  return out;
}`;

export const ellipsoidPickingVertCode = /*wgsl*/`
@vertex
fn vs_ellipsoid(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) iPos: vec3<f32>,
  @location(3) iScale: vec3<f32>,
  @location(4) pickID: f32
)-> VSOut {
  let wp = iPos + (inPos * iScale);
  var out: VSOut;
  out.pos = camera.mvp*vec4<f32>(wp,1.0);
  out.pickID = pickID;
  return out;
}`;

export const ellipsoidFragCode = /*wgsl*/`
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>,
  @location(5) instancePos: vec3<f32>
)-> @location(0) vec4<f32> {
  let color = calculateLighting(baseColor, normal, worldPos);
  return vec4<f32>(color, alpha);
}`;



export const ringVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) color: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>,
};

@vertex
fn vs_main(
  @builtin(instance_index) instID: u32,
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) scale: vec3<f32>,
  @location(4) color: vec3<f32>,
  @location(5) alpha: f32
)-> VSOut {
  let ringIndex = i32(instID % 3u);
  var lp = inPos;
  // rotate the ring geometry differently for x-y-z rings
  if(ringIndex==0){
    let tmp = lp.z;
    lp.z = -lp.y;
    lp.y = tmp;
  } else if(ringIndex==1){
    let px = lp.x;
    lp.x = -lp.y;
    lp.y = px;
    let pz = lp.z;
    lp.z = lp.x;
    lp.x = pz;
  }
  lp *= scale;
  let wp = center + lp;
  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(wp,1.0);
  out.normal = inNorm;
  out.color = color;
  out.alpha = alpha;
  out.worldPos = wp;
  return out;
}`;


export const ringPickingVertCode = /*wgsl*/`
@vertex
fn vs_rings(
  @builtin(instance_index) instID:u32,
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) scale: vec3<f32>,
  @location(4) pickID: f32
)-> VSOut {
  let ringIndex=i32(instID%3u);
  var lp=inPos;
  if(ringIndex==0){
    let tmp=lp.z; lp.z=-lp.y; lp.y=tmp;
  } else if(ringIndex==1){
    let px=lp.x; lp.x=-lp.y; lp.y=px;
    let pz=lp.z; lp.z=lp.x; lp.x=pz;
  }
  lp*=scale;
  let wp=center+lp;
  var out:VSOut;
  out.pos=camera.mvp*vec4<f32>(wp,1.0);
  out.pickID=pickID;
  return out;
}`;

export const ringFragCode = /*wgsl*/`
@fragment
fn fs_main(
  @location(1) n: vec3<f32>,
  @location(2) c: vec3<f32>,
  @location(3) a: f32,
  @location(4) wp: vec3<f32>
)-> @location(0) vec4<f32> {
  // simple color (no shading)
  return vec4<f32>(c, a);
}`;



export const cuboidVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
};

@vertex
fn vs_main(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) size: vec3<f32>,
  @location(4) color: vec3<f32>,
  @location(5) alpha: f32
)-> VSOut {
  let worldPos = center + (inPos * size);
  let scaledNorm = normalize(inNorm / size);
  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos,1.0);
  out.normal = scaledNorm;
  out.baseColor = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  return out;
}`;

export const cuboidPickingVertCode = /*wgsl*/`
@vertex
fn vs_cuboid(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,
  @location(2) center: vec3<f32>,
  @location(3) size: vec3<f32>,
  @location(4) pickID: f32
)-> VSOut {
  let worldPos = center + (inPos * size);
  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;

export const cuboidFragCode = /*wgsl*/`
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
)-> @location(0) vec4<f32> {
  let color = calculateLighting(baseColor, normal, worldPos);
  return vec4<f32>(color, alpha);
}`;



export const lineBeamVertCode = /*wgsl*/`// lineBeamVertCode.wgsl
${cameraStruct}
${lightingConstants}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>,
};

@vertex
fn vs_main(
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,

  @location(2) startPos: vec3<f32>,
  @location(3) endPos: vec3<f32>,
  @location(4) size: f32,
  @location(5) color: vec3<f32>,
  @location(6) alpha: f32
) -> VSOut
{
  // The unit beam is from z=0..1 along local Z, size=1 in XY
  // We'll transform so it goes from start->end with size=size.
  let segDir = endPos - startPos;
  let length = max(length(segDir), 0.000001);
  let zDir   = normalize(segDir);

  // build basis xDir,yDir from zDir
  var tempUp = vec3<f32>(0,0,1);
  if (abs(dot(zDir, tempUp)) > 0.99) {
    tempUp = vec3<f32>(0,1,0);
  }
  let xDir = normalize(cross(zDir, tempUp));
  let yDir = cross(zDir, xDir);

  // For cuboid, we want corners at ±size in both x and y
  let localX = inPos.x * size;
  let localY = inPos.y * size;
  let localZ = inPos.z * length;
  let worldPos = startPos
    + xDir * localX
    + yDir * localY
    + zDir * localZ;

  // transform normal similarly
  let rawNormal = vec3<f32>(inNorm.x, inNorm.y, inNorm.z);
  let nWorld = normalize(
    xDir*rawNormal.x +
    yDir*rawNormal.y +
    zDir*rawNormal.z
  );

  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.normal = nWorld;
  out.baseColor = color;
  out.alpha = alpha;
  out.worldPos = worldPos;
  return out;
}`;

export const lineBeamFragCode = /*wgsl*/`// lineBeamFragCode.wgsl
${cameraStruct}
${lightingConstants}
${lightingCalc}

@fragment
fn fs_main(
  @location(1) normal: vec3<f32>,
  @location(2) baseColor: vec3<f32>,
  @location(3) alpha: f32,
  @location(4) worldPos: vec3<f32>
)-> @location(0) vec4<f32>
{
  let color = calculateLighting(baseColor, normal, worldPos);
  return vec4<f32>(color, alpha);
}`

export const lineBeamPickingVertCode = /*wgsl*/`
@vertex
fn vs_lineBeam(  // Rename from vs_lineCyl to vs_lineBeam
  @location(0) inPos: vec3<f32>,
  @location(1) inNorm: vec3<f32>,

  @location(2) startPos: vec3<f32>,
  @location(3) endPos: vec3<f32>,
  @location(4) size: f32,
  @location(5) pickID: f32
) -> VSOut {
  let segDir = endPos - startPos;
  let length = max(length(segDir), 0.000001);
  let zDir = normalize(segDir);

  var tempUp = vec3<f32>(0,0,1);
  if (abs(dot(zDir, tempUp)) > 0.99) {
    tempUp = vec3<f32>(0,1,0);
  }
  let xDir = normalize(cross(zDir, tempUp));
  let yDir = cross(zDir, xDir);

  let localX = inPos.x * size;
  let localY = inPos.y * size;
  let localZ = inPos.z * length;
  let worldPos = startPos
    + xDir*localX
    + yDir*localY
    + zDir*localZ;

  var out: VSOut;
  out.pos = camera.mvp * vec4<f32>(worldPos, 1.0);
  out.pickID = pickID;
  return out;
}`;



export const pickingVertCode = /*wgsl*/`
${cameraStruct}

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) pickID: f32
};

@fragment
fn fs_pick(@location(0) pickID: f32)-> @location(0) vec4<f32> {
  let iID = u32(pickID);
  let r = f32(iID & 255u)/255.0;
  let g = f32((iID>>8)&255u)/255.0;
  let b = f32((iID>>16)&255u)/255.0;
  return vec4<f32>(r,g,b,1.0);
}

${billboardPickingVertCode}
${ellipsoidPickingVertCode}
${ringPickingVertCode}
${cuboidPickingVertCode}
${lineBeamPickingVertCode}
`;
