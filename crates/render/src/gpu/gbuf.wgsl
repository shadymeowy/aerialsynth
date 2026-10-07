// G-buffer pass: the CPU-built meshes (vertices already projected through the camera model, in
// supersampled pixel coordinates, with their range) rasterized into (u, v, range, unit + 1 as
// bits; 0 = no terrain: the clear value, a NaN pattern would not survive the clear).
// clip.w = range makes the interpolation of u, v and range perspective-correct with weights
// 1 / range, exactly like the CPU rasterizer (which interpolates 1/z); the CPU samples pixel
// (x, y) at integer coordinates, the GPU at x + 0.5, hence the half-pixel shift.

struct GU {
    dims: vec4<f32>, // supersampled width, height
};
@group(0) @binding(0) var<uniform> gu: GU;

struct VIn {
    @location(0) pos: vec3<f32>, // sx, sy, range
    @location(1) uv: vec2<f32>,
    @location(2) unit: u32,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uvr: vec3<f32>,
    @location(1) @interpolate(flat) unit: u32,
};

@vertex
fn vs(v: VIn) -> VOut {
    let r = v.pos.z;
    let nx = (v.pos.x + 0.5) / gu.dims.x * 2.0 - 1.0;
    let ny = 1.0 - (v.pos.y + 0.5) / gu.dims.y * 2.0;
    var o: VOut;
    o.clip = vec4<f32>(nx * r, ny * r, 0.5 * r, r);
    o.uvr = vec3<f32>(v.uv, r);
    o.unit = v.unit;
    return o;
}

struct FOut {
    @location(0) g: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(i: VOut) -> FOut {
    var o: FOut;
    o.g = vec4<f32>(i.uvr.x, i.uvr.y, i.uvr.z, bitcast<f32>(i.unit + 1u));
    o.depth = clamp(i.uvr.z * 1.0e-7, 0.0, 1.0);
    return o;
}
