// G-buffer pass with GPU-resident meshes: each unit's mesh is cached on the GPU in coordinates
// relative to its origin (f32 is enough there); per frame only the origin relative to the camera
// (computed in f64 on the CPU) and the camera are uploaded. The vertex shader transforms to the
// camera frame and applies the camera model (ported from camera.rs; supersampled intrinsics),
// with the same validity rules as the CPU mesh builder. Output: (u, v, range, unit + 1 as bits).
// clip.w = range makes the interpolation of u, v and range perspective-correct with weights
// 1 / range, like the CPU rasterizer; the CPU samples pixel (x, y) at integer coordinates, the
// GPU at x + 0.5, hence the half-pixel shift. Triangles touching a non-imageable vertex are
// dropped (the CPU rasterizer skips them): their fragments see an interpolated validity < 1.

struct MU {
    rt0: vec4<f32>,      // rows of the ECEF → camera rotation
    rt1: vec4<f32>,
    rt2: vec4<f32>,
    p0: vec4<f32>,       // camera model parameters (camera.rs GpuCamera)
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    lim: vec4<f32>,      // cos(angle limit), cos(mesh half-angle limit), supersampled width, height
    kind: vec4<u32>,     // camera model kind
};
@group(0) @binding(0) var<uniform> mu: MU;
// per draw (instance index): unit origin minus camera position (ECEF), unit index bits in w
@group(0) @binding(1) var<storage, read> draws: array<vec4<f32>>;

fn prm(i: u32) -> f32 {
    let q = i / 4u;
    let r = i % 4u;
    var v: vec4<f32>;
    switch q {
        case 0u: { v = mu.p0; }
        case 1u: { v = mu.p1; }
        case 2u: { v = mu.p2; }
        default: { v = mu.p3; }
    }
    return v[r];
}

fn radtan4(p: vec2<f32>, k1: f32, k2: f32, p1: f32, p2: f32) -> vec2<f32> {
    let x2 = p.x * p.x;
    let y2 = p.y * p.y;
    let xy = p.x * p.y;
    let r2 = x2 + y2;
    let rad = k1 * r2 + k2 * r2 * r2;
    return vec2<f32>(p.x * rad + 2.0 * p1 * xy + p2 * (r2 + 2.0 * x2), p.y * rad + 2.0 * p2 * xy + p1 * (r2 + 2.0 * y2));
}

// pixel position (x, y) and validity (z: 1 = imageable)
fn project(p: vec3<f32>) -> vec3<f32> {
    let n = length(p);
    if n < 1e-12 || p.z / n < mu.lim.x || p.z / n <= mu.lim.y {
        return vec3<f32>(0.0);
    }
    let kind = mu.kind.x;
    if kind == 0u || kind == 1u {
        if p.z <= 1e-9 {
            return vec3<f32>(0.0);
        }
        let x = p.x / p.z;
        let y = p.y / p.z;
        var d: vec2<f32>;
        if kind == 0u {
            d = vec2<f32>(x, y) + radtan4(vec2<f32>(x, y), prm(4u), prm(5u), prm(6u), prm(7u));
        } else {
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let a1 = 2.0 * x * y;
            let a2 = r2 + 2.0 * x * x;
            let a3 = r2 + 2.0 * y * y;
            let cd = 1.0 + prm(4u) * r2 + prm(5u) * r4 + prm(8u) * r6;
            let icd = 1.0 / (1.0 + prm(9u) * r2 + prm(10u) * r4 + prm(11u) * r6);
            d = vec2<f32>(x * cd * icd + prm(6u) * a1 + prm(7u) * a2, y * cd * icd + prm(6u) * a3 + prm(7u) * a1);
        }
        return vec3<f32>(prm(0u) * d.x + prm(2u), prm(1u) * d.y + prm(3u), 1.0);
    }
    if kind == 2u {
        let theta = acos(clamp(p.z / n, -1.0, 1.0));
        if theta > prm(8u) {
            return vec3<f32>(0.0);
        }
        let phi = atan2(p.y, p.x);
        let t2 = theta * theta;
        let r = theta * (1.0 + t2 * (prm(4u) + t2 * (prm(5u) + t2 * (prm(6u) + t2 * prm(7u)))));
        return vec3<f32>(prm(0u) * r * cos(phi) + prm(2u), prm(1u) * r * sin(phi) + prm(3u), 1.0);
    }
    if kind == 3u {
        let z = p.z + prm(4u) * n;
        if z <= 1e-9 {
            return vec3<f32>(0.0);
        }
        var u = vec2<f32>(p.x / z, p.y / z);
        u += radtan4(u, prm(5u), prm(6u), prm(7u), prm(8u));
        return vec3<f32>(prm(0u) * u.x + prm(2u), prm(1u) * u.y + prm(3u), 1.0);
    }
    // scaramuzza
    let nn = length(p.xy);
    if nn < 1e-12 {
        return vec3<f32>(0.0);
    }
    let theta = atan2(-p.z, nn);
    let nc = u32(prm(5u));
    var rho = 0.0;
    var t = 1.0;
    for (var i = 0u; i < nc; i++) {
        rho += t * prm(6u + i);
        t *= theta;
    }
    let xn = vec2<f32>(p.x / nn * rho, p.y / nn * rho);
    return vec3<f32>(xn.x * prm(0u) + xn.y * prm(1u) + prm(3u), xn.x * prm(2u) + xn.y + prm(4u), 1.0);
}

struct VIn {
    @builtin(instance_index) inst: u32,
    @location(0) pos: vec3<f32>, // relative to the unit origin (ECEF axes)
    @location(1) uv: vec2<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uvr: vec3<f32>,
    @location(1) valid: f32,
    @location(2) @interpolate(flat) unit: u32,
};

@vertex
fn vs(v: VIn) -> VOut {
    let dr = draws[v.inst];
    let q = dr.xyz + v.pos;
    let pc = vec3<f32>(dot(mu.rt0.xyz, q), dot(mu.rt1.xyz, q), dot(mu.rt2.xyz, q));
    let r = max(length(pc), 1e-3);
    let px = project(pc);
    let nx = (px.x + 0.5) / mu.lim.z * 2.0 - 1.0;
    let ny = 1.0 - (px.y + 0.5) / mu.lim.w * 2.0;
    var o: VOut;
    o.clip = vec4<f32>(nx * r, ny * r, 0.5 * r, r);
    o.uvr = vec3<f32>(v.uv, r);
    o.valid = px.z;
    o.unit = bitcast<u32>(dr.w);
    return o;
}

struct FOut {
    @location(0) g: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(i: VOut) -> FOut {
    if i.valid < 0.9999 {
        discard;
    }
    var o: FOut;
    o.g = vec4<f32>(i.uvr.x, i.uvr.y, i.uvr.z, bitcast<f32>(i.unit + 1u));
    o.depth = clamp(i.uvr.z * 1.0e-7, 0.0, 1.0);
    return o;
}
