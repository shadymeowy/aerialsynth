// The geometric normals pass: the normal of the rendered surface at each supersampled pixel,
// from its neighbours' points in the G-buffer (Renderer::geo_normal), for the walls of the
// shading pass (shade.wgsl, F_WALLS).
//
// A pass of its own rather than a function of the shading kernel: gathering the neighbouring
// G-buffer texels inside the shading kernel's sample loop made Windows' shader compiler (FXC,
// DX12) / WARP produce a kernel that wrote garbage or nothing (no terrain at all in the
// renderer's self-check), with no error reported. Here it is a small kernel of its own, and the
// shading kernel reads one texel.

struct NU {
    // supersampled width, height (the shading pass's uniform buffer: only its first field)
    dims: vec4<u32>,
}

@group(0) @binding(0) var<uniform> u: NU;
// u, v (tile texel), range, unit + 1 (bits; 0: sky)
@group(0) @binding(1) var gbuf: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> rays: array<vec4<f32>>;
@group(0) @binding(3) var out_n: texture_storage_2d<rgba32float, write>;

// geometric normal (camera frame, facing the camera; zero when unknown) of the rendered surface
// at supersampled pixel (x, y), from its neighbours' points (Renderer::geo_normal: the next pixel
// on each axis, the previous one where that is missing or across a silhouette)
fn ss_point(x: i32, y: i32) -> vec4<f32> {
    if x < 0 || y < 0 || x >= i32(u.dims.x) || y >= i32(u.dims.y) {
        return vec4<f32>(0.0);
    }
    let g = textureLoad(gbuf, vec2<i32>(x, y), 0);
    let ray = rays[u32(y) * u.dims.x + u32(x)].xyz;
    if bitcast<u32>(g.w) == 0u || all(ray == vec3<f32>(0.0)) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(ray * g.z, 1.0);
}

// the step from point c to its neighbour at +d (pixels), or from the one at -d where that is
// missing or across a silhouette (farther than a tenth of the range); w = 0: neither
fn geo_step(c: vec3<f32>, x: i32, y: i32, d: vec2<i32>) -> vec4<f32> {
    let lim = 0.01 * dot(c, c);
    let hi = ss_point(x + d.x, y + d.y);
    if hi.w > 0.0 {
        let b = hi.xyz - c;
        if dot(b, b) < lim {
            return vec4<f32>(b, 1.0);
        }
    }
    let lo = ss_point(x - d.x, y - d.y);
    if lo.w > 0.0 {
        let a = c - lo.xyz;
        if dot(a, a) < lim {
            return vec4<f32>(a, 1.0);
        }
    }
    return vec4<f32>(0.0);
}

// (c: the sample's own point, camera frame)
fn geo_normal(x: i32, y: i32, c: vec3<f32>) -> vec3<f32> {
    let dx = geo_step(c, x, y, vec2<i32>(1, 0));
    let dy = geo_step(c, x, y, vec2<i32>(0, 1));
    if dx.w == 0.0 || dy.w == 0.0 {
        return vec3<f32>(0.0);
    }
    let n = cross(dx.xyz, dy.xyz);
    let l = length(n);
    if l < 1e-12 {
        return vec3<f32>(0.0);
    }
    return select(n / l, -n / l, dot(n, c) > 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= u.dims.x || gid.y >= u.dims.y {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let c = ss_point(x, y);
    var n = vec3<f32>(0.0);
    if c.w > 0.0 {
        n = geo_normal(x, y, c.xyz);
    }
    textureStore(out_n, vec2<i32>(x, y), vec4<f32>(n, 0.0));
}
