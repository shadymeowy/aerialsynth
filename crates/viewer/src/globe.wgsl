// Globe of XYZ tiles. Positions are camera-relative (f64 on the CPU, f32 here), so the view
// stays precise from orbit down to a few metres.

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // camera ECEF (m) - only for the sky (direction / altitude), not for positions
    cam: vec4<f32>,
    // xyz sun direction (ECEF), w ambient
    sun: vec4<f32>,
    // x exaggeration, y mode, z tile borders, w haze distance (m)
    params: vec4<f32>,
    // 1/a², 1/a², 1/b², mean radius
    ell: vec4<f32>,
};

struct Draw {
    rel_center: vec3<f32>,
    layer: u32,
    abs_center: vec3<f32>,
    skirt: f32,
    uv_off: vec2<f32>,
    uv_scale: f32,
    zoom: f32,
    cap_color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var color_tex: texture_2d_array<f32>;
@group(0) @binding(2) var elev_tex: texture_2d_array<f32>;
@group(0) @binding(3) var grad_tex: texture_2d_array<f32>;
@group(0) @binding(4) var samp: sampler;
@group(1) @binding(0) var<uniform> d: Draw;

const CAP: u32 = 0xffffffffu;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) suv: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) up: vec3<f32>,
    @location(3) rel: vec3<f32>,
};

fn elev_at(suv: vec2<f32>, layer: u32) -> f32 {
    let t = clamp(suv * 256.0 - 0.5, vec2<f32>(0.0), vec2<f32>(255.0));
    let i0 = vec2<i32>(floor(t));
    let f = t - floor(t);
    let i1 = min(i0 + vec2<i32>(1), vec2<i32>(255));
    let a = textureLoad(elev_tex, i0, layer, 0).r;
    let b = textureLoad(elev_tex, vec2<i32>(i1.x, i0.y), layer, 0).r;
    let c = textureLoad(elev_tex, vec2<i32>(i0.x, i1.y), layer, 0).r;
    let e = textureLoad(elev_tex, i1, layer, 0).r;
    return mix(mix(a, b, f.x), mix(c, e, f.x), f.y);
}

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) skirt: f32) -> VOut {
    var o: VOut;
    let absp = pos + d.abs_center;
    let up = normalize(absp * g.ell.xyz);
    let suv = d.uv_off + uv * d.uv_scale;
    // (a polar cap carries its height in uv_scale)
    var h = d.uv_scale;
    if (d.layer != CAP) {
        h = elev_at(suv, d.layer);
    }
    let p = pos + d.rel_center + up * (max(h, 0.0) * g.params.x - skirt * d.skirt);
    o.clip = g.view_proj * vec4<f32>(p, 1.0);
    o.suv = suv;
    o.uv = uv;
    o.up = up;
    o.rel = p;
    return o;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

fn palette(c: u32) -> vec3<f32> {
    var p = vec3<f32>(255.0, 0.0, 255.0);
    switch c {
        case 1u: { p = vec3<f32>(20.0, 50.0, 110.0); }
        case 2u: { p = vec3<f32>(40.0, 90.0, 160.0); }
        case 3u: { p = vec3<f32>(60.0, 130.0, 200.0); }
        case 4u: { p = vec3<f32>(240.0, 220.0, 160.0); }
        case 5u: { p = vec3<f32>(220.0, 190.0, 120.0); }
        case 6u: { p = vec3<f32>(130.0, 120.0, 110.0); }
        case 7u: { p = vec3<f32>(250.0, 250.0, 255.0); }
        case 8u: { p = vec3<f32>(140.0, 190.0, 80.0); }
        case 9u: { p = vec3<f32>(150.0, 150.0, 70.0); }
        case 10u: { p = vec3<f32>(30.0, 100.0, 40.0); }
        case 11u: { p = vec3<f32>(230.0, 200.0, 60.0); }
        case 12u: { p = vec3<f32>(200.0, 60.0, 60.0); }
        case 13u: { p = vec3<f32>(60.0, 60.0, 60.0); }
        case 14u: { p = vec3<f32>(70.0, 140.0, 130.0); }
        case 15u: { p = vec3<f32>(160.0, 160.0, 130.0); }
        case 16u: { p = vec3<f32>(160.0, 120.0, 90.0); }
        case 17u: { p = vec3<f32>(180.0, 150.0, 150.0); }
        default: {}
    }
    return srgb_to_linear(p / 255.0);
}

// hypsometric tint, sea level to ~5 km
fn hypso(h: f32, water: bool) -> vec3<f32> {
    if (water) {
        return srgb_to_linear(vec3<f32>(0.16, 0.32, 0.55));
    }
    let t = clamp(h / 5000.0, 0.0, 1.0);
    let c0 = vec3<f32>(0.25, 0.52, 0.30);
    let c1 = vec3<f32>(0.78, 0.74, 0.45);
    let c2 = vec3<f32>(0.55, 0.38, 0.25);
    let c3 = vec3<f32>(0.97, 0.97, 0.97);
    var c = mix(c0, c1, smoothstep(0.0, 0.2, t));
    c = mix(c, c2, smoothstep(0.2, 0.55, t));
    c = mix(c, c3, smoothstep(0.65, 0.9, t));
    return srgb_to_linear(c);
}

fn zoom_color(z: f32) -> vec3<f32> {
    let k = fract(z * 0.618034);
    return vec3<f32>(0.5 + 0.5 * cos(6.2832 * (k + vec3<f32>(0.0, 0.33, 0.67))));
}

// sky radiance along a ray: glow from the closest approach of the ray to the planet
fn sky(cam: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let t = max(-dot(cam, dir), 0.0);
    let p = cam + dir * t;
    let alt = max(length(p) - g.ell.w, 0.0);
    let up = normalize(p);
    let lit = smoothstep(-0.18, 0.25, dot(up, g.sun.xyz));
    let dusk = smoothstep(-0.2, 0.0, dot(up, g.sun.xyz)) * (1.0 - smoothstep(0.0, 0.3, dot(up, g.sun.xyz)));
    let dens = exp(-alt / 9000.0);
    let thin = exp(-alt / 40000.0);
    let blue = vec3<f32>(0.18, 0.42, 0.95);
    let horizon = vec3<f32>(0.62, 0.78, 1.0);
    var c = mix(blue * 0.6, horizon, dens) * thin * lit * 1.3;
    c += vec3<f32>(1.0, 0.45, 0.2) * dusk * thin * 0.5;
    // the sun disc
    let s = max(dot(dir, g.sun.xyz), 0.0);
    c += vec3<f32>(1.0, 0.95, 0.85) * (pow(s, 2000.0) * 20.0 + pow(s, 60.0) * 0.12 * thin);
    return c;
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0) - exp(-c * 1.25);
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let up = normalize(i.up);
    let east = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), up) + vec3<f32>(1e-6, 0.0, 0.0));
    let north = cross(up, east);
    let exag = g.params.x;
    let mode = u32(g.params.y + 0.5);
    var base: vec3<f32>;
    var n = up;
    var water = false;
    if (d.layer == CAP) {
        base = d.cap_color.rgb;
        water = d.cap_color.a > 0.5;
    } else {
        let c = textureSample(color_tex, samp, i.suv, d.layer);
        let lc = u32(round(c.a * 255.0));
        water = lc == 1u || lc == 2u || lc == 3u;
        var gr = textureSample(grad_tex, samp, i.suv, d.layer).xy * exag;
        // walls of buildings are steps of metres over one texel: cap the slope (~50 deg) so they
        // shade as edges, not as needles
        let gm = length(gr);
        if (gm > 1.2) {
            gr *= 1.2 / gm;
        }
        n = normalize(up - east * gr.x - north * gr.y);
        switch mode {
            // (the texel heights: the mesh heights are box-filtered over its grid spacing)
            case 1u: { base = hypso(elev_at(i.suv, d.layer), water); }
            case 2u: { base = palette(lc); }
            case 3u: { base = vec3<f32>(0.6); }
            default: { base = c.rgb; }
        }
    }
    let sun = g.sun.xyz;
    let day = smoothstep(-0.08, 0.12, dot(up, sun));
    let ndl = max(dot(n, sun), 0.0) * day;
    var col = base * (g.sun.w + 1.35 * ndl);
    let v = normalize(-i.rel);
    if (water && mode != 3u) {
        let hv = normalize(sun + v);
        // a tight glint on a faint sheen (a broad bright lobe turns oceans into a white blob)
        let s = max(dot(up, hv), 0.0);
        col += vec3<f32>(1.0, 0.95, 0.85) * (pow(s, 600.0) * 0.35 + pow(s, 40.0) * 0.03) * day;
    }
    // aerial perspective towards the sky colour behind the point: the path through the lower
    // atmosphere (~25 km thick), not the whole distance (seen from orbit, the disc stays clear
    // and the limb hazes)
    let dist = length(i.rel);
    let path = min(dist, 25000.0 / max(dot(up, v), 0.04));
    let haze = (1.0 - exp(-path / g.params.w)) * 0.7;
    // the sky at the horizon above the point (as `sky` gives it for a low, level ray)
    let haze_col = vec3<f32>(0.30, 0.48, 0.92) * 0.9 * smoothstep(-0.18, 0.25, dot(up, sun));
    col = mix(col, haze_col, haze);
    if (g.params.z > 0.5) {
        let e = min(min(i.uv.x, 1.0 - i.uv.x), min(i.uv.y, 1.0 - i.uv.y));
        let w = max(fwidth(i.uv.x), fwidth(i.uv.y));
        let line = 1.0 - smoothstep(0.0, 1.5 * w, e);
        col = mix(col, zoom_color(d.zoom), line * 0.85);
    }
    return vec4<f32>(linear_to_srgb(tonemap(col)), 1.0);
}

// ---- background: sky / space
struct SOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) k: u32) -> SOut {
    var o: SOut;
    let p = vec2<f32>(f32((k << 1u) & 2u), f32(k & 2u)) * 2.0 - 1.0;
    o.clip = vec4<f32>(p, 0.0, 1.0);
    o.ndc = p;
    return o;
}

// integer hash of a cell of the star grid (a sin-based hash loses precision for large
// arguments: stars showed in a slice of the sky only)
fn hash(p: vec3<f32>) -> f32 {
    let q = vec3<i32>(p);
    var h = (u32(q.x) * 0x8da6b343u) ^ (u32(q.y) * 0xd8163841u) ^ (u32(q.z) * 0xcb1ab31fu);
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return f32(h) / 4294967296.0;
}

@fragment
fn fs_sky(i: SOut) -> @location(0) vec4<f32> {
    let a = g.inv_view_proj * vec4<f32>(i.ndc, 1.0, 1.0);
    let b = g.inv_view_proj * vec4<f32>(i.ndc, 0.5, 1.0);
    let dir = normalize(b.xyz / b.w - a.xyz / a.w);
    var c = sky(g.cam.xyz, dir);
    // a sparse star field, faded where the sky is bright
    let q = floor(dir * 700.0);
    let s = hash(q);
    let star = step(0.9985, s) * (0.4 + 2.0 * fract(s * 91.7));
    c += vec3<f32>(star) * max(0.0, 1.0 - 4.0 * max(c.r, max(c.g, c.b)));
    return vec4<f32>(linear_to_srgb(tonemap(c)), 1.0);
}
