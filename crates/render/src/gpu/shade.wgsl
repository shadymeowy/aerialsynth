// Deferred shading pass: a port of the CPU renderer's per-pixel shading (raster.rs:
// `pixel_shade`, `texture`, `sun_visibility`, `TileView` sampling) and atmo.rs (sky, aerial
// perspective). One invocation per output pixel; it loops over the pixel's supersamples like the
// CPU path. Tile data comes from the slot pool (texture arrays) through a per-frame open-addressing
// table of the frame's tiles, so neighbour and ancestor fallbacks behave as on the CPU.
//
// Positions on the tile pyramid are (zoom, integer global pixel, fraction) per axis: global pixel
// coordinates reach 2^27 at zoom 19, beyond f32 precision.

struct U {
    dims: vec4<u32>,     // supersampled w, h; output ow, oh
    cfg: vec4<u32>,      // ss, max_aniso, flags, table mask
    cam_pos: vec4<f32>,  // camera ECEF (m), camera height above the ellipsoid
    r0: vec4<f32>,       // rows of r_ecef_cam (world = R · cam)
    r1: vec4<f32>,
    r2: vec4<f32>,
    cam_up: vec4<f32>,   // local up at the camera; w: sub-sample angular size
    sun_dir: vec4<f32>,  // towards the sun (ECEF); w: tan(max(sun elevation, 0.005))
    moon_dir: vec4<f32>, // w: lunar disc radiance
    sun_col: vec4<f32>,
    moon_col: vec4<f32>, // w: Mie extinction β_m
    ray_col: vec4<f32>,  // w: Rayleigh scale height
    mie_col: vec4<f32>,  // w: Mie scale height
    zenith: vec4<f32>,   // w: in-scatter strength
    horizon: vec4<f32>,  // w: highest DSM point of the frame's tiles
    sun: vec4<f32>,      // direct, sky, lights, light pollution
    sun2: vec4<f32>,     // sun azimuth, sun elevation
    flick: vec4<f32>,    // cos(ωt), sin(ωt), exposure sinc, LED fraction
    flick2: vec4<f32>,   // LED depth, depth
    ell: vec4<f32>,      // a, e², 2π
};

const F_RELIT: u32 = 1u;
const F_GLINT: u32 = 2u;
const F_SHADOWS: u32 = 4u;
const F_SPLIT: u32 = 8u;
const F_GEOM: u32 = 16u;
const F_ATMO: u32 = 32u;
const F_FLICKER: u32 = 64u;
const F_LIGHTS: u32 = 128u;
const F_POLLUTION: u32 = 256u;
const F_WALLS: u32 = 512u;

const HAS_COLOR: u32 = 1u;
const HAS_NORMAL: u32 = 2u;
const HAS_EMISSION: u32 = 4u;
const HAS_ELEV: u32 = 8u;
const HAS_LC: u32 = 16u;
const EMPTY: u32 = 0xFFFFFFFFu;
const PI: f32 = 3.14159265358979;

@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var gbuf: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> rays: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> units: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> table: array<vec4<u32>>;
@group(0) @binding(5) var t_color: texture_2d_array<f32>;
@group(0) @binding(6) var t_normal: texture_2d_array<f32>;
@group(0) @binding(7) var t_emis: texture_2d_array<f32>;
@group(0) @binding(8) var t_elev: texture_2d_array<f32>;
@group(0) @binding(9) var t_lc: texture_2d_array<u32>;
@group(0) @binding(10) var t_bmax: texture_2d_array<f32>;
@group(0) @binding(11) var<storage, read_write> out_rad: array<vec4<f32>>;
// depth, range of the central sample, land cover, unused
@group(0) @binding(12) var<storage, read_write> out_geo: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> out_fc: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> out_fs: array<vec4<f32>>;

fn has(f: u32) -> bool {
    return (u.cfg.z & f) != 0u;
}

// ---------------------------------------------------------------- tile lookup

fn tile_hash(z: u32, x: u32, y: u32) -> u32 {
    var h = (x * 0x9E3779B1u) ^ (y * 0x85EBCA77u) ^ (z * 0xC2B2AE3Du);
    h = h ^ (h >> 15u);
    h = h * 0x2C1B3C6Du;
    h = h ^ (h >> 12u);
    return h;
}

// slot | flags << 16 of tile (z, x, y), or EMPTY
fn lookup(z: u32, x: u32, y: u32) -> u32 {
    let mask = u.cfg.w;
    var i = tile_hash(z, x, y) & mask;
    for (var k = 0u; k <= mask; k++) {
        let e = table[i];
        if e.w == EMPTY {
            return EMPTY;
        }
        if e.x == z && e.y == x && e.z == y {
            return e.w;
        }
        i = (i + 1u) & mask;
    }
    return EMPTY;
}

// tile entry (slot | flags << 16) holding global integer pixel (ix, iy) of zoom z, or EMPTY
fn tile_of(z: i32, ix: i32, iy: i32) -> u32 {
    let n = 1 << u32(z);
    let ty = iy >> 8u;
    if ty < 0 || ty >= n {
        return EMPTY;
    }
    let tx = (ix >> 8u) & (n - 1);
    return lookup(u32(z), u32(tx), u32(ty));
}

// ---------------------------------------------------------------- pyramid positions

struct GP {
    z: i32,
    ix: i32,
    iy: i32,
    fx: f32,
    fy: f32,
};

fn gp_coarser(p: GP, k: i32) -> GP {
    if k <= 0 {
        return p;
    }
    let m = (1 << u32(k)) - 1;
    let s = 1.0 / f32(1 << u32(k));
    return GP(p.z - k, p.ix >> u32(k), p.iy >> u32(k), (f32(p.ix & m) + p.fx) * s, (f32(p.iy & m) + p.fy) * s);
}

fn gp_add(p: GP, d: vec2<f32>) -> GP {
    let x = p.fx + d.x;
    let y = p.fy + d.y;
    let cx = floor(x);
    let cy = floor(y);
    return GP(p.z, p.ix + i32(cx), p.iy + i32(cy), x - cx, y - cy);
}

fn gp_lat(p: GP) -> f32 {
    let n = 256.0 * exp2(f32(p.z));
    let y = (f32(p.iy) + p.fy) / n;
    let a = PI * (1.0 - 2.0 * y);
    return atan(sinh(a));
}

fn gp_lon(p: GP) -> f32 {
    let n = 256.0 * exp2(f32(p.z));
    return (f32(p.ix) + p.fx) / n * 2.0 * PI - PI;
}

// ---------------------------------------------------------------- sampling

// 0 colour, 1 normal, 2 emission; rgb and 1 if found
fn texel(z: i32, ix: i32, iy: i32, which: u32) -> vec4<f32> {
    let e = tile_of(z, ix, iy);
    if e == EMPTY {
        return vec4<f32>(0.0);
    }
    let slot = i32(e & 0xFFFFu);
    let flags = e >> 16u;
    let c = vec2<i32>(ix & 255, iy & 255);
    if which == 0u {
        if (flags & HAS_COLOR) == 0u {
            return vec4<f32>(0.0);
        }
        return vec4<f32>(textureLoad(t_color, c, slot, 0).rgb, 1.0);
    } else if which == 1u {
        if (flags & HAS_NORMAL) == 0u {
            return vec4<f32>(0.0);
        }
        return vec4<f32>(textureLoad(t_normal, c, slot, 0).rgb, 1.0);
    }
    if (flags & HAS_EMISSION) == 0u {
        return vec4<f32>(0.0);
    }
    let v = textureLoad(t_emis, c, slot, 0).rgb;
    return vec4<f32>(16.0 * v * v * v, 1.0);
}

// bilinear sample with ancestor fallback (CPU: TileView::sample3)
fn sample3(p0: GP, which: u32) -> vec4<f32> {
    var p = p0;
    loop {
        let x = p.fx - 0.5;
        let y = p.fy - 0.5;
        let cx = floor(x);
        let cy = floor(y);
        let fx = x - cx;
        let fy = y - cy;
        let ix = p.ix + i32(cx);
        let iy = p.iy + i32(cy);
        var acc = vec3<f32>(0.0);
        var ws = 0.0;
        var found = false;
        let w = array<f32, 4>((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy);
        for (var t = 0; t < 4; t++) {
            let s = texel(p.z, ix + (t & 1), iy + (t >> 1u), which);
            if s.w > 0.0 {
                acc += s.rgb * w[t];
                ws += w[t];
                found = true;
            }
        }
        if found && ws > 1e-6 {
            return vec4<f32>(acc / ws, 1.0);
        }
        if p.z == 0 {
            return vec4<f32>(0.0);
        }
        p = gp_coarser(p, 1);
    }
    return vec4<f32>(0.0);
}

fn elev_texel(z: i32, ix: i32, iy: i32) -> vec2<f32> {
    let e = tile_of(z, ix, iy);
    if e == EMPTY || ((e >> 16u) & HAS_ELEV) == 0u {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(textureLoad(t_elev, vec2<i32>(ix & 255, iy & 255), i32(e & 0xFFFFu), 0).r, 1.0);
}

// bilinear DSM height with ancestor fallback (CPU: TileView::elev); y = 1 if found
fn elev(p0: GP) -> vec2<f32> {
    var p = p0;
    loop {
        let x = p.fx - 0.5;
        let y = p.fy - 0.5;
        let cx = floor(x);
        let cy = floor(y);
        let fx = x - cx;
        let fy = y - cy;
        let ix = p.ix + i32(cx);
        let iy = p.iy + i32(cy);
        var acc = 0.0;
        var ws = 0.0;
        let w = array<f32, 4>((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy);
        for (var t = 0; t < 4; t++) {
            let s = elev_texel(p.z, ix + (t & 1), iy + (t >> 1u));
            if s.y > 0.0 {
                acc += s.x * w[t];
                ws += w[t];
            }
        }
        if ws > 1e-6 {
            return vec2<f32>(acc / ws, 1.0);
        }
        if p.z == 0 {
            return vec2<f32>(0.0);
        }
        p = gp_coarser(p, 1);
    }
    return vec2<f32>(0.0);
}

fn landcover(p: GP) -> u32 {
    let e = tile_of(p.z, p.ix, p.iy);
    if e == EMPTY || ((e >> 16u) & HAS_LC) == 0u {
        return 0u;
    }
    return textureLoad(t_lc, vec2<i32>(p.ix & 255, p.iy & 255), i32(e & 0xFFFFu), 0).r;
}

// max of the 16x16 block holding the pixel; y = 1 if known
fn block_max(p: GP) -> vec2<f32> {
    let e = tile_of(p.z, p.ix, p.iy);
    if e == EMPTY || ((e >> 16u) & HAS_ELEV) == 0u {
        return vec2<f32>(0.0);
    }
    return vec2<f32>(textureLoad(t_bmax, vec2<i32>((p.ix & 255) >> 4u, (p.iy & 255) >> 4u), i32(e & 0xFFFFu), 0).r, 1.0);
}

fn prime_vertical(sl: f32) -> f32 {
    return u.ell.x / sqrt(1.0 - u.ell.y * sl * sl);
}

// ---------------------------------------------------------------- 64-bit hashing (u64 emulation)

fn mulw(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    let lo = (p00 & 0xFFFFu) | (mid << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = mulw(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}

fn shr64(a: vec2<u32>, n: u32) -> vec2<u32> {
    if n == 0u {
        return a;
    }
    if n >= 32u {
        return vec2<u32>(a.y >> (n - 32u), 0u);
    }
    return vec2<u32>((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}

fn i64of(v: i32) -> vec2<u32> {
    return vec2<u32>(bitcast<u32>(v), select(0u, 0xFFFFFFFFu, v < 0));
}

// value of (h >> 11) / 2^53, i.e. h / 2^64
fn unit64(h: vec2<u32>) -> f32 {
    return f32(h.y) * 2.3283064e-10 + f32(h.x) * 5.421011e-20;
}

fn mod64(h: vec2<u32>, m: u32, two32_mod_m: u32) -> u32 {
    return ((h.y % m) * two32_mod_m + h.x % m) % m;
}

// lamp cell of a pyramid position: global zoom-17 grid of 32 px (CPU: pixel_shade)
fn flicker_cell(p: GP) -> vec2<u32> {
    let m = 17 - p.z;
    var cx: i32;
    var cy: i32;
    if m >= 5 {
        let s = f32(1 << u32(m - 5));
        cx = (p.ix << u32(m - 5)) + i32(floor(p.fx * s));
        cy = (p.iy << u32(m - 5)) + i32(floor(p.fy * s));
    } else {
        cx = p.ix >> u32(5 - m);
        cy = p.iy >> u32(5 - m);
    }
    let a = mul64(i64of(cx), vec2<u32>(0x7F4A7C15u, 0x9E3779B9u));
    return a ^ i64of(cy);
}

// (modulation depth, phase) of a lamp cell (CPU: FlickerConfig::modulation)
fn modulation(cell: vec2<u32>) -> vec2<f32> {
    var h = cell ^ vec2<u32>(0xF11Cu, 0u);
    h = mul64(h ^ shr64(h, 33u), vec2<u32>(0xed558ccdu, 0xff51afd7u));
    h = h ^ shr64(h, 33u);
    let uu = unit64(h);
    let d = select(u.flick2.y, u.flick2.x, uu < u.flick.w);
    let phase = f32(mod64(h, 3u, 1u)) * 2.0 * PI / 3.0 + 0.2 * f32(mod64(shr64(h, 8u), 7u, 4u)) / 7.0;
    return vec2<f32>(d, phase);
}

// ---------------------------------------------------------------- atmosphere (atmo.rs)

fn integ(h0: f32, h1: f32, hs: f32) -> f32 {
    let a = exp(-max(h0, -500.0) / hs);
    let b = exp(-max(h1, -500.0) / hs);
    let dh = h1 - h0;
    if abs(dh) < 1.0 {
        return 0.5 * (a + b);
    }
    return (a - b) * hs / dh;
}

struct TI {
    t: vec3<f32>,
    ins: vec3<f32>,
};

fn transmittance(h_cam: f32, h_pt: f32, d: f32, view: vec3<f32>) -> TI {
    if !has(F_ATMO) {
        return TI(vec3<f32>(1.0), vec3<f32>(0.0));
    }
    let tr = vec3<f32>(5.8e-6, 13.5e-6, 33.1e-6) * (integ(h_cam, h_pt, u.ray_col.w) * d);
    let tm = u.moon_col.w * integ(h_cam, h_pt, u.mie_col.w) * d;
    let tau = tr + vec3<f32>(tm);
    let t = exp(-tau);
    let cos_s = dot(view, u.sun_dir.xyz);
    let mie_phase = 0.7 + 0.8 * pow(max(cos_s, 0.0), 4.0);
    let wr = tr.x + tr.y + tr.z;
    let wm = 3.0 * tm;
    let mixc = (u.ray_col.xyz * wr + u.mie_col.xyz * mie_phase * wm) / max(wr + wm, 1e-12);
    return TI(t, mixc * (vec3<f32>(1.0) - t) * u.zenith.w);
}

fn sky(dir: vec3<f32>, up: vec3<f32>) -> vec3<f32> {
    let e = dot(dir, up);
    let t = pow(1.0 - max(e, 0.0), 3.0);
    var c = u.zenith.xyz * (1.0 - t) + u.horizon.xyz * t;
    if e < 0.0 {
        c = u.horizon.xyz * (0.9 + 0.1 * e);
    }
    let cs = dot(dir, u.sun_dir.xyz);
    c += u.sun_col.xyz * (0.25 * pow(max(cs, 0.0), 8.0) + 0.4 * pow(max(cs, 0.0), 64.0));
    if cs > 0.99996 && e > -0.01 {
        c += u.sun_col.xyz * 20.0;
    }
    let disc = u.moon_dir.w;
    if disc > 0.0 {
        let cm = dot(dir, u.moon_dir.xyz);
        if cm > 0.99999 {
            c += vec3<f32>(0.95, 0.95, 1.0) * disc;
        }
        c += vec3<f32>(0.8, 0.85, 1.0) * (disc * 2e-3 * pow(max(cm, 0.0), 512.0));
    }
    return c;
}

// ---------------------------------------------------------------- shadows

fn sun_visibility(p0: GP) -> f32 {
    let e0 = elev(p0);
    if e0.y == 0.0 {
        return 1.0;
    }
    let h0 = e0.x;
    let lat = gp_lat(p0);
    let tan_e = u.sun_dir.w;
    let dir = vec2<f32>(sin(u.sun2.x), -cos(u.sun2.x));
    let r_earth = u.ell.x;
    var p = p0;
    let sl = sin(lat);
    var texel = 2.0 * PI * prime_vertical(sl) * cos(lat) / (256.0 * exp2(f32(p.z)));
    var dist = 0.0;
    var step = 1.0;
    let bias = 0.4 * texel + 0.3;
    var i = 0;
    loop {
        if i >= 160 {
            break;
        }
        i++;
        let ray_here = h0 + bias + dist * tan_e - dist * dist / (2.0 * r_earth);
        if ray_here > u.horizon.w + 1.0 {
            break;
        }
        let bm = block_max(p);
        if bm.y > 0.0 && ray_here > bm.x + 0.01 {
            let fx = f32(p.ix & 15) + p.fx;
            let fy = f32(p.iy & 15) + p.fy;
            var tx = 3.4e38;
            var ty = 3.4e38;
            if dir.x > 1e-9 {
                tx = (16.0 - fx) / dir.x;
            } else if dir.x < -1e-9 {
                tx = fx / -dir.x;
            }
            if dir.y > 1e-9 {
                ty = (16.0 - fy) / dir.y;
            } else if dir.y < -1e-9 {
                ty = fy / -dir.y;
            }
            let adv = min(tx, ty) + 0.05;
            p = gp_add(p, dir * adv);
            dist += adv * texel;
            if dist > 40000.0 || ray_here > 9000.0 {
                break;
            }
            if dist > 64.0 * texel && p.z > 0 {
                p = gp_coarser(p, 1);
                texel *= 2.0;
            }
            continue;
        }
        p = gp_add(p, dir * step);
        dist += step * texel;
        let ray_h = h0 + bias + dist * tan_e - dist * dist / (2.0 * r_earth);
        if ray_h > 9000.0 {
            break;
        }
        let hh = elev(p);
        if hh.y > 0.0 && hh.x > ray_h {
            return 0.0;
        }
        if i % 12 == 11 && p.z > 0 {
            p = gp_coarser(p, 1);
            texel *= 2.0;
        } else if i % 12 == 5 {
            step = min(step * 1.5, 2.0);
        }
        if dist > 40000.0 {
            break;
        }
    }
    return 1.0;
}

// ---------------------------------------------------------------- steep walls (render::walls)

const WK_NONE: u32 = 0u;
const WK_BUILT: u32 = 1u;
const WK_GLASS: u32 = 2u;
const WK_ROCK: u32 = 3u;
const WK_VEG: u32 = 4u;
const WK_SNOW: u32 = 5u;
const WALL_WRAP: i32 = 4095;

fn wall_weight(nz: f32) -> f32 {
    let t = clamp((0.62 - nz) / 0.22, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// height above the ellipsoid of the point at `range` along a view direction with dot `e` with
// the camera's up (walls::sample_height)
fn sample_height(h_cam: f32, e: f32, range: f32) -> f32 {
    let dv = range * e;
    return h_cam + dv + (range * range - dv * dv) / (2.0 * (6371000.0 + h_cam));
}

fn wall_is_built(c: u32) -> bool {
    return lc_group(c) == LG_BUILT || c == LC_GREENHOUSE;
}

fn wall_kind(top: u32, here: u32) -> u32 {
    if wall_is_built(top) {
        return select(WK_BUILT, WK_GLASS, lc_material(top).x > 0.0);
    }
    var g = lc_group(here);
    if g == LG_WATER || g == LG_UNKNOWN || g == LG_BUILT || g == LG_TRANSPORT {
        g = lc_group(top);
    }
    if g == LG_BARE || g == LG_WATER || g == LG_UNKNOWN || g == LG_BUILT || g == LG_TRANSPORT {
        return WK_ROCK;
    }
    if g == LG_SNOW_ICE {
        return WK_SNOW;
    }
    return WK_VEG;
}

fn wall_hash(a: i32, b: i32) -> f32 {
    var h = (bitcast<u32>(a) * 0x8DA6B343u) ^ (bitcast<u32>(b) * 0xD8163841u);
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return f32(h >> 8u) / 16777216.0;
}

fn wall_inside(f: f32, a: f32, b: f32) -> f32 {
    return select(0.0, 1.0, f >= a && f < b);
}

// colour and night emission of a wall sample (walls::texel)
fn wall_texel(kind: u32, base: vec3<f32>, s: f32, h: f32, ground: f32, top: f32, foot: f32) -> TX {
    if kind == WK_BUILT || kind == WK_GLASS {
        let glass = kind == WK_GLASS;
        let rel = h - ground;
        let fh = select(3.1, 3.8, glass);
        let bw = select(2.4, 1.6, glass);
        let fl = rel / fh;
        let fi = floor(fl);
        let fy = fl - fi;
        let sx = s / bw;
        let bi = floor(sx);
        let fx = sx - bi;
        let body = wall_inside(rel, 0.0, top - ground - 0.9);
        let lit_p = select(0.28, 0.4, glass);
        let lit = select(0.0, 1.0, wall_hash(i32(bi), i32(fi)) < lit_p);
        var col: vec3<f32>;
        var mean: vec3<f32>;
        var win: f32;
        var wf: f32;
        var e_col: vec3<f32>;
        if glass {
            let pane = mix(base, vec3<f32>(0.07, 0.09, 0.11), 0.75);
            let mullion = vec3<f32>(0.30, 0.31, 0.33);
            let spandrel = pane * 0.6;
            let m = wall_inside(fx, 0.0, 0.07);
            let sp = wall_inside(fy, 0.0, 0.2) * (1.0 - m);
            col = pane * (1.0 - m - sp) + mullion * m + spandrel * sp;
            mean = pane * (0.93 - 0.186) + mullion * 0.07 + spandrel * 0.186;
            win = (1.0 - m) * (1.0 - sp);
            wf = 0.93 * 0.8;
            e_col = vec3<f32>(0.85, 0.9, 1.0);
        } else {
            let l = dot(base, vec3<f32>(0.2126, 0.7152, 0.0722));
            let facade = min(mix(base, vec3<f32>(l), 0.35) * 1.1 + vec3<f32>(0.03), vec3<f32>(0.8));
            let glass_c = vec3<f32>(0.05, 0.06, 0.075);
            let slab = wall_inside(fy, 0.0, 0.06);
            win = wall_inside(fy, 0.32, 0.82) * wall_inside(fx, 0.22, 0.78);
            col = mix(facade * (1.0 - 0.15 * slab), glass_c, win);
            wf = 0.5 * 0.56;
            mean = mix(facade * (1.0 - 0.15 * 0.06), glass_c, wf);
            e_col = vec3<f32>(1.0, 0.78, 0.5);
        }
        let k = clamp((1.4 - foot) / 0.7, 0.0, 1.0) * body;
        let e = e_col * (0.9 * (win * lit * k + wf * lit_p * (1.0 - k)) * body);
        return TX(mix(mean, col, k), e);
    }
    if kind == WK_ROCK {
        let b = h / 2.6 + 0.35 * sin(s / 23.0);
        let bi = floor(b);
        let fb = b - bi;
        let layer = 0.8 + 0.4 * wall_hash(i32(bi), 7);
        let bed = 1.08 - 0.16 * fb;
        let j = s / 3.5 + wall_hash(i32(bi), 3);
        let joint = select(1.0, 0.7, j - floor(j) < 0.08);
        let k = clamp((4.0 - foot) / 2.5, 0.0, 1.0);
        let f = 1.0 + (layer * bed * joint / 0.976 - 1.0) * k;
        let rock = mix(base, vec3<f32>(dot(base, vec3<f32>(0.2126, 0.7152, 0.0722))), 0.3);
        return TX(rock * f, vec3<f32>(0.0));
    }
    if kind == WK_VEG {
        let c = wall_hash(i32(floor(s / 3.0)), i32(floor(h / 3.0)));
        let k = clamp((4.0 - foot) / 2.5, 0.0, 1.0);
        return TX(base * (0.8 * (1.0 + (0.5 * c - 0.25) * k)), vec3<f32>(0.0));
    }
    return TX(base * vec3<f32>(0.95, 0.97, 1.0), vec3<f32>(0.0));
}

// The texels the wall reads (nearest, no fallback): the tile entry of q (`e` when q is in p's
// tile: one table lookup for most walls), its land cover (0: none), DSM height (h0: none),
// colour (w = 0: none).
fn wall_entry(q: GP, p: GP, e: u32) -> u32 {
    if q.z == p.z && (q.ix >> 8u) == (p.ix >> 8u) && (q.iy >> 8u) == (p.iy >> 8u) {
        return e;
    }
    return tile_of(q.z, q.ix, q.iy);
}

fn wall_lc(e: u32, q: GP) -> u32 {
    if e == EMPTY || ((e >> 16u) & HAS_LC) == 0u {
        return 0u;
    }
    return textureLoad(t_lc, vec2<i32>(q.ix & 255, q.iy & 255), i32(e & 0xFFFFu), 0).r;
}

fn wall_h(e: u32, q: GP, h0: f32) -> f32 {
    if e == EMPTY || ((e >> 16u) & HAS_ELEV) == 0u {
        return h0;
    }
    return textureLoad(t_elev, vec2<i32>(q.ix & 255, q.iy & 255), i32(e & 0xFFFFu), 0).r;
}

fn wall_col(e: u32, q: GP) -> vec4<f32> {
    if e == EMPTY || ((e >> 16u) & HAS_COLOR) == 0u {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(textureLoad(t_color, vec2<i32>(q.ix & 255, q.iy & 255), i32(e & 0xFFFFu), 0).rgb, 1.0);
}

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

// ---------------------------------------------------------------- shading

struct PS {
    z: i32,
    range: f32,
    lam: f32,
    aniso: f32,
    hdir: vec2<f32>,
    major_texels: f32,
    mul: vec3<f32>,
    add: vec3<f32>,
    emis: vec3<f32>,
    flicker: f32,
    fdepth: f32,
    fphase: f32,
    // the steep wall at the pixel (render::walls), packed (fewer registers): kind + weight / 2,
    // base colour (sqrt, unorm8 x 3), ground and top heights, direction along it (texel space,
    // snorm16 x 2)
    wkw: f32,
    wbase: u32,
    wground: f32,
    wtop: f32,
    wtan: u32,
};

fn pixel_shade(p: GP, range: f32, dir_w: vec3<f32>, shadow: f32, n_geo: vec3<f32>, hs: f32) -> PS {
    let lat = gp_lat(p);
    let lon = gp_lon(p);
    let sl = sin(lat);
    let cl = cos(lat);
    let so = sin(lon);
    let co = cos(lon);
    let up = vec3<f32>(cl * co, cl * so, sl);
    let east = vec3<f32>(-so, co, 0.0);
    let north = vec3<f32>(-sl * co, -sl * so, cl);
    let v = -dir_w;
    let v_enu = vec3<f32>(dot(v, east), dot(v, north), dot(v, up));
    let cos_inc = max(v_enu.z, 0.03);
    let nrad = prime_vertical(sl);
    let texel = 2.0 * PI * nrad * cl / (256.0 * exp2(f32(p.z)));
    let minor = range * u.cam_up.w;
    let major = minor / cos_inc;
    let lam = log2(minor / texel);
    let aniso = clamp(major / minor, 1.0, f32(u.cfg.y));
    var hdir = vec2<f32>(v_enu.x, -v_enu.y);
    if dot(hdir, hdir) > 1e-12 {
        hdir = normalize(hdir);
    } else {
        hdir = vec2<f32>(1.0, 0.0);
    }
    let sky_amb = vec3<f32>(0.80, 0.90, 1.10) * 0.32 * u.sun.y;
    // a steep wall: the geometric normal's slope; kind, base colour, ground and top heights
    var wk = WK_NONE;
    var ww = 0.0;
    var wbase = vec3<f32>(0.0);
    var wground = 0.0;
    var wtop = 0.0;
    var wtan = vec2<f32>(0.0);
    var cls_top = 0u;
    let ng = vec3<f32>(dot(n_geo, east), dot(n_geo, north), dot(n_geo, up));
    let hz = vec2<f32>(ng.x, -ng.y);
    let wt = wall_weight(ng.z);
    if wt > 0.0 && dot(hz, hz) > 1e-6 {
        let out = normalize(hz);
        let e0 = tile_of(p.z, p.ix, p.iy);
        let pt = gp_add(p, -2.5 * out);
        let pg = gp_add(p, 2.5 * out);
        let et = wall_entry(pt, p, e0);
        cls_top = wall_lc(et, pt);
        wk = wall_kind(cls_top, wall_lc(e0, p));
        var b: vec4<f32>;
        if wk == WK_BUILT || wk == WK_GLASS {
            b = wall_col(et, pt);
        } else {
            let pc = gp_coarser(p, min(p.z, 2));
            b = wall_col(tile_of(pc.z, pc.ix, pc.iy), pc);
        }
        wbase = select(vec3<f32>(0.2), b.xyz, b.w > 0.0);
        wtop = wall_h(et, pt, hs);
        wground = wall_h(wall_entry(pg, p, e0), pg, hs);
        wtan = vec2<f32>(-out.y, out.x);
        ww = wt;
    }
    // the class's material (lc_material, generated from terragen::landcover): self-emission
    // with the lights, the glint of water and other glossy classes (as raster.rs)
    var mat = vec4<f32>(0.0);
    if has(F_RELIT) {
        if ww > 0.5 {
            if wk == WK_GLASS {
                mat = lc_material(cls_top);
            }
        } else {
            mat = lc_material(landcover(p));
        }
    }
    var mul: vec3<f32>;
    var add = vec3<f32>(0.0);
    if has(F_RELIT) {
        var n_enu = vec3<f32>(0.0, 0.0, 1.0);
        let ns = sample3(p, 1u);
        if ns.w > 0.0 {
            n_enu = ns.xyz;
        }
        let n = normalize(normalize(east * n_enu.x + north * n_enu.y + up * n_enu.z) * (1.0 - ww) + n_geo * ww);
        let ndl = max(dot(n, u.sun_dir.xyz), 0.0);
        let ndm = max(dot(n, u.moon_dir.xyz), 0.0);
        mul = u.sun_col.xyz * (ndl * 1.25 * shadow) + u.moon_col.xyz * (ndm * 1.25) + sky_amb * (0.6 + 0.4 * dot(n, up));
    } else {
        mul = vec3<f32>(min(0.75 * u.sun.x + 0.25 * u.sun.y, 1.0));
    }
    if has(F_RELIT) && has(F_LIGHTS) {
        let k = min(i32(max(ceil(log2(25.0 / texel)), 0.0)), p.z);
        let e = sample3(gp_coarser(p, k), 2u);
        if e.w > 0.0 {
            mul += e.xyz * (0.2 * u.sun.z);
        }
    }
    if mat.w > 0.0 && u.sun.z > 1e-3 {
        mul += vec3<f32>(mat.w * u.sun.z);
    }
    if has(F_GLINT) && mat.x > 0.0 {
        let sun = u.sun_dir.xyz;
        var nm = up;
        if ww > 0.0 {
            nm = normalize(up * (1.0 - ww) + n_geo * ww);
        }
        let hv = normalize(v + sun);
        let nh = max(dot(nm, hv), 0.0);
        let fres = mat.y + (1.0 - mat.y) * pow(1.0 - clamp(dot(v, nm), 0.0, 1.0), 5.0);
        let sky_c = sky(normalize(dir_w - nm * 2.0 * dot(dir_w, nm)), up);
        mul *= 1.0 - mat.x * fres;
        add += (sky_c * fres + u.sun_col.xyz * (shadow * 1.5 * pow(nh, mat.z))) * mat.x;
    }
    let p_w = u.cam_pos.xyz - v * range;
    var h_pt: f32;
    if abs(cl) > 0.1 {
        h_pt = sqrt(p_w.x * p_w.x + p_w.y * p_w.y) / cl - nrad;
    } else {
        h_pt = abs(p_w.z) / abs(sl) - nrad * (1.0 - u.ell.y);
    }
    let tt = transmittance(u.cam_pos.w, h_pt, range, dir_w);
    var ins = tt.ins;
    if has(F_LIGHTS) && has(F_POLLUTION) {
        let k = min(i32(max(ceil(log2(1500.0 / texel)), 0.0)), p.z);
        let e = sample3(gp_coarser(p, k), 2u);
        if e.w > 0.0 {
            ins += e.xyz * (vec3<f32>(1.0) - tt.t) * (u.sun.z * u.sun.w * 1.5);
        }
    }
    var flicker = 1.0;
    var fd = 0.0;
    var fph = 0.0;
    if has(F_FLICKER) {
        let m = modulation(flicker_cell(p));
        fd = m.x;
        fph = m.y;
        // 1 + d · sinc · cos(ωt + phase)
        flicker = 1.0 + m.x * u.flick.z * (u.flick.x * cos(m.y) - u.flick.y * sin(m.y));
    }
    return PS(p.z, range, lam, aniso, hdir, major / texel, mul * tt.t, add * tt.t + ins, tt.t * u.sun.z, flicker, fd, fph, f32(wk) + 0.5 * ww, pack4x8unorm(vec4<f32>(sqrt(clamp(wbase, vec3<f32>(0.0), vec3<f32>(1.0))), 0.0)), wground, wtop, pack2x16snorm(wtan));
}

struct TX {
    col: vec3<f32>,
    emis: vec3<f32>,
};

fn texture_fetch(ps: PS, p: GP, hs: f32) -> TX {
    // a wall's material at the sample's height and position along the wall
    var wall = TX(vec3<f32>(0.0), vec3<f32>(0.0));
    let ww = 2.0 * fract(ps.wkw);
    if ww > 0.0 {
        // (metres per texel of the sample's zoom, and the patterns' footprint: two sub-samples)
        let minor = ps.range * u.cam_up.w;
        let tz = minor * exp2(f32(ps.z - p.z) - ps.lam);
        let lx = f32(p.ix & WALL_WRAP) + p.fx;
        let ly = f32(p.iy & WALL_WRAP) + p.fy;
        let b = unpack4x8unorm(ps.wbase).xyz;
        let tan = unpack2x16snorm(ps.wtan);
        wall = wall_texel(u32(ps.wkw), b * b, (lx * tan.x + ly * tan.y) * tz, hs, ps.wground, ps.wtop, 2.0 * minor);
        if ww >= 1.0 {
            return TX(wall.col * ps.mul + ps.add, wall.emis * ps.emis);
        }
    }
    let dz = f32(p.z - ps.z);
    let lam = max(ps.lam + dz, 0.0);
    let major_texels = ps.major_texels * exp2(dz);
    let taps = i32(ceil(ps.aniso));
    let lights_on = max(ps.emis.x, max(ps.emis.y, ps.emis.z)) > 1e-9;
    let l0 = floor(lam);
    let t = lam - l0;
    var col = vec3<f32>(0.0);
    var emis = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var li = 0; li < 2; li++) {
        let wl = select(1.0 - t, t, li == 1);
        if wl <= 1e-4 {
            continue;
        }
        let k = min(i32(l0) + li, p.z);
        let s = exp2(-f32(k));
        let ext = min(major_texels * s, 64.0);
        let pk = gp_coarser(p, k);
        for (var i = 0; i < taps; i++) {
            var o = 0.0;
            if taps > 1 {
                o = (f32(i) + 0.5) / f32(taps) - 0.5;
            }
            let c = sample3(gp_add(pk, ps.hdir * (o * ext)), 0u);
            if c.w > 0.0 {
                col += c.xyz * wl;
                wsum += wl;
            }
        }
    }
    var ewsum = 0.0;
    if lights_on {
        let lam_e = max(lam - 1.5, 0.0);
        let l0e = floor(lam_e);
        let te = lam_e - l0e;
        for (var li = 0; li < 2; li++) {
            let wl = select(1.0 - te, te, li == 1);
            if wl <= 1e-4 {
                continue;
            }
            let k = min(i32(l0e) + li, p.z);
            let e = sample3(gp_coarser(p, k), 2u);
            if e.w > 0.0 {
                emis += e.xyz * wl;
                ewsum += wl;
            }
        }
    }
    var e0 = vec3<f32>(0.0);
    if ewsum > 0.0 {
        e0 = emis / ewsum;
    }
    var c0 = vec3<f32>(0.2);
    if wsum > 0.0 {
        c0 = col / wsum;
    }
    if ww > 0.0 {
        c0 += (wall.col - c0) * ww;
        e0 += (wall.emis - e0) * ww;
    }
    return TX(c0 * ps.mul + ps.add, e0 * ps.emis);
}

fn gp_of_sample(g: vec4<f32>) -> GP {
    let un = units[bitcast<u32>(g.w) - 1u];
    let fu = floor(g.x);
    let fv = floor(g.y);
    return GP(i32(un.x), i32(un.y) * 256 + i32(fu), i32(un.z) * 256 + i32(fv), g.x - fu, g.y - fv);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let ow = u.dims.z;
    let oh = u.dims.w;
    if ox >= ow || oy >= oh {
        return;
    }
    let o = oy * ow + ox;
    let ss = u.cfg.x;
    let w = u.dims.x;
    let cs = ss / 2u;
    let r_cam = mat3x3<f32>(vec3<f32>(u.r0.x, u.r1.x, u.r2.x), vec3<f32>(u.r0.y, u.r1.y, u.r2.y), vec3<f32>(u.r0.z, u.r1.z, u.r2.z));
    var geo = vec4<f32>(3.4e38, 0.0, 255.0, 0.0);
    if has(F_GEOM) {
        let x = ox * ss + cs;
        let y = oy * ss + cs;
        let g = textureLoad(gbuf, vec2<u32>(x, y), 0);
        let ray = rays[y * w + x].xyz;
        if bitcast<u32>(g.w) != 0u && any(ray != vec3<f32>(0.0)) {
            geo = vec4<f32>(g.z * ray.z, g.z, f32(landcover(gp_of_sample(g))), 0.0);
        }
        out_geo[o] = geo;
        out_rad[o] = vec4<f32>(0.0);
        return;
    }
    let split = has(F_SPLIT);
    var acc0 = vec3<f32>(0.0);
    var acc1 = vec3<f32>(0.0);
    var acc2 = vec3<f32>(0.0);
    var have_ctx = false;
    var ctx: PS;
    let n = ss * ss;
    for (var k = 0u; k < n; k++) {
        // central sub-sample first, then the others in row order
        var sx: u32;
        var sy: u32;
        if k == 0u {
            sx = cs;
            sy = cs;
        } else {
            var j = k - 1u;
            if j >= cs * ss + cs {
                j += 1u;
            }
            sx = j % ss;
            sy = j / ss;
        }
        let x = ox * ss + sx;
        let y = oy * ss + sy;
        let g = textureLoad(gbuf, vec2<u32>(x, y), 0);
        let ray = rays[y * w + x].xyz;
        if all(ray == vec3<f32>(0.0)) {
            continue;
        }
        let dir_w = r_cam * ray;
        if bitcast<u32>(g.w) == 0u {
            acc0 += sky(dir_w, u.cam_up.xyz);
            continue;
        }
        let range = g.z;
        let p = gp_of_sample(g);
        if sx == cs && sy == cs {
            geo = vec4<f32>(range * ray.z, range, f32(landcover(p)), 0.0);
        }
        // height of the sample (wall patterns)
        let hs = sample_height(u.cam_pos.w, dot(dir_w, u.cam_up.xyz), range);
        var pc: PS;
        let reuse = have_ctx && abs(range - ctx.range) < 0.03 * ctx.range;
        if reuse {
            pc = ctx;
        } else {
            var shadow = 1.0;
            if has(F_SHADOWS) {
                shadow = sun_visibility(p);
            }
            var n_geo = vec3<f32>(0.0);
            if has(F_WALLS) {
                n_geo = r_cam * geo_normal(i32(x), i32(y), ray * range);
            }
            pc = pixel_shade(p, range, dir_w, shadow, n_geo, hs);
            if !have_ctx {
                ctx = pc;
                have_ctx = true;
            }
        }
        let tx = texture_fetch(pc, p, hs);
        if split {
            acc0 += tx.col + tx.emis;
            acc1 += tx.emis * (pc.fdepth * cos(pc.fphase));
            acc2 -= tx.emis * (pc.fdepth * sin(pc.fphase));
        } else {
            acc0 += tx.col + tx.emis * pc.flicker;
        }
    }
    let nf = f32(n);
    out_rad[o] = vec4<f32>(acc0 / nf, 0.0);
    out_geo[o] = geo;
    if split {
        out_fc[o] = vec4<f32>(acc1 / nf, 0.0);
        out_fs[o] = vec4<f32>(acc2 / nf, 0.0);
    }
}
