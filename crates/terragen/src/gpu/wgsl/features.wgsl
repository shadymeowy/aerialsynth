// Linear features and stamps (`features.rs`): host-built, binned per tile like the pass-A
// pixels (a sample's bin: `(*s).l.blk`).

struct FSeg {
    a: vec4<f64>,
    b: vec4<f64>,
    ha: f32,
    hb: f32,
    hw: f32,
    s0: f32,
    deck_a: f32,
    deck_b: f32,
    kind: u32,
    cls: u32,
    flags: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    v: vec4<f32>,
}

struct FStamp {
    center: vec4<f64>,
    ex: vec4<f32>,
    ey: vec4<f32>,
    half: vec2<f32>,
    tmpl: u32,
    h: f32,
    v0: vec4<f32>,
    v1: vec4<f32>,
    _p: vec4<f32>,
}

@group(1) @binding(14) var<storage, read> feat_segs: array<FSeg>;
@group(1) @binding(15) var<storage, read> feat_stamps: array<FStamp>;
/// per bin: first segment index, segments, first stamp index, stamps (into `feat_idx`)
@group(1) @binding(16) var<storage, read> feat_bins: array<vec4<u32>>;
@group(1) @binding(17) var<storage, read> feat_idx: array<u32>;

fn feat_seg_n(blk: u32) -> u32 {
    if (blk >= arrayLength(&feat_bins)) {
        return 0u;
    }
    return feat_bins[blk].y;
}

fn feat_seg(blk: u32, k: u32) -> FSeg {
    return feat_segs[feat_idx[feat_bins[blk].x + k]];
}

fn feat_stamp_n(blk: u32) -> u32 {
    if (blk >= arrayLength(&feat_bins)) {
        return 0u;
    }
    return feat_bins[blk].w;
}

fn feat_stamp(blk: u32, k: u32) -> FStamp {
    return feat_stamps[feat_idx[feat_bins[blk].z + k]];
}

/// Signed distance (m, + left of a→b), along-distance (m, + s0) and fraction of `p` on `s`
/// (`features::seg_frame`).
fn seg_frame(s: FSeg, p: vec3<f64>) -> vec3<f32> {
    let ab = vec3<f32>(s.b.xyz - s.a.xyz);
    let ap = vec3<f32>(p - s.a.xyz);
    let l2 = max(dot(ab, ab), 1e-9);
    let t = clamp(dot(ap, ab) / l2, 0.0, 1.0);
    let foot = s.a.xyz + vec3<f64>(ab * t);
    let up = normalize(vec3<f32>(foot * (1.0lf / 6.4e6lf)));
    let left = normalize_or_zero(cross(up, ab));
    let fp = vec3<f32>(p - foot);
    let side = select(1.0, -1.0, dot(fp, left) < 0.0);
    return vec3<f32>(side * length(fp), s.s0 + t * sqrt(l2), t);
}

/// A point in a stamp's box: (u, v) along its axes (m).
fn stamp_uv(s: FStamp, p: vec3<f64>) -> vec2<f32> {
    let d = vec3<f32>(p - s.center.xyz);
    return vec2<f32>(dot(d, s.ex.xyz), dot(d, s.ey.xyz));
}
