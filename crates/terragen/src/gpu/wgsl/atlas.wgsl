// The planetary atlas (`atlas/mod.rs`): a cube map of world-scale fields (climate, tectonics,
// lithology, cultures), computed once per world on the CPU and bound read-only. `atlas_sample`
// returns the same values as `Atlas::sample` (to f32 precision): continuous fields by a cubic
// B-spline over 4 x 4 texels, discrete ones from the nearest texel.
//
// Buffer: a header of ATLAS_HEADER words (magic, version, resolution r, apron, words per texel,
// offset of the texel data, number of cultures, offset of the culture table), then 6 faces of
// (r + 2 apron)^2 texels of ATLAS_WORDS words; each word holds two 16-bit slots (low first):
// f16 values for the continuous fields (slots 0..14), bit fields for plate, lithology, culture.

const ATLAS_APRON: u32 = 2u;
const ATLAS_WORDS: u32 = 9u;
const ATLAS_HEADER: u32 = 16u;
const ATLAS_INV_PI_4: f32 = 1.2732395447351628;

@group(0) @binding(5) var<storage, read> atlas: array<u32>;

struct AtlasSample {
    /// smooth elevation (m)
    elevation: f32,
    /// signed distance to the coast (km, > 0 on land)
    coast_km: f32,
    /// annual-mean surface wind (m/s): east, north
    wind: vec2<f32>,
    /// annual precipitation (mm/year)
    precip_mm: f32,
    /// annual mean temperature at sea level (°C)
    temp_c: f32,
    /// warmest − coldest month (°C)
    temp_range_c: f32,
    /// −1 dry summers … +1 summer rains
    regime: f32,
    /// distance to the nearest plate boundary (km)
    plate_dist_km: f32,
    /// closing speed of that boundary (mm/year, < 0 diverging)
    convergence: f32,
    /// tectonic relief potential −1..1
    uplift: f32,
    volcanism: f32,
    glaciation: f32,
    development: f32,
    population: f32,
    /// bit fields of the nearest texel: plate (id 0–5, boundary type 6–8, continental 9,
    /// hotspot track 10), lithology (dominant 0–2, secondary 3–5, secondary share × 510 8–15),
    /// culture (id 0–11, archetype 12–15)
    plate: u32,
    litho: u32,
    culture: u32,
}

fn atlas_plate_id(s: AtlasSample) -> u32 { return s.plate & 63u; }
fn atlas_boundary(s: AtlasSample) -> u32 { return (s.plate >> 6u) & 7u; }
fn atlas_continental(s: AtlasSample) -> bool { return (s.plate & 512u) != 0u; }
fn atlas_hotspot(s: AtlasSample) -> bool { return (s.plate & 1024u) != 0u; }
fn atlas_litho(s: AtlasSample) -> u32 { return s.litho & 7u; }
fn atlas_litho2(s: AtlasSample) -> u32 { return (s.litho >> 3u) & 7u; }
fn atlas_litho2_frac(s: AtlasSample) -> f32 { return f32((s.litho >> 8u) & 255u) / 510.0; }
fn atlas_culture_id(s: AtlasSample) -> u32 { return s.culture & 0xFFFu; }
fn atlas_archetype(s: AtlasSample) -> u32 { return s.culture >> 12u; }

/// Face (x) and continuous texel coordinates (y, z; centres at integers, clamped to the face)
/// of a direction, for faces of `r` texels.
fn atlas_locate(d: vec3<f32>, r: f32) -> vec3<f32> {
    let a = abs(d);
    var axis = 2u;
    if (a.x >= a.y && a.x >= a.z) {
        axis = 0u;
    } else if (a.y >= a.z) {
        axis = 1u;
    }
    let fm = atlas_face(d, axis);
    let x = clamp(atlas_coords(fm, r), vec2<f32>(-0.5), vec2<f32>(r - 0.5));
    return vec3<f32>(fm.x, x);
}

/// The face of `axis` on the side of `d` (x), and the coordinates of `d` along its u and v axes
/// (y, z) and normal (w = |d[axis]|).
fn atlas_face(d: vec3<f32>, axis: u32) -> vec4<f32> {
    if (axis == 0u) {
        if (d.x < 0.0) {
            return vec4<f32>(1.0, -d.y, d.z, -d.x);
        }
        return vec4<f32>(0.0, d.y, d.z, d.x);
    }
    if (axis == 1u) {
        if (d.y < 0.0) {
            return vec4<f32>(3.0, d.x, d.z, -d.y);
        }
        return vec4<f32>(2.0, -d.x, d.z, d.y);
    }
    if (d.z < 0.0) {
        return vec4<f32>(5.0, d.y, d.x, -d.z);
    }
    return vec4<f32>(4.0, d.y, -d.x, d.z);
}

/// Continuous texel coordinates (tangent warp) of `atlas_face`'s result.
fn atlas_coords(fm: vec4<f32>, r: f32) -> vec2<f32> {
    return (atan(fm.yz / fm.w) * ATLAS_INV_PI_4 + 1.0) * (0.5 * r) - 0.5;
}

/// Weight of a face at texel coordinate `x`: 1 inside, ½ on the edge, 0 half a texel beyond.
fn atlas_ramp(x: f32, r: f32) -> f32 {
    let lo = clamp(x + 1.0, 0.0, 1.0);
    let hi = clamp(r - x, 0.0, 1.0);
    return lo * lo * (3.0 - 2.0 * lo) * hi * hi * (3.0 - 2.0 * hi);
}

fn atlas_bspline(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    let s = 1.0 - t;
    return vec4<f32>(s * s * s, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) * (1.0 / 6.0);
}

/// First word of padded texel (f, i, j) (i, j in 0 .. r + 2 apron).
fn atlas_texel(f: u32, i: u32, j: u32) -> u32 {
    let p = atlas[2u] + 2u * ATLAS_APRON;
    return atlas[5u] + ((f * p + j) * p + i) * ATLAS_WORDS;
}

/// Add `w` × the B-spline of face `f` at texel coordinates `x` (in (−1, r)²) to `acc`.
fn atlas_accumulate(f: u32, x: vec2<f32>, w: f32, acc: ptr<function, array<vec2<f32>, 8>>) {
    let c = floor(x);
    let wx = atlas_bspline(x.x - c.x) * w;
    let wy = atlas_bspline(x.y - c.y);
    // first tap in padded indices: c - 1 + apron >= 0
    let i0 = u32(i32(c.x) + 1);
    let j0 = u32(i32(c.y) + 1);
    for (var b = 0u; b < 4u; b++) {
        for (var a = 0u; a < 4u; a++) {
            let wt = wx[a] * wy[b];
            let t = atlas_texel(f, i0 + a, j0 + b);
            for (var k = 0u; k < 8u; k++) {
                (*acc)[k] += wt * unpack2x16float(atlas[t + k]);
            }
        }
    }
}

/// The atlas fields at a direction (any length; ECEF directions are fine). Within half a texel
/// of a face edge the faces meeting there are blended (weights ½ on the edge): no seams.
fn atlas_sample(dir: vec3<f32>) -> AtlasSample {
    let r = f32(atlas[2u]);
    var acc: array<vec2<f32>, 8>;
    var wsum = 0.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let fm = atlas_face(dir, axis);
        if (fm.w <= 0.0) {
            continue;
        }
        let x = atlas_coords(fm, r);
        let w = atlas_ramp(x.x, r) * atlas_ramp(x.y, r);
        if (w <= 0.0) {
            continue;
        }
        atlas_accumulate(u32(fm.x), x, w, &acc);
        wsum += w;
    }
    for (var k = 0u; k < 8u; k++) {
        acc[k] /= wsum;
    }
    var s: AtlasSample;
    s.elevation = acc[0].x;
    s.coast_km = acc[0].y;
    s.wind = acc[1];
    s.precip_mm = acc[2].x;
    s.temp_c = acc[2].y;
    s.temp_range_c = acc[3].x;
    s.regime = acc[3].y;
    s.plate_dist_km = acc[4].x;
    s.convergence = acc[4].y;
    s.uplift = acc[5].x;
    s.volcanism = acc[5].y;
    s.glaciation = acc[6].x;
    s.development = acc[6].y;
    s.population = acc[7].x;
    // the discrete fields: the nearest texel of the direction's face
    let l = atlas_locate(dir, r);
    let n = min(vec2<u32>(floor(l.yz + 0.5)), vec2<u32>(atlas[2u] - 1u));
    let t = atlas_texel(u32(l.x), n.x + ATLAS_APRON, n.y + ATLAS_APRON);
    s.plate = atlas[t + 7u] >> 16u;
    s.litho = atlas[t + 8u] & 0xFFFFu;
    s.culture = atlas[t + 8u] >> 16u;
    return s;
}
