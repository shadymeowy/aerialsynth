//! Steep walls: building facades, cliffs and steep canopy.
//!
//! The renderers drape the top-down layers over the DSM, so a near-vertical face (a building
//! wall, a cliff, a karst tower's flank) would take its colour from the one or two texels its
//! foot and top project to and smear them over its whole height. Where the rendered surface is
//! steep (its geometric normal, from the G-buffer's neighbouring points) the shading replaces
//! that colour by a wall material derived from the class and colour around it:
//!
//! * **built** (the class at the top of the wall is built-up, or a greenhouse): a facade of the
//!   roof's colour (desaturated, lighter) with floors and window bays; lit windows at night;
//!   glass (the top's material glints: towers, greenhouses) is a curtain wall of dark glass,
//!   mullions and spandrels, reflecting the sky with the wall's normal;
//! * **rock** (bare ground or cliff at or above the wall): the local mean colour with strata
//!   (soft beds and bands of hashed tone and saturation, keyed on the elevation only, wavy),
//!   rare faint fractures, and scree at the foot of a high wall;
//! * **vegetation**: the colour at the top (the crown's own), darker (the side of the canopy),
//!   in clumps;
//! * **snow and ice**: the local mean colour, slightly blue.
//!
//! The patterns are functions of the sample's height (from its range along the view ray) and
//! of its position along the wall, faded to their mean where the pixel footprint is too coarse
//! to resolve them. The top and the foot of the wall are the DSM texels 2.5 texels against and
//! along the direction the wall faces. Lighting uses the geometric normal on walls. `shade.wgsl` has the same
//! functions (`wall_*`); both renderers compute the same patterns, except on DX12, where the GPU
//! renderer has no walls (`gpu::walls_compiled`).

use glam::DVec3;
use terragen::landcover::{self as lc, Group};

pub(crate) const NONE: u32 = 0;
pub(crate) const BUILT: u32 = 1;
pub(crate) const GLASS: u32 = 2;
pub(crate) const ROCK: u32 = 3;
pub(crate) const VEG: u32 = 4;
pub(crate) const SNOW: u32 = 5;

/// Mean Earth radius (m) of the sample heights' curvature term.
pub(crate) const R_EARTH: f64 = 6_371_000.0;
/// Texel coordinates along the wall wrap at this many texels (f32 precision on the GPU).
pub(crate) const WRAP: f64 = 4096.0;

/// Weight 0..1 of the wall material for a surface whose geometric normal has up component `nz`
/// (0 below ~52° of slope, 1 above ~66°).
pub(crate) fn weight(nz: f64) -> f64 {
    let t = ((0.62 - nz) / 0.22).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The weight of a **rock** wall: the same ramp, from ~40° of slope (full at ~57°): the
/// top-down texture stretched down a terraced canyon face is a comb of vertical stripes
/// (the renderers evaluate it for every face steeper than ~40° and take the larger).
pub(crate) fn weight_rock(nz: f64) -> f64 {
    let t = ((0.76 - nz) / 0.22).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Height above the ellipsoid (m) of a point at `range` along the unit view direction whose dot
/// product with the camera's up is `e`, for a camera at height `h_cam` (second order in the
/// range: exact enough for the patterns, and precise in f32 on the GPU).
pub(crate) fn sample_height(h_cam: f64, e: f64, range: f64) -> f64 {
    let dv = range * e;
    h_cam + dv + (range * range - dv * dv) / (2.0 * (R_EARTH + h_cam))
}

fn is_built(c: u8) -> bool {
    lc::group(c) == Group::Built || c == lc::GREENHOUSE
}

/// Wall kind from the class at the top of the wall and the class at the sample.
pub(crate) fn kind(top: u8, here: u8) -> u32 {
    if is_built(top) {
        return if lc::material(top).glint > 0.0 { GLASS } else { BUILT };
    }
    let g = match lc::group(here) {
        Group::Water | Group::Unknown | Group::Built | Group::Transport => lc::group(top),
        g => g,
    };
    match g {
        Group::Bare | Group::Water | Group::Unknown | Group::Built | Group::Transport => ROCK,
        Group::SnowIce => SNOW,
        _ => VEG,
    }
}

/// Uniform 0..1 hash of two integers (the same bits as `wall_hash` in shade.wgsl).
pub(crate) fn hash(a: i32, b: i32) -> f64 {
    let mut h = (a as u32).wrapping_mul(0x8DA6_B343) ^ (b as u32).wrapping_mul(0xD816_3841);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h >> 8) as f64 / 16_777_216.0
}

fn mix(a: DVec3, b: DVec3, t: f64) -> DVec3 {
    a + (b - a) * t
}

fn lum(c: DVec3) -> f64 {
    c.dot(DVec3::new(0.2126, 0.7152, 0.0722))
}

fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn inside(f: f64, a: f64, b: f64) -> f64 {
    if f >= a && f < b {
        1.0
    } else {
        0.0
    }
}

/// Colour (linear albedo) and night emission of a wall sample: `kind`, the base colour (the
/// top's for buildings and vegetation, the local mean for rock and snow), `s` metres along the wall, height `h` (m
/// above the ellipsoid), the ground and top heights at the wall, the pixel footprint `foot` (m).
#[allow(clippy::too_many_arguments)]
pub(crate) fn texel(kind: u32, base: DVec3, s: f64, h: f64, ground: f64, top: f64, foot: f64) -> (DVec3, DVec3) {
    match kind {
        BUILT | GLASS => {
            let glass = kind == GLASS;
            let rel = h - ground;
            let (fh, bw) = if glass { (3.8, 1.6) } else { (3.1, 2.4) };
            let fl = rel / fh;
            let fi = fl.floor();
            let fy = fl - fi;
            let sx = s / bw;
            let bi = sx.floor();
            let fx = sx - bi;
            // no windows in the parapet and below the ground
            let body = inside(rel, 0.0, top - ground - 0.9);
            let lit_p = if glass { 0.4 } else { 0.28 };
            let lit = if hash(bi as i32, fi as i32) < lit_p { 1.0 } else { 0.0 };
            let (col, mean, win, wf, e_col) = if glass {
                let pane = mix(base, DVec3::new(0.07, 0.09, 0.11), 0.75);
                let mullion = DVec3::new(0.30, 0.31, 0.33);
                let spandrel = pane * 0.6;
                let m = inside(fx, 0.0, 0.07);
                let sp = inside(fy, 0.0, 0.2) * (1.0 - m);
                let c = pane * (1.0 - m - sp) + mullion * m + spandrel * sp;
                let mean = pane * (0.93 - 0.186) + mullion * 0.07 + spandrel * 0.186;
                let win = (1.0 - m) * (1.0 - sp);
                (c, mean, win, 0.93 * 0.8, DVec3::new(0.85, 0.9, 1.0))
            } else {
                let l = lum(base);
                let facade = (mix(base, DVec3::splat(l), 0.35) * 1.1 + DVec3::splat(0.03)).min(DVec3::splat(0.8));
                let glass_c = DVec3::new(0.05, 0.06, 0.075);
                let slab = inside(fy, 0.0, 0.06);
                let win = inside(fy, 0.32, 0.82) * inside(fx, 0.22, 0.78);
                let c = mix(facade * (1.0 - 0.15 * slab), glass_c, win);
                let wf = 0.5 * 0.56;
                let mean = mix(facade * (1.0 - 0.15 * 0.06), glass_c, wf);
                (c, mean, win, wf, DVec3::new(1.0, 0.78, 0.5))
            };
            // fade the pattern where the footprint cannot resolve the windows
            let k = ((1.4 - foot) / 0.7).clamp(0.0, 1.0) * body;
            let e = e_col * (0.3 * (win * lit * k + wf * lit_p * (1.0 - k)) * body);
            (mix(mean, col, k), e)
        }
        ROCK => {
            // strata follow the elevation (never the position along the wall): beds of hashed
            // tone, soft at their top, gently warped along the wall; broad bands; rare faint
            // fractures; scree at the foot of a high wall
            let b = h / 2.6 + 0.35 * (s / 23.0).sin() + 0.15 * (s / 7.3 + 1.3).sin();
            let bi = b.floor();
            let fb = b - bi;
            let (l0, l1) = (hash(bi as i32, 7), hash(bi as i32 + 1, 7));
            let layer = 0.88 + 0.24 * (l0 + (l1 - l0) * smooth(0.75, 1.0, fb));
            let band = 0.95 + 0.10 * hash((h / 19.0).floor() as i32, 11);
            let bed = 1.0 + 0.04 * (std::f64::consts::TAU * fb).cos();
            let jw = ((0.9 - foot) / 0.5).clamp(0.0, 1.0);
            let j = s / (6.0 + 8.0 * hash(bi as i32, 5)) + hash(bi as i32, 3);
            let joint = if j - j.floor() < 0.03 && hash(bi as i32, 9) < 0.3 { 1.0 - 0.15 * jw } else { 1.0 };
            let k = ((3.2 - foot) / 2.0).clamp(0.0, 1.0);
            let k2 = ((12.0 - foot) / 8.0).clamp(0.0, 1.0);
            let f = 1.0 + (layer * bed * joint - 1.0) * k + (band - 1.0) * k2;
            let sat = 0.3 + 0.3 * (hash(bi as i32, 13) - 0.5) * k;
            let rock = mix(base, DVec3::splat(lum(base)), sat);
            let tl = (1.0 - smooth(1.0, 9.0, h - ground)) * ((top - ground - 6.0) / 6.0).clamp(0.0, 1.0);
            let scree = mix(rock, DVec3::splat(lum(rock) * 1.12), 0.5);
            (mix(rock * f, scree, 0.6 * tl), DVec3::ZERO)
        }
        VEG => {
            let c = hash((s / 3.0).floor() as i32, (h / 3.0).floor() as i32);
            let k = ((4.0 - foot) / 2.5).clamp(0.0, 1.0);
            (base * (0.9 * (1.0 + (0.3 * c - 0.15) * k)), DVec3::ZERO)
        }
        _ => (base * DVec3::new(0.95, 0.97, 1.0), DVec3::ZERO),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds() {
        assert_eq!(kind(lc::BUILDING, lc::ROAD), BUILT);
        assert_eq!(kind(lc::BUILDING_TALL, lc::ROAD_MINOR), GLASS);
        assert_eq!(kind(lc::FOREST, lc::ROCK), ROCK);
        assert_eq!(kind(lc::FOREST, lc::FOREST), VEG);
        assert_eq!(kind(lc::ROCK, lc::OCEAN), ROCK);
        assert_eq!(kind(lc::GLACIER, lc::SNOW), SNOW);
    }

    #[test]
    fn patterns_fade_to_their_mean() {
        // averaged over a wall, the resolved pattern matches the faded one (no brightness jump
        // as the camera moves away)
        let base = DVec3::new(0.3, 0.25, 0.2);
        for kind in [BUILT, GLASS, ROCK, VEG] {
            let (mut near, mut far) = (DVec3::ZERO, DVec3::ZERO);
            let (mut en, mut ef) = (DVec3::ZERO, DVec3::ZERO);
            let n = 400;
            for i in 0..n {
                for j in 0..n {
                    let (s, h) = (i as f64 * 0.511, 100.0 + j as f64 * 0.253);
                    let (c, e) = texel(kind, base, s, h, 100.0, 300.0, 0.1);
                    near += c;
                    en += e;
                    let (c, e) = texel(kind, base, s, h, 100.0, 300.0, 50.0);
                    far += c;
                    ef += e;
                }
            }
            let r = near / far;
            assert!(r.max_element() < 1.08 && r.min_element() > 0.92, "kind {kind}: {r}");
            assert!((en - ef).abs().max_element() <= 0.08 * ef.max_element() + 1e-9, "kind {kind}: {en} vs {ef}");
        }
    }

    #[test]
    fn heights() {
        assert!((sample_height(1000.0, -1.0, 500.0) - 500.0).abs() < 1e-9);
        // 10 km level: ~7.8 m of curvature
        assert!((sample_height(0.0, 0.0, 10_000.0) - 7.85).abs() < 0.01);
    }
}
