//! Slot 3, azonal overrides: rock on steep slopes, sand seas, beaches; then the ground's
//! micro-relief and the masks of natural land use, and the riparian belt.

use crate::landcover as lc;
use crate::noise::*;
use crate::registry::pal;
use crate::stack::{Layer, Stack};
use crate::surface::{mixc, PF_STRATA, PF_STRATA2};

pub fn layer(s: &mut Stack) {
    let t = s.t;
    let pf = s.pf;
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let st = &t.style;
    let detail = pf.detail;
    let patch = pf.patch;
    let slope = s.l.slope;
    // ---- rock (slope + expected): at coarse zooms the resolved slope underestimates the true
    // one, so the slope expected from the relief type is blended in (rock cover stays
    // consistent across zooms)
    let resolve = 1.0 - smoothstep(8.0, 80.0, gsd);
    let exp_slope = 0.12 + 0.75 * t.rock_expect;
    let slope_eff = lerp(exp_slope, slope.max(exp_slope * 0.6), resolve);
    let rock_n = 0.7 * patch + 0.3 * pf.land;
    let rock = smoothstep(0.55, 0.85, slope_eff + 0.25 * detail + 0.25 * rock_n + 0.2 * (t.rock_expect - 0.4) + 0.15 * t.gully)
        * (1.0 - 0.5 * smoothstep(0.3, 0.8, s.m.cover) * (1.0 - t.mountain));
    s.m.rock = rock;
    if rock > 0.0 {
        let ri = st[2] * 2.0;
        let j = (ri.floor() as usize).min(1);
        let mut rc = mixc(s.pal(pal::ROCK + j), s.pal(pal::ROCK + j + 1), ri - j as f64);
        // bands along the contours: their horizontal wavelength shrinks with the slope (a
        // fixed band limit aliased them into hairlines on steep valley walls)
        let strata_h = 6.0 + 10.0 * st[3];
        let strata = (s.l.ground / strata_h + 3.0 * s.sm.pf_lazy(pf, PF_STRATA)).sin();
        let strata_w = std::f64::consts::TAU * strata_h / slope.max(0.05);
        rc *= 1.0 + 0.06 * strata * band(strata_w, 1.5 * gsd) + 0.25 * detail + 0.12 * s.sm.pf_lazy(pf, PF_STRATA2);
        s.composite(Layer::paint_cls(rock, rc, lc::ROCK));
    }
    // ---- sand seas
    s.m.sand = t.sand;
    if t.sand > 0.0 {
        let si = st[1] * 2.0;
        let j = (si.floor() as usize).min(1);
        let sc = mixc(s.pal(pal::SAND + j), s.pal(pal::SAND + j + 1), si - j as f64) * (1.0 + 0.06 * detail);
        let a = smoothstep(0.2, 0.6, t.sand + 0.2 * patch);
        s.composite(Layer::paint_cls(a, sc, lc::SAND));
    }
    // ---- beaches: land within a few metres of sea level is coastal plain (river floodplains
    // near sea level, deltas, are not); beaches up to ~4 m above the sea with a ragged inland
    // edge; nothing is farmed or wooded on them and fields keep back from the shore
    let ground = s.l.ground;
    let coastal = 1.0 - smoothstep(0.3, 0.7, t.floodplain);
    if coastal > 0.0 {
        s.m.shore_keep = 1.0 - coastal * (1.0 - smoothstep(4.0, 8.0, ground + 2.0 * patch));
    }
    if ground < 6.0 && coastal > 0.0 && slope < 0.3 {
        let b = (1.0 - smoothstep(2.6, 4.2, ground + 1.0 * detail + 0.8 * patch)) * coastal * (1.0 - smoothstep(0.15, 0.3, slope));
        s.m.beach = b;
        // dry sand mottled by wind and footprints, darker wet sand only at the water's edge
        let beach = s.pal(pal::BEACH);
        let bc = beach * (1.0 + 0.05 * perlin3(0xBE1, p / 9.0) * band(9.0, gsd) + 0.04 * perlin3(0xBE2, p / 2.5) * band(2.5, gsd));
        let bc = mixc(bc, mixc(beach, s.pal(pal::WET_SAND), 0.7), 1.0 - smoothstep(0.05, 0.3, ground + 0.08 * perlin3(0xBE3, p / 15.0)));
        s.composite(Layer::paint_cls(b, bc, lc::BEACH));
    }
}

/// After the azonal layers: the ground's micro-relief (hummocks, tussocks: decimetres over
/// metres, band-limited), where natural land use can be, the riparian belt.
pub fn finish_ground(s: &mut Stack) {
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let t = s.t;
    let m = &mut s.m;
    m.micro = (0.30 * perlin3(0x9A01, p / 9.0) * band(9.0, gsd) + 0.14 * perlin3(0x9A02, p / 3.2) * band(3.2, gsd) + 0.06 * perlin3(0x9A03, p / 1.1) * band(1.1, gsd))
        * (1.0 - 0.5 * m.rock)
        * (1.0 - 0.7 * m.beach);
    s.height = s.l.ground + s.m.micro;
    let m = &mut s.m;
    m.natural_ok *= (1.0 - m.rock) * (1.0 - m.snow) * (1.0 - m.sand) * (1.0 - m.beach);
    m.flat_ok = 1.0 - smoothstep(0.22, 0.32, s.l.slope);
    // riparian belt along rivers
    let l = s.l;
    if l.river_hw > 0.0 && t.river_wet > 0.3 {
        let ad = l.river_d.abs();
        let belt = 4.0 + 0.6 * l.river_hw.min(60.0);
        s.m.riparian = (1.0 - smoothstep(l.river_hw + 0.3 * belt, l.river_hw + belt, ad))
            * smoothstep(0.35, 0.6, t.moist)
            * smoothstep(0.3, 0.7, t.river_wet)
            * s.m.natural_ok
            * (0.55 + 0.45 * smoothstep(-0.3, 0.3, s.pf.patch));
    }
}
