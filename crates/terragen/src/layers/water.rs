//! Water: the surface of standing water (slot 0: the sample is water) and rivers (slot 8, on
//! top of the land).

use crate::landcover as lc;
use crate::noise::*;
use crate::stack::{hmode, Layer, Stack};
use crate::surface::{mixc, srgb, PF_WATER};
use crate::world::water;

/// Oceans and lakes: colour by depth, sediment, a sandy bottom and surf in the shallows.
pub fn standing(s: &mut Stack) {
    let pal = &s.sm.pal;
    let l = s.l;
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let depth = l.water - l.ground;
    let mut col = match l.water_kind {
        water::OCEAN => mixc(pal.ocean_shallow, pal.ocean_deep, smoothstep(0.0, 28.0, depth)),
        _ => mixc(mixc(pal.river, pal.ocean_shallow, 0.25), pal.lake_deep, smoothstep(0.0, 5.0, depth)),
    };
    // sediment / plankton variation
    col *= 1.0 + 0.06 * s.sm.pf_lazy(s.pf, PF_WATER);
    if l.water_kind == water::OCEAN && depth < 3.0 {
        // the sandy bottom shows through clear shallow water
        let sh = 1.0 - smoothstep(0.0, 3.0, depth);
        col = mixc(col, mixc(pal.ocean_shallow, pal.beach, 0.55) * 1.05, 0.75 * sh * sh);
        // surf: thin broken lines of foam along the shore (swash at the water's edge, breakers
        // a little further out), not a blotchy band
        let wave = depth + 0.18 * perlin3(0x5F1, p / 40.0);
        let broken = smoothstep(-0.25, 0.35, perlin3(0x5F2, p / 22.0) + 0.5 * perlin3(0x5F3, p / 7.0));
        let line = |c: f64, w: f64| (-((wave - c) / w).powi(2)).exp();
        let foam = (0.85 * line(0.06, 0.05) + 0.6 * broken * line(0.45, 0.05) + 0.4 * broken * line(1.0, 0.06)) * (0.75 + 0.25 * perlin3(7, p / 3.0)) * band(3.0, gsd);
        col = mixc(col, srgb(225.0, 232.0, 230.0), foam.min(1.0));
    }
    let cls = if l.water_kind == water::OCEAN { lc::OCEAN } else { lc::LAKE };
    s.composite(Layer { cov: 1.0, albedo: col, dh: l.water, hmode: hmode::ABS, cls, relit: 1.0, water: true, ..Default::default() });
}

/// Rivers, on top of everything below the seasonal layer: wet channels, frozen ones, dry beds.
pub fn rivers(s: &mut Stack) {
    let l = s.l;
    let t = s.t;
    if l.river_hw <= 0.0 {
        return;
    }
    let gsd = s.ctx.gsd;
    let pal = &s.sm.pal;
    let fwr = l.fw.max(gsd * 0.35);
    let cov = crate::surface::band_cov(l.river_d, l.river_hw, fwr);
    if cov <= 0.0 {
        return;
    }
    let temp = t.temp;
    let detail = s.pf.detail;
    let wet_r = t.river_wet;
    let mut wcol = mixc(pal.river, pal.lake_deep, smoothstep(30.0, 200.0, l.river_hw * 2.0));
    // mountain rivers carry glacial flour: milky turquoise, not deep dark water
    wcol = mixc(wcol, srgb(96.0, 138.0, 140.0), 0.7 * t.mountain * smoothstep(8.0, 0.0, temp));
    // in the cold the river freezes over and is snowed on (a dark channel through the
    // snowfields read as a crack)
    let ice = smoothstep(-1.5, -4.0, temp + 1.5 * s.pf.snow);
    wcol = mixc(wcol, mixc(pal.snow * 0.9, srgb(170.0, 190.0, 200.0), 0.35 * (0.5 + 0.5 * detail)), ice);
    // dry beds are only a subtle pale line (gravel / sand with some vegetation)
    let dry_col = mixc(s.col, mixc(pal.gravel, pal.sand[2], 0.5) * (1.0 + 0.1 * detail), 0.55);
    let rc = mixc(dry_col, wcol, wet_r);
    let frozen = ice > 0.5 && wet_r > 0.5;
    let cls = if frozen {
        lc::SNOW
    } else if wet_r > 0.5 {
        lc::RIVER
    } else {
        lc::SAND
    };
    // (the snow cover leaves open water free, and frozen rivers a faint line)
    s.m.river_cov = cov * if frozen { 0.4 } else if wet_r > 0.5 { 1.0 } else { 0.0 };
    s.composite(Layer { cov, albedo: rc, dh: l.river_level, hmode: hmode::ABS, cls, relit: 1.0, water: !frozen && wet_r > 0.5, ..Default::default() });
}
