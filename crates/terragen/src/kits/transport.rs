//! The transport kit: roads (motorways, trunk, secondary and local roads, tracks), railways and
//! power lines between the settlements kit's places and sites (`docs/design/terrain-next.md`
//! §6.3, §6.4, P2/P3; `docs/kits/transport.md`).
//!
//! * [`net`]: the graph (relative-neighbourhood graphs of places per network, track ends).
//! * [`route`]: routing an edge through its scene (the relief with the valleys carved in, lakes,
//!   reservoirs, the sea, river crossings, site footprints; corridor DP, smoothing, switchbacks).
//! * [`build`]: the features of an area: engineered profiles (cut and fill, bridges, tunnels),
//!   segments, interchanges, roundabouts, stations, yards, roads and lines to the sites.
//! * [`px`]: the looks per sample (shared by slot 6 and slot 8).
//!
//! Slot 7 (built, after the settlements kit) draws the network and its stamps over the towns;
//! slot 8 (water) the decks over rivers, lakes and the sea. The kit replaces the core's noise-network roads (`Kit::replaces_roads`). WGSL
//! twin: `gpu/wgsl/kits/transport.wgsl`.

pub mod build;
pub mod net;
pub mod px;
pub mod route;

use crate::features::{Area, FSeg, FStamp, Features, HostCtx};
use crate::kits::Kit;
use crate::stack::{hmode, slot, Layer, Stack};
use glam::DVec3;

pub static KIT: Kit = Kit {
    name: "transport",
    wgsl: include_str!("../gpu/wgsl/kits/transport.wgsl"),
    layers: &[(slot::BUILT, built), (slot::WATER, water)],
    host: Some(host),
    prepare: Some(build::prepare),
    replaces_roads: true,
    ..Kit::EMPTY
};

/// Classes of the network (`FSeg::kind` = `KIND0` + class).
pub mod cls {
    pub const MOTORWAY: usize = 0;
    pub const TRUNK: usize = 1;
    pub const SECONDARY: usize = 2;
    pub const LOCAL: usize = 3;
    pub const TRACK: usize = 4;
    pub const RAMP: usize = 5;
    pub const RAIL: usize = 6;
    pub const POWER: usize = 7;
    pub const N: usize = 8;
    /// the kit's kind ids (its kernel id range, 208…)
    pub const KIND0: u32 = 208;
    /// A class is listed for areas whose pixel is below this size (m): it covers ≤ ~5 % of a
    /// pixel above (power lines: their cleared corridors).
    pub const GSD_MAX: [f64; N] = [600.0, 240.0, 160.0, 110.0, 60.0, 160.0, 200.0, 800.0];
    /// land-cover classes
    pub const LANDCOVER: [u8; N] = [
        crate::landcover::MOTORWAY,
        crate::landcover::ROAD_MAJOR,
        crate::landcover::ROAD_MAJOR,
        crate::landcover::ROAD_MINOR,
        crate::landcover::TRACK,
        crate::landcover::MOTORWAY,
        crate::landcover::RAILWAY,
        0,
    ];
}

/// Stamp templates.
pub mod tpl {
    pub const PORTAL: u32 = 216;
    pub const STATION: u32 = 217;
    pub const YARD: u32 = 218;
    pub const ROUNDABOUT: u32 = 219;
}

/// Segment flags.
pub mod flag {
    pub const BRIDGE: u32 = 1;
    pub const TUNNEL: u32 = 2;
    pub const URBAN: u32 = 4;
    pub const PAVED: u32 = 8;
    pub const ELECTRIC: u32 = 16;
    /// over another road at a crossing (on top where no deck is known)
    pub const OVER: u32 = 32;
    /// the deck is a lower bound (approach ramps of overpasses)
    pub const FILL: u32 = 64;
    /// lanes per direction (roads) or tracks (rail): 4 bits
    pub const LANES_SHIFT: u32 = 8;
}

/// Along-distances are kept modulo this (m): every pattern period divides it (dashes 12, 16;
/// sleepers 0.6; car slots 24, 30; lamps 35, 40; masts 60; pylons 300, 350), and f32 keeps
/// millimetres.
pub const S_WRAP: f64 = 8_400.0;

/// "No engineered height" in `FSeg::deck_a`/`deck_b` and portal stamps (on the ground): a
/// sentinel, as WGSL cannot test for NaN.
pub const NO_DECK: f32 = -1.0e7;

#[inline]
pub fn has_deck(d: f32) -> bool {
    d > -1.0e6
}

/// f32-rounded unit vector (the GPU's stamp axes are f32: the CPU uses the same).
pub fn f32v(d: DVec3) -> DVec3 {
    let v = d.as_vec3();
    DVec3::new(v.x as f64, v.y as f64, v.z as f64)
}

fn host(ctx: &mut HostCtx, area: &Area) -> Features {
    build::features(ctx, area)
}

#[inline]
fn is_seg(k: u32) -> bool {
    (cls::KIND0..cls::KIND0 + cls::N as u32).contains(&k)
}

#[inline]
fn is_stamp(t: u32) -> bool {
    (tpl::PORTAL..=tpl::ROUNDABOUT).contains(&t)
}

/// Slot 7 (after the settlements kit: roads run over the towns, the network meets their
/// streets): the network, its stamps; cleared corridors, cast shadows, lamps.
fn built(s: &mut Stack) {
    if s.world.cfg.landuse.roads <= 0.0 {
        return;
    }
    let segs: Vec<FSeg> = s.segments().filter(|g| is_seg(g.kind)).copied().collect();
    let stamps: Vec<FStamp> = s.stamps().filter(|t| is_stamp(t.template)).copied().collect();
    if segs.is_empty() && stamps.is_empty() {
        return;
    }
    let e = px::Env {
        p: s.ctx.p,
        ground: s.l.ground,
        gsd: s.ctx.gsd,
        fw: s.l.fw,
        sun: s.ctx.east * s.sm.sun_h.x + s.ctx.north * s.sm.sun_h.y,
        sun_tan: s.sm.sun_tan,
        below: s.col,
    };
    let mut best = px::Px::none();
    let mut second = px::Px::none();
    let mut emit = DVec3::ZERO;
    let (mut shade, mut shade_low, mut clear, mut road): (f64, f64, f64, f64) = (0.0, 0.0, 0.0, 0.0);
    for sg in &segs {
        // (on a volcano's cone and lava only motorways, trunk roads and railways)
        if s.no_town && !matches!((sg.kind - cls::KIND0) as usize, cls::MOTORWAY | cls::TRUNK | cls::RAMP | cls::RAIL) {
            continue;
        }
        let o = px::seg_px(sg, &e);
        emit += o.emit;
        shade = shade.max(o.shade);
        shade_low = shade_low.max(o.shade_low);
        clear = clear.max(o.clear);
        road = road.max(o.road);
        if o.cov > 0.0 {
            if o.z > best.z {
                second = best;
                best = o;
            } else if o.z > second.z {
                second = o;
            }
        }
    }
    for o in [second, best] {
        if o.cov > 0.0 {
            s.composite(Layer { cov: o.cov, albedo: o.col, dh: o.h, hmode: hmode::ABS, cls: o.cls, relit: 0.5, ..Default::default() });
        }
    }
    for st in &stamps {
        let o = px::stamp_px(st, &e);
        emit += o.emit;
        shade = shade.max(o.shade);
        clear = clear.max(o.clear);
        road = road.max(o.road);
        if o.cov > 0.0 {
            s.composite(Layer { cov: o.cov, albedo: o.col, dh: o.h, hmode: hmode::ABS, cls: o.cls, relit: 0.5, ..Default::default() });
        }
    }
    // (decks shade the ground below them, not the road surfaces)
    let shade = shade.max(shade_low * (1.0 - road));
    // cast shadows (bridges, pylons, wires, cars) without painting; lamps
    s.composite(Layer { cov: if shade > 0.0 { 1e-4 } else { 0.0 }, albedo: s.col, lit: 1.0 - shade, emit, ..Default::default() });
    s.m.veg *= 1.0 - clear;
    s.m.natural_ok *= 1.0 - road;
    s.m.road_cov = s.m.road_cov.max(road);
}

/// Slot 8: decks over rivers, reservoirs, lakes and the sea.
fn water(s: &mut Stack) {
    if s.world.cfg.landuse.roads <= 0.0 {
        return;
    }
    let l = s.l;
    let standing = l.water > l.ground && l.water_kind != crate::world::water::NONE;
    let river = l.river_hw > 0.0 && l.river_d.abs() < l.river_hw + 0.5 * l.fw.max(0.35 * s.ctx.gsd);
    if !standing && !river && !s.is_water {
        return;
    }
    let segs: Vec<FSeg> = s.segments().filter(|g| is_seg(g.kind)).copied().collect();
    if segs.is_empty() {
        return;
    }
    let p = s.ctx.p;
    let fw = l.fw;
    let wl = s.height;
    let mut best = px::Px::none();
    for sg in &segs {
        let o = px::deck_px(sg, p, l.ground, wl, fw);
        if o.cov > 0.0 && o.z > best.z {
            best = o;
        }
    }
    if best.cov > 0.0 {
        s.composite(Layer { cov: best.cov, albedo: best.col, dh: best.h, hmode: hmode::ABS, cls: best.cls, relit: 1.0, solid: true, ..Default::default() });
        s.m.road_cov = s.m.road_cov.max(best.road);
    }
}

#[cfg(test)]
mod tests;
