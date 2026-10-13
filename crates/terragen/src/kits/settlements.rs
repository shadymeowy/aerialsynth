//! The settlements kit: hamlets to megacities, airports, ports, dams and reservoirs, quarries
//! and mines, solar and wind farms, gas flares (`docs/design/terrain-next.md` §5.9, §6.1, §6.2,
//! §6.4; `docs/kits/settlements.md`).
//!
//! * [`places`]: the central-place hierarchy (host-side, analytic): where settlements are, their
//!   level, population, radius, culture, form; the API other kits use
//!   ([`places::settlements_near`]).
//! * [`sites`]: infrastructure placement (host) and its pixel looks.
//! * [`town`]: the settlement pixel model (zones, districts, street patterns, lots, buildings,
//!   village forms, lights).
//!
//! The host function turns places and sites into stamps and segments (per tile area, binned like
//! the pass-A pixels); slot 7 (built) draws the places and the land sites, slot 8 (water) the
//! reservoirs, dam walls and the water side of ports. The kit replaces the core's grid towns
//! (`Kit::replaces_towns`): the town masks (`town_urban`, `town_cov`) are published here.
//! WGSL twin: `gpu/wgsl/kits/settlements.wgsl`.

pub mod places;
pub mod sites;
pub mod town;

use crate::features::{Area, FStamp, Features, HostCtx};
use crate::kits::Kit;
use crate::noise::*;
use crate::stack::{hmode, slot, Layer, Stack};
use glam::{DVec2, DVec3};
use sites::tpl;

pub static KIT: Kit = Kit {
    name: "settlements",
    wgsl: include_str!("../gpu/wgsl/kits/settlements.wgsl"),
    layers: &[(slot::BUILT, built), (slot::WATER, water)],
    host: Some(host),
    replaces_towns: true,
    ..Kit::EMPTY
};

/// Places of a level are listed for areas whose pixel is below this size (m): hamlet, village,
/// town, city, metropolis (a place covers at most ~0.2 pixel above it).
pub const GSD_MAX: [f64; 5] = [150.0, 800.0, 3_000.0, 12_000.0, 50_000.0];

/// A place's stamp.
pub fn place_stamp(w: &crate::world::World, pl: &places::Place) -> FStamp {
    let (ex, ey) = sites::axes(w, pl.center, pl.rot);
    let (s, c) = pl.rot.sin_cos();
    let (we, wn) = (pl.wind[0], pl.wind[1]);
    let wind = (-we * s + wn * c).atan2(we * c + wn * s);
    let (s0, s1) = sites::seed_v(pl.id);
    let ext = pl.extent();
    let lv = pl.level as u32 + 8 * pl.form as u32 + 64 * (pl.parent != 0) as u32;
    FStamp {
        center: pl.center,
        ex,
        ey,
        half: [ext as f32, ext as f32],
        template: tpl::PLACE,
        h: pl.population as f32,
        v: [pl.radius as f32, lv as f32, s0, s1, pl.arch as f32, pl.dev as f32, pl.elong as f32, wind as f32],
    }
}

/// The kit's host features for an area: the places of the levels resolved at its pixel size,
/// and the sites.
fn host(ctx: &mut HostCtx, area: &Area) -> Features {
    let mut f = Features::default();
    let w = ctx.world;
    if w.cfg.landuse.towns <= 0.0 || area.gsd >= GSD_MAX[4] {
        return f;
    }
    // hamlets and villages drowned by a reservoir are not drawn (a dam never floods a town):
    // the reservoirs that can reach any of their centres listed here
    let ext_v = places::LEVELS[1].radius[1] * 1.15 * 1.62 * 1.62 + places::WALDHUFEN_STRIP;
    let mut res: Option<Vec<crate::features::FSeg>> = None;
    for (level, &gmax) in GSD_MAX.iter().enumerate() {
        if area.gsd >= gmax {
            continue;
        }
        let ext = places::LEVELS[level].radius[1] * 1.15 * 1.62 * 1.62 + if level <= 1 { places::WALDHUFEN_STRIP } else { 0.0 };
        for pl in places::places_near(w, level, area.center, area.radius + 1.42 * ext) {
            if (pl.center - area.center).length() <= area.radius + 1.42 * pl.extent() + 2.0 {
                if level <= places::VILLAGE as usize {
                    let res = res.get_or_insert_with(|| {
                        let r = area.radius + 1.42 * ext_v + 2.0;
                        let mut towns = Vec::new();
                        for l in places::TOWN as usize..places::NLEVELS {
                            towns.extend(places::places_near(w, l, area.center, r + 32_000.0));
                        }
                        sites::dams_near(ctx, area.center, r, area.gsd, &towns).into_iter().flat_map(|s| s.segs).collect()
                    });
                    // (pending point evaluations: the host is run again once they are in)
                    if sites::drowned(ctx, res, pl.center) == Some(true) {
                        continue;
                    }
                }
                f.stamps.push(place_stamp(w, &pl));
            }
        }
    }
    sites::sites_near(ctx, area.center, area.radius, area.gsd, &mut f);
    f
}

/// Crown colour of town trees.
fn crown(s: &Stack) -> DVec3 {
    s.pal(crate::registry::pal::CROWN_DECID) * 0.9
}

/// The place model's mean over a place's footprint per R² (its radius squared), per level:
/// street-light emission of the CBD and of the other zones (before the development factors),
/// built cover (`town::lamps`, `town::density`, `town::zone_of` at unresolved blocks, the
/// district density factor averaged; `glow_integrals`).
pub const GLOW_K_CBD: [f64; 5] = [0.0, 0.0, 0.00024, 0.00035, 0.00064];
pub const GLOW_K: [f64; 5] = [0.03597, 0.03597, 0.01737, 0.01728, 0.01706];
pub const GLOW_J: [f64; 5] = [1.8895, 1.8895, 1.5689, 1.5689, 1.5689];

/// The low-zoom glow: where a level's places are no longer listed (pixels ≥ `GSD_MAX[level]`:
/// hamlets below z11, villages below z8, towns below z6 …), the pixel shows their expected
/// built cover and street lights — the places expected there ([`places::expected_places`],
/// calibrated against the final places by `GLOW_CAL`) times the place model's mean over a
/// footprint — so the mean of a coarse pixel matches the finer zooms' and night views from
/// altitude keep their city lights.
fn glow(s: &mut Stack) {
    let gsd = s.ctx.gsd;
    let towns = s.world.cfg.landuse.towns;
    if gsd < GSD_MAX[0] || s.is_water || towns <= 0.0 {
        return;
    }
    let t = s.t;
    // (no places on ice sheets or where it is too cold for any: `base_place`'s candidates)
    let habitable = (1.0 - smoothstep(0.5, 0.6, t.glaciation)) * smoothstep(-16.0, -12.0, t.temp);
    if habitable <= 0.0 {
        return;
    }
    let (pop, dev) = (t.population, t.development);
    let arch = s.bio.style.archetype as usize;
    const LAMPS: [DVec3; 3] = [DVec3::new(1.0, 0.48, 0.12), DVec3::new(1.0, 0.72, 0.38), DVec3::new(0.86, 0.92, 1.0)];
    let sodium = 0.25 + 0.25 * (1.0 - dev);
    let lamp = mixc3(LAMPS[1], LAMPS[0], sodium);
    let devm = (0.5 + 0.45 * dev.min(1.0)) * (0.55 + 0.9 * dev);
    let (mut cover, mut emit) = (0.0, DVec3::ZERO);
    for l in 0..places::NLEVELS {
        if gsd < GSD_MAX[l] {
            continue;
        }
        let (n, r2) = places::expected_places(l, pop, arch, dev, towns);
        let a = n * r2 * places::GLOW_CAL[l] * habitable;
        cover += a * GLOW_J[l];
        emit += (LAMPS[2] * GLOW_K_CBD[l] + lamp * GLOW_K[l]) * (a * devm);
    }
    if cover <= 0.0 {
        return;
    }
    let col = srgb3(128.0, 122.0, 112.0) * (0.92 + 0.1 * s.pf.detail);
    s.composite(Layer { cov: cover.min(0.9), albedo: col, emit, clear: 1.0, ..Default::default() });
}

/// Slot 7: the land sites and the place at the sample.
fn built(s: &mut Stack) {
    if s.no_town {
        return;
    }
    glow(s);
    let stamps: Vec<FStamp> = s.stamps().copied().collect();
    if stamps.is_empty() {
        return;
    }
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let fw = s.l.fw;
    let ground = s.l.ground;
    let sun = s.ctx.east * s.sm.sun_h.x + s.ctx.north * s.sm.sun_h.y;
    let mut best: Option<(town::Pl, f64, f64, f64, DVec2)> = None;
    for st in &stamps {
        let sp = match st.template {
            tpl::PLACE => {
                let pl = town::decode(st);
                let q = town::local(&pl, p);
                let r0 = town::rel0(&pl, q);
                let strips = town::in_strips(&pl, q);
                if r0 < 1.62 || strips {
                    let n = town::foot_noise(&pl, p, gsd);
                    let rel = r0 / (1.0 + n);
                    let score = if rel < 1.0 {
                        (1.0 - rel) * if pl.sub { 1.15 } else { 1.0 }
                    } else if strips {
                        0.001
                    } else {
                        -1.0
                    };
                    if score > 0.0 && best.as_ref().is_none_or(|b| score > b.1) {
                        best = Some((pl, score, rel, n, q));
                    }
                }
                continue;
            }
            tpl::AIRPORT => sites::airport_px(st, p, ground, gsd, fw),
            tpl::PORT => sites::port_px(st, p, false, 0.0, gsd, fw),
            tpl::QUARRY => sites::quarry_px(st, p, gsd, fw),
            tpl::SOLAR => sites::solar_px(st, p, gsd, fw),
            tpl::WIND => sites::wind_px(st, p, DVec2::new(sun.dot(st.ex), sun.dot(st.ey)), s.sm.sun_tan, gsd, fw),
            tpl::FLARE => sites::flare_px(st, p, gsd, fw),
            _ => continue,
        };
        if sp.cov > 0.0 || sp.emit != DVec3::ZERO || sp.shade > 0.0 {
            let (dh, hm) = if sp.abs { (sp.dh, hmode::ABS) } else { (sp.dh, hmode::BLEND) };
            s.composite(Layer {
                cov: sp.cov,
                albedo: sp.col,
                dh,
                hmode: hm,
                cls: sp.cls,
                emit: sp.emit,
                clear: sp.clear,
                lit: 1.0 - sp.shade,
                ..Default::default()
            });
            s.m.natural_ok *= 1.0 - sp.cov * sp.clear;
        }
    }
    if let Some((pl, _, rel, n, q)) = best {
        place(s, &pl, rel, n, q, sun);
    }
}

/// The place at the sample: its pixel (explicit where its blocks are resolved, else the zone
/// mean), cast shadows, lights, trees; the town masks.
fn place(s: &mut Stack, pl: &town::Pl, rel: f64, n: f64, q: DVec2, sun: DVec3) {
    use town::*;
    let world = s.world;
    let t = s.t;
    let gsd = s.ctx.gsd;
    let fw = s.l.fw;
    let town_slope = 0.12 + 0.75 * t.rock_expect;
    let slope_ok = 1.0 - smoothstep(0.45, 0.8, town_slope);
    let dens = density(pl, rel) * slope_ok * s.m.river_clear;
    let strips = in_strips(pl, q);
    if dens <= 0.0 && !strips {
        return;
    }
    let env = Env { gsd, fw, n, moist: t.moist, detail: s.pf.detail };
    let city = pl.form == crate::kits::settlements::places::form::CITY as u32;
    let block = if city { 90.0 } else { 40.0 };
    let e_blk = band(block, gsd);
    let d = district(pl, q, rel);
    let z = d.zone;
    let dens = dens * d.dmul;
    let (mcol, mh) = zone_mean(pl, &d, t.moist);
    let mean_cov = dens.powf(0.7);
    let mut ev = Ev { zone: z, pattern: d.pattern, lamp_sp: if city { 24.0 } else { 34.0 }, ds: 1e9, ..Default::default() };
    let mut shade = 0.0;
    if e_blk > 0.0 {
        let sq = DVec2::new(sun.dot(pl.ex), sun.dot(pl.ey)).normalize_or_zero();
        let steps = if world.cfg.satellite.shadows { 4 } else { 1 };
        let tan_el = s.sm.sun_tan;
        for k in 0..steps {
            let qk = q + sq * (3.5 * k as f64);
            let e = place_eval(pl, qk, &d, &env);
            if k == 0 {
                ev = e;
                if e.h > 0.5 || e.cov <= 0.0 || e.street > 0.5 {
                    break;
                }
            } else if e.h > 3.5 * k as f64 * tan_el {
                shade = 0.75;
                break;
            }
        }
    }
    let cov = mean_cov + (ev.cov - mean_cov) * e_blk;
    // no natural vegetation in the built-up area (also between its lots: the same at every zoom)
    s.m.veg *= 1.0 - smoothstep(0.0, 0.08, dens) * if city { 1.0 } else { 0.6 };
    s.m.town_urban = dens;
    if cov <= 0.0 && ev.tree <= 0.0 {
        return;
    }
    let col = mixc3(mcol * (0.92 + 0.1 * s.pf.detail), ev.col, e_blk);
    let h = mh * dens + (ev.h - mh * dens) * e_blk;
    let cls = if e_blk >= 0.5 { ev.cls } else { mean_class(pl, &d, q) };
    let dev = 0.55 + 0.9 * pl.dev;
    let lamp = lamps(pl, &ev, dens, gsd, fw, block, e_blk);
    let win_lit = 0.1 + if z == zone::CBD { 0.25 } else { 0.0 };
    let windows = ev.win * (0.55 * ev.rim * win_lit * band(2.6, gsd));
    let emit = (lamp * (1.0 - ev.roof) + windows + ev.em * e_blk) * dev;
    let in_dsm = if world.cfg.landuse.buildings_in_dsm { hmode::BLEND } else { hmode::NONE };
    s.composite(Layer { cov, albedo: col, dh: h, hmode: in_dsm, cls, emit, lit: 1.0 - shade * e_blk, clear: 1.0, ..Default::default() });
    if ev.tree > 0.0 && e_blk > 0.0 {
        let tc = crown(s) * (0.85 + 0.3 * u01k(ev.sid ^ 0x7EE, 1));
        s.composite(Layer { cov: ev.tree * e_blk, albedo: tc, dh: ev.tree_h, hmode: hmode::BLEND, ..Default::default() });
    }
    s.m.town_cov = cov;
    s.m.natural_ok *= 1.0 - cov;
}

#[inline]
fn mixc3(a: DVec3, b: DVec3, t: f64) -> DVec3 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Slot 8: reservoirs and dam walls on land, the water side of ports on standing water.
fn water(s: &mut Stack) {
    let has = s.l.feat.is_some_and(|(b, i)| !b.segs[i].is_empty() || !b.stamps[i].is_empty());
    if !has || s.no_town {
        return;
    }
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let fw = s.l.fw;
    let l = s.l;
    let standing = l.water > l.ground && l.water_kind != crate::world::water::NONE;
    if standing {
        if l.water_kind != crate::world::water::OCEAN {
            return;
        }
        let stamps: Vec<FStamp> = s.stamps().filter(|st| st.template == tpl::PORT).copied().collect();
        for st in &stamps {
            let sp = sites::port_px(st, p, true, l.water, gsd, fw);
            if sp.cov > 0.0 || sp.emit != DVec3::ZERO {
                s.composite(Layer {
                    cov: sp.cov,
                    albedo: sp.col,
                    dh: sp.dh,
                    hmode: hmode::ABS,
                    cls: sp.cls,
                    emit: sp.emit,
                    solid: true,
                    relit: 1.0,
                    ..Default::default()
                });
            }
        }
        return;
    }
    let ground = l.ground;
    let segs: Vec<crate::features::FSeg> = s.segments().filter(|sg| sg.kind == tpl::RESERVOIR || sg.kind == tpl::DAM).copied().collect();
    if segs.is_empty() {
        return;
    }
    let mut wet: f64 = 0.0;
    let mut ring: f64 = 0.0;
    let mut level = 0.0;
    let mut dam = (0.0, 0.0, 0.0);
    for sg in &segs {
        if sg.kind == tpl::RESERVOIR {
            let (wv, rv) = sites::reservoir_at(sg, p, ground, fw);
            if wv > wet {
                wet = wv;
                level = sg.ha as f64;
            }
            ring = ring.max(rv);
        } else {
            let d = sites::dam_at(sg, p, ground, fw);
            if d.0 > dam.0 {
                dam = d;
            }
        }
    }
    if wet > 0.0 {
        let depth = (level - ground).max(0.0);
        let col = mixc3(srgb3(70.0, 96.0, 84.0), srgb3(28.0, 58.0, 76.0), smoothstep(0.0, 25.0, depth));
        s.composite(Layer {
            cov: wet,
            albedo: col,
            dh: level,
            hmode: hmode::ABS,
            cls: crate::landcover::RESERVOIR,
            water: true,
            relit: 1.0,
            clear: 1.0,
            ..Default::default()
        });
    }
    if ring > 0.0 {
        s.composite(Layer {
            cov: ring * (1.0 - wet) * band(8.0, gsd).max(0.5),
            albedo: srgb3(178.0, 166.0, 140.0),
            cls: crate::landcover::BARE,
            clear: 1.0,
            ..Default::default()
        });
    }
    if dam.0 > 0.0 {
        let col = mixc3(srgb3(190.0, 188.0, 180.0), srgb3(150.0, 148.0, 142.0), dam.2);
        s.composite(Layer { cov: dam.0, albedo: col, dh: dam.1, hmode: hmode::ABS, cls: crate::landcover::DAM, solid: true, relit: 1.0, ..Default::default() });
    }
}

#[inline]
fn srgb3(r: f64, g: f64, b: f64) -> DVec3 {
    crate::surface::srgb(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Places of the default world where the kit draws: (name, lat, lon), for the tests and
    /// stills (`find_settlement_places`, `find_sites`).
    pub(crate) const PLACES: &[(&str, f64, f64)] = &[
        ("metro", 30.35, 128.8487),
        ("metro_outer", 30.45, 128.95),
        ("survey_city", -6.1773, -114.2395),
        ("desert_town", -12.198, -21.4536),
        ("poor_city", -20.8647, -142.6153),
        ("waldhufen", 20.7151, -22.6459),
        ("linear", -9.1620, 96.5530),
        ("kraal", -15.9711, 62.8364),
        ("perched", -38.7777, -166.8082),
        ("airport", -10.5390, 115.3568),
        ("port", -11.7166, -173.5204),
        ("quarry", -20.4582, -152.1378),
        ("solar", 40.4972, 46.9202),
        ("wind", 42.4183, 33.3454),
        ("flare", 24.0306, -57.4828),
        ("dam", -0.6990, -17.6874),
    ];

    fn tile_at(name: &str, z: u8) -> geodesy::tiles::TileId {
        let &(_, lat, lon) = PLACES.iter().find(|p| p.0 == name).unwrap();
        geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)
    }

    /// A point source of GPU point evaluations in rounds (as the GPU generator's settle rounds):
    /// unknown points are pending, evaluated in one batch between runs of the host.
    #[cfg(feature = "gpu")]
    struct GpuRounds<'g> {
        gen: &'g crate::gpu::GpuGenerator,
        done: crate::noise::FxHashMap<(u64, u64, u64), crate::tile::PointTerrain>,
        missing: Vec<(f64, f64, f64)>,
    }

    #[cfg(feature = "gpu")]
    impl crate::features::PointSource for GpuRounds<'_> {
        fn terrain(&mut self, c: &crate::world::Ctx) -> Option<crate::tile::PointTerrain> {
            let key = (c.lat.to_bits(), c.lon.to_bits(), c.gsd.to_bits());
            if let Some(t) = self.done.get(&key) {
                return Some(*t);
            }
            self.missing.push((c.lat, c.lon, c.gsd));
            None
        }
    }

    #[cfg(feature = "gpu")]
    impl GpuRounds<'_> {
        /// The kits' host features of an area, its point evaluations on the GPU.
        fn features(&mut self, w: &crate::world::World, area: &crate::features::Area) -> crate::features::Features {
            for _ in 0..64 {
                self.missing.clear();
                let f = crate::features::build(w, self, area);
                if self.missing.is_empty() {
                    return f;
                }
                let mut pts = std::mem::take(&mut self.missing);
                pts.sort_by(|a, b| a.partial_cmp(b).unwrap());
                pts.dedup();
                let ts = self.gen.terrain_points(&pts).unwrap();
                for (p, t) in pts.iter().zip(ts) {
                    let t = crate::tile::PointTerrain {
                        ground: t.ground as f64,
                        water: if t.water_kind != 0 { t.water as f64 } else { f64::NEG_INFINITY },
                        water_kind: t.water_kind as u8,
                    };
                    self.done.insert((p.0.to_bits(), p.1.to_bits(), p.2.to_bits()), t);
                }
            }
            panic!("host features did not settle");
        }
    }

    /// The sites (airports, ports, dams, quarries, solar, wind, flares) and the transport
    /// network the kits build from GPU point evaluations are the CPU's, bit for bit: close
    /// decisions are re-taken with exact samples (`sites::TIE_H`), the engineered profiles'
    /// samples are quantized (`transport::build::deck_sample`). Run alone (it clears the shared
    /// caches).
    #[cfg(feature = "gpu")]
    #[test]
    fn sites_and_routes_match_the_cpu() {
        let cfg = crate::Config::default();
        let gen = match crate::gpu::GpuGenerator::new(cfg.clone()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU generator ({e}): skipped");
                return;
            }
        };
        let w = crate::world::World::new(cfg);
        let clear = || {
            sites::clear_cache();
            crate::kits::transport::net::clear_cache();
            crate::kits::transport::build::clear_cache();
        };
        let key_st = |st: &FStamp| (st.template, st.center.to_array().map(f64::to_bits), st.half.map(f32::to_bits), st.h.to_bits(), st.v.map(f32::to_bits));
        let key_sg = |sg: &crate::features::FSeg| {
            (
                sg.kind,
                sg.class,
                sg.a.to_array().map(f64::to_bits),
                sg.b.to_array().map(f64::to_bits),
                sg.hw.to_bits(),
                sg.ha.to_bits(),
                sg.hb.to_bits(),
                sg.v.map(f32::to_bits),
                // (the engineered profiles: decks, bridge and tunnel flags)
                [sg.deck_a.to_bits(), sg.deck_b.to_bits(), sg.flags],
            )
        };
        let (mut nst, mut nsg, mut ndeck) = (0, 0, 0);
        for &(name, lat, lon) in PLACES.iter().filter(|p| ["airport", "port", "dam", "solar", "flare", "survey_city"].contains(&p.0)) {
            let center = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &w.ell);
            let area = crate::features::Area { center, radius: 4_000.0, gsd: 20.0 };
            clear();
            let g = GpuRounds { gen: &gen, done: Default::default(), missing: Vec::new() }.features(&w, &area);
            clear();
            let c = crate::features::build(&w, &mut crate::features::CpuPoints(&w), &area);
            // (tunnel portals included: their decks come from the engineered profiles)
            let mut gs: Vec<_> = g.stamps.iter().map(key_st).collect();
            let mut cs: Vec<_> = c.stamps.iter().map(key_st).collect();
            gs.sort();
            cs.sort();
            assert_eq!(gs.len(), cs.len(), "{name}: {} stamps on the GPU, {} on the CPU", gs.len(), cs.len());
            assert!(gs == cs, "{name}: stamps differ");
            // (sites' segments, the routes' polylines and their engineered profiles)
            let mut gg: Vec<_> = g.segs.iter().map(key_sg).collect();
            let mut cg: Vec<_> = c.segs.iter().map(key_sg).collect();
            gg.sort();
            cg.sort();
            let no_deck = crate::kits::transport::NO_DECK.to_bits();
            ndeck += gg.iter().filter(|k| k.8[0] != no_deck || k.8[1] != no_deck).count();
            assert_eq!(gg.len(), cg.len(), "{name}: {} segments on the GPU, {} on the CPU", gg.len(), cg.len());
            assert!(gg == cg, "{name}: segments differ");
            eprintln!("{name}: {} stamps, {} segments identical (decks included)", gs.len(), gg.len());
            nst += gs.len();
            nsg += gg.len();
        }
        eprintln!("{ndeck} segments with engineered decks");
        assert!(nst > 0 && nsg > 0 && ndeck > 0);
    }

    /// The glow's WGSL constants are the Rust ones (`places::GLOW_C1`, `GLOW_CAL`).
    #[test]
    fn glow_wgsl_constants() {
        let w = include_str!("../gpu/wgsl/kits/settlements.wgsl");
        let arr = |v: &[f64; 5]| format!("array<f32, 5>({})", v.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join(", "));
        assert!(w.contains(&format!("var c1 = {};", arr(&places::GLOW_C1))), "c1: {}", arr(&places::GLOW_C1));
        assert!(w.contains(&format!("var cal = {};", arr(&places::GLOW_CAL))), "cal: {}", arr(&places::GLOW_CAL));
        assert!(w.contains(&format!("var kc = {};", arr(&GLOW_K_CBD))));
        assert!(w.contains(&format!("var kr = {};", arr(&GLOW_K))));
        assert!(w.contains(&format!("var kj = {};", arr(&GLOW_J))));
    }

    /// The low-zoom glow keeps the mean light: at each zoom where a level's places stop being
    /// listed, the parent tile's mean emission against its four children's (which draw the
    /// places), summed over populated places. Without the glow the parents lost the level's light.
    #[cfg(feature = "gpu")]
    #[test]
    fn glow_keeps_the_mean() {
        let gen = match crate::gpu::GpuGenerator::new(crate::Config::default()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU generator ({e}): skipped");
                return;
            }
        };
        let mean = |t: &tilestore::TileData| t.emission.iter().map(|&b| 16.0 * (b as f64 / 255.0).powi(3)).sum::<f64>() / t.emission.len() as f64;
        let names = ["metro", "survey_city", "poor_city", "desert_town", "airport", "linear", "waldhufen", "kraal"];
        // (parent zooms: hamlets drop below z11, villages below z8, towns below z6, cities below z4)
        for zp in [10u8, 7, 5, 3] {
            let (mut sp, mut sc) = (0.0, 0.0);
            for name in names {
                let pid = tile_at(name, zp);
                let kids: Vec<_> = (0..4).map(|k| geodesy::tiles::TileId { z: zp + 1, x: 2 * pid.x + k % 2, y: 2 * pid.y + k / 2 }).collect();
                let mut ids = vec![pid];
                ids.extend(kids);
                let t = gen.tiles(&ids).unwrap();
                let p = mean(&t[0]);
                let c = t[1..].iter().map(mean).sum::<f64>() / 4.0;
                eprintln!("z{zp} {name}: parent {p:.3e} children {c:.3e}");
                sp += p;
                sc += c;
            }
            let r = sp / sc.max(1e-12);
            eprintln!("z{zp}: parent / children mean emission {r:.3}");
            assert!((0.6..=1.6).contains(&r), "z{zp}: ratio {r:.3}");
        }
    }

    /// Share of samples differing by more than `tol`.
    #[cfg(feature = "gpu")]
    fn off<T: Copy>(a: &[T], b: &[T], tol: f64, f: impl Fn(T) -> f64) -> f64 {
        a.iter().zip(b).filter(|(x, y)| (f(**x) - f(**y)).abs() > tol).count() as f64 / a.len() as f64
    }

    /// The GPU's tiles are the CPU's where the kit draws.
    #[cfg(feature = "gpu")]
    #[test]
    fn tiles_match_the_cpu() {
        let cfg = crate::Config::default();
        let gen = match crate::gpu::GpuGenerator::new(cfg.clone()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU generator ({e}): skipped");
                return;
            }
        };
        let cpu = crate::Generator::new(cfg);
        let mut worst: f64 = 0.0;
        for (name, z) in [
            ("metro", 12u8),
            ("metro", 15),
            ("metro", 17),
            ("metro_outer", 14),
            ("survey_city", 13),
            ("survey_city", 16),
            ("desert_town", 15),
            ("poor_city", 14),
            ("waldhufen", 14),
            ("linear", 16),
            ("kraal", 17),
            ("perched", 16),
            ("airport", 13),
            ("airport", 15),
            ("port", 14),
            ("port", 16),
            ("quarry", 15),
            ("solar", 16),
            ("wind", 16),
            ("flare", 16),
            ("dam", 11),
            ("dam", 14),
        ] {
            let id = tile_at(name, z);
            let g = gen.tiles(&[id]).unwrap().remove(0);
            let c = cpu.tile_cpu(id);
            let rgb = off(&g.rgb, &c.rgb, 2.0, |v| v as f64);
            let el = off(&g.elevation, &c.elevation, 0.05, |v| v as f64);
            let lc = off(&g.landcover, &c.landcover, 0.5, |v| v as f64);
            let em = off(&g.emission, &c.emission, 2.0, |v| v as f64);
            eprintln!(
                "{name} z{z}: rgb off {:.3}% elevation off {:.3}% landcover off {:.3}% emission off {:.3}%",
                rgb * 100.0,
                el * 100.0,
                lc * 100.0,
                em * 100.0
            );
            worst = worst.max(rgb).max(el).max(lc).max(em);
        }
        assert!(worst < 0.005, "worst {:.3}%", worst * 100.0);
    }

    /// Parent ≈ mean of children where the kit draws (elevation, albedo).
    #[test]
    fn lod_consistent_at_places() {
        let g = crate::Generator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in
            [("metro", 11u8), ("metro", 13), ("metro_outer", 12), ("survey_city", 14), ("desert_town", 13), ("waldhufen", 13), ("airport", 13), ("dam", 12)]
        {
            let parent = tile_at(name, z);
            let p = g.tile(parent);
            let kids = parent.children().map(|c| g.tile(c));
            let (mut eh, mut ea, mut cnt) = (0.0, 0.0, 0.0);
            for (k, kid) in kids.iter().enumerate() {
                let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
                for j in (0..n).step_by(2) {
                    for i in (0..n).step_by(2) {
                        let ix = [j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1];
                        let m: f32 = ix.iter().map(|&q| kid.elevation[q]).sum::<f32>() / 4.0;
                        let pi = (oy + j / 2) * n + ox + i / 2;
                        eh += (m - p.elevation[pi]).abs() as f64;
                        for c in 0..3 {
                            let ma: f64 = ix.iter().map(|&q| kid.albedo[3 * q + c] as f64).sum::<f64>() / 4.0;
                            ea += (ma - p.albedo[3 * pi + c] as f64).abs();
                        }
                        cnt += 1.0;
                    }
                }
            }
            let (eh, ea) = (eh / cnt, ea / (3.0 * cnt));
            eprintln!("{name} z{z}: |parent − mean(children)| elevation {eh:.2} m, albedo {ea:.2} DN");
            assert!(eh < 6.0, "{name}: elevation {eh}");
            assert!(ea < 14.0, "{name}: albedo {ea}");
        }
    }

    /// No seam between neighbouring tiles where the kit draws (E-W and N-S).
    #[test]
    fn seamless_at_places() {
        let g = crate::Generator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in [("metro", 14u8), ("survey_city", 15), ("desert_town", 16), ("waldhufen", 14), ("linear", 15)] {
            let id = tile_at(name, z);
            let a = g.tile(id);
            let (mut ea, mut ei, mut na, mut ni) = (0.0, 0.0, 0.0, 0.0);
            let e = g.tile(geodesy::tiles::TileId::new(z, id.x + 1, id.y));
            let s = g.tile(geodesy::tiles::TileId::new(z, id.x, id.y + 1));
            for j in 0..n {
                for c in 0..3 {
                    ea += (a.albedo[3 * (j * n + n - 1) + c] as f64 - e.albedo[3 * (j * n) + c] as f64).abs();
                    ei += (a.albedo[3 * (j * n + n - 2) + c] as f64 - a.albedo[3 * (j * n + n - 1) + c] as f64).abs();
                    na += (a.albedo[3 * ((n - 1) * n + j) + c] as f64 - s.albedo[3 * j + c] as f64).abs();
                    ni += (a.albedo[3 * ((n - 2) * n + j) + c] as f64 - a.albedo[3 * ((n - 1) * n + j) + c] as f64).abs();
                    ei += (e.albedo[3 * (j * n) + c] as f64 - e.albedo[3 * (j * n + 1) + c] as f64).abs();
                    ni += (s.albedo[3 * j + c] as f64 - s.albedo[3 * (n + j) + c] as f64).abs();
                }
            }
            eprintln!("{name} z{z}: albedo E-W across {ea:.0} inside {ei:.0}; N-S across {na:.0} inside {ni:.0}");
            assert!(ea < 2.0 * ei + 2.0 * n as f64, "{name}: E-W across {ea} inside {ei}");
            assert!(na < 2.0 * ni + 2.0 * n as f64, "{name}: N-S across {na} inside {ni}");
        }
    }

    /// Tiles are a pure function of the world: the same tile twice (fresh generators, cleared
    /// place and site caches) is the same bits.
    #[test]
    fn deterministic_at_places() {
        let cfg = crate::Config { tile_supersample: 1, ..crate::Config::default() };
        let a = crate::Generator::new(cfg.clone());
        let ta: Vec<_> = [("metro", 15u8), ("desert_town", 16), ("linear", 15)].iter().map(|&(nm, z)| a.tile(tile_at(nm, z))).collect();
        places::clear_cache();
        sites::clear_cache();
        let b = crate::Generator::new(cfg);
        for (k, &(nm, z)) in [("metro", 15u8), ("desert_town", 16), ("linear", 15)].iter().enumerate() {
            let y = b.tile(tile_at(nm, z));
            let x = &ta[k];
            assert!(x.elevation == y.elevation && x.albedo == y.albedo && x.landcover == y.landcover && x.emission == y.emission, "{nm}");
        }
    }

    /// Every place generates on the GPU at all zooms (no CPU fallback).
    #[cfg(feature = "gpu")]
    #[test]
    fn places_stay_on_the_gpu() {
        let Ok(g) = crate::gpu::shared() else { return };
        if g.check_generator().is_err() {
            return;
        }
        let gen = crate::Generator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        for &(name, lat, lon) in PLACES {
            let ids: Vec<_> = [2u8, 4, 6, 8, 10, 12, 14, 16].iter().map(|&z| geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)).collect();
            if let Err(e) = gen.tiles(&ids) {
                panic!("{name}: GPU generation failed: {e:#}");
            }
        }
        assert!(gen.backend_name().starts_with("GPU"), "{}", gen.backend_name());
    }

    /// GPU tiles/s at the kit's places (performance budget): `bench_places -- --ignored
    /// --nocapture`.
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn bench_places() {
        let gen = crate::Generator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        gen.tiles(&[geodesy::tiles::TileId::new(3, 4, 3)]).unwrap();
        for z in [8u8, 10, 12, 14, 16] {
            let mut ids = Vec::new();
            for &(_, lat, lon) in PLACES {
                let c = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
                let n = 1i64 << z;
                for dy in 0..4i64 {
                    for dx in 0..4i64 {
                        ids.push(geodesy::tiles::TileId::new(z, (c.x as i64 + dx - 2).rem_euclid(n) as u32, (c.y as i64 + dy - 2).clamp(0, n - 1) as u32));
                    }
                }
            }
            let t0 = std::time::Instant::now();
            for chunk in ids.chunks(16) {
                gen.tiles(chunk).unwrap();
            }
            println!("bench z{z}: {} tiles, {:.1} tiles/s", ids.len(), ids.len() as f64 / t0.elapsed().as_secs_f64());
        }
    }

    /// Debug: render tiles to PNG (`SETTLE_OUT` dir, `SETTLE_SPEC` "name:lat,lon,z;…", 2 × 2
    /// tiles each, `SETTLE_GPU=1` on the GPU): `render_tiles -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn render_tiles() {
        let out = std::path::PathBuf::from(std::env::var("SETTLE_OUT").unwrap_or("/tmp/settle".into()));
        std::fs::create_dir_all(&out).unwrap();
        let spec = std::env::var("SETTLE_SPEC").unwrap_or_default();
        let backend = if std::env::var("SETTLE_GPU").is_ok() { crate::Backend::Gpu } else { crate::Backend::Cpu };
        let night = std::env::var("SETTLE_NIGHT").is_ok();
        let gen = crate::Generator::try_with_backend(crate::Config::default(), backend).unwrap();
        for item in spec.split(';').filter(|s| !s.is_empty()) {
            let (name, rest) = item.split_once(':').unwrap();
            let v: Vec<f64> = rest.split(',').map(|x| x.trim().parse().unwrap()).collect();
            let z = v[2] as u8;
            // the 2 × 2 tiles centred on the point
            let px = geodesy::tiles::latlon_to_pixel(v[0].to_radians(), v[1].to_radians(), z, crate::TILE_SIZE as u32);
            let (x0, y0) = ((px.x / crate::TILE_SIZE as f64 - 0.5).floor() as u32, (px.y / crate::TILE_SIZE as f64 - 0.5).floor() as u32);
            let ids: Vec<_> = (0..4).map(|k| geodesy::tiles::TileId::new(z, x0 + (k % 2), y0 + (k / 2))).collect();
            let t0 = std::time::Instant::now();
            let tiles = gen.tiles(&ids).unwrap();
            let n = crate::TILE_SIZE;
            let mut img = image::RgbImage::new(2 * n as u32, 2 * n as u32);
            for (k, t) in tiles.iter().enumerate() {
                for j in 0..n {
                    for i in 0..n {
                        let o = 3 * (j * n + i);
                        let px = if night {
                            let e = |c: usize| (255.0 * (t.emission[o + c] as f64 / 255.0).sqrt()) as u8;
                            [e(0), e(1), e(2)]
                        } else if std::env::var("SETTLE_ALBEDO").is_ok() {
                            [t.albedo[o], t.albedo[o + 1], t.albedo[o + 2]]
                        } else {
                            [t.rgb[o], t.rgb[o + 1], t.rgb[o + 2]]
                        };
                        img.put_pixel(((k % 2) * n + i) as u32, ((k / 2) * n + j) as u32, image::Rgb(px));
                    }
                }
            }
            img.save(out.join(format!("{name}_z{z}{}.png", if night { "_night" } else { "" }))).unwrap();
            eprintln!("{name} z{z}: {:.1} s", t0.elapsed().as_secs_f64());
        }
    }

    /// Debug: class histograms of a parent and its children (`SETTLE_SPEC` "lat,lon,z").
    #[test]
    #[ignore]
    fn debug_class_lod() {
        let spec = std::env::var("SETTLE_SPEC").unwrap_or("47.092,-2.889,12".into());
        let v: Vec<f64> = spec.split(',').map(|x| x.trim().parse().unwrap()).collect();
        let g = crate::Generator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let parent = geodesy::tiles::tile_for_latlon(v[0].to_radians(), v[1].to_radians(), v[2] as u8);
        let p = g.tile(parent);
        let mut hp = [0f64; 256];
        let mut hk = [0f64; 256];
        for c in &p.landcover {
            hp[*c as usize] += 1.0;
        }
        for kid in parent.children().map(|c| g.tile(c)) {
            for c in &kid.landcover {
                hk[*c as usize] += 0.25;
            }
        }
        let n = (crate::TILE_SIZE * crate::TILE_SIZE) as f64;
        for c in 0..256 {
            if hp[c] + hk[c] > 0.0 {
                eprintln!("class {c:3} {:>14}: parent {:.4} children {:.4}", crate::landcover::name(c as u8), hp[c] / n, hk[c] / n);
            }
        }
    }

    /// Debug: ports of coastal towns: `debug_ports -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn debug_ports() {
        let w = crate::world::World::new(crate::Config::default());
        let mut pts = crate::features::CpuPoints(&w);
        let mut ctx = HostCtx { world: &w, points: &mut pts };
        let mut n = 0;
        for i in 0..20000u64 {
            let h = mix64(0x9027 ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            let a = w.atlas().sample(p.normalize());
            if a.coast_km < 0.0 || a.coast_km > 8.0 || a.population < 0.05 {
                continue;
            }
            for l in 2..5 {
                for pl in places::places_near(&w, l, p, 30_000.0) {
                    if !pl.port || pl.parent != 0 {
                        continue;
                    }
                    let r = sites::port(&mut ctx, &pl);
                    eprintln!(
                        "town {} level {} coast_km {:.1} R {:.0}: {:?}",
                        pl.id,
                        pl.level,
                        w.atlas().sample(pl.center.normalize()).coast_km,
                        pl.radius,
                        r.map(|x| x.map(|s| geodesy::ecef2geodetic(s.center, &w.ell)).map(|g| (g.lat.to_degrees(), g.lon.to_degrees())))
                    );
                    n += 1;
                }
            }
            if n > 12 {
                break;
            }
        }
    }

    /// Debug: sites near the default world's towns: `find_sites -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_sites() {
        let w = crate::world::World::new(crate::Config::default());
        let mut pts = crate::features::CpuPoints(&w);
        let mut ctx = HostCtx { world: &w, points: &mut pts };
        let mut found = std::collections::BTreeMap::new();
        for i in 0..3000u64 {
            let h = mix64(0x517E ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            let a = w.atlas().sample(p.normalize());
            if a.coast_km < 0.0 || a.population < 0.1 {
                continue;
            }
            let mut f = Features::default();
            sites::sites_near(&mut ctx, p, 20_000.0, 50.0, &mut f);
            for st in &f.stamps {
                let g = geodesy::ecef2geodetic(st.center, &w.ell);
                let v = found.entry(st.template).or_insert_with(Vec::new);
                let c = (g.lat.to_degrees(), g.lon.to_degrees());
                if !v.iter().any(|x: &(f64, f64)| (x.0 - c.0).abs() + (x.1 - c.1).abs() < 0.05) {
                    v.push(c);
                }
            }
            for sg in f.segs.iter().filter(|sg| sg.kind == sites::tpl::DAM) {
                let g = geodesy::ecef2geodetic(sg.a, &w.ell);
                let v = found.entry(sg.kind).or_insert_with(Vec::new);
                let c = (g.lat.to_degrees(), g.lon.to_degrees());
                if !v.iter().any(|x: &(f64, f64)| (x.0 - c.0).abs() + (x.1 - c.1).abs() < 0.05) {
                    v.push(c);
                }
            }
            if found.len() >= 8 && found.values().all(|v| v.len() >= 3) {
                break;
            }
        }
        for (k, v) in &found {
            let l: Vec<String> = v.iter().take(4).map(|(a, b)| format!("({a:.4}, {b:.4})")).collect();
            eprintln!("template {k}: {} found: {}", v.len(), l.join(" "));
        }
    }

    #[test]
    fn registry_compiles_with_the_settlements_kit() {
        crate::registry::Registry::builtin().unwrap();
    }

    #[test]
    fn legacy_towns_are_off() {
        assert!(!crate::kits::legacy_towns());
    }

    /// Stamp parameters round-trip (seed, level, form) through f32.
    #[test]
    fn stamp_encoding() {
        let w = crate::world::World::new(crate::Config::default());
        let pl = places::Place {
            id: 0xDEAD_BEEF_1234_5678,
            level: 3,
            center: DVec3::new(6.37e6, 0.0, 0.0),
            lat: 0.0,
            lon: 0.0,
            population: 1.2e6,
            radius: 8000.0,
            culture: 3,
            arch: 6,
            dev: 0.7,
            coastal: false,
            airport: true,
            port: false,
            parent: 0,
            form: places::form::CITY,
            rot: 0.3,
            elong: 1.4,
            score: 10,
            wind: [3.0, 1.0],
            temp: 12.0,
            precip: 700.0,
            elev: 100.0,
            litho: 0,
        };
        let st = place_stamp(&w, &pl);
        let d = town::decode(&st);
        assert_eq!(d.level, 3);
        assert_eq!(d.form, places::form::CITY as u32);
        assert!(!d.sub);
        assert_eq!(d.seed, mix64((pl.id & 0xFF_FFFF) | (((pl.id >> 24) & 0xFF_FFFF) << 24)));
        assert!((d.r - 8000.0).abs() < 1e-3);
        assert!((d.ex.length() - 1.0).abs() < 1e-6 && d.ex.dot(d.ey).abs() < 1e-6);
    }

    /// The BSP partitions a block: every point lies in its lot, lots are within the block.
    #[test]
    fn bsp_lots_partition_the_block() {
        for k in 0..200u64 {
            let h = mix64(k);
            let (bw, bd) = (40.0 + 200.0 * u01k(h, 1), 30.0 + 150.0 * u01k(h, 2));
            let (u, v) = (bw * u01k(h, 3), bd * u01k(h, 4));
            let (_, r) = town::bsp(h, u, v, bw, bd, 4, 9.0);
            assert!(r[0] <= u && u <= r[2] && r[1] <= v && v <= r[3], "{r:?} {u} {v}");
            assert!(r[0] >= 0.0 && r[1] >= 0.0 && r[2] <= bw && r[3] <= bd);
        }
    }
}
