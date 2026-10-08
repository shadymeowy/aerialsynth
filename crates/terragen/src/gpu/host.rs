//! Host side of the GPU generator: the parts of the world that are graphs or site lists rather
//! than per-pixel fields (the drainage network of `hydro.rs`, lake levels, sink lakes), kept
//! in caches across batches. Everything they need from the terrain itself comes from GPU point
//! evaluations: a computation that lacks one records the request and goes on with a stand-in
//! value; its result is then discarded, the requests are evaluated on the GPU in one batch, and
//! the computation runs again (`Prep::missing`).

use super::types::*;
use crate::noise::*;
use crate::world::{Ctx, Seg, World};
use geodesy::Geodetic;
use glam::DVec3;

pub(crate) type Cell = (i64, i64, i64);

const KM: f64 = 1000.0;

#[derive(Clone, Copy)]
struct FlowPt {
    s: DVec3,
    h: f64,
    active: bool,
}

/// Key of a point evaluation: mode, surface point, pixel size.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PointKey(u32, u64, u64, u64, u64);

impl PointKey {
    pub(crate) fn new(mode: u32, c: &Ctx) -> Self {
        PointKey(mode, c.p.x.to_bits(), c.p.y.to_bits(), c.p.z.to_bits(), c.gsd.to_bits())
    }
}

/// A point evaluation the GPU is to run.
pub(crate) struct PointReq {
    pub ctx: Ctx,
    pub mode: u32,
    pub segs: Vec<Seg>,
    pub sinks: Vec<GSink>,
}

/// The GPU input of a point.
pub(crate) fn point_in(c: &Ctx, mode: u32, dr: GDrain) -> GPointIn {
    GPointIn {
        p: [c.p.x, c.p.y, c.p.z, 0.0],
        sl: c.up.z as f32,
        cl: c.north.z as f32,
        so: (-c.east.x) as f32,
        co: c.east.y as f32,
        lat: c.lat as f32,
        gsd: c.gsd as f32,
        mode,
        _p: 0,
        dr,
        _q: [0; 4],
    }
}

/// Results of GPU evaluations and what was derived from them (pure functions of the world).
#[derive(Default)]
pub(crate) struct Cache {
    flow: FxHashMap<(usize, Cell), FlowPt>,
    tgt: FxHashMap<(usize, Cell), Option<Cell>>,
    src: FxHashMap<(usize, Cell), bool>,
    pub points: FxHashMap<PointKey, GTerrain>,
    /// lake levels by (id, radius bits)
    lakes: FxHashMap<(u64, u64), Option<f64>>,
    lakes_forced: FxHashMap<(u64, u64), Option<f64>>,
    /// levels of the lattice lakes by id (what the GPU looks up)
    pub lattice_lakes: FxHashMap<u64, Option<f64>>,
    pub regions: FxHashMap<u64, GRegion>,
    towns_base: FxHashMap<u64, TownBase>,
    towns: FxHashMap<u64, TownBase>,
    /// existing towns around each cell of the town lattice: (id, town)
    pub town_cands: FxHashMap<Cell, Vec<(u64, GTown)>>,
}

/// A town site before / after resolving overlaps (`TownInfo`).
#[derive(Clone, Copy)]
struct TownBase {
    exists: bool,
    center: DVec3,
    ex: DVec3,
    ey: DVec3,
    radius: f64,
    block: f64,
    street: f64,
    organic: f64,
    roof_style: f64,
    height: f64,
    lot: f64,
    elong: f64,
    seed: u64,
    sun: DVec3,
}

impl TownBase {
    fn gpu(&self) -> GTown {
        let v = |d: DVec3| [d.x as f32, d.y as f32, d.z as f32, 0.0];
        GTown {
            center: [self.center.x, self.center.y, self.center.z, 0.0],
            inv_r09: 1.0 / (0.9 * self.radius),
            inv_r035: 1.0 / (0.35 * self.radius),
            seed: self.seed,
            _p: 0,
            ex: v(self.ex),
            ey: v(self.ey),
            sun: v(self.sun),
            radius: self.radius as f32,
            block: self.block as f32,
            street: self.street as f32,
            organic: self.organic as f32,
            roof_style: self.roof_style as f32,
            height: self.height as f32,
            lot: self.lot as f32,
            elong: self.elong as f32,
            _q: [0.0; 4],
        }
    }
}

impl Cache {
    /// Bound the memory of a long-lived cache.
    pub fn trim(&mut self) {
        if self.flow.len() + self.tgt.len() + self.src.len() + self.points.len() + self.lakes.len() + self.regions.len() + self.towns_base.len() + self.town_cands.len() > 4_000_000 {
            *self = Cache::default();
        }
    }
}

/// One pass of host preparation over a cache: what it computes is complete only when nothing
/// was missing.
pub(crate) struct Prep<'a> {
    pub w: &'a World,
    pub c: &'a mut Cache,
    /// GPU evaluations still needed (deduplicated)
    pub need: FxHashMap<PointKey, PointReq>,
    /// count of stand-in values used so far
    pub missing: usize,
}

impl<'a> Prep<'a> {
    pub fn new(w: &'a World, c: &'a mut Cache) -> Self {
        Prep { w, c, need: FxHashMap::default(), missing: 0 }
    }

    /// The point evaluation `mode` at `ctx`, if done; else it is requested (once its own inputs
    /// are complete).
    pub fn point(&mut self, mode: u32, ctx: &Ctx) -> Option<GTerrain> {
        let key = PointKey::new(mode, ctx);
        if let Some(t) = self.c.points.get(&key) {
            return Some(*t);
        }
        self.missing += 1;
        if self.need.contains_key(&key) {
            return None;
        }
        let before = self.missing;
        let (segs, sinks) = self.drainage(mode, ctx);
        if self.missing == before {
            self.need.insert(key, PointReq { ctx: *ctx, mode, segs, sinks });
        }
        None
    }

    /// The drainage data of a point evaluation (`terrain_impl` without precomputed pieces).
    fn drainage(&mut self, mode: u32, ctx: &Ctx) -> (Vec<Seg>, Vec<GSink>) {
        if mode == MODE_RELIEF || !self.w.cfg.hydro.rivers {
            return (vec![], vec![]);
        }
        let segs = self.river_segments(ctx.p, 0.0, ctx.gsd);
        let mut sinks = Vec::new();
        if mode == MODE_FULL {
            for (id, c, rad) in World::sink_lakes(&segs) {
                if (ctx.p - c).length() < 1.6 * rad {
                    let level = self.lake_level_forced(id, c, rad);
                    sinks.push(gsink(id, c, rad, level));
                }
            }
            self.lattice_lakes_at(ctx.p, 0.0);
        }
        (segs, sinks)
    }

    /// Make sure the levels of the lattice lakes a point within `radius` of `p` may look up are
    /// known (the two nearest lake sites of the point, see `terrain_impl`).
    pub fn lattice_lakes_at(&mut self, p: DVec3, radius: f64) {
        let cfg = &self.w.cfg.hydro;
        if cfg.lake_density <= 0.0 {
            return;
        }
        let cell = cfg.lake_cell_km * KM;
        let sites = worley3_sites(self.w.seed ^ 0x1A4E, p, cell, 0.85);
        let _ = radius;
        for (id, pt) in sites {
            self.lattice_lake(id, pt * cell, p);
        }
    }

    /// The level of lattice lake `id` (site `pt`) if a point at `p` may need it.
    pub fn lattice_lake(&mut self, id: u64, pt: DVec3, p: DVec3) {
        let cfg = &self.w.cfg.hydro;
        let cell = cfg.lake_cell_km * KM;
        // (prob <= density · 1.2)
        if u01k(id, 1) > cfg.lake_density * 1.2 {
            return;
        }
        let rad = (300.0 * (u01k(id, 2) * 2.7).exp()).min(cell * 0.3);
        let pc = pt.normalize() * p.length();
        if (p - pc).length() > rad * 1.5 + 2.0 {
            return;
        }
        self.lattice_lake_level(id, pt);
    }

    /// The level of lattice lake `id` (site `pt`), kept for the GPU's lookups.
    pub fn lattice_lake_level(&mut self, id: u64, pt: DVec3) {
        if self.c.lattice_lakes.contains_key(&id) {
            return;
        }
        let cell = self.w.cfg.hydro.lake_cell_km * KM;
        let rad = (300.0 * (u01k(id, 2) * 2.7).exp()).min(cell * 0.3);
        let before = self.missing;
        let level = self.lake_level(id, pt, rad);
        if self.missing == before {
            self.c.lattice_lakes.insert(id, level);
        }
    }

    // ------------------------------------------------------------ drainage (`hydro.rs`)

    fn flow_point(&mut self, lvl: usize, c: Cell) -> FlowPt {
        if let Some(f) = self.c.flow.get(&(lvl, c)) {
            return *f;
        }
        let cell = self.w.cfg.hydro.levels[lvl].cell_km * KM;
        let hh = hash3(self.w.level_key(lvl), c.0, c.1, c.2);
        let p = DVec3::new(
            c.0 as f64 + 0.5 + 0.8 * (u01k(hh, 1) - 0.5),
            c.1 as f64 + 0.5 + 0.8 * (u01k(hh, 2) - 0.5),
            c.2 as f64 + 0.5 + 0.8 * (u01k(hh, 3) - 0.5),
        ) * cell;
        let g = geodesy::ecef2geodetic(p, &self.w.ell);
        let active = g.h.abs() < 0.5 * cell;
        let fp = if active {
            let ctx = Ctx::new(g.lat, g.lon, cell / 4.0, &self.w.ell);
            match self.point(MODE_RELIEF, &ctx) {
                Some(t) => FlowPt { s: ctx.p, h: t.ground as f64, active: true },
                // a stand-in (the result is discarded)
                None => return FlowPt { s: ctx.p, h: 0.0, active: true },
            }
        } else {
            FlowPt { s: DVec3::ZERO, h: 0.0, active: false }
        };
        self.c.flow.insert((lvl, c), fp);
        fp
    }

    /// Downstream neighbour (steepest descent) of an active point (`World::flow_target`).
    fn flow_target(&mut self, lvl: usize, c: Cell) -> Option<Cell> {
        if let Some(t) = self.c.tgt.get(&(lvl, c)) {
            return *t;
        }
        let before = self.missing;
        let cell = self.w.cfg.hydro.levels[lvl].cell_km * KM;
        let me = self.flow_point(lvl, c);
        let mut best = None;
        if me.active && me.h > -150.0 {
            let mut best_slope = 0.0;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let n = (c.0 + dx, c.1 + dy, c.2 + dz);
                        let o = self.flow_point(lvl, n);
                        if !o.active {
                            continue;
                        }
                        let d = (o.s - me.s).length();
                        if d < 0.2 * cell || d > 1.8 * cell {
                            continue;
                        }
                        let slope = (o.h - me.h) / d;
                        if slope < best_slope {
                            best_slope = slope;
                            best = Some(n);
                        }
                    }
                }
            }
            if best.is_none() {
                for dz in -2..=2i64 {
                    for dy in -2..=2i64 {
                        for dx in -2..=2i64 {
                            if dx.abs().max(dy.abs()).max(dz.abs()) < 2 {
                                continue;
                            }
                            let n = (c.0 + dx, c.1 + dy, c.2 + dz);
                            let o = self.flow_point(lvl, n);
                            if !o.active {
                                continue;
                            }
                            let d = (o.s - me.s).length();
                            if d > 3.0 * cell {
                                continue;
                            }
                            let slope = (o.h - me.h) / d;
                            if slope < best_slope {
                                best_slope = slope;
                                best = Some(n);
                            }
                        }
                    }
                }
            }
        }
        if self.missing == before {
            self.c.tgt.insert((lvl, c), best);
        }
        best
    }

    /// No other point drains into `c` (`World::is_source`).
    fn is_source(&mut self, lvl: usize, c: Cell) -> bool {
        if let Some(v) = self.c.src.get(&(lvl, c)) {
            return *v;
        }
        let before = self.missing;
        let mut src = true;
        'n: for dz in -2..=2i64 {
            for dy in -2..=2i64 {
                for dx in -2..=2i64 {
                    let n = (c.0 + dx, c.1 + dy, c.2 + dz);
                    if n != c && self.flow_point(lvl, n).active && self.flow_target(lvl, n) == Some(c) {
                        src = false;
                        break 'n;
                    }
                }
            }
        }
        if self.missing == before {
            self.c.src.insert((lvl, c), src);
        }
        src
    }

    /// All drainage pieces whose valley could reach within `radius` of `center`
    /// (`World::river_segments`).
    pub fn river_segments(&mut self, center: DVec3, radius: f64, gsd: f64) -> Vec<Seg> {
        let mut out = Vec::new();
        if !self.w.cfg.hydro.rivers {
            return out;
        }
        let levels = self.w.cfg.hydro.levels.clone();
        for (lvl, lc) in levels.iter().enumerate() {
            let first = out.len();
            let cell = lc.cell_km * KM;
            if lc.valley_m < 0.2 * gsd && lc.width_m[1] < 0.15 * gsd {
                continue;
            }
            let reach = radius + 4.6 * cell + lc.valley_m;
            let lo = ((center - DVec3::splat(reach)) / cell).floor();
            let hi = ((center + DVec3::splat(reach)) / cell).floor();
            let width = |c: Cell| {
                let hh = hash3(self.w.level_key(lvl) ^ 0x51DE, c.0, c.1, c.2);
                (0.5 * (lc.width_m[0] + (lc.width_m[1] - lc.width_m[0]) * u01k(hh, 1)), lc.valley_m * (0.6 + 0.8 * u01k(hh, 2)))
            };
            for c in self.w.shell_cells(center, lo, hi, cell) {
                if !self.w.maybe_active(lvl, c, cell) {
                    continue;
                }
                let fp = self.flow_point(lvl, c);
                if !fp.active || (fp.s - center).length() > reach {
                    continue;
                }
                let Some(tc) = self.flow_target(lvl, c) else { continue };
                let tp = self.flow_point(lvl, tc);
                let (hw, valley) = width(c);
                let seg = |a: DVec3, b: DVec3, ha: f64, hb: f64, hw: f64, hw_b: f64| Seg { a, b, ha, hb, level: lvl as u8, hw, valley, hw_b, sink: false };
                let mid = 0.5 * (fp.s + tp.s);
                let hmid = 0.5 * (fp.h + tp.h);
                if self.is_source(lvl, c) {
                    out.push(seg(fp.s, mid, fp.h, hmid, 0.08 * hw, hw));
                }
                match self.flow_target(lvl, tc) {
                    Some(ttc) => {
                        let tq = self.flow_point(lvl, ttc);
                        let (hw2, _) = width(tc);
                        let mid2 = 0.5 * (tp.s + tq.s);
                        let hmid2 = 0.5 * (tp.h + tq.h);
                        let at = |t: f64| {
                            let (u, v) = ((1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t));
                            (mid * u + tp.s * v + mid2 * (t * t), hmid * u + tp.h * v + hmid2 * (t * t), hw + (hw2 - hw) * t)
                        };
                        let n = 6usize;
                        for k in 0..n {
                            let (a, ha, wa) = at(k as f64 / n as f64);
                            let (b, hb, wb) = at((k + 1) as f64 / n as f64);
                            out.push(seg(a, b, ha, hb, wa, wb));
                        }
                    }
                    None => {
                        out.push(Seg { sink: tp.h > 0.0, ..seg(mid, tp.s, hmid, tp.h, hw, hw) });
                    }
                }
            }
            let keep = radius + 0.4 * cell + 1.4 * lc.valley_m + 0.35 * lc.meander * cell;
            let mut k = first;
            for i in first..out.len() {
                let s = out[i];
                let ab = s.b - s.a;
                let u = ((center - s.a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                if (center - (s.a + ab * u)).length() <= keep {
                    out[k] = s;
                    k += 1;
                }
            }
            out.truncate(k);
        }
        out
    }

    // ------------------------------------------------------------ lakes

    /// The spill level of a lake basin, or None on a slope / in the sea (`World::lake_level`).
    pub fn lake_level(&mut self, id: u64, center: DVec3, rad: f64) -> Option<f64> {
        let key = (id, rad.to_bits());
        if let Some(v) = self.c.lakes.get(&key) {
            return *v;
        }
        let before = self.missing;
        let g = geodesy::ecef2geodetic(center, &self.w.ell);
        let cctx = Ctx::new(g.lat, g.lon, 20.0, &self.w.ell);
        let tc = self.point(MODE_NOLAKES, &cctx);
        let mut rim = f64::MAX;
        for k in 0..10 {
            let a = k as f64 * std::f64::consts::TAU / 10.0;
            let q = cctx.offset(rad * a.cos(), rad * a.sin());
            let gq = geodesy::ecef2geodetic(q, &self.w.ell);
            let qctx = Ctx::new(gq.lat, gq.lon, 20.0, &self.w.ell);
            if let Some(tq) = self.point(MODE_NOLAKES, &qctx) {
                rim = rim.min(tq.ground as f64);
            }
        }
        let tc = tc?;
        if self.missing != before {
            return None;
        }
        let ground = tc.ground as f64;
        let v = if tc.water_kind != W_NONE || ground < 1.0 {
            None
        } else if rim < ground - 4.0 {
            None
        } else {
            Some((rim - 0.7).max(ground + 1.0))
        };
        self.c.lakes.insert(key, v);
        v
    }

    /// Level of a lake that must exist (a river's closed basin) (`World::lake_level_forced`).
    pub fn lake_level_forced(&mut self, id: u64, center: DVec3, rad: f64) -> Option<f64> {
        let key = (id, rad.to_bits());
        if let Some(v) = self.c.lakes_forced.get(&key) {
            return *v;
        }
        let before = self.missing;
        let mut v = self.lake_level(id, center, rad);
        if v.is_none() {
            let g = geodesy::ecef2geodetic(center, &self.w.ell);
            let cctx = Ctx::new(g.lat, g.lon, 20.0, &self.w.ell);
            v = self.point(MODE_NOLAKES, &cctx).and_then(|tc| (tc.water_kind == W_NONE && tc.ground > 1.0).then_some(tc.ground as f64 + 1.0));
        }
        if self.missing != before {
            return None;
        }
        self.c.lakes_forced.insert(key, v);
        v
    }
}

impl<'a> Prep<'a> {
    // ------------------------------------------------------------ land-use regions and towns

    /// The field system of land-use region `id` with site `center` (`region_info`).
    pub fn region(&mut self, id: u64, center: DVec3) -> Option<GRegion> {
        if let Some(r) = self.c.regions.get(&id) {
            return Some(*r);
        }
        let cctx = site_ctx(self.w, center, 400.0);
        let tc = self.point(MODE_FULL, &cctx)?;
        let (c, east, north) = (cctx.p, cctx.east, cctx.north);
        let ang = u01k(id, 1) * std::f64::consts::PI;
        let (sa, ca) = ang.sin_cos();
        let ex = east * ca + north * sa;
        let ey = north * ca - east * sa;
        // climate at the region centre decides the field style
        let dry = 1.0 - smoothstep(0.2, 0.4, tc.moist as f64);
        let u = u01k(id, 2);
        let style = if dry > 0.5 && u < 0.6 * dry {
            2
        } else if u < 0.5 {
            0
        } else if u < 0.88 {
            1
        } else {
            3
        };
        let scale = 0.6 + 1.1 * u01k(id, 3);
        let (fw, fh) = match style {
            0 => (220.0 * scale, 220.0 * scale * (1.0 + 2.0 * u01k(id, 4))),
            1 => (300.0 * scale, 0.0),
            2 => (if u01k(id, 4) < 0.5 { 805.0 } else { 402.0 }, 0.0),
            _ => (60.0 + 90.0 * u01k(id, 4), 400.0 + 600.0 * u01k(id, 5)),
        };
        let v = |d: DVec3| [d.x as f32, d.y as f32, d.z as f32, 0.0];
        let r = GRegion {
            center: [c.x, c.y, c.z, 0.0],
            ex: v(ex),
            ey: v(ey),
            east: v(east),
            north: v(north),
            split: u01k(id, 6).to_bits(),
            style,
            _p: 0,
            fw: fw as f32,
            fh: fh as f32,
            hedge: (if u01k(id, 7) < 0.4 { u01k(id, 8) } else { 0.0 }) as f32,
            track: (0.2 + 0.6 * u01k(id, 9)) as f32,
            border_w: (1.5 + 3.0 * u01k(id, 10)) as f32,
            palette: u01k(id, 11) as f32,
            agri: tc.agri,
            season: (tc.style[3] as f64 * 0.7 + 0.3 * u01k(id, 12)).clamp(0.0, 1.0) as f32,
            _q: [0.0; 4],
        };
        self.c.regions.insert(id, r);
        Some(r)
    }

    /// A town site before resolving overlaps (`town_base`); None while its terrain is pending.
    fn town_base(&mut self, id: u64, center: DVec3) -> Option<TownBase> {
        if let Some(t) = self.c.towns_base.get(&id) {
            return Some(*t);
        }
        let lu = &self.w.cfg.landuse;
        let cell = lu.town_cell_km * KM;
        let ctx = site_ctx(self.w, center, 300.0);
        let (east, north) = (ctx.east, ctx.north);
        let near_surface = (center.length() - ctx.p.length()).abs() < 0.8 * cell;
        let exists = if near_surface && u01k(id, 1) < 0.95 {
            let tc = self.point(MODE_FULL, &ctx)?;
            let p_exist = (tc.habit as f64 * 1.1 * lu.towns).min(0.95);
            u01k(id, 1) < p_exist && tc.water_kind == W_NONE && tc.ground > 2.0 && tc.ground < 4000.0
        } else {
            false
        };
        let ang = u01k(id, 2) * std::f64::consts::FRAC_PI_2;
        let (sa, ca) = ang.sin_cos();
        let mut radius = 160.0 * (u01k(id, 3).powf(1.6) * 2.4).exp();
        if u01k(id, 4) < 0.03 {
            radius *= 4.0;
        }
        let elong = 1.0 + 1.6 * u01k(id, 11) * u01k(id, 12);
        let s = &self.w.cfg.satellite;
        let (az, _) = (s.sun_azimuth_deg.to_radians(), ());
        let sun_h = glam::DVec2::new(az.sin(), az.cos());
        let t = TownBase {
            exists,
            center: ctx.p,
            ex: east * ca + north * sa,
            ey: north * ca - east * sa,
            radius: radius.min(cell * 0.45),
            block: 70.0 + 70.0 * u01k(id, 5),
            street: 6.5 + 6.0 * u01k(id, 6),
            organic: u01k(id, 7),
            roof_style: u01k(id, 8),
            height: u01k(id, 9),
            lot: 13.0 + 12.0 * u01k(id, 10),
            elong,
            seed: mix64(id ^ 0x70E5),
            sun: east * sun_h.x + north * sun_h.y,
        };
        self.c.towns_base.insert(id, t);
        Some(t)
    }

    /// A town site with overlaps resolved: of two towns whose footprints would overlap only
    /// the larger exists (`town_info`).
    fn town_info(&mut self, id: u64, center: DVec3) -> Option<TownBase> {
        if let Some(t) = self.c.towns.get(&id) {
            return Some(*t);
        }
        let mut info = self.town_base(id, center)?;
        if info.exists {
            let cell = self.w.cfg.landuse.town_cell_km * KM;
            let extent = |t: &TownBase| 1.6 * t.radius * t.elong.sqrt();
            let k = (center / cell).floor();
            let mut pending = false;
            'search: for dz in -2..=2i64 {
                for dy in -2..=2i64 {
                    for dx in -2..=2i64 {
                        let (nid, nc) = worley3_site(self.w.seed ^ 0x70E1, (k.x as i64 + dx, k.y as i64 + dy, k.z as i64 + dz), cell, 0.8);
                        if nid == id {
                            continue;
                        }
                        let Some(n) = self.town_base(nid, nc) else {
                            pending = true;
                            continue;
                        };
                        if !n.exists || (n.center - info.center).length() > extent(&n) + extent(&info) {
                            continue;
                        }
                        if n.radius > info.radius || (n.radius == info.radius && nid > id) {
                            info.exists = false;
                            break 'search;
                        }
                    }
                }
            }
            if pending {
                return None;
            }
        }
        self.c.towns.insert(id, info);
        Some(info)
    }

    /// The existing towns of the town lattice cells within ±2 cells of `key` (`select_town`'s
    /// candidates); None while pending.
    pub fn town_candidates(&mut self, key: Cell) -> Option<Vec<(u64, GTown)>> {
        if let Some(v) = self.c.town_cands.get(&key) {
            return Some(v.clone());
        }
        let cell = self.w.cfg.landuse.town_cell_km * KM;
        let mut v = Vec::new();
        let mut pending = false;
        for dz in -2..=2i64 {
            for dy in -2..=2i64 {
                for dx in -2..=2i64 {
                    let (id, c) = worley3_site(self.w.seed ^ 0x70E1, (key.0 + dx, key.1 + dy, key.2 + dz), cell, 0.8);
                    match self.town_info(id, c) {
                        Some(t) if t.exists => v.push((id, t.gpu())),
                        Some(_) => {}
                        None => pending = true,
                    }
                }
            }
        }
        if pending {
            return None;
        }
        self.c.town_cands.insert(key, v.clone());
        Some(v)
    }
}

/// A sink lake for the GPU.
pub(crate) fn gsink(id: u64, c: DVec3, rad: f64, level: Option<f64>) -> GSink {
    GSink { c: [c.x, c.y, c.z, 0.0], id, rad: rad as f32, level: level.map_or(NONE_F, |l| l as f32), _p: [0; 2] }
}

/// Context at the surface point below / above an ECEF point.
pub(crate) fn site_ctx(w: &World, pt: DVec3, gsd: f64) -> Ctx {
    let g = geodesy::ecef2geodetic(pt, &w.ell);
    Ctx::new(g.lat, g.lon, gsd, &w.ell)
}

#[allow(dead_code)]
fn geodetic(lat: f64, lon: f64) -> Geodetic {
    Geodetic::new(lat, lon, 0.0)
}
