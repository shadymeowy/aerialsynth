//! Host side of the GPU generator: the parts of the world that are site lists rather than
//! fields (lake levels, sink lakes, land-use regions, towns), kept in caches across batches.
//! Everything they need from the terrain comes from GPU point evaluations (the drainage network
//! itself is on the GPU, `drain.wgsl`): a computation that lacks one records the request and
//! goes on with a stand-in value; its result is then discarded, the requests are evaluated on
//! the GPU in one batch, and the computation runs again (`GpuGenerator::settle`).

use super::types::*;
use crate::noise::*;
use crate::world::{Ctx, Seg, World};
use glam::DVec3;

pub(crate) type Cell = (i64, i64, i64);

const KM: f64 = 1000.0;

/// Key of a point evaluation: mode, surface point, pixel size.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PointKey(u32, u64, u64, u64, u64);

impl PointKey {
    pub(crate) fn new(mode: u32, c: &Ctx) -> Self {
        PointKey(mode, c.p.x.to_bits(), c.p.y.to_bits(), c.p.z.to_bits(), c.gsd.to_bits())
    }
}

/// A point evaluation the GPU is to run (its drainage pieces come from the GPU's drainage
/// network; a full evaluation brings its sink lakes with their levels).
pub(crate) struct PointReq {
    pub ctx: Ctx,
    pub mode: u32,
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
        _r: [0; 4],
    }
}

/// Results of GPU evaluations and what was derived from them (pure functions of the world).
#[derive(Default)]
pub(crate) struct Cache {
    pub points: FxHashMap<PointKey, GTerrain>,
    /// the sink pieces (end point, half width) a full point evaluation keeps, in order
    pub point_sinks: FxHashMap<PointKey, Vec<(DVec3, f64)>>,
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
    /// ecoregions by id
    pub eco: FxHashMap<u64, crate::eco::EcoParams>,
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
        if self.points.len() + self.point_sinks.len() + self.lakes.len() + self.regions.len() + self.towns_base.len() + self.town_cands.len() + self.eco.len() > 2_000_000 {
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
    /// town sites whose base / overlap resolution is still pending in this pass
    towns_pending: FxHashSet<u64>,
    towns_info_pending: FxHashSet<u64>,
    /// point evaluations asked for whose inputs are still to be prepared (`prepare`)
    pending: FxHashMap<PointKey, (Ctx, u32)>,
}

impl<'a> Prep<'a> {
    pub fn new(w: &'a World, c: &'a mut Cache) -> Self {
        Prep {
            w,
            c,
            need: FxHashMap::default(),
            missing: 0,
            towns_pending: FxHashSet::default(),
            towns_info_pending: FxHashSet::default(),
            pending: FxHashMap::default(),
        }
    }

    /// The point evaluation `mode` at `ctx`, if done; else it is requested (its inputs are
    /// prepared by [`Prep::prepare`]).
    pub fn point(&mut self, mode: u32, ctx: &Ctx) -> Option<GTerrain> {
        let key = PointKey::new(mode, ctx);
        if let Some(t) = self.c.points.get(&key) {
            return Some(*t);
        }
        self.missing += 1;
        if !self.need.contains_key(&key) {
            self.pending.entry(key).or_insert((*ctx, mode));
        }
        None
    }

    /// The inputs of the requested point evaluations: the requests whose inputs are complete go
    /// to `need`. A full evaluation needs the levels of its sink lakes (`terrain_impl`), which
    /// follow from the sink pieces it keeps: those are reported by the GPU first.
    pub fn prepare(&mut self) {
        while !self.pending.is_empty() {
            let mut pending: Vec<(PointKey, (Ctx, u32))> = std::mem::take(&mut self.pending).into_iter().collect();
            pending.sort_unstable_by_key(|p| (p.0 .0, p.0 .1, p.0 .2, p.0 .3, p.0 .4));
            for (key, (ctx, mode)) in pending {
                if mode != MODE_FULL || !self.w.cfg.hydro.rivers {
                    let before = self.missing;
                    if mode == MODE_FULL {
                        self.lattice_lakes_at(ctx.p);
                    }
                    if self.missing == before {
                        self.need.insert(key, PointReq { ctx, mode, sinks: vec![] });
                    }
                    continue;
                }
                let Some(pieces) = self.c.point_sinks.get(&key).cloned() else {
                    let rk = PointKey::new(MODE_REPORT, &ctx);
                    self.need.entry(rk).or_insert(PointReq { ctx, mode: MODE_REPORT, sinks: vec![] });
                    continue;
                };
                let segs: Vec<Seg> = pieces.iter().map(|&(b, hw)| Seg { a: b, b, ha: 0.0, hb: 0.0, level: 0, hw, valley: 0.0, hw_b: hw, sink: true }).collect();
                let before = self.missing;
                let mut sinks = Vec::new();
                for (id, sc, rad) in World::sink_lakes(&segs) {
                    if (ctx.p - sc).length() < 1.6 * rad {
                        let level = self.lake_level_forced(id, sc, rad);
                        sinks.push(gsink(id, sc, rad, level));
                    }
                }
                self.lattice_lakes_at(ctx.p);
                if self.missing == before {
                    self.need.insert(key, PointReq { ctx, mode, sinks });
                }
            }
        }
    }

    /// Make sure the levels of the lattice lakes a point within `radius` of `p` may look up are
    /// known (the two nearest lake sites of the point, see `terrain_impl`).
    pub fn lattice_lakes_at(&mut self, p: DVec3) {
        let cfg = &self.w.cfg.hydro;
        if cfg.lake_density <= 0.0 {
            return;
        }
        let cell = cfg.lake_cell_km * KM;
        let sites = worley3_sites(self.w.seed ^ 0x1A4E, p, cell, 0.85);
        for (id, pt) in sites {
            self.lattice_lake(id, pt * cell, p);
        }
    }

    /// The level of lattice lake `id` (site `pt`) if a point at `p` may need it.
    pub fn lattice_lake(&mut self, id: u64, pt: DVec3, p: DVec3) {
        if lattice_lake_wanted(self.w, id, pt, p) {
            self.lattice_lake_level(id, pt);
        }
    }

    /// The level of lattice lake `id` (site `pt`), kept for the GPU's lookups.
    pub fn lattice_lake_level(&mut self, id: u64, pt: DVec3) {
        if self.c.lattice_lakes.contains_key(&id) {
            return;
        }
        let rad = lattice_lake_radius(self.w, id);
        let before = self.missing;
        let level = self.lake_level(id, pt, rad);
        if self.missing == before {
            self.c.lattice_lakes.insert(id, level);
        }
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
        let v = if tc.water_kind != W_NONE || ground < 1.0 || rim < ground - 4.0 { None } else { Some((rim - 0.7).max(ground + 1.0)) };
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

    /// The parameters of ecoregion `id` near `p` (cached; the same as the CPU's).
    fn eco(&mut self, id: u64, p: DVec3) -> Option<crate::eco::EcoParams> {
        if let Some(e) = self.c.eco.get(&id) {
            return Some(*e);
        }
        let reg = crate::registry::Registry::for_config(&self.w.cfg).ok()?;
        let e = crate::eco::params_near(self.w, &reg, id, p)?;
        self.c.eco.insert(id, e);
        Some(e)
    }

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
        // climate at the region centre and its ecoregion's culture decide the field system
        let eco = self.eco(tc.eco_id, c)?;
        let rs = crate::eco::region_style(id, tc.moist as f64, tc.style[3] as f64, &eco.style);
        let (style, fw, fh) = (rs.style as u32, rs.fw, rs.fh);
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
            hedge: rs.hedge as f32,
            track: (0.2 + 0.6 * u01k(id, 9)) as f32,
            border_w: (1.5 + 3.0 * u01k(id, 10)) as f32,
            palette: u01k(id, 11) as f32,
            agri: tc.agri,
            season: rs.season as f32,
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
        if self.towns_pending.contains(&id) {
            self.missing += 1;
            return None;
        }
        let lu = &self.w.cfg.landuse;
        let cell = lu.town_cell_km * KM;
        let ctx = site_ctx(self.w, center, 300.0);
        let (east, north) = (ctx.east, ctx.north);
        let near_surface = (center.length() - ctx.p.length()).abs() < 0.8 * cell;
        let mut culture = (1.0, 1.0, u01k(id, 8), u01k(id, 9));
        let exists = if near_surface && u01k(id, 1) < 0.95 {
            let Some(tc) = self.point(MODE_FULL, &ctx) else {
                self.towns_pending.insert(id);
                return None;
            };
            let eco = self.eco(tc.eco_id, ctx.p)?;
            culture = crate::eco::town_style(id, &eco.style);
            let p_exist = (tc.habit as f64 * 1.1 * lu.towns * culture.0).min(0.95);
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
            block: (70.0 + 70.0 * u01k(id, 5)) * culture.1,
            street: 6.5 + 6.0 * u01k(id, 6),
            organic: u01k(id, 7),
            roof_style: culture.2,
            height: culture.3,
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
        if self.towns_info_pending.contains(&id) {
            self.missing += 1;
            return None;
        }
        let Some(mut info) = self.town_base(id, center) else {
            self.towns_info_pending.insert(id);
            return None;
        };
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
                self.towns_info_pending.insert(id);
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

/// The radius of lattice lake `id`.
fn lattice_lake_radius(w: &World, id: u64) -> f64 {
    let cell = w.cfg.hydro.lake_cell_km * KM;
    (300.0 * (u01k(id, 2) * 2.7).exp()).min(cell * 0.3)
}

/// Can a point at `p` look up the level of lattice lake `id` (site `pt`)? (prob <= density · 1.2)
fn lattice_lake_wanted(w: &World, id: u64, pt: DVec3, p: DVec3) -> bool {
    if u01k(id, 1) > w.cfg.hydro.lake_density * 1.2 {
        return false;
    }
    let pc = pt.normalize() * p.length();
    (p - pc).length() <= lattice_lake_radius(w, id) * 1.5 + 2.0
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
