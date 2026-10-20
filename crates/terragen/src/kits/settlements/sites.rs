//! Infrastructure sites (`docs/design/terrain-next.md` §6.4): airports, ports, dams and
//! reservoirs, quarries and open-pit mines, solar farms, wind farms, gas flares. Placement is host
//! Rust (shared by both backends): settlements from [`super::places`], terrain from point
//! evaluations (`HostCtx::terrain`), drainage from `World::river_segments`; each site becomes a
//! stamp (or segments) whose look is drawn per pixel (`*_px`, WGSL twins `st_*_px`).

use super::places::{self, Place, CITY, METRO, TOWN};
use crate::features::{FSeg, FStamp, HostCtx};
use crate::noise::*;
use crate::surface::{band_cov, mixc, point_light, srgb};
use crate::world::World;
use glam::{DVec2, DVec3};
use std::sync::RwLock;

/// Stamp templates and segment kinds of the kit (its kernel id range).
pub mod tpl {
    pub const PLACE: u32 = 192;
    pub const AIRPORT: u32 = 193;
    pub const PORT: u32 = 194;
    pub const QUARRY: u32 = 195;
    pub const SOLAR: u32 = 196;
    pub const WIND: u32 = 197;
    pub const FLARE: u32 = 198;
    /// segments: a reservoir's upstream drainage piece, a dam wall
    pub const RESERVOIR: u32 = 200;
    pub const DAM: u32 = 201;
}

/// Seed split into two f32-exact 24-bit halves (stamp parameters).
pub fn seed_v(id: u64) -> (f32, f32) {
    ((id & 0xFF_FFFF) as f32, ((id >> 24) & 0xFF_FFFF) as f32)
}

/// The seed of a stamp (`v[2]`, `v[3]`).
pub fn seed_of(v2: f32, v3: f32) -> u64 {
    mix64((v2 as u64) | ((v3 as u64) << 24))
}

/// f32-rounded unit vector (the GPU's stamp axes are f32: the CPU uses the same).
pub fn f32v(d: DVec3) -> DVec3 {
    let v = d.as_vec3();
    DVec3::new(v.x as f64, v.y as f64, v.z as f64)
}

/// Unit axes of a surface point at bearing `a` (radians from east, counter-clockwise).
pub fn axes(w: &World, p: DVec3, a: f64) -> (DVec3, DVec3) {
    let (_, _, east, north) = places::frame_at(w, p);
    let (s, c) = a.sin_cos();
    (f32v(east * c + north * s), f32v(north * c - east * s))
}

fn latlon(w: &World, p: DVec3) -> (f64, f64) {
    let (lat, lon, _, _) = places::frame_at(w, p);
    (lat, lon)
}

/// The point at local offset (de, dn) along the axes, projected to the ellipsoid surface.
fn offset(w: &World, p: DVec3, ex: DVec3, ey: DVec3, u: f64, v: f64) -> DVec3 {
    let q = p + ex * u + ey * v;
    let g = geodesy::ecef2geodetic(q, &w.ell);
    geodesy::geodetic2ecef(geodesy::Geodetic::new(g.lat, g.lon, 0.0), &w.ell)
}

/// Point terrain (None: pending on the GPU).
fn pt(ctx: &mut HostCtx, p: DVec3) -> Option<crate::tile::PointTerrain> {
    let (lat, lon) = latlon(ctx.world, p);
    ctx.terrain(lat, lon, 20.0)
}

/// Point terrain evaluated exactly: the CPU's `World::terrain` (f64), which is also the CPU
/// backend's point source. Site decisions use the backend's samples where they are clear by a
/// margin ([`TIE_H`]) and re-take close ones with exact samples, so both backends choose the
/// same sites (the GPU's f32 samples differ from the CPU's by millimetres to centimetres).
fn pt_exact(w: &World, p: DVec3) -> crate::tile::PointTerrain {
    let (lat, lon) = latlon(w, p);
    let t = w.terrain(&crate::world::Ctx::new(lat, lon, 20.0, &w.ell));
    crate::tile::PointTerrain { ground: t.ground, water: t.water, water_kind: t.water_kind }
}

/// The backend's sample `t` at `p`, or the exact one where its water decision is close.
fn pt_exact_if(w: &World, p: DVec3, t: crate::tile::PointTerrain) -> crate::tile::PointTerrain {
    if water_fragile(&t) {
        pt_exact(w, p)
    } else {
        t
    }
}

/// Heights within this of a decision threshold, or of a competitor, are re-evaluated exactly
/// (m; far above the backends' differences).
pub const TIE_H: f64 = 0.5;

/// Can the backend's sample `t` be on the other side of a water decision than the exact one
/// (ground near sea level or near its lake's level)?
pub(crate) fn water_fragile(t: &crate::tile::PointTerrain) -> bool {
    t.ground.abs() < TIE_H || (t.water_kind != crate::world::water::NONE && (t.water - t.ground).abs() < TIE_H)
}

// ---------------------------------------------------------------- the site cache

#[derive(Clone, Debug, Default)]
pub struct Site {
    pub stamps: Vec<FStamp>,
    pub segs: Vec<FSeg>,
    /// centre and reach (m) of the whole site (area selection)
    pub center: DVec3,
    pub reach: f64,
}

type Key = (u64, u32, u64);
static CACHE: RwLock<Option<FxHashMap<Key, Option<Site>>>> = RwLock::new(None);

/// A memoized site (`kind`, `id`): `f` returns None while point evaluations are pending (not
/// cached), `Some(None)` where there is no site.
fn cached(w: &World, kind: u32, id: u64, f: impl FnOnce() -> Option<Option<Site>>) -> Option<Option<Site>> {
    let key = (w.cache_key, kind, id);
    {
        let g = CACHE.read().unwrap();
        if let Some(v) = g.as_ref().and_then(|m| m.get(&key)) {
            return Some(v.clone());
        }
    }
    let v = f()?;
    let mut g = CACHE.write().unwrap();
    let m = g.get_or_insert_with(FxHashMap::default);
    if m.len() > 200_000 {
        m.clear();
    }
    m.insert(key, v.clone());
    Some(v)
}

// ---------------------------------------------------------------- airports

/// An airport for a place: candidates on 8 bearings at two distances beyond the built-up
/// area, scored by flatness (5 point samples over the runway); runways along the wind.
pub fn airport(ctx: &mut HostCtx, pl: &Place) -> Option<Option<Site>> {
    let w = ctx.world;
    cached(w, tpl::AIRPORT, pl.id, || {
        let id = mix64(pl.id ^ 0xA1F);
        let len = match pl.level {
            METRO => 3500.0 + 500.0 * u01k(id, 1),
            CITY => 2500.0 + 1000.0 * u01k(id, 1),
            _ => 1800.0 + 600.0 * u01k(id, 1),
        };
        let layout = match pl.level {
            METRO => 2u32,
            CITY if pl.population > 1.0e6 => 1,
            _ => 0,
        };
        let spacing = if layout > 0 { 1300.0 + 400.0 * u01k(id, 2) } else { 0.0 };
        let heading = pl.wind[1].atan2(pl.wind[0]) + 0.5 * (u01k(id, 3) - 0.5);
        // a candidate's runway samples: (ok: dry land ≥ 2 m, height range + distance penalty,
        // mean height)
        let score = |ts: &[crate::tile::PointTerrain], dist: f64| {
            let ok = ts.iter().all(|t| t.water_kind == crate::world::water::NONE && t.ground >= 2.0);
            let lo = ts.iter().map(|t| t.ground).fold(f64::MAX, f64::min);
            let hi = ts.iter().map(|t| t.ground).fold(f64::MIN, f64::max);
            (ok, hi - lo + 0.002 * dist, ts.iter().map(|t| t.ground).sum::<f64>() / ts.len() as f64)
        };
        let mut cands: Vec<(f64, u32, DVec3, Vec<DVec3>, f64)> = Vec::new();
        let mut pending = false;
        for k in 0..16u32 {
            let b = (k % 8) as f64 * std::f64::consts::FRAC_PI_4 + 0.4 * u01k(id, 10 + k as u64);
            let dist = pl.radius + if k < 8 { 2500.0 } else { 6000.0 } + 0.5 * len;
            let (ex0, ey0) = axes(w, pl.center, b);
            let c = offset(w, pl.center, ex0, ey0, dist, 0.0);
            let (ex, ey) = axes(w, c, heading);
            let ps: Vec<DVec3> = [(0.0, 0.0), (0.5 * len, 0.0), (-0.5 * len, 0.0), (0.0, 0.5 * spacing + 450.0), (0.0, -0.5 * spacing - 450.0)]
                .iter()
                .map(|&(u, v)| offset(w, c, ex, ey, u, v))
                .collect();
            let ts: Option<Vec<_>> = ps.iter().map(|&q| pt(ctx, q)).collect();
            match ts {
                // (the range of a wet candidate too: its water may be a hair's breadth away)
                Some(ts) => cands.push((score(&ts, dist).1, k, c, ps, dist)),
                None => pending = true,
            }
        }
        if pending {
            return None;
        }
        // the flattest (smallest range, then the first): candidates in the order of their
        // backend range, re-evaluated exactly until the next cannot beat the best exact one
        cands.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut best: Option<(f64, u32, DVec3, f64)> = None;
        for (rg, k, c, ps, dist) in cands {
            if rg > best.as_ref().map_or(30.0, |b| b.0) + TIE_H {
                break;
            }
            let ts: Vec<_> = ps.iter().map(|&q| pt_exact(w, q)).collect();
            let (ok, range, mean) = score(&ts, dist);
            if ok && range < 30.0 && best.as_ref().is_none_or(|b| (range, k) < (b.0, b.1)) {
                best = Some((range, k, c, mean));
            }
        }
        let Some((_, _, c, h)) = best else { return Some(None) };
        let (ex, ey) = axes(w, c, heading);
        let side = if u01k(id, 4) < 0.5 { 1.0 } else { -1.0 };
        let hv = if layout > 0 { 0.5 * spacing + 650.0 } else { 900.0 };
        let (s0, s1) = seed_v(id);
        let center = if layout > 0 { c } else { offset(w, c, ex, ey, 0.0, side * 250.0) };
        let st = FStamp {
            center,
            ex,
            ey,
            half: [(0.5 * len + 450.0) as f32, hv as f32],
            template: tpl::AIRPORT,
            h: h as f32,
            v: [len as f32, layout as f32, s0, s1, spacing as f32, pl.dev as f32, side as f32, (0.9 + 0.5 * u01k(id, 5)) as f32],
        };
        Some(Some(Site { stamps: vec![st], segs: vec![], center, reach: (0.5 * len + 450.0).hypot(hv) }))
    })
}

// ---------------------------------------------------------------- ports

/// A port for a coastal place: the coast found by marching point samples from the centre
/// seaward (the atlas' coast-distance gradient), its direction from two neighbouring marches.
pub fn port(ctx: &mut HostCtx, pl: &Place) -> Option<Option<Site>> {
    let w = ctx.world;
    cached(w, tpl::PORT, pl.id, || {
        let id = mix64(pl.id ^ 0x9027);
        let a = w.atlas();
        let (_, _, east, north) = places::frame_at(w, pl.center);
        let ck = |p: DVec3| a.sample(p.normalize()).coast_km;
        let d = 6000.0;
        let ge = ck(pl.center + east * d) - ck(pl.center - east * d);
        let gn = ck(pl.center + north * d) - ck(pl.center - north * d);
        if ge.abs() + gn.abs() < 1e-6 {
            return Some(None);
        }
        let sea0 = (-gn).atan2(-ge);
        // the nearest sea along 8 bearings (the atlas' gradient first), then the coast at two
        // lateral offsets along that bearing (the coast direction)
        let step = 400.0;
        let n = ((pl.radius + 12_000.0) / step) as u32;
        // every sample of a ray is requested at once (one GPU round per phase, not per sample)
        let ray = |ctx: &mut HostCtx, bearing: f64, off: f64| -> Option<Option<(u32, DVec3)>> {
            let (ex, ey) = axes(w, pl.center, bearing);
            let start = offset(w, pl.center, ex, ey, 0.0, off);
            let ts: Vec<Option<crate::tile::PointTerrain>> = (0..=n).map(|i| pt(ctx, offset(w, start, ex, ey, i as f64 * step, 0.0))).collect();
            if ts.iter().any(|t| t.is_none()) {
                return None;
            }
            // (samples whose water kind could differ between the backends: exactly)
            let ts: Vec<crate::tile::PointTerrain> = ts
                .into_iter()
                .flatten()
                .enumerate()
                .map(|(i, t)| if water_fragile(&t) { pt_exact(w, offset(w, start, ex, ey, i as f64 * step, 0.0)) } else { t })
                .collect();
            let mut prev: Option<f64> = None;
            for (i, t) in ts.iter().enumerate() {
                let u = i as f64 * step;
                if t.water_kind == crate::world::water::OCEAN {
                    return Some(prev.map(|pu| (i as u32, offset(w, start, ex, ey, 0.5 * (pu + u), 0.0))));
                }
                // (rivers are crossed: ports sit at river mouths)
                if t.water_kind == crate::world::water::LAKE {
                    return Some(None);
                }
                if t.water_kind == crate::world::water::NONE {
                    prev = Some(u);
                }
            }
            Some(None)
        };
        let mut best: Option<(u32, f64)> = None;
        let mut pending = false;
        for k in 0..8 {
            let b = sea0 + k as f64 * std::f64::consts::FRAC_PI_4;
            match ray(ctx, b, 0.0) {
                None => pending = true,
                Some(Some((i, _))) => {
                    if best.is_none_or(|x| i < x.0) {
                        best = Some((i, b));
                    }
                }
                Some(None) => {}
            }
        }
        if pending {
            return None;
        }
        let Some((_, sea)) = best else { return Some(None) };
        let mut coast = [DVec3::ZERO; 3];
        for (j, off) in [-500.0, 0.0, 500.0].iter().enumerate() {
            match ray(ctx, sea, *off) {
                None => pending = true,
                Some(Some((_, c))) => coast[j] = c,
                Some(None) => return Some(None),
            }
        }
        if pending {
            return None;
        }
        let along = (coast[2] - coast[0]).normalize_or_zero();
        let up = coast[1].normalize();
        let mut ey = f32v(up.cross(along).normalize_or_zero());
        let (_, sy) = axes(w, coast[1], sea);
        if ey.dot(sy) < 0.0 {
            ey = -ey;
        }
        let ex = f32v(ey.cross(up).normalize_or_zero());
        let lvl = (pl.level as f64 - 2.0).max(0.0);
        let len = 700.0 + 900.0 * lvl + 900.0 * u01k(id, 1);
        let pier = 220.0 + 120.0 * lvl + 80.0 * u01k(id, 2);
        let land = 320.0 + 120.0 * lvl;
        let (s0, s1) = seed_v(id);
        let st = FStamp {
            center: coast[1],
            ex,
            ey,
            half: [(0.5 * len + 250.0) as f32, (land.max(pier + 220.0)) as f32],
            template: tpl::PORT,
            h: 0.0,
            v: [len as f32, land as f32, s0, s1, pier as f32, pl.dev as f32, 0.0, 0.0],
        };
        Some(Some(Site { center: coast[1], reach: (0.5 * len + 250.0).hypot(land.max(pier + 220.0)), stamps: vec![st], segs: vec![] }))
    })
}

// ---------------------------------------------------------------- dams and reservoirs

/// A dam on a level 0–1 river near a place (or a mountain lattice site): the narrowest valley
/// cross-section among candidate pieces (point samples), the crest below the lower bank; the
/// reservoir is the drainage upstream (≤ 40 pieces, same level) with the water level.
pub fn dam(ctx: &mut HostCtx, key: u64, center: DVec3, r_min: f64, r_max: f64) -> Option<Option<Site>> {
    let w = ctx.world;
    cached(w, tpl::DAM, key, || {
        let segs: Vec<_> = w.river_segments(center, r_max, 800.0).into_iter().filter(|s| s.level <= 1 && !s.sink).collect();
        if segs.is_empty() {
            return Some(None);
        }
        // candidate pieces by hash order
        let mut cand: Vec<(u64, usize)> = segs
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                let m = 0.5 * (s.a + s.b);
                let d = (m - center).length();
                d > r_min && d < r_max && s.ha > 5.0 && (s.b - s.a).length() > 300.0
            })
            .map(|(i, s)| (hash3(key, (s.a.x * 0.01) as i64, (s.a.y * 0.01) as i64, (s.a.z * 0.01) as i64), i))
            .collect();
        cand.sort_unstable();
        cand.truncate(6);
        // a candidate's cross-section from its samples (floor, then the banks at increasing
        // distances on both sides: the first that rises 60 m): (score, floor, height; None:
        // too low or at sea) and whether the samples are near a decision's threshold
        const BANKS: [f64; 5] = [200.0, 400.0, 700.0, 1100.0, 1600.0];
        let section = |tm: &crate::tile::PointTerrain, banks: &[crate::tile::PointTerrain]| {
            let floor = tm.ground;
            let mut side = [(0.0, 0.0); 2];
            let mut fragile = water_fragile(tm);
            for k in 0..2 {
                for (j, dd) in BANKS.iter().enumerate() {
                    let g = banks[k * 5 + j].ground;
                    side[k] = (*dd, g - floor);
                    fragile |= (g - floor - 60.0).abs() < TIE_H;
                    if g - floor > 60.0 {
                        break;
                    }
                }
            }
            let hgt = side[0].1.min(side[1].1) - 4.0;
            let width = side[0].0 + side[1].0;
            fragile |= (hgt - 25.0).abs() < TIE_H;
            let v = (hgt >= 25.0 && tm.water_kind != crate::world::water::OCEAN).then(|| (hgt.min(150.0) / width, floor, hgt.min(150.0)));
            (v, fragile)
        };
        let mut evals: Vec<(f64, bool, usize, Vec<DVec3>)> = Vec::new();
        let mut pending = false;
        for (ci, &(_, i)) in cand.iter().enumerate() {
            let s = &segs[i];
            let m = 0.5 * (s.a + s.b);
            let up = m.normalize();
            let dir = (s.b - s.a).normalize();
            let n = f32v(up.cross(dir).normalize());
            // (all requested at once)
            let mut ps = vec![m];
            ps.extend([1.0, -1.0].iter().flat_map(|sg| BANKS.map(|dd| offset(w, m, n, dir, sg * dd, 0.0))));
            let ts: Option<Vec<_>> = ps.iter().map(|&q| pt(ctx, q)).collect();
            let Some(ts) = ts else {
                pending = true;
                continue;
            };
            let (v, fragile) = section(&ts[0], &ts[1..]);
            evals.push((v.map_or(f64::NEG_INFINITY, |v| v.0), fragile, ci, ps));
        }
        if pending {
            return None;
        }
        // the best score (then the first candidate): candidates near a threshold first, then by
        // their backend score, re-evaluated exactly until the next cannot beat the best exact
        // one (a height error moves a score by at most TIE_H / 400 m)
        evals.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.total_cmp(&a.0)).then(a.2.cmp(&b.2)));
        let mut best: Option<(f64, usize, f64, f64)> = None;
        for (sc, fragile, ci, ps) in evals {
            if !fragile && best.as_ref().is_some_and(|b| sc < b.0 - TIE_H / 400.0) {
                break;
            }
            let ts: Vec<_> = ps.iter().map(|&q| pt_exact(w, q)).collect();
            if let (Some((score, floor, hgt)), _) = section(&ts[0], &ts[1..]) {
                if best.as_ref().is_none_or(|b| score > b.0 || (score == b.0 && ci < b.1)) {
                    best = Some((score, ci, floor, hgt));
                }
            }
        }
        let best = best.map(|(sc, ci, floor, hgt)| (sc, cand[ci].1, floor, hgt, 0.0));
        let Some((_, i0, floor, hgt, _)) = best else { return Some(None) };
        let crest = floor + hgt;
        let level = crest - 3.0 - 0.06 * hgt;
        let s0 = &segs[i0];
        let m = 0.5 * (s0.a + s0.b);
        let dir = (s0.b - s0.a).normalize();
        let n = f32v(m.normalize().cross(dir).normalize());
        // the upstream pieces (same level, ending where a piece of the set starts)
        let mut set = vec![(s0.a, m)];
        let mut todo = vec![s0.a];
        let lvl = s0.level;
        while let Some(a) = todo.pop() {
            if set.len() >= 40 {
                break;
            }
            for s in &segs {
                if s.level == lvl && (s.b - a).length() < 1.0 && (s.a - m).length() < 25_000.0 && !set.iter().any(|x| (x.0 - s.a).length() < 1.0) {
                    set.push((s.a, s.b));
                    todo.push(s.a);
                }
            }
        }
        let reach = (35.0 * hgt).clamp(600.0, 4000.0);
        // the dam point and the downstream direction in each piece's frame
        let mut out = Site { center: m, reach: 0.0, ..Default::default() };
        let (dk0, dk1) = seed_v(key);
        for &(a, b) in &set {
            let ab = b - a;
            let len = ab.length().max(1.0);
            let dx = ab / len;
            let left = f32v(a.normalize().cross(dx).normalize());
            let dm = m - a;
            let (da, dl) = (dm.dot(dx), dm.dot(left));
            let nd = dir.dot(left).atan2(dir.dot(dx));
            out.segs.push(FSeg {
                a,
                b,
                ha: level as f32,
                hb: crest as f32,
                kind: tpl::RESERVOIR,
                class: crate::landcover::RESERVOIR,
                hw: reach as f32,
                s0: 0.0,
                deck_a: f32::NAN,
                deck_b: f32::NAN,
                flags: 0,
                v: [da as f32, dl as f32, nd as f32, dk0],
            });
            out.reach = out.reach.max((a - m).length() + reach).max((b - m).length() + reach);
        }
        // a reservoir never drowns a town or a city (hamlets and villages under its water are
        // drowned: not drawn, see `drowned`)
        for l in TOWN as usize..places::NLEVELS {
            for pl in places::places_near(w, l, m, out.reach) {
                if out.segs.iter().any(|sg| reservoir_at(sg, pl.center, f64::NEG_INFINITY, 0.1).0 > 0.5) && pt_exact(w, pl.center).ground < level - 1.0 {
                    return Some(None);
                }
            }
        }
        // the wall across the valley: as wide as the cross-section at the crest
        let wl = side_width(ctx, m, n, dir, floor + hgt, 1.0);
        let wr = side_width(ctx, m, n, dir, floor + hgt, -1.0);
        let (Some(wl), Some(wr)) = (wl, wr) else { return None };
        out.segs.push(FSeg {
            a: offset(w, m, n, dir, -wr, 0.0),
            b: offset(w, m, n, dir, wl, 0.0),
            ha: crest as f32,
            hb: crest as f32,
            kind: tpl::DAM,
            class: crate::landcover::DAM,
            hw: (6.0 + 0.04 * hgt) as f32,
            s0: 0.0,
            deck_a: f32::NAN,
            deck_b: f32::NAN,
            flags: 0,
            v: [hgt as f32, dk1, 0.0, 0.0],
        });
        Some(Some(out))
    })
}

/// Distance across the valley from `m` (side `sg`) where the ground reaches `crest`.
fn side_width(ctx: &mut HostCtx, m: DVec3, n: DVec3, dir: DVec3, crest: f64, sg: f64) -> Option<f64> {
    let w = ctx.world;
    const D: [f64; 7] = [100.0, 200.0, 350.0, 550.0, 800.0, 1100.0, 1600.0];
    // (exact samples: the crest threshold decides the wall's ends)
    let ts: Vec<Option<crate::tile::PointTerrain>> = D.iter().map(|dd| Some(pt_exact(w, offset(w, m, n, dir, sg * dd, 0.0)))).collect();
    let mut last = 0.0;
    for (dd, t) in D.iter().zip(ts) {
        if t?.ground >= crest {
            return Some(0.5 * (last + dd) + 30.0);
        }
        last = *dd;
    }
    Some(last + 30.0)
}

// ---------------------------------------------------------------- lattice sites (quarries, wind, flares)

/// The candidate of a lattice cell of `cell` m (seed `key`): its id and surface point.
pub(crate) fn lattice_site(w: &World, key: u64, cell: f64, c: (i64, i64, i64)) -> Option<(u64, DVec3)> {
    let h = hash3(w.seed() ^ key, c.0, c.1, c.2);
    let pt = DVec3::new(c.0 as f64 + 0.5 + 0.8 * (u01k(h, 1) - 0.5), c.1 as f64 + 0.5 + 0.8 * (u01k(h, 2) - 0.5), c.2 as f64 + 0.5 + 0.8 * (u01k(h, 3) - 0.5))
        * cell;
    let g = geodesy::ecef2geodetic(pt, &w.ell);
    let s = geodesy::geodetic2ecef(geodesy::Geodetic::new(g.lat, g.lon, 0.0), &w.ell);
    if (pt - s).length() >= 0.5 * cell {
        return None;
    }
    Some((h, s))
}

/// The lattice cells of `cell` whose sites can lie within `r` of `p`.
pub(crate) fn lattice_cells(w: &World, cell: f64, p: DVec3, r: f64) -> Vec<(i64, i64, i64)> {
    let rr = r + 0.5 * cell;
    let lo = ((p - DVec3::splat(rr)) / cell).floor();
    let hi = ((p + DVec3::splat(rr)) / cell).floor();
    let mut v = Vec::new();
    for z in lo.z as i64..=hi.z as i64 {
        for y in lo.y as i64..=hi.y as i64 {
            for x in lo.x as i64..=hi.x as i64 {
                let cc = (DVec3::new(x as f64, y as f64, z as f64) + 0.5) * cell;
                if (cc.length() - w.ell.a).abs() > 1.4 * cell || (cc - p).length() > rr + 0.87 * cell {
                    continue;
                }
                v.push((x, y, z));
            }
        }
    }
    v
}

pub const QUARRY_CELL: f64 = 16_000.0;
pub const WIND_CELL: f64 = 45_000.0;
pub const FLARE_CELL: f64 = 30_000.0;
pub const MTN_DAM_CELL: f64 = 55_000.0;

/// Is `p` within `margin` m of a settlement's built-up area (towns and larger)?
fn near_town(w: &World, p: DVec3, margin: f64) -> bool {
    for l in TOWN as usize..places::NLEVELS {
        for pl in places::places_near(w, l, p, places::LEVELS[l].radius[1] * 1.7 + margin) {
            if (pl.center - p).length() < pl.radius * 1.3 + margin {
                return true;
            }
        }
    }
    false
}

/// A quarry or open-pit mine at a lattice cell: near towns (or in the hills), by lithology.
pub fn quarry(ctx: &mut HostCtx, c: (i64, i64, i64)) -> Option<Option<Site>> {
    let w = ctx.world;
    let Some((id, p)) = lattice_site(w, 0x0A77, QUARRY_CELL, c) else { return Some(None) };
    cached(w, tpl::QUARRY, id, || {
        let a = w.atlas().sample(p.normalize());
        if a.coast_km < 2.0 || a.glaciation > 0.5 || w.cfg.landuse.towns <= 0.0 {
            return Some(None);
        }
        let (h0, amp) = w.smooth_relief(p, 2000.0);
        if h0 < 5.0 {
            return Some(None);
        }
        let near = !places::places_near(w, TOWN as usize, p, 22_000.0).is_empty();
        let hills = smoothstep(80.0, 600.0, amp);
        let prob = if near { 0.32 } else { 0.08 * hills } * (0.4 + 0.6 * a.development.max(0.3));
        if u01k(id, 1) >= prob || near_town(w, p, 1500.0) {
            return Some(None);
        }
        use crate::atlas::Lithology as L;
        let hard = matches!(a.litho, L::Crystalline | L::Volcanic);
        let open_pit = hard && u01k(id, 2) < 0.3;
        let r = if open_pit { 600.0 + 1400.0 * u01k(id, 3) } else { 140.0 + 320.0 * u01k(id, 3) };
        let depth = if open_pit { 120.0 + 300.0 * u01k(id, 4) } else { 25.0 + 50.0 * u01k(id, 4) };
        // (exact: the stamp keeps its height, the same on both backends)
        let t = pt_exact(w, p);
        if t.water_kind != crate::world::water::NONE {
            return Some(None);
        }
        let kind = if open_pit {
            1.0
        } else if matches!(a.litho, L::Carbonate) {
            0.0
        } else if hard {
            2.0
        } else {
            3.0
        };
        let tail_r = if open_pit { 0.5 * r + 300.0 * u01k(id, 6) } else { 0.0 };
        let (ex, ey) = axes(w, p, u01k(id, 5) * std::f64::consts::TAU);
        let (s0, s1) = seed_v(id);
        let ext = 1.3 * r + 2.0 * tail_r + 300.0;
        let st = FStamp {
            center: p,
            ex,
            ey,
            half: [ext as f32, ext as f32],
            template: tpl::QUARRY,
            h: t.ground as f32,
            v: [r as f32, depth as f32, s0, s1, kind as f32, if open_pit { 15.0 } else { 10.0 }, (u01k(id, 7) * std::f64::consts::TAU) as f32, tail_r as f32],
        };
        Some(Some(Site { center: p, reach: ext * std::f64::consts::SQRT_2, stamps: vec![st], segs: vec![] }))
    })
}

/// A wind farm at a lattice cell: windy, developed plains and ridges, away from towns.
pub fn wind(ctx: &mut HostCtx, c: (i64, i64, i64)) -> Option<Option<Site>> {
    let w = ctx.world;
    let Some((id, p)) = lattice_site(w, 0x3D1, WIND_CELL, c) else { return Some(None) };
    cached(w, tpl::WIND, id, || {
        let a = w.atlas();
        let s = a.sample(p.normalize());
        let speed = s.wind_e.hypot(s.wind_n);
        let arch_w = [0.3, 0.3, 0.4, 0.5, 1.0, 0.7, 1.2, 1.4, 0.5, 0.7, 0.3, 0.6][(s.archetype as usize).min(11)];
        let prob = 0.55 * smoothstep(3.5, 7.0, speed) * smoothstep(0.3, 0.6, s.development) * arch_w;
        if s.coast_km < 1.0 || s.glaciation > 0.3 || s.temp_c < -6.0 || u01k(id, 1) >= prob || w.cfg.landuse.towns <= 0.0 {
            return Some(None);
        }
        let (h0, amp) = w.smooth_relief(p, 3000.0);
        if h0 < 2.0 || amp > 700.0 || near_town(w, p, 4000.0) {
            return Some(None);
        }
        let t = pt_exact_if(w, p, pt(ctx, p)?);
        if t.water_kind != crate::world::water::NONE {
            return Some(None);
        }
        let across = s.wind_n.atan2(s.wind_e) + std::f64::consts::FRAC_PI_2;
        let (ex, ey) = axes(w, p, across);
        let hx = 2500.0 + 3000.0 * u01k(id, 2);
        let hy = 1500.0 + 2000.0 * u01k(id, 3);
        let (s0, s1) = seed_v(id);
        let st = FStamp {
            center: p,
            ex,
            ey,
            half: [hx as f32, hy as f32],
            template: tpl::WIND,
            h: 0.0,
            v: [
                (420.0 + 220.0 * u01k(id, 4)) as f32,
                (80.0 + 45.0 * u01k(id, 5)) as f32,
                s0,
                s1,
                (40.0 + 25.0 * u01k(id, 6)) as f32,
                (0.55 + 0.35 * u01k(id, 7)) as f32,
                0.0,
                0.0,
            ],
        };
        Some(Some(Site { center: p, reach: hx.hypot(hy), stamps: vec![st], segs: vec![] }))
    })
}

/// Gas flares at a lattice cell: arid sedimentary basins (oil fields).
pub fn flare(ctx: &mut HostCtx, c: (i64, i64, i64)) -> Option<Option<Site>> {
    let w = ctx.world;
    let Some((id, p)) = lattice_site(w, 0xF1A, FLARE_CELL, c) else { return Some(None) };
    cached(w, tpl::FLARE, id, || {
        let s = w.atlas().sample(p.normalize());
        use crate::atlas::Lithology as L;
        let basin = matches!(s.litho, L::Sedimentary | L::Unconsolidated | L::Carbonate);
        if !basin || s.precip_mm > 380.0 || s.temp_c < 4.0 || s.coast_km < 1.0 || u01k(id, 1) >= 0.32 || w.cfg.landuse.towns <= 0.0 {
            return Some(None);
        }
        let (h0, amp) = w.smooth_relief(p, 3000.0);
        if h0 < 2.0 || amp > 500.0 || near_town(w, p, 2000.0) {
            return Some(None);
        }
        let t = pt_exact_if(w, p, pt(ctx, p)?);
        if t.water_kind != crate::world::water::NONE {
            return Some(None);
        }
        let (ex, ey) = axes(w, p, u01k(id, 2) * std::f64::consts::TAU);
        let (s0, s1) = seed_v(id);
        let st = FStamp {
            center: p,
            ex,
            ey,
            half: [1600.0, 1600.0],
            template: tpl::FLARE,
            h: 0.0,
            v: [(2.0 + 3.0 * u01k(id, 3)).floor() as f32, 0.0, s0, s1, 0.0, 0.0, 0.0, 0.0],
        };
        Some(Some(Site { center: p, reach: 1600.0 * std::f64::consts::SQRT_2, stamps: vec![st], segs: vec![] }))
    })
}

/// Solar farms of a town (sunny, dry, flat, developed): up to two, beyond its edge.
pub fn solar(ctx: &mut HostCtx, pl: &Place, k: u32) -> Option<Option<Site>> {
    let w = ctx.world;
    let id = mix64(pl.id ^ 0x501A ^ ((k as u64) << 40));
    cached(w, tpl::SOLAR, id, || {
        let sunny = (1.0 - smoothstep(500.0, 1100.0, pl.precip)) * (1.0 - smoothstep(38.0, 52.0, pl.lat.to_degrees().abs()));
        let prob = 0.75 * sunny * smoothstep(0.25, 0.5, pl.dev) * if pl.level >= CITY { 1.0 } else { 0.6 };
        if u01k(id, 1) >= prob {
            return Some(None);
        }
        let b = u01k(id, 2) * std::f64::consts::TAU;
        let desert = 1.0 - smoothstep(150.0, 400.0, pl.precip);
        let hx = 220.0 + 500.0 * u01k(id, 3) + 700.0 * desert;
        let hy = 160.0 + 300.0 * u01k(id, 4) + 400.0 * desert;
        let dist = pl.radius * 1.2 + 800.0 + 2500.0 * u01k(id, 5) + hx;
        let (ex0, ey0) = axes(w, pl.center, b);
        let c = offset(w, pl.center, ex0, ey0, dist, 0.0);
        let (ex, ey) = axes(w, c, 0.0);
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        let ts: Vec<Option<crate::tile::PointTerrain>> =
            [(0.0, 0.0), (hx, hy), (-hx, hy), (hx, -hy), (-hx, -hy)].iter().map(|&(u, v)| pt(ctx, offset(w, c, ex, ey, u, v))).collect();
        let ts: Vec<crate::tile::PointTerrain> = ts.into_iter().collect::<Option<_>>()?;
        let mut fragile = false;
        for t in &ts {
            fragile |= water_fragile(t);
            lo = lo.min(t.ground);
            hi = hi.max(t.ground);
        }
        let lim = 12.0 + 0.004 * hx;
        // (close to a decision: exactly)
        let ts = if fragile || (hi - lo - lim).abs() < 2.0 * TIE_H {
            lo = f64::MAX;
            hi = f64::MIN;
            let ps = [(0.0, 0.0), (hx, hy), (-hx, hy), (hx, -hy), (-hx, -hy)].map(|(u, v)| offset(w, c, ex, ey, u, v));
            let ts: Vec<_> = ps.iter().map(|&q| pt_exact(w, q)).collect();
            for t in &ts {
                lo = lo.min(t.ground);
                hi = hi.max(t.ground);
            }
            ts
        } else {
            ts
        };
        if ts.iter().any(|t| t.water_kind != crate::world::water::NONE) || hi - lo > lim {
            return Some(None);
        }
        let (s0, s1) = seed_v(id);
        let st = FStamp {
            center: c,
            ex,
            ey,
            half: [hx as f32, hy as f32],
            template: tpl::SOLAR,
            h: 0.0,
            v: [0.0, 0.0, s0, s1, (6.5 + 3.0 * u01k(id, 6)) as f32, (3.6 + 1.0 * u01k(id, 7)) as f32, 0.0, 0.0],
        };
        Some(Some(Site { center: c, reach: hx.hypot(hy), stamps: vec![st], segs: vec![] }))
    })
}

/// The sites whose reach meets the area (`gsd` its pixel size): their stamps and segments.
pub fn sites_near(ctx: &mut HostCtx, center: DVec3, radius: f64, gsd: f64, out: &mut crate::features::Features) {
    let w = ctx.world;
    let push = |s: Option<Option<Site>>, out: &mut crate::features::Features| {
        if let Some(Some(s)) = s {
            if (s.center - center).length() <= radius + s.reach {
                out.stamps.extend(s.stamps);
                out.segs.extend(s.segs);
            }
        }
    };
    // place-bound sites: airports, ports, solar farms, dams
    // (only at the pixel sizes any of them is drawn at, see below and `dams_near`: a low-zoom
    // tile's disc holds the towns of a continent, each resolved against its neighbours)
    let mut towns: Vec<Place> = Vec::new();
    if gsd < DAMS_GSD_MAX {
        for l in TOWN as usize..places::NLEVELS {
            towns.extend(places::places_near(w, l, center, radius + 32_000.0));
        }
    }
    for pl in &towns {
        let d = (pl.center - center).length();
        if pl.airport && gsd < 600.0 && d < radius + pl.radius + 14_000.0 {
            push(airport(ctx, pl), out);
        }
        if pl.port && pl.parent == 0 && gsd < 600.0 && d < radius + pl.radius + 12_000.0 {
            push(port(ctx, pl), out);
        }
        if gsd < 300.0 && d < radius + pl.radius + 6_000.0 {
            for k in 0..2 {
                push(solar(ctx, pl, k), out);
            }
        }
    }
    for s in dams_near(ctx, center, radius, gsd, &towns) {
        push(Some(Some(s)), out);
    }
    if gsd < 300.0 {
        for c in lattice_cells(w, QUARRY_CELL, center, radius + 3_500.0) {
            push(quarry(ctx, c), out);
        }
    }
    if gsd < 150.0 {
        for c in lattice_cells(w, WIND_CELL, center, radius + 6_500.0) {
            push(wind(ctx, c), out);
        }
    }
    if gsd < 2500.0 {
        for c in lattice_cells(w, FLARE_CELL, center, radius + 2_300.0) {
            push(flare(ctx, c), out);
        }
    }
}

/// The largest pixel size (m) dams are drawn at (the other place-bound sites, airports, ports
/// and solar farms, only at finer ones).
pub const DAMS_GSD_MAX: f64 = 1200.0;

/// The dams and reservoirs whose reach meets the area (`towns`: the places of levels ≥ town
/// within its radius + 32 km).
pub fn dams_near(ctx: &mut HostCtx, center: DVec3, radius: f64, gsd: f64, towns: &[Place]) -> Vec<Site> {
    let w = ctx.world;
    let mut out = Vec::new();
    if gsd >= DAMS_GSD_MAX {
        return out;
    }
    let mut push = |s: Option<Option<Site>>| {
        if let Some(Some(s)) = s {
            if (s.center - center).length() <= radius + s.reach {
                out.push(s);
            }
        }
    };
    for pl in towns {
        let d = (pl.center - center).length();
        if pl.parent == 0 && d < radius + 30_000.0 && u01k(pl.id, 50) < if pl.level >= CITY { 0.5 } else { 0.15 } {
            push(dam(ctx, mix64(pl.id ^ 0xDA4), pl.center, pl.radius + 2500.0, 22_000.0));
        }
    }
    // mountain dams
    for c in lattice_cells(w, MTN_DAM_CELL, center, radius + 30_000.0) {
        if let Some((id, p)) = lattice_site(w, 0xDA7, MTN_DAM_CELL, c) {
            if u01k(id, 1) < 0.2 && w.smooth_relief(p, 3000.0).1 > 350.0 && w.atlas().sample(p.normalize()).glaciation < 0.4 {
                push(dam(ctx, id, p, 0.0, 20_000.0));
            }
        }
    }
    out
}

/// Is the place centre `p` drowned by one of the reservoir pieces `res` (its ground > 1 m below
/// the water level)? None while a point evaluation is pending.
pub fn drowned(ctx: &mut HostCtx, res: &[FSeg], p: DVec3) -> Option<bool> {
    let mut level = f64::NEG_INFINITY;
    for sg in res.iter().filter(|sg| sg.kind == tpl::RESERVOIR) {
        if reservoir_at(sg, p, f64::NEG_INFINITY, 0.1).0 > 0.5 {
            level = level.max(sg.ha as f64);
        }
    }
    if level == f64::NEG_INFINITY {
        return Some(false);
    }
    let g = pt(ctx, p)?.ground;
    // (near the level: exactly, the same on both backends)
    Some(if (g - (level - 1.0)).abs() < TIE_H { pt_exact(ctx.world, p).ground } else { g } < level - 1.0)
}

/// Clear the site cache (tests).
#[cfg(test)]
pub fn clear_cache() {
    *CACHE.write().unwrap() = None;
}

// ================================================================ pixel looks

/// A site's look at a sample (`Layer` fields; `cov` 0: nothing).
#[derive(Clone, Copy, Debug, Default)]
pub struct SitePx {
    pub cov: f64,
    pub col: DVec3,
    /// height: `abs` absolute (m), else above ground (`dh`)
    pub dh: f64,
    pub abs: bool,
    pub cls: u8,
    pub emit: DVec3,
    pub clear: f64,
    /// cast shadow on what lies below (0..1)
    pub shade: f64,
}

#[inline]
fn aa(d: f64, fw: f64) -> f64 {
    (d / fw + 0.5).clamp(0.0, 1.0)
}

#[inline]
fn boxd(r: DVec2, c: DVec2, half: DVec2) -> f64 {
    (half - (r - c).abs()).min_element()
}

/// Stamp-local coordinates of `p`.
pub fn uv(st: &FStamp, p: DVec3) -> DVec2 {
    let d = p - st.center;
    DVec2::new(d.dot(st.ex), d.dot(st.ey))
}

/// An airport: runways (markings, thresholds), taxiways, apron, terminal, hangars, car park,
/// mown grass; the graded area flattened to the plane `h` (cut and fill margin); lights.
pub fn airport_px(st: &FStamp, p: DVec3, ground: f64, gsd: f64, fw: f64) -> SitePx {
    let q = uv(st, p);
    let (hx, hy) = (st.half[0] as f64, st.half[1] as f64);
    let len = st.v[0] as f64;
    let layout = st.v[1] as u32;
    let seed = seed_of(st.v[2], st.v[3]);
    let spacing = st.v[4] as f64;
    let side = st.v[6] as f64;
    // (round 13: a 220 m cut-and-fill margin, the grass beyond the perimeter road fading into
    // the land along a wavy edge, the grass toned over ~170 m: the airport sat on a flat green
    // pad with hard straight edges, gallery g33)
    let margin = 220.0;
    let din = (hx - q.x.abs()).min(hy - q.y.abs());
    if din <= 0.0 {
        return SitePx::default();
    }
    let wn = perlin3(seed ^ 0xED1, p / 170.0);
    let edge = smoothstep(0.0, 0.8 * margin, din + 65.0 * wn);
    let mut o = SitePx { cov: edge, abs: true, clear: edge, cls: crate::landcover::GRASS, ..Default::default() };
    // grass, mown in stripes by patches of ~260 m (each its own width, direction and contrast;
    // a fifth unmown, rough; round 14: one stripe pattern over the whole field read as a
    // painted pad, gallery g33)
    let mc = (q / 260.0).floor();
    let mh = hash2(seed ^ 0x30A, mc.x as i64, mc.y as i64);
    let sw = 9.0 + 12.0 * u01k(mh, 1);
    let mq = if u01k(mh, 2) < 0.65 { q.x } else { q.y };
    let stripe = ((mq / sw).floor() as i64).rem_euclid(2) as f64;
    let mown = if u01k(mh, 3) < 0.8 { 0.02 + 0.05 * u01k(mh, 4) } else { 0.0 };
    let tone = 1.0 + 0.12 * wn * band(170.0, gsd) + 0.06 * (u01k(mh, 5) - 0.5) * band(260.0, gsd);
    o.col = srgb(132.0, 146.0, 92.0) * (0.97 + mown * (stripe - 0.5) * band(sw, gsd)) * tone;
    // graded plane, cut-and-fill margin to the ground
    let wgt = smoothstep(0.0, margin, din);
    o.dh = ground + (st.h as f64 - ground) * wgt;
    let wr = if layout == 2 { 30.0 } else { 22.5 };
    // runways: 0 at v = +spacing/2 (or 0), 1 at v = −spacing/2
    let yr0 = if layout > 0 { 0.5 * spacing } else { -side * 250.0 };
    let mut rw = 0.0;
    let mut rx = 0.0;
    let mut ry = 0.0;
    for k in 0..2u32 {
        if k == 0 || layout > 0 {
            let y = if k == 0 { yr0 } else { -0.5 * spacing };
            let c = aa(boxd(q, DVec2::new(0.0, y), DVec2::new(0.5 * len, wr)), fw);
            if c > rw {
                rw = c;
                rx = q.x;
                ry = q.y - y;
            }
        }
    }
    // the cross runway (large airports)
    if layout == 2 {
        let a = st.v[7] as f64;
        let (s, c) = a.sin_cos();
        let cq = q - DVec2::new(0.15 * len, 0.0);
        let r = DVec2::new(c * cq.x + s * cq.y, -s * cq.x + c * cq.y);
        let lc = (0.6 * len).min(2.0 * hy - 300.0);
        let cc = aa(boxd(r, DVec2::ZERO, DVec2::new(0.5 * lc, 22.5)), fw);
        if cc > rw {
            rw = cc;
            rx = r.x;
            ry = r.y;
        }
    }
    // taxiways parallel to each runway (towards the apron), connectors every 600 m
    let ty = if layout > 0 { 0.5 * spacing - 190.0 } else { -side * 250.0 + side * 190.0 };
    let ty2 = -0.5 * spacing + 190.0;
    let mut tw = aa(boxd(q, DVec2::new(0.0, ty), DVec2::new(0.5 * len - 100.0, 11.5)), fw);
    if layout > 0 {
        tw = tw.max(aa(boxd(q, DVec2::new(0.0, ty2), DVec2::new(0.5 * len - 100.0, 11.5)), fw));
    }
    let ck = (q.x / 600.0).round();
    if ck.abs() * 600.0 < 0.5 * len {
        let (y0, y1) = if layout > 0 {
            if q.y > 0.0 {
                (ty, yr0)
            } else {
                (-0.5 * spacing, ty2)
            }
        } else {
            (yr0.min(ty), yr0.max(ty))
        };
        tw = tw.max(aa(boxd(q, DVec2::new(ck * 600.0, 0.5 * (y0 + y1)), DVec2::new(11.5, 0.5 * (y1 - y0).abs())), fw));
    }
    // apron and terminal: between the runways, or beyond the taxiway
    let (ay0, ay1) = if layout > 0 {
        (ty2 + 60.0, ty - 60.0)
    } else if side > 0.0 {
        (ty + 60.0, ty + 360.0)
    } else {
        (ty - 360.0, ty - 60.0)
    };
    let ac = DVec2::new(0.0, 0.5 * (ay0 + ay1));
    let ah = DVec2::new(0.33 * len, 0.5 * (ay1 - ay0).abs());
    let apron = aa(boxd(q, ac, ah), fw);
    let term_c = DVec2::new(0.0, ac.y + 0.25 * ah.y * if layout > 0 { 0.0 } else { side });
    let term_h = DVec2::new(0.22 * len, 32.0);
    let term = aa(boxd(q, term_c, term_h), fw);
    let hang = (0..3)
        .map(|i| {
            let hc = DVec2::new(ac.x + ah.x - 60.0 - 110.0 * i as f64, ac.y - 0.5 * ah.y * side.max(0.5));
            aa(boxd(q, hc, DVec2::new(40.0, 30.0)), fw)
        })
        .fold(0.0, f64::max);
    let park = aa(boxd(q, DVec2::new(-0.6 * ah.x, term_c.y), DVec2::new(0.25 * ah.x, 0.6 * ah.y)), fw) * (1.0 - term);
    if apron > 0.0 {
        o.col = mixc(o.col, srgb(168.0, 166.0, 160.0), apron);
        o.cls = crate::landcover::TAXIWAY;
        // stand markings, parked aircraft
        let sk = (q.x / 70.0).round();
        let sx = q.x - sk * 70.0;
        let sy = q.y - (term_c.y - side * (term_h.y + 45.0));
        let ah2 = hash2(seed ^ 0xA9, sk as i64, 0);
        if u01k(ah2, 1) < 0.6 && layout == 0 || u01k(ah2, 1) < 0.7 && layout > 0 {
            let fus = aa(boxd(DVec2::new(sx, sy), DVec2::ZERO, DVec2::new(2.2, 22.0)), fw);
            let wing = aa(boxd(DVec2::new(sx, sy - 2.0), DVec2::ZERO, DVec2::new(18.0, 2.6)), fw);
            o.col = mixc(o.col, srgb(232.0, 234.0, 236.0), fus.max(wing) * band(4.0, gsd));
        }
    }
    if park > 0.0 {
        o.col = mixc(o.col, srgb(70.0, 70.0, 74.0), park);
        let (pc, pcol) = crate::kits::transport::px::parked(seed ^ 0xA1C, q.x + 0.6 * ah.x, q.y - term_c.y, fw, gsd);
        o.col = mixc(o.col, pcol, pc * park);
        o.cls = crate::landcover::PARKING;
    }
    if tw > 0.0 {
        o.col = mixc(o.col, srgb(92.0, 92.0, 98.0), tw);
        o.cls = crate::landcover::TAXIWAY;
    }
    if rw > 0.0 {
        let mut c = srgb(56.0, 56.0, 60.0) * (1.0 + 0.04 * (1.0 - smoothstep(0.3 * len, 0.5 * len, rx.abs())));
        // centre-line dashes, edge lines, threshold piano keys, touchdown bars
        let dash = (rx / 50.0 - (rx / 50.0).floor() < 0.6) as u32 as f64;
        let m1 = band_cov(ry, 0.45, fw) * dash;
        let m2 = band_cov(ry.abs() - (wr - 1.5), 0.45, fw);
        let thr = 0.5 * len - rx.abs();
        let keys = if thr > 6.0 && thr < 50.0 && ry.abs() < wr - 3.0 && (ry.abs() / 3.6).fract() < 0.55 { 1.0 } else { 0.0 };
        let td = if thr > 150.0 && thr < 600.0 && ((thr - 150.0) / 150.0).fract() < 0.2 && (ry.abs() - 0.45 * wr).abs() < 0.2 * wr { 1.0 } else { 0.0 };
        let mk = (m1 + m2 + keys + td).min(1.0) * band(1.5, gsd);
        c = mixc(c, srgb(232.0, 232.0, 228.0), mk);
        o.col = mixc(o.col, c, rw);
        o.cls = crate::landcover::RUNWAY;
        // edge lights every 60 m, approach lights at the ends (white)
        let lk = (rx / 60.0).round();
        let d2 = (rx - lk * 60.0).powi(2) + (ry.abs() - wr - 1.0).powi(2);
        o.emit += DVec3::new(1.0, 0.95, 0.85) * point_light(d2, 8.0, 0.35, fw);
    }
    // approach light bars beyond the thresholds
    if ry.abs() < 15.0 && rx.abs() > 0.5 * len && rx.abs() < 0.5 * len + 420.0 {
        let ak = ((rx.abs() - 0.5 * len) / 30.0).round();
        let d2 = (rx.abs() - 0.5 * len - ak * 30.0).powi(2);
        let bar = (1.0 - smoothstep(6.0, 12.0, ry.abs())) * point_light(d2, 6.0, 0.4, fw);
        o.emit += DVec3::new(1.0, 0.92, 0.8) * bar;
    }
    // taxiway edge lights (blue)
    if tw > 0.0 && tw < 1.0 {
        o.emit += DVec3::new(0.25, 0.45, 1.0) * 0.6 * band(30.0, gsd);
    }
    if term > 0.0 {
        o.col = mixc(o.col, srgb(196.0, 198.0, 204.0) * (1.0 - 0.1 * band_cov(boxd(q, term_c, term_h), 1.2, fw)), term);
        o.dh += 18.0 * term * wgt;
        if term > 0.5 {
            o.cls = crate::landcover::BUILDING;
        }
    }
    if hang > 0.0 && term <= 0.0 {
        o.col = mixc(o.col, srgb(176.0, 182.0, 188.0), hang);
        o.dh += 16.0 * hang * wgt;
        if hang > 0.5 {
            o.cls = crate::landcover::BUILDING;
        }
    }
    // apron floods
    o.emit += DVec3::new(1.0, 0.86, 0.62) * 0.12 * apron * (1.0 - term);
    // perimeter road and fence
    o.col = mixc(o.col, srgb(90.0, 90.0, 92.0), band_cov(din - margin - 6.0, 3.5, fw));
    o
}

/// A port: on land a container yard, warehouses, cranes along the quay; on the water piers,
/// moored ships and a breakwater (`water`: the sample is on standing water at level `wl`).
pub fn port_px(st: &FStamp, p: DVec3, water: bool, wl: f64, gsd: f64, fw: f64) -> SitePx {
    let q = uv(st, p);
    let len = st.v[0] as f64;
    let land = st.v[1] as f64;
    let seed = seed_of(st.v[2], st.v[3]);
    let pier = st.v[4] as f64;
    let mut o = SitePx::default();
    if q.x.abs() > 0.5 * len + 250.0 {
        return o;
    }
    let along = q.x.abs() < 0.5 * len;
    if water {
        // piers every 260 m, ships alongside, a breakwater beyond
        let k = (q.x / 260.0).round();
        let px = q.x - k * 260.0;
        let pw = 26.0 + 10.0 * u01k(hash1(seed, k as i64), 1);
        let ins = aa(boxd(DVec2::new(px, q.y), DVec2::new(0.0, 0.5 * pier - 40.0), DVec2::new(pw, 0.5 * pier + 40.0)), fw);
        let on_pier = if along { ins } else { 0.0 };
        let bw = band_cov(q.y - (pier + 160.0), 14.0, fw) * if q.x.abs() < 0.5 * len + 200.0 { 1.0 } else { 0.0 };
        // ships: hulls beside the piers
        let sh = hash2(seed ^ 0x5A1, k as i64, (px > 0.0) as i64);
        let sx = px.abs() - pw - 18.0;
        let slen = 60.0 + 180.0 * u01k(sh, 1);
        let ship = if along && u01k(sh, 2) < 0.65 { aa(boxd(DVec2::new(sx, q.y), DVec2::new(0.0, 0.5 * pier), DVec2::new(14.0, 0.5 * slen)), fw) } else { 0.0 };
        if on_pier > 0.0 {
            o.cov = on_pier;
            o.col = srgb(150.0, 148.0, 142.0) * (0.95 + 0.1 * u01k(hash2(seed, (q.y / 12.0).floor() as i64, k as i64), 1));
            o.dh = wl + 2.5;
            o.abs = true;
            o.cls = crate::landcover::PORT;
            let fl = DVec2::new(px, (q.y / 70.0).round() * 70.0 - q.y);
            o.emit += DVec3::new(1.0, 0.82, 0.55) * (point_light(fl.length_squared(), 10.0, 0.6, fw) + 0.03 * on_pier);
        } else if ship > 0.0 {
            o.cov = ship;
            let deck = match (u01k(sh, 3) * 3.99) as u32 {
                0 => srgb(120.0, 40.0, 36.0),
                1 => srgb(40.0, 60.0, 90.0),
                2 => srgb(60.0, 60.0, 64.0),
                _ => srgb(150.0, 150.0, 146.0),
            };
            // containers stacked on deck
            let cont = ((q.y / 6.5).floor() as i64 + (sx / 2.6).floor() as i64).rem_euclid(3) as f64;
            o.col = mixc(deck, srgb(170.0, 90.0, 60.0) * (0.8 + 0.2 * cont), 0.5 * band(6.0, gsd) * (u01k(sh, 4) < 0.5) as u32 as f64);
            o.dh = wl + 12.0;
            o.abs = true;
            o.cls = crate::landcover::PORT;
        } else if bw > 0.0 {
            o.cov = bw;
            o.col = srgb(128.0, 124.0, 116.0) * (0.85 + 0.25 * u01k(hash1(seed, (q.x / 4.0).floor() as i64), 1));
            o.dh = wl + 4.0;
            o.abs = true;
            o.cls = crate::landcover::ROCK;
        }
        return o;
    }
    // land: from the quay inland
    let depth = -q.y;
    if !along || depth < -60.0 || depth > land {
        return o;
    }
    o.cov = 1.0;
    o.clear = 1.0;
    o.cls = crate::landcover::PORT;
    o.col = srgb(132.0, 130.0, 126.0);
    // quay edge
    let quay = band_cov(depth - 8.0, 8.0, fw);
    // container stacks: rows of multicolour boxes
    let row = ((depth - 40.0) / 16.0).floor();
    let ry = depth - 40.0 - row * 16.0;
    let bay = (q.x / 13.0).floor();
    let ch = hash2(seed ^ 0xC07, bay as i64, row as i64);
    let in_yard = depth > 40.0 && depth < 0.62 * land;
    let zone = u01k(hash1(seed, (q.x / 200.0).floor() as i64), 1);
    if in_yard && zone < 0.7 {
        let box_ = aa((6.0 - (ry - 8.0).abs()).min(6.0 - (q.x - (bay + 0.5) * 13.0).abs()), fw);
        if u01k(ch, 1) < 0.85 {
            let cc = match (u01k(ch, 2) * 5.99) as u32 {
                0 => srgb(170.0, 60.0, 44.0),
                1 => srgb(50.0, 90.0, 150.0),
                2 => srgb(200.0, 170.0, 70.0),
                3 => srgb(70.0, 120.0, 80.0),
                4 => srgb(200.0, 200.0, 196.0),
                _ => srgb(120.0, 70.0, 50.0),
            };
            let e = band(6.0, gsd);
            o.col = mixc(o.col, cc, box_ * e + 0.45 * (1.0 - e));
            o.dh = (2.6 * (1.0 + (u01k(ch, 3) * 3.99).floor())) * box_;
        }
    } else if depth > 0.62 * land || (in_yard && zone >= 0.7) {
        // warehouses
        let wk = (q.x / 120.0).floor();
        let wh = hash1(seed ^ 0x3A2, wk as i64);
        let wc = DVec2::new((wk + 0.5) * 120.0, -(0.62 * land + 0.19 * land));
        let ins = aa(boxd(q, wc, DVec2::new(48.0, 0.15 * land)), fw);
        if ins > 0.0 && u01k(wh, 1) < 0.8 {
            o.col = mixc(o.col, if u01k(wh, 2) < 0.5 { srgb(176.0, 178.0, 180.0) } else { srgb(110.0, 130.0, 156.0) }, ins);
            o.dh = 12.0 * ins;
            if ins > 0.5 {
                o.cls = crate::landcover::BUILDING;
            }
        }
    }
    // gantry cranes along the quay
    let ck = (q.x / 90.0).round();
    let cx = q.x - ck * 90.0;
    let crane = if u01k(hash1(seed ^ 0xC4A, ck as i64), 1) < 0.7 {
        let legs = band_cov(cx.abs() - 9.0, 1.2, fw) * band_cov(depth - 20.0, 20.0, fw);
        let boom = band_cov(cx, 1.6, fw) * band_cov(depth - 10.0, 50.0, fw);
        legs.max(boom)
    } else {
        0.0
    };
    o.col = mixc(o.col, srgb(186.0, 182.0, 176.0), quay);
    if crane > 0.0 {
        o.col = mixc(o.col, if u01k(seed, 5) < 0.5 { srgb(160.0, 48.0, 40.0) } else { srgb(60.0, 100.0, 150.0) }, crane);
        o.dh = o.dh.max(45.0 * crane);
    }
    // floodlight masts on a grid
    let fl = DVec2::new(q.x - (q.x / 80.0).round() * 80.0, depth - ((depth / 80.0).round() * 80.0));
    o.emit += DVec3::new(1.0, 0.84, 0.6) * (point_light(fl.length_squared(), 14.0, 0.8, fw) + 0.025);
    o
}

/// A quarry or open-pit mine: benches (stepped bowl), haul road, pit lake, spoil heap,
/// tailings pond; `dh` is relative to the ground (pass B only).
pub fn quarry_px(st: &FStamp, p: DVec3, gsd: f64, fw: f64) -> SitePx {
    let q = uv(st, p);
    let r = st.v[0] as f64;
    let depth = st.v[1] as f64;
    let seed = seed_of(st.v[2], st.v[3]);
    let kind = st.v[4] as u32;
    let bh = st.v[5] as f64;
    let ta = st.v[6] as f64;
    let tr = st.v[7] as f64;
    let mut o = SitePx { clear: 1.0, cls: crate::landcover::QUARRY, ..Default::default() };
    // an irregular outline (lobes by the angle; per-quarry phases)
    let qa = DVec2::new(q.x / 1.25, q.y * 1.1);
    let th0 = qa.y.atan2(qa.x);
    let wob = 1.0 + 0.13 * (3.0 * th0 + std::f64::consts::TAU * u01k(seed, 8)).sin() + 0.07 * (5.0 * th0 + std::f64::consts::TAU * u01k(seed, 9)).sin();
    let rn = qa.length() / (r * wob);
    let (rock, bench) = match kind {
        0 => (srgb(196.0, 192.0, 180.0), srgb(222.0, 218.0, 204.0)),
        1 => (srgb(150.0, 112.0, 82.0), srgb(186.0, 150.0, 112.0)),
        2 => (srgb(150.0, 134.0, 128.0), srgb(184.0, 170.0, 160.0)),
        _ => (srgb(170.0, 150.0, 118.0), srgb(196.0, 180.0, 148.0)),
    };
    if rn < 1.0 {
        let floor_r = 0.25;
        let dn = ((1.0 - rn) / (1.0 - floor_r)).clamp(0.0, 1.0);
        let d = depth * dn;
        let res = band(bh * 1.2 / (depth / ((1.0 - floor_r) * r)).max(0.05), gsd);
        let b = (d / bh).floor();
        let f = d / bh - b;
        let step = b + smoothstep(0.75, 1.0, f);
        let z = -bh * (step * res + (d / bh) * (1.0 - res));
        o.cov = 1.0;
        o.dh = z;
        let face = smoothstep(0.7, 0.95, f) * res;
        o.col = mixc(bench, rock * 0.75, face) * (0.9 + 0.15 * u01k(hash1(seed, b as i64), 1));
        // spiral haul road
        let th = qa.y.atan2(qa.x);
        let turns = 1.5 + (depth / 120.0).min(2.5);
        let road_th = dn * turns * std::f64::consts::TAU;
        let dth = (th - road_th) - std::f64::consts::TAU * ((th - road_th) / std::f64::consts::TAU).round();
        o.col = mixc(o.col, bench * 1.05, band_cov(dth * rn * r, 7.0, fw));
        // a pit lake in the deepest part (open pits)
        if kind == 1 && dn > 0.92 {
            o.col = srgb(70.0, 140.0, 130.0);
            o.cls = crate::landcover::LAKE;
        }
        return o;
    }
    // spoil heap beside the pit
    let sc = DVec2::new(ta.cos(), ta.sin()) * (1.3 * r + 60.0);
    let qs = q - sc;
    let ths = qs.y.atan2(qs.x);
    let ds = qs.length() / ((0.35 * r + 40.0) * (1.0 + 0.25 * (4.0 * ths + std::f64::consts::TAU * u01k(seed, 10)).sin()));
    if ds < 1.0 {
        o.cov = 1.0;
        o.dh = (0.12 * r).min(35.0) * (1.0 - ds * ds);
        o.col = bench * (0.92 + 0.12 * (1.0 - ds));
        return o;
    }
    // tailings pond (open pits)
    if tr > 0.0 {
        let tc = DVec2::new((ta + 2.2).cos(), (ta + 2.2).sin()) * (r * 1.35 + tr + 120.0);
        let dt = (q - tc).length() - tr;
        if dt < 25.0 {
            o.cov = 1.0;
            let beach = smoothstep(-0.25 * tr, 0.0, dt);
            o.col = mixc(srgb(96.0, 150.0, 146.0), srgb(196.0, 184.0, 156.0), beach);
            o.col = mixc(o.col, srgb(150.0, 140.0, 120.0), band_cov(dt - 10.0, 9.0, fw));
            o.dh = 3.0 * band_cov(dt - 10.0, 9.0, fw);
            o.cls = if beach < 0.5 { crate::landcover::AQUACULTURE } else { crate::landcover::QUARRY };
            return o;
        }
    }
    // the haul track out of the pit and a cleared rim
    let rim = band_cov(rn - 1.04, 0.04, fw / r);
    if rim > 0.0 {
        o.cov = rim;
        o.col = bench * 0.95;
    }
    o
}

/// A solar farm: rows of panels (facing the equator), service roads, fence.
pub fn solar_px(st: &FStamp, p: DVec3, gsd: f64, fw: f64) -> SitePx {
    let q = uv(st, p);
    let (hx, hy) = (st.half[0] as f64, st.half[1] as f64);
    let pitch = st.v[4] as f64;
    let pw = st.v[5] as f64;
    let din = (hx - q.x.abs()).min(hy - q.y.abs());
    if din <= 0.0 {
        return SitePx::default();
    }
    let mut o = SitePx { cov: 1.0, clear: 1.0, cls: crate::landcover::SOLAR_FARM, ..Default::default() };
    let ground = srgb(150.0, 140.0, 116.0);
    let panel = srgb(34.0, 42.0, 70.0);
    let e = band(pitch, gsd);
    let fy = q.y / pitch - (q.y / pitch).floor();
    let blockx = (q.x + hx) / 130.0;
    let road = band_cov((blockx - blockx.round()) * 130.0, 3.0, fw).max(band_cov(q.y, 4.0, fw));
    let row = if (fy * pitch) < pw { 1.0 } else { 0.0 };
    let mean = pw / pitch;
    let cov_p = (row * e + mean * (1.0 - e)) * (1.0 - road) * smoothstep(8.0, 12.0, din);
    o.col = mixc(ground, panel * (1.0 + 0.25 * (fy * pitch / pw).min(1.0) * e), cov_p);
    o.col = mixc(o.col, srgb(170.0, 160.0, 140.0), road * 0.7);
    o.col = mixc(o.col, srgb(80.0, 80.0, 82.0), band_cov(din - 4.0, 0.6, fw));
    o.dh = 1.6 * cov_p;
    o
}

/// A wind farm: turbines on a jittered grid (rows across the wind): white tower and nacelle,
/// three blades, a long shadow, a gravel pad and access track; red aviation lights at night.
pub fn wind_px(st: &FStamp, p: DVec3, sun: DVec2, sun_tan: f64, gsd: f64, fw: f64) -> SitePx {
    let q = uv(st, p);
    let (hx, hy) = (st.half[0] as f64, st.half[1] as f64);
    let sp = st.v[0] as f64;
    let hub = st.v[1] as f64;
    let seed = seed_of(st.v[2], st.v[3]);
    let rr = st.v[4] as f64;
    let dens = st.v[5] as f64;
    let mut o = SitePx::default();
    if (q.x / hx).powi(2) + (q.y / hy).powi(2) > 1.0 {
        return o;
    }
    let cell = DVec2::new(sp, 1.6 * sp);
    let ci = (q / cell).round();
    let th = hash2(seed, ci.x as i64, ci.y as i64);
    let tc = ci * cell + (DVec2::new(u01k(th, 1), u01k(th, 2)) - 0.5) * 0.3 * cell;
    let tcn = DVec2::new(tc.x / hx, tc.y / hy);
    if u01k(th, 3) >= dens || tcn.length() > 0.95 {
        return o;
    }
    let r = q - tc;
    let fwt = fw.max(0.3 * gsd);
    // access track along the row, pad
    let track = band_cov(r.y, 2.5, fwt) * (r.x.abs() < 0.5 * cell.x) as u32 as f64;
    let pad = aa(18.0 - r.length(), fwt);
    let tower = aa(2.4 - r.length(), fwt);
    // blades: three thin arms from the hub (seen from above), a hashed rotor angle
    let a0 = u01k(th, 4) * std::f64::consts::TAU;
    let mut blade: f64 = 0.0;
    for k in 0..3 {
        let a = a0 + k as f64 * std::f64::consts::TAU / 3.0;
        let d = DVec2::new(a.cos(), a.sin());
        let t = r.dot(d);
        let n = (r - d * t).length();
        if t > 0.0 && t < rr {
            blade = blade.max(band_cov(n, 1.3 * (1.0 - 0.6 * t / rr), fwt));
        }
    }
    // tower shadow: a line away from the sun
    let sl = hub / sun_tan.max(0.1);
    let ts = r.dot(-sun);
    let tn = (r + sun * ts).length();
    let shadow = if ts > 0.0 && ts < sl { band_cov(tn, 1.8, fwt) } else { 0.0 };
    let gravel = track.max(pad);
    o.emit = DVec3::new(1.0, 0.1, 0.05) * point_light(r.length_squared(), 4.0, 0.5, fw);
    // the shadow darkens what lies below (light, not a feature)
    o.shade = 0.7 * shadow * (1.0 - blade.max(tower));
    let white = blade.max(tower);
    o.cov = gravel.max(white);
    if o.cov <= 0.0 {
        return o;
    }
    o.col = mixc(srgb(160.0, 150.0, 132.0), srgb(236.0, 238.0, 240.0), white / o.cov);
    o.cls = if gravel > 0.5 { crate::landcover::TRACK } else { 0 };
    o.clear = gravel;
    o.dh = hub * tower;
    o
}

/// Gas flares: well pads joined by tracks, a burning flare (bright orange at night, a
/// scorched ring by day).
pub fn flare_px(st: &FStamp, p: DVec3, gsd: f64, fw: f64) -> SitePx {
    let q = uv(st, p);
    let n = st.v[0] as u32;
    let seed = seed_of(st.v[2], st.v[3]);
    let mut o = SitePx::default();
    let fwt = fw.max(0.4 * gsd);
    let mut best = 0.0;
    let mut track: f64 = 0.0;
    for k in 0..5u32 {
        if k < n {
            let h = hash1(seed, k as i64);
            let c = if k == 0 { DVec2::ZERO } else { (DVec2::new(u01k(h, 1), u01k(h, 2)) - 0.5) * 2400.0 };
            let pad = aa(boxd(q, c, DVec2::splat(35.0 + 20.0 * u01k(h, 3))), fwt);
            best = f64::max(best, pad);
            if k > 0 {
                let d = c.length().max(1.0);
                let dir = c / d;
                let t = q.dot(dir).clamp(0.0, d);
                track = track.max(band_cov((q - dir * t).length(), 2.5, fwt));
            }
        }
    }
    let fc = DVec2::new(20.0, 20.0);
    let dfl = (q - fc).length();
    let scorch = aa(22.0 - dfl, fwt);
    o.emit = DVec3::new(1.0, 0.55, 0.16) * point_light(dfl * dfl, 60.0, 3.0, fw);
    let cov = best.max(track).max(scorch);
    if cov > 0.0 {
        o.cov = cov;
        o.col = mixc(srgb(176.0, 160.0, 130.0), srgb(46.0, 40.0, 36.0), scorch * (1.0 - smoothstep(10.0, 22.0, dfl)));
        o.cls = crate::landcover::BARE;
        o.clear = cov;
    }
    o
}

/// A reservoir piece at a sample: inside its reach, upstream of the dam (half plane), below
/// the water level. Returns (water coverage, bathtub ring coverage).
pub fn reservoir_at(sg: &FSeg, p: DVec3, ground: f64, fw: f64) -> (f64, f64) {
    let ab = sg.b - sg.a;
    let len = ab.length().max(1.0);
    let dx = ab / len;
    let left = sg.a.normalize().cross(dx).normalize();
    let ap = p - sg.a;
    let (x, y) = (ap.dot(dx), ap.dot(left));
    let t = x.clamp(0.0, len);
    let dist = DVec2::new(x - t, y).length();
    if dist > sg.hw as f64 {
        return (0.0, 0.0);
    }
    // upstream of the dam line: the dam point and downstream direction in this frame
    let (da, dl, nd) = (sg.v[0] as f64, sg.v[1] as f64, sg.v[2] as f64);
    let rel = DVec2::new(x - da, y - dl);
    if rel.dot(DVec2::new(nd.cos(), nd.sin())) > 0.0 {
        return (0.0, 0.0);
    }
    let level = sg.ha as f64;
    let crest = sg.hb as f64;
    let wet = aa(level - ground, fw * 0.05 + 0.3);
    let ring = if ground < crest - 1.0 { 1.0 - wet } else { 0.0 };
    (wet, ring)
}

/// A dam wall at a sample: (coverage, crest height, the downstream face shade).
pub fn dam_at(sg: &FSeg, p: DVec3, ground: f64, fw: f64) -> (f64, f64, f64) {
    let (d, _, t) = crate::features::seg_frame(sg, p);
    let crest = sg.ha as f64;
    if t <= 0.0 || t >= 1.0 || ground > crest + 1.0 {
        return (0.0, crest, 0.0);
    }
    let cov = band_cov(d, sg.hw as f64, fw);
    (cov, crest, smoothstep(-(sg.hw as f64), sg.hw as f64, d))
}
