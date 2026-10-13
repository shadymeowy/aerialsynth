//! The features of an area (`docs/design/terrain-next.md` §6.3, §6.4): the edges whose routes come
//! near it, their engineered profiles (deck heights from point samples: cut and fill, bridges,
//! tunnels), segments with their cross-section parameters, crossings (interchanges, overpasses,
//! roundabouts), spurs from towns to passing motorways (trumpets), roads and power lines to the
//! settlements kit's sites, stations and yards. Everything is a pure function of the world and
//! the edge (memoized per edge, per edge pair and per sample), so neighbouring tiles agree.

use super::net::{self, network as nw, Edge, Node};
use super::route::{self, Line};
use super::{cls, flag, tpl, S_WRAP};
use crate::features::{Area, FSeg, FStamp, Features, HostCtx};
use crate::kits::settlements::{places, sites};
use crate::noise::*;
use crate::tile::PointTerrain;
use crate::world::{water, World};
use glam::{DVec2, DVec3};
use std::sync::{Arc, RwLock};

/// Areas whose pixel is below this size (m) get the engineered profiles (decks, bridges,
/// tunnels, embankments); coarser ones draw the roads on the ground.
pub const PROFILE_GSD: f64 = 40.0;
/// Deck sample spacing along a route (m).
pub const DS: f64 = 40.0;
/// Overpass clearance (m) and approach grade.
const CLEAR: f64 = 7.5;
const RAMP_G: f64 = 0.05;

// ---------------------------------------------------------------- cross-sections

/// Carriageway half width (m) of a class (lanes per direction for motorways; tracks for rail).
pub fn half_width(class: usize, lanes: u32) -> f64 {
    match class {
        cls::MOTORWAY => 1.6 + 3.6 * lanes as f64 + 3.0,
        cls::TRUNK => 5.0,
        cls::SECONDARY => 3.5,
        cls::LOCAL => 2.6,
        cls::TRACK => 1.5,
        cls::RAMP => 4.2,
        cls::RAIL => 0.6 + 2.0 * lanes as f64,
        _ => 0.0,
    }
}

/// Verge width beyond the carriageway (m): cleared, gravel and grass.
pub fn verge(class: usize) -> f64 {
    match class {
        cls::MOTORWAY => 6.0,
        cls::TRUNK => 3.5,
        cls::SECONDARY => 2.5,
        cls::LOCAL => 1.5,
        cls::TRACK => 0.8,
        cls::RAMP => 3.0,
        cls::RAIL => 3.0,
        _ => 0.0,
    }
}

fn profiled(class: usize) -> bool {
    matches!(class, cls::MOTORWAY | cls::TRUNK | cls::RAIL)
}

/// Rank for crossings: which road goes over.
fn rank(class: usize) -> u32 {
    [6, 4, 3, 2, 1, 3, 5, 0][class]
}

// ---------------------------------------------------------------- routes

type RKey = (u64, u64);
/// Analytic routes (no sites) and their scenes; the final routes.
#[derive(Clone)]
pub(crate) struct Base {
    pub line: Option<Arc<Line>>,
    pub scene: Arc<route::Scene>,
    /// the route's ends (on dry land)
    pub ends: [DVec3; 2],
}
static BASE: RwLock<Option<FxHashMap<RKey, Base>>> = RwLock::new(None);
static ROUTES: RwLock<Option<FxHashMap<RKey, Option<Arc<Line>>>>> = RwLock::new(None);

/// Where a route of `class` ends at node `n` (coming from `toward`): a road enters a town or
/// larger at the nearest arterial (a district edge of the settlements kit's town model) on its
/// way to the centre, a power line ends at the built-up edge; railways run to the centre (the
/// station), other nodes are points.
pub fn end_point(w: &World, n: &Node, toward: DVec3, class: usize) -> DVec3 {
    if n.level < places::TOWN || n.level > places::METRO || class == cls::RAIL || class == cls::RAMP {
        return n.p;
    }
    let Some(pl) = places::places_near(w, n.level as usize, n.p, 1.0).into_iter().find(|p| p.id == n.id) else { return n.p };
    let tp = crate::kits::settlements::town::decode(&crate::kits::settlements::place_stamp(w, &pl));
    let d = toward - tp.c;
    let dir = DVec2::new(d.dot(tp.ex), d.dot(tp.ey)).normalize_or_zero();
    let se = tp.elong.sqrt();
    let k = DVec2::new(dir.x / se, dir.y * se).length().max(1e-9);
    let rel = match class {
        cls::POWER => 1.0,
        cls::MOTORWAY => 0.65,
        _ => 0.8,
    };
    let mut q = dir * (rel * tp.r / k);
    if class != cls::POWER && tp.form == places::form::CITY as u32 {
        // the bisector of the two nearest district sites (as `town::district`), within reach
        let dcell = [400.0, 400.0, 600.0, 1000.0, 1400.0][tp.level as usize];
        let dc = worley2(tp.seed ^ 0xD15, q, dcell, 0.85);
        let nn = (dc.point2 - dc.point).normalize_or_zero();
        let off = (q - (dc.point + dc.point2) * 0.5).dot(nn);
        let lim = route::ENTRY.min(0.25 * tp.r);
        q -= nn * off.clamp(-lim, lim);
    }
    route::surface(w, tp.c + tp.ex * q.x + tp.ey * q.y)
}

/// The ends of an edge's route.
pub(crate) fn ends(w: &World, e: &Edge) -> (DVec3, DVec3) {
    (end_point(w, &e.a, e.b.p, e.class), end_point(w, &e.b, e.a.p, e.class))
}

/// The analytic part of an edge's route (memoized): its scene (towns to go around, drainage,
/// lakes) and the route through it.
pub(crate) fn base(w: &World, e: &Edge) -> Base {
    let key = (w.cache_key, e.key);
    if let Some(v) = BASE.read().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return v.clone();
    }
    let avoid: Vec<(DVec3, f64)> = if matches!(e.class, cls::MOTORWAY | cls::TRUNK | cls::RAIL | cls::POWER) {
        let mid = (e.a.p + e.b.p) * 0.5;
        let dev = route::max_deviation(e.class, e.len());
        let mut v = Vec::new();
        for l in places::TOWN as usize..places::NLEVELS {
            for p in places::places_near(w, l, mid, 0.5 * e.len() + dev + 15_000.0) {
                if p.id != e.a.id && p.id != e.b.id && p.parent != e.a.id && p.parent != e.b.id && net::chord_dist(e.a.p, e.b.p, p.center) < dev + p.radius {
                    v.push((p.center, 0.9 * p.radius));
                }
            }
        }
        v
    } else {
        Vec::new()
    };
    let (mut a, mut b) = ends(w, e);
    let half = route::max_deviation(e.class, (b - a).length());
    let fine = route::PARAMS[e.class].span > 0.0 && route::PARAMS[e.class].span < 300.0;
    let sc = if e.class == cls::POWER { route::Scene { avoid, ..Default::default() } } else { route::Scene::of_corridor(w, a, b, half, fine, avoid) };
    if e.class != cls::POWER {
        // (a node in a lake or by the shore: the road ends on the shore)
        let a0 = a;
        a = route::dry_end(w, &sc, a0, b);
        b = route::dry_end(w, &sc, b, a0);
    }
    let r = route::route(w, a, b, e.class, &sc).map(Arc::new);
    let v = Base { line: r, scene: Arc::new(sc), ends: [a, b] };
    let mut g = BASE.write().unwrap();
    let m = g.get_or_insert_with(FxHashMap::default);
    if m.len() > 50_000 {
        m.clear();
    }
    m.insert(key, v.clone());
    v
}

/// The route of an edge (memoized): the analytic route, routed again around the sites near it
/// (no-go footprints, reservoirs) where it touches any. None: pending point evaluations (the
/// sites'); Some(None): no route (e.g. across the sea).
pub fn routed(ctx: &mut HostCtx, e: &Edge) -> Option<Option<Arc<Line>>> {
    let w = ctx.world;
    let key = (w.cache_key, e.key);
    if let Some(v) = ROUTES.read().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return Some(v.clone());
    }
    let Base { line: r0, scene: sc0, ends: [mut a, mut b] } = base(w, e);
    let mut line = r0;
    let mut sc: Option<route::Scene> = None;
    let mut tries = 0;
    while let Some(l) = line.clone() {
        let (blocks, res) = obstacles(ctx, &l, e.class, [a, b])?;
        if blocks.is_empty() && res.is_empty() {
            break;
        }
        let sc = sc.get_or_insert_with(|| (*sc0).clone());
        let mut more = false;
        for k in blocks {
            if !sc.blocks.contains(&k) {
                sc.blocks.push(k);
                more = true;
            }
        }
        for r in res {
            if !sc.res.iter().any(|x| x.a == r.a && x.b == r.b && x.ha == r.ha) {
                sc.res.push(r);
                more = true;
            }
        }
        if !more || route::check(w, sc, &l, e.class, false) {
            break;
        }
        tries += 1;
        if tries > 2 {
            line = None;
            break;
        }
        if e.class != cls::POWER {
            // (an end in a reservoir: on its shore)
            let a0 = a;
            a = route::dry_end(w, sc, a0, b);
            b = route::dry_end(w, sc, b, a0);
        }
        line = route::route(w, a, b, e.class, sc).map(Arc::new);
    }
    let mut g = ROUTES.write().unwrap();
    let m = g.get_or_insert_with(FxHashMap::default);
    if m.len() > 100_000 {
        m.clear();
    }
    m.insert(key, line.clone());
    // (the scene served its purpose)
    if let Some(m) = BASE.write().unwrap().as_mut() {
        m.remove(&key);
    }
    Some(line)
}

/// The sites near a route of `class` (its ends excepted): footprints of airports, quarries,
/// solar and wind farms (widened by the road), and reservoir pieces. None: pending.
pub(crate) fn obstacles(ctx: &mut HostCtx, line: &Line, class: usize, ends: [DVec3; 2]) -> Option<(Vec<route::Block>, Vec<FSeg>)> {
    let w = ctx.world;
    let margin = if class == cls::POWER { 30.0 } else { half_width(class, 2) + verge(class) + 15.0 };
    // the line in chunks of ≤ 15 km: centre and radius
    let mut chunks: Vec<(DVec3, f64)> = Vec::new();
    let n = line.pts.len();
    let mut k0 = 0;
    while k0 + 1 < n {
        let mut k1 = k0 + 1;
        while k1 + 1 < n && line.s[k1] - line.s[k0] < 15_000.0 {
            k1 += 1;
        }
        let c = (line.pts[k0] + line.pts[k1]) * 0.5;
        let r = line.pts[k0..=k1].iter().map(|p| (*p - c).length()).fold(0.0, f64::max);
        chunks.push((c, r));
        k0 = k1;
    }
    let near = |c: DVec3, r: f64| chunks.iter().any(|&(q, rq)| (q - c).length() < rq + r);
    let mut towns: Vec<places::Place> = Vec::new();
    let mut qc: Vec<(i64, i64, i64)> = Vec::new();
    let mut wc: Vec<(i64, i64, i64)> = Vec::new();
    let mut mc: Vec<(i64, i64, i64)> = Vec::new();
    for &(c, r) in &chunks {
        for l in places::TOWN as usize..places::NLEVELS {
            towns.extend(places::places_near(w, l, c, r + 50_000.0));
        }
        qc.extend(sites::lattice_cells(w, sites::QUARRY_CELL, c, r + 8_000.0));
        wc.extend(sites::lattice_cells(w, sites::WIND_CELL, c, r + 8_000.0));
        mc.extend(sites::lattice_cells(w, sites::MTN_DAM_CELL, c, r + 45_000.0));
    }
    towns.sort_unstable_by_key(|p| p.id);
    towns.dedup_by_key(|p| p.id);
    for v in [&mut qc, &mut wc, &mut mc] {
        v.sort_unstable();
        v.dedup();
    }
    let mut pending = false;
    let mut blocks: Vec<route::Block> = Vec::new();
    let mut res: Vec<FSeg> = Vec::new();
    let add = |s: &sites::Site, blocks: &mut Vec<route::Block>, res: &mut Vec<FSeg>| {
        for st in &s.stamps {
            let b = match st.template {
                sites::tpl::QUARRY => route::Block { c: st.center, ex: st.ex, ey: st.ey, half: DVec2::splat(1.6 * st.v[0] as f64 + margin), round: true },
                sites::tpl::AIRPORT | sites::tpl::SOLAR | sites::tpl::WIND => {
                    route::Block { c: st.center, ex: st.ex, ey: st.ey, half: DVec2::new(st.half[0] as f64, st.half[1] as f64) + margin, round: false }
                }
                _ => continue,
            };
            // (a route ending at or in the site keeps it)
            let grown = route::Block { half: b.half + 60.0, ..b };
            if near(b.c, b.half.length() + 100.0) && !ends.iter().any(|&p| grown.inside(p)) {
                blocks.push(b);
            }
        }
        if class != cls::POWER {
            for sg in s.segs.iter().filter(|g| g.kind == sites::tpl::RESERVOIR) {
                if near((sg.a + sg.b) * 0.5, 0.5 * (sg.b - sg.a).length() + sg.hw as f64 + 100.0) {
                    res.push(*sg);
                }
            }
        }
    };
    for pl in &towns {
        let d = line.dist(pl.center).0;
        if pl.airport && d < pl.radius + 13_000.0 {
            match sites::airport(ctx, pl) {
                None => pending = true,
                Some(Some(s)) => add(&s, &mut blocks, &mut res),
                Some(None) => {}
            }
        }
        if d < 1.2 * pl.radius + 6_500.0 {
            for k in 0..2 {
                match sites::solar(ctx, pl, k) {
                    None => pending = true,
                    Some(Some(s)) => add(&s, &mut blocks, &mut res),
                    Some(None) => {}
                }
            }
        }
        if class != cls::POWER && pl.parent == 0 && d < 50_000.0 && u01k(pl.id, 50) < if pl.level >= places::CITY { 0.5 } else { 0.15 } {
            match sites::dam(ctx, mix64(pl.id ^ 0xDA4), pl.center, pl.radius + 2500.0, 22_000.0) {
                None => pending = true,
                Some(Some(s)) => add(&s, &mut blocks, &mut res),
                Some(None) => {}
            }
        }
    }
    for &c in &qc {
        match sites::quarry(ctx, c) {
            None => pending = true,
            Some(Some(s)) => add(&s, &mut blocks, &mut res),
            Some(None) => {}
        }
    }
    for &c in &wc {
        match sites::wind(ctx, c) {
            None => pending = true,
            Some(Some(s)) => add(&s, &mut blocks, &mut res),
            Some(None) => {}
        }
    }
    if class != cls::POWER {
        for &c in &mc {
            if let Some((id, p)) = sites::lattice_site(w, 0xDA7, sites::MTN_DAM_CELL, c) {
                if u01k(id, 1) < 0.2 && w.smooth_relief(p, 3000.0).1 > 350.0 && w.atlas().sample(p.normalize()).glaciation < 0.4 {
                    match sites::dam(ctx, id, p, 0.0, 20_000.0) {
                        None => pending = true,
                        Some(Some(s)) => add(&s, &mut blocks, &mut res),
                        Some(None) => {}
                    }
                }
            }
        }
    }
    if pending {
        return None;
    }
    Some((blocks, res))
}

// ---------------------------------------------------------------- profiles

type SKey = (u64, u64, i64);
static SAMPLES: RwLock<Option<FxHashMap<SKey, PointTerrain>>> = RwLock::new(None);

/// Point terrain at the deck samples `j0..=j1` of an edge (None: pending on the GPU; all are
/// requested at once).
fn samples(ctx: &mut HostCtx, key: u64, line: &Line, j0: i64, j1: i64) -> Option<Vec<PointTerrain>> {
    let w = ctx.world;
    let mut out: Vec<Option<PointTerrain>> = Vec::with_capacity((j1 - j0 + 1).max(0) as usize);
    {
        let g = SAMPLES.read().unwrap();
        for j in j0..=j1 {
            out.push(g.as_ref().and_then(|m| m.get(&(w.cache_key, key, j))).copied());
        }
    }
    let mut got = Vec::new();
    for (k, j) in (j0..=j1).enumerate() {
        if out[k].is_none() {
            let p = line.at(j as f64 * DS).0;
            let gd = geodesy::ecef2geodetic(p, &w.ell);
            if let Some(t) = ctx.terrain(gd.lat, gd.lon, 20.0) {
                got.push((k, j, gd.lat, gd.lon, t));
            }
        }
    }
    // (some need an exact sample: in parallel)
    use rayon::prelude::*;
    let exact = ctx.points.exact();
    let new: Vec<(i64, PointTerrain)> = got.par_iter().map(|&(_, j, lat, lon, t)| (j, deck_sample(w, lat, lon, t, exact))).collect();
    for (g, &(_, t)) in got.iter().zip(&new) {
        out[g.0] = Some(t);
    }
    if !new.is_empty() {
        let mut g = SAMPLES.write().unwrap();
        let m = g.get_or_insert_with(FxHashMap::default);
        if m.len() > 2_000_000 {
            m.clear();
        }
        for (j, t) in new {
            m.insert((w.cache_key, key, j), t);
        }
    }
    out.into_iter().collect()
}

/// Deck samples are quantized to this (m), so that the profiles (decks, bridges, tunnels,
/// portals) are the same on both backends: the GPU's samples differ from the CPU's by
/// millimetres to centimetres, which moved every deck a little.
const DECK_Q: f64 = 1.0;
/// A backend sample this close (m) to a rounding boundary of [`DECK_Q`] is re-evaluated exactly.
const DECK_EPS: f64 = 0.04;

/// The deck sample from the backend's point terrain `t` at (lat, lon): ground and water level
/// quantized to [`DECK_Q`], from the exact sample (the CPU's `World::terrain`, the CPU backend's
/// own) where `t` is within [`DECK_EPS`] of a rounding boundary or near a water decision
/// (`sites::water_fragile`), else from `t`.
fn deck_sample(w: &World, lat: f64, lon: f64, t: PointTerrain, exact: bool) -> PointTerrain {
    let near = |h: f64| ((h / DECK_Q).fract().abs() - 0.5).abs() < DECK_EPS / DECK_Q;
    // (`exact`: `t` is the exact sample already, the CPU backend's)
    let t = if !exact && (sites::water_fragile(&t) || near(t.ground) || (t.water.is_finite() && near(t.water))) {
        let e = w.terrain(&crate::world::Ctx::new(lat, lon, 20.0, &w.ell));
        PointTerrain { ground: e.ground, water: e.water, water_kind: e.water_kind }
    } else {
        t
    };
    let q = |h: f64| (h / DECK_Q).round() * DECK_Q;
    PointTerrain { ground: q(t.ground), water: if t.water.is_finite() { q(t.water) } else { t.water }, water_kind: t.water_kind }
}

/// The engineered profile of a stretch: deck heights, ground, bridge and tunnel flags at the
/// samples `j0 ..` (pieces j → j + 1).
#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub j0: i64,
    pub deck: Vec<f64>,
    pub ground: Vec<f64>,
    pub bridge: Vec<bool>,
    pub tunnel: Vec<bool>,
}

impl Profile {
    fn idx(&self, j: i64) -> usize {
        (j - self.j0).clamp(0, self.deck.len() as i64 - 1) as usize
    }
    pub fn deck_at(&self, s: f64) -> f64 {
        let j = (s / DS).floor() as i64;
        let t = s / DS - j as f64;
        let (a, b) = (self.deck[self.idx(j)], self.deck[self.idx(j + 1)]);
        a + (b - a) * t
    }
    pub fn piece(&self, s: f64) -> (bool, bool) {
        let k = self.idx((s / DS).floor() as i64);
        (self.bridge[k], self.tunnel[k])
    }
}

/// Smoothing half window and grade-limit radius (samples), grade limit of a profiled class.
fn deck_params(class: usize) -> (i64, i64, f64) {
    match class {
        cls::MOTORWAY => (5, 30, 0.045),
        cls::TRUNK => (4, 30, 0.065),
        _ => (10, 45, 0.02),
    }
}

/// The profile of the stretch `s_lo..s_hi` of a route: a local operator on the samples (a box
/// mean of the ground, raised over water, then the mean of its largest grade-limited minorant
/// and smallest majorant within the radius), so any stretch of the route gets the same decks.
pub fn profile(ctx: &mut HostCtx, e: &Edge, line: &Line, s_lo: f64, s_hi: f64) -> Option<Profile> {
    let (wn, rn, g) = deck_params(e.class);
    let nmax = (line.len() / DS).ceil() as i64;
    let j0 = ((s_lo / DS).floor() as i64 - 3).max(0);
    let j1 = ((s_hi / DS).ceil() as i64 + 3).min(nmax);
    let f0 = (j0 - wn - rn - 16).max(0);
    let f1 = (j1 + wn + rn + 16).min(nmax);
    let smp = samples(ctx, e.key, line, f0, f1)?;
    let at = |j: i64| &smp[(j.clamp(f0, f1) - f0) as usize];
    let sj = |j: i64| (j as f64 * DS).min(line.len());
    let lifted = |t: &PointTerrain| -> f64 {
        match t.water_kind {
            water::NONE => t.ground,
            water::RIVER => t.ground + 8.0,
            _ => t.water.max(t.ground) + CLEAR,
        }
    };
    let m: Vec<f64> = (f0..=f1)
        .map(|j| {
            let (a, b) = ((j - wn).max(0), (j + wn).min(nmax));
            (a..=b).map(|i| lifted(at(i))).sum::<f64>() / (b - a + 1) as f64
        })
        .collect();
    let mm = |j: i64| m[(j.clamp(f0, f1) - f0) as usize];
    let (e0, e1) = ((j0 - 6).max(0), (j1 + 6).min(nmax));
    let deck_of = |j: i64| -> f64 {
        let (mut up, mut lo) = (f64::MAX, f64::MIN);
        for i in (j - rn).max(0)..=(j + rn).min(nmax) {
            let ds = (sj(i) - sj(j)).abs() * g;
            up = up.min(mm(i) + ds);
            lo = lo.max(mm(i) - ds);
        }
        let d = 0.5 * (up + lo);
        // over water and next to it the deck clears the water (a bridge, approach ramps at
        // the grade limit: never a tunnel under a river)
        let mut lift = f64::MIN;
        for i in (j - 8).max(0)..=(j + 8).min(nmax) {
            let t = at(i);
            if t.water_kind != water::NONE {
                lift = lift.max(lifted(t) - (sj(i) - sj(j)).abs() * g);
            }
        }
        d.max(lift)
    };
    let decks: Vec<f64> = (e0..=e1).map(deck_of).collect();
    let dk = |j: i64| decks[(j.clamp(e0, e1) - e0) as usize];
    let gr = |j: i64| at(j).ground;
    let wet = |j: i64| at(j).water_kind != water::NONE;
    let raw_tunnel = |j: i64| j >= 0 && j < nmax && !wet(j) && !wet(j + 1) && gr(j) - dk(j) > 30.0 && gr(j + 1) - dk(j + 1) > 30.0;
    let mut p = Profile { j0, ..Default::default() };
    for j in j0..=j1 {
        p.deck.push(dk(j));
        p.ground.push(gr(j));
        let br = j < nmax && ((dk(j) - gr(j) > 6.0 && dk(j + 1) - gr(j + 1) > 6.0) || wet(j) || wet(j + 1));
        // tunnels of at least five pieces (200 m; a shorter one stays a deep cutting)
        let tn = raw_tunnel(j) && (-4..=0).any(|o| (o..o + 5).all(|i| raw_tunnel(j + i)));
        p.bridge.push(br && !tn);
        p.tunnel.push(tn);
    }
    Some(p)
}

// ---------------------------------------------------------------- crossings

/// Two routes crossing: the arc lengths on each, the point, the unit directions there.
#[derive(Clone, Copy, Debug)]
pub struct Crossing {
    pub sa: f64,
    pub sb: f64,
    pub p: DVec3,
    pub da: DVec3,
    pub db: DVec3,
}

type XKey = (u64, u64, u64);
static CROSS: RwLock<Option<FxHashMap<XKey, Arc<Vec<Crossing>>>>> = RwLock::new(None);

fn bounds(l: &Line, k0: usize, k1: usize) -> (DVec3, f64) {
    let c = (l.pts[k0] + l.pts[k1]) * 0.5;
    let r = l.pts[k0..=k1].iter().map(|p| (*p - c).length()).fold(0.0, f64::max);
    (c, r)
}

/// The crossings of two routes (memoized per pair; none within 300 m of a route end).
pub fn crossings(w: &World, ka: u64, a: &Line, kb: u64, b: &Line) -> Arc<Vec<Crossing>> {
    let key = (w.cache_key, ka, kb);
    if let Some(v) = CROSS.read().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return v.clone();
    }
    const CH: usize = 24;
    let mut out = Vec::new();
    let (na, nb) = (a.pts.len(), b.pts.len());
    let mut ca = 0;
    while ca + 1 < na {
        let ea = (ca + CH).min(na - 1);
        let (c_a, r_a) = bounds(a, ca, ea);
        let mut cb = 0;
        while cb + 1 < nb {
            let eb = (cb + CH).min(nb - 1);
            let (c_b, r_b) = bounds(b, cb, eb);
            if (c_a - c_b).length() <= r_a + r_b + 1.0 {
                for i in ca..ea {
                    let (p0, p1) = (a.pts[i], a.pts[i + 1]);
                    let up = p0.normalize();
                    let ex = (p1 - p0).normalize_or_zero();
                    let ey = up.cross(ex);
                    let lp = (p1 - p0).length();
                    for k in cb..eb {
                        let (q0, q1) = (b.pts[k], b.pts[k + 1]);
                        let u0 = DVec2::new((q0 - p0).dot(ex), (q0 - p0).dot(ey));
                        let u1 = DVec2::new((q1 - p0).dot(ex), (q1 - p0).dot(ey));
                        if (u0.y > 0.0) == (u1.y > 0.0) {
                            continue;
                        }
                        let t = u0.y / (u0.y - u1.y);
                        let x = u0.x + (u1.x - u0.x) * t;
                        if !(0.0..lp).contains(&x) {
                            continue;
                        }
                        let sa = a.s[i] + x;
                        let sb = b.s[k] + t * (b.s[k + 1] - b.s[k]);
                        if sa < 300.0 || sb < 300.0 || a.len() - sa < 300.0 || b.len() - sb < 300.0 {
                            continue;
                        }
                        out.push(Crossing { sa, sb, p: p0 + ex * x, da: ex, db: (q1 - q0).normalize_or_zero() });
                    }
                }
            }
            cb = eb;
        }
        ca = ea;
    }
    // (a route crossing another several times within 2.5 km: keep the first)
    out.sort_by(|x, y| x.sa.total_cmp(&y.sa));
    out.dedup_by(|y, x| (y.sa - x.sa).abs() < 2_500.0);
    let v = Arc::new(out);
    let mut g = CROSS.write().unwrap();
    let m = g.get_or_insert_with(FxHashMap::default);
    if m.len() > 200_000 {
        m.clear();
    }
    m.insert(key, v.clone());
    v
}

/// A crossing this route passes over (an overpass).
#[derive(Clone, Copy, Debug)]
struct Mod {
    /// arc length of the crossing on this route
    s: f64,
    /// the route under it (index) and the arc length there
    under: usize,
    su: f64,
    /// the deck of the route under it there (NaN: no profile) and its half width
    deck: f64,
    hw: f64,
}

/// Interchange templates.
mod ix {
    pub const NONE: u32 = 0;
    pub const DIAMOND: u32 = 1;
    pub const PARCLO: u32 = 2;
    pub const CLOVERLEAF: u32 = 3;
    pub const TRUMPET: u32 = 4;
    pub const ROUNDABOUT: u32 = 5;
}

/// Is `p` inside a town or larger (its built-up radius)?
pub(crate) fn in_town(w: &World, p: DVec3) -> bool {
    (places::TOWN as usize..places::NLEVELS)
        .any(|l| places::places_near(w, l, p, places::LEVELS[l].radius[1] * 1.15).iter().any(|pl| (pl.center - p).length() < pl.radius))
}

/// Roundabout culture: European-type archetypes, some developed others.
fn roundabouts(arch: u8, dev: f64, h: u64) -> bool {
    matches!(arch, 5 | 6) || (dev > 0.6 && u01k(h, 9) < 0.35)
}

// ---------------------------------------------------------------- the area

pub(crate) struct Routed {
    pub e: Edge,
    pub line: Arc<Line>,
}

/// Vertex spacing (m) of the emitted polylines at a pixel size: 20 m × 2^k ≥ 1.5 gsd.
fn spacing(gsd: f64) -> f64 {
    let k = (gsd * 1.5 / 20.0).max(1.0).log2().floor().clamp(0.0, 12.0);
    20.0 * k.exp2()
}

/// The routes near an area (the graph's edges, spurs to motorways with their crossings, roads and
/// lines to the sites) and whether point evaluations are pending.
/// The graph edges near an area (`area_routes`) and the motorways its towns may have spurs to.
fn graph_edges(w: &World, area: &Area) -> (Vec<Edge>, Vec<Edge>) {
    let gsd = area.gsd;
    let vis = |c: usize| gsd < cls::GSD_MAX[c];
    let margin = 1_500.0;
    let mut edges: Vec<Edge> = Vec::new();
    for n in 0..nw::N {
        if gsd >= net::net_class_max_gsd(n) {
            continue;
        }
        edges.extend(net::edges_near(w, n, area.center, area.radius + margin).into_iter().filter(|e| vis(e.class)));
    }
    let mut mways: Vec<Edge> = Vec::new();
    if vis(cls::TRUNK) {
        mways = net::edges_near(w, nw::ARTERIAL, area.center, area.radius + 25_000.0);
        mways.retain(|e| e.class == cls::MOTORWAY);
    }
    (edges, mways)
}

/// The kit's `prepare` hook: route the graph edges and motorways near a batch of areas, each
/// once and in parallel (longest first), before the areas' features are built in parallel (each
/// would route the edges it shares with its neighbours itself).
pub fn prepare(w: &World, areas: &[Area]) {
    use rayon::prelude::*;
    if w.cfg.landuse.roads <= 0.0 || w.cfg.landuse.towns <= 0.0 {
        return;
    }
    let lists: Vec<Vec<Edge>> = areas
        .par_iter()
        .filter(|a| a.gsd < cls::GSD_MAX[cls::POWER])
        .map(|a| {
            let (mut e, m) = graph_edges(w, a);
            e.extend(m);
            e
        })
        .collect();
    let mut all: Vec<Edge> = lists.concat();
    all.sort_unstable_by_key(|e| e.key);
    all.dedup_by_key(|e| e.key);
    all.sort_by(|a, b| b.len().total_cmp(&a.len()).then(a.key.cmp(&b.key)));
    all.par_iter().for_each(|e| {
        let mut pts = crate::features::CpuPoints(w);
        let mut ctx = HostCtx { world: w, points: &mut pts };
        let _ = routed(&mut ctx, e);
    });
}

pub(crate) fn area_routes(ctx: &mut HostCtx, area: &Area) -> (Vec<Routed>, Vec<(Edge, u64, Crossing)>, bool) {
    let w = ctx.world;
    let gsd = area.gsd;
    let vis = |c: usize| gsd < cls::GSD_MAX[c];
    let margin = 1_500.0;
    // ---- the graph edges near the area
    let (mut edges, mways) = graph_edges(w, area);
    let mut pending = false;
    // ---- spurs from towns to passing motorways
    let mut spurs: Vec<(Edge, u64, Crossing)> = Vec::new();
    if vis(cls::TRUNK) {
        for m in &mways {
            let ml = match routed(ctx, m) {
                Some(Some(l)) => l,
                Some(None) => continue,
                None => {
                    pending = true;
                    continue;
                }
            };
            for l in places::TOWN as usize..=places::CITY as usize {
                for pl in places::places_near(w, l, area.center, area.radius + 14_000.0) {
                    if pl.id == m.a.id || pl.id == m.b.id || pl.parent != 0 {
                        continue;
                    }
                    let (d, s) = ml.dist(pl.center);
                    if d < 0.7 * pl.radius || d > 1.5 * pl.radius + 4_000.0 || s < 2_000.0 || ml.len() - s < 2_000.0 {
                        continue;
                    }
                    let (foot, dir) = ml.at(s);
                    let side = if (pl.center - foot).dot(foot.normalize().cross(dir)) > 0.0 { 1.0 } else { -1.0 };
                    let hw = half_width(cls::MOTORWAY, 2);
                    let stem = route::surface(w, foot + foot.normalize().cross(dir) * side * (hw + 220.0));
                    let site = Node { id: mix64(pl.id ^ m.key ^ 0x5B0), p: stem, level: net::TRACK_END, pop: 0.0, radius: 0.0, dev: pl.dev, arch: pl.arch };
                    let e = Edge::new(site, Node::of(&pl), cls::TRUNK, nw::REGIONAL);
                    if net::chord_dist(e.a.p, e.b.p, area.center) < area.radius + route::max_deviation(cls::TRUNK, e.len()) + margin + 1_500.0 {
                        spurs.push((e, m.key, Crossing { sa: s, sb: 0.0, p: foot, da: dir, db: (pl.center - foot).normalize_or_zero() }));
                        edges.push(e);
                    }
                }
            }
        }
    }
    // ---- roads, rail spurs and power lines to the sites
    site_edges(ctx, area, &mut edges, &mut pending);
    edges.sort_unstable_by_key(|e| e.key);
    edges.dedup_by_key(|e| e.key);
    let mut routes: Vec<Routed> = Vec::new();
    for e in &edges {
        match routed(ctx, e) {
            Some(Some(line)) => routes.push(Routed { e: *e, line }),
            Some(None) => {}
            None => pending = true,
        }
    }
    (routes, spurs, pending)
}

/// The kit's features for an area.
pub fn features(ctx: &mut HostCtx, area: &Area) -> Features {
    let mut f = Features::default();
    let w = ctx.world;
    if w.cfg.landuse.roads <= 0.0 || w.cfg.landuse.towns <= 0.0 || area.gsd >= cls::GSD_MAX[cls::POWER] {
        return f;
    }
    let gsd = area.gsd;
    let vis = |c: usize| gsd < cls::GSD_MAX[c];
    let margin = 1_500.0;
    let (routes, spurs, mut pending) = area_routes(ctx, area);
    // ---- crossings among the routes near the area
    let mut mods: Vec<Vec<Mod>> = vec![Vec::new(); routes.len()];
    let mut ixs: Vec<(u32, usize, usize, Crossing)> = Vec::new();
    let near = |p: DVec3, r: f64| (p - area.center).length() <= area.radius + r;
    for i in 0..routes.len() {
        for j in 0..routes.len() {
            let (a, b) = (&routes[i], &routes[j]);
            if i == j || a.e.class == cls::POWER || b.e.class == cls::POWER {
                continue;
            }
            // (a, b) with a the higher rank (ties: lower key)
            if rank(a.e.class) < rank(b.e.class) || (rank(a.e.class) == rank(b.e.class) && a.e.key > b.e.key) {
                continue;
            }
            let (ca, cb) = (a.e.class, b.e.class);
            let grade_sep = ca == cls::MOTORWAY || (ca == cls::RAIL && matches!(cb, cls::TRUNK | cls::SECONDARY | cls::MOTORWAY));
            let at_grade = matches!(ca, cls::TRUNK | cls::SECONDARY) && matches!(cb, cls::TRUNK | cls::SECONDARY);
            if !grade_sep && !at_grade {
                continue;
            }
            if net::chord_dist(a.e.a.p, a.e.b.p, area.center) > area.radius + route::max_deviation(ca, a.e.len()) + margin
                || net::chord_dist(b.e.a.p, b.e.b.p, area.center) > area.radius + route::max_deviation(cb, b.e.len()) + margin
            {
                continue;
            }
            for x in crossings(w, a.e.key, &a.line, b.e.key, &b.line).iter() {
                if !near(x.p, 900.0) {
                    continue;
                }
                // not at a shared place (inside places the town draws the junction)
                let inside = [a.e.a, a.e.b, b.e.a, b.e.b].iter().any(|n| n.level <= places::METRO && (x.p - n.p).length() < (1.1 * n.radius).max(300.0));
                if inside {
                    continue;
                }
                if grade_sep {
                    mods[j].push(Mod { s: x.sb, under: i, su: x.sa, deck: f64::NAN, hw: half_width(ca, 2) });
                    let t = if ca == cls::MOTORWAY {
                        match cb {
                            cls::MOTORWAY => ix::CLOVERLEAF,
                            cls::TRUNK => ix::PARCLO,
                            cls::SECONDARY => ix::DIAMOND,
                            _ => ix::NONE,
                        }
                    } else {
                        ix::NONE
                    };
                    // (no ramps inside towns: their streets are the junctions)
                    if t != ix::NONE && vis(cls::RAMP) && !in_town(w, x.p) {
                        ixs.push((t, i, j, *x));
                    }
                } else if roundabouts(a.e.a.arch, a.e.a.dev.min(a.e.b.dev), a.e.key ^ b.e.key) {
                    ixs.push((ix::ROUNDABOUT, i, j, *x));
                }
            }
        }
    }
    // spurs end in trumpets on their motorway
    for (e, mk, x) in &spurs {
        let k = routes.iter().position(|r| r.e.key == e.key);
        let i = routes.iter().position(|r| r.e.key == *mk);
        if let (Some(k), Some(i)) = (k, i) {
            if vis(cls::RAMP) && near(x.p, 900.0) {
                ixs.push((ix::TRUMPET, i, k, *x));
            }
        }
    }
    // ---- segments
    let prof_area = gsd < PROFILE_GSD;
    for (k, r) in routes.iter().enumerate() {
        // the decks of the routes this one passes over
        let mut md = mods[k].clone();
        if prof_area {
            for m in md.iter_mut() {
                let u = &routes[m.under];
                if profiled(u.e.class) {
                    match profile(ctx, &u.e, &u.line, m.su - 1.0, m.su + 1.0) {
                        Some(p) => m.deck = p.deck_at(m.su),
                        None => pending = true,
                    }
                }
            }
        }
        if !emit(ctx, area, &r.e, &r.line, prof_area, &md, &mut f) {
            pending = true;
        }
    }
    // ---- interchanges, roundabouts
    for &(t, i, j, x) in &ixs {
        interchange(ctx, area, t, i, j, &x, &routes, prof_area, &mut f, &mut pending);
    }
    // ---- stations, yards, town-edge roundabouts
    stations(ctx, area, &routes, &mut f);
    for r in &routes {
        if matches!(r.e.class, cls::TRUNK | cls::SECONDARY) {
            for (node, at_a) in [(r.e.a, true), (r.e.b, false)] {
                if node.level >= places::TOWN && node.level <= places::METRO && roundabouts(node.arch, node.dev, node.id) {
                    let s = if at_a { edge_entry(&r.line, node.p, 0.95 * node.radius, true) } else { edge_entry(&r.line, node.p, 0.95 * node.radius, false) };
                    if let Some(s) = s {
                        let (p, d) = r.line.at(s);
                        if near(p, 100.0) {
                            f.stamps.push(roundabout(w, p, d, r.e.class, r.e.key));
                        }
                    }
                }
            }
        }
    }
    // (pending: the GPU host re-runs us once its point evaluations are in)
    let _ = pending;
    f
}

/// Arc length where a route leaves (from its start) or enters (toward its end) the disc
/// (c, r).
fn edge_entry(l: &Line, c: DVec3, r: f64, from_start: bool) -> Option<f64> {
    let n = l.pts.len();
    if from_start {
        (0..n).find(|&k| (l.pts[k] - c).length() > r).map(|k| l.s[k])
    } else {
        (0..n).rev().find(|&k| (l.pts[k] - c).length() > r).map(|k| l.s[k])
    }
}

/// A deck height for the GPU tables (`NO_DECK`: on the ground; no NaN in WGSL).
fn deck32(d: f64) -> f32 {
    if d.is_finite() {
        d as f32
    } else {
        super::NO_DECK
    }
}

fn seed24(h: u64) -> f32 {
    (h & 0xFF_FFFF) as f32
}

/// Car density 0..1 and lamp spacing (m; 0: unlit) of a stretch: by class and development,
/// more in towns, the more the larger the town (`size`: its level 0–4).
fn traffic(class: usize, urban: bool, size: u8, dev: f64, near_city: bool) -> (f64, f64) {
    let base = [0.24, 0.18, 0.1, 0.04, 0.0, 0.12, 0.0, 0.0][class];
    let town = if urban { 1.2 + 0.3 * size as f64 } else { 1.0 };
    let dens = (base * (0.45 + 0.8 * dev) * town).min(0.9);
    let lit = match class {
        cls::MOTORWAY => dev > 0.75 || near_city || urban,
        cls::RAMP => dev > 0.3,
        cls::TRUNK | cls::SECONDARY | cls::LOCAL => urban && dev > 0.15,
        _ => false,
    };
    (
        dens,
        if lit {
            if class == cls::MOTORWAY || class == cls::RAMP {
                40.0
            } else {
                35.0
            }
        } else {
            0.0
        },
    )
}

/// Emit the segments of a route near the area (false: point samples pending).
fn emit(ctx: &mut HostCtx, area: &Area, e: &Edge, line: &Line, prof_area: bool, mods: &[Mod], f: &mut Features) -> bool {
    let class = e.class;
    let lanes: u32 = match class {
        cls::MOTORWAY => 2 + (e.a.dev.min(e.b.dev) > 0.6 && e.a.pop.min(e.b.pop) > 1.0e6) as u32,
        cls::RAIL => 1 + (e.a.dev.min(e.b.dev) > 0.45 && e.a.level.min(e.b.level) >= places::CITY) as u32,
        _ => 1,
    };
    let (hw, med) = if class == cls::POWER {
        (if e.net == nw::GRID { 22.0 } else { 14.0 }, 0.0)
    } else {
        (half_width(class, lanes), if class == cls::MOTORWAY { 1.6 } else { 0.0 })
    };
    let vg = verge(class);
    let reach = if class == cls::POWER { hw.max(90.0) } else { hw + vg + 40.0 };
    let sp = spacing(area.gsd);
    let n = line.pts.len();
    if n < 2 {
        return true;
    }
    // the stretch near the area
    let lim = area.radius + reach + sp + 2.0;
    let mut lo = usize::MAX;
    let mut hi = 0;
    for k in 0..n - 1 {
        if net::chord_dist(line.pts[k], line.pts[k + 1], area.center) <= lim {
            lo = lo.min(k);
            hi = hi.max(k + 1);
        }
    }
    if lo == usize::MAX {
        return true;
    }
    let (s_lo, s_hi) = (line.s[lo], line.s[hi]);
    let prof = prof_area && profiled(class);
    let pr = if prof {
        match profile(ctx, e, line, s_lo, s_hi) {
            Some(p) => Some(p),
            None => return false,
        }
    } else {
        None
    };
    // vertices: the route's, one per spacing bucket (and the ends of the stretch)
    let mut vs: Vec<f64> = Vec::new();
    let mut last = i64::MIN;
    for k in lo..=hi {
        let b = (line.s[k] / sp).floor() as i64;
        if b != last || k == hi || k == lo {
            vs.push(line.s[k]);
            last = b;
        }
    }
    // flag changes of the profile and the overpass approaches
    if let Some(p) = &pr {
        for j in p.j0..p.j0 + p.deck.len() as i64 - 1 {
            let s = j as f64 * DS;
            if s > s_lo && s < s_hi && j > p.j0 && p.piece(s - 1.0) != p.piece(s + 1.0) {
                vs.push(s);
            }
        }
    }
    for m in mods {
        for ds in [-260.0, -120.0, -(m.hw + 15.0), 0.0, m.hw + 15.0, 120.0, 260.0] {
            let s = m.s + ds;
            if s > s_lo && s < s_hi {
                vs.push(s);
            }
        }
    }
    vs.sort_by(|a, b| a.total_cmp(b));
    vs.dedup_by(|b, a| (*b - *a).abs() < 0.5);
    let urban_at = |p: DVec3| [e.a, e.b].iter().any(|nd| nd.level <= places::METRO && (p - nd.p).length() < 0.92 * nd.radius);
    let near_city = [e.a, e.b]
        .iter()
        .any(|nd| nd.level >= places::CITY && nd.level <= places::METRO && (line.at(0.5 * (s_lo + s_hi)).0 - nd.p).length() < nd.radius + 4_000.0);
    let dev = e.a.dev.min(e.b.dev);
    let paved = match class {
        cls::LOCAL => dev > 0.3,
        cls::SECONDARY => dev > 0.1,
        cls::TRACK | cls::POWER | cls::RAIL => false,
        _ => true,
    };
    let deck_at = |s: f64| -> (f64, bool) {
        let mut d = pr.as_ref().map_or(f64::NAN, |p| p.deck_at(s));
        let mut fill = false;
        for m in mods {
            if m.deck.is_finite() {
                let h = m.deck + CLEAR - RAMP_G * ((s - m.s).abs() - (m.hw + 15.0)).max(0.0);
                if (s - m.s).abs() < 260.0 && (d.is_nan() || d < h) {
                    if !d.is_finite() {
                        fill = true;
                    }
                    d = h;
                }
            }
        }
        (d, fill)
    };
    let mut prev_tunnel = None;
    for k in 0..vs.len().saturating_sub(1) {
        let (s0, s1) = (vs[k], vs[k + 1]);
        let (pa, _) = line.at(s0);
        let (pb, _) = line.at(s1);
        let mid = 0.5 * (s0 + s1);
        let (bridge, tunnel) = pr.as_ref().map_or((false, false), |p| p.piece(mid));
        // portals where a tunnel starts or ends
        if prof && class != cls::POWER {
            if let Some(pt) = prev_tunnel {
                if pt != tunnel {
                    let (pp, d) = line.at(s0);
                    let into = if tunnel { d } else { -d };
                    if (pp - area.center).length() < area.radius + 60.0 {
                        f.stamps.push(portal(pp, into, deck_at(s0).0, hw, class, e.key ^ (s0 as u64)));
                    }
                }
            }
            prev_tunnel = Some(tunnel);
        }
        if net::chord_dist(pa, pb, area.center) > area.radius + reach + 2.0 {
            continue;
        }
        let over = mods.iter().any(|m| (mid - m.s).abs() < m.hw + 15.0);
        let urban = urban_at(0.5 * (pa + pb));
        let size = [e.a, e.b].iter().filter(|nd| nd.level <= places::METRO).map(|nd| nd.level).max().unwrap_or(0);
        let (dens, lamp) = traffic(class, urban, size, dev, near_city);
        let (da, fa) = deck_at(s0);
        let (db, fb) = deck_at(s1);
        let mut fl = 0u32;
        if bridge || (over && da.is_finite()) {
            fl |= flag::BRIDGE;
        }
        if tunnel {
            fl |= flag::TUNNEL;
        }
        if urban {
            fl |= flag::URBAN;
        }
        if paved {
            fl |= flag::PAVED;
        }
        if class == cls::RAIL && dev > 0.55 {
            fl |= flag::ELECTRIC;
        }
        if over {
            fl |= flag::OVER;
        }
        if fa || fb {
            fl |= flag::FILL;
        }
        fl |= lanes << flag::LANES_SHIFT;
        let wrap = (s0 / S_WRAP).floor();
        let (v1, v2) = if class == cls::POWER {
            if e.net == nw::GRID {
                (350.0, 46.0)
            } else {
                (300.0, 30.0)
            }
        } else {
            (med, vg)
        };
        f.segs.push(FSeg {
            a: pa,
            b: pb,
            ha: dens as f32,
            hb: lamp as f32,
            kind: cls::KIND0 + class as u32,
            class: cls::LANDCOVER[class],
            hw: reach as f32,
            s0: (s0 - wrap * S_WRAP) as f32,
            deck_a: deck32(da),
            deck_b: deck32(db),
            flags: fl,
            v: [hw as f32, v1 as f32, v2 as f32, seed24(mix64(e.key ^ (wrap as u64).wrapping_mul(0x9E37_79B9)))],
        });
    }
    true
}

fn portal(p: DVec3, into: DVec3, deck: f64, hw: f64, class: usize, h: u64) -> FStamp {
    let up = p.normalize();
    let ex = super::f32v(into);
    let ey = super::f32v(up.cross(into).normalize_or_zero());
    FStamp {
        center: p,
        ex,
        ey,
        half: [16.0, (hw + 8.0) as f32],
        template: tpl::PORTAL,
        h: deck32(deck),
        v: [hw as f32, class as f32, seed24(mix64(h)), 0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

fn roundabout(w: &World, p: DVec3, d: DVec3, class: usize, h: u64) -> FStamp {
    let _ = w;
    let up = p.normalize();
    let ex = super::f32v(d);
    let ey = super::f32v(up.cross(d).normalize_or_zero());
    let r = if class == cls::TRUNK { 24.0 } else { 17.0 };
    FStamp {
        center: p,
        ex,
        ey,
        half: [r + 10.0, r + 10.0],
        template: tpl::ROUNDABOUT,
        h: 0.0,
        v: [r, half_width(class, 1) as f32, seed24(mix64(h ^ 0x80D)), 0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

/// A ramp polyline as segments. Its deck is a lower bound (`flag::FILL`) falling from the
/// overpass height `over` at the crossing `x` with the approach grade (NaN: on the ground).
#[allow(clippy::too_many_arguments)]
fn ramp(area: &Area, pts: &[DVec3], x: DVec3, over: f64, lit: bool, dev: f64, h: u64, f: &mut Features) {
    let l = Line::new(pts.to_vec());
    let hw = half_width(cls::RAMP, 1);
    let reach = hw + verge(cls::RAMP) + 40.0;
    let deck = |p: DVec3| if over.is_finite() { over - RAMP_G * ((p - x).length() - 40.0).max(0.0) } else { f64::NAN };
    for k in 0..pts.len() - 1 {
        if net::chord_dist(pts[k], pts[k + 1], area.center) > area.radius + reach + 2.0 {
            continue;
        }
        let (da, db) = (deck(pts[k]), deck(pts[k + 1]));
        let mut fl = flag::PAVED | (1 << flag::LANES_SHIFT);
        if da.is_finite() {
            fl |= flag::FILL;
        }
        let (dens, lamp) = traffic(cls::RAMP, false, 0, dev, lit);
        f.segs.push(FSeg {
            a: pts[k],
            b: pts[k + 1],
            ha: dens as f32,
            hb: if lit { lamp as f32 } else { 0.0 },
            kind: cls::KIND0 + cls::RAMP as u32,
            class: cls::LANDCOVER[cls::RAMP],
            hw: reach as f32,
            s0: (l.s[k] % S_WRAP) as f32,
            deck_a: deck32(da),
            deck_b: deck32(db),
            flags: fl,
            v: [hw as f32, 0.0, verge(cls::RAMP) as f32, seed24(mix64(h))],
        });
    }
}

/// A cubic Bézier through 3D control points, projected to the surface.
fn bezier(w: &World, c: [DVec3; 4], n: usize) -> Vec<DVec3> {
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            let u = 1.0 - t;
            route::surface(w, c[0] * (u * u * u) + c[1] * (3.0 * u * u * t) + c[2] * (3.0 * u * t * t) + c[3] * (t * t * t))
        })
        .collect()
}

/// The 270° loop around `c` (radius `r`) from the tangent point in direction `from` to the one
/// in direction `to` (unit tangent-plane vectors from the centre), the long way round.
fn loop270(w: &World, c: DVec3, r: f64, from: DVec3, to: DVec3, n: usize) -> Vec<DVec3> {
    let up = c.normalize();
    let ex = from.normalize_or_zero();
    let ey = up.cross(ex);
    let d = to.dot(ey).atan2(to.dot(ex));
    let da = d - d.signum() * std::f64::consts::TAU;
    (0..=n)
        .map(|i| {
            let a = da * i as f64 / n as f64;
            route::surface(w, c + (ex * a.cos() + ey * a.sin()) * r)
        })
        .collect()
}

/// A point of a route at arc length `s`, offset `lat` m to its left; its direction.
fn on(l: &Line, s: f64, lat: f64) -> (DVec3, DVec3) {
    let (p, d) = l.at(s);
    let left = p.normalize().cross(d).normalize_or_zero();
    (p + left * lat, d)
}

/// Join the end `q` of a ramp to the nearest point of route `l` (a short connector).
fn join(l: &Line, q: DVec3) -> Option<DVec3> {
    let (d, s) = l.dist(q);
    (d > 2.0).then(|| l.at(s).0)
}

/// The ramps of an interchange (or a roundabout) at the crossing `x` of the major route `i` and
/// the route `j` (for a trumpet: the spur meeting motorway `i`).
#[allow(clippy::too_many_arguments)]
fn interchange(ctx: &mut HostCtx, area: &Area, t: u32, i: usize, j: usize, x: &Crossing, routes: &[Routed], prof: bool, f: &mut Features, pending: &mut bool) {
    let w = ctx.world;
    let major = &routes[i];
    let minor = &routes[j];
    let dev = minor.e.a.dev.min(minor.e.b.dev);
    let h0 = minor.e.key ^ x.sa.to_bits();
    if t == ix::ROUNDABOUT {
        if (x.p - area.center).length() < area.radius + 60.0 {
            f.stamps.push(roundabout(w, x.p, x.db, major.e.class.max(minor.e.class), h0));
        }
        return;
    }
    let (lm, lx) = (&major.line, &minor.line);
    let hw_m = half_width(cls::MOTORWAY, 2);
    let hw_x = half_width(minor.e.class, 1);
    // the overpass deck above the major's
    let mut over = f64::NAN;
    if prof {
        match profile(ctx, &major.e, lm, x.sa - 1.0, x.sa + 1.0) {
            Some(p) => over = p.deck_at(x.sa) + CLEAR,
            None => {
                *pending = true;
                return;
            }
        }
    }
    let lit = dev > 0.3;
    let up = x.p.normalize();
    let left_m = up.cross(x.da).normalize_or_zero();
    let mut emit_ramp = |pts: Vec<DVec3>, k: u64| ramp(area, &pts, x.p, over, lit, dev, h0 ^ k, f);
    let r0 = 62.0;
    match t {
        ix::DIAMOND | ix::PARCLO | ix::CLOVERLEAF => {
            for (qi, &(sa, sb)) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)].iter().enumerate() {
                // along the minor road toward side sb of the major; its side facing sa
                let gx = if x.db.dot(left_m) * sb > 0.0 { 1.0 } else { -1.0 };
                let left_x = up.cross(x.db).normalize_or_zero();
                let side_x = if left_x.dot(x.da * sa) > 0.0 { 1.0 } else { -1.0 };
                let loop_here = t == ix::CLOVERLEAF || (t == ix::PARCLO && sa > 0.0);
                let far = if t == ix::DIAMOND { 0.0 } else { 2.0 * r0 + 60.0 };
                // the outer ramp (diamond / directional)
                let (p1, d1) = on(lm, x.sa + sa * (420.0 + far), sb * (hw_m + 6.0));
                let (p2, d2) = on(lx, x.sb + gx * (hw_m + 70.0 + far), side_x * (hw_x + 5.0));
                emit_ramp(bezier(w, [p1, p1 - d1 * (sa * 220.0), p2 + d2 * (gx * 120.0), p2], 18), qi as u64);
                if loop_here {
                    // a loop in the quadrant, touching the major (side sb) and the minor (side sa)
                    let cc = x.p + x.da * (sa * (hw_x + 2.0 + r0)) + left_m * (sb * (hw_m + 2.0 + r0));
                    let mut lp = loop270(w, cc, r0, -left_m * sb, -x.da * sa, 28);
                    if let Some(q) = join(lx, *lp.last().unwrap()) {
                        lp.push(q);
                    }
                    if let Some(q) = join(lm, lp[0]) {
                        lp.insert(0, q);
                    }
                    emit_ramp(lp, 10 + qi as u64);
                }
            }
        }
        ix::TRUMPET => {
            // the spur's stem ends on side sb of the motorway: a ramp and a loop to it, and a
            // flyover to the far carriageway
            let stem = lx.pts[0];
            let sb = if (stem - x.p).dot(left_m) > 0.0 { 1.0 } else { -1.0 };
            let (p1, d1) = on(lm, x.sa - 480.0, sb * (hw_m + 6.0));
            let toward = (x.p - stem).normalize_or_zero();
            emit_ramp(bezier(w, [p1, p1 + d1 * 220.0, stem + toward * 120.0 - x.da * 80.0, stem], 18), 1);
            let cc = x.p + x.da * (hw_x + 2.0 + r0) + left_m * (sb * (hw_m + 2.0 + r0));
            let mut lp = loop270(w, cc, r0, -left_m * sb, -x.da, 28);
            lp.push(stem);
            if let Some(q) = join(lm, lp[0]) {
                lp.insert(0, q);
            }
            emit_ramp(lp, 2);
            let (p3, d3) = on(lm, x.sa + 420.0, -sb * (hw_m + 6.0));
            emit_ramp(bezier(w, [stem, stem + toward * 160.0 + x.da * 60.0, p3 - d3 * 240.0, p3], 20), 3);
        }
        _ => {}
    }
}

/// Stations where a railway ends at or passes through a town or city, yards at cities.
fn stations(ctx: &mut HostCtx, area: &Area, routes: &[Routed], f: &mut Features) {
    let w = ctx.world;
    if area.gsd >= cls::GSD_MAX[cls::RAIL] {
        return;
    }
    let mut seen: Vec<u64> = Vec::new();
    for r in routes.iter().filter(|r| r.e.class == cls::RAIL) {
        // the end stations (once per node: on its lowest-key rail edge)
        for (nd, at_a) in [(r.e.a, true), (r.e.b, false)] {
            if nd.level > places::METRO || seen.contains(&nd.id) || (nd.p - area.center).length() > area.radius + 3_000.0 {
                continue;
            }
            let first = net::neighbours(w, nw::RAIL, &nd)
                .iter()
                .map(|o| Edge::new(if nd.id < o.id { nd } else { *o }, if nd.id < o.id { *o } else { nd }, cls::RAIL, nw::RAIL).key)
                .min();
            if first != Some(r.e.key) {
                continue;
            }
            seen.push(nd.id);
            let s = if at_a { 0.18 * nd.radius } else { r.line.len() - 0.18 * nd.radius };
            let (p, d) = r.line.at(s);
            f.stamps.push(station(p, d, nd.level, nd.pop, nd.id));
            if nd.level >= places::CITY {
                let sy = if at_a { 0.6 * nd.radius } else { r.line.len() - 0.6 * nd.radius };
                let (p, d) = r.line.at(sy);
                f.stamps.push(yard(p, d, nd.level, nd.id));
            }
        }
        // a port's rail spur ends in a yard by the quay
        if r.e.a.level == net::TRACK_END {
            let (p, d) = r.line.at((0.4 * r.line.len()).min(700.0));
            if (p - area.center).length() < area.radius + 1_500.0 {
                f.stamps.push(yard(p, d, places::CITY, r.e.key));
            }
            continue;
        }
        // through stations: towns within half their radius of the line
        if r.line.len() < 2.0 {
            continue;
        }
        for pl in places::places_near(w, places::TOWN as usize, area.center, area.radius + 4_000.0) {
            if pl.id == r.e.a.id || pl.id == r.e.b.id {
                continue;
            }
            let (d, s) = r.line.dist(pl.center);
            if d < 0.5 * pl.radius && s > 500.0 && r.line.len() - s > 500.0 {
                let (p, dir) = r.line.at(s);
                f.stamps.push(station(p, dir, places::TOWN, pl.population, pl.id ^ r.e.key));
            }
        }
    }
}

fn station(p: DVec3, d: DVec3, level: u8, pop: f64, h: u64) -> FStamp {
    let up = p.normalize();
    let tracks = match level {
        places::METRO => 8.0,
        places::CITY => 5.0 + (pop > 1.0e6) as u32 as f64,
        _ => 3.0,
    };
    let len = 160.0 + 60.0 * level as f64 + 60.0 * u01k(h, 1);
    FStamp {
        center: p,
        ex: super::f32v(d),
        ey: super::f32v(up.cross(d).normalize_or_zero()),
        half: [(0.5 * len + 30.0) as f32, (tracks * 5.5 + 30.0) as f32],
        template: tpl::STATION,
        h: 0.0,
        v: [tracks as f32, len as f32, seed24(mix64(h ^ 0x57A)), if u01k(h, 2) < 0.5 { 1.0 } else { -1.0 }, 0.0, 0.0, 0.0, 0.0],
    }
}

fn yard(p: DVec3, d: DVec3, level: u8, h: u64) -> FStamp {
    let up = p.normalize();
    let tracks = (8.0 + 4.0 * (level as f64 - 3.0) + (6.0 * u01k(h, 3)).floor()).min(18.0);
    let len = 900.0 + 500.0 * u01k(h, 4) + 300.0 * (level as f64 - 3.0);
    FStamp {
        center: p,
        ex: super::f32v(d),
        ey: super::f32v(up.cross(d).normalize_or_zero()),
        half: [(0.5 * len) as f32, (tracks * 2.6 + 12.0) as f32],
        template: tpl::YARD,
        h: 0.0,
        v: [tracks as f32, len as f32, seed24(mix64(h ^ 0x7A2D)), if u01k(h, 5) < 0.5 { 1.0 } else { -1.0 }, 0.0, 0.0, 0.0, 0.0],
    }
}

// ---------------------------------------------------------------- sites

/// The nearest place of levels `l0..=4` to `p` within `r` (by distance, then id).
fn nearest_place(w: &World, p: DVec3, l0: u8, r: f64) -> Option<Node> {
    let mut best: Option<(f64, Node)> = None;
    for l in l0 as usize..places::NLEVELS {
        for pl in places::places_near(w, l, p, r) {
            let d = (pl.center - p).length();
            if best.as_ref().is_none_or(|b| d < b.0 || (d == b.0 && pl.id < b.1.id)) {
                best = Some((d, Node::of(&pl)));
            }
        }
    }
    best.map(|b| b.1)
}

/// Where the road from `toward` meets a site's box (its edge, not its centre: roads stop at the
/// airport fence, the quarry rim).
fn box_edge(w: &World, st: &FStamp, toward: DVec3) -> DVec3 {
    let d = toward - st.center;
    let (u, v) = (d.dot(st.ex), d.dot(st.ey));
    let t = (st.half[0] as f64 / u.abs().max(1e-9)).min(st.half[1] as f64 / v.abs().max(1e-9)).min(1.0);
    route::surface(w, st.center + (st.ex * u + st.ey * v) * t)
}

/// The edge of a site toward `toward` (its first stamp; else its centre).
fn site_edge(w: &World, s: &sites::Site, toward: DVec3) -> DVec3 {
    s.stamps.first().map_or(s.center, |st| box_edge(w, st, toward))
}

fn site_node(id: u64, p: DVec3, dev: f64) -> Node {
    Node { id, p, level: net::TRACK_END, pop: 0.0, radius: 0.0, dev, arch: 0 }
}

/// Roads to airports, ports, dams, quarries, solar and wind farms; rail spurs to ports; power
/// lines from dams, solar and wind farms to the nearest town or city.
fn site_edges(ctx: &mut HostCtx, area: &Area, edges: &mut Vec<Edge>, pending: &mut bool) {
    let w = ctx.world;
    let gsd = area.gsd;
    let vis = |c: usize| gsd < cls::GSD_MAX[c];
    let r_area = area.radius + 1_500.0;
    let push = |e: Edge, edges: &mut Vec<Edge>| {
        if vis(e.class) && net::chord_dist(e.a.p, e.b.p, area.center) < r_area + route::max_deviation(e.class, e.len()) {
            edges.push(e);
        }
    };
    let power_to = |w: &World, p: DVec3| nearest_place(w, p, places::TOWN, 55_000.0).filter(|n| n.pop > 25_000.0 || n.level >= places::CITY);
    let mut towns: Vec<Place2> = Vec::new();
    for l in places::TOWN as usize..places::NLEVELS {
        for pl in places::places_near(w, l, area.center, area.radius + 60_000.0) {
            towns.push(pl);
        }
    }
    for pl in &towns {
        let d = (pl.center - area.center).length();
        let me = Node::of(pl);
        // airports (trunk road), ports (secondary road, rail spur for cities)
        if pl.airport && d < r_area + pl.radius + 20_000.0 && vis(cls::SECONDARY) {
            match sites::airport(ctx, pl) {
                None => *pending = true,
                Some(Some(s)) => {
                    let c = if pl.level >= places::CITY { cls::TRUNK } else { cls::SECONDARY };
                    push(Edge::new(site_node(mix64(pl.id ^ 0xA1F0), site_edge(w, &s, pl.center), pl.dev), me, c, nw::REGIONAL), edges);
                }
                Some(None) => {}
            }
        }
        if pl.port && pl.parent == 0 && d < r_area + pl.radius + 16_000.0 && vis(cls::SECONDARY) {
            match sites::port(ctx, pl) {
                None => *pending = true,
                Some(Some(s)) => {
                    let a = site_node(mix64(pl.id ^ 0x0009_0270), site_edge(w, &s, pl.center), pl.dev);
                    push(Edge::new(a, me, cls::SECONDARY, nw::LOCAL), edges);
                    if pl.level >= places::CITY && pl.dev > 0.2 {
                        push(Edge::new(site_node(mix64(pl.id ^ 0x0009_0271), s.center, pl.dev), me, cls::RAIL, nw::RAIL), edges);
                    }
                }
                Some(None) => {}
            }
        }
        // solar farms: a local road and a line to the town
        if d < r_area + pl.radius + 8_000.0 {
            for k in 0..2 {
                match sites::solar(ctx, pl, k) {
                    None => *pending = true,
                    Some(Some(s)) => {
                        let a = site_node(mix64(pl.id ^ 0x501A ^ ((k as u64) << 40)), site_edge(w, &s, pl.center), pl.dev);
                        push(Edge::new(a, me, cls::LOCAL, nw::HAMLET), edges);
                        push(Edge::new(site_node(a.id ^ 0x90, a.p, pl.dev), me, cls::POWER, nw::HAMLET), edges);
                    }
                    Some(None) => {}
                }
            }
        }
        // town dams: a local road from the town, a line to the nearest larger town
        if pl.parent == 0 && d < r_area + 60_000.0 && u01k(pl.id, 50) < if pl.level >= places::CITY { 0.5 } else { 0.15 } && gsd < cls::GSD_MAX[cls::POWER] {
            {
                match sites::dam(ctx, mix64(pl.id ^ 0xDA4), pl.center, pl.radius + 2500.0, 22_000.0) {
                    None => *pending = true,
                    Some(Some(s)) => {
                        if let Some(dam) = s.segs.iter().find(|g| g.kind == sites::tpl::DAM) {
                            let a = site_node(mix64(pl.id ^ 0xDA40), dam.a, pl.dev);
                            push(Edge::new(a, me, cls::LOCAL, nw::HAMLET), edges);
                            if let Some(t) = power_to(w, dam.b) {
                                push(Edge::new(site_node(a.id ^ 0x90, dam.b, pl.dev), t, cls::POWER, nw::HAMLET), edges);
                            }
                        }
                    }
                    Some(None) => {}
                }
            }
        }
    }
    // mountain dams, quarries, wind farms (lattice sites)
    if gsd < cls::GSD_MAX[cls::POWER] {
        for c in sites::lattice_cells(w, sites::MTN_DAM_CELL, area.center, r_area + 60_000.0) {
            if let Some((id, p)) = sites::lattice_site(w, 0xDA7, sites::MTN_DAM_CELL, c) {
                if u01k(id, 1) < 0.2 && w.smooth_relief(p, 3000.0).1 > 350.0 && w.atlas().sample(p.normalize()).glaciation < 0.4 {
                    match sites::dam(ctx, id, p, 0.0, 20_000.0) {
                        None => *pending = true,
                        Some(Some(s)) => {
                            if let Some(dam) = s.segs.iter().find(|g| g.kind == sites::tpl::DAM) {
                                if let Some(v) = nearest_place(w, dam.a, places::VILLAGE, 25_000.0) {
                                    push(Edge::new(site_node(mix64(id ^ 0xDA41), dam.a, v.dev), v, cls::LOCAL, nw::HAMLET), edges);
                                }
                                if let Some(t) = power_to(w, dam.b) {
                                    push(Edge::new(site_node(mix64(id ^ 0xDA42), dam.b, t.dev), t, cls::POWER, nw::HAMLET), edges);
                                }
                            }
                        }
                        Some(None) => {}
                    }
                }
            }
        }
    }
    if vis(cls::TRACK) {
        for c in sites::lattice_cells(w, sites::QUARRY_CELL, area.center, r_area + 12_000.0) {
            match sites::quarry(ctx, c) {
                None => *pending = true,
                Some(Some(s)) => {
                    if let Some(v) = nearest_place(w, s.center, places::VILLAGE, 12_000.0) {
                        let r = s.stamps.first().map_or(300.0, |st| st.v[0] as f64 * 1.1);
                        let rim = route::surface(w, s.center + (v.p - s.center).normalize_or_zero() * r);
                        push(Edge::new(site_node(mix64(v.id ^ s.center.x.to_bits() ^ 0x0A7), rim, v.dev), v, cls::LOCAL, nw::HAMLET), edges);
                    }
                }
                Some(None) => {}
            }
        }
    }
    if gsd < cls::GSD_MAX[cls::POWER] {
        for c in sites::lattice_cells(w, sites::WIND_CELL, area.center, r_area + 60_000.0) {
            match sites::wind(ctx, c) {
                None => *pending = true,
                Some(Some(s)) => {
                    let h = mix64(s.center.x.to_bits() ^ 0x3D10);
                    if let Some(v) = nearest_place(w, s.center, places::VILLAGE, 20_000.0) {
                        push(Edge::new(site_node(h, site_edge(w, &s, v.p), v.dev), v, cls::LOCAL, nw::HAMLET), edges);
                    }
                    if let Some(t) = power_to(w, s.center) {
                        push(Edge::new(site_node(h ^ 0x90, site_edge(w, &s, t.p), t.dev), t, cls::POWER, nw::HAMLET), edges);
                    }
                }
                Some(None) => {}
            }
        }
    }
}

type Place2 = places::Place;

/// Clear the caches (tests).
#[cfg(test)]
pub fn clear_cache() {
    *BASE.write().unwrap() = None;
    *ROUTES.write().unwrap() = None;
    *SAMPLES.write().unwrap() = None;
    *CROSS.write().unwrap() = None;
}
