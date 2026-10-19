//! Routing one edge of the network over the terrain (`docs/design/terrain-next.md` §6.3): a
//! dynamic programme on a corridor grid between the ends (bounded: ≤ 41 × 11 nodes), refined
//! piece by piece where the coarse grid is coarser than 260 m (≤ 41 × 7 nodes per piece). The
//! costs use host knowledge only, the same on both backends (a [`Scene`] per edge): the
//! CPU-analytic relief with the valleys of the major drainage carved in (the core's valley
//! profile, so routes see the flat valley floors), the sea, the core's lattice and closed-basin
//! lakes, the cold kit's large kettle lakes, river crossings (few and near perpendicular), and
//! from the caller the towns to go around, reservoirs and site footprints (no-go). A route that
//! still crosses wider standing water than its class may bridge ([`span`]) or enters a site is
//! tried once more in a wider corridor, then rejected. Moving averages, Chaikin smoothing, and
//! switchbacks where a minor road climbs steeper than it may. A pure function of the edge and
//! its scene.

use super::cls;
use crate::features::FSeg;
use crate::noise::*;
use crate::world::{Ctx, Seg, Terrain, World};
use glam::{DVec2, DVec3};

/// Routing parameters of a class.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// coarse grid step along the edge (m), corridor half width (share of the length)
    pub step: f64,
    pub width: f64,
    /// cost of grade: length × (1 + k g² + k2 (g − glim)²₊)
    pub k: f64,
    pub k2: f64,
    pub glim: f64,
    /// the steepest grade a road climbs without switchbacks (0: none)
    pub gmax: f64,
    /// cost of a change of the lateral step (m per node of lateral change)
    pub turn: f64,
    /// extra cost per metre over the sea, per metre inside a town that is not an end
    pub wet: f64,
    pub town: f64,
    /// extra cost per metre over standing water (lakes, reservoirs)
    pub lake: f64,
    /// cost of a river crossing (m), plus [`BRIDGE`] per metre of bridge (the channel's width
    /// over the sine of the crossing angle)
    pub xing: f64,
    /// the longest lake a route crosses (m, a narrow neck on a bridge; reservoirs and the sea:
    /// × 3; 0: any)
    pub span: f64,
    /// Chaikin rounds (0: straight between the grid nodes)
    pub smooth: u32,
    /// at most one lateral step change per node
    pub smooth_turns: bool,
    /// moving-average window along the route (m; 0: none): wide curves for major roads
    pub window: f64,
}

#[rustfmt::skip]
pub const PARAMS: [Params; cls::N] = [
    // motorway, trunk, secondary, local, track, ramp, rail, power
    Params { step: 2_500.0, width: 0.16, k: 300.0, k2: 30_000.0, glim: 0.05, gmax: 0.0, turn: 150.0, wet: 40.0, town: 6.0, lake: 60.0, xing: 600.0, span: 500.0, smooth: 1, smooth_turns: true, window: 400.0 },
    Params { step: 1_500.0, width: 0.18, k: 160.0, k2: 12_000.0, glim: 0.07, gmax: 0.0, turn: 60.0, wet: 25.0, town: 2.0, lake: 50.0, xing: 500.0, span: 500.0, smooth: 1, smooth_turns: true, window: 250.0 },
    Params { step: 400.0, width: 0.2, k: 100.0, k2: 6_000.0, glim: 0.10, gmax: 0.11, turn: 15.0, wet: 20.0, town: 0.0, lake: 40.0, xing: 300.0, span: 160.0, smooth: 3, smooth_turns: false, window: 0.0 },
    Params { step: 150.0, width: 0.22, k: 60.0, k2: 4_000.0, glim: 0.12, gmax: 0.13, turn: 6.0, wet: 15.0, town: 0.0, lake: 30.0, xing: 200.0, span: 160.0, smooth: 3, smooth_turns: false, window: 0.0 },
    Params { step: 100.0, width: 0.22, k: 40.0, k2: 3_000.0, glim: 0.16, gmax: 0.16, turn: 4.0, wet: 10.0, town: 0.0, lake: 25.0, xing: 100.0, span: 160.0, smooth: 3, smooth_turns: false, window: 0.0 },
    Params { step: 100.0, width: 0.0, k: 0.0, k2: 0.0, glim: 1.0, gmax: 0.0, turn: 0.0, wet: 0.0, town: 0.0, lake: 0.0, xing: 0.0, span: 0.0, smooth: 0, smooth_turns: false, window: 0.0 },
    Params { step: 2_500.0, width: 0.16, k: 2_500.0, k2: 80_000.0, glim: 0.022, gmax: 0.0, turn: 200.0, wet: 40.0, town: 6.0, lake: 60.0, xing: 600.0, span: 500.0, smooth: 1, smooth_turns: true, window: 600.0 },
    Params { step: 2_500.0, width: 0.08, k: 4.0, k2: 40.0, glim: 0.3, gmax: 0.0, turn: 30.0, wet: 0.5, town: 3.0, lake: 0.5, xing: 0.0, span: 0.0, smooth: 0, smooth_turns: false, window: 0.0 },
];

/// The longest stretch of a lake (`wide`: of a reservoir, which floods the valleys roads have to
/// cross, or of the sea) a route of `class` crosses on a bridge (m; 0: any).
pub fn span(class: usize, wide: bool) -> f64 {
    PARAMS[class].span * if wide { 3.0 } else { 1.0 }
}

/// Cost per metre of a river bridge (on top of the length).
pub const BRIDGE: f64 = 12.0;

/// Extra corridor half width of the second try (m).
pub const WIDEN: f64 = 600.0;
/// How far a route's ends may lie off its edge's chord (m): town entries snapped to an
/// arterial.
pub const ENTRY: f64 = 600.0;

/// The largest lateral deviation of a route of length `d` from its edge's chord (m).
pub fn max_deviation(class: usize, d: f64) -> f64 {
    let p = &PARAMS[class];
    p.width * d + 300.0 + if p.width > 0.0 { WIDEN } else { 0.0 } + ENTRY + if p.gmax > 0.0 { 250.0 } else { 0.0 }
}

/// The cold kit's lake scale (its `lake_mask`, cube root; 0: no lakes) estimated from pass A's
/// values at a point and the ground slope (m/m): thermokarst and kettle lakes of flat, low,
/// moist lowlands (`kits::cold`).
pub fn cold_lake_scale(t: &Terrain, slope: f64) -> f64 {
    let thermo = smoothstep(-1.0, -4.0, t.temp);
    let kettle = t.glaciation * smoothstep(9.0, 4.0, t.temp);
    let m = thermo.max(0.8 * kettle);
    if m <= 0.0 {
        return 0.0;
    }
    let lm = m
        * (1.0 - smoothstep(0.03, 0.07, slope))
        * (1.0 - smoothstep(350.0, 700.0, t.ground))
        * (1.0 - smoothstep(0.2, 0.5, t.mountain))
        * smoothstep(0.02, 0.12, t.moist)
        * (1.0 - smoothstep(0.1, 0.3, t.sand))
        * (0.3 + 0.7 * smoothstep(0.25, 0.65, t.style[0]));
    if lm < 0.03 {
        0.0
    } else {
        lm.cbrt()
    }
}

/// Is `p` in a kettle or thermokarst lake of the cold kit's large lattice (its ponds, the small
/// lattice, are crossed on the road's fill)? `t`: pass A's values there (the relief's), `slope`
/// the ground's.
pub fn in_cold_lake(t: &Terrain, slope: f64, p: DVec3) -> bool {
    let sc = cold_lake_scale(t, slope);
    if sc <= 0.0 {
        return false;
    }
    let h = crate::kits::cold::lake_at(&crate::kits::cold::LAKE_LATS[0], p, sc, 0.0);
    h.cov > 0.5 && h.kind == 0
}

/// A polyline of surface points and the arc length at each.
#[derive(Clone, Debug, Default)]
pub struct Line {
    pub pts: Vec<DVec3>,
    pub s: Vec<f64>,
}

impl Line {
    pub fn new(pts: Vec<DVec3>) -> Line {
        let mut s = Vec::with_capacity(pts.len());
        let mut acc = 0.0;
        for k in 0..pts.len() {
            if k > 0 {
                acc += (pts[k] - pts[k - 1]).length();
            }
            s.push(acc);
        }
        Line { pts, s }
    }

    pub fn len(&self) -> f64 {
        self.s.last().copied().unwrap_or(0.0)
    }

    /// The vertex index of the piece containing arc length `s`.
    pub fn piece(&self, s: f64) -> usize {
        let n = self.pts.len();
        if n < 2 {
            return 0;
        }
        match self.s.binary_search_by(|v| v.total_cmp(&s)) {
            Ok(k) => k.min(n - 2),
            Err(k) => k.saturating_sub(1).min(n - 2),
        }
    }

    /// The point at arc length `s` and the unit direction there.
    pub fn at(&self, s: f64) -> (DVec3, DVec3) {
        let n = self.pts.len();
        if n < 2 {
            return (self.pts.first().copied().unwrap_or_default(), DVec3::X);
        }
        let s = s.clamp(0.0, self.len());
        let k = self.piece(s);
        let (a, b) = (self.pts[k], self.pts[k + 1]);
        let l = (self.s[k + 1] - self.s[k]).max(1e-9);
        let t = ((s - self.s[k]) / l).clamp(0.0, 1.0);
        (a + (b - a) * t, (b - a).normalize_or(DVec3::X))
    }

    /// Distance from `p` to the polyline and the arc length of the closest point.
    pub fn dist(&self, p: DVec3) -> (f64, f64) {
        let mut best = (f64::MAX, 0.0);
        for k in 0..self.pts.len().saturating_sub(1) {
            let (a, b) = (self.pts[k], self.pts[k + 1]);
            let ab = b - a;
            let t = ((p - a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
            let d = (p - (a + ab * t)).length();
            if d < best.0 {
                best = (d, self.s[k] + t * (self.s[k + 1] - self.s[k]));
            }
        }
        best
    }
}

/// The surface point below / above `p`.
pub fn surface(w: &World, p: DVec3) -> DVec3 {
    let g = geodesy::ecef2geodetic(p, &w.ell);
    geodesy::geodetic2ecef(geodesy::Geodetic::new(g.lat, g.lon, 0.0), &w.ell)
}

/// The analytic relief (before rivers, lakes and the sea; < 0.5: sea) at `p`, band-limited at
/// `gsd`.
pub fn relief_h(w: &World, p: DVec3, gsd: f64) -> f64 {
    let g = geodesy::ecef2geodetic(p, &w.ell);
    let ctx = Ctx::new(g.lat, g.lon, gsd, &w.ell);
    w.relief_height(&ctx, &w.macro_at(ctx.p, gsd))
}

/// A lake a route goes around: a lattice lake (`pt` its 3D feature point) or a closed-basin lake
/// (`sink`: `pt` its centre), its radius and level (`World::terrain`'s lakes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lake {
    pub id: u64,
    pub pt: DVec3,
    pub rad: f64,
    pub level: f64,
    pub sink: bool,
}

/// Routes keep off the high mountains where the mean annual temperature at the ground is below
/// this (°C: glaciers and perennial snow): a high cost per metre, and no route with more than
/// [`ICE_MAX`] m of it.
const ICE_TEMP: f64 = -3.0;
const ICE_MAX: f64 = 300.0;

/// Water stands this much above a lake's level at its shore for routing (m).
const SHORE: f64 = 0.5;

impl Lake {
    /// The outline's centre below `p`'s radius, its axes and semi-axes (without the shore noise).
    fn frame(&self, r: f64) -> (DVec3, DVec3, DVec3, f64, f64) {
        let pc = self.pt.normalize() * r;
        let up = pc.normalize();
        if self.sink {
            let ax = up.cross(DVec3::new(u01k(self.id, 2) - 0.5, u01k(self.id, 3) - 0.5, u01k(self.id, 4) - 0.5)).normalize_or_zero();
            let el = 1.0 + 1.2 * u01k(self.id, 5);
            (pc, ax, up.cross(ax), self.rad * el.sqrt(), self.rad / el.sqrt())
        } else {
            let ax = up.cross(if up.z.abs() < 0.9 { DVec3::Z } else { DVec3::X }).normalize();
            (pc, ax, up.cross(ax), self.rad, self.rad)
        }
    }

    /// Is `p` (ground `h`, before the lake's bowl) under the lake's water? (as `World::terrain`)
    pub fn wet(&self, p: DVec3, h: f64, gsd: f64) -> bool {
        let pc = self.pt.normalize() * p.length();
        let (id, rad, level) = (self.id, self.rad, self.level);
        let depth = 3.0 + 0.01 * rad;
        let lw = rad * 0.6;
        if !self.sink {
            let d = (p - pc).length();
            if d > rad * 1.5 {
                return false;
            }
            let warpn = perlin3(id, p / lw) * 0.3 + perlin3(id ^ 7, p / (lw * 0.3)) * 0.12 * band(lw * 0.3, gsd);
            let de = d / rad * (1.0 + warpn);
            if de >= 1.0 {
                return false;
            }
            let bowl = level - depth * (1.0 - de * de);
            h.min(lerp(h, bowl, smoothstep(1.0, 0.7, de))) < level + SHORE
        } else {
            if (p - self.pt).length() >= 1.6 * rad {
                return false;
            }
            let (_, ax, _, _, _) = self.frame(p.length());
            let el = 1.0 + 1.2 * u01k(id, 5);
            let dv = p - pc;
            let (da, db) = (dv.dot(ax), (dv - ax * dv.dot(ax)).length());
            let d = (da * da / el + db * db * el).sqrt();
            let warpn = perlin3(id, p / (1.4 * lw)) * 0.45
                + perlin3(id ^ 7, p / (lw * 0.45)) * 0.18 * band(lw * 0.45, gsd)
                + perlin3(id ^ 9, p / (lw * 0.12)) * 0.06 * band(lw * 0.12, gsd);
            let de = d / rad * (1.0 + warpn);
            let near = 1.0 - smoothstep(1.5, 4.0, h - level);
            if de >= 1.0 || near <= 0.0 {
                return false;
            }
            let bowl = level - depth * (1.0 - de * de) - 0.3;
            h.min(lerp(h, bowl, near)) < level + SHORE
        }
    }

    /// The length of the chord p0–p1 inside the lake's outline (without the shore noise).
    pub fn chord(&self, p0: DVec3, p1: DVec3) -> f64 {
        let (pc, ax, ay, a, b) = self.frame(p0.length());
        let l = (p1 - p0).length();
        if super::net::chord_dist(p0, p1, pc) > a.max(b) {
            return 0.0;
        }
        let q = |p: DVec3| DVec2::new((p - pc).dot(ax) / a, (p - pc).dot(ay) / b);
        let (q0, q1) = (q(p0), q(p1));
        let d = q1 - q0;
        let (aa, bb, cc) = (d.length_squared(), 2.0 * q0.dot(d), q0.length_squared() - 1.0);
        let disc = bb * bb - 4.0 * aa * cc;
        if aa < 1e-12 || disc <= 0.0 {
            return 0.0;
        }
        let s = disc.sqrt();
        let (t0, t1) = (((-bb - s) / (2.0 * aa)).max(0.0), ((-bb + s) / (2.0 * aa)).min(1.0));
        (t1 - t0).max(0.0) * l
    }
}

/// A site footprint a route keeps out of: an oriented box, or a disc (`round`, radius
/// `half.x`), already widened by the road's half width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Block {
    pub c: DVec3,
    pub ex: DVec3,
    pub ey: DVec3,
    pub half: DVec2,
    pub round: bool,
}

impl Block {
    fn local(&self, p: DVec3) -> DVec2 {
        let d = p - self.c;
        DVec2::new(d.dot(self.ex), d.dot(self.ey))
    }

    pub fn inside(&self, p: DVec3) -> bool {
        let q = self.local(p);
        if self.round {
            q.length() < self.half.x
        } else {
            q.x.abs() < self.half.x && q.y.abs() < self.half.y
        }
    }

    /// Does the segment p0–p1 touch the footprint?
    pub fn hits(&self, p0: DVec3, p1: DVec3) -> bool {
        if super::net::chord_dist(p0, p1, self.c) > self.half.length() {
            return false;
        }
        let (q0, q1) = (self.local(p0), self.local(p1));
        if self.round {
            let d = q1 - q0;
            let t = (-q0.dot(d) / d.length_squared().max(1e-9)).clamp(0.0, 1.0);
            return (q0 + d * t).length() < self.half.x;
        }
        // slab clipping
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for k in 0..2 {
            let (a, d, h) = (q0[k], q1[k] - q0[k], self.half[k]);
            if d.abs() < 1e-12 {
                if a.abs() >= h {
                    return false;
                }
            } else {
                let (u, v) = ((-h - a) / d, (h - a) / d);
                t0 = t0.max(u.min(v));
                t1 = t1.min(u.max(v));
            }
        }
        t0 < t1
    }
}

/// What a route sees of its corridor: built-up areas to go around (centre, radius), the
/// drainage of levels 0–1 (valley floors, river crossings), lakes, reservoir pieces (the
/// settlements kit's dams) and site footprints; `cold`: kettle lakes are possible.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub avoid: Vec<(DVec3, f64)>,
    pub rivers: Vec<Seg>,
    pub lakes: Vec<Lake>,
    pub res: Vec<FSeg>,
    pub blocks: Vec<Block>,
    pub cold: bool,
}

/// Lattice lakes cell (the core's `SITE_LAKE` lattice seed).
const LAKE_KEY: u64 = 0x1A4E;

impl Scene {
    /// The drainage, lakes and cold-lake flag of the corridor of half width `half` around a–b
    /// (`fine`: also the closed-basin lakes of the streams, level 2: minor roads).
    pub fn of_corridor(w: &World, a: DVec3, b: DVec3, half: f64, fine: bool, avoid: Vec<(DVec3, f64)>) -> Scene {
        let mut sc = Scene { avoid, ..Default::default() };
        let d = (b - a).length();
        let nch = ((d / 30_000.0).ceil() as usize).max(1);
        let rch = 0.5 * d / nch as f64 + half;
        let mut seen: FxHashSet<(u64, u64, u64, u64)> = FxHashSet::default();
        let key = |s: &Seg| (s.a.x.to_bits(), s.a.y.to_bits(), s.b.x.to_bits(), s.level as u64);
        let mut sinks: Vec<Seg> = Vec::new();
        let lake_cell = w.cfg.hydro.lake_cell_km * 1000.0;
        let mut cells: FxHashSet<(i64, i64, i64)> = FxHashSet::default();
        for k in 0..nch {
            let c = surface(w, a + (b - a) * ((k as f64 + 0.5) / nch as f64));
            for s in w.river_segments_of(c, rch, 1_000.0, 0..2) {
                if seen.insert(key(&s)) {
                    if s.sink {
                        sinks.push(s);
                    }
                    sc.rivers.push(s);
                }
            }
            if fine {
                for s in w.river_segments_of(c, rch, 100.0, 2..3) {
                    if s.sink && seen.insert(key(&s)) {
                        sinks.push(s);
                    }
                }
            }
            // the lattice lakes' cells (their feature points may lie off the surface)
            let rr = rch + 1.2 * lake_cell;
            let lo = ((c - DVec3::splat(rr)) / lake_cell).floor();
            let hi = ((c + DVec3::splat(rr)) / lake_cell).floor();
            for z in lo.z as i64..=hi.z as i64 {
                for y in lo.y as i64..=hi.y as i64 {
                    for x in lo.x as i64..=hi.x as i64 {
                        cells.insert((x, y, z));
                    }
                }
            }
            let at = w.atlas().sample(c.normalize());
            if at.temp_c < 3.0 || (at.glaciation > 0.2 && at.temp_c < 13.0) {
                sc.cold = true;
            }
        }
        // the rivers within reach of the corridor (valley, meander warp)
        sc.rivers.retain(|s| {
            let lc = &w.cfg.hydro.levels[s.level as usize];
            let reach = 2.0 * s.valley + 0.9 * lc.meander * lc.cell_km * 1000.0 + 500.0;
            seg_dist(a, b, s.a, s.b) < half + reach
        });
        // closed basins (as `World::sink_lakes` over the pieces)
        let mut sl: Vec<(u64, DVec3, f64)> = Vec::new();
        for sg in &sinks {
            let hh = hash3(0x51A7, (sg.b.x / 10.0) as i64, (sg.b.y / 10.0) as i64, (sg.b.z / 10.0) as i64);
            match sl.iter_mut().find(|l| l.0 == hh) {
                Some(l) => l.2 += sg.hw,
                None => sl.push((hh, sg.b, sg.hw)),
            }
        }
        for (id, c, w0) in sl {
            let rad = (6.0 * w0).clamp(200.0, 2500.0) * (0.8 + 0.4 * u01k(id, 1));
            if super::net::chord_dist(a, b, c) > half + 2.0 * rad {
                continue;
            }
            if let Some(level) = w.sink_lake_level(id, c, rad) {
                sc.lakes.push(Lake { id, pt: c, rad, level, sink: true });
            }
        }
        // lattice lakes (existence by the values at the lake's centre)
        let dens = w.cfg.hydro.lake_density;
        let mut cells: Vec<_> = cells.into_iter().collect();
        cells.sort_unstable();
        for cc in cells {
            if dens <= 0.0 {
                break;
            }
            let (id, pt) = worley3_site(w.seed() ^ LAKE_KEY, cc, lake_cell, 0.85);
            if u01k(id, 1) > dens * 1.2 {
                continue;
            }
            let pc = surface(w, pt);
            if super::net::chord_dist(a, b, pc) > half + 0.3 * lake_cell * 1.8 {
                continue;
            }
            let g = geodesy::ecef2geodetic(pc, &w.ell);
            let ctx = Ctx::new(g.lat, g.lon, 100.0, &w.ell);
            let t = w.relief_terrain(&ctx, &w.macro_at(ctx.p, 100.0));
            let prob = dens * (0.3 + 0.9 * t.moist) * (1.0 - 0.8 * t.mountain);
            if u01k(id, 1) > prob || smoothstep(-0.06, 0.05, t.cont) <= 0.3 {
                continue;
            }
            let Some((rad, level)) = w.lattice_lake(id, pt) else { continue };
            if super::net::chord_dist(a, b, pc) > half + 1.8 * rad {
                continue;
            }
            sc.lakes.push(Lake { id, pt, rad, level, sink: false });
        }
        sc
    }

    /// The part of the scene near the corridor of half width `half` around a–b.
    fn local(&self, w: &World, a: DVec3, b: DVec3, half: f64) -> Scene {
        Scene {
            avoid: self.avoid.iter().filter(|(c, r)| super::net::chord_dist(a, b, *c) < half + r).copied().collect(),
            rivers: self
                .rivers
                .iter()
                .filter(|s| {
                    let lc = &w.cfg.hydro.levels[s.level as usize];
                    seg_dist(a, b, s.a, s.b) < half + 2.0 * s.valley + 0.9 * lc.meander * lc.cell_km * 1000.0 + 500.0
                })
                .copied()
                .collect(),
            lakes: self.lakes.iter().filter(|l| super::net::chord_dist(a, b, surface(w, l.pt)) < half + 2.0 * l.rad).copied().collect(),
            res: self.res.iter().filter(|s| seg_dist(a, b, s.a, s.b) < half + s.hw as f64).copied().collect(),
            blocks: self.blocks.iter().filter(|k| super::net::chord_dist(a, b, k.c) < half + k.half.length()).copied().collect(),
            cold: self.cold,
        }
    }

    /// Is there anything for [`check`] to test?
    fn has_water(&self) -> bool {
        !self.lakes.is_empty() || !self.res.is_empty() || self.cold
    }
}

/// Distance between the segments a–b and c–d (sampled: ends and the closest approach of each).
fn seg_dist(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    use super::net::chord_dist;
    chord_dist(a, b, c).min(chord_dist(a, b, d)).min(chord_dist(c, d, a)).min(chord_dist(c, d, b)).min({
        // (crossing segments: 0)
        let up = a.normalize();
        let ex = (b - a).normalize_or_zero();
        let ey = up.cross(ex);
        let (yc, yd) = ((c - a).dot(ey), (d - a).dot(ey));
        if (yc > 0.0) != (yd > 0.0) {
            let t = yc / (yc - yd);
            let x = (c - a).dot(ex) + ((d - a).dot(ex) - (c - a).dot(ex)) * t;
            if (0.0..=(b - a).length()).contains(&x) {
                0.0
            } else {
                f64::MAX
            }
        } else {
            f64::MAX
        }
    })
}

/// What a grid node knows: the ground (the relief with the valleys carved in), the sea, standing
/// water, a site footprint, a town, the ice of high mountains; the meander-warped positions of
/// drainage levels 0 and 1.
#[derive(Clone, Copy, Debug, Default)]
struct Probe {
    h: f64,
    sea: bool,
    ice: bool,
    stand: bool,
    res: bool,
    block: bool,
    town: bool,
    pw: [DVec3; 2],
}

/// Probe the scene at the surface point `p` (relief band-limited at `gsd`).
fn probe(w: &World, sc: &Scene, p: DVec3, gsd: f64) -> Probe {
    let g = geodesy::ecef2geodetic(p, &w.ell);
    let ctx = Ctx::new(g.lat, g.lon, gsd, &w.ell);
    let t = w.relief_terrain(&ctx, &w.macro_at(ctx.p, gsd));
    let h0 = t.ground;
    // (the glaciers and perennial snow of high mountains: roads went over 6 km summits in
    // switchbacks, a comb of hairpins on the ice, alpine_south@type)
    let ice = t.mountain > 0.25 && t.temp < ICE_TEMP;
    let mut o = Probe { h: h0, sea: h0 < 0.5, ice, pw: [ctx.p; 2], ..Default::default() };
    o.block = sc.blocks.iter().any(|k| k.inside(p));
    o.town = sc.avoid.iter().any(|&(q, r)| (p - q).length() < r);
    if !sc.rivers.is_empty() {
        for l in 0..2 {
            let lam = w.meander_wavelength(l);
            let [w1, w2] = w.meander_warp(l, ctx.p);
            o.pw[l] = ctx.p + (ctx.east * w1 + ctx.north * w2) * (0.22 * lam);
        }
        // the valleys (`World::terrain`'s profile without the kits' carve hooks)
        let land_fade = smoothstep(0.0, 0.1, smoothstep(-0.06, 0.05, t.cont));
        let shelf = smoothstep(-4.0, 0.5, h0);
        for s in &sc.rivers {
            let li = s.level as usize;
            let pw = o.pw[li];
            let ab = s.b - s.a;
            let u = ((pw - s.a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
            let dist = (pw - (s.a + ab * u)).length();
            if dist > 3.0 * s.valley + 3_000.0 {
                continue;
            }
            let hw = s.hw + (s.hw_b - s.hw) * u;
            let raw = s.ha + (s.hb - s.ha) * u;
            let floor = raw.max(1.0) - (1.0 + 0.04 * hw);
            let reach = (s.valley * 1.5 + s.hw + 200.0).max(11.3 * s.hw + 6.0 * (h0 - raw + 2.0 + 0.04 * s.hw) + 50.0);
            let fp_w = 6.0 * hw;
            let valley = s.valley.max(fp_w + lerp(6.0, 2.2, t.mountain) * (h0 - floor)).min(reach);
            if h0 > floor && h0 > -4.0 && dist < valley {
                let wall = smoothstep(fp_w, valley, dist);
                let wall = wall * wall * (3.0 - 2.0 * wall);
                let target = if dist < hw { floor - 0.8 - 0.04 * hw } else { (floor + 0.8).max(0.5) };
                let carved = h0.min(lerp(target, h0, wall)).max(h0 - w.cfg.hydro.levels[li].max_depth_m * (1.0 - 0.3 * wall));
                o.h = o.h.min(h0 + (carved - h0) * land_fade * shelf);
            }
        }
    }
    o.stand = sc.lakes.iter().any(|l| l.wet(ctx.p, o.h, gsd));
    o.res = sc.res.iter().any(|sg| crate::kits::settlements::sites::reservoir_at(sg, ctx.p, o.h, 0.1).0 > 0.5);
    if !o.stand
        && !o.res
        && sc.cold
        && t.ground < 700.0
        && (t.temp < -1.0 || (t.glaciation > 0.0 && t.temp < 9.0))
        // (inside a lake at the largest scale: its outline at any scale lies within)
        && crate::kits::cold::lake_at(&crate::kits::cold::LAKE_LATS[0], ctx.p, 1.0, 0.0).cov > 0.0
    {
        // (the slope over ±40 m)
        let at = |de: f64, dn: f64| {
            let q = ctx.offset(de, dn);
            let gq = geodesy::ecef2geodetic(q, &w.ell);
            let c = Ctx::new(gq.lat, gq.lon, gsd, &w.ell);
            w.relief_height(&c, &w.macro_at(c.p, gsd))
        };
        let slope = DVec2::new(at(40.0, 0.0) - at(-40.0, 0.0), at(0.0, 40.0) - at(0.0, -40.0)).length() / 80.0;
        o.stand = in_cold_lake(&Terrain { ground: o.h, ..t }, slope, ctx.p);
    }
    o
}

/// A route end on dry land: `p`, or where the way from `p` toward `toward` first leaves the
/// standing water or the sea (within a third of the way, ≤ 1.5 km; then 30 m on).
pub fn dry_end(w: &World, sc: &Scene, p: DVec3, toward: DVec3) -> DVec3 {
    let wet = |q: DVec3| {
        let o = probe(w, sc, q, 30.0);
        o.stand || o.res || o.sea
    };
    if !wet(p) {
        return p;
    }
    let d = (toward - p).length();
    let n = ((d / 3.0).min(1_500.0) / 25.0) as usize;
    for k in 1..=n {
        let t = (k as f64 * 25.0 + 30.0) / d;
        if !wet(surface(w, p + (toward - p) * ((k as f64 * 25.0) / d))) {
            return surface(w, p + (toward - p) * t.min(1.0));
        }
    }
    p
}

/// The cost of the river crossings of the step n0 → n1 (in each level's warped space).
fn crossings(sc: &Scene, pr: &Params, n0: &Probe, n1: &Probe) -> f64 {
    if pr.xing <= 0.0 {
        return 0.0;
    }
    let mut c = 0.0;
    for s in &sc.rivers {
        let l = s.level as usize;
        let (p0, p1) = (n0.pw[l], n1.pw[l]);
        let d = p1 - p0;
        let ln = d.length();
        if ln < 1.0 || (s.a - p0).length() > ln + (s.b - s.a).length() + 1.0 {
            continue;
        }
        let ex = d / ln;
        let ey = p0.normalize().cross(ex);
        let (u0, u1) = (s.a - p0, s.b - p0);
        let (y0, y1) = (u0.dot(ey), u1.dot(ey));
        if (y0 > 0.0) == (y1 > 0.0) {
            continue;
        }
        let t = y0 / (y0 - y1);
        let x = u0.dot(ex) + (u1.dot(ex) - u0.dot(ex)) * t;
        if !(0.0..ln).contains(&x) {
            continue;
        }
        let sin = ex.cross((s.b - s.a).normalize_or_zero()).length();
        let hw = s.hw + (s.hw_b - s.hw) * t;
        c += pr.xing + BRIDGE * 2.0 * hw / sin.max(0.25);
    }
    c
}

/// Grid DP between `a` and `b` over a corridor of `na` nodes along and `nc` across (half width
/// `half`, tapered to the ends): the node polyline and the metres over the sea. Monotone along
/// the edge; ties broken by node index (strict `<` in index order). None: no way through (site
/// footprints). Also the metres over the ice of high mountains.
#[allow(clippy::too_many_arguments)]
fn corridor(w: &World, a: DVec3, b: DVec3, na: usize, nc: usize, half: f64, pr: &Params, sc: &Scene) -> Option<(Vec<DVec3>, f64, f64)> {
    let c = nc / 2;
    // (a lateral step at most ~0.45 of the step along: a node-to-node bend of ≤ 24°)
    let step_a = (b - a).length() / (na - 1) as f64;
    let dy = if c > 0 { (half / c as f64).min(0.45 * step_a) } else { 0.0 };
    let lat = (a + b).normalize().cross(b - a).normalize_or_zero();
    let gsd = 0.5 * step_a;
    let sc = sc.local(w, a, b, c as f64 * dy + 0.5 * step_a);
    // (the nodes' probes in parallel: a long edge has thousands)
    use rayon::prelude::*;
    let mut pos = vec![DVec3::ZERO; na * nc];
    let mut nd_: Vec<Option<Probe>> = vec![None; na * nc];
    let probes: Vec<(usize, DVec3, Probe)> = (0..na * nc)
        .into_par_iter()
        .filter_map(|k| {
            let (i, j) = (k / nc, k % nc);
            if (i == 0 || i == na - 1) && j != c {
                return None;
            }
            let t = i as f64 / (na - 1) as f64;
            let taper = (std::f64::consts::PI * t).sin().sqrt();
            let base = a + (b - a) * t;
            let p = if i == 0 {
                a
            } else if i == na - 1 {
                b
            } else {
                surface(w, base + lat * ((j as f64 - c as f64) * dy * taper))
            };
            let mut o = probe(w, &sc, p, gsd);
            if i == 0 || i == na - 1 {
                // (the ends are where they are)
                o.block = false;
            }
            Some((k, p, o))
        })
        .collect();
    for (k, p, o) in probes {
        pos[k] = p;
        nd_[k] = Some(o);
    }
    // the standing water between the nodes of a step, sampled where a lake's outline or a
    // reservoir is near (a coarse step jumps a lake between two dry nodes)
    const DJ: i64 = 2;
    let nd = (2 * DJ + 1) as usize;
    let mut between = vec![0.0f64; na * nc * nd];
    if !sc.lakes.is_empty() || !sc.res.is_empty() {
        between.par_chunks_mut(nd).enumerate().for_each(|(k0, out)| {
            let (i, j) = (k0 / nc, k0 % nc);
            if i + 1 >= na || nd_[k0].is_none() {
                return;
            }
            for dj in -DJ..=DJ {
                let j1 = j as i64 + dj;
                if j1 < 0 || j1 >= nc as i64 || nd_[(i + 1) * nc + j1 as usize].is_none() {
                    continue;
                }
                let (p0, p1) = (pos[k0], pos[(i + 1) * nc + j1 as usize]);
                let near = sc.lakes.iter().any(|l| super::net::chord_dist(p0, p1, l.pt.normalize() * p0.length()) < 2.4 * l.rad)
                    || sc.res.iter().any(|r| seg_dist(p0, p1, r.a, r.b) < r.hw as f64);
                if !near {
                    continue;
                }
                let l = (p1 - p0).length();
                if l < 300.0 {
                    // (short steps: the nodes see the lakes)
                    continue;
                }
                let ns = (l / (l / 8.0).clamp(100.0, 300.0)).ceil().max(2.0) as usize;
                let wet = (1..ns)
                    .filter(|&k| {
                        let o = probe(w, &sc, surface(w, p0 + (p1 - p0) * (k as f64 / ns as f64)), gsd);
                        o.stand || o.res
                    })
                    .count();
                out[(dj + DJ) as usize] = l * wet as f64 / (ns - 1) as f64;
            }
        });
    }
    let first = corridor_dp(&pos, &nd_, &between, na, nc, dy, pr, &sc, pr.smooth_turns);
    first.or_else(|| if pr.smooth_turns { corridor_dp(&pos, &nd_, &between, na, nc, dy, pr, &sc, false) } else { None })
}

#[allow(clippy::too_many_arguments)]
fn corridor_dp(
    pos: &[DVec3],
    nodes: &[Option<Probe>],
    between: &[f64],
    na: usize,
    nc: usize,
    dy: f64,
    pr: &Params,
    sc: &Scene,
    smooth_turns: bool,
) -> Option<(Vec<DVec3>, f64, f64)> {
    let c = nc / 2;
    const DJ: i64 = 2;
    let nd = (2 * DJ + 1) as usize;
    let idx = |i: usize, j: usize, k: usize| (i * nc + j) * nd + k;
    let mut cost = vec![f64::INFINITY; na * nc * nd];
    let mut from = vec![u32::MAX; na * nc * nd];
    let mut water = vec![0.0f64; na * nc * nd];
    let mut icem = vec![0.0f64; na * nc * nd];
    cost[idx(0, c, DJ as usize)] = 0.0;
    for i in 0..na - 1 {
        for j in 0..nc {
            let Some(n0) = nodes[i * nc + j] else { continue };
            if (0..nd).all(|k| !cost[idx(i, j, k)].is_finite()) {
                continue;
            }
            let p0 = pos[i * nc + j];
            for dj in -DJ..=DJ {
                let j1 = j as i64 + dj;
                if j1 < 0 || j1 >= nc as i64 {
                    continue;
                }
                let j1 = j1 as usize;
                let n1i = (i + 1) * nc + j1;
                let Some(n1) = nodes[n1i] else { continue };
                let p1 = pos[n1i];
                if n1.block || sc.blocks.iter().any(|k| k.hits(p0, p1)) {
                    continue;
                }
                // the step's cost (before the turn)
                let l = (p1 - p0).length().max(1.0);
                let g = (n1.h - n0.h).abs() / l;
                let over = (g - pr.glim).max(0.0);
                let wet = if n0.sea || n1.sea { l } else { 0.0 };
                let ice = 0.5 * (n0.ice as u32 + n1.ice as u32) as f64 * l;
                let geo: f64 = sc.lakes.iter().map(|lk| lk.chord(p0, p1)).sum();
                let stand = (((n0.stand || n0.res) as u32 + (n1.stand || n1.res) as u32) as f64 * 0.5 * l)
                    .max(0.85 * geo)
                    .max(between[(i * nc + j) * nd + (dj + DJ) as usize])
                    .min(l);
                let inside = 0.5 * (n0.town as u32 + n1.town as u32) as f64 * l;
                let base = l * (1.0 + pr.k * g * g + pr.k2 * over * over)
                    + pr.wet * wet
                    + pr.lake * stand
                    + pr.town * inside
                    + 2.0 * pr.lake * ice
                    + crossings(sc, pr, &n0, &n1);
                for k in 0..nd {
                    let c0 = cost[idx(i, j, k)];
                    if !c0.is_finite() {
                        continue;
                    }
                    let dturn = (dj - (k as i64 - DJ)).abs();
                    // (major roads and railways turn by at most one lateral step per node: no
                    // hairpins)
                    if smooth_turns && dturn > 1 && i > 0 {
                        continue;
                    }
                    let turn = pr.turn * dy / 100.0 * dturn as f64;
                    let k1 = (dj + DJ) as usize;
                    let c1 = c0 + base + turn;
                    let t = idx(i + 1, j1, k1);
                    if c1 < cost[t] {
                        cost[t] = c1;
                        from[t] = idx(i, j, k) as u32;
                        water[t] = water[idx(i, j, k)] + wet;
                        icem[t] = icem[idx(i, j, k)] + ice;
                    }
                }
            }
        }
    }
    let mut best = (f64::INFINITY, usize::MAX);
    for k in 0..nd {
        let t = idx(na - 1, c, k);
        if cost[t] < best.0 {
            best = (cost[t], t);
        }
    }
    if best.1 == usize::MAX {
        return None;
    }
    let mut path = Vec::with_capacity(na);
    let mut t = best.1;
    loop {
        path.push(pos[t / nd]);
        if from[t] == u32::MAX {
            break;
        }
        t = from[t] as usize;
    }
    path.reverse();
    Some((path, water[best.1], icem[best.1]))
}

/// Route from `a` to `b` (surface points) for `class` through its scene. None: the edge would
/// cross too much sea, wider standing water than the class bridges, or a site.
pub fn route(w: &World, a: DVec3, b: DVec3, class: usize, sc: &Scene) -> Option<Line> {
    let pr = &PARAMS[class];
    for extra in [0.0, WIDEN] {
        if extra > 0.0 && pr.width <= 0.0 {
            break;
        }
        let Some((line, raw, wet)) = route_once(w, a, b, class, sc, extra) else { continue };
        if !(sc.has_water() || !sc.blocks.is_empty() || wet > 0.0) {
            return Some(line);
        }
        if check(w, sc, &line, class, wet > 0.0) {
            return Some(line);
        }
        // (the smoothing may cut a shore: the route without the moving average)
        if let Some(raw) = raw {
            if check(w, sc, &raw, class, wet > 0.0) {
                return Some(raw);
            }
        }
    }
    None
}

/// Why [`route`] finds no route (tests): "corridor" (no way through the footprints), "sea" (too
/// much sea), or the first check failure ("site", "water", "sea run") with its arc length.
#[cfg(test)]
pub fn diagnose(w: &World, a: DVec3, b: DVec3, class: usize, sc: &Scene) -> String {
    let Some((line, _, wet)) = route_once(w, a, b, class, sc, WIDEN) else {
        let pr = &PARAMS[class];
        let d = (b - a).length();
        let na = ((d / pr.step).ceil() as usize).clamp(3, 40) + 1;
        let nc = if pr.width > 0.0 { 11 } else { 1 };
        let half = pr.width * d + 300.0 + WIDEN;
        return if corridor(w, a, b, na, nc, half, pr, sc).is_none() { "corridor".into() } else { "sea".into() };
    };
    let len = line.len();
    for k in 0..=((len / 25.0) as usize) {
        let s = k as f64 * 25.0;
        if s < 60.0 || len - s < 60.0 {
            continue;
        }
        let p = line.at(s).0;
        let o = probe(w, sc, p, 30.0);
        if o.block || sc.blocks.iter().any(|b| b.inside(p)) {
            return format!("site at {s:.0} of {len:.0}");
        }
        if o.stand || o.res {
            return format!("{} at {s:.0} of {len:.0} (wet {wet:.0})", if o.res { "reservoir" } else { "lake" });
        }
    }
    format!("sea run? (wet {wet:.0}, len {len:.0})")
}

/// One try of [`route`] with the corridor `extra` m wider: the route, the route without its
/// moving average (if it has one), the metres over the sea.
fn route_once(w: &World, a: DVec3, b: DVec3, class: usize, sc: &Scene, extra: f64) -> Option<(Line, Option<Line>, f64)> {
    let pr = &PARAMS[class];
    let d = (b - a).length();
    if d < 1.0 {
        return None;
    }
    let na = ((d / pr.step).ceil() as usize).clamp(3, 40) + 1;
    let nc = if pr.width > 0.0 { 11 } else { 1 };
    let half = pr.width * d + 300.0 * (pr.width > 0.0) as u32 as f64 + extra;
    let (coarse, wet, ice) = corridor(w, a, b, na, nc, half, pr, sc)?;
    if wet > (0.12 * d).clamp(400.0, 3_000.0) || ice > ICE_MAX {
        return None;
    }
    // refine the pieces of a coarse grid (lateral resolution: a third of the coarse spacing)
    let step = d / (na - 1) as f64;
    let pts = if step > 260.0 && class != cls::POWER {
        // (the pieces in parallel)
        use rayon::prelude::*;
        let fine: Vec<Option<Vec<DVec3>>> = (0..coarse.len() - 1)
            .into_par_iter()
            .map(|k| {
                let (p, q) = (coarse[k], coarse[k + 1]);
                let l = (q - p).length();
                let nf = ((l / 120.0).ceil() as usize).clamp(3, 40) + 1;
                // (fine lateral steps of 0.45 of the fine step: the local adaptation of the
                // coarse route)
                let hf = 3.0 * 0.45 * l / (nf - 1) as f64;
                corridor(w, p, q, nf, 7, hf, pr, sc).map(|(f, _, _)| f)
            })
            .collect();
        let mut v: Vec<DVec3> = vec![coarse[0]];
        for (k, f) in fine.into_iter().enumerate() {
            match f {
                Some(f) => v.extend_from_slice(&f[1..]),
                None => v.push(coarse[k + 1]),
            }
        }
        v
    } else {
        coarse
    };
    let finish = |mut pts: Vec<DVec3>, avg: bool| {
        if avg && pr.window > 0.0 {
            pts = average(w, &pts, pr.window);
        }
        if pr.smooth > 0 {
            pts = chaikin(w, &pts, pr.smooth as usize);
        }
        if pr.gmax > 0.0 {
            pts = switchbacks(w, &pts, pr);
        }
        Line::new(pts)
    };
    let raw = (pr.window > 0.0).then(|| finish(pts.clone(), false));
    Some((finish(pts, true), raw, wet))
}

/// Does the route keep out of the scene's site footprints, and cross lakes, reservoirs (`sea`:
/// and the sea) only where narrower than its class bridges ([`span`])? (point probes every span / 6
/// where water can be, the ends' first and last 60 m skipped)
pub fn check(w: &World, sc: &Scene, line: &Line, class: usize, sea: bool) -> bool {
    let n = line.pts.len();
    // site footprints: exactly, piece by piece
    if !sc.blocks.is_empty() {
        for k in 0..n.saturating_sub(1) {
            let (s0, s1) = (line.s[k], line.s[k + 1]);
            if s1 < 60.0 || s0 > line.len() - 60.0 {
                continue;
            }
            let (p0, p1) = (line.at(s0.max(60.0)).0, line.at(s1.min(line.len() - 60.0)).0);
            if sc.blocks.iter().any(|b| b.hits(p0, p1)) {
                return false;
            }
        }
    }
    let span_w = span(class, false);
    if span_w <= 0.0 {
        return true;
    }
    let len = line.len();
    let ds = (span_w / 6.0).clamp(25.0, 100.0);
    let ns = (len / ds).ceil().max(1.0) as usize;
    let (mut run_w, mut run_r, mut run_s) = (0.0, 0.0, 0.0);
    // (probed in chunks, each with its part of the scene)
    const CH: usize = 64;
    let mut k0 = 0;
    while k0 <= ns {
        let k1 = (k0 + CH).min(ns);
        let (pa, pb) = (line.at(len * k0 as f64 / ns as f64).0, line.at(len * k1 as f64 / ns as f64).0);
        let part = sc.local(w, pa, pb, (len * (k1 - k0) as f64 / ns as f64) + 100.0);
        for k in k0..=k1 {
            if k == k1 && k1 != ns {
                break;
            }
            let s = len * k as f64 / ns as f64;
            if s < 60.0 || len - s < 60.0 {
                continue;
            }
            let p = line.at(s).0;
            // (water can be here: near a lake's or reservoir's outline, in cold lowlands)
            let near = (part.cold && crate::kits::cold::lake_at(&crate::kits::cold::LAKE_LATS[0], p, 1.0, 0.0).cov > 0.0)
                || part.lakes.iter().any(|l| (p - l.pt.normalize() * p.length()).length() < 2.4 * l.rad)
                || part.res.iter().any(|r| super::net::chord_dist(r.a, r.b, p) < r.hw as f64);
            let o = if near {
                probe(w, &part, p, 30.0)
            } else if sea {
                Probe { sea: relief_h(w, p, 30.0) < 0.5, ..Default::default() }
            } else {
                Probe::default()
            };
            run_w = if o.stand { run_w + ds } else { 0.0 };
            run_r = if o.res { run_r + ds } else { 0.0 };
            run_s = if o.sea { run_s + ds } else { 0.0 };
            if run_w > span_w || run_r.max(run_s) > span(class, true) {
                return false;
            }
        }
        k0 = k1 + if k1 == ns { 1 } else { 0 };
    }
    true
}

/// A moving average of the polyline resampled every 40 m over `window` m (the window shrinks
/// toward the ends, which stay), points back on the surface.
fn average(w: &World, pts: &[DVec3], window: f64) -> Vec<DVec3> {
    let l = Line::new(pts.to_vec());
    let len = l.len();
    let n = ((len / 40.0).ceil() as usize).max(2);
    let r: Vec<DVec3> = (0..=n).map(|k| l.at(len * k as f64 / n as f64).0).collect();
    let hk = ((0.5 * window / (len / n as f64)).round() as usize).max(1);
    // prefix sums for the window means
    let mut acc = vec![DVec3::ZERO; r.len() + 1];
    for k in 0..r.len() {
        acc[k + 1] = acc[k] + r[k];
    }
    (0..r.len())
        .map(|k| {
            let h = hk.min(k).min(r.len() - 1 - k);
            if h == 0 {
                r[k]
            } else {
                surface(w, (acc[k + h + 1] - acc[k - h]) / (2 * h + 1) as f64)
            }
        })
        .collect()
}

/// Corner cutting (Chaikin), ends kept, points back on the surface.
pub fn chaikin(w: &World, p: &[DVec3], rounds: usize) -> Vec<DVec3> {
    let mut cur = p.to_vec();
    for _ in 0..rounds {
        if cur.len() < 3 {
            break;
        }
        let mut nx = Vec::with_capacity(cur.len() * 2);
        nx.push(cur[0]);
        for k in 0..cur.len() - 1 {
            let (a, b) = (cur[k], cur[k + 1]);
            if k > 0 {
                nx.push(a * 0.75 + b * 0.25);
            }
            if k < cur.len() - 2 {
                nx.push(a * 0.25 + b * 0.75);
            }
        }
        nx.push(*cur.last().unwrap());
        cur = nx;
    }
    let n = cur.len();
    cur.iter().enumerate().map(|(k, &q)| if k == 0 || k == n - 1 { q } else { surface(w, q) }).collect()
}

/// Replace the stretches steeper than the road may climb by zigzags across the slope (legs
/// joined by rounded hairpins), long enough to keep the grade.
fn switchbacks(w: &World, pts: &[DVec3], pr: &Params) -> Vec<DVec3> {
    let r = Line::new(pts.to_vec());
    let len = r.len();
    let ds = 60.0;
    let n = ((len / ds).ceil() as usize).max(1);
    let s: Vec<f64> = (0..=n).map(|k| len * k as f64 / n as f64).collect();
    let h: Vec<f64> = s.iter().map(|&v| relief_h(w, r.at(v).0, 40.0)).collect();
    let steep: Vec<bool> = (0..n).map(|k| (h[k + 1] - h[k]).abs() / (s[k + 1] - s[k]) > pr.gmax * 1.1).collect();
    let mut runs = Vec::new();
    let mut k = 0;
    while k < n {
        if steep[k] {
            let k0 = k;
            let up = h[k + 1] > h[k];
            while k < n && steep[k] && (h[k + 1] > h[k]) == up {
                k += 1;
            }
            runs.push((k0, k));
        } else {
            k += 1;
        }
    }
    if runs.is_empty() {
        return pts.to_vec();
    }
    let amp_max = if pr.gmax > 0.12 { 110.0 } else { 160.0 };
    let mut out: Vec<DVec3> = vec![r.pts[0]];
    let mut s_done = 0.0;
    let push_range = |out: &mut Vec<DVec3>, s0: f64, s1: f64| {
        for (k, &v) in r.s.iter().enumerate() {
            if v > s0 && v < s1 {
                out.push(r.pts[k]);
            }
        }
        out.push(r.at(s1).0);
    };
    for (k0, k1) in runs {
        let (s0, s1) = (s[k0], s[k1]);
        let lr = s1 - s0;
        let climb = (h[k1] - h[k0]).abs();
        let need = climb / (0.9 * pr.gmax);
        if lr < 150.0 || need < 1.25 * lr || s0 < s_done {
            continue;
        }
        let extra = (need * need - lr * lr).sqrt();
        let legs = ((extra / (2.0 * amp_max)).ceil() as usize).clamp(2, 14);
        let a = (extra / (2.0 * legs as f64)).min(amp_max);
        push_range(&mut out, s_done, s0);
        let (pa, _) = r.at(s0);
        let (pb, _) = r.at(s1);
        let dir = pb - pa;
        let side = pa.normalize().cross(dir).normalize();
        let mut zz = vec![pa];
        for l in 0..legs {
            let sign = if l % 2 == 0 { 1.0 } else { -1.0 };
            let f = (l as f64 + 0.5) / legs as f64;
            zz.push(surface(w, pa + dir * f + side * (sign * a)));
        }
        zz.push(pb);
        let zz = chaikin(w, &zz, 3);
        out.extend_from_slice(&zz[1..]);
        s_done = s1;
    }
    push_range(&mut out, s_done, len);
    out.dedup_by(|b, a| (*b - *a).length() < 0.5);
    out
}
