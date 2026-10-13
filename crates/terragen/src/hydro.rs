//! Drainage network: rivers as a downhill flow graph.
//!
//! For each level (major rivers, tributaries, streams) a jittered 3D lattice provides points; the
//! points of cells near the ellipsoid surface are projected onto it ("active" points). Every active
//! point drains to the neighbouring active point of steepest descent of the relief, so the graph
//! is acyclic and rivers always run downhill on the actual terrain. River channels are the edges of
//! this graph, meandered by a domain warp; the water level is interpolated along each edge and so
//! decreases monotonically downstream. Points without a lower neighbour are sinks (lakes form
//! there from the basin-filling lake model).

use super::*;
use std::cell::RefCell;

/// One drainage edge (channel segment), surface points in ECEF.
#[derive(Clone, Copy, Debug)]
pub struct Seg {
    pub a: DVec3,
    pub b: DVec3,
    pub ha: f64,
    pub hb: f64,
    pub level: u8,
    /// half width of the channel (m)
    pub hw: f64,
    /// valley half width (m)
    pub valley: f64,
    /// channel half width at `b` (narrower where a channel starts at a spring)
    pub hw_b: f64,
    /// the channel ends in a closed basin at `b`, where it feeds a lake
    pub sink: bool,
}

/// Closest drainage channel to a point.
#[derive(Clone, Copy, Debug)]
pub struct RiverHit {
    /// signed distance to the channel centre line (m)
    pub d: f64,
    pub hw: f64,
    pub valley: f64,
    /// water surface / valley floor level at the closest point
    pub floor: f64,
    pub level: u8,
    /// position along the piece (0 at its start, 1 at its end); where the piece is a source
    /// (`spring`), 0 is the head of the channel
    pub u: f64,
    /// the piece is the first of a channel that starts at a spring
    pub spring: bool,
    /// the bound of the carve extent used to select the piece (m)
    pub reach: f64,
}

#[derive(Clone, Copy)]
struct FlowPt {
    s: DVec3,
    h: f64,
    active: bool,
}

/// Per-thread cache key of a lattice point: (level / world key, lattice cell x, y, z).
type LatticeKey = (u64, i64, i64, i64);
/// A lattice cell (x, y, z).
type Cell = (i64, i64, i64);

thread_local! {
    static PTS: RefCell<FxHashMap<LatticeKey, FlowPt>> = RefCell::new(FxHashMap::default());
    static TGT: RefCell<FxHashMap<LatticeKey, Option<Cell>>> = RefCell::new(FxHashMap::default());
    static SRC: RefCell<FxHashMap<LatticeKey, bool>> = RefCell::new(FxHashMap::default());
    static MAIN: RefCell<FxHashMap<LatticeKey, Option<Cell>>> = RefCell::new(FxHashMap::default());
}

/// The junction angle of a tributary with the main stem: its inflow direction's angle to the
/// main stem, clamped to 35°–80° (cosines).
const JOIN_COS: [f32; 2] = [0.173_648_18, 0.819_152_04];

/// Where the tributary of rank `r` joins the main stem's bend (its curve parameter, 0.5–0.875:
/// distinct for distinct ranks; f32 as in WGSL).
fn junction(r: u32) -> f32 {
    0.5 + 0.375 * (r as f32 * 0.618_034).fract()
}

/// The inner control points of a tributary's last reach, a cubic from `p0` (the middle of its
/// edge, entered along `u_in`: it continues the bend upstream smoothly) to the junction `j` on
/// the main stem, which it meets at 35°–80° to the main stem's direction `tan` (f32 directions,
/// as WGSL `trib_controls`).
fn tributary_controls(p0: DVec3, u_in: glam::Vec3, j: DVec3, tan: glam::Vec3) -> (DVec3, DVec3) {
    let up = j.as_vec3().normalize();
    let tn = tan.normalize();
    let nl = up.cross(tn).normalize();
    let side = if (p0 - j).as_vec3().dot(nl) >= 0.0 { 1.0f32 } else { -1.0 };
    let c = u_in.dot(tn).clamp(JOIN_COS[0], JOIN_COS[1]);
    let s = (1.0 - c * c).sqrt();
    let d = tn * c - nl * (side * s);
    let l = (j - p0).as_vec3().length() / 3.0;
    (p0 + (u_in * l).as_dvec3(), j - (d * l).as_dvec3())
}

/// A cubic Bézier curve at `k`.
fn cubic(p: [DVec3; 4], k: f64) -> DVec3 {
    let r = 1.0 - k;
    p[0] * (r * r * r) + p[1] * (3.0 * r * r * k) + p[2] * (3.0 * r * k * k) + p[3] * (k * k * k)
}

impl World {
    pub(crate) fn level_key(&self, lvl: usize) -> u64 {
        self.seed.rotate_left(23) ^ (lvl as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    fn flow_point(&self, lvl: usize, c: (i64, i64, i64)) -> FlowPt {
        let key = (self.level_key(lvl) ^ self.cache_key, c.0, c.1, c.2);
        if let Some(p) = PTS.with(|m| m.borrow().get(&key).copied()) {
            return p;
        }
        let cell = self.cfg.hydro.levels[lvl].cell_km * KM;
        let hh = hash3(self.level_key(lvl), c.0, c.1, c.2);
        let p = DVec3::new(
            c.0 as f64 + 0.5 + 0.8 * (u01k(hh, 1) - 0.5),
            c.1 as f64 + 0.5 + 0.8 * (u01k(hh, 2) - 0.5),
            c.2 as f64 + 0.5 + 0.8 * (u01k(hh, 3) - 0.5),
        ) * cell;
        let g = geodesy::ecef2geodetic(p, &self.ell);
        let active = g.h.abs() < 0.5 * cell;
        let fp = if active {
            let gsd = cell / 4.0;
            let ctx = Ctx::new(g.lat, g.lon, gsd, &self.ell);
            let m = self.macro_at(ctx.p, gsd);
            let t = self.terrain_impl(&ctx, &m, Mode::Relief, None, None);
            FlowPt { s: ctx.p, h: t.ground, active: true }
        } else {
            FlowPt { s: DVec3::ZERO, h: 0.0, active: false }
        };
        PTS.with(|m| {
            let mut m = m.borrow_mut();
            if m.len() > 400_000 {
                m.clear();
            }
            m.insert(key, fp);
        });
        fp
    }

    /// Downstream neighbour (steepest descent) of an active point; None for sinks and the sea.
    fn flow_target(&self, lvl: usize, c: (i64, i64, i64)) -> Option<(i64, i64, i64)> {
        let key = (self.level_key(lvl) ^ self.cache_key, c.0, c.1, c.2);
        if let Some(t) = TGT.with(|m| m.borrow().get(&key).copied()) {
            return t;
        }
        let cell = self.cfg.hydro.levels[lvl].cell_km * KM;
        let me = self.flow_point(lvl, c);
        let mut best = None;
        // shallow sea points (the coarse relief's coast is not the real one) drain on too, so
        // rivers run on to the actual shoreline instead of ending in a blunt cap on land; the
        // channel over the sea is under water (and the seabed is not carved)
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
            // no lower neighbour: look a little farther for an outlet to the sea (beyond a
            // shallow dip of the coarse relief; rivers ended there on land in a blunt cap).
            // Inland, a point without a lower neighbour is a closed basin and holds a lake: a
            // 2-ring edge to a lower inland point drew a channel up to 3 cells long, straight
            // across the higher ground between (a canal-like trench), and several of them
            // converged side by side on the next basin's lake.
            // Targets are always lower, so the graph stays acyclic.
            if best.is_none() {
                for dz in -2..=2i64 {
                    for dy in -2..=2i64 {
                        for dx in -2..=2i64 {
                            if dx.abs().max(dy.abs()).max(dz.abs()) < 2 {
                                continue;
                            }
                            let n = (c.0 + dx, c.1 + dy, c.2 + dz);
                            let o = self.flow_point(lvl, n);
                            if !o.active || o.h >= 0.0 {
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
        TGT.with(|m| {
            let mut m = m.borrow_mut();
            if m.len() > 400_000 {
                m.clear();
            }
            m.insert(key, best);
        });
        best
    }

    /// Lattice cells of the box [lo, hi] (cell coordinates) that the ellipsoid shell of possibly
    /// active points (radius within [b − m, a + m], see `maybe_active`) can pass through, in the
    /// order of a z, y, x loop over the box. Columns run along the axis closest to the surface
    /// normal at `center`; per column only the few cells the shell crosses are visited.
    pub(crate) fn shell_cells(&self, center: DVec3, lo: DVec3, hi: DVec3, cell: f64) -> Vec<(i64, i64, i64)> {
        self.shell_cells_in(center, lo, hi, cell, true)
    }

    /// [`World::shell_cells`] in any order (`sorted`: in the z, y, x loop order).
    pub(crate) fn shell_cells_in(&self, center: DVec3, lo: DVec3, hi: DVec3, cell: f64, sorted: bool) -> Vec<(i64, i64, i64)> {
        let m = 0.5 * cell + 200.0;
        // local geocentric radius range over the box: the ellipsoid radius changes by at most
        // (a − b)·|sin 2φ|·Δφ ≤ 2 (a − b) Δφ over the box's latitude span
        let half = ((hi - lo) + DVec3::ONE).max_element() * cell * 0.5 * 3f64.sqrt();
        let r = center.length();
        let sz = center.z / r.max(1.0);
        let (a, b) = (self.ell.a, self.ell.b);
        let r_gc = a * b / ((b * b * (1.0 - sz * sz)) + (a * a * sz * sz)).sqrt();
        let dr = 2.0 * (a - b) * (half / b) + 100.0;
        let (r_lo, r_hi) = (r_gc - dr - m, r_gc + dr + m);
        let c = center.to_array();
        let ax = (0..3).max_by(|&i, &j| c[i].abs().total_cmp(&c[j].abs())).unwrap();
        let (ua, va) = ((ax + 1) % 3, (ax + 2) % 3);
        let (lo, hi) = (lo.to_array(), hi.to_array());
        let sign = c[ax].signum();
        // squared extent of a cell interval [i, i+1] * cell: (min, max) of x²
        let sq = |i: i64| -> (f64, f64) {
            let (x0, x1) = (i as f64 * cell, (i + 1) as f64 * cell);
            let mx = (x0 * x0).max(x1 * x1);
            let mn = if x0 <= 0.0 && x1 >= 0.0 { 0.0 } else { (x0 * x0).min(x1 * x1) };
            (mn, mx)
        };
        let mut out = Vec::new();
        for iu in lo[ua] as i64..=hi[ua] as i64 {
            let (u2lo, u2hi) = sq(iu);
            for iv in lo[va] as i64..=hi[va] as i64 {
                let (v2lo, v2hi) = sq(iv);
                let dmax2 = r_hi * r_hi - u2lo - v2lo;
                if dmax2 <= 0.0 {
                    continue;
                }
                let dmin = (r_lo * r_lo - u2hi - v2hi).max(0.0).sqrt();
                let dmax = dmax2.sqrt();
                let (a0, a1) = if sign >= 0.0 { (dmin, dmax) } else { (-dmax, -dmin) };
                let k0 = ((a0 / cell).floor() as i64).max(lo[ax] as i64);
                let k1 = ((a1 / cell).floor() as i64).min(hi[ax] as i64);
                for k in k0..=k1 {
                    let mut idx = [0i64; 3];
                    idx[ax] = k;
                    idx[ua] = iu;
                    idx[va] = iv;
                    out.push((idx[0], idx[1], idx[2]));
                }
            }
        }
        if sorted {
            out.sort_unstable_by_key(|&(x, y, z)| (z, y, x));
        }
        out
    }

    /// The jittered lattice point of cell `c` of drainage level `lvl` (m, ECEF; not projected).
    pub(crate) fn lattice_point(&self, lvl: usize, c: Cell) -> DVec3 {
        let cell = self.cfg.hydro.levels[lvl].cell_km * KM;
        let hh = hash3(self.level_key(lvl), c.0, c.1, c.2);
        DVec3::new(c.0 as f64 + 0.5 + 0.8 * (u01k(hh, 1) - 0.5), c.1 as f64 + 0.5 + 0.8 * (u01k(hh, 2) - 0.5), c.2 as f64 + 0.5 + 0.8 * (u01k(hh, 3) - 0.5)) * cell
    }

    /// Cheap test without the cache: false only for lattice points certainly not active (far
    /// from the surface by a geocentric height estimate; `flow_point` decides with the geodetic
    /// height |h| < cell / 2, which differs from the estimate by far less than the margin).
    #[inline]
    pub(crate) fn maybe_active(&self, lvl: usize, c: (i64, i64, i64), cell: f64) -> bool {
        let hh = hash3(self.level_key(lvl), c.0, c.1, c.2);
        let p = DVec3::new(
            c.0 as f64 + 0.5 + 0.8 * (u01k(hh, 1) - 0.5),
            c.1 as f64 + 0.5 + 0.8 * (u01k(hh, 2) - 0.5),
            c.2 as f64 + 0.5 + 0.8 * (u01k(hh, 3) - 0.5),
        ) * cell;
        let r = p.length();
        let (a, b) = (self.ell.a, self.ell.b);
        let sz = p.z / r.max(1.0);
        let cz2 = 1.0 - sz * sz;
        let r_gc = a * b / ((b * b * cz2) + (a * a * sz * sz)).sqrt();
        (r - r_gc).abs() < 0.5 * cell + 200.0
    }

    /// No other point drains into `c` (cached).
    fn is_source(&self, lvl: usize, c: (i64, i64, i64)) -> bool {
        let key = (self.level_key(lvl) ^ self.cache_key ^ 0x50C, c.0, c.1, c.2);
        if let Some(v) = SRC.with(|m| m.borrow().get(&key).copied()) {
            return v;
        }
        let mut src = true;
        // targets lie within ±2 cells (see `flow_target`)
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
        SRC.with(|m| {
            let mut m = m.borrow_mut();
            if m.len() > 400_000 {
                m.clear();
            }
            m.insert(key, src);
        });
        src
    }

    /// The main stem among the channels joining at `t` (downstream `tt`): of the active points of
    /// its 1-ring that drain into it, the one whose inflow continues straightest into its outflow
    /// (the first of the best in z, y, x order); the others join it at an angle (cached).
    fn main_upstream(&self, lvl: usize, t: Cell, tt: Cell) -> Option<Cell> {
        let key = (self.level_key(lvl) ^ self.cache_key ^ 0x3A1, t.0, t.1, t.2);
        if let Some(v) = MAIN.with(|m| m.borrow().get(&key).copied()) {
            return v;
        }
        let tp = self.flow_point(lvl, t);
        let out = (self.flow_point(lvl, tt).s - tp.s).as_vec3().normalize();
        let mut best: Option<(f32, Cell)> = None;
        for dz in -1..=1i64 {
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let n = (t.0 + dx, t.1 + dy, t.2 + dz);
                    if n == t || !self.flow_point(lvl, n).active || self.flow_target(lvl, n) != Some(t) {
                        continue;
                    }
                    let sc = (tp.s - self.flow_point(lvl, n).s).as_vec3().normalize().dot(out);
                    if best.is_none_or(|b| sc > b.0) {
                        best = Some((sc, n));
                    }
                }
            }
        }
        let v = best.map(|b| b.1);
        MAIN.with(|m| {
            let mut m = m.borrow_mut();
            if m.len() > 400_000 {
                m.clear();
            }
            m.insert(key, v);
        });
        v
    }

    /// The rank of the tributary `c` among the channels joining the main stem `main` at `t`
    /// (the 1-ring of `t` in z, y, x order).
    fn tributary_rank(&self, lvl: usize, t: Cell, c: Cell, main: Cell) -> u32 {
        let mut r = 0;
        for dz in -1..=1i64 {
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let n = (t.0 + dx, t.1 + dy, t.2 + dz);
                    if n == c {
                        return r;
                    }
                    if n != t && n != main && self.flow_point(lvl, n).active && self.flow_target(lvl, n) == Some(t) {
                        r += 1;
                    }
                }
            }
        }
        r
    }

    /// All drainage edges whose valley could reach within `radius` of `center`.
    pub fn river_segments(&self, center: DVec3, radius: f64, gsd: f64) -> Vec<Seg> {
        self.river_segments_of(center, radius, gsd, 0..self.cfg.hydro.levels.len())
    }

    /// [`World::river_segments`] of the drainage levels `levels` only.
    pub fn river_segments_of(&self, center: DVec3, radius: f64, gsd: f64, levels: std::ops::Range<usize>) -> Vec<Seg> {
        let mut out = Vec::new();
        if !self.cfg.hydro.rivers {
            return out;
        }
        for (lvl, lc) in self.cfg.hydro.levels.iter().enumerate() {
            if !levels.contains(&lvl) {
                continue;
            }
            let first = out.len();
            let cell = lc.cell_km * KM;
            // skip levels far below the resolution (the carve fades out per pixel before that,
            // see `World::terrain_impl`, so tiles at slightly different GSD agree)
            if lc.valley_m < 0.2 * gsd && lc.width_m[1] < 0.15 * gsd {
                continue;
            }
            // a node's channels reach to the middle of its downstream node's outgoing edge
            // (edges up to 3 cells, see `flow_target`: a curve piece lies up to 4.5 cells from
            // the node that owns it; a shorter reach dropped pieces in some tiles only, leaving
            // elevation steps at tile edges)
            let reach = radius + 4.6 * cell + lc.valley_m;
            let lo = ((center - DVec3::splat(reach)) / cell).floor();
            let hi = ((center + DVec3::splat(reach)) / cell).floor();
            for c in self.shell_cells(center, lo, hi, cell) {
                if !self.maybe_active(lvl, c, cell) {
                    continue;
                }
                let fp = self.flow_point(lvl, c);
                if !fp.active || (fp.s - center).length() > reach {
                    continue;
                }
                self.node_pieces(lvl, c, &mut out);
            }
            // keep only the pieces that can reach the area: as conservative as selecting by the
            // owning node within radius + 2.2 cells + valley with ≤ 1.8-cell edges (every piece within
            // radius + 0.4 cell + valley of the centre), plus the meander warp; the far pieces of the
            // wide node search cost most of the river queries
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

    /// Wavelength (m) of the meander warp of drainage level `li` (its shorter octave: × 0.37).
    pub fn meander_wavelength(&self, li: usize) -> f64 {
        let lc = &self.cfg.hydro.levels[li];
        lc.meander.max(1e-3) * (lc.cell_km * KM)
    }

    /// The meander warp (east, north; × 0.22 wavelength) of drainage level `li` at `p`.
    pub fn meander_warp(&self, li: usize, p: DVec3) -> [f64; 2] {
        let lam = self.meander_wavelength(li);
        let k = self.level_key(li);
        let w1 = perlin3(k ^ 1, p / lam) + 0.45 * perlin3(k ^ 2, p / (0.37 * lam));
        let w2 = perlin3(k ^ 3, p / lam) + 0.45 * perlin3(k ^ 4, p / (0.37 * lam));
        [w1, w2]
    }

    /// Channels among `segs` whose valley may reach `ctx` with ground height `h` (meandered by
    /// a domain warp per level). The valley widens with the height above the floor, so the
    /// reach depends on `h`; the caller carves with every hit (min) and takes the channel
    /// attributes from the nearest one.
    /// `pre_warp`: the meander warps per level where known (from the tile's coarse grid).
    pub fn river_query(&self, ctx: &Ctx, segs: &[Seg], h: f64, pre_warp: &[Option<[f64; 2]>]) -> Vec<RiverHit> {
        let mut hits = Vec::new();
        let mut warped: Vec<Option<DVec3>> = vec![None; self.cfg.hydro.levels.len()];
        for s in segs {
            let li = s.level as usize;
            let pw = *warped[li].get_or_insert_with(|| {
                let lam = self.meander_wavelength(li);
                let [w1, w2] = pre_warp.get(li).copied().flatten().unwrap_or_else(|| self.meander_warp(li, ctx.p));
                ctx.p + (ctx.east * w1 + ctx.north * w2) * (0.22 * lam)
            });
            let ab = s.b - s.a;
            let l2 = ab.length_squared().max(1e-9);
            let u = ((pw - s.a).dot(ab) / l2).clamp(0.0, 1.0);
            let q = s.a + ab * u;
            let dv = pw - q;
            let dist = dv.length();
            let floor = s.ha + (s.hb - s.ha) * u;
            // bound of the carve extent (floodplain ≤ 9 hw × 1.25 edge noise, walls ≤ 6 m per m
            // above the floor)
            let reach = (s.valley * 1.5 + s.hw + 200.0).max(11.3 * s.hw + 6.0 * (h - floor + 2.0 + 0.04 * s.hw) + 50.0);
            if dist > reach {
                continue;
            }
            let sign = if ab.cross(dv).dot(ctx.up) >= 0.0 { 1.0 } else { -1.0 };
            hits.push(RiverHit {
                d: sign * dist,
                hw: s.hw + (s.hw_b - s.hw) * u,
                valley: s.valley,
                floor,
                level: s.level,
                u,
                spring: s.hw < 0.2 * s.hw_b,
                reach,
            });
        }
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Channels joining at a node no longer meet tangentially (both bent into the outflow, they
    /// ran side by side for kilometres): pieces of one level that end at the same point meet at
    /// an angle (a tributary's last reach ends on the main stem's bend at 35°–80°).
    #[test]
    fn tributaries_join_at_an_angle() {
        let w = World::new(crate::Config::default());
        let (mut pairs, mut joins, mut flat) = (0, 0, 0);
        for i in 0..400u64 {
            let h = mix64(0xC0F1 ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            let a = w.atlas().sample(p.normalize());
            if a.coast_km < 60.0 || a.glaciation > 0.3 {
                continue;
            }
            for lvl in 1..3 {
                let segs = w.river_segments_of(p, 15_000.0, 20.0, lvl..lvl + 1);
                for (k, s) in segs.iter().enumerate() {
                    if s.sink || s.hb <= 0.0 {
                        continue;
                    }
                    let ds = (s.b - s.a).normalize();
                    for o in &segs[k + 1..] {
                        if o.sink || o.hb <= 0.0 || o.b != s.b {
                            continue;
                        }
                        pairs += 1;
                        let ang = ds.dot((o.b - o.a).normalize()).clamp(-1.0, 1.0).acos().to_degrees();
                        // (the pieces are chords of curves that turn up to ~10° each)
                        if ang < 12.0 {
                            flat += 1;
                            let g = geodesy::ecef2geodetic(s.b, &w.ell);
                            eprintln!("level {lvl}: channels meet at {ang:.1}° at ({:.4}, {:.4})", g.lat.to_degrees(), g.lon.to_degrees());
                        } else {
                            joins += 1;
                        }
                    }
                }
            }
            if pairs > 400 {
                break;
            }
        }
        eprintln!("{pairs} pairs of channels meeting at a point: {joins} at an angle, {flat} tangential");
        assert!(joins > 20 && flat == 0, "{flat} of {pairs} meet tangentially");
    }
}
