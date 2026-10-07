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
use std::collections::HashMap;

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
}

#[derive(Clone, Copy)]
struct FlowPt {
    s: DVec3,
    h: f64,
    active: bool,
}

thread_local! {
    static PTS: RefCell<HashMap<(u64, i64, i64, i64), FlowPt>> = RefCell::new(HashMap::new());
    static TGT: RefCell<HashMap<(u64, i64, i64, i64), Option<(i64, i64, i64)>>> = RefCell::new(HashMap::new());
    static SRC: RefCell<HashMap<(u64, i64, i64, i64), bool>> = RefCell::new(HashMap::new());
}

impl World {
    fn level_key(&self, lvl: usize) -> u64 {
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
            let t = self.terrain_impl(&ctx, &m, Mode::Relief, None);
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
            // no lower neighbour: look a little farther for an outlet (often the sea beyond a
            // shallow dip of the coarse relief; rivers ended there on land in a blunt cap).
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
        TGT.with(|m| {
            let mut m = m.borrow_mut();
            if m.len() > 400_000 {
                m.clear();
            }
            m.insert(key, best);
        });
        best
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

    /// All drainage edges whose valley could reach within `radius` of `center`.
    pub fn river_segments(&self, center: DVec3, radius: f64, gsd: f64) -> Vec<Seg> {
        let mut out = Vec::new();
        if !self.cfg.hydro.rivers {
            return out;
        }
        for (lvl, lc) in self.cfg.hydro.levels.iter().enumerate() {
            let cell = lc.cell_km * KM;
            // skip levels far below the resolution (the carve fades out per pixel before that,
            // see `World::terrain_impl`, so tiles at slightly different GSD agree)
            if lc.valley_m < 0.2 * gsd && lc.width_m[1] < 0.15 * gsd {
                continue;
            }
            // a node's channels reach to the middle of its downstream node's outgoing edge
            // (edges are ≤ 1.8 cells, rarely up to 3, see `flow_target`)
            let reach = radius + 2.9 * cell + lc.valley_m;
            let lo = ((center - DVec3::splat(reach)) / cell).floor();
            let hi = ((center + DVec3::splat(reach)) / cell).floor();
            let width = |c: (i64, i64, i64)| {
                let hh = hash3(self.level_key(lvl) ^ 0x51DE, c.0, c.1, c.2);
                (0.5 * (lc.width_m[0] + (lc.width_m[1] - lc.width_m[0]) * u01k(hh, 1)), lc.valley_m * (0.6 + 0.8 * u01k(hh, 2)))
            };
            for cz in lo.z as i64..=hi.z as i64 {
                for cy in lo.y as i64..=hi.y as i64 {
                    for cx in lo.x as i64..=hi.x as i64 {
                        let c = (cx, cy, cz);
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
                        // a source (nothing drains into it): straight from the source point to the
                        // middle of its edge, where the bends start
                        if self.is_source(lvl, c) {
                            out.push(seg(fp.s, mid, fp.h, hmid, 0.08 * hw, hw)); // widening from a spring
                        }
                        match self.flow_target(lvl, tc) {
                            // the bend at the downstream node: a quadratic curve from the middle
                            // of this edge to the middle of the next (straight edges met in sharp
                            // corners: bulging wedges where a wide river turned)
                            Some(ttc) => {
                                let tq = self.flow_point(lvl, ttc);
                                let (hw2, _) = width(tc);
                                let mid2 = 0.5 * (tp.s + tq.s);
                                let hmid2 = 0.5 * (tp.h + tq.h);
                                let at = |t: f64| {
                                    let (u, v) = ((1.0 - t) * (1.0 - t), 2.0 * t * (1.0 - t));
                                    (mid * u + tp.s * v + mid2 * (t * t), hmid * u + tp.h * v + hmid2 * (t * t), hw + (hw2 - hw) * t)
                                };
                                // pieces of ~8 pixels at most (fewer at coarse zooms, where every
                                // pixel tests every piece)
                                let n = (((mid2 - mid).length() / (8.0 * gsd)).ceil() as usize).clamp(1, 6);
                                for k in 0..n {
                                    let (a, ha, wa) = at(k as f64 / n as f64);
                                    let (b, hb, wb) = at((k + 1) as f64 / n as f64);
                                    out.push(seg(a, b, ha, hb, wa, wb));
                                }
                            }
                            // into deep sea, or into a closed basin, where the river feeds a lake
                            // (it ended there in a blunt cap on land)
                            None => {
                                out.push(Seg { sink: tp.h > 0.0, ..seg(mid, tp.s, hmid, tp.h, hw, hw) });
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Channels among `segs` whose valley may reach `ctx` with ground height `h` (meandered by
    /// a domain warp per level). The valley widens with the height above the floor, so the
    /// reach depends on `h`; the caller carves with every hit (min) and takes the channel
    /// attributes from the nearest one.
    pub fn river_query(&self, ctx: &Ctx, segs: &[Seg], h: f64) -> Vec<RiverHit> {
        let mut hits = Vec::new();
        let mut warped: Vec<Option<DVec3>> = vec![None; self.cfg.hydro.levels.len()];
        for s in segs {
            let li = s.level as usize;
            let lc = &self.cfg.hydro.levels[li];
            let pw = *warped[li].get_or_insert_with(|| {
                let cell = lc.cell_km * KM;
                let lam = lc.meander.max(1e-3) * cell;
                let k = self.level_key(li);
                let w1 = perlin3(k ^ 1, ctx.p / lam) + 0.45 * perlin3(k ^ 2, ctx.p / (0.37 * lam));
                let w2 = perlin3(k ^ 3, ctx.p / lam) + 0.45 * perlin3(k ^ 4, ctx.p / (0.37 * lam));
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
            hits.push(RiverHit { d: sign * dist, hw: s.hw + (s.hw_b - s.hw) * u, valley: s.valley, floor, level: s.level });
        }
        hits
    }
}
