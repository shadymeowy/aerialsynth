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
}

impl World {
    fn level_key(&self, lvl: usize) -> u64 {
        self.seed.rotate_left(23) ^ (lvl as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    fn flow_point(&self, lvl: usize, c: (i64, i64, i64)) -> FlowPt {
        let key = (self.level_key(lvl), c.0, c.1, c.2);
        if let Some(p) = PTS.with(|m| m.borrow().get(&key).copied()) {
            return p;
        }
        let cell = self.cfg.hydro.levels[lvl].cell_km * KM;
        let hh = hash3(key.0, c.0, c.1, c.2);
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
        let key = (self.level_key(lvl), c.0, c.1, c.2);
        if let Some(t) = TGT.with(|m| m.borrow().get(&key).copied()) {
            return t;
        }
        let cell = self.cfg.hydro.levels[lvl].cell_km * KM;
        let me = self.flow_point(lvl, c);
        let mut best = None;
        if me.active && me.h > 0.0 {
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

    /// All drainage edges whose valley could reach within `radius` of `center`.
    pub fn river_segments(&self, center: DVec3, radius: f64, gsd: f64) -> Vec<Seg> {
        let mut out = Vec::new();
        if !self.cfg.hydro.rivers {
            return out;
        }
        for (lvl, lc) in self.cfg.hydro.levels.iter().enumerate() {
            let cell = lc.cell_km * KM;
            // skip levels that cannot be resolved at all (channel and valley sub-pixel)
            if lc.valley_m < 0.5 * gsd && lc.width_m[1] < 0.3 * gsd {
                continue;
            }
            let reach = radius + 2.0 * cell + lc.valley_m + 0.2 * cell;
            let lo = ((center - DVec3::splat(reach)) / cell).floor();
            let hi = ((center + DVec3::splat(reach)) / cell).floor();
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
                        let hh = hash3(self.level_key(lvl) ^ 0x51DE, cx, cy, cz);
                        let w = lc.width_m[0] + (lc.width_m[1] - lc.width_m[0]) * u01k(hh, 1);
                        out.push(Seg { a: fp.s, b: tp.s, ha: fp.h, hb: tp.h, level: lvl as u8, hw: 0.5 * w, valley: lc.valley_m * (0.6 + 0.8 * u01k(hh, 2)) });
                    }
                }
            }
        }
        out
    }

    /// Closest channel to `ctx` among `segs` (meandered by a domain warp per level).
    pub fn river_query(&self, ctx: &Ctx, segs: &[Seg]) -> Option<RiverHit> {
        let mut best: Option<RiverHit> = None;
        let mut warped: [Option<DVec3>; 4] = [None; 4];
        for s in segs {
            let li = s.level as usize;
            let lc = &self.cfg.hydro.levels[li];
            let pw = *warped[li.min(3)].get_or_insert_with(|| {
                let cell = lc.cell_km * KM;
                let lam = lc.meander * cell;
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
            if dist > s.valley * 1.5 + s.hw + 200.0 {
                continue;
            }
            let sign = if ab.cross(dv).dot(ctx.up) >= 0.0 { 1.0 } else { -1.0 };
            let floor = s.ha + (s.hb - s.ha) * u;
            let better = match &best {
                None => true,
                Some(b) => dist - s.hw < b.d.abs() - b.hw,
            };
            if better {
                best = Some(RiverHit { d: sign * dist, hw: s.hw, valley: s.valley, floor, level: s.level });
            }
        }
        best
    }
}
