//! Routing one edge of the network over the terrain (`docs/design/terrain-next.md` §6.3): a
//! dynamic programme on a corridor grid between the ends (bounded: ≤ 41 × 11 nodes), refined
//! piece by piece where the coarse grid is coarser than 260 m (≤ 41 × 7 nodes per piece), costs
//! from the CPU-analytic relief (no drainage: the same on both backends), Chaikin smoothing, and
//! switchbacks where a minor road climbs steeper than it may. A pure function of the edge.

use super::cls;
use crate::world::{Ctx, World};
use glam::DVec3;

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
    /// Chaikin rounds (0: straight between the grid nodes)
    pub smooth: u32,
}

#[rustfmt::skip]
pub const PARAMS: [Params; cls::N] = [
    // motorway, trunk, secondary, local, track, ramp, rail, power
    Params { step: 2_500.0, width: 0.16, k: 300.0, k2: 30_000.0, glim: 0.05, gmax: 0.0, turn: 150.0, wet: 40.0, town: 6.0, smooth: 3 },
    Params { step: 1_500.0, width: 0.18, k: 160.0, k2: 12_000.0, glim: 0.07, gmax: 0.0, turn: 60.0, wet: 25.0, town: 2.0, smooth: 3 },
    Params { step: 400.0, width: 0.2, k: 100.0, k2: 6_000.0, glim: 0.10, gmax: 0.11, turn: 15.0, wet: 20.0, town: 0.0, smooth: 3 },
    Params { step: 150.0, width: 0.22, k: 60.0, k2: 4_000.0, glim: 0.12, gmax: 0.13, turn: 6.0, wet: 15.0, town: 0.0, smooth: 3 },
    Params { step: 100.0, width: 0.22, k: 40.0, k2: 3_000.0, glim: 0.16, gmax: 0.16, turn: 4.0, wet: 10.0, town: 0.0, smooth: 3 },
    Params { step: 100.0, width: 0.0, k: 0.0, k2: 0.0, glim: 1.0, gmax: 0.0, turn: 0.0, wet: 0.0, town: 0.0, smooth: 0 },
    Params { step: 2_500.0, width: 0.16, k: 2_500.0, k2: 80_000.0, glim: 0.022, gmax: 0.0, turn: 200.0, wet: 40.0, town: 6.0, smooth: 3 },
    Params { step: 2_500.0, width: 0.08, k: 4.0, k2: 40.0, glim: 0.3, gmax: 0.0, turn: 30.0, wet: 0.5, town: 3.0, smooth: 0 },
];

/// The largest lateral deviation of a route of length `d` from its chord (m).
pub fn max_deviation(class: usize, d: f64) -> f64 {
    let p = &PARAMS[class];
    p.width * d + 300.0 + if p.gmax > 0.0 { 250.0 } else { 0.0 }
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

/// Built-up areas a route avoids (centre, radius).
pub type Avoid = [(DVec3, f64)];

/// Grid DP between `a` and `b` over a corridor of `na` nodes along and `nc` across (half width
/// `half`, tapered to the ends): the node polyline and the metres over the sea. Monotone along
/// the edge; ties broken by node index (strict `<` in index order).
#[allow(clippy::too_many_arguments)]
fn corridor(w: &World, a: DVec3, b: DVec3, na: usize, nc: usize, half: f64, pr: &Params, avoid: &Avoid) -> Option<(Vec<DVec3>, f64)> {
    let c = nc / 2;
    let dy = if c > 0 { half / c as f64 } else { 0.0 };
    let lat = (a + b).normalize().cross(b - a).normalize_or_zero();
    let gsd = 0.5 * (b - a).length() / (na - 1) as f64;
    let mut pos = vec![DVec3::ZERO; na * nc];
    let mut h = vec![0.0f64; na * nc];
    let mut town = vec![0.0f64; na * nc];
    for i in 0..na {
        let t = i as f64 / (na - 1) as f64;
        let taper = (std::f64::consts::PI * t).sin().sqrt();
        let base = a + (b - a) * t;
        for j in 0..nc {
            let k = i * nc + j;
            if (i == 0 || i == na - 1) && j != c {
                h[k] = f64::NAN;
                continue;
            }
            let p = surface(w, base + lat * ((j as f64 - c as f64) * dy * taper));
            pos[k] = p;
            h[k] = relief_h(w, p, gsd);
            town[k] = if avoid.iter().any(|&(q, r)| (p - q).length() < r) { 1.0 } else { 0.0 };
        }
    }
    pos[c] = a;
    pos[(na - 1) * nc + c] = b;
    const DJ: i64 = 2;
    let nd = (2 * DJ + 1) as usize;
    let idx = |i: usize, j: usize, k: usize| (i * nc + j) * nd + k;
    let mut cost = vec![f64::INFINITY; na * nc * nd];
    let mut from = vec![u32::MAX; na * nc * nd];
    let mut water = vec![0.0f64; na * nc * nd];
    cost[idx(0, c, DJ as usize)] = 0.0;
    for i in 0..na - 1 {
        for j in 0..nc {
            for k in 0..nd {
                let c0 = cost[idx(i, j, k)];
                if !c0.is_finite() {
                    continue;
                }
                let (p0, h0) = (pos[i * nc + j], h[i * nc + j]);
                for dj in -DJ..=DJ {
                    let j1 = j as i64 + dj;
                    if j1 < 0 || j1 >= nc as i64 {
                        continue;
                    }
                    let j1 = j1 as usize;
                    let n1 = (i + 1) * nc + j1;
                    let h1 = h[n1];
                    if !h1.is_finite() {
                        continue;
                    }
                    let l = (pos[n1] - p0).length().max(1.0);
                    let g = (h1 - h0).abs() / l;
                    let over = (g - pr.glim).max(0.0);
                    let wet = if h0 < 0.5 || h1 < 0.5 { l } else { 0.0 };
                    let turn = pr.turn * dy / 100.0 * (dj - (k as i64 - DJ)).abs() as f64;
                    let inside = 0.5 * (town[i * nc + j] + town[n1]) * l;
                    let step = l * (1.0 + pr.k * g * g + pr.k2 * over * over) + pr.wet * wet + pr.town * inside + turn;
                    let k1 = (dj + DJ) as usize;
                    let c1 = c0 + step;
                    let t = idx(i + 1, j1, k1);
                    if c1 < cost[t] {
                        cost[t] = c1;
                        from[t] = idx(i, j, k) as u32;
                        water[t] = water[idx(i, j, k)] + wet;
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
    Some((path, water[best.1]))
}

/// Route from `a` to `b` (surface points) for `class`. None: the edge would cross too much sea.
pub fn route(w: &World, a: DVec3, b: DVec3, class: usize, avoid: &Avoid) -> Option<Line> {
    let pr = &PARAMS[class];
    let d = (b - a).length();
    if d < 1.0 {
        return None;
    }
    let na = ((d / pr.step).ceil() as usize).clamp(3, 40) + 1;
    let nc = if pr.width > 0.0 { 11 } else { 1 };
    let half = pr.width * d + 300.0 * (pr.width > 0.0) as u32 as f64;
    let (coarse, wet) = corridor(w, a, b, na, nc, half, pr, avoid)?;
    if wet > (0.12 * d).clamp(400.0, 3_000.0) {
        return None;
    }
    // refine the pieces of a coarse grid (lateral resolution: a third of the coarse spacing)
    let step = d / (na - 1) as f64;
    let mut pts = if step > 260.0 && class != cls::POWER {
        let mut v: Vec<DVec3> = vec![coarse[0]];
        for k in 0..coarse.len() - 1 {
            let (p, q) = (coarse[k], coarse[k + 1]);
            let l = (q - p).length();
            let nf = ((l / 120.0).ceil() as usize).clamp(3, 40) + 1;
            let hf = (0.5 * half / 5.0).min(0.25 * l) + 60.0;
            match corridor(w, p, q, nf, 7, hf, pr, avoid) {
                Some((f, _)) => v.extend_from_slice(&f[1..]),
                None => v.push(q),
            }
        }
        v
    } else {
        coarse
    };
    if pr.smooth > 0 {
        pts = chaikin(w, &pts, pr.smooth as usize);
    }
    if pr.gmax > 0.0 {
        pts = switchbacks(w, &pts, pr);
    }
    Some(Line::new(pts))
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
