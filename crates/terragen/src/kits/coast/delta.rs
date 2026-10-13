//! Deltas at the mouths of the major rivers (`docs/design/terrain-next.md` §5.5): fan (arcuate)
//! and bird's-foot deltas of lobate flat land 0.5–2 m high that extends the shoreline, cut by
//! distributary channels; their cover follows the climate (marsh, mud flats, fields; mangroves
//! from the tropical kit's biomes). WGSL twin: `coast_delta_shape` and `coast_astamp` in
//! `gpu/wgsl/kits/coast_relief.wgsl`.
//!
//! **Mouths (host).** A delta sits where a level-0 river meets the *drawn* coast: along the
//! channel pieces of a drainage node ([`World::node_pieces`]), samples every 250 m are moved
//! by the inverse of the meander warp onto the drawn centre line, and the relief before the
//! drainage (`World::relief_height`, exact f64) is evaluated there. The mouth is the last
//! land-to-sea crossing of the node's channel after which it stays at sea, refined by
//! bisection; the node whose channel starts on land owns it. The direction is the drawn
//! channel's there. A pure function of the world (cached per node), the same for both
//! backends; it reaches pass A as a `kits::AStamp` (`Kit::astamps`) and pass B as a stamp
//! ([`TEMPLATE`]).
//!
//! **Relief (pass A, after the drainage carve).** In a frame at the mouth (apex a quarter radius
//! inland): a fan of ±72° with lobate margins (three angular harmonics) or four fingers along
//! the distributaries, the longest ahead (bird's-foot); inside, low ground is raised (and a little higher ground
//! flattened) to 2 m at the apex falling to 0.5 m at the radius; a shallow front (−2 m) around
//! the lobes; distributaries radiate from the apex with a gentle wiggle (fan: forking halfway),
//! carved to 1.5 m below the sea, so they hold sea water; the river's own channel upstream of
//! the mouth is kept. Lobes only grow over the shelf (fading out from −35 to −60 m). Channels are
//! box-filtered by the pixel (the mean below it), the margin's edge is at least a pixel wide.

use crate::kits::{astamp_frame, AStamp, AStampIn};
use crate::noise::*;
use crate::world::{Ctx, World};
use glam::{DVec3, Vec3};
use std::sync::RwLock;

/// `AStamp::kind` of a delta.
pub const KIND: u8 = 1;
/// The pass-B stamp of a delta (the coast kit's range 128–143).
pub const TEMPLATE: u32 = 128;
/// Deltas are listed for areas whose pixels are at most this large (m): coarser tiles do not
/// draw them (a 2–12 km delta covers a few pixels there; the drainage of level 0 itself is
/// listed up to 12.5 km).
pub const GSD_MAX: f64 = 3_000.0;
/// The drainage level whose rivers build deltas.
const LEVEL: usize = 0;
/// Sample spacing along a channel (m) and the pixel size of the samples (m).
const STEP: f64 = 250.0;
const SAMPLE_GSD: f64 = 30.0;
/// The largest delta radius (m), and the reach of a delta in radii (lobes and front).
const R_MAX: f64 = 12_000.0;
pub const REACH: f32 = 1.4;

/// Uniform in [0, 1) with 24 bits, sub-stream `k`: the GPU's `u01k` exactly.
#[inline]
pub(crate) fn u24(h: u64, k: u64) -> f32 {
    ((mix64(h ^ k.wrapping_mul(0xD6E8_FEB8_6659_FD93)) >> 40) as u32 as f32) * (1.0 / 16_777_216.0)
}

#[inline]
fn sm(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ============================================================================= mouths (host)

type Key = (u64, i64, i64, i64);
static CACHE: RwLock<Option<FxHashMap<Key, Option<AStamp>>>> = RwLock::new(None);

/// The deltas that can reach the disc (`center`, `radius`) at pixel size `gsd`
/// (`Kit::astamps`), by node in lattice order.
pub fn astamps(w: &World, center: DVec3, radius: f64, gsd: f64) -> Vec<AStamp> {
    if gsd > GSD_MAX || !w.cfg.hydro.rivers || w.cfg.hydro.levels.is_empty() {
        return Vec::new();
    }
    let cell = w.cfg.hydro.levels[LEVEL].cell_km * 1000.0;
    // a node's pieces lie within 4.5 cells of it (`river_segments`); a delta reaches 1.4 radii
    let reach = radius + 4.6 * cell + REACH as f64 * R_MAX;
    let lo = ((center - DVec3::splat(reach)) / cell).floor();
    let hi = ((center + DVec3::splat(reach)) / cell).floor();
    let cells: Vec<_> = w
        .shell_cells(center, lo, hi, cell)
        .into_iter()
        .filter(|&c| w.maybe_active(LEVEL, c, cell) && (w.lattice_point(LEVEL, c).normalize() * center.length() - center).length() <= reach + cell)
        .collect();
    use rayon::prelude::*;
    let found: Vec<Option<AStamp>> = cells.par_iter().map(|&c| node_delta(w, c)).collect();
    found.into_iter().flatten().filter(|st| (st.c - center).length() <= radius + st.r).collect()
}

/// The pass-B stamps of the deltas reaching an area (the coast kit's host function).
pub fn host(ctx: &mut crate::features::HostCtx, area: &crate::features::Area) -> crate::features::Features {
    let mut f = crate::features::Features::default();
    for st in astamps(ctx.world, area.center, area.radius, area.gsd) {
        let [_, _, up, d] = astamp_frame(st.c, st.bearing);
        let ex = d.as_dvec3();
        let ey = up.cross(d).as_dvec3();
        f.stamps.push(crate::features::FStamp {
            center: st.c,
            ex,
            ey,
            half: [st.r as f32, st.r as f32],
            template: TEMPLATE,
            h: st.v,
            v: [st.bearing as f32, st.r as f32, (st.id & 0xFF_FFFF) as f32, ((st.id >> 24) & 0xFF_FFFF) as f32, ((st.id >> 48) & 0xFFFF) as f32, 0.0, 0.0, 0.0],
        });
    }
    f
}

/// The delta at the mouth of node `c`'s channel, if any (cached).
fn node_delta(w: &World, c: (i64, i64, i64)) -> Option<AStamp> {
    let key = (w.cache_key, c.0, c.1, c.2);
    if let Some(v) = CACHE.read().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return *v;
    }
    let v = find(w, c);
    let mut g = CACHE.write().unwrap();
    let m = g.get_or_insert_with(FxHashMap::default);
    if m.len() > 200_000 {
        m.clear();
    }
    m.insert(key, v);
    v
}

/// The surface point below / above `q`.
fn surface(w: &World, q: DVec3) -> DVec3 {
    let g = geodesy::ecef2geodetic(q, &w.ell);
    geodesy::geodetic2ecef(geodesy::Geodetic::new(g.lat, g.lon, 0.0), &w.ell)
}

fn ctx_at(w: &World, p: DVec3, gsd: f64) -> Ctx {
    let g = geodesy::ecef2geodetic(p, &w.ell);
    Ctx::new(g.lat, g.lon, gsd, &w.ell)
}

/// The drawn centre line's point for the channel point `q`: pass A draws a channel where the
/// meander-warped pixel position lies on it (`World::river_query`), so the inverse of the warp
/// (a fixed-point iteration).
fn drawn(w: &World, q: DVec3) -> DVec3 {
    let k = 0.22 * w.meander_wavelength(LEVEL);
    let mut p = surface(w, q);
    for _ in 0..6 {
        let c = ctx_at(w, p, SAMPLE_GSD);
        let [w1, w2] = w.meander_warp(LEVEL, c.p);
        p = surface(w, q - (c.east * w1 + c.north * w2) * k);
    }
    p
}

/// The ground before the drainage at `p` (the coast as drawn, before the channels).
fn height(w: &World, p: DVec3) -> f64 {
    let c = ctx_at(w, p, SAMPLE_GSD);
    w.relief_height(&c, &w.macro_at(c.p, SAMPLE_GSD))
}

fn find(w: &World, c: (i64, i64, i64)) -> Option<AStamp> {
    // near a coast only (the atlas' coast distance at the lattice site: cheap)
    let q0 = w.lattice_point(LEVEL, c);
    if w.atlas().sample(q0.normalize()).coast_km.abs() > 80.0 {
        return None;
    }
    let mut pieces = Vec::new();
    w.node_pieces(LEVEL, c, &mut pieces);
    if pieces.is_empty() {
        return None;
    }
    // (the drainage floor far above or below the sea all along)
    let (lo, hi) = pieces.iter().fold((f64::MAX, f64::MIN), |a, s| (a.0.min(s.ha.min(s.hb)), a.1.max(s.ha.max(s.hb))));
    if lo > 400.0 || hi < -400.0 {
        return None;
    }
    let lens: Vec<f64> = pieces.iter().map(|s| (s.b - s.a).length()).collect();
    let total: f64 = lens.iter().sum();
    // the point at arc length `x` along the pieces, and the channel's half width there
    let at = |x: f64| -> (DVec3, f64) {
        let mut x = x.clamp(0.0, total);
        for (s, &l) in pieces.iter().zip(&lens) {
            if x <= l {
                let t = x / l.max(1e-9);
                return (s.a + (s.b - s.a) * t, s.hw + (s.hw_b - s.hw) * t);
            }
            x -= l;
        }
        let s = pieces.last().unwrap();
        (s.b, s.hw_b)
    };
    let n = ((total / STEP).ceil() as usize).max(2);
    let xs: Vec<f64> = (0..=n).map(|i| total * i as f64 / n as f64).collect();
    // starting at sea: the mouth is an upstream node's; ending on land: none (the two ends
    // first, the same samples as below: most nodes stop here)
    let (h_first, h_last) = (height(w, drawn(w, at(xs[0]).0)), height(w, drawn(w, at(xs[n]).0)));
    if h_first < 0.0 || h_last >= 0.0 {
        return None;
    }
    let hs: Vec<f64> = xs
        .iter()
        .enumerate()
        .map(|(k, &x)| {
            if k == 0 {
                h_first
            } else if k == n {
                h_last
            } else {
                height(w, drawn(w, at(x).0))
            }
        })
        .collect();
    // the last sample on land, after which the channel stays at sea
    let i = hs.iter().rposition(|&h| h >= 0.0)?;
    if i == n {
        return None;
    }
    let (mut a, mut b) = (xs[i], xs[i + 1]);
    for _ in 0..8 {
        let m = 0.5 * (a + b);
        if height(w, drawn(w, at(m).0)) >= 0.0 {
            a = m;
        } else {
            b = m;
        }
    }
    let xm = 0.5 * (a + b);
    let (qm, hw) = at(xm);
    let m = drawn(w, qm);
    let (d0, d1) = (drawn(w, at(xm - 150.0).0), drawn(w, at(xm + 150.0).0));
    let up = m.normalize();
    let dv = d1 - d0;
    let dir = (dv - up * dv.dot(up)).normalize_or_zero();
    if dir == DVec3::ZERO {
        return None;
    }
    let id = (hash3(w.seed() ^ 0xDE17_A000, c.0, c.1, c.2) & !0xFF) | KIND as u64;
    if u24(id, 1) > 0.85 {
        return None;
    }
    let r = ((2_500.0 + 6_000.0 * u24(id, 3) as f64) * (hw / 60.0).sqrt()).clamp(2_000.0, R_MAX);
    // a delta grows on the shelf
    if height(w, surface(w, m + dir * 0.4 * r)) < -60.0 {
        return None;
    }
    let [east, north, _, _] = astamp_frame(m, 0.0);
    let d32 = dir.as_vec3();
    let bearing = (d32.dot(north) as f64).atan2(d32.dot(east) as f64);
    Some(AStamp { id, c: m, bearing, r: REACH as f64 * r, v: hw as f32 })
}

// ============================================================================= shape

/// A delta's geometry at a point (f32, as WGSL `coast_delta_shape`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Shape {
    /// the delta plain: 1 inside the lobes (a soft edge of 5 % of the radius, at least a pixel)
    pub plain: f32,
    /// its height (m): 2 at the apex, 0.5 at the radius
    pub hd: f32,
    /// the shallow front around the lobes (0..1)
    pub front: f32,
    /// the distributaries' share of the pixel (box-filtered)
    pub chan: f32,
    /// the along-coordinate from the mouth (m, + seaward) and the radius
    pub u: f32,
    pub r0: f32,
}

/// The fingers of a bird's-foot delta: angle (rad from the river's direction), length (radii),
/// angular half width (rad).
#[inline]
fn finger(id: u64, k: u32) -> (f32, f32, f32) {
    // the river's own finger ahead (the longest), one on either side, a fourth on a side by hash
    let base = [0.0, -0.52, 0.5, if u24(id, 19) < 0.5 { -0.98 } else { 0.98 }][k as usize];
    let a = base + 0.16 * (u24(id, 20 + k as u64) - 0.5);
    let l = if k == 0 { 0.8 + 0.2 * u24(id, 10) } else { 0.5 + 0.4 * u24(id, 10 + k as u64) };
    (a, l, 0.09 + 0.05 * u24(id, 14 + k as u64))
}

/// The shape of the delta (centre `c`, `bearing`, `reach`, `id`, the river's half width `hw`
/// at the mouth) at `p` for pixels of `gsd` m.
pub fn shape(c: DVec3, bearing: f64, reach: f32, id: u64, hw: f32, p: DVec3, gsd: f32) -> Shape {
    let f = astamp_frame(c, bearing);
    let (up, d) = (f[2], f[3]);
    let left = up.cross(d);
    let dp: Vec3 = (p - c).as_vec3();
    let (u, v) = (dp.dot(d), dp.dot(left));
    let r0 = reach / REACH;
    let bird = u24(id, 4) < 0.35;
    // polar coordinates from the apex, a quarter radius inland
    let x = u + 0.25 * r0;
    let r = (x * x + v * v).sqrt();
    let phi = v.atan2(x);
    let tau = std::f32::consts::TAU;
    // the margin: a fan of ±72° with lobes, or fingers along the distributaries
    let g = (1.0 - (phi / 1.25) * (phi / 1.25)).max(0.0);
    let mut rm;
    if bird {
        rm = 0.4 * r0 * g.sqrt();
        for k in 0..4u32 {
            let (a, l, wk) = finger(id, k);
            let e = (phi - a) / wk;
            rm = rm.max(l * r0 * (-e * e).exp());
        }
    } else {
        let lob = 0.5 * (5.0 * phi + tau * u24(id, 5)).cos() + 0.3 * (9.0 * phi + tau * u24(id, 6)).cos() + 0.2 * (14.0 * phi + tau * u24(id, 7)).cos();
        rm = r0 * g.sqrt() * (1.0 + 0.14 * lob);
    }
    let edge = (0.05 * r0).max(gsd);
    let plain = if rm > 0.0 { sm(rm, rm - edge, r) } else { 0.0 };
    let hd = 0.5 + 1.5 * (1.0 - r / r0).clamp(0.0, 1.0);
    let front = if rm > 0.0 { sm(1.3 * rm + edge, rm, r) * (1.0 - plain) } else { 0.0 };
    // distributaries from the apex: along the fingers (bird's-foot) or five forking ones (fan)
    // (half width at the apex: half the river's, at least 10 m; narrowing seaward)
    let cw0 = (0.5 * hw).max(10.0);
    let mut chan = 0.0f32;
    let nk = if bird { 4u32 } else { 5 };
    for k in 0..nk {
        let mut a = if bird { finger(id, k).0 } else { (k as f32 - 2.0) * 0.42 + 0.14 * (u24(id, 24 + k as u64) - 0.5) };
        a += 0.05 * (r / (0.21 * r0) + tau * u24(id, 30 + k as u64)).sin();
        // (a fan's channels fork halfway)
        let fork = if bird { 0.0 } else { 0.11 * sm(0.5 * r0, 0.75 * r0, r) };
        for s in [-1.0f32, 1.0] {
            let dphi = (phi - a - s * fork).abs();
            if dphi > 1.2 {
                continue;
            }
            let dist = r * dphi.sin();
            let cw = (1.0 - 0.45 * (r / r0).min(1.0)) * cw0;
            let cov = ((dist + 0.5 * gsd).min(cw) - (dist - 0.5 * gsd).max(-cw)) / gsd;
            let ext = sm(rm + 2.0 * cw + edge, rm, r);
            chan = chan.max(cov.clamp(0.0, 1.0) * ext);
            if fork == 0.0 {
                break;
            }
        }
    }
    Shape { plain, hd, front, chan, u, r0 }
}

/// A delta's relief (`Kit::astamp`): the plain, the front and the distributaries.
pub fn relief(_w: &World, st: &AStamp, ai: &AStampIn, h: &mut f64) {
    if st.kind() != KIND {
        return;
    }
    let sh = shape(st.c, st.bearing, st.r as f32, st.id, st.v, ai.ctx.p, ai.ctx.gsd as f32);
    if sh.plain <= 0.0 && sh.front <= 0.0 && sh.chan <= 0.0 {
        return;
    }
    let h0 = *h as f32;
    // the river's own channel upstream of the mouth stays
    let keep = if ai.river_hw > 0.0 && (ai.river_d.abs() as f32) < 1.2 * ai.river_hw as f32 { 1.0 - sm(-0.05 * sh.r0, 0.1 * sh.r0, sh.u) } else { 0.0 };
    // lobes grow on the shelf only
    let shelf = sm(-60.0, -35.0, h0);
    let ht = if h0 < sh.hd {
        sh.hd
    } else if h0 < sh.hd + 4.0 {
        sh.hd + (h0 - sh.hd) * (h0 - sh.hd) / 4.0
    } else {
        h0
    };
    let mut hn = h0 + (ht - h0) * sh.plain * shelf * (1.0 - keep);
    if hn < -2.0 {
        hn += (-2.0 - hn) * sh.front * shelf;
    }
    hn += (hn.min(-1.5) - hn) * sh.chan * (1.0 - keep);
    *h += (hn - h0) as f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> World {
        World::new(crate::Config::default())
    }

    /// Coastal points of the default world (atlas coast within 30 km, land).
    fn coastal_points(w: &World, n: usize) -> Vec<DVec3> {
        let mut v = Vec::new();
        for i in 0..20_000u64 {
            let h = mix64(0xDE17 ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            if lat.abs() > 1.1 {
                continue;
            }
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            let a = w.atlas().sample(p.normalize());
            if a.coast_km > 0.0 && a.coast_km < 30.0 && a.glaciation < 0.3 {
                v.push(p);
                if v.len() == n {
                    break;
                }
            }
        }
        v
    }

    fn ctx(w: &World, p: DVec3, gsd: f64) -> Ctx {
        ctx_at(w, p, gsd)
    }

    /// Deltas sit where a major river meets the drawn coast: the mouth is on the coast line of
    /// the relief (|h| < 1 m), the river's channel passes it, and the delta extends the land
    /// seaward (its plain is above the sea in front of the mouth).
    #[test]
    fn deltas_sit_at_river_mouths() {
        let w = world();
        let mut seen = std::collections::BTreeMap::new();
        for p in coastal_points(&w, 24) {
            for st in astamps(&w, p, 150_000.0, 100.0) {
                seen.insert(st.id, st);
            }
        }
        let (mut n, mut on_river, mut grown) = (0, 0, 0);
        for st in seen.values() {
            n += 1;
            let h = height(&w, st.c);
            assert!(h.abs() < 1.0, "mouth {:?} at {h:.2} m", st.c);
            let t = w.terrain(&ctx(&w, st.c, 20.0));
            if t.river_d.abs() < t.river_hw + 60.0 {
                on_river += 1;
            }
            let [_, _, _, d] = astamp_frame(st.c, st.bearing);
            let r0 = st.r / REACH as f64;
            let q = surface(&w, st.c + d.as_dvec3() * 0.35 * r0);
            let tq = w.terrain(&ctx(&w, q, 20.0));
            // (the plain, or a distributary cut into it ahead of the mouth)
            if tq.ground > -1.6 && tq.ground < 2.5 {
                grown += 1;
            }
            let g = geodesy::ecef2geodetic(st.c, &w.ell);
            eprintln!(
                "delta ({:.4}, {:.4}) r {:.1} km {} hw {:.0} m: river d {:.0} hw {:.0}; seaward ground {:.1} kind {}; {:.0} °C {:.0} mm",
                g.lat.to_degrees(),
                g.lon.to_degrees(),
                r0 / 1000.0,
                if u24(st.id, 4) < 0.35 { "bird's-foot" } else { "fan" },
                st.v,
                t.river_d,
                t.river_hw,
                tq.ground,
                tq.water_kind,
                w.atlas().sample(st.c.normalize()).temp_c,
                w.atlas().sample(st.c.normalize()).precip_mm
            );
        }
        eprintln!("{n} deltas: {on_river} on their river at the mouth, {grown} grown seaward");
        assert!(n >= 5, "{n} deltas");
        assert!(on_river * 10 >= n * 8, "{on_river} of {n} on their river");
        assert!(grown * 10 >= n * 7, "{grown} of {n} grown seaward");
    }

    /// The same deltas from any query (pure function of the world).
    #[test]
    fn deltas_are_tile_independent() {
        let w = world();
        let p = coastal_points(&w, 3)[2];
        let a = astamps(&w, p, 120_000.0, 100.0);
        *CACHE.write().unwrap() = None;
        let up = p.normalize();
        let east = DVec3::Z.cross(up).normalize();
        let b = astamps(&w, p + east * 50_000.0, 200_000.0, 20.0);
        for st in &a {
            assert!(b.contains(st), "{st:?}");
        }
    }
}
