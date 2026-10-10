//! Computation of the atlas fields. Everything is f64 and deterministic: parallel maps over
//! texels (each a pure function of the inputs), iterations that read only the previous buffer
//! (jump flooding, moisture advection), and sequential steps in a fixed order (plate sites,
//! hotspot tracks, culture ids).

use super::geom::{ExtGrid, Grid};
use super::*;
use crate::noise::*;
use rayon::prelude::*;

const KM: f64 = 1000.0;
const NONE: u32 = u32::MAX;
/// Relative humidity of the air over the open ocean.
const RH_OCEAN: f64 = 0.8;
/// Relaxation time of the evaporation over the ocean (days).
const TAU_EVAP: f64 = 1.5;
/// Orographic rain-out per metre of ascent (the moisture left after rising `dh` m is
/// exp(−K_ORO·dh), on top of the lower saturation of the cooled air), and the suppression of
/// rain per metre of descent (föhn).
const K_ORO: f64 = 2.5e-4;
const K_DESCENT: f64 = 1.0e-3;
/// e-folding length (m) of the maritime influence over land, along the wind.
const L_MARITIME: f64 = 1200.0 * KM;
/// Scale (m²/s) of the stream function of the wind perturbations.
const PSI_SCALE: f64 = 2.4e6;

/// Stage timings (printed with `TERRAGEN_PROFILE`).
struct Lap(Option<std::time::Instant>);
impl Lap {
    fn new() -> Self {
        Lap(std::env::var_os("TERRAGEN_PROFILE").map(|_| std::time::Instant::now()))
    }
    fn lap(&mut self, what: &str) {
        if let Some(t) = self.0 {
            eprintln!("  atlas: {what} {:.3} s", t.elapsed().as_secs_f64());
            self.0 = Some(std::time::Instant::now());
        }
    }
}

fn par<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    (0..n).into_par_iter().map(f).collect()
}

/// Geography of a direction on the world's ellipsoid.
#[derive(Clone, Copy)]
struct Geo {
    dir: DVec3,
    /// ECEF point on the ellipsoid (m)
    p: DVec3,
    /// geodetic latitude (degrees)
    lat: f64,
    east: DVec3,
    north: DVec3,
}

fn geo(w: &World, dir: DVec3) -> Geo {
    let (a, b) = (w.ell.a, w.ell.b);
    let rho = (dir.x * dir.x + dir.y * dir.y).sqrt();
    let r = 1.0 / (rho * rho / (a * a) + dir.z * dir.z / (b * b)).sqrt();
    let lat = (dir.z * a * a).atan2(rho * b * b);
    let lon = dir.y.atan2(dir.x);
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    Geo { dir, p: dir * r, lat: lat.to_degrees(), east: DVec3::new(-so, co, 0.0), north: DVec3::new(-sl * co, -sl * so, cl) }
}

/// Angle (radians) between unit vectors, from their chord.
#[inline]
fn angle(a: DVec3, b: DVec3) -> f64 {
    2.0 * (0.5 * (a - b).length()).min(1.0).asin()
}

#[inline]
fn gauss(d: f64, mu: f64, w: f64) -> f64 {
    (-((d - mu) / w).powi(2)).exp()
}

/// A uniform direction from a hash.
fn hash_dir(h: u64) -> DVec3 {
    let z = 2.0 * u01k(h, 1) - 1.0;
    let ph = std::f64::consts::TAU * u01k(h, 2);
    let s = (1.0 - z * z).max(0.0).sqrt();
    DVec3::new(s * ph.cos(), s * ph.sin(), z)
}

/// Saturation precipitable water (mm) at temperature `t` (°C) (Clausius–Clapeyron-like).
#[inline]
fn qsat(t: f64) -> f64 {
    50.0 * (0.065 * (t.clamp(-60.0, 40.0) - 27.0)).exp()
}

/// Warmest − coldest month (°C) far inland at latitude `lat` (degrees).
#[inline]
fn range_continental(lat: f64) -> f64 {
    1.5 + 50.0 * lat.abs().to_radians().sin().powf(1.4)
}

/// Base precipitation rate (1/day) of the atmospheric moisture at (seasonally shifted)
/// latitude `lat`: ITCZ convergence, subtropical highs (of strength `high`: weak over heated
/// continents in summer), storm tracks.
#[inline]
fn rain_rate(lat: f64, high: f64) -> f64 {
    let a = lat.abs();
    0.04 + 0.14 * gauss(lat, 0.0, 8.0) - 0.034 * high * gauss(a, 26.0, 8.0) + 0.17 * gauss(a, 52.0, 14.0)
}

/// Jump flooding: per texel the nearest seed texel (by the chord between texel centres; ties to
/// the lower index), `NONE` without seeds.
fn jfa(e: &ExtGrid, dirs: &[DVec3], seed: &[bool]) -> Vec<u32> {
    let g = &e.g;
    let n = g.n();
    let mut cur: Vec<u32> = (0..n).map(|k| if seed[k] { k as u32 } else { NONE }).collect();
    let mut steps = Vec::new();
    let mut s = g.r / 2;
    while s >= 1 {
        steps.push(s as i64);
        s /= 2;
    }
    steps.push(1);
    for st in steps {
        let prev = &cur;
        cur = par(n, |t| {
            let (f, i, j) = g.fij(t);
            let d = dirs[t];
            let mut best = prev[t];
            let mut bd = if best == NONE { f64::MAX } else { dirs[best as usize].distance_squared(d) };
            for (di, dj) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                let Some(m) = e.at(f, i as i64 + di * st, j as i64 + dj * st) else { continue };
                let s = prev[m];
                if s == NONE || s == best {
                    continue;
                }
                let dd = dirs[s as usize].distance_squared(d);
                if dd < bd || (dd == bd && s < best) {
                    best = s;
                    bd = dd;
                }
            }
            best
        });
    }
    cur
}

/// Tangent gradient (per radian of arc) of field `v`, by central differences over the edge
/// neighbours.
fn gradient(dirs: &[DVec3], nb: &[[usize; 4]], v: &[f64]) -> Vec<DVec3> {
    par(dirs.len(), |k| {
        let d = dirs[k];
        let [a, b, c, e] = nb[k];
        let t = |x: DVec3| x - d * d.dot(x);
        let (e1, e2) = (t(dirs[a] - dirs[b]), t(dirs[c] - dirs[e]));
        let (f1, f2) = (v[a] - v[b], v[c] - v[e]);
        let (g11, g12, g22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
        let det = g11 * g22 - g12 * g12;
        if det <= 1e-12 * (g11 * g22).max(1e-300) {
            return DVec3::ZERO;
        }
        let al = (f1 * g22 - f2 * g12) / det;
        let be = (f2 * g11 - f1 * g12) / det;
        e1 * al + e2 * be
    })
}

/// Nominal texel spacing (radians) of a grid of `n` texels.
fn texel_spacing(n: usize) -> f64 {
    std::f64::consts::FRAC_PI_2 / ((n / 6) as f64).sqrt()
}

/// `iters` steps of neighbour smoothing (a Gaussian of σ = √iters / 2 texels).
fn smooth(nb: &[[usize; 4]], v: &[f64], iters: usize) -> Vec<f64> {
    let mut v = v.to_vec();
    for _ in 0..iters {
        let o = &v;
        v = par(o.len(), |k| 0.5 * o[k] + 0.125 * nb[k].iter().map(|&m| o[m]).sum::<f64>());
    }
    v
}

/// Average of the 2 × 2 children of each texel of the half-resolution grid.
fn downsample(g2: &Grid, g: &Grid, v: &[f64]) -> Vec<f64> {
    par(g2.n(), |k| {
        let (f, i, j) = g2.fij(k);
        let c = |di: usize, dj: usize| v[g.idx(f, 2 * i + di, 2 * j + dj)];
        0.25 * (c(0, 0) + c(1, 0) + c(0, 1) + c(1, 1))
    })
}

/// Bilinear interpolation of a field of the half-resolution grid `g2` at the texels of `g`
/// (the warped coordinates nest: fine texel i is at coarse i/2 − 1/4).
fn upsample(g2: &Grid, g: &Grid, v: &[f64]) -> Vec<f64> {
    par(g.n(), |k| {
        let (f, i, j) = g.fij(k);
        let (x, y) = (0.5 * i as f64 - 0.25, 0.5 * j as f64 - 0.25);
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let (i0, j0) = (x0 as i64, y0 as i64);
        let t = |di: i64, dj: i64| v[g2.at(f, i0 + di, j0 + dj).unwrap_or_else(|| g2.idx(f, i / 2, j / 2))];
        (1.0 - ty) * ((1.0 - tx) * t(0, 0) + tx * t(1, 0)) + ty * ((1.0 - tx) * t(0, 1) + tx * t(1, 1))
    })
}

/// [`upsample`] of vectors.
fn upsample3(g2: &Grid, g: &Grid, v: &[DVec3]) -> Vec<DVec3> {
    let c = |a: usize| upsample(g2, g, &v.iter().map(|x| x[a]).collect::<Vec<_>>());
    let (x, y, z) = (c(0), c(1), c(2));
    par(g.n(), |k| DVec3::new(x[k], y[k], z[k]))
}

/// What the wind depends on at a texel.
struct WindIn {
    geo: Geo,
    coast_km: f64,
    /// gradient of the land fraction smoothed to ~300 km, per 800 km (points inland)
    grad_land: DVec3,
    /// tangent gradient of the pressure-perturbation stream function (per m)
    grad_psi: DVec3,
}

/// The latitude whose circulation belt (ITCZ, trades, subtropical high, westerlies) lies at
/// `lat` in a season (`season` +1 July, −1 January): the belts follow the sun, more over land.
fn belt_latitude(ac: &crate::config::AtlasConfig, lat: f64, coast_km: f64, season: f64) -> f64 {
    let landness = smoothstep(-300.0, 1500.0, coast_km);
    (lat - (4.0 + 7.0 * landness * ac.monsoon.min(1.5)) * season).clamp(-90.0, 90.0)
}

/// Prevailing surface wind (m/s, tangent vector) in a season: `season` +1 July (belts shifted
/// north), −1 January.
fn wind(ac: &crate::config::AtlasConfig, w: &WindIn, season: f64) -> DVec3 {
    let lat = w.geo.lat;
    let le = belt_latitude(ac, lat, w.coast_km, season);
    let al = le.abs();
    let amp = 6.0 + 2.0 * smoothstep(25.0, 35.0, al) - 4.0 * smoothstep(55.0, 65.0, al);
    let s6 = (6.0 * al).to_radians().sin();
    // easterly trades, westerlies, polar easterlies; equatorward / poleward / equatorward
    let mut v = w.geo.east * (-amp * s6) + w.geo.north * (-0.3 * amp * s6 * le.signum());
    v += w.geo.dir.cross(w.grad_psi) * PSI_SCALE;
    // monsoon: onshore towards heated continents in summer, offshore in winter
    let summer = season * lat >= 0.0;
    let sf = if summer { 1.0 } else { -0.5 };
    v += w.grad_land * (4.5 * ac.monsoon * sf * gauss(lat.abs(), 20.0, 15.0));
    v
}

/// The seasonal moisture advection on the climate grid: precipitation (mm/day) and the
/// maritime influence (0..1) of each texel.
struct Season {
    precip: Vec<f64>,
    maritime: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn advect(
    ac: &crate::config::AtlasConfig,
    world: &World,
    g2: &Grid,
    dirs2: &[DVec3],
    wind_in: &[WindIn],
    moist2: &[f64],
    oro2: &[f64],
    land2: &[f64],
    temp2: &[f64],
    cold2: &[f64],
    season: f64,
) -> Season {
    let n2 = g2.n();
    let a_m = world.ell.a;
    let h2 = g2.spacing() * a_m;
    let dt = 3.0 * h2 / 10.0;
    let dt_d = dt / 86400.0;
    let lapse = world.cfg.climate.lapse_rate_c_per_km;
    /// per texel: the moisture kept by rain over a step (rate and orographic rain-out), the
    /// evaporation coefficient over sea and its target, the recycling over land, the
    /// saturation at the ground, the maritime influence kept over a step, the land fraction
    #[derive(Clone, Copy)]
    struct Tex {
        keep: f32,
        e_sea: f32,
        q_sea: f32,
        eta: f32,
        q_air: f32,
        keep_m: f32,
        lf: f32,
    }
    type Taps = ([u32; 4], [f32; 4]);
    let (taps, tex): (Vec<Taps>, Vec<Tex>) = par(n2, |k| {
        let wi = &wind_in[k];
        let lat = wi.geo.lat;
        let w = wind(ac, wi, season);
        let back = (dirs2[k] - w * (dt / a_m)).normalize();
        let taps = g2.bilinear(back);
        let hb: f64 = taps.iter().map(|&(t, wt)| wt * oro2[t].max(0.0)).sum();
        let h = oro2[k].max(0.0);
        let lf = land2[k];
        let summer = season * lat >= 0.0;
        // the half-year mean of a sinusoidal year: ± range/π
        let sgn = if summer { 1.0 / std::f64::consts::PI } else { -1.0 / std::f64::consts::PI };
        let range = range_continental(lat) * (0.22 + 0.38 * lf);
        let t_season = temp2[k] + sgn * range;
        let le = belt_latitude(ac, lat, wi.coast_km, season);
        // convective summer rains over heated land: monsoon in the tropics and subtropics (drier
        // winters), thunderstorms in mid-latitude continental interiors (under cold highs in
        // winter: the storm tracks bring their rain to the west coasts)
        let lw = gauss(lat.abs(), 18.0, 14.0) * lf * ac.monsoon;
        let interior = gauss(lat.abs(), 42.0, 16.0) * lf * smoothstep(200.0, 1200.0, wi.coast_km);
        let mons = if summer { 1.0 + 1.2 * lw + 0.8 * interior } else { 1.0 / (1.0 + lw + 1.5 * interior) };
        let stab = 1.0 - 0.85 * cold2[k];
        let high = if summer { 1.0 - 0.7 * lf } else { 1.0 };
        let rate = rain_rate(le, high) * stab * mons * (0.45 * moist2[k]).exp() * (-K_DESCENT * (hb - h).max(0.0)).exp();
        let oro_keep = (-K_ORO * ac.rain_shadow * (h - hb).max(0.0)).exp();
        (
            (taps.map(|t| t.0 as u32), taps.map(|t| t.1 as f32)),
            Tex {
                keep: ((-rate * dt_d).exp() * oro_keep) as f32,
                e_sea: ((1.0 - lf) / TAU_EVAP * dt_d) as f32,
                q_sea: (RH_OCEAN * qsat(t_season)) as f32,
                eta: (lf * 0.45 * ((t_season + 5.0) / 25.0).clamp(0.0, 1.0)) as f32,
                q_air: qsat(t_season - lapse * h / KM) as f32,
                keep_m: (lf * (-(w.length() * dt) / L_MARITIME).exp()) as f32,
                lf: lf as f32,
            },
        )
    })
    .into_iter()
    .unzip();
    // state: moisture (mm), precipitation over the last step (mm), maritime influence
    let mut q: Vec<f32> = par(n2, |k| tex[k].q_sea * (1.0 - 0.4 * tex[k].lf));
    let mut p: Vec<f32> = par(n2, |k| q[k] * (1.0 - tex[k].keep));
    let mut m: Vec<f32> = par(n2, |k| 1.0 - tex[k].lf);
    let (mut q2, mut p2, mut m2) = (q.clone(), p.clone(), m.clone());
    for _ in 0..ac.advection_steps {
        q2.par_iter_mut().zip(p2.par_iter_mut()).zip(m2.par_iter_mut()).enumerate().for_each(|(k, ((qn, pn), mn))| {
            let t = tex[k];
            let (ti, tw) = taps[k];
            let (mut qb, mut mb) = (0.0f32, 0.0f32);
            for a in 0..4 {
                qb += tw[a] * q[ti[a] as usize];
                mb += tw[a] * m[ti[a] as usize];
            }
            // evaporation over sea (towards the humidity of the sea air), recycling over land
            let q1 = qb + t.e_sea * (t.q_sea - qb).max(0.0) + t.eta * p[k];
            let q2 = q1 * t.keep;
            let q3 = q2.min(t.q_air);
            *qn = q3;
            *pn = q1 - q3;
            *mn = (1.0 - t.lf) + t.keep_m * mb;
        });
        std::mem::swap(&mut q, &mut q2);
        std::mem::swap(&mut p, &mut p2);
        std::mem::swap(&mut m, &mut m2);
    }
    Season { precip: p.iter().map(|&x| x as f64 / dt_d).collect(), maritime: m.iter().map(|&x| x as f64).collect() }
}

pub(super) fn build(world: &World) -> Atlas {
    let cfg = &world.cfg;
    let ac = &cfg.atlas;
    let seed = world.seed();
    let r = ac.resolution as usize;
    let g = Grid::new(r);
    let n = g.n();
    let a_m = world.ell.a;
    let gsd = g.spacing() * a_m;
    let lapse = cfg.climate.lapse_rate_c_per_km;
    let fbm = |k: u64, lam_km: f64, oct: usize| Fbm::new(mix64(seed ^ 0xA71A_5000 ^ k.wrapping_mul(0x9E37_79B9)), lam_km * KM, oct, 2.0, 0.5);

    let mut lap = Lap::new();
    let dirs: Vec<DVec3> = par(n, |k| g.dir(k));
    let nb4: Vec<[usize; 4]> = par(n, |k| g.neighbours4(k));
    let geos: Vec<Geo> = par(n, |k| geo(world, dirs[k]));
    lap.lap("geometry");
    let e = ExtGrid::new(g);
    lap.lap("extended faces");

    // the half-resolution grid (climate), and the long-wavelength noises evaluated there
    let g2 = Grid::new(r / 2);
    let n2 = g2.n();
    let dirs2: Vec<DVec3> = par(n2, |k| g2.dir(k));
    let nb2: Vec<[usize; 4]> = par(n2, |k| g2.neighbours4(k));
    let geos2: Vec<Geo> = par(n2, |k| geo(world, dirs2[k]));
    let on2 = |f: &(dyn Fn(DVec3) -> f64 + Sync)| -> Vec<f64> { par(n2, |k| f(geos2[k].p)) };
    let on2_up = |f: &(dyn Fn(DVec3) -> f64 + Sync)| -> Vec<f64> { upsample(&g2, &g, &on2(f)) };
    let nval = |f: &Fbm, p: DVec3| f.eval(p, 2.0 * gsd) * f.norm() * 1.6;
    let temp_noise = on2_up(&|p| 5.0 * world.temp_n.eval(p, 50.0 * KM));
    let moist2 = on2(&|p| world.moist_n.eval(p, 20.0 * KM) * world.moist_n.norm() * 1.6);
    let pw = [fbm(1, 2000.0, 4), fbm(2, 2000.0, 4), fbm(3, 2000.0, 4)];
    let plate_warp = |p: DVec3| DVec3::new(pw[0].eval(p, 2.0 * gsd), pw[1].eval(p, 2.0 * gsd), pw[2].eval(p, 2.0 * gsd)) * 0.12;
    let plate_warp_up = upsample3(&g2, &g, &par(n2, |k| plate_warp(geos2[k].p)));
    let litho_noise: Vec<Vec<f64>> = (0..7u64)
        .map(|i| {
            let f = fbm(10 + i, if i < 5 { 900.0 } else { 1300.0 }, 3);
            on2_up(&|p| nval(&f, p))
        })
        .collect();
    let cell = ac.culture_cell_km * KM;
    let cw = [fbm(20, 0.8 * ac.culture_cell_km, 3), fbm(21, 0.8 * ac.culture_cell_km, 3), fbm(22, 0.8 * ac.culture_cell_km, 3)];
    let culture_warp =
        upsample3(&g2, &g, &par(n2, |k| DVec3::new(nval(&cw[0], geos2[k].p), nval(&cw[1], geos2[k].p), nval(&cw[2], geos2[k].p)) * (0.2 * cell)));
    let n_dev = fbm(30, 400.0, 3);
    let dev_noise = on2_up(&|p| nval(&n_dev, p));
    lap.lap("geometry, noises");

    // ---- smooth elevation and mountain amplitude, band-limited at the texel size: first
    // without the tectonic ranges (the plates' crust comes from it), then with them (coasts,
    // climate and everything else follow the final relief)
    let relief0: Vec<(f64, f64)> = par(n, |k| world.smooth_relief_u(geos[k].p, gsd, 0.0));
    let elev: Vec<f64> = relief0.iter().map(|r| r.0).collect();
    lap.lap("elevation");

    // ---- plates: a weighted, warped spherical Voronoi diagram
    let np = ac.plates as usize;
    let mut sites: Vec<DVec3> = Vec::with_capacity(np);
    for i in 0..np {
        // best of 8 candidates (the farthest from the sites so far): plates of similar size
        let mut best = (f64::MIN, DVec3::Z);
        for cand in 0..8i64 {
            let d = hash_dir(hash2(seed ^ 0x9_1A7E, i as i64, cand));
            let sep = sites.iter().map(|s| angle(*s, d)).fold(f64::MAX, f64::min);
            if sep > best.0 {
                best = (sep, d);
            }
        }
        sites.push(best.1);
    }
    let plate_hash = |i: usize| hash1(seed ^ 0x9_1A7F, i as i64);
    // additive weights w (radians): the plate of q maximizes cos(angle(q, site) − w)
    let weights: Vec<(f64, f64)> = (0..np).map(|i| (0.12 * u01k(plate_hash(i), 3)).sin_cos()).collect();
    let axes: Vec<DVec3> = (0..np).map(|i| hash_dir(mix64(plate_hash(i) ^ 0x4A))).collect();
    let omega: Vec<f64> = (0..np).map(|i| (0.3 + 0.8 * u01k(plate_hash(i), 5)).to_radians()).collect();
    let plate_at = |q: DVec3| -> usize {
        let mut best = (f64::MIN, 0);
        for (i, s) in sites.iter().enumerate() {
            let c = q.dot(*s).clamp(-1.0, 1.0);
            let (sw, cw) = weights[i];
            let sc = c * cw + (1.0 - c * c).sqrt() * sw;
            if sc > best.0 {
                best = (sc, i);
            }
        }
        best.1
    };
    let plate_of = |d: DVec3| plate_at((d + plate_warp(d * a_m)).normalize());
    // velocity (mm/yr) of plate `i` at `d`: ω (rad/Myr) × r (m) = m/Myr = 1e-3 mm/yr
    let vel = |i: usize, d: DVec3| axes[i].cross(d) * (omega[i] * a_m * 1e-3);
    lap.lap("upsampling, seasonality, wind");
    let plate: Vec<usize> = par(n, |k| plate_at((dirs[k] + plate_warp_up[k]).normalize()));
    lap.lap("plates");
    let cont: Vec<bool> = elev.iter().map(|&h| h > -600.0).collect();
    struct BSeed {
        plates: [usize; 2],
        kinds: [Boundary; 2],
        conv: f64,
    }
    let bseed: Vec<Option<BSeed>> = par(n, |k| {
        let m = *nb4[k].iter().find(|&&m| plate[m] != plate[k])?;
        let d = dirs[k];
        let (a, b) = (plate[k], plate[m]);
        // the normal of the (unwarped) bisector of the two sites: smooth along the boundary
        let ab = sites[b] - sites[a];
        let nrm = (ab - d * d.dot(ab)).normalize_or_zero();
        let dv = vel(a, d) - vel(b, d);
        let conv = dv.dot(nrm);
        let shear = dv.dot(d.cross(nrm));
        let (ca, cb) = (cont[k], cont[m]);
        let kinds = if conv.abs() < 0.6 * shear.abs() && conv.abs() < 15.0 {
            [Boundary::Transform; 2]
        } else if conv > 0.0 {
            match (ca, cb) {
                (true, true) => [Boundary::Collision; 2],
                (true, false) => [Boundary::Overriding, Boundary::Subducting],
                (false, true) => [Boundary::Subducting, Boundary::Overriding],
                (false, false) => [Boundary::IslandArc; 2],
            }
        } else if ca || cb {
            [Boundary::Rift; 2]
        } else {
            [Boundary::Ridge; 2]
        };
        Some(BSeed { plates: [a, b], kinds, conv })
    });
    let is_seed: Vec<bool> = bseed.iter().map(|s| s.is_some()).collect();
    let bnear = jfa(&e, &dirs, &is_seed);
    lap.lap("plate boundaries");
    // per texel: distance, kind, closing speed, tectonic uplift and volcanism of the boundary
    let tect: Vec<(f64, Boundary, f64, f64, f64)> = par(n, |k| {
        if bnear[k] == NONE {
            return (20000.0, Boundary::None, 0.0, 0.0, 0.0);
        }
        let s = bseed[bnear[k] as usize].as_ref().unwrap();
        let dist = angle(dirs[k], dirs[bnear[k] as usize]) * a_m / KM + 0.5 * gsd / KM;
        let kind = if plate[k] == s.plates[1] { s.kinds[1] } else { s.kinds[0] };
        let sg = (s.conv.abs() / 40.0).clamp(0.25, 1.6);
        let d = dist;
        let (up, vol) = match kind {
            Boundary::Overriding => (sg * (0.55 * gauss(d, 60.0, 70.0)).max(0.9 * gauss(d, 220.0, 120.0)), sg.min(1.0) * gauss(d, 200.0, 70.0)),
            Boundary::Subducting => (-0.6 * gauss(d, 30.0, 60.0), 0.0),
            Boundary::Collision => (sg * gauss(d, 0.0, 320.0), 0.08 * gauss(d, 0.0, 200.0)),
            Boundary::IslandArc => (0.45 * sg * gauss(d, 90.0, 60.0), 0.9 * gauss(d, 110.0, 50.0)),
            Boundary::Rift => (sg * (-0.55 * gauss(d, 0.0, 45.0) + 0.4 * gauss(d, 90.0, 45.0)), 0.5 * gauss(d, 0.0, 90.0)),
            Boundary::Ridge => (0.25 * gauss(d, 0.0, 180.0), 0.35 * gauss(d, 0.0, 35.0)),
            Boundary::Transform => (0.12 * gauss(d, 0.0, 50.0), 0.0),
            Boundary::None => (0.0, 0.0),
        };
        (dist, kind, s.conv, up, vol)
    });
    // hotspot tracks: the crust that was over a hotspot t Myr ago is now rotated by the plate's
    // motion; volcanism fading with age (splatted in a fixed order, combined with max)
    let mut track = vec![0.0f64; n];
    let win = (250.0 * KM / gsd).ceil() as i64;
    for h in 0..ac.hotspots as u64 {
        let pos = hash_dir(hash1(seed ^ 0x4075, h as i64));
        let pl = plate_of(pos);
        for t in 0..=60 {
            let x = glam::DQuat::from_axis_angle(axes[pl], omega[pl] * t as f64) * pos;
            if plate_of(x) != pl {
                break;
            }
            let (f, i, j) = g.fij(g.nearest(x));
            for dj in -win..=win {
                for di in -win..=win {
                    let Some(k) = e.at(f, i as i64 + di, j as i64 + dj) else { continue };
                    let d = angle(dirs[k], x) * a_m / KM;
                    let v = gauss(d, 0.0, 70.0) * (-(t as f64) / 25.0).exp();
                    track[k] = track[k].max(v);
                }
            }
        }
    }

    lap.lap("boundary fields, hotspots");
    // the tectonic fields, smoothed over ~30 km (no steps where the boundary kind changes)
    let uplift = smooth(&nb4, &par(n, |k| (tect[k].3 + 0.25 * track[k]).clamp(-1.0, 1.0)), 9);
    let volcanism = smooth(&nb4, &par(n, |k| tect[k].4.max(track[k]).clamp(0.0, 1.0)), 9);

    let relief: Vec<(f64, f64)> = par(n, |k| world.smooth_relief_u(geos[k].p, gsd, uplift[k]));
    let elev: Vec<f64> = relief.iter().map(|r| r.0).collect();
    let amp_m: Vec<f64> = relief.iter().map(|r| r.1).collect();
    lap.lap("elevation with the tectonic ranges");

    // ---- signed coast distance: jump flooding from the texels next to the other class
    let land: Vec<bool> = elev.iter().map(|&h| h > 0.0).collect();
    let coast_seeds: Vec<bool> = par(n, |k| nb4[k].iter().any(|&m| land[m] != land[k]));
    let near = jfa(&e, &dirs, &coast_seeds);
    let coast: Vec<f64> = par(n, |k| {
        let d = if near[k] == NONE { 20000.0 } else { angle(dirs[k], dirs[near[k] as usize]) * a_m / KM + 0.5 * gsd / KM };
        if land[k] {
            d
        } else {
            -d
        }
    });
    lap.lap("coast distance");

    // ---- ocean currents (on the half-resolution grid, from the land fractions along rays to
    // the west and east: smooth) and the sea-level temperature before continentality
    let elev2 = downsample(&g2, &g, &elev);
    let land2 = downsample(&g2, &g, &land.iter().map(|&l| l as u8 as f64).collect::<Vec<_>>());
    let coast2 = downsample(&g2, &g, &coast);
    let currents = if ac.currents { 1.0 } else { 0.0 };
    let land_along = |k: usize, sign: f64| -> f64 {
        let ge = &geos2[k];
        let (mut s, mut ws) = (0.0, 0.0);
        for (d_km, w) in [(150.0, 1.0), (300.0, 1.0), (500.0, 0.8), (800.0, 0.6), (1200.0, 0.4)] {
            let a = d_km * KM / a_m;
            let q = ge.dir * a.cos() + ge.east * (sign * a.sin());
            s += w * g2.bilinear(q).iter().map(|&(t, wt)| wt * land2[t]).sum::<f64>();
            ws += w;
        }
        s / ws
    };
    let cur2: Vec<[f64; 3]> = par(n2, |k| {
        let (lw, le) = (land_along(k, -1.0), land_along(k, 1.0));
        let al = geos2[k].lat.abs();
        let ad = coast2[k].abs();
        let lf = land2[k];
        // cold eastern-boundary currents and upwelling off subtropical west coasts
        let cold = (1.0 - lw)
            * le.max(lf)
            * (-ad / if coast2[k] > 0.0 { 350.0 } else { 600.0 }).exp()
            * smoothstep(8.0, 16.0, al)
            * (1.0 - smoothstep(32.0, 42.0, al));
        // warm western-boundary currents off subtropical east coasts
        let warm_e = lw.max(lf) * (1.0 - le) * (-ad / 450.0).exp() * smoothstep(12.0, 20.0, al) * (1.0 - smoothstep(38.0, 48.0, al));
        // warm drift reaching high-latitude west coasts (carried inland by the westerlies)
        let warm_w = (1.0 - lw) * le.max(lf) * (-ad / 900.0).exp() * smoothstep(42.0, 52.0, al) * (1.0 - smoothstep(68.0, 78.0, al));
        [cold, warm_e, warm_w].map(|x| x * currents)
    });
    let cold2: Vec<f64> = cur2.iter().map(|c| c[0]).collect();
    let cur_t = upsample(&g2, &g, &cur2.iter().map(|c| -7.0 * c[0] + 2.5 * c[1] + 6.0 * c[2]).collect::<Vec<_>>());
    let c = &cfg.climate;
    let temp0: Vec<f64> = par(n, |k| {
        let la = geos[k].lat.abs() / 90.0;
        c.equator_temp_c - c.pole_drop_c * la.powf(1.6) + temp_noise[k] + cur_t[k]
    });

    // ---- winds: the stream function of the pressure perturbations and the land heating
    let psi = Fbm::new(mix64(seed ^ 0x9517), 3000.0 * KM, 3, 2.0, 0.5);
    let wind_inputs = |dirs: &[DVec3], geos: &[Geo], coast: &[f64], nb: &[[usize; 4]]| -> Vec<WindIn> {
        // the land fraction smoothed to ~300 km (smooth gradients, unlike the coast distance's
        // at its medial axes): the monsoon flows up its gradient
        let land: Vec<f64> = coast.iter().map(|&c| (c > 0.0) as u8 as f64).collect();
        let sigma = 300.0 * KM / (a_m * texel_spacing(dirs.len()));
        let gl = gradient(dirs, nb, &smooth(nb, &land, (4.0 * sigma * sigma) as usize));
        par(dirs.len(), |k| {
            let (_, gp) = psi.eval_d(geos[k].p, gsd, 99);
            let d = dirs[k];
            WindIn { geo: geos[k], coast_km: coast[k], grad_land: gl[k] * (800.0 * KM / a_m), grad_psi: gp - d * d.dot(gp) }
        })
    };
    lap.lap("temperature");

    // ---- moisture advection on the half-resolution grid, two seasons; the orography is the
    // elevation smoothed to ~100 km (only ranges cast rain shadows, not every hill)
    let temp2 = downsample(&g2, &g, &temp0);
    let oro2 = smooth(&nb2, &elev2.iter().map(|&h| h.max(0.0)).collect::<Vec<_>>(), 25);
    let wind2 = wind_inputs(&dirs2, &geos2, &coast2, &nb2);
    lap.lap("climate grid");
    let jul = advect(ac, world, &g2, &dirs2, &wind2, &moist2, &oro2, &land2, &temp2, &cold2, 1.0);
    let jan = advect(ac, world, &g2, &dirs2, &wind2, &moist2, &oro2, &land2, &temp2, &cold2, -1.0);
    let p_scale = 365.0 * ac.precipitation * (1.5 * c.moisture_bias).exp();
    // the condensate falls over some distance: spread it (~40 km)
    let jul = Season { precip: smooth(&nb2, &jul.precip, 4), ..jul };
    let jan = Season { precip: smooth(&nb2, &jan.precip, 4), ..jan };
    let precip2: Vec<f64> = par(n2, |k| 0.5 * (jul.precip[k] + jan.precip[k]) * p_scale);
    // (the regime is a regional shape of the year: smoothed to ~80 km, else every windward
    // slope of one season's wind shows)
    let regime2: Vec<f64> = par(n2, |k| {
        let (s, w) = if geos2[k].lat >= 0.0 { (jul.precip[k], jan.precip[k]) } else { (jan.precip[k], jul.precip[k]) };
        // (no seasons at the equator: fade out the flip of the hemispheres' summer)
        (s - w) / (s + w + 0.05) * smoothstep(1.0, 8.0, geos2[k].lat.abs())
    });
    let regime2 = smooth(&nb2, &regime2, 16);
    let maritime2: Vec<f64> = par(n2, |k| 0.5 * (jul.maritime[k] + jan.maritime[k]));
    lap.lap("moisture advection");
    let precip = upsample(&g2, &g, &precip2);
    let regime = upsample(&g2, &g, &regime2);
    let maritime = upsample(&g2, &g, &maritime2);

    // ---- temperature, seasonality, annual wind
    let (temp, trange): (Vec<f64>, Vec<f64>) = par(n, |k| {
        let lat = geos[k].lat;
        let m = if coast[k] <= 0.0 { 1.0 } else { maritime[k].max((-coast[k] / 120.0).exp()).clamp(0.0, 1.0) };
        let range = range_continental(lat) * (0.22 + 0.78 * (1.0 - m).powf(0.8));
        (temp0[k] - 5.0 * (1.0 - m) * smoothstep(35.0, 70.0, lat.abs()), range)
    })
    .into_iter()
    .unzip();
    let wind_ann = upsample3(&g2, &g, &par(n2, |k| 0.5 * (wind(ac, &wind2[k], 1.0) + wind(ac, &wind2[k], -1.0))));

    // ---- glaciation (ice at the last glacial maximum: ~6 °C colder, more towards the poles)
    let glaciation: Vec<f64> = par(n, |k| {
        let lat = geos[k].lat.to_radians();
        let lgm = temp[k] - 6.0 - 6.0 * lat.sin().powi(2);
        let h = elev[k].max(0.0);
        let tl = lgm - lapse * h / KM;
        // (where it never melts, little snow suffices)
        let sheet = smoothstep(1.0, -6.0, tl) * smoothstep(150.0, 450.0, precip[k]).max(smoothstep(-8.0, -16.0, tl));
        let peaks = h + 0.66 * amp_m[k];
        let alpine = smoothstep(-2.0, -9.0, lgm - lapse * peaks / KM) * smoothstep(200.0, 600.0, precip[k]) * smoothstep(200.0, 900.0, amp_m[k]);
        (sheet.max(0.9 * alpine) * smoothstep(-200.0, 50.0, elev[k])).clamp(0.0, 1.0)
    });

    // ---- lithology
    let g_elev = gradient(&dirs, &nb4, &elev);
    let flat: Vec<f64> = par(n, |k| 1.0 / (1.0 + (g_elev[k].length() / a_m / 0.006).powi(2)));
    let litho: Vec<(usize, usize, f64)> = par(n, |k| {
        let e = elev[k];
        let (bd, kind, _, _, _) = tect[k];
        let vol = volcanism[k];
        let ct = cont[k] as u8 as f64;
        let orogen = matches!(kind, Boundary::Collision | Boundary::Overriding) as u8 as f64;
        let w = [
            0.75 + 0.4 * orogen * gauss(bd, 380.0, 220.0),
            0.7 * (1.0 - smoothstep(800.0, 1500.0, e)) * ct * smoothstep(-0.3, 0.3, litho_noise[6][k]),
            0.45 * smoothstep(1000.0, 2500.0, bd) * ct + 0.8 * orogen * gauss(bd, 0.0, 150.0) + 0.5 * glaciation[k] + 0.3 * smoothstep(1500.0, 3000.0, e),
            2.0 * vol + 0.9 * smoothstep(0.45, 0.8, litho_noise[5][k]) * ct,
            0.6 * (1.0 - smoothstep(100.0, 500.0, e)) * flat[k]
                + 0.6 * (1.0 - smoothstep(150.0, 400.0, precip[k])) * (1.0 - smoothstep(300.0, 1200.0, e))
                + 0.4 * glaciation[k] * (1.0 - smoothstep(300.0, 800.0, e)),
        ];
        let w2: [f64; 5] = std::array::from_fn(|c| w[c] * (0.7 * litho_noise[c][k]).exp());
        // largest and second largest (ties to the lower class)
        let dom = (1..5).fold(0, |b, c| if w2[c] > w2[b] { c } else { b });
        let sec = (0..5).filter(|&c| c != dom).fold(usize::MAX, |b, c| if b == usize::MAX || w2[c] > w2[b] { c } else { b });
        (dom, sec, w2[sec] / (w2[dom] + w2[sec]).max(1e-12))
    });

    lap.lap("glaciation, lithology");
    // ---- cultures: a warped Voronoi diagram of ~cell-sized areas, dense ids in hash order
    let csite: Vec<(u64, DVec3)> = par(n, |k| {
        let p = geos[k].p;
        let warp = culture_warp[k];
        let c = worley3(seed ^ 0x00C0_170E, p + warp, cell, 0.85);
        (c.id, c.point)
    });
    let mut hashes: Vec<(u64, DVec3)> = csite.clone();
    hashes.dedup_by_key(|h| h.0);
    hashes.sort_unstable_by_key(|h| h.0);
    hashes.dedup_by_key(|h| h.0);
    let cid: Vec<usize> = par(n, |k| hashes.binary_search_by_key(&csite[k].0, |h| h.0).unwrap());
    // the climate of each culture: the mean over its land texels (its site texel without land)
    let nc = hashes.len();
    let mut acc = vec![[0.0f64; 6]; nc];
    for k in 0..n {
        if land[k] {
            let a = &mut acc[cid[k]];
            for (x, v) in a.iter_mut().zip([1.0, temp[k], trange[k], precip[k], regime[k], elev[k]]) {
                *x += v;
            }
        }
    }
    let comfort = |t: f64, p: f64| {
        smoothstep(-6.0, 8.0, t) * (1.0 - smoothstep(26.0, 32.0, t)) * smoothstep(150.0, 700.0, p) * (1.0 - 0.35 * smoothstep(2500.0, 5000.0, p))
    };
    let cultures: Vec<Culture> = hashes
        .iter()
        .enumerate()
        .map(|(i, &(h, centre))| {
            let site = centre.normalize_or(DVec3::Z);
            let a = acc[i];
            let [t, tr, p, rg, e] = if a[0] > 0.0 {
                [a[1] / a[0], a[2] / a[0], a[3] / a[0], a[4] / a[0], a[5] / a[0]]
            } else {
                let k = g.nearest(site);
                [temp[k], trange[k], precip[k], regime[k], elev[k]]
            };
            let s = AtlasSample { temp_c: t, temp_range_c: tr, precip_mm: p, regime: rg, ..Default::default() };
            let kp = koppen_with_lapse(&s, e, lapse);
            let archetype = draw_archetype(kp, e, u01k(h, 7));
            let bias = match archetype {
                Archetype::SurveyGrid => 0.15,
                Archetype::TemperateVillage | Archetype::Mediterranean => 0.08,
                Archetype::Arctic | Archetype::SavannaPastoral | Archetype::DesertOasis => -0.12,
                _ => 0.0,
            };
            let development = (0.2 + 0.6 * u01k(h, 31) + bias + 0.2 * (comfort(t - lapse * e.max(0.0) / KM, p) - 0.5)).clamp(0.0, 1.0);
            Culture { hash: h, site, archetype, development }
        })
        .collect();

    lap.lap("cultures");
    // ---- development and population potential
    let (development, population): (Vec<f64>, Vec<f64>) = par(n, |k| {
        let e = elev[k];
        let te = temp[k] - lapse * e.max(0.0) / KM;
        let cf = comfort(te, precip[k]);
        let dev = (cultures[cid[k]].development + 0.15 * dev_noise[k] + 0.1 * (cf - 0.5)).clamp(0.0, 1.0);
        let (dom, sec, fr) = litho[k];
        const FERT: [f64; 5] = [0.8, 0.65, 0.5, 0.95, 1.0];
        let fert = ((1.0 - fr) * FERT[dom] + fr * FERT[sec]) * (1.0 - 0.5 * glaciation[k]);
        let water = (-coast[k].max(0.0) / 150.0).exp().max(0.85 * smoothstep(300.0, 1000.0, precip[k]));
        let pop = if land[k] { flat[k].sqrt() * water.sqrt() * cf * fert * (1.0 - smoothstep(2500.0, 4500.0, e)) } else { 0.0 };
        (dev, pop.clamp(0.0, 1.0))
    })
    .into_iter()
    .unzip();

    lap.lap("development, population");
    // ---- pack
    let texel_words: Vec<[u32; WORDS]> = par(n, |k| {
        let wv = wind_ann[k];
        let (bd, kind, conv, _, _) = tect[k];
        let mut v = [0.0f64; slot::CONTINUOUS];
        v[slot::ELEVATION] = elev[k];
        v[slot::COAST] = coast[k];
        v[slot::WIND_E] = wv.dot(geos[k].east);
        v[slot::WIND_N] = wv.dot(geos[k].north);
        v[slot::PRECIP] = precip[k].max(0.0);
        v[slot::TEMP] = temp[k];
        v[slot::TEMP_RANGE] = trange[k];
        v[slot::REGIME] = regime[k].clamp(-1.0, 1.0);
        v[slot::PLATE_DIST] = bd;
        v[slot::CONVERGENCE] = conv;
        v[slot::UPLIFT] = uplift[k];
        v[slot::VOLCANISM] = volcanism[k];
        v[slot::GLACIATION] = glaciation[k];
        v[slot::DEVELOPMENT] = development[k];
        v[slot::POPULATION] = population[k];
        let mut s = [0u16; slot::COUNT];
        for (i, x) in v.iter().enumerate() {
            s[i] = f16_bits(*x as f32);
        }
        s[slot::PLATE] = (plate[k] as u16 & 63) | (kind as u16) << 6 | (cont[k] as u16) << 9 | ((track[k] > 0.2) as u16) << 10;
        let (dom, sec, fr) = litho[k];
        s[slot::LITHO] = dom as u16 | (sec as u16) << 3 | ((fr * 510.0).round().clamp(0.0, 255.0) as u16) << 8;
        let ci = cid[k] as u16 & 0xFFF;
        s[slot::CULTURE] = ci | (cultures[cid[k]].archetype as u16) << 12;
        let mut w = [0u32; WORDS];
        for (i, x) in w.iter_mut().enumerate() {
            *x = s[2 * i] as u32 | (s[2 * i + 1] as u32) << 16;
        }
        w
    });
    let pd = r + 2 * APRON;
    let data_words = 6 * pd * pd * WORDS;
    let mut words = vec![0u32; HEADER + data_words + 4 * nc];
    words[..10].copy_from_slice(&[
        MAGIC,
        ATLAS_VERSION,
        r as u32,
        APRON as u32,
        WORDS as u32,
        HEADER as u32,
        nc as u32,
        (HEADER + data_words) as u32,
        np as u32,
        ac.hotspots,
    ]);
    let padded = |f: usize, i: usize, j: usize| HEADER + ((f * pd + j) * pd + i) * WORDS;
    for (k, w) in texel_words.iter().enumerate() {
        let (f, i, j) = g.fij(k);
        let o = padded(f, i + APRON, j + APRON);
        words[o..o + WORDS].copy_from_slice(w);
    }
    // aprons: the fields at the (extended) texel centres, bilinear from the neighbouring faces
    let apron: Vec<(usize, [u32; WORDS])> = (0..6 * pd * pd)
        .into_par_iter()
        .filter_map(|t| {
            let (f, j, i) = (t / (pd * pd), (t / pd) % pd, t % pd);
            let inside = |x: usize| (APRON..APRON + r).contains(&x);
            if inside(i) && inside(j) {
                return None;
            }
            let d = g.dir_ext(f, i as i64 - APRON as i64, j as i64 - APRON as i64)?;
            let taps = g.bilinear(d);
            let mut s = [0u16; slot::COUNT];
            for (si, sv) in s.iter_mut().enumerate().take(slot::CONTINUOUS) {
                let v: f64 = taps.iter().map(|&(k, w)| w * f16_value(half(&texel_words[k], si)) as f64).sum();
                *sv = f16_bits(v as f32);
            }
            let (kmax, _) = taps.iter().fold(taps[0], |b, &x| if x.1 > b.1 { x } else { b });
            for (si, sv) in s.iter_mut().enumerate().skip(slot::CONTINUOUS) {
                *sv = half(&texel_words[kmax], si);
            }
            let mut w = [0u32; WORDS];
            for (i, x) in w.iter_mut().enumerate() {
                *x = s[2 * i] as u32 | (s[2 * i + 1] as u32) << 16;
            }
            Some((HEADER + t * WORDS, w))
        })
        .collect();
    for (o, w) in apron {
        words[o..o + WORDS].copy_from_slice(&w);
    }
    lap.lap("packing, aprons");
    for (i, c) in cultures.iter().enumerate() {
        let o = HEADER + data_words + 4 * i;
        words[o..o + 4].copy_from_slice(&[c.hash as u32, (c.hash >> 32) as u32, c.archetype as u32, (c.development as f32).to_bits()]);
    }
    Atlas::from_parts(words, cultures, Atlas::key(cfg))
}

/// Slot `s` of a texel's words.
#[inline]
fn half(w: &[u32; WORDS], s: usize) -> u16 {
    (w[s / 2] >> (16 * (s % 2))) as u16
}

/// Archetype of a culture from its climate class and elevation (weighted draw by `u`).
fn draw_archetype(k: Koppen, elev: f64, u: f64) -> Archetype {
    use Archetype::*;
    use Koppen::*;
    let mut w: Vec<(Archetype, f64)> = match k {
        Af | Am => vec![(TropicalSmallholder, 3.0), (TropicalPlantation, 2.0), (MonsoonPaddy, 2.0)],
        Aw => vec![(SavannaPastoral, 3.0), (TropicalSmallholder, 2.0), (TropicalPlantation, 1.0), (MonsoonPaddy, 1.0)],
        BWh | BWk => vec![(DesertOasis, 4.0), (SteppeNomadic, 1.0)],
        BSh | BSk => vec![(SteppeNomadic, 3.0), (SavannaPastoral, 2.0), (SurveyGrid, 1.0), (DesertOasis, 1.0)],
        Csa | Csb | Csc => vec![(Mediterranean, 5.0), (SurveyGrid, 1.0)],
        Cwa | Cwb | Cwc => vec![(MonsoonPaddy, 4.0), (SavannaPastoral, 1.0), (Highland, 1.0)],
        Cfa | Cfb | Cfc => vec![(TemperateVillage, 3.0), (SurveyGrid, 3.0), (MonsoonPaddy, 1.0)],
        Dsa | Dsb | Dsc | Dsd => vec![(SteppeNomadic, 2.0), (Highland, 1.0), (TemperateVillage, 1.0)],
        Dwa | Dwb | Dwc | Dwd => vec![(MonsoonPaddy, 2.0), (SteppeNomadic, 2.0), (Boreal, 1.0)],
        Dfa | Dfb => vec![(TemperateVillage, 3.0), (SurveyGrid, 3.0), (Boreal, 1.0)],
        Dfc | Dfd => vec![(Boreal, 4.0), (Arctic, 1.0)],
        ET | EF => vec![(Arctic, 4.0), (Boreal, 1.0)],
    };
    if elev > 1800.0 {
        w.push((Highland, 6.0));
    }
    let total: f64 = w.iter().map(|x| x.1).sum();
    let mut x = u * total;
    for (a, wt) in &w {
        if x < *wt {
            return *a;
        }
        x -= wt;
    }
    w.last().unwrap().0
}
