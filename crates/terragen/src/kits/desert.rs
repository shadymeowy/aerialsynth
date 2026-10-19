//! The desert kit (`docs/design/terrain-next.md` §5.1, `docs/next/kits/desert.md`): sand seas whose
//! dunes follow the wind regime, the stony deserts and fringes between them, playas, wadis,
//! yardangs and inselbergs. The WGSL twins are `gpu/wgsl/kits/desert_relief.wgsl` (pass A and
//! the shared helpers) and `gpu/wgsl/kits/desert.wgsl` (pass B), in the same order.
//!
//! * **Relief (pass A, [`relief`])**: the core's transverse dunes are replaced by four dune
//!   types picked by the wind regime ([`wind`]): asymmetric transverse ridges (steady wind),
//!   long linear (seif) ridges along the wind with Y-junctions (bimodal winds), star dunes
//!   (multidirectional winds) and small dunes superimposed on the big ones; isolated
//!   inselbergs (granite domes with pediments), wind-streamlined yardang ridges in the
//!   hyper-arid zone and flat playa floors.
//! * **Surface (pass B, [`azonal`], [`washes`])**: sand coloured by palette and lithology (red,
//!   orange, yellow, grey, black), barchans on a darker stony pavement in the sand-poor fringe of
//!   the sand seas, white salt crust with polygonal cracks, brine pools and paleo-shorelines on
//!   the playas, braided dry washes.
//! * **Biomes (`biomes/desert.yaml`)**: hot and cold deserts by palette and lithology, with
//!   desert varnish, gravel mottling, scrub and oasis groves as kernel layers.
//!
//! Sparse landforms (barchans, star dunes, inselbergs, playas) are instances of jittered 3D
//! lattices whose sites are kept within half a cell of the ellipsoid ([`slab_sites`]): seamless
//! on the sphere, decided from integer hashes and values at the site only. Until the core's
//! instance families (milestone 2) land they are enumerated per sample (≤ 27 cells).

use crate::instances::Family;
use crate::kits::{Kit, ReliefIn};
use crate::noise::*;
use crate::registry::pal;
use crate::stack::{hmode, slot, Layer, Stack};
use crate::surface::{mixc, srgb};
use crate::world::{Ctx, World};
use glam::{DVec2, DVec3};
use std::f64::consts::{PI, TAU};

pub static KIT: Kit = Kit {
    name: "desert",
    biomes_yaml: include_str!("../../biomes/desert.yaml"),
    wgsl: include_str!("../gpu/wgsl/kits/desert.wgsl"),
    layers: &[(slot::AZONAL, azonal), (slot::DISTURBANCE, barchan_field), (slot::LINEAR, qanats), (slot::WATER, washes)],
    host: Some(qanat_host),
    relief: Some(relief),
    relief_wgsl: include_str!("../gpu/wgsl/kits/desert_relief.wgsl"),
    shared_exists: true,
    families: &[
        Family { name: "desert_playas", cell: PLAYA_CELL, jitter: 0.8, reach: PLAYA_REACH, relief: true, exists: playa_exists },
        Family { name: "desert_inselbergs", cell: INS_CELL, jitter: 0.8, reach: INS_REACH, relief: true, exists: inselberg_exists },
    ],
    ..Kit::EMPTY
};

// ============================================================================= constants

// lattice / noise keys (xor the world seed)
const S_WIND2: u64 = 0xDE5E_0002;
const S_REG1: u64 = 0xDE5E_0003;
const S_REG2: u64 = 0xDE5E_0004;
const S_TRANS: u64 = 0xDE5E_0010;
const S_SEIF: u64 = 0xDE5E_0011;
const S_YARD: u64 = 0xDE5E_0013;
const S_YZONE: u64 = 0xDE5E_0014;
const S_STAR: u64 = 0xDE5E_0020;
const S_BAR: u64 = 0xDE5E_0023;
const S_CRACK: u64 = 0xDE5E_0030;
const S_BRINE: u64 = 0xDE5E_0031;
const S_VARN: u64 = 0xDE5E_0032;
const S_STREAK: u64 = 0xDE5E_0033;
const S_SANDTEX: u64 = 0xDE5E_0034;
const S_INSROCK: u64 = 0xDE5E_0035;
const S_RIPPLE: u64 = 0xDE5E_0036;
const S_WASH: u64 = 0xDE5E_0040;
const S_BRAID: u64 = 0xDE5E_0041;
const S_BRAID2: u64 = 0xDE5E_0042;

// land-cover classes (v2 ids; `tests::class_ids` checks them against the registry)
const LC_SAND: u8 = 5;
const LC_ROCK: u8 = 6;
const LC_DRY_RIVERBED: u8 = 29;
const LC_SALT_FLAT: u8 = 30;
const LC_VOLCANIC_ASH: u8 = 32;
const LC_GRAVEL: u8 = 33;

// dune fields: wavelengths (m), waves per lattice cell, kernel extents (crest, wave, up; cells)
const TRANS_LAM: f64 = 600.0;
const SEIF_LAM: f64 = 1700.0;
const YARD_LAM: f64 = 180.0;
const TRANS_EXT: [f64; 3] = [1.3, 0.9, 0.9];
const SEIF_EXT: [f64; 3] = [1.5, 0.8, 0.8];
const YARD_EXT: [f64; 3] = [1.5, 0.6, 0.6];
// the means of the ridge fields (`tests::ridge_means`): what a pixel shows where unresolved
const TRANS_MEAN: f64 = 0.4821;
const SEIF_MEAN: f64 = 0.2633;
const YARD_MEAN: f64 = 0.2272;

// star dunes: cell (m), density, radius range (m), the mean volume per instance / (H R²)
const STAR_CELL: f64 = 2600.0;
const STAR_DENS: f64 = 1.0;
const STAR_R: [f64; 2] = [750.0, 1150.0];
const STAR_CV: f64 = 0.3239;
const ARM_MEAN: f64 = 0.375;
// inselbergs: cell (m), density, the volume of the profile / (H R²)
const INS_CELL: f64 = 16000.0;
const INS_REACH: f64 = 7200.0;
const INS_DENS: f64 = 0.45;
const INS_CV: f64 = 1.4049;
// playas: cell (m), density, the largest hill amplitude (macro field) at the site
const PLAYA_CELL: f64 = 36000.0;
const PLAYA_REACH: f64 = 16000.0;
const PLAYA_DENS: f64 = 0.55;
// barchans: cell (m), density, radius range (m), area and volume of the shape / R², (H R²)
const BAR_CELL: f64 = 380.0;
const BAR_DENS: f64 = 0.5;
const BAR_R: [f64; 2] = [35.0, 95.0];
const BAR_CA: f64 = 1.4777;
const BAR_CV: f64 = 0.2252;

// ============================================================================= proxies

/// Aridity 0 (moist) .. 1 (hyper-arid), from the moisture: a proxy of the atlas precipitation
/// (BWh below ~250 mm, moisture ≲ 0.2).
#[inline]
pub fn aridity(moist: f64) -> f64 {
    1.0 - smoothstep(0.16, 0.30, moist)
}

/// The prevailing wind: the unit direction it blows to (east, north) and its directional
/// variability 0 (one direction) .. 1 (many). The direction is the atlas' annual wind `wa`
/// (m/s), with a latitude-belt proxy (trades, westerlies) where it is calm, turned a little by
/// regional noise; the variability grows where the annual wind is weak (it blows from changing
/// directions), with continental-scale noise (the atlas keeps no seasonal winds).
pub fn wind(seed: u64, p: DVec3, lat: f64, wa: DVec2) -> (DVec2, f64) {
    let latd = lat.abs().to_degrees();
    let ns = if lat >= 0.0 { 1.0 } else { -1.0 };
    let trade = 1.0 - smoothstep(25.0, 35.0, latd);
    let belt = DVec2::new(-std::f64::consts::FRAC_1_SQRT_2, -std::f64::consts::FRAC_1_SQRT_2 * ns) * trade + DVec2::new(0.966, 0.259 * ns) * (1.0 - trade);
    let v = wa + belt * 0.6;
    let th = v.y.atan2(v.x) + 0.3 * perlin3(seed ^ S_WIND2, p / 260_000.0);
    let speed = wa.length();
    let rho = saturate(0.9 - 0.11 * speed + 0.6 * perlin3(seed ^ S_REG1, p / 800_000.0) + 0.25 * perlin3(seed ^ S_REG2, p / 230_000.0));
    (DVec2::new(th.cos(), th.sin()), rho)
}

/// The atlas' annual wind (m/s, east and north) sampled at `p` itself: its cubic B-spline is
/// smooth, while the tile's grid-interpolated value (`Macro`) kinks along the grid lines, which
/// ridge phases a few cells from their lattice point turn into creases.
pub fn atlas_wind(w: &World, p: DVec3) -> DVec2 {
    let a = w.atlas().sample(p.normalize());
    DVec2::new(a.wind_e, a.wind_n)
}

/// Weights of the dune types by the wind variability: transverse, linear, star.
#[inline]
fn regime(rho: f64) -> (f64, f64, f64) {
    let wt = 1.0 - smoothstep(0.30, 0.42, rho);
    let ws = smoothstep(0.62, 0.74, rho);
    (wt, 1.0 - wt - ws, ws)
}

// ============================================================================= helpers

/// Distance of a point (offset `o` of the lattice cell, jitter [0, 0.5)) from `f` along one axis,
/// squared (`World::gully_octave`).
#[inline]
fn gap2(f: f64, o: i64) -> f64 {
    let t = f - o as f64;
    if t < 0.0 {
        t * t
    } else if t > 0.5 {
        (t - 0.5) * (t - 0.5)
    } else {
        0.0
    }
}

/// A periodic ridge profile at phase `x` (0..1): 0 transverse (gentle stoss, a lee face of
/// `lee` of the wavelength), 1 linear (sharp crest, flat corridors), 2 yardang (rounded hulls,
/// wide troughs).
#[inline]
fn profile(kind: u32, x: f64, lee: f64) -> f64 {
    match kind {
        0 => {
            if x < 1.0 - lee {
                let s = x / (1.0 - lee);
                s * s * (3.0 - 2.0 * s)
            } else {
                let s = (1.0 - x) / lee;
                s * s * (3.0 - 2.0 * s)
            }
        }
        1 => {
            let t = 1.0 - (2.0 * x - 1.0).abs();
            ((t - 0.35) / 0.65).max(0.0).powf(1.3)
        }
        _ => {
            let t = 1.0 - (2.0 * x - 1.0).abs();
            let y = ((t - 0.5) / 0.5).max(0.0);
            y * (2.0 - y)
        }
    }
}

/// Oriented ridges (0..1): every point of a jittered 3D lattice (`q` in cells) carries ridges
/// across `ew` (crests along `ec`), `k` per cell, with a random phase, weighted by an anisotropic
/// kernel of extents `ext` (along the crests, across, up; ≤ 1.5 cells). Overlapping points with
/// other phases fork and end the crests (Y-junctions); gaps between points are flat. No
/// wavelength or direction is modulated at ECEF coordinates (R-wave): phases are per point.
#[allow(clippy::too_many_arguments)]
pub fn ridges(seed: u64, q: DVec3, ew: DVec3, ec: DVec3, up: DVec3, k: f64, ext: [f64; 3], kind: u32, lee: f64, warp: f64) -> f64 {
    // sinuous crests: a smooth phase shift (a domain warp at a constant wavelength)
    let x0 = warp * perlin3(seed ^ 0x3A9, q * 1.3);
    let qf = q.floor();
    let (ix, iy, iz) = (qf.x as i64, qf.y as i64, qf.z as i64);
    let f = q - qf;
    let rmax = ext[0].max(ext[1]).max(ext[2]);
    let r2 = rmax * rmax + 1e-9;
    let ia = 1.0 / (ext[0] * ext[0]);
    let ib = 1.0 / (ext[1] * ext[1]);
    let iu = 1.0 / (ext[2] * ext[2]);
    let (mut v, mut wt) = (0.0, 0.0);
    for dz in -2..=2i64 {
        let gz = gap2(f.z, dz);
        if gz > r2 {
            continue;
        }
        for dy in -2..=2i64 {
            let gzy = gz + gap2(f.y, dy);
            if gzy > r2 {
                continue;
            }
            for dx in -2..=2i64 {
                if gzy + gap2(f.x, dx) > r2 {
                    continue;
                }
                let h = hash3(seed, ix + dx, iy + dy, iz + dz);
                let jit = DVec3::new(u01k(h, 1), u01k(h, 2), u01k(h, 3)) * 0.5;
                let pp = f - DVec3::new(dx as f64, dy as f64, dz as f64) - jit;
                let a = pp.dot(ec);
                let b = pp.dot(ew);
                let u = pp.dot(up);
                let d2 = a * a * ia + b * b * ib + u * u * iu;
                if d2 >= 1.0 {
                    continue;
                }
                let w = (1.0 - d2) * (1.0 - d2) * (1.0 - d2);
                let x = k * b + u01k(h, 4) + x0;
                let x = x - x.floor();
                v += w * profile(kind, x, lee);
                wt += w;
            }
        }
    }
    v / wt.max(0.3)
}

/// Up to 4 sites of a jittered 3D lattice (cell `cell`, jitter 0.8) lying within half a cell of
/// the ellipsoid (so each surface point sees about one site per cell², seamless on the sphere)
/// and within `reach` of `p` in the tangent plane: their hash, the offset of the site from `p`
/// (east, north; m) and the surface point below the site.
pub struct Sites {
    pub n: usize,
    pub id: [u64; 4],
    pub rel: [DVec2; 4],
    pub c: [DVec3; 4],
}

pub fn slab_sites(w: &World, seed: u64, p: DVec3, east: DVec3, north: DVec3, cell: f64, reach: f64) -> Sites {
    let mut s = Sites { n: 0, id: [0; 4], rel: [DVec2::ZERO; 4], c: [DVec3::ZERO; 4] };
    let q = p / cell;
    let qf = q.floor();
    let (ix, iy, iz) = (qf.x as i64, qf.y as i64, qf.z as i64);
    let f = q - qf;
    let r3 = (reach * reach + 0.25 * cell * cell) / (cell * cell);
    let gap = |f: f64, o: i64| -> f64 {
        let lo = o as f64 + 0.1;
        let hi = o as f64 + 0.9;
        let g = (lo - f).max(f - hi).max(0.0);
        g * g
    };
    let (ia2, ib2) = (1.0 / (w.ell.a * w.ell.a), 1.0 / (w.ell.b * w.ell.b));
    for dz in -1..=1i64 {
        let gz = gap(f.z, dz);
        for dy in -1..=1i64 {
            let gzy = gz + gap(f.y, dy);
            for dx in -1..=1i64 {
                if gzy + gap(f.x, dx) > r3 {
                    continue;
                }
                let (cx, cy, cz) = (ix + dx, iy + dy, iz + dz);
                let h = hash3(seed, cx, cy, cz);
                let pt = DVec3::new(
                    cx as f64 + 0.5 + 0.8 * (u01k(h, 1) - 0.5),
                    cy as f64 + 0.5 + 0.8 * (u01k(h, 2) - 0.5),
                    cz as f64 + 0.5 + 0.8 * (u01k(h, 3) - 0.5),
                ) * cell;
                // the height of the site above the ellipsoid (along the ray from the centre)
                let s2 = (pt.x * pt.x + pt.y * pt.y) * ia2 + pt.z * pt.z * ib2;
                let sq = s2.sqrt();
                let hg = pt.length() * (1.0 - 1.0 / sq);
                if hg.abs() >= 0.5 * cell {
                    continue;
                }
                let d = pt - p;
                let rel = DVec2::new(d.dot(east), d.dot(north));
                if rel.length_squared() > reach * reach || s.n >= 4 {
                    continue;
                }
                s.id[s.n] = h;
                s.rel[s.n] = rel;
                s.c[s.n] = pt / sq;
                s.n += 1;
            }
        }
    }
    s
}

/// The tangent-plane direction of a wind (east, north) and the direction across it (ECEF).
#[inline]
fn wind_axes(c: &Ctx, wd: DVec2) -> (DVec3, DVec3) {
    let ew = c.east * wd.x + c.north * wd.y;
    (ew, c.up.cross(ew))
}

// ============================================================================= relief (pass A)

/// Star dunes (m, for 35 m dunes): pyramids of 3–5 sinuous arms on the lattice of `STAR_CELL`,
/// thinned by `amount`; their mean where unresolved.
fn stars(w: &World, seed: u64, c: &Ctx, amount: f64) -> f64 {
    let gsd = c.gsd;
    let er3 = (STAR_R[0] + STAR_R[1]) * (STAR_R[0] * STAR_R[0] + STAR_R[1] * STAR_R[1]) / 4.0;
    let mean = amount * STAR_DENS * STAR_CV * 0.2 * er3 / (STAR_CELL * STAR_CELL);
    // (switched off where the explicit share is a few per cent: R-cutoff)
    let e = smoothstep(1.2 * gsd, 3.0 * gsd, STAR_R[0]);
    if e <= 0.02 {
        return mean;
    }
    let st = slab_sites(w, seed ^ S_STAR, c.p, c.east, c.north, STAR_CELL, 0.45 * STAR_CELL);
    let mut best: f64 = 0.0;
    for i in 0..st.n {
        let h = st.id[i];
        if u01k(h, 5) >= STAR_DENS * amount {
            continue;
        }
        let r = STAR_R[0] + (STAR_R[1] - STAR_R[0]) * u01k(h, 6);
        let hh = r * (0.14 + 0.12 * u01k(h, 7));
        let v = -st.rel[i];
        let d = v.length();
        if d >= r {
            continue;
        }
        let n = 3.0 + (3.0 * u01k(h, 8)).floor();
        let th = v.y.atan2(v.x);
        let arm = 0.5 + 0.5 * (n * (th - TAU * u01k(h, 9)) + 0.7 * (th + TAU * u01k(h, 10)).sin()).cos();
        let a2 = ARM_MEAN + (arm * arm - ARM_MEAN) * band(TAU * d / (3.0 * n), gsd);
        let x = d / (r * (0.42 + 0.58 * a2));
        if x < 1.0 {
            best = best.max(hh * (1.0 - x).powf(1.5));
        }
    }
    e * best + (1.0 - e) * mean
}

/// The macro fields at a family site that the existence tests read: moisture, mountain mask,
/// hill amplitude (m), sand-sea and mesa masks, continent value.
struct SiteFields {
    moist: f64,
    temp: f64,
    mountain: f64,
    hills: f64,
    sand: f64,
    mesa: f64,
    cont: f64,
}

fn site_fields(w: &World, c: DVec3) -> SiteFields {
    let m = w.macro_at(c, 20_000.0);
    let r = &w.cfg.relief;
    // `World::mountain_mask` and `World::hill_amplitude` (private in the core: the same formulas)
    let b1 = 1.0 - m.belt.abs();
    let b2 = 1.0 - m.belt2.abs();
    let belt = (b1 * 0.75 + b2 * 0.45 + 0.35 * m.belt_var).max(0.0);
    let mb = smoothstep(0.62, 0.92, belt) * smoothstep(-0.04, 0.08, m.cont);
    let u = m.uplift.max(0.0) * r.tectonic_mountains;
    let mt = smoothstep(0.08, 0.5, u) * smoothstep(-0.2, 0.05, m.cont);
    let wb = (r.belt_mountains / 0.45 * 0.55).clamp(0.0, 1.0);
    let mountain = (1.0 - (1.0 - wb * mb) * (1.0 - mt)).clamp(0.0, 1.0);
    let hills = r.hill_height_m * (0.15 + 0.85 * smoothstep(-0.5, 0.6, m.hill_amp)) * (0.25 + 0.75 * smoothstep(-0.06, 0.05, m.cont));
    let moist = m.moist.clamp(0.0, 1.0);
    let sand = (1.0 - smoothstep(0.1, 0.28, moist)) * smoothstep(-0.25, 0.15, m.sand);
    SiteFields { moist, temp: m.temp, mountain, hills, sand, mesa: smoothstep(0.1, 0.45, m.mesa), cont: m.cont }
}

/// Family indices (`FAM_DESERT_*` in WGSL).
fn fam(name: &str) -> usize {
    crate::instances::families().position(|f| f.name == name).expect("a desert family")
}
fn fam_playas() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| fam("desert_playas"))
}
fn fam_inselbergs() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| fam("desert_inselbergs"))
}

/// Playas exist at arid, flat sites (no mountains, low hills, outside sand seas and mesas).
fn playa_exists(w: &World, s: &crate::instances::InstSite) -> Option<[f32; 8]> {
    if u01k(s.id, 5) >= PLAYA_DENS {
        return None;
    }
    // cheap tests on the atlas first (the macro fields cost ~60 noise octaves)
    let a = w.atlas().sample(s.center.normalize());
    if World::moisture_index(a.precip_mm, a.temp_c) > 0.33 || a.temp_c < 2.0 || a.coast_km < 0.0 {
        return None;
    }
    let f = site_fields(w, s.center);
    if aridity(f.moist) < 0.5 || f.temp < 4.0 || f.cont < 0.05 || f.mountain > 0.1 || f.hills > 130.0 || f.sand > 0.4 || f.mesa > 0.3 {
        return None;
    }
    Some([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

/// Inselbergs exist at dry sites off the mountains and sand seas, mostly on crystalline rock
/// (the atlas' lithology at the site).
fn inselberg_exists(w: &World, s: &crate::instances::InstSite) -> Option<[f32; 8]> {
    if u01k(s.id, 5) >= INS_DENS {
        return None;
    }
    let a = w.atlas().sample(s.center.normalize());
    if (a.litho as u8 != 2 && u01k(s.id, 13) >= 0.3) || World::moisture_index(a.precip_mm, a.temp_c) > 0.52 || a.coast_km < 0.0 {
        return None;
    }
    let f = site_fields(w, s.center);
    if f.moist > 0.42 || f.cont < 0.05 || f.mountain > 0.15 || f.sand > 0.6 || f.mesa > 0.3 {
        return None;
    }
    Some([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

/// The offset (east, north; m) of `p` from an instance centre.
#[inline]
fn offset(p: DVec3, center: DVec3, east: DVec3, north: DVec3) -> DVec2 {
    let d = p - center;
    DVec2::new(d.dot(east), d.dot(north))
}

/// The shape of inselberg `id` at the offset `v` (m) from its centre: the normalized distance
/// (elongated, with a lobed outline), its radius and height (m).
fn ins_shape(id: u64, v: DVec2) -> (f64, f64, f64) {
    let u6 = u01k(id, 6);
    let r = 500.0 + 2100.0 * u6 * u6;
    let hh = (r * (0.18 + 0.22 * u01k(id, 7))).min(600.0);
    let ex = 1.0 + 0.7 * u01k(id, 8);
    let (sa, ca) = (TAU * u01k(id, 9)).sin_cos();
    let (vx, vy) = (v.x * ca + v.y * sa, -v.x * sa + v.y * ca);
    let dn = (vx * vx / ex + vy * vy * ex).sqrt();
    let th = vy.atan2(vx);
    let edge =
        1.0 + 0.10 * (3.0 * th + TAU * u01k(id, 10)).sin() + 0.06 * (5.0 * th + TAU * u01k(id, 11)).sin() + 0.035 * (9.0 * th + TAU * u01k(id, 12)).sin();
    (dn / (r * edge), r, hh)
}

/// The profile of an inselberg (× its height): a dome with steep flanks and a pediment.
#[inline]
fn ins_profile(x: f64) -> f64 {
    let dome = if x < 1.0 { (1.0 - x.powf(1.5)).max(0.0).powf(1.2) } else { 0.0 };
    dome + 0.04 * (1.0 - smoothstep(0.9, 1.6, x))
}

/// Inselbergs (m): granite domes with a pediment; widened with their volume kept where below
/// the pixel.
fn inselbergs(inst: &[crate::instances::Instance], c: &Ctx) -> f64 {
    let gsd = c.gsd;
    let mut best: f64 = 0.0;
    for i in inst {
        let v = offset(c.p, i.center, c.east, c.north);
        let (x, r, hh) = ins_shape(i.id, v);
        let e = smoothstep(1.2 * gsd, 3.0 * gsd, r);
        let hx = if e > 0.0 && x < 1.6 { hh * ins_profile(x) } else { 0.0 };
        let rb = (0.95 * INS_REACH).min((1.6 * r).max(3.0 * gsd));
        let hp = 2.0 * hh * r * r * INS_CV / (PI * rb * rb);
        let db = v.length() / rb;
        let blob = if db < 1.0 { hp * (1.0 - db * db) } else { 0.0 };
        best = best.max(e * hx + (1.0 - e) * blob);
    }
    best
}

/// The shape of playa `id` at the offset `v` (m) from its centre: the normalized distance with
/// a ragged shore, and its radius (m).
fn playa_shape(id: u64, v: DVec2) -> (f64, f64) {
    let r = 2000.0 + 4500.0 * u01k(id, 6);
    let ex = 1.0 + 0.8 * u01k(id, 7);
    let (sa, ca) = (TAU * u01k(id, 8)).sin_cos();
    let (vx, vy) = (v.x * ca + v.y * sa, -v.x * sa + v.y * ca);
    let dn = (vx * vx / ex + vy * vy * ex).sqrt();
    let th = vy.atan2(vx);
    let edge = 1.0 + 0.14 * (2.0 * th + TAU * u01k(id, 9)).sin() + 0.08 * (3.0 * th + TAU * u01k(id, 10)).sin() + 0.04 * (7.0 * th + TAU * u01k(id, 11)).sin();
    (dn / (r * edge), r)
}

/// The playa nearest (in normalized distance) to `p`: (x, radius, hash).
fn playa(inst: &[crate::instances::Instance], p: DVec3, east: DVec3, north: DVec3) -> Option<(f64, f64, u64)> {
    let mut best: Option<(f64, f64, u64)> = None;
    for i in inst {
        let (x, r) = playa_shape(i.id, offset(p, i.center, east, north));
        if best.is_none_or(|b| x < b.0) {
            best = Some((x, r, i.id));
        }
    }
    best
}

/// The kit's relief: dunes by wind regime (replacing the core's), inselbergs, playa floors,
/// yardangs.
pub fn relief(w: &World, r: &ReliefIn, h: &mut f64) {
    let c = r.ctx;
    let gsd = c.gsd;
    let seed = w.seed;
    let dh = w.cfg.relief.dune_height_m;
    // ---- dunes: the core's transverse ridges out, the types of the wind regime in
    if r.sand > 1e-3 && dh > 0.0 {
        let core = dh * r.sand * w.dunes(c, r.m, gsd);
        let (wd, rho) = wind(seed, c.p, c.lat, atlas_wind(w, c.p));
        let (wt, wl, ws) = regime(rho);
        let env = smoothstep(0.35, 0.75, r.sand);
        let (ew, ec) = wind_axes(c, wd);
        let mut sum = 0.0;
        if env > 0.0 {
            // (star dunes rise from a network of transverse ridges)
            let wtr = wt + 0.5 * ws;
            if wtr > 0.0 {
                let b = band(TRANS_LAM, gsd);
                let mut v = TRANS_MEAN;
                if b > 0.0 {
                    let lee = (2.5 * gsd / TRANS_LAM).clamp(0.14, 0.5);
                    v += (ridges(seed ^ S_TRANS, c.p / (2.0 * TRANS_LAM), ew, ec, c.up, 2.0, TRANS_EXT, 0, lee, 0.18) - TRANS_MEAN) * b;
                }
                sum += wtr * 1.4 * v;
            }
            if wl > 0.0 {
                let b = band(SEIF_LAM, gsd);
                let mut v = SEIF_MEAN;
                if b > 0.0 {
                    v += (ridges(seed ^ S_SEIF, c.p / (3.0 * SEIF_LAM), ec, ew, c.up, 3.0, SEIF_EXT, 1, 0.5, 0.22) - SEIF_MEAN) * b;
                }
                sum += wl * 1.9 * v;
            }
        }
        let mut star = 0.0;
        if ws * env > 0.0 {
            star = stars(w, seed, c, ws * env) * (dh / 35.0);
        }
        // (the small dunes over the big ones: v0.2.0's network of ridged noise in three frames, the
        // core's, kept at half its height; the kit's own small transverse ridges drew a regular
        // comb in the corridors)
        *h += dh * r.sand * env * sum + star - 0.5 * core;
    }
    // ---- inselbergs
    let ins = r.instances(w, fam_inselbergs());
    if !ins.is_empty() {
        *h += inselbergs(&ins, c);
    }
    // ---- playa floors: flattened towards the smooth relief, a little below it
    if let Some((x, _, _)) = playa(&r.instances(w, fam_playas()), c.p, c.east, c.north) {
        if x < 1.0 {
            let k = 1.0 - smoothstep(0.8, 1.0, x);
            *h += (r.smooth - 2.0 - *h) * k;
        }
    }
    // ---- yardangs: streamlined ridges along the wind on the hyper-arid plains
    let hyper = (1.0 - smoothstep(0.05, 0.12, r.moist)) * smoothstep(-2.0, 2.0, r.temp);
    if hyper > 0.0 && r.sand < 0.2 {
        let zone = hyper * smoothstep(0.1, 0.3, perlin3(seed ^ S_YZONE, c.p / 120_000.0)) * (1.0 - smoothstep(0.02, 0.2, r.sand)) * (1.0 - r.mountain);
        if zone > 0.0 {
            let b = band(YARD_LAM, gsd);
            let mut v = YARD_MEAN;
            if b > 0.0 {
                let (wd, _) = wind(seed, c.p, c.lat, atlas_wind(w, c.p));
                let (ew, ec) = wind_axes(c, wd);
                v += (ridges(seed ^ S_YARD, c.p / (2.0 * YARD_LAM), ec, ew, c.up, 2.0, YARD_EXT, 2, 0.5, 0.1) - YARD_MEAN) * b;
            }
            *h += 12.0 * zone * v;
        }
    }
}

// ============================================================================= surface (pass B)

/// Barchans at `p`: (coverage, height) of sand crescents, horns downwind, on the lattice of
/// `BAR_CELL` thinned by `amount`; the mean where unresolved.
#[allow(clippy::too_many_arguments)]
fn barchans(w: &World, seed: u64, p: DVec3, east: DVec3, north: DVec3, wd: DVec2, amount: f64, gsd: f64, fw: f64) -> (f64, f64) {
    let (r0, r1) = (BAR_R[0], BAR_R[1]);
    let er2 = (r0 * r0 + r0 * r1 + r1 * r1) / 3.0;
    let er3 = (r0 + r1) * (r0 * r0 + r1 * r1) / 4.0;
    let mc = amount * BAR_DENS * BAR_CA * er2 / (BAR_CELL * BAR_CELL);
    let mh = BAR_CV / BAR_CA * 0.12 * er3 / er2;
    let e = smoothstep(1.2 * gsd, 3.0 * gsd, r0);
    if e <= 0.0 {
        return (mc, mh);
    }
    let st = slab_sites(w, seed ^ S_BAR, p, east, north, BAR_CELL, 0.45 * BAR_CELL);
    let (mut cov, mut dh): (f64, f64) = (0.0, 0.0);
    for i in 0..st.n {
        let h = st.id[i];
        if u01k(h, 5) >= BAR_DENS * amount {
            continue;
        }
        let r = r0 + (r1 - r0) * u01k(h, 6);
        // (low crescents: 16-26 % of the radius drew bright lit crests and slip-face rims on the plain, a11)
        let hh = r * (0.09 + 0.06 * u01k(h, 7));
        let a = 0.5 * (u01k(h, 8) - 0.5);
        let (sa, ca) = a.sin_cos();
        let dw = DVec2::new(wd.x * ca - wd.y * sa, wd.x * sa + wd.y * ca);
        let v = -st.rel[i] / r;
        let (u, s) = (v.dot(dw), v.x * dw.y - v.y * dw.x);
        let so = 1.0 - (u * u + s * s).sqrt();
        let si = ((u - 0.55) * (u - 0.55) + s * s).sqrt() - 0.85;
        let ci = ((so * r / fw + 0.5).clamp(0.0, 1.0)) * ((si * r / fw + 0.5).clamp(0.0, 1.0));
        if ci <= 0.0 {
            continue;
        }
        let a1 = (so / 0.7).clamp(0.0, 1.0);
        let a2 = (si / 0.3).clamp(0.0, 1.0);
        let taper = (1.0 - s * s).max(0.0).powf(1.5);
        let hx = hh * (a1 * a1 * (3.0 - 2.0 * a1)).min(a2 * a2 * (3.0 - 2.0 * a2)) * taper;
        cov = cov.max(ci);
        dh = dh.max(hx);
    }
    let c = e * cov + (1.0 - e) * mc;
    if c <= 0.0 {
        return (0.0, 0.0);
    }
    (c, (e * cov * dh + (1.0 - e) * mc * mh) / c)
}

/// The sand colour of the sample: the biome's sand palette by the regional style, tinted by
/// the ecoregion's lithology (iron-red on crystalline rock, pale on carbonate, yellow on
/// unconsolidated sediment, dark grey to black on volcanic rock).
fn sand_colour(s: &Stack) -> DVec3 {
    let si = s.t.style[1] * 2.0;
    let j = (si.floor() as usize).min(1);
    let base = mixc(s.pal(pal::SAND + j), s.pal(pal::SAND + j + 1), si - j as f64);
    let lum = base.dot(DVec3::new(0.2126, 0.7152, 0.0722));
    match s.bio.style.litho {
        1 => mixc(base, DVec3::splat(lum * 1.15), 0.35),
        2 => base * DVec3::new(1.10, 0.94, 0.84),
        3 => mixc(base, DVec3::splat(0.045), 0.8),
        4 => base * DVec3::new(1.04, 1.0, 0.9),
        _ => base,
    }
}

/// The instances of family `f` at a sample (`Stack::instances` without borrowing the stack).
fn insts<'a>(l: &'a crate::surface::Local, w: &World, p: DVec3, f: usize) -> std::borrow::Cow<'a, [crate::instances::Instance]> {
    match l.inst.get(f) {
        Some(v) => std::borrow::Cow::Borrowed(&v[..]),
        None => std::borrow::Cow::Owned(crate::instances::near(w, f, p, 0.0)),
    }
}

/// Salt crust: white to beige (`tone`, `shade` 0..1) plates 8–20 m between dark cracks, a
/// darker, wetter floor where `mid` → 1, and pink or green (`pink`) brine pools where `pools`
/// < 0.5.
fn crust(s: &Stack, tone: f64, shade: f64, mid: f64, pools: f64, pink: f64) -> DVec3 {
    let seed = s.world.seed;
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let fw = s.l.fw;
    let crust_w = DVec3::new(0.80, 0.78, 0.72) * (0.92 + 0.08 * shade);
    let tint = [DVec3::new(1.0, 1.0, 1.0), DVec3::new(1.0, 0.95, 0.86), DVec3::new(1.0, 0.93, 0.92)][(3.0 * tone) as usize % 3];
    let mut col = crust_w * tint;
    let ce = smoothstep(1.2 * gsd, 3.0 * gsd, 12.0);
    let mut crack = 0.06;
    let mut plate = 1.0;
    if ce > 0.0 {
        // (a 3D partition: seamless, and the same on standing water, which has no local frame)
        let cl = worley3(seed ^ S_CRACK, p, 12.0, 0.85);
        let ed = worley_edge_dist(&cl, p);
        crack = ce * crate::kernels::band_cov(ed, 0.35, fw) + (1.0 - ce) * 0.06;
        plate = 1.0 + ce * 0.06 * (u01k(cl.id, 1) - 0.5);
    }
    col = col * plate * (1.0 - 0.35 * crack);
    col = mixc(col, col * DVec3::new(0.78, 0.76, 0.72), 0.5 * mid);
    if pools < 0.5 {
        let pn = perlin3(seed ^ S_BRINE ^ 7, p / 400.0) + 0.5 * perlin3(seed ^ S_BRINE ^ 9, p / 130.0) * band(130.0, gsd);
        let pool = smoothstep(0.25, 0.32, pn + 0.6 * mid - 0.4) * mid;
        let brine = if pink < 0.5 { srgb(214.0, 150.0, 156.0) } else { srgb(120.0, 156.0, 120.0) };
        col = mixc(col, brine, 0.85 * pool);
    }
    col
}

/// Slot 3: playas, the sand seas' surface, their stony fringe and its barchans.
pub fn azonal(s: &mut Stack) {
    let t = *s.t;
    let w = s.world;
    let seed = w.seed;
    let c = *s.ctx;
    let p = c.p;
    let gsd = c.gsd;
    let fw = s.l.fw;
    // ---- playas: salt crust with polygonal cracks, brine pools, paleo-shorelines
    if let Some((x, r, id)) = playa(&insts(s.l, w, p, fam_playas()), p, c.east, c.north) {
        if x < 1.4 {
            let dm = (1.0 - x) * r;
            let cov = (dm / fw.max(1e-3) + 0.5).clamp(0.0, 1.0);
            if cov > 0.0 {
                // the wetter, darker floor in the middle
                let mid = 1.0 - smoothstep(0.2, 0.6, x + 0.15 * perlin3(seed ^ S_BRINE, p / 900.0));
                let col = crust(s, u01k(id, 13), u01k(id, 12), mid, u01k(id, 14), u01k(id, 15));
                s.composite(Layer { cov, albedo: col, cls: LC_SALT_FLAT, clear: 1.0, ..Default::default() });
            }
            // paleo-shorelines: pale rings on the slopes around the floor
            if x > 0.98 {
                let mut ring = 0.0;
                for xi in [1.05, 1.13, 1.24, 1.36] {
                    ring += crate::kernels::band_cov((x - xi) * r, 0.006 * r, fw);
                }
                let ring = ring.min(1.0) * band(0.03 * r, gsd);
                if ring > 0.0 {
                    let crust_w = DVec3::new(0.80, 0.78, 0.72) * (0.92 + 0.08 * u01k(id, 12));
                    s.composite(Layer::paint(0.45 * ring, mixc(s.col, crust_w, 0.7)));
                }
            }
        }
    }
    // ---- inselbergs: bare granite, streaked dark by varnish down the flanks
    let l = s.l;
    for i in insts(l, w, p, fam_inselbergs()).iter() {
        let (x, r, _) = ins_shape(i.id, offset(p, i.center, c.east, c.north));
        let cov = ((0.9 - x) * r / fw.max(1e-3) + 0.5).clamp(0.0, 1.0);
        if cov > 0.0 {
            // granite: the rock palette towards grey, darker varnish streaks and joints
            let r0 = s.pal(pal::ROCK + 1);
            let rock = mixc(r0, DVec3::splat(r0.dot(DVec3::new(0.2126, 0.7152, 0.0722))), 0.45) * 1.15;
            let sn = perlin3(seed ^ S_INSROCK, p / 60.0) * band(60.0, gsd) + 0.5 * perlin3(seed ^ S_INSROCK ^ 1, p / 15.0) * band(15.0, gsd);
            let mut col = mixc(rock, rock * DVec3::new(0.45, 0.42, 0.40), (0.25 + 0.5 * sn).clamp(0.0, 1.0));
            let je = smoothstep(1.2 * gsd, 3.0 * gsd, 45.0);
            if je > 0.0 {
                let jc = worley3(seed ^ S_INSROCK ^ 2, p, 45.0, 0.9);
                col *= 1.0 - 0.45 * je * crate::kernels::band_cov(worley_edge_dist(&jc, p), 0.8, fw);
            }
            s.composite(Layer { cov, albedo: col, cls: LC_ROCK, clear: 0.9, ..Default::default() });
        }
    }
    if t.sand <= 0.0 {
        return;
    }
    // ---- sand seas
    let (wd, rho) = wind(seed, p, c.lat, DVec2::from_array(t.wind));
    let (_, wl, _) = regime(rho);
    let sc = sand_colour(s);
    let black = s.bio.style.litho == 3;
    let patch = s.pf.patch;
    let erg = smoothstep(0.47, 0.53, t.sand + 0.1 * patch);
    // (the stony desert pavement is the arid lands': semi-arid scrub keeps its biome's ground)
    let fringe = (1.0 - erg) * smoothstep(0.0, 0.08, t.sand) * aridity(t.moist);
    if fringe > 0.0 {
        // the stony pavement between the barchans: varnished gravel, sand sheets streaked
        // along the wind
        let soil = s.pal(pal::SOIL + 1);
        let mut pav = mixc(soil, s.pal(pal::ROCK), 0.3) * 0.9;
        let vn = perlin3(seed ^ S_VARN, p / 420.0) + 0.5 * perlin3(seed ^ S_VARN ^ 1, p / 130.0) * band(130.0, gsd);
        let ve = smoothstep(1.2 * gsd, 3.0 * gsd, 130.0);
        // (a gradual mottling: a crisp threshold drew a camouflage of dark and pale patches)
        let varn = ve * smoothstep(-0.15, 0.45, vn) + (1.0 - ve) * 0.3;
        pav = mixc(pav, pav * DVec3::new(0.72, 0.68, 0.66), 0.5 * varn);
        let sb = band(45.0, gsd);
        let mut streak = 0.2;
        if sb > 0.0 {
            let ec = c.up.cross(c.east * wd.x + c.north * wd.y);
            let (g, _) = World::gully_octave(seed ^ S_STREAK, p / 45.0, ec);
            // (faint: crisp pale streaks read as scratches over the plain)
            streak = 0.2 + 0.4 * (smoothstep(0.2, 0.9, g) - 0.2) * sb;
        }
        pav = mixc(pav, sc, 0.35 * streak * (2.0 * t.sand).min(1.0));
        s.composite(Layer { cov: fringe, albedo: pav, cls: LC_GRAVEL, clear: 0.6, ..Default::default() });
    }
    if erg > 0.0 {
        let mut col =
            sc * (1.0 + 0.05 * perlin3(seed ^ S_SANDTEX, p / 35.0) * band(35.0, gsd) + 0.03 * perlin3(seed ^ S_SANDTEX ^ 1, p / 9.0) * band(9.0, gsd));
        // wind ripples across the wind (seen from a few hundred metres)
        let rb = band(5.0, gsd);
        if rb > 0.0 {
            let (rp, _) = World::gully_octave(seed ^ S_RIPPLE, p / 5.0, c.east * wd.x + c.north * wd.y);
            col *= 1.0 + 0.09 * rp * rb;
        }
        // slip faces: finer, paler sand
        col *= 1.0 + 0.07 * smoothstep(0.15, 0.5, s.l.slope) * (1.0 - smoothstep(10.0, 40.0, gsd));
        if wl > 0.0 {
            // the corridors between linear dunes: coarser, darker lag and sparse plants
            let res = 1.0 - smoothstep(10.0, 80.0, gsd);
            let flat = 1.0 - smoothstep(0.02, 0.08, s.l.slope);
            let k = wl * (res * flat + (1.0 - res) * 0.45) * 0.5;
            col = mixc(col, mixc(s.pal(pal::SOIL + 1), col, 0.5) * 0.92, k);
        }
        let cls = if black { LC_VOLCANIC_ASH } else { LC_SAND };
        s.composite(Layer { cov: erg, albedo: col, cls, clear: 1.0, ..Default::default() });
    }
}

/// Slot 4 (after the core's micro-relief resets the height): barchans on the sand-poor fringe
/// of the sand seas of hyper-arid deserts, most where the wind is steady.
pub fn barchan_field(s: &mut Stack) {
    let t = *s.t;
    // (only near the ergs of hyper-arid deserts: crescents strewn over semi-arid scrub, steppe
    // and even paddy land read as toys)
    let z0 = smoothstep(0.08, 0.2, t.sand) * (1.0 - smoothstep(0.42, 0.58, t.sand)) * (1.0 - smoothstep(0.1, 0.2, t.moist));
    if z0 <= 0.0 {
        return;
    }
    let w = s.world;
    let c = *s.ctx;
    let (wd, rho) = wind(w.seed, c.p, c.lat, DVec2::from_array(t.wind));
    let zb = z0 * (0.35 + 0.65 * (1.0 - smoothstep(0.35, 0.6, rho)));
    let (cov, dh) = barchans(w, w.seed, c.p, c.east, c.north, wd, zb, c.gsd, s.l.fw);
    if cov > 0.0 {
        let cls = if s.bio.style.litho == 3 { LC_VOLCANIC_ASH } else { LC_SAND };
        // (the pavement's tone in their sand: pale crescents on a dark plain read as a pattern)
        let sc = mixc(sand_colour(s), s.pal(pal::SOIL + 1), 0.5) * 0.98;
        s.composite(Layer { cov, albedo: sc, dh, hmode: hmode::ADD, cls, clear: 1.0, ..Default::default() });
    }
}

/// Slot 8: dry washes (wadis) wider than the channel, with braided threads of gravel and sand
/// bars parallel to the banks that merge and split.
pub fn washes(s: &mut Stack) {
    if s.is_water {
        dry_lake(s);
        return;
    }
    let l = s.l;
    let t = s.t;
    if l.river_hw <= 0.0 || t.river_wet >= 0.5 || t.sand > 0.45 {
        return;
    }
    // (arid land only: in sub-humid farmland the streams carry tree lines, not sandy washes)
    let ar = (1.0 - smoothstep(0.2, 0.34, t.moist)) * smoothstep(-4.0, 0.0, t.temp) * (1.0 - smoothstep(0.3, 0.45, t.sand));
    if ar <= 0.0 {
        return;
    }
    let seed = s.world.seed;
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let hw = l.river_hw;
    let half = hw * (1.0 + 1.4 * ar) * (1.0 + 0.4 * perlin3(seed ^ S_WASH, p / 700.0));
    let fwr = l.fw.max(gsd * 0.35);
    // (washes on the floors, not down steep walls: their sand and gravel drew pale ribbons up
    // the canyon walls, canyon@type; and fresh sand only along some stretches, desert pavement
    // between them: continuous pale ribbons of one width read as roads)
    let fresh = 0.35 + 0.65 * smoothstep(-0.45, 0.35, perlin3(seed ^ S_WASH ^ 0x55, p / 380.0));
    // (none along the streams of the arid canyon plateaus: their straight pale ribbons ran across
    // the canyon's floor and up its walls like roads, canyon@type)
    let plateau = if hw < 15.0 { 1.0 - smoothstep(0.05, 0.3, t.mesa) } else { 1.0 };
    let cov = crate::kernels::band_cov(l.river_d, half, fwr) * (1.0 - smoothstep(0.25, 0.55, l.slope)) * fresh * plateau;
    if cov <= 0.0 {
        return;
    }
    let lam = (0.25 * half).max(3.0);
    let ph = l.river_d / lam + 0.9 * perlin3(seed ^ S_BRAID, p / 90.0) + 0.4 * perlin3(seed ^ S_BRAID2, p / 31.0) * band(31.0, gsd);
    let th = 0.5 + 0.5 * (TAU * ph).cos();
    let bw = band(lam, gsd);
    // (a third of the way to the ground's colour: pale sand ribbons stood out like roads)
    let bar = mixc(mixc(s.pal(pal::SAND + 2), s.pal(pal::SOIL + 1), 0.25) * 1.02, s.col, 0.45);
    let chan = mixc(bar, s.pal(pal::ROCK), 0.25) * 0.9;
    let col = mixc(bar, chan, smoothstep(0.7, 0.9, th) * bw + (1.0 - bw) * 0.15);
    s.composite(Layer { cov, albedo: col, cls: LC_DRY_RIVERBED, clear: 0.8, ..Default::default() });
}

/// Slot 8 on lakes: arid lakes are dry salt pans, from their shallow margins inwards (the
/// deeper the lake, the more arid it must be to dry out completely): a crisp shore that moves
/// with the aridity, the crust at the lake's level (solid: the sample is land again).
fn dry_lake(s: &mut Stack) {
    let l = s.l;
    if l.water_kind != crate::world::water::LAKE {
        return;
    }
    let ar = aridity(s.t.moist) * smoothstep(0.0, 4.0, s.t.temp);
    if ar <= 0.0 {
        return;
    }
    let depth = l.water - l.ground;
    let dmax = 40.0 * smoothstep(0.4, 0.9, ar).powi(2);
    let cov = ((dmax - depth) / 0.4 + 0.5).clamp(0.0, 1.0);
    if cov <= 0.0 {
        return;
    }
    let st = s.t.style;
    let mid = smoothstep(0.3, 1.0, depth / dmax.max(1e-3));
    let col = crust(s, st[3], st[0], mid, st[2], st[1]);
    s.composite(Layer { cov, albedo: col, cls: LC_SALT_FLAT, solid: true, ..Default::default() });
}

// ============================================================================= qanats

/// Qanats (`docs/design/terrain-next.md` §5.1, P2): underground channels from a mother well at
/// the foot of the mountains down to an oasis, seen as dotted lines of shaft mounds (rings of
/// spoil around a dark shaft every 30–55 m). Host-built (shared by both backends): a lattice of
/// 9 km cells, a qanat group where the culture digs them (desert oasis, highland, steppe and
/// Mediterranean archetypes), the climate is arid and the ground is a piedmont (the smooth
/// relief's slope 0.8–6 % with mountains near); 1–3 lines fan out upslope from a common outlet,
/// each 3–8 km long with a bend. The layer (slot 6) draws the mounds, or their mean along the
/// line where they are below the pixel.
pub const QANAT_CELL: f64 = 9_000.0;
/// The feature kind of a qanat's segments (the kit's kind range: 64…).
pub const KIND_QANAT: u32 = 64;
/// Qanats are listed for areas whose pixel is below this size (m): a line of mounds covers
/// ≤ ~5 % of a pixel above.
pub const QANAT_GSD_MAX: f64 = 60.0;

/// The segments of the qanat group of lattice cell `c` (empty: none).
pub fn qanat_group(w: &World, c: (i64, i64, i64)) -> Vec<crate::features::FSeg> {
    use crate::kits::settlements::sites::{lattice_site, seed_v};
    let mut out = Vec::new();
    let Some((id, p)) = lattice_site(w, 0x0A4A7, QANAT_CELL, c) else { return out };
    let a = w.atlas().sample(p.normalize());
    let arch = [0.0, 0.0, 0.05, 0.65, 0.2, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35][(a.archetype as usize).min(11)];
    let prob = arch * (1.0 - 0.6 * a.development) * (1.0 - smoothstep(250.0, 450.0, a.precip_mm)) * smoothstep(6.0, 12.0, a.temp_c);
    if u01k(id, 1) >= prob || a.coast_km < 5.0 || a.glaciation > 0.1 || w.cfg.landuse.towns <= 0.0 {
        return out;
    }
    // a piedmont: a moderate regional slope, mountains near
    let (h0, amp) = w.smooth_relief(p, 2_000.0);
    if h0 < 100.0 || amp < 120.0 {
        return out;
    }
    let (_, _, east, north) = crate::kits::settlements::places::frame_at(w, p);
    let d = 1_500.0;
    let ge = (w.smooth_relief(p + east * d, 1_000.0).0 - w.smooth_relief(p - east * d, 1_000.0).0) / (2.0 * d);
    let gn = (w.smooth_relief(p + north * d, 1_000.0).0 - w.smooth_relief(p - north * d, 1_000.0).0) / (2.0 * d);
    let g = DVec2::new(ge, gn);
    let slope = g.length();
    if !(0.008..=0.06).contains(&slope) {
        return out;
    }
    // the outlet downslope of the site, the lines upslope from it
    let down = -g / slope;
    let surf = |q: DVec3| crate::kits::transport::route::surface(w, q);
    let at = |o: DVec3, v: DVec2, l: f64| surf(o + (east * v.x + north * v.y) * l);
    let outlet = at(p, down, 1_500.0 * (u01k(id, 2) - 0.2));
    let n = 1 + (u01k(id, 3) * 2.99) as usize;
    let (s0, s1) = seed_v(id);
    for i in 0..n {
        let hi = mix64(id ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9));
        let ang = (i as f64 - 0.5 * (n as f64 - 1.0)) * 0.3 + 0.15 * (u01k(hi, 1) - 0.5);
        let (sa, ca) = ang.sin_cos();
        let up = DVec2::new(-down.x * ca + down.y * sa, -down.x * sa - down.y * ca);
        let len = 3_000.0 + 5_000.0 * u01k(hi, 2);
        let side = DVec2::new(-up.y, up.x) * (0.06 * len * (u01k(hi, 3) - 0.5));
        let mid = surf(outlet + (east * (up.x * 0.5 * len + side.x) + north * (up.y * 0.5 * len + side.y)));
        let end = at(outlet, up, len);
        let spacing = 30.0 + 25.0 * u01k(hi, 4);
        let r = 4.0 + 2.5 * u01k(hi, 5);
        let mut s_acc = 0.0;
        for (a0, b0) in [(outlet, mid), (mid, end)] {
            out.push(crate::features::FSeg {
                a: a0,
                b: b0,
                ha: 0.0,
                hb: 0.0,
                kind: KIND_QANAT,
                class: crate::landcover::BARE,
                hw: (r + 1.0) as f32,
                s0: s_acc as f32,
                deck_a: f32::NAN,
                deck_b: f32::NAN,
                flags: 0,
                v: [spacing as f32, r as f32, s0, s1],
            });
            s_acc += (b0 - a0).length();
        }
    }
    out
}

/// The kit's host features: the qanats whose lines can reach the area.
fn qanat_host(ctx: &mut crate::features::HostCtx, area: &crate::features::Area) -> crate::features::Features {
    let mut f = crate::features::Features::default();
    let w = ctx.world;
    if area.gsd >= QANAT_GSD_MAX {
        return f;
    }
    // (a line reaches ≤ 1.5 km + 8 km + its bend from its cell's site)
    for c in crate::kits::settlements::sites::lattice_cells(w, QANAT_CELL, area.center, area.radius + 10_500.0) {
        for sg in qanat_group(w, c) {
            if crate::kits::transport::net::chord_dist(sg.a, sg.b, area.center) <= area.radius + sg.hw as f64 + 2.0 {
                f.segs.push(sg);
            }
        }
    }
    f
}

/// Slot 6: the shaft mounds of the qanats (explicit rings of spoil around a dark shaft where
/// they are resolved, their mean along the line where not).
pub fn qanats(s: &mut Stack) {
    let gsd = s.ctx.gsd;
    if gsd >= QANAT_GSD_MAX || s.t.sand > 0.5 || s.l.slope > 0.25 {
        return;
    }
    let p = s.ctx.p;
    let fw = s.l.fw.max(0.5 * gsd);
    let spoil = mixc(s.pal(pal::SAND + 2), s.pal(pal::SOIL + 1), 0.4) * 1.22;
    let hole = srgb(46.0, 38.0, 30.0);
    let mut best: Option<(f64, DVec3, f64)> = None;
    for sg in s.segments() {
        if sg.kind != KIND_QANAT {
            continue;
        }
        let (d, along, _) = crate::features::seg_frame(sg, p);
        let (sp, r) = (sg.v[0] as f64, sg.v[1] as f64);
        let hw_l = r.max(0.5 * gsd);
        if d.abs() > hw_l + fw {
            continue;
        }
        let k = (along / sp).round();
        let u = along - k * sp;
        let rr = (d * d + u * u).sqrt();
        let e = smoothstep(1.2 * gsd, 3.0 * gsd, 2.0 * r);
        // explicit: the ring of spoil, the shaft
        let ring = crate::surface::band_cov(rr, r, fw);
        let shaft = crate::surface::band_cov(rr, 0.9, fw) * band(1.8, gsd);
        let col_e = mixc(spoil, hole, shaft / ring.max(1e-6));
        // mean: the mounds' area per metre of line over the line's width
        let m = (PI * r * r / sp) / (2.0 * hw_l);
        let cov_m = crate::surface::band_cov(d, hw_l, fw) * m.min(1.0);
        let col_m = mixc(spoil, hole, 0.9 * 0.9 / (r * r));
        let cov = cov_m + (ring - cov_m) * e;
        if cov > 0.0 && best.as_ref().is_none_or(|b| cov > b.0) {
            let dh = 1.4 * (1.0 - (rr / r).min(1.0).powi(2)) - 2.0 * shaft;
            best = Some((cov, mixc(col_m, col_e, e), dh * e));
        }
    }
    if let Some((cov, col, dh)) = best {
        s.composite(Layer { cov, albedo: col, dh, hmode: hmode::BLEND, cls: crate::landcover::BARE, clear: 0.6, ..Default::default() });
    }
}

#[cfg(test)]
mod tests {
    /// Debug: qanat groups of the default world (their outlets): `find_qanats -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn find_qanats() {
        let w = World::new(crate::Config::default());
        let mut n = 0;
        for i in 0..400_000u64 {
            let h = mix64(0x0A4A7 ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = (u01k(h, 2) - 0.5) * TAU;
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            let c = (p / QANAT_CELL).floor();
            let segs = qanat_group(&w, (c.x as i64, c.y as i64, c.z as i64));
            if let Some(sg) = segs.first() {
                let g = geodesy::ecef2geodetic(sg.a, &w.ell);
                println!("qanats: {} lines, outlet ({:.4}, {:.4})", segs.len() / 2, g.lat.to_degrees(), g.lon.to_degrees());
                n += 1;
                if n >= 12 {
                    break;
                }
            }
        }
    }

    use super::*;

    #[test]
    fn class_ids() {
        use crate::registry::classes::id;
        assert_eq!(id("sand"), Some(LC_SAND));
        assert_eq!(id("rock"), Some(LC_ROCK));
        assert_eq!(id("dry_riverbed"), Some(LC_DRY_RIVERBED));
        assert_eq!(id("salt_flat"), Some(LC_SALT_FLAT));
        assert_eq!(id("volcanic_ash"), Some(LC_VOLCANIC_ASH));
        assert_eq!(id("gravel"), Some(LC_GRAVEL));
    }

    #[test]
    fn registry_compiles() {
        let r = crate::registry::Registry::builtin().unwrap();
        for b in ["hot_desert_erg", "desert_red"] {
            assert!(r.index(b).is_some(), "{b}");
        }
    }

    /// A random orthonormal frame (ECEF-like) from a hash.
    fn frame(h: u64) -> (DVec3, DVec3, DVec3) {
        let up = DVec3::new(u01k(h, 1) - 0.5, u01k(h, 2) - 0.5, u01k(h, 3) - 0.5).normalize();
        let t = up.cross(DVec3::new(u01k(h, 4) - 0.5, u01k(h, 5) - 0.5, u01k(h, 6) - 0.5)).normalize();
        (up, t, up.cross(t))
    }

    /// The ridge fields' means (what a pixel shows where they are unresolved).
    #[test]
    fn ridge_means() {
        let cases = [("transverse", TRANS_EXT, 0u32, 0.14, TRANS_MEAN), ("seif", SEIF_EXT, 1, 0.5, SEIF_MEAN), ("yardang", YARD_EXT, 2, 0.5, YARD_MEAN)];
        let mut bad = Vec::new();
        for (name, ext, kind, lee, mean) in cases {
            let n = 40_000;
            let mut sum = 0.0;
            for i in 0..n {
                let h = mix64(0x3EA2 ^ i as u64);
                let (up, ew, ec) = frame(h);
                let q = DVec3::new(u01k(h, 7), u01k(h, 8), u01k(h, 9)) * 1000.0 - 500.0;
                sum += ridges(0x51, q, ew, ec, up, 2.0, ext, kind, lee, 0.2);
            }
            let m = sum / n as f64;
            eprintln!("{name}: mean {m:.4} (constant {mean})");
            bad.push((name, m, mean));
        }
        for (name, m, mean) in bad {
            assert!((m - mean).abs() < 0.02 * mean, "{name}: mean {m:.4}, constant {mean}");
        }
    }

    /// The shape constants of the instances: star dune, inselberg and barchan volumes and areas.
    #[test]
    fn shape_constants() {
        // star dunes: ∫ (1 − x)^1.5 dA / (H R²) = B(2, 2.5) ∫ (0.42 + 0.58 arm²)² dθ
        let (mut acc, mut cnt) = (0.0, 0.0);
        for i in 0..400 {
            let h = mix64(0x57A ^ i);
            let n = 3.0 + (3.0 * u01k(h, 8)).floor();
            let (p1, p2) = (TAU * u01k(h, 9), TAU * u01k(h, 10));
            let m = 720;
            let mut s = 0.0;
            for j in 0..m {
                let th = -PI + TAU * (j as f64 + 0.5) / m as f64;
                let arm = 0.5 + 0.5 * (n * (th - p1) + 0.7 * (th + p2).sin()).cos();
                let f = 0.42 + 0.58 * arm * arm;
                s += f * f * TAU / m as f64;
            }
            acc += s * 0.114_286;
            cnt += 1.0;
        }
        let star = acc / cnt;
        // inselbergs: ∫ (dome + pediment) u du · ∫ edge² dθ
        let mut radial = 0.0;
        let m = 16_000;
        for j in 0..m {
            let x = 1.6 * (j as f64 + 0.5) / m as f64;
            radial += ins_profile(x) * x * 1.6 / m as f64;
        }
        let ins = radial * TAU * (1.0 + 0.5 * (0.10f64.powi(2) + 0.06f64.powi(2) + 0.035f64.powi(2)));
        // barchans: area / R² and volume / (H R²) of the crescent
        let (mut area, mut vol) = (0.0, 0.0);
        let g = 800;
        let dx = 2.4 / g as f64;
        for jy in 0..g {
            for jx in 0..g {
                let u = -1.2 + (jx as f64 + 0.5) * dx;
                let s = -1.2 + (jy as f64 + 0.5) * dx;
                let so = 1.0 - (u * u + s * s).sqrt();
                let si = ((u - 0.55) * (u - 0.55) + s * s).sqrt() - 0.85;
                if so <= 0.0 || si <= 0.0 {
                    continue;
                }
                let a1 = (so / 0.7).clamp(0.0, 1.0);
                let a2 = (si / 0.3).clamp(0.0, 1.0);
                let taper = (1.0 - s * s).max(0.0).powf(1.5);
                area += dx * dx;
                vol += dx * dx * (a1 * a1 * (3.0 - 2.0 * a1)).min(a2 * a2 * (3.0 - 2.0 * a2)) * taper;
            }
        }
        eprintln!("star CV {star:.4} ({STAR_CV}), inselberg CV {ins:.4} ({INS_CV}), barchan CA {area:.4} ({BAR_CA}) CV {vol:.4} ({BAR_CV})");
        for (name, v, c) in [("star", star, STAR_CV), ("inselberg", ins, INS_CV), ("barchan area", area, BAR_CA), ("barchan volume", vol, BAR_CV)] {
            assert!((v - c).abs() < 0.02 * c, "{name}: {v:.4}, constant {c}");
        }
    }

    /// Places of the default world where the kit's landforms are (for stills):
    /// `cargo test --release -p terragen desert_places -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn desert_places() {
        use rayon::prelude::*;
        let w = World::new(crate::Config::default());
        let seed = w.seed;
        let names = ["transverse", "linear", "star", "barchans", "playa", "inselberg", "yardangs", "wash", "dry lake"];
        let hits: Vec<(usize, String)> = (0..60_000u64)
            .into_par_iter()
            .flat_map_iter(|i| {
                let h = mix64(0xD1CE ^ i);
                let lat = (2.0 * u01k(h, 1) - 1.0).asin();
                let lon = TAU * u01k(h, 2) - PI;
                let c = Ctx::new(lat, lon, 30.0, &w.ell);
                let t = w.terrain(&c);
                let mut out = Vec::new();
                if t.moist > 0.45 {
                    return out;
                }
                let (_, rho) = wind(seed, c.p, c.lat, DVec2::from_array(t.wind));
                let (wt, wl, ws) = regime(rho);
                let at = format!(
                    "{:.4},{:.4} (h {:.0} m, T {:.0}, moist {:.2}, sand {:.2}, wind {:.1} m/s, rho {rho:.2})",
                    lat.to_degrees(),
                    lon.to_degrees(),
                    t.ground,
                    t.temp,
                    t.moist,
                    t.sand,
                    DVec2::from_array(t.wind).length()
                );
                if t.water_kind == crate::world::water::LAKE && aridity(t.moist) > 0.5 {
                    out.push((8, at.clone()));
                }
                if t.water_kind != 0 || t.ground < 1.0 {
                    return out;
                }
                if t.sand > 0.7 {
                    if wt > 0.9 {
                        out.push((0, at.clone()));
                    } else if wl > 0.9 {
                        out.push((1, at.clone()));
                    } else if ws > 0.9 {
                        out.push((2, at.clone()));
                    }
                }
                if (0.12..0.4).contains(&t.sand) && rho < 0.35 {
                    out.push((3, at.clone()));
                }
                if playa(&crate::instances::near(&w, fam_playas(), c.p, 0.0), c.p, c.east, c.north).is_some_and(|p| p.0 < 0.7) {
                    out.push((4, at.clone()));
                }
                if inselbergs(&crate::instances::near(&w, fam_inselbergs(), c.p, 0.0), &c) > 120.0 {
                    out.push((5, at.clone()));
                }
                let hyper = (1.0 - smoothstep(0.05, 0.12, t.moist)) * smoothstep(-2.0, 2.0, t.temp);
                if hyper > 0.5 && t.sand < 0.02 && perlin3(seed ^ S_YZONE, c.p / 120_000.0) > 0.35 {
                    out.push((6, at.clone()));
                }
                if t.river_hw > 20.0 && t.river_wet < 0.5 && t.river_d.abs() < t.river_hw {
                    out.push((7, at.clone()));
                }
                out
            })
            .collect();
        for (k, name) in names.iter().enumerate() {
            let v: Vec<&String> = hits.iter().filter(|h| h.0 == k).map(|h| &h.1).collect();
            eprintln!("{name}: {} hits", v.len());
            for s in v.iter().take(6) {
                eprintln!("    {s}");
            }
        }
    }

    /// Places of the default world where the kit draws (from `desert_places`): transverse,
    /// linear and star dunes, barchans, a playa, an inselberg, a dry lake, a wash.
    const PLACES: [(f64, f64); 9] = [
        (-19.9225, 40.0243),
        (23.5356, 43.2255),
        (-31.1491, -166.3463),
        (20.8406, -50.0567),
        (-11.8581, -162.1509),
        (22.2101, 127.5214),
        (-12.4109, -155.0993),
        (-16.1256, 53.6542),
        // qanats
        (27.0493, 34.4005),
    ];

    fn tile_at(i: usize, z: u8) -> geodesy::tiles::TileId {
        let (lat, lon) = PLACES[i];
        geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)
    }

    fn gen1() -> crate::NextGenerator {
        crate::NextGenerator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() })
    }

    /// Parent ≈ mean of children where the kit draws: elevation, albedo and the share of the
    /// kit's classes.
    #[test]
    fn lod_consistency() {
        use rayon::prelude::*;
        let g = gen1();
        let n = crate::TILE_SIZE;
        let res: Vec<(usize, f64, f64, f64)> = (0..PLACES.len())
            .into_par_iter()
            .map(|i| {
                let parent = tile_at(i, 12);
                let p = g.tile(parent);
                let kids = parent.children().map(|c| g.tile(c));
                let (mut eh, mut ea, mut cnt) = (0.0, 0.0, 0.0);
                let (mut cp, mut ck) = (0.0, 0.0);
                let kit = |c: u8| [LC_SAND, LC_GRAVEL, LC_SALT_FLAT, LC_DRY_RIVERBED, LC_VOLCANIC_ASH].contains(&c) as u8 as f64;
                for (k, kid) in kids.iter().enumerate() {
                    let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
                    for j in (0..n).step_by(2) {
                        for i in (0..n).step_by(2) {
                            let ix = [j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1];
                            let m = ix.iter().map(|&o| kid.elevation[o] as f64).sum::<f64>() / 4.0;
                            let pi = (oy + j / 2) * n + ox + i / 2;
                            eh += (m - p.elevation[pi] as f64).abs();
                            for c in 0..3 {
                                let ma = ix.iter().map(|&o| kid.albedo[3 * o + c] as f64).sum::<f64>() / 4.0;
                                ea += (ma - p.albedo[3 * pi + c] as f64).abs() / 3.0;
                            }
                            ck += ix.iter().map(|&o| kit(kid.landcover[o])).sum::<f64>() / 4.0;
                            cp += kit(p.landcover[pi]);
                            cnt += 1.0;
                        }
                    }
                }
                (i, eh / cnt, ea / cnt, (cp - ck).abs() / cnt)
            })
            .collect();
        for (i, eh, ea, dc) in res {
            eprintln!("place {i}: |parent - mean(children)| elevation {eh:.2} m, albedo {ea:.2} DN, kit classes {:.2}%", dc * 100.0);
            assert!(eh < 6.0, "place {i}: elevation {eh}");
            assert!(ea < 6.0, "place {i}: albedo {ea}");
            assert!(dc < 0.05, "place {i}: class share {dc}");
        }
    }

    /// No steps along tile borders where the kit draws, and the same tile twice.
    #[test]
    fn seamless_and_deterministic() {
        use rayon::prelude::*;
        let g = gen1();
        // (one second generator for the determinism check, not one per place: nine generators
        // at once on the GPU ran an 8 GB card out of memory under the parallel test load)
        let g2 = gen1();
        let n = crate::TILE_SIZE;
        (0..PLACES.len()).into_par_iter().for_each(|i| {
            for z in [11u8, 15] {
                let id = tile_at(i, z);
                let a = g.tile(id);
                let b = g.tile(geodesy::tiles::TileId::new(z, id.x + 1, id.y));
                let (mut across, mut inside) = (0.0, 0.0);
                for j in 0..n {
                    across += (a.elevation[j * n + n - 1] - b.elevation[j * n]).abs() as f64;
                    inside += (a.elevation[j * n + n - 2] - a.elevation[j * n + n - 1]).abs() as f64;
                }
                assert!(across < 3.0 * inside + 1.0 * n as f64, "place {i} z{z}: across {across} inside {inside}");
                if z == 15 {
                    let a2 = g2.tile(id);
                    assert_eq!(a.elevation, a2.elevation, "place {i}");
                    assert_eq!(a.rgb, a2.rgb, "place {i}");
                }
            }
        });
    }

    /// GPU tiles agree with the CPU's where the kit draws (≤ 0.5 % of the pixels off: rgb by
    /// more than 2 DN, elevation by more than 5 cm, land cover).
    #[cfg(feature = "gpu")]
    #[test]
    fn tiles_match_the_cpu() {
        let cfg = crate::Config::default();
        let Ok(gpu) = crate::gpu::GpuGenerator::new(cfg.clone()) else {
            eprintln!("no GPU: skipped");
            return;
        };
        let cpu = crate::NextGenerator::new(cfg);
        let mut ids = Vec::new();
        for i in 0..PLACES.len() {
            for z in [10u8, 13, 16] {
                ids.push(tile_at(i, z));
            }
        }
        let gt = gpu.tiles(&ids).unwrap();
        use rayon::prelude::*;
        let ct: Vec<_> = ids.par_iter().map(|&id| cpu.tile_cpu(id)).collect();
        let mut worst: f64 = 0.0;
        for ((id, g), c) in ids.iter().zip(&gt).zip(&ct) {
            let off = |a: &[u8], b: &[u8], tol: i32| a.iter().zip(b).filter(|(x, y)| (**x as i32 - **y as i32).abs() > tol).count() as f64 / a.len() as f64;
            let e = g.elevation.iter().zip(&c.elevation).filter(|(x, y)| (**x - **y).abs() > 0.05).count() as f64 / g.elevation.len() as f64;
            let (r, l) = (off(&g.rgb, &c.rgb, 2), off(&g.landcover, &c.landcover, 0));
            eprintln!("{id}: off elevation {:.3}% rgb {:.3}% landcover {:.3}%", e * 100.0, r * 100.0, l * 100.0);
            worst = worst.max(e).max(r).max(l);
        }
        assert!(worst <= 0.005, "worst share of pixels off: {:.3}%", worst * 100.0);
    }

    /// Nadir previews of the kit's places (rgb and a hillshade of the DSM) at z11, z13, z15 on
    /// the default backend: `DESERT_PNG=dir cargo test --release -p terragen desert_previews --
    /// --ignored` (`DESERT_AT="lat,lon;lat,lon"` for other places).
    #[test]
    #[ignore]
    fn desert_previews() {
        let Ok(dir) = std::env::var("DESERT_PNG") else { return };
        std::fs::create_dir_all(&dir).unwrap();
        let places: Vec<(f64, f64)> = match std::env::var("DESERT_AT") {
            Ok(v) => v
                .split(';')
                .map(|p| {
                    let x: Vec<f64> = p.split(',').map(|c| c.trim().parse().unwrap()).collect();
                    (x[0], x[1])
                })
                .collect(),
            Err(_) => PLACES.to_vec(),
        };
        let zooms: Vec<u8> = std::env::var("DESERT_Z").map(|v| v.split(',').map(|z| z.parse().unwrap()).collect()).unwrap_or(vec![11, 13, 15]);
        let be = if std::env::var("DESERT_CPU").is_ok() { crate::Backend::Cpu } else { crate::Backend::Auto };
        let g = crate::NextGenerator::try_with_backend(crate::Config::default(), be).unwrap();
        let n = crate::TILE_SIZE;
        for (k, &(lat, lon)) in places.iter().enumerate() {
            let ids: Vec<_> = zooms.iter().map(|&z| geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)).collect();
            let tiles = g.tiles(&ids).unwrap();
            let mut img = image::RgbImage::new((n * ids.len()) as u32, (2 * n) as u32);
            for (j, (id, t)) in ids.iter().zip(&tiles).enumerate() {
                let gsd = geodesy::tiles::gsd_ew(lat.to_radians(), id.z, n as u32, &g.world.ell);
                for y in 0..n {
                    for x in 0..n {
                        let o = y * n + x;
                        img.put_pixel((j * n + x) as u32, y as u32, image::Rgb([t.rgb[3 * o], t.rgb[3 * o + 1], t.rgb[3 * o + 2]]));
                        let h = |xx: usize, yy: usize| t.elevation[yy.min(n - 1) * n + xx.min(n - 1)] as f64;
                        let dx = (h(x + 1, y) - h(x.saturating_sub(1), y)) / (2.0 * gsd);
                        let dy = (h(x, y + 1) - h(x, y.saturating_sub(1))) / (2.0 * gsd);
                        let nv = DVec3::new(-dx, dy, 1.0).normalize();
                        let sh = (nv.dot(DVec3::new(-0.5, 0.5, 0.7).normalize()) * 230.0).clamp(0.0, 255.0) as u8;
                        img.put_pixel((j * n + x) as u32, (n + y) as u32, image::Rgb([sh, sh, sh]));
                    }
                }
            }
            img.save(format!("{dir}/p{k}_{lat:.3}_{lon:.3}.png")).unwrap();
        }
    }

    /// Debug: the inselbergs and playas near `DESERT_AT` and the ground along a transect.
    #[test]
    #[ignore]
    fn desert_probe() {
        let Ok(at) = std::env::var("DESERT_AT") else { return };
        let x: Vec<f64> = at.split(',').map(|c| c.trim().parse().unwrap()).collect();
        let w = World::new(crate::Config::default());
        let c = Ctx::new(x[0].to_radians(), x[1].to_radians(), 30.0, &w.ell);
        for (name, f) in [("inselberg", fam_inselbergs()), ("playa", fam_playas())] {
            for i in crate::instances::near(&w, f, c.p, 5000.0) {
                let g = geodesy::ecef2geodetic(i.center, &w.ell);
                let v = offset(c.p, i.center, c.east, c.north);
                let sh = if name == "inselberg" {
                    ins_shape(i.id, v)
                } else {
                    let (x, r) = playa_shape(i.id, v);
                    (x, r, 0.0)
                };
                eprintln!("{name} at {:.4},{:.4}: x {:.2} r {:.0} h {:.0}", g.lat.to_degrees(), g.lon.to_degrees(), sh.0, sh.1, sh.2);
                for k in -10..=10 {
                    let p = i.center + c.east * (k as f64 * 0.15 * sh.1);
                    let g = geodesy::ecef2geodetic(p, &w.ell);
                    let t = w.terrain(&Ctx::new(g.lat, g.lon, 30.0, &w.ell));
                    eprint!("{:.0} ", t.ground);
                }
                eprintln!();
            }
        }
    }
}
