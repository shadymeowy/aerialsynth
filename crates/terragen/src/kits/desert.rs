//! The desert kit (`docs/design/terrain-next.md` §5.1, `docs/kits/desert.md`): sand seas whose
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
    layers: &[(slot::AZONAL, azonal), (slot::WATER, washes)],
    relief: Some(relief),
    relief_wgsl: include_str!("../gpu/wgsl/kits/desert_relief.wgsl"),
    ..Kit::EMPTY
};

// ============================================================================= constants

// lattice / noise keys (xor the world seed)
const S_WIND1: u64 = 0xDE5E_0001;
const S_WIND2: u64 = 0xDE5E_0002;
const S_REG1: u64 = 0xDE5E_0003;
const S_REG2: u64 = 0xDE5E_0004;
const S_TRANS: u64 = 0xDE5E_0010;
const S_SEIF: u64 = 0xDE5E_0011;
const S_SMALL: u64 = 0xDE5E_0012;
const S_YARD: u64 = 0xDE5E_0013;
const S_YZONE: u64 = 0xDE5E_0014;
const S_STAR: u64 = 0xDE5E_0020;
const S_INS: u64 = 0xDE5E_0021;
const S_PLAYA: u64 = 0xDE5E_0022;
const S_BAR: u64 = 0xDE5E_0023;
const S_CRACK: u64 = 0xDE5E_0030;
const S_BRINE: u64 = 0xDE5E_0031;
const S_VARN: u64 = 0xDE5E_0032;
const S_STREAK: u64 = 0xDE5E_0033;
const S_SANDTEX: u64 = 0xDE5E_0034;
const S_WASH: u64 = 0xDE5E_0040;
const S_BRAID: u64 = 0xDE5E_0041;
const S_BRAID2: u64 = 0xDE5E_0042;

// land-cover classes (v2 ids; `tests::class_ids` checks them against the registry)
const LC_SAND: u8 = 5;
const LC_DRY_RIVERBED: u8 = 29;
const LC_SALT_FLAT: u8 = 30;
const LC_VOLCANIC_ASH: u8 = 32;
const LC_GRAVEL: u8 = 33;

// dune fields: wavelengths (m), waves per lattice cell, kernel extents (crest, wave, up; cells)
const TRANS_LAM: f64 = 600.0;
const SEIF_LAM: f64 = 1700.0;
const SMALL_LAM: f64 = 240.0;
const YARD_LAM: f64 = 180.0;
const TRANS_EXT: [f64; 3] = [1.2, 0.7, 0.7];
const SEIF_EXT: [f64; 3] = [1.5, 0.55, 0.55];
const YARD_EXT: [f64; 3] = [1.5, 0.45, 0.45];
// the means of the ridge fields (`tests::ridge_means`): what a pixel shows where unresolved
const TRANS_MEAN: f64 = 0.3711;
const SEIF_MEAN: f64 = 0.1685;
const YARD_MEAN: f64 = 0.1409;

// star dunes: cell (m), density, radius range (m), the mean volume per instance / (H R²)
const STAR_CELL: f64 = 2200.0;
const STAR_DENS: f64 = 0.8;
const STAR_R: [f64; 2] = [500.0, 950.0];
const STAR_CV: f64 = 0.2383;
const ARM_MEAN: f64 = 0.3125;
// inselbergs: cell (m), density, the volume of the profile / (H R²)
const INS_CELL: f64 = 16000.0;
const INS_DENS: f64 = 0.35;
const INS_CV: f64 = 2.4046;
// playas: cell (m), density, the largest hill amplitude (macro field) at the site
const PLAYA_CELL: f64 = 36000.0;
const PLAYA_DENS: f64 = 0.6;
const PLAYA_FLAT: f64 = -0.25;
// barchans: cell (m), density, radius range (m), area and volume of the shape / R², (H R²)
const BAR_CELL: f64 = 380.0;
const BAR_DENS: f64 = 0.8;
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
/// variability 0 (one direction) .. 1 (many). A proxy of the atlas wind channels: trade winds
/// (to the south-west in the north, the north-west in the south) and westerlies, turned by
/// continental-scale noise.
pub fn wind(seed: u64, p: DVec3, lat: f64) -> (DVec2, f64) {
    let latd = lat.abs().to_degrees();
    let ns = if lat >= 0.0 { 1.0 } else { -1.0 };
    let trade = 1.0 - smoothstep(25.0, 35.0, latd);
    let base = DVec2::new(-0.7071, -0.7071 * ns) * trade + DVec2::new(0.966, 0.259 * ns) * (1.0 - trade);
    let th = base.y.atan2(base.x) + 1.1 * perlin3(seed ^ S_WIND1, p / 900_000.0) + 0.45 * perlin3(seed ^ S_WIND2, p / 260_000.0);
    let rho = saturate(0.5 + 0.9 * perlin3(seed ^ S_REG1, p / 800_000.0) + 0.3 * perlin3(seed ^ S_REG2, p / 230_000.0));
    (DVec2::new(th.cos(), th.sin()), rho)
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
pub fn ridges(seed: u64, q: DVec3, ew: DVec3, ec: DVec3, up: DVec3, k: f64, ext: [f64; 3], kind: u32, lee: f64) -> f64 {
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
                let x = k * b + u01k(h, 4);
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
    let mean = amount * STAR_DENS * STAR_CV * 0.185 * er3 / (STAR_CELL * STAR_CELL);
    let e = smoothstep(1.2 * gsd, 3.0 * gsd, STAR_R[0]);
    if e <= 0.0 {
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
        let hh = r * (0.12 + 0.13 * u01k(h, 7));
        let v = -st.rel[i];
        let d = v.length();
        if d >= r {
            continue;
        }
        let n = 3.0 + (3.0 * u01k(h, 8)).floor();
        let th = v.y.atan2(v.x);
        let arm = 0.5 + 0.5 * (n * (th - TAU * u01k(h, 9)) + 0.7 * (th + TAU * u01k(h, 10)).sin()).cos();
        let a3 = ARM_MEAN + (arm * arm * arm - ARM_MEAN) * band(TAU * d / (3.0 * n), gsd);
        let x = d / (r * (0.3 + 0.7 * a3));
        if x < 1.0 {
            best = best.max(hh * (1.0 - x).powf(1.5));
        }
    }
    e * best + (1.0 - e) * mean
}

/// Inselbergs (m): steep granite domes with a pediment, sized by the mask `m`; widened with
/// their volume kept where below the pixel.
fn inselbergs(w: &World, seed: u64, c: &Ctx, m: f64) -> f64 {
    let gsd = c.gsd;
    let reach = 0.45 * INS_CELL;
    let st = slab_sites(w, seed ^ S_INS, c.p, c.east, c.north, INS_CELL, reach);
    let mut best: f64 = 0.0;
    for i in 0..st.n {
        let h = st.id[i];
        if u01k(h, 5) >= INS_DENS {
            continue;
        }
        let u6 = u01k(h, 6);
        let r = (500.0 + 2100.0 * u6 * u6) * m;
        if r < 1.0 {
            continue;
        }
        let hh = (r * (0.12 + 0.22 * u01k(h, 7))).min(450.0);
        let ex = 1.0 + 0.7 * u01k(h, 8);
        let (sa, ca) = (TAU * u01k(h, 9)).sin_cos();
        let v = -st.rel[i];
        let (vx, vy) = (v.x * ca + v.y * sa, -v.x * sa + v.y * ca);
        let dn = (vx * vx / ex + vy * vy * ex).sqrt();
        let th = vy.atan2(vx);
        let edge = 1.0 + 0.10 * (3.0 * th + TAU * u01k(h, 10)).sin() + 0.06 * (5.0 * th + TAU * u01k(h, 11)).sin() + 0.035 * (9.0 * th + TAU * u01k(h, 12)).sin();
        let x = dn / (r * edge);
        let e = smoothstep(1.2 * gsd, 3.0 * gsd, r);
        let mut hx = 0.0;
        if e > 0.0 && x < 1.6 {
            let dome = if x < 1.0 { (1.0 - x.powf(2.4)).max(0.0).powf(0.55) } else { 0.0 };
            let ped = 0.05 * (1.0 - smoothstep(0.9, 1.6, x));
            hx = hh * (dome + ped);
        }
        let rb = (0.95 * reach).min((1.6 * r).max(3.0 * gsd));
        let hp = 2.0 * hh * r * r * INS_CV / (PI * rb * rb);
        let db = v.length() / rb;
        let blob = if db < 1.0 { hp * (1.0 - db * db) } else { 0.0 };
        best = best.max(e * hx + (1.0 - e) * blob);
    }
    best
}

/// The size mask of playas (0..1) from pass-A values both passes know: arid, not in sand
/// seas, mountains or mesas.
#[inline]
fn playa_mask(moist: f64, sand: f64, mountain: f64, mesa: f64) -> f64 {
    aridity(moist) * (1.0 - 0.85 * sand) * (1.0 - mountain) * (1.0 - smoothstep(0.2, 0.5, mesa))
}

/// The playa at `p`, if any: (normalized distance from its centre with a ragged shore, its
/// radius (m), its hash). Playas exist at flat sites of their lattice (the macro hill amplitude
/// at the site) and shrink with the mask `m` (the shore moves, it never fades).
fn playa(w: &World, seed: u64, p: DVec3, east: DVec3, north: DVec3, m: f64) -> Option<(f64, f64, u64)> {
    if m <= 0.0 {
        return None;
    }
    let st = slab_sites(w, seed ^ S_PLAYA, p, east, north, PLAYA_CELL, 0.45 * PLAYA_CELL);
    let mut best: Option<(f64, f64, u64)> = None;
    for i in 0..st.n {
        let h = st.id[i];
        if u01k(h, 5) >= PLAYA_DENS {
            continue;
        }
        let r = (2000.0 + 5500.0 * u01k(h, 6)) * m;
        let ex = 1.0 + 0.8 * u01k(h, 7);
        let (sa, ca) = (TAU * u01k(h, 8)).sin_cos();
        let v = -st.rel[i];
        let (vx, vy) = (v.x * ca + v.y * sa, -v.x * sa + v.y * ca);
        let dn = (vx * vx / ex + vy * vy * ex).sqrt();
        if dn > 1.4 * 1.27 * r {
            continue;
        }
        let ha = w.hill_amp.eval(st.c[i], 20_000.0) * w.hill_amp.norm() * 1.6;
        if ha > PLAYA_FLAT {
            continue;
        }
        let th = vy.atan2(vx);
        let edge = 1.0 + 0.14 * (2.0 * th + TAU * u01k(h, 9)).sin() + 0.08 * (3.0 * th + TAU * u01k(h, 10)).sin() + 0.04 * (7.0 * th + TAU * u01k(h, 11)).sin();
        let x = dn / (r * edge).max(1e-3);
        if best.is_none_or(|b| x < b.0) {
            best = Some((x, r, h));
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
    let semi = 1.0 - smoothstep(0.30, 0.45, r.moist);
    if semi <= 0.0 && r.sand <= 0.0 {
        return;
    }
    let dh = w.cfg.relief.dune_height_m;
    // ---- dunes: the core's transverse ridges out, the types of the wind regime in
    if r.sand > 1e-3 && dh > 0.0 {
        let core = dh * r.sand * w.dunes(c, r.m, gsd);
        let (wd, rho) = wind(seed, c.p, c.lat);
        let (wt, wl, ws) = regime(rho);
        let env = smoothstep(0.35, 0.75, r.sand);
        let (ew, ec) = wind_axes(c, wd);
        let mut sum = 0.0;
        if env > 0.0 {
            if wt > 0.0 {
                let b = band(TRANS_LAM, gsd);
                let mut v = TRANS_MEAN;
                if b > 0.0 {
                    let lee = (2.5 * gsd / TRANS_LAM).clamp(0.14, 0.5);
                    v += (ridges(seed ^ S_TRANS, c.p / (2.0 * TRANS_LAM), ew, ec, c.up, 2.0, TRANS_EXT, 0, lee) - TRANS_MEAN) * b;
                }
                sum += wt * v;
            }
            if wl > 0.0 {
                let b = band(SEIF_LAM, gsd);
                let mut v = SEIF_MEAN;
                if b > 0.0 {
                    v += (ridges(seed ^ S_SEIF, c.p / (3.0 * SEIF_LAM), ec, ew, c.up, 3.0, SEIF_EXT, 1, 0.5) - SEIF_MEAN) * b;
                }
                sum += wl * 1.9 * v;
            }
            if wl + ws > 0.0 {
                let b = band(SMALL_LAM, gsd);
                let mut v = TRANS_MEAN;
                if b > 0.0 {
                    let lee = (2.5 * gsd / SMALL_LAM).clamp(0.14, 0.5);
                    v += (ridges(seed ^ S_SMALL, c.p / (2.0 * SMALL_LAM), ew, ec, c.up, 2.0, TRANS_EXT, 0, lee) - TRANS_MEAN) * b;
                }
                sum += (wl + ws) * 0.14 * v;
            }
        }
        let mut star = 0.0;
        if ws * env > 0.0 {
            star = stars(w, seed, c, ws * env) * (dh / 35.0);
        }
        *h += dh * r.sand * env * sum + star - core;
    }
    if semi <= 0.0 {
        return;
    }
    // ---- inselbergs
    let mi = semi * (1.0 - r.mountain) * (1.0 - 0.6 * r.sand) * (1.0 - smoothstep(0.2, 0.5, r.mesa));
    if mi > 0.0 {
        *h += inselbergs(w, seed, c, mi);
    }
    // ---- playa floors: flattened towards the smooth relief, a little below it
    let mp = playa_mask(r.moist, r.sand, r.mountain, r.mesa);
    if let Some((x, _, _)) = playa(w, seed, c.p, c.east, c.north, mp) {
        if x < 1.0 {
            let k = 1.0 - smoothstep(0.8, 1.0, x);
            *h += (r.smooth - 2.0 - *h) * k;
        }
    }
    // ---- yardangs: streamlined ridges along the wind on the hyper-arid plains
    let hyper = 1.0 - smoothstep(0.05, 0.12, r.moist);
    if hyper > 0.0 && r.sand < 0.2 {
        let zone = hyper * smoothstep(0.1, 0.3, perlin3(seed ^ S_YZONE, c.p / 120_000.0)) * (1.0 - smoothstep(0.02, 0.2, r.sand)) * (1.0 - r.mountain);
        if zone > 0.0 {
            let b = band(YARD_LAM, gsd);
            let mut v = YARD_MEAN;
            if b > 0.0 {
                let (wd, _) = wind(seed, c.p, c.lat);
                let (ew, ec) = wind_axes(c, wd);
                v += (ridges(seed ^ S_YARD, c.p / (2.0 * YARD_LAM), ec, ew, c.up, 2.0, YARD_EXT, 2, 0.5) - YARD_MEAN) * b;
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
    let mh = BAR_CV / BAR_CA * 0.21 * er3 / er2;
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
        let hh = r * (0.16 + 0.10 * u01k(h, 7));
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

/// Slot 3: playas, the sand seas' surface, their stony fringe and its barchans.
pub fn azonal(s: &mut Stack) {
    let t = *s.t;
    let ar = aridity(t.moist);
    if t.sand <= 0.0 && ar <= 0.0 {
        return;
    }
    let w = s.world;
    let seed = w.seed;
    let c = *s.ctx;
    let p = c.p;
    let gsd = c.gsd;
    let fw = s.l.fw;
    // ---- playas: salt crust with polygonal cracks, brine pools, paleo-shorelines
    let mp = playa_mask(t.moist, t.sand, t.mountain, t.mesa);
    if let Some((x, r, id)) = playa(w, seed, p, c.east, c.north, mp) {
        if x < 1.4 {
            let crust_w = DVec3::new(0.80, 0.78, 0.72) * (0.92 + 0.08 * u01k(id, 12));
            let tone = [DVec3::new(1.0, 1.0, 1.0), DVec3::new(1.0, 0.95, 0.86), DVec3::new(1.0, 0.93, 0.92)][(3.0 * u01k(id, 13)) as usize % 3];
            let mut col = crust_w * tone;
            let dm = (1.0 - x) * r;
            let cov = (dm / fw.max(1e-3) + 0.5).clamp(0.0, 1.0);
            if cov > 0.0 {
                // polygons 8–20 m: dark cracks, slightly different plates
                let ce = smoothstep(1.2 * gsd, 3.0 * gsd, 12.0);
                let mut crack = 0.06;
                let mut plate = 1.0;
                if ce > 0.0 {
                    let cl = crate::noise::worley2(seed ^ S_CRACK, s.q_loc, 12.0, 0.85);
                    let ed = worley2_edge_dist(&cl, s.q_loc);
                    crack = ce * crate::kernels::band_cov(ed, 0.35, fw) + (1.0 - ce) * 0.06;
                    plate = 1.0 + ce * 0.06 * (u01k(cl.id, 1) - 0.5);
                }
                col = col * plate * (1.0 - 0.35 * crack);
                // the wetter, darker floor in the middle, brine pools in some playas
                let mid = 1.0 - smoothstep(0.2, 0.6, x + 0.15 * perlin3(seed ^ S_BRINE, p / 900.0));
                col = mixc(col, col * DVec3::new(0.78, 0.76, 0.72), 0.5 * mid);
                if u01k(id, 14) < 0.5 {
                    let pn = perlin3(seed ^ S_BRINE ^ 7, p / 400.0) + 0.5 * perlin3(seed ^ S_BRINE ^ 9, p / 130.0) * band(130.0, gsd);
                    let pool = smoothstep(0.25, 0.32, pn + 0.6 * mid - 0.4) * mid;
                    let brine = if u01k(id, 15) < 0.5 { srgb(214.0, 150.0, 156.0) } else { srgb(120.0, 156.0, 120.0) };
                    col = mixc(col, brine, 0.85 * pool);
                }
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
                    s.composite(Layer::paint(0.45 * ring, mixc(s.col, crust_w, 0.7)));
                }
            }
        }
    }
    if t.sand <= 0.0 {
        return;
    }
    // ---- sand seas
    let (wd, rho) = wind(seed, p, c.lat);
    let (_, wl, _) = regime(rho);
    let sc = sand_colour(s);
    let black = s.bio.style.litho == 3;
    let patch = s.pf.patch;
    let erg = smoothstep(0.47, 0.53, t.sand + 0.1 * patch);
    let fringe = (1.0 - erg) * smoothstep(0.0, 0.08, t.sand);
    if fringe > 0.0 {
        // the stony pavement between the barchans: varnished gravel, sand sheets streaked
        // along the wind
        let soil = s.pal(pal::SOIL + 1);
        let mut pav = mixc(soil, s.pal(pal::ROCK), 0.3) * 0.82;
        let vn = perlin3(seed ^ S_VARN, p / 420.0) + 0.5 * perlin3(seed ^ S_VARN ^ 1, p / 130.0) * band(130.0, gsd);
        let ve = smoothstep(1.2 * gsd, 3.0 * gsd, 130.0);
        let varn = ve * smoothstep(0.2, 0.3, vn) + (1.0 - ve) * 0.3;
        pav = mixc(pav, pav * DVec3::new(0.55, 0.5, 0.48), 0.8 * varn);
        let sb = band(90.0, gsd);
        let mut streak = 0.35;
        if sb > 0.0 {
            let ec = c.up.cross(c.east * wd.x + c.north * wd.y);
            let (g, _) = World::gully_octave(seed ^ S_STREAK, p / 90.0, ec);
            streak = 0.35 + (smoothstep(0.1, 0.7, g) - 0.35) * sb;
        }
        pav = mixc(pav, sc, streak * (2.0 * t.sand).min(1.0));
        s.composite(Layer { cov: fringe, albedo: pav, cls: LC_GRAVEL, clear: 0.6, ..Default::default() });
    }
    if erg > 0.0 {
        let mut col = sc * (1.0 + 0.05 * perlin3(seed ^ S_SANDTEX, p / 35.0) * band(35.0, gsd) + 0.03 * perlin3(seed ^ S_SANDTEX ^ 1, p / 9.0) * band(9.0, gsd));
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
    // ---- barchans: on the sand-poor fringe, most where the wind is steady
    let zb = smoothstep(0.02, 0.1, t.sand) * (1.0 - smoothstep(0.42, 0.58, t.sand)) * (0.35 + 0.65 * (1.0 - smoothstep(0.35, 0.6, rho)));
    if zb > 0.0 {
        let (cov, dh) = barchans(w, seed, p, c.east, c.north, wd, zb, gsd, fw);
        if cov > 0.0 {
            let cls = if black { LC_VOLCANIC_ASH } else { LC_SAND };
            s.composite(Layer { cov, albedo: sc * 1.06, dh, hmode: hmode::ADD, cls, clear: 1.0, ..Default::default() });
        }
    }
}

/// Slot 8: dry washes (wadis) wider than the channel, with braided threads of gravel and sand
/// bars parallel to the banks that merge and split.
pub fn washes(s: &mut Stack) {
    if s.is_water {
        return;
    }
    let l = s.l;
    let t = s.t;
    if l.river_hw <= 0.0 || t.river_wet >= 0.5 {
        return;
    }
    let ar = 1.0 - smoothstep(0.25, 0.42, t.moist);
    if ar <= 0.0 {
        return;
    }
    let seed = s.world.seed;
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let hw = l.river_hw;
    let half = hw * (1.0 + 1.4 * ar) * (1.0 + 0.25 * perlin3(seed ^ S_WASH, p / 700.0));
    let fwr = l.fw.max(gsd * 0.35);
    let cov = crate::kernels::band_cov(l.river_d, half, fwr);
    if cov <= 0.0 {
        return;
    }
    let lam = (0.45 * half).max(4.0);
    let ph = l.river_d / lam + 0.9 * perlin3(seed ^ S_BRAID, p / 90.0) + 0.4 * perlin3(seed ^ S_BRAID2, p / 31.0) * band(31.0, gsd);
    let th = 0.5 + 0.5 * (TAU * ph).cos();
    let bw = band(lam, gsd);
    let bar = s.pal(pal::SAND + 2) * 1.05;
    let chan = mixc(s.pal(pal::SOIL + 1), s.pal(pal::ROCK), 0.4) * 0.85;
    let col = mixc(bar, chan, smoothstep(0.55, 0.8, th) * bw + (1.0 - bw) * 0.3);
    s.composite(Layer { cov, albedo: col, cls: LC_DRY_RIVERBED, clear: 0.8, ..Default::default() });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_ids() {
        use crate::registry::classes::id;
        assert_eq!(id("sand"), Some(LC_SAND));
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
                sum += ridges(0x51, q, ew, ec, up, 2.0, ext, kind, lee);
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
        // star dunes: ∫ (1 − x)^1.5 dA / (H R²) = B(2, 2.5) ∫ (0.3 + 0.7 arm³)² dθ
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
                let f = 0.3 + 0.7 * arm * arm * arm;
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
            let dome = if x < 1.0 { (1.0 - x.powf(2.4)).max(0.0).powf(0.55) } else { 0.0 };
            let ped = 0.05 * (1.0 - smoothstep(0.9, 1.6, x));
            radial += (dome + ped) * x * 1.6 / m as f64;
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
        let w = World::new(crate::Config::default());
        let seed = w.seed;
        let mut found: Vec<(&str, Vec<String>)> = ["transverse", "linear", "star", "barchans", "playa", "inselberg", "yardangs", "wash"].iter().map(|n| (*n, Vec::new())).collect();
        for i in 0..60_000u64 {
            let h = mix64(0xD1CE ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = TAU * u01k(h, 2) - PI;
            let c = Ctx::new(lat, lon, 30.0, &w.ell);
            let t = w.terrain(&c);
            if t.water_kind != 0 || t.ground < 1.0 || t.moist > 0.4 {
                continue;
            }
            let (_, rho) = wind(seed, c.p, c.lat);
            let (wt, wl, ws) = regime(rho);
            let at = format!("{:.4},{:.4} (h {:.0} m, moist {:.2}, sand {:.2}, litho?)", lat.to_degrees(), lon.to_degrees(), t.ground, t.moist, t.sand);
            let mut add = |k: usize| {
                if found[k].1.len() < 6 {
                    found[k].1.push(at.clone());
                }
            };
            if t.sand > 0.7 {
                if wt > 0.9 {
                    add(0);
                } else if wl > 0.9 {
                    add(1);
                } else if ws > 0.9 {
                    add(2);
                }
            }
            if (0.12..0.4).contains(&t.sand) && rho < 0.35 {
                add(3);
            }
            let mp = playa_mask(t.moist, t.sand, t.mountain, t.mesa);
            if playa(&w, seed, c.p, c.east, c.north, mp).is_some_and(|p| p.0 < 0.7) {
                add(4);
            }
            let semi = 1.0 - smoothstep(0.30, 0.45, t.moist);
            let mi = semi * (1.0 - t.mountain) * (1.0 - 0.6 * t.sand) * (1.0 - smoothstep(0.2, 0.5, t.mesa));
            if mi > 0.0 && inselbergs(&w, seed, &c, mi) > 120.0 {
                add(5);
            }
            let hyper = 1.0 - smoothstep(0.05, 0.12, t.moist);
            if hyper > 0.5 && t.sand < 0.02 && perlin3(seed ^ S_YZONE, c.p / 120_000.0) > 0.35 {
                add(6);
            }
            if t.river_hw > 20.0 && t.river_wet < 0.5 && t.river_d.abs() < t.river_hw {
                add(7);
            }
        }
        for (k, v) in &found {
            eprintln!("{k}:");
            for s in v {
                eprintln!("    {s}");
            }
        }
    }
}
