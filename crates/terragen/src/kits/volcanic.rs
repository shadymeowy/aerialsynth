//! The volcanic kit (`docs/design/terrain-next.md` §5.2, `docs/next/kits/volcanic.md`): stratovolcanoes
//! with craters, radial gullies, lava flows and ash; calderas with lakes; shield volcanoes;
//! cinder-cone fields with maars; black-sand beaches. The WGSL twins are
//! `gpu/wgsl/kits/volcanic_relief.wgsl` (families, relief, shared geometry) and
//! `gpu/wgsl/kits/volcanic.wgsl` (the layer): the same functions in the same order of operations.
//!
//! **Families** (instances of jittered 3D lattices, existence from the atlas' volcanism, plate
//! boundary and hotspot fields at the site):
//!
//! | family | cell | what |
//! |---|---|---|
//! | `volcanic_volcanoes` | 40 km | stratovolcanoes (crater) and calderas (flat floor, lake) on arcs, rifts and hotspot tracks; R 6–25 km |
//! | `volcanic_shields` | 110 km | shield volcanoes on hotspot tracks: R 22–45 km, summit pit |
//! | `volcanic_fields` | 28 km | cinder-cone fields: a lattice of cones, maars and lava tongues per field (dense objects are a per-pixel lattice inside one instance, never a dense family) |
//!
//! **Relief (pass A).** A volcano is an *absolute* shape: its summit is `E + H` above the atlas'
//! smooth elevation `E` at its site (so oceanic volcanoes grow from the sea floor and make
//! islands), the flank `E + H (1 - r/R)^1.6` (shield `^1.3`), raised smoothly over the ground
//! (`smooth max`, fading to nothing at the foot). The crater (stratovolcano: a bowl, caldera: a
//! flat floor with a ring wall) overrides the ground: its radius is widened to `1.5 gsd` with the
//! volume preserved (depth × (rc / rc_eff)²). Radial gullies (three tiers of 14, 30 and 62
//! ravines around the cone, zero mean, band-limited) and flank noise are carved into the flank;
//! the drainage then follows them. The rim is ragged by three sine harmonics of the azimuth.
//!
//! **Pass B (slot 3).** Ash and scoria on the upper cone and as an ash-fall apron (noise
//! threshold, crisp), lava tongues running down the flanks (young black, weathered brown,
//! old moss-green), hydrothermal patches around the crater, crater lakes (a flat water level of
//! the instance, `Layer::water` with an absolute height), black-sand beaches. The cinder-cone
//! fields: dark cones with craters, maar lakes inside tephra rings, short lava tongues, an ash
//! apron. Snow caps come from the climate (the cold kit's snow model, the lapse rate).

use crate::instances::{Family, InstSite, Instance};
use crate::kits::{Kit, ReliefIn};
use crate::noise::*;
use crate::stack::{hmode, slot, Layer, Stack};
use crate::surface::{mixc, srgb};
use glam::DVec3;

pub static KIT: Kit = Kit {
    name: "volcanic",
    biomes_yaml: include_str!("../../biomes/volcanic.yaml"),
    wgsl: include_str!("../gpu/wgsl/kits/volcanic.wgsl"),
    layers: &[(slot::AZONAL, azonal)],
    relief: Some(relief),
    relief_wgsl: include_str!("../gpu/wgsl/kits/volcanic_relief.wgsl"),
    families: &[
        Family { name: "volcanic_volcanoes", cell: 40_000.0, jitter: 0.8, reach: 26_000.0, relief: true, exists: volcano_exists },
        Family { name: "volcanic_shields", cell: 110_000.0, jitter: 0.7, reach: 46_000.0, relief: true, exists: shield_exists },
        Family { name: "volcanic_fields", cell: 28_000.0, jitter: 0.8, reach: 9_000.0, relief: true, exists: field_exists },
    ],
    ..Kit::EMPTY
};

/// Land-cover v2 class ids (`registry::classes`; checked by a test).
mod cls {
    pub const LAKE: u8 = 2;
    pub const LAVA: u8 = 31;
    pub const VOLCANIC_ASH: u8 = 32;
    pub const BARE: u8 = 16;
}

/// Noise of wavelength `lam` at `p`, band-limited (0 where the pixel cannot resolve it: no noise
/// evaluated).
#[inline]
fn nz(seed: u64, p: DVec3, lam: f64, gsd: f64) -> f64 {
    let b = band(lam, gsd);
    if b > 0.0 {
        perlin3(seed, p / lam) * b
    } else {
        0.0
    }
}

const SEED_ROUGH: u64 = 0x701C_A002;
const SEED_FLANK: u64 = 0x701C_A001;
const SEED_LAVA: u64 = 0x701C_A003;
const SEED_ASH: u64 = 0x701C_A004;
const SEED_HYD: u64 = 0x701C_A005;

fn fam(name: &str) -> usize {
    crate::instances::families().position(|f| f.name == name).unwrap()
}
fn fam_volcanoes() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| fam("volcanic_volcanoes"))
}
fn fam_shields() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| fam("volcanic_shields"))
}
fn fam_fields() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| fam("volcanic_fields"))
}

// ============================================================================= existence

/// Volcanism at the atlas texel nearest to the site (a few loads: the sites' existence tests run
/// at every drainage lattice point, so the full B-spline sample is kept for the sites that pass
/// this; `volcanic_pre` in WGSL).
fn pre_volcanism(w: &crate::world::World, center: DVec3) -> f64 {
    use crate::atlas::{f16_value, slot, APRON};
    let a = w.atlas();
    let r = a.resolution();
    let rf = r as f64;
    let (f, x, y) = crate::atlas::geom::Grid::new(r).locate(center.normalize());
    let near = |x: f64| ((x.clamp(-0.5, rf - 0.5) + 0.5).floor() as usize).min(r - 1);
    f16_value(a.raw(f, near(x) + APRON, near(y) + APRON, slot::VOLCANISM)) as f64
}

/// Texels with less volcanism than this have no volcanoes.
const PRE_VOLCANISM: f64 = 0.08;

/// Stratovolcanoes and calderas exist on arcs, rifts and hotspot tracks (the atlas' volcanism at
/// the site: strong enough, and on land or on an island-arc / hotspot sea floor). v: radius,
/// base elevation E, height H above it, crater radius, depth, lake level (absolute; < −1e5:
/// none), activity, kind (0 stratovolcano, 1 caldera, 2 shield).
fn volcano_exists(w: &crate::world::World, s: &InstSite) -> Option<[f32; 8]> {
    if u01k(s.id, 1) >= 0.55 || pre_volcanism(w, s.center) < PRE_VOLCANISM {
        return None;
    }
    let a = w.atlas().sample(s.center.normalize());
    let oceanic = a.elevation_m < -150.0;
    let vol = a.volcanism;
    let need = if oceanic { 0.36 } else { 0.2 };
    if vol < need {
        return None;
    }
    if oceanic && !(a.hotspot || a.boundary == crate::atlas::Boundary::IslandArc) {
        return None;
    }
    if u01k(s.id, 2) >= 0.15 + 0.85 * smoothstep(need, need + 0.35, vol) {
        return None;
    }
    let kind = if u01k(s.id, 3) < 0.2 { 1.0 } else { 0.0 };
    let e = a.elevation_m;
    // the summit: 2.0–3.7 km over land, 250–1750 m a.s.l. over the sea
    let summit = if oceanic { 250.0 + 1500.0 * u01k(s.id, 4) } else { e.max(0.0) + 2000.0 + 1700.0 * u01k(s.id, 4) };
    let h = summit - e;
    let r = (h * (3.6 + 0.9 * u01k(s.id, 5))).clamp(6000.0, 25_500.0);
    let (rc, d) = if kind > 0.5 {
        (1600.0 + 2200.0 * u01k(s.id, 6), 350.0 + 450.0 * u01k(s.id, 7))
    } else {
        (220.0 + 500.0 * u01k(s.id, 6), 70.0 + 150.0 * u01k(s.id, 7))
    };
    let zrim = e + h * (1.0 - rc / r).max(0.0).powf(1.6);
    let lake = if kind > 0.5 && u01k(s.id, 8) < 0.8 && zrim - d > 30.0 {
        zrim - 0.82 * d
    } else if kind < 0.5 && rc > 350.0 && u01k(s.id, 8) < 0.25 && zrim - d > 30.0 {
        zrim - 0.65 * d
    } else {
        -1.0e6
    };
    let f = |x: f64| x as f32;
    Some([f(r), f(e), f(h), f(rc), f(d), f(lake), f(u01k(s.id, 9)), f(kind)])
}

/// Shield volcanoes: wide and low, on hotspot tracks and (less) in rifts. Same layout as above.
fn shield_exists(w: &crate::world::World, s: &InstSite) -> Option<[f32; 8]> {
    if u01k(s.id, 1) >= 0.6 || pre_volcanism(w, s.center) < PRE_VOLCANISM {
        return None;
    }
    let a = w.atlas().sample(s.center.normalize());
    let vol = a.volcanism;
    let rift = a.boundary == crate::atlas::Boundary::Rift && a.continental;
    if vol < 0.3 || !(a.hotspot || (rift && u01k(s.id, 12) < 0.4)) {
        return None;
    }
    let oceanic = a.elevation_m < -150.0;
    let e = a.elevation_m;
    let summit = if oceanic { 400.0 + 2000.0 * u01k(s.id, 4) } else { e.max(0.0) + 700.0 + 900.0 * u01k(s.id, 4) };
    let h = summit - e;
    let r = (h * (7.0 + 3.0 * u01k(s.id, 5))).clamp(22_000.0, 45_000.0);
    let (rc, d) = (900.0 + 1100.0 * u01k(s.id, 6), 90.0 + 110.0 * u01k(s.id, 7));
    let zrim = e + h * (1.0 - rc / r).max(0.0).powf(1.3);
    let lake = if u01k(s.id, 8) < 0.15 && zrim - d > 30.0 { zrim - 0.7 * d } else { -1.0e6 };
    let f = |x: f64| x as f32;
    Some([f(r), f(e), f(h), f(rc), f(d), f(lake), f(u01k(s.id, 9)), 2.0])
}

/// Cinder-cone fields on land with moderate volcanism (arcs, rifts, hotspot tracks). v: field
/// radius, cone lattice cell, amplitude, share of cones with a lava tongue, tongue phase, lava
/// age, share of maars.
fn field_exists(w: &crate::world::World, s: &InstSite) -> Option<[f32; 8]> {
    if u01k(s.id, 1) >= 0.7 || pre_volcanism(w, s.center) < PRE_VOLCANISM {
        return None;
    }
    let a = w.atlas().sample(s.center.normalize());
    let vol = a.volcanism;
    if a.elevation_m < -30.0 || vol < 0.18 || u01k(s.id, 2) >= 0.2 + 0.8 * smoothstep(0.18, 0.45, vol) {
        return None;
    }
    let f = |x: f64| x as f32;
    Some([
        f(5500.0 + 3000.0 * u01k(s.id, 3)),
        f(550.0 + 350.0 * u01k(s.id, 4)),
        f(0.8 + 0.5 * u01k(s.id, 5)),
        f(0.2 + 0.3 * u01k(s.id, 6)),
        f(u01k(s.id, 7)),
        f(u01k(s.id, 8)),
        f(0.04 + 0.14 * u01k(s.id, 9)),
        0.0,
    ])
}

// ============================================================================= geometry

/// The east and north axes of an instance's frame (at its centre).
fn frame(center: DVec3) -> (DVec3, DVec3) {
    let up = center.normalize();
    let e = if up.z.abs() > 0.999 { DVec3::X } else { DVec3::Z.cross(up).normalize() };
    (e, up.cross(e))
}

#[inline]
fn smax(a: f64, b: f64, k: f64) -> f64 {
    a.max(b) + (k - (a - b).abs()).max(0.0).powi(2) / (4.0 * k)
}

const TWO_OVER_PI: f64 = std::f64::consts::FRAC_2_PI;
/// Ravines per tier and their relative depth; the depth of the first tier over H.
const GULLY_N: [f64; 3] = [14.0, 30.0, 62.0];
const GULLY_A: [f64; 3] = [1.0, 0.6, 0.35];
const GULLY_AMP: f64 = 0.04;
/// Flank noise: wavelengths (m) and amplitudes (× H).
const ROUGH_L: [f64; 3] = [1600.0, 520.0, 170.0];
const ROUGH_A: [f64; 3] = [0.02, 0.01, 0.0045];

/// The relative depth (mean 1) of the ravine at the phase `phi` of tier `j`: a random depth per
/// ravine (between two crests: `phi / pi`), interpolated across the crest so the surface stays
/// continuous.
fn ravine_amp(id: u64, j: usize, phi: f64) -> f64 {
    let x = phi / std::f64::consts::PI;
    let k = x.floor();
    let n = GULLY_N[j] as i64;
    let a = |i: i64| 0.55 + 0.9 * u01k(hash2(id, j as i64, i.rem_euclid(n)), 1);
    let (k0, f) = (k as i64, x - k);
    lerp(a(k0), a(k0 + 1), smoothstep(0.0, 1.0, f))
}

/// A volcano instance.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Vol {
    pub id: u64,
    pub center: DVec3,
    pub r: f64,
    pub e: f64,
    pub h: f64,
    pub rc: f64,
    pub d: f64,
    pub lake: f64,
    pub act: f64,
    pub kind: u32,
}

impl Vol {
    pub(crate) fn of(it: &Instance) -> Vol {
        let v = it.v.map(|x| x as f64);
        Vol { id: it.id, center: it.center, r: v[0], e: v[1], h: v[2], rc: v[3], d: v[4], lake: v[5], act: v[6], kind: v[7] as u32 }
    }
    /// The flank profile (1 at the axis, 0 at the foot) at `q` = r / R.
    fn pf(&self, q: f64) -> f64 {
        (1.0 - q).max(0.0).powf(if self.kind == 2 { 1.3 } else { 1.6 })
    }
}

/// The shape of a volcano at a point.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Cone {
    /// offset (m) in the instance's frame
    pub x: f64,
    pub y: f64,
    /// radius over the rim's raggedness (m), and over R
    pub rr: f64,
    pub u: f64,
    /// azimuth in the instance's frame (rad)
    pub th: f64,
    /// crater radius after widening to the pixel (m)
    pub rce: f64,
    /// absolute elevation of the shape (m); inside the crater the ground itself
    pub z: f64,
    pub bowl: bool,
}

/// The volcano `v` at `p` (`full`: with the gullies and the flank noise, for the relief).
pub(crate) fn cone(v: &Vol, p: DVec3, gsd: f64, full: bool) -> Option<Cone> {
    // (a volcano smaller than two pixels is widened with its volume kept)
    let rw = v.r.max(2.0 * gsd);
    let vh = v.h * (v.r / rw) * (v.r / rw);
    let (e, n) = frame(v.center);
    let rel = p - v.center;
    let (x, y) = (rel.dot(e), rel.dot(n));
    let r = x.hypot(y);
    if r >= 1.14 * rw {
        return None;
    }
    let th = y.atan2(x);
    let tau = std::f64::consts::TAU;
    let edge =
        1.0 + 0.07 * (3.0 * th + tau * u01k(v.id, 10)).sin() + 0.04 * (5.0 * th + tau * u01k(v.id, 11)).sin() + 0.025 * (9.0 * th + tau * u01k(v.id, 12)).sin();
    let rr = r / edge;
    if rr >= rw {
        return None;
    }
    let u = rr / rw;
    let rce = v.rc.max(1.5 * gsd);
    let de = v.d * (v.rc / rce) * (v.rc / rce);
    let zrim = v.e + vh * v.pf(rce / rw);
    if rr < rce {
        let q = rr / rce;
        let z = if v.kind == 1 { zrim - de + de * smoothstep(0.5, 1.0, q) } else { zrim - de * (1.0 - q * q) };
        return Some(Cone { x, y, rr, u, th, rce, z, bowl: true });
    }
    let mut z = v.e + vh * v.pf(u);
    if full {
        let fl = smoothstep(1.0, 1.5, rr / rce);
        let f = fl * (1.0 - smoothstep(0.7, 1.0, u));
        if f > 0.0 {
            for k in 0..3 {
                let b = band(ROUGH_L[k], gsd);
                if b > 0.0 {
                    z += ROUGH_A[k] * vh * perlin3(SEED_ROUGH ^ v.id ^ k as u64, DVec3::new(x / ROUGH_L[k], y / ROUGH_L[k], 0.7)) * b * f;
                }
            }
        }
        if v.kind != 2 {
            let a = GULLY_AMP * vh * smoothstep(0.03, 0.16, u) * (1.0 - smoothstep(0.5, 0.92, u)) * fl;
            if a > 0.0 {
                let thw = th + 0.06 * perlin3(SEED_FLANK ^ v.id, DVec3::new(x / 1800.0, y / 1800.0, 0.37));
                for j in 0..3 {
                    let lam = tau * rr / GULLY_N[j];
                    let b = band(lam, gsd);
                    if b > 0.0 {
                        let phi = 0.5 * GULLY_N[j] * thw + tau * u01k(v.id, 13 + j as u64);
                        z += a * GULLY_A[j] * (phi.cos().abs() - TWO_OVER_PI) * ravine_amp(v.id, j, phi) * b;
                    }
                }
            }
        }
    }
    Some(Cone { x, y, rr, u, th, rce, z, bowl: false })
}

/// The ground height with the volcano: the crater overrides it, the flank rises out of it.
fn apply(h: &mut f64, c: &Cone) {
    if c.bowl {
        *h = c.z;
        return;
    }
    let w = smoothstep(1.0, 0.7, c.u);
    let m = smax(*h, c.z, 40.0);
    *h += w * (m - *h);
}

// ---- cinder-cone fields

/// Mean heights of the lattice's cones and maars over cell × amplitude (the profiles' volumes
/// and the size distributions; a test checks them against explicit averages), and the mean
/// cover of cinder cones and lava tongues.
const K_CONE: f64 = 0.008_581_6;
const K_MAAR: f64 = -0.001_225_1;
const MEAN_CINDER: f64 = 0.2190;
/// mean lava tongue cover × cell (m)
const MEAN_LAVA: f64 = 51.0;

/// Profile of a cinder cone at ρ = d / R (1 at the crater rim's top).
fn cone_profile(rho: f64) -> f64 {
    let g = |r: f64| (1.0 - r).max(0.0).powf(1.35);
    g(rho.max(0.22)) - 0.45 * g(0.22) * (1.0 - (rho / 0.22) * (rho / 0.22)).max(0.0)
}

/// What a cinder-cone field is at a point.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Field {
    /// height (m)
    pub z: f64,
    /// cinder cone cover, and the cone's hash
    pub cinder: f64,
    pub chash: u64,
    pub crho: f64,
    /// the tephra ring of a maar, and its lake
    pub ring: f64,
    pub lake: f64,
    /// lava tongue cover and age
    pub lava: f64,
    pub age: f64,
    /// ash apron strength
    pub apron: f64,
    pub dens: f64,
}

/// ρ = distance over the radius `r` of a lattice object at the offset (qx, qy), the outline ragged
/// by two harmonics of the azimuth.
fn ragged(qx: f64, qy: f64, r: f64, h: u64) -> f64 {
    let th = qy.atan2(qx);
    let t = std::f64::consts::TAU;
    qx.hypot(qy) / (r * (1.0 + 0.12 * (3.0 * th + t * u01k(h, 15)).sin() + 0.08 * (5.0 * th + t * u01k(h, 16)).sin()))
}

/// Half width of a lava tongue at the fraction `sg` of its length: narrow at the vent, widening
/// downstream to a blunt toe (a lens of `sin(π sg)^0.6`, short and wide, drew round black discs
/// in the cinder fields, braided).
fn tongue_w(sg: f64, w0: f64) -> f64 {
    w0 * (sg.max(0.0) * (1.0 - sg.powi(4)).max(0.0)).sqrt()
}

/// A lava tongue from the origin along `dir` (rad) between `s0` and `len` (m): (cover, half
/// width). The axis wanders, the margin is lobate (`nzv`: noise in the frame of the instance,
/// ~[-1, 1], band-limited).
#[allow(clippy::too_many_arguments)]
fn tongue(x: f64, y: f64, dir: f64, s0: f64, len: f64, w0: f64, ph: f64, nzv: f64, fw: f64) -> (f64, f64) {
    let (sn, cs) = dir.sin_cos();
    let s = x * cs + y * sn;
    let t = -x * sn + y * cs;
    let sg = (s - s0) / (len - s0);
    if sg <= 0.0 || sg >= 1.0 {
        return (0.0, 0.0);
    }
    let wander = 0.035 * len * (s / (0.22 * len) + ph).sin();
    let w = tongue_w(sg, w0) * (1.0 + 0.22 * nzv);
    (((w - (t - wander).abs()) / fw + 0.5).clamp(0.0, 1.0), w)
}

/// The margin noise of the lava tongues at (x, y) of an instance's frame.
fn margin_noise(id: u64, x: f64, y: f64, gsd: f64) -> f64 {
    let b = band(90.0, gsd);
    if b > 0.0 {
        perlin3(SEED_LAVA ^ id, DVec3::new(x / 90.0, y / 90.0, 0.37)) * b
    } else {
        0.0
    }
}

/// The field `it` at `p` (`look`: also the lava tongues).
pub(crate) fn field(it: &Instance, p: DVec3, gsd: f64, fw: f64, look: bool) -> Field {
    let v = it.v.map(|x| x as f64);
    let (rf, cs, amp, lshare, lage, mshare) = (v[0], v[1], v[2], v[3], v[5], v[6]);
    let mut o = Field::default();
    let (e, n) = frame(it.center);
    let rel = p - it.center;
    let (x, y) = (rel.dot(e), rel.dot(n));
    let d = x.hypot(y);
    if d >= rf + 1.2 * cs {
        return o;
    }
    let dens_at = |dc: f64| 0.8 * (1.0 - smoothstep(0.35 * rf, rf, dc));
    let dm = dens_at(d);
    o.dens = dm;
    o.apron = 1.0 - smoothstep(0.55 * rf, 1.1 * rf, d);
    let bnd = band(0.35 * cs, gsd);
    let mean_z = dm * cs * amp * ((1.0 - mshare) * K_CONE + mshare * K_MAAR);
    o.z = (1.0 - bnd) * mean_z;
    o.cinder = (1.0 - bnd) * dm * (1.0 - mshare) * MEAN_CINDER;
    o.lava = (1.0 - bnd) * dm * (1.0 - mshare) * lshare * MEAN_LAVA / cs;
    o.age = (lage + 0.35).min(1.0);
    if bnd <= 0.0 {
        return o;
    }
    let (ix, iy) = ((x / cs).floor() as i64, (y / cs).floor() as i64);
    let (mut zs, mut cinder, mut lava, mut age) = (0.0, 0.0f64, 0.0f64, lage);
    let nzv = if look { margin_noise(it.id, x, y, gsd) } else { 0.0 };
    // (without the lava tongues only the cells within 0.43 cells of the pixel matter)
    let (fx, fy) = (x / cs - ix as f64, y / cs - iy as f64);
    for dy in -1..=1i64 {
        for dx in -1..=1i64 {
            if !look && ((dx < 0 && fx > 0.35) || (dx > 0 && fx < 0.65) || (dy < 0 && fy > 0.35) || (dy > 0 && fy < 0.65)) {
                continue;
            }
            let (gx, gy) = (ix + dx, iy + dy);
            let h = hash2(it.id, gx, gy);
            let cx = (gx as f64 + 0.5 + 0.7 * (u01k(h, 2) - 0.5)) * cs;
            let cy = (gy as f64 + 0.5 + 0.7 * (u01k(h, 3) - 0.5)) * cs;
            if u01k(h, 1) >= dens_at(cx.hypot(cy)) {
                continue;
            }
            let (qx, qy) = (x - cx, y - cy);
            if u01k(h, 7) < mshare {
                let rm = cs * (0.12 + 0.12 * u01k(h, 4));
                let dp = rm * (0.18 + 0.12 * u01k(h, 5)) * amp;
                let rho = ragged(qx, qy, rm, h);
                if rho < 1.45 {
                    if rho < 1.0 {
                        zs -= dp * (1.0 - smoothstep(0.55, 1.0, rho));
                    }
                    zs += 0.3 * dp * (1.0 - ((rho - 1.0) / 0.45) * ((rho - 1.0) / 0.45)).max(0.0);
                    o.ring = o.ring.max(smoothstep(0.7, 0.95, rho) * smoothstep(1.45, 1.1, rho) * bnd);
                    // (a crisp shore: over 13 % of the radius it blurred into a pale halo)
                    o.lake = o.lake.max(smoothstep(0.71, 0.66, rho) * bnd);
                }
            } else {
                let rc = cs * (0.16 + 0.2 * u01k(h, 4));
                let hc = 2.0 * rc * (0.2 + 0.16 * u01k(h, 5)) * amp;
                let rho = ragged(qx, qy, rc, h);
                if rho < 1.0 {
                    zs += hc * cone_profile(rho);
                }
                // (the cone and a faint apron of scoria around its foot: an outline that ended at
                // the foot looked stamped on, braided)
                let cv = smoothstep(1.0, 0.8, rho).max(0.25 * smoothstep(1.6, 0.95, rho));
                if cv > cinder {
                    cinder = cv;
                    o.chash = h;
                    o.crho = rho;
                }
                if look && u01k(h, 8) < lshare {
                    let dir = std::f64::consts::TAU * u01k(h, 9);
                    let len = cs * (0.55 + 0.45 * u01k(h, 10));
                    let w0 = 55.0 + 60.0 * u01k(h, 11);
                    let (c, _) = tongue(qx, qy, dir, 0.8 * rc, len, w0, std::f64::consts::TAU * u01k(h, 12), nzv, fw);
                    if c > lava {
                        lava = c;
                        // (the tongues of a cinder field mostly weathered: fresh black flows drew holes in the plain)
                        age = (lage + 0.35 + 0.5 * (u01k(h, 13) - 0.5)).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
    o.z += bnd * zs;
    o.cinder += bnd * cinder;
    if lava > 0.0 {
        o.lava += bnd * lava;
        o.age = age;
    }
    o
}

// ============================================================================= relief

/// Pass A: the volcanoes (the drainage follows them), the cinder-cone fields.
fn relief(w: &crate::world::World, r: &ReliefIn, h: &mut f64) {
    let (p, gsd) = (r.ctx.p, r.ctx.gsd);
    for f in [fam_volcanoes(), fam_shields()] {
        for it in r.instances(w, f).iter() {
            if let Some(c) = cone(&Vol::of(it), p, gsd, true) {
                apply(h, &c);
            }
        }
    }
    for it in r.instances(w, fam_fields()).iter() {
        *h += field(it, p, gsd, gsd, false).z;
    }
}

// ============================================================================= pass B

// colours (sRGB)
fn ash_col(t: f64) -> DVec3 {
    mixc(srgb(40.0, 38.0, 38.0), srgb(70.0, 63.0, 58.0), t)
}

/// The colour of lava by age: fresh black, weathered brown, old moss-green; and its class and
/// the share of the vegetation it removes.
fn lava_look(age: f64, p: DVec3, gsd: f64) -> (DVec3, u8, f64) {
    // (the three colours mixed by age: only the noises of the colours that count)
    let t1 = smoothstep(0.22, 0.5, age);
    let t2 = 0.85 * smoothstep(0.55, 0.95, age);
    let (wf, wm, wo) = ((1.0 - t2) * (1.0 - t1), (1.0 - t2) * t1, t2);
    let mut c = DVec3::ZERO;
    if wf > 0.0 {
        // (fresh basalt at an albedo of ~4 %, rough: 24 DN read as a hole in the ground)
        c += wf * srgb(66.0, 62.0, 62.0) * (1.0 + 0.3 * nz(SEED_LAVA, p, 14.0, gsd));
    }
    if wm > 0.0 {
        c += wm * mixc(srgb(66.0, 58.0, 54.0), srgb(106.0, 70.0, 50.0), 0.5 + 0.5 * nz(SEED_LAVA + 1, p, 40.0, gsd));
    }
    if wo > 0.0 {
        c += wo * mixc(srgb(88.0, 100.0, 66.0), srgb(124.0, 124.0, 98.0), 0.5 + 0.5 * nz(SEED_LAVA + 2, p, 25.0, gsd));
    }
    (c, if age < 0.65 { cls::LAVA } else { 0 }, 1.0 - smoothstep(0.5, 0.95, age))
}

#[inline]
fn mix1(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Slot 3: black-sand beaches, the volcanoes, the cinder-cone fields.
pub fn azonal(s: &mut Stack) {
    beach(s);
    for f in [fam_volcanoes(), fam_shields()] {
        let inst = s.instances(f).into_owned();
        for it in inst.iter() {
            volcano_look(s, &Vol::of(it));
        }
    }
    let inst = s.instances(fam_fields()).into_owned();
    for it in inst.iter() {
        field_look(s, it);
    }
}

/// Black sand on the beaches of volcanic coasts (the core's beach, recoloured).
fn beach(s: &mut Stack) {
    let b = s.m.beach;
    if b <= 0.0 || !(s.t.volcanism > 0.3 || s.bio.style.litho == 3) {
        return;
    }
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let dry = mixc(srgb(52.0, 50.0, 50.0), srgb(78.0, 72.0, 68.0), 0.5 + 0.5 * nz(SEED_ASH + 9, p, 9.0, gsd));
    let wet = srgb(26.0, 26.0, 28.0);
    let col = mixc(dry, wet, 1.0 - smoothstep(0.05, 0.3, s.l.ground));
    s.composite(Layer::paint_cls(b, col, cls::VOLCANIC_ASH));
}

fn volcano_look(s: &mut Stack, v: &Vol) {
    let p = s.ctx.p;
    let gsd = s.ctx.gsd;
    let Some(k) = cone(v, p, gsd, false) else { return };
    // no fields in the crater, no towns on the cone (on the ash apron of an active one)
    if k.u < 0.45 + 0.55 * v.act || k.rr < 1.6 * k.rce {
        s.no_town = true;
    }
    s.m.natural_ok *= smoothstep(1.0, 1.6, k.rr / k.rce);
    ash(s, v, &k);
    flows(s, v, &k);
    hydrothermal(s, v, &k);
    crater_lake(s, v, &k);
}

/// Ash and scoria: bare upper cone, an ash-fall apron around active volcanoes (crisp threshold of
/// a noise, wider where the noise is below the pixel).
fn ash(s: &mut Stack, v: &Vol, k: &Cone) {
    let (p, gsd) = (s.ctx.p, s.ctx.gsd);
    let upper = (0.45 + 0.55 * v.act) * (1.0 - smoothstep(0.1, 0.45, k.u));
    let apron = 0.75 * v.act * v.act * (1.0 - smoothstep(0.15, 0.8, k.u));
    let m = upper.max(apron);
    if m < 0.02 {
        return;
    }
    // the ash edge follows the cone's radial structure up high (ash lies in the ravines and
    // streaks down the summit cone: harmonics of a wandering azimuth, each only where its
    // wavelength at this radius is resolved); lower down the apron thins out in soft blotches
    // (rays reaching across the apron drew a dark star from above and spikes in oblique views)
    let tau = std::f64::consts::TAU;
    let wl = |n: f64| band(tau * k.rr / n, gsd);
    let hi = 1.0 - smoothstep(0.06, 0.22, k.u);
    let mut n = 0.3 * nz(SEED_ASH + 1, p, 90.0, gsd) + 0.3 * nz(SEED_ASH + 4, p, 320.0, gsd) + 0.25 * nz(SEED_ASH + 2, p, 900.0, gsd);
    if hi > 0.0 {
        // (a sideways shift of up to ~100 m, not an angle: that folded the flanks into swirls)
        let th = k.th + 100.0 * nz(SEED_ASH + 3, p, 500.0, gsd) / k.rr.max(300.0);
        n += hi
            * (0.15 * (7.0 * th + tau * u01k(v.id, 31)).sin() * wl(7.0)
                + 0.2 * (17.0 * th + tau * u01k(v.id, 32)).sin() * wl(17.0)
                + 0.2 * (41.0 * th + tau * u01k(v.id, 33)).sin() * wl(41.0));
    }
    let wd = (0.07 + 0.3 * (1.0 - band(260.0, gsd))).max(0.3 * (1.0 - hi));
    let cov = smoothstep(-wd, wd, m - 0.35 + 0.5 * n);
    if cov <= 0.0 {
        return;
    }
    // (dark scoria up the cone, paler weathered ash lower down)
    let mut col = mixc(ash_col((0.5 + 0.6 * n).clamp(0.0, 1.0)), srgb(112.0, 104.0, 94.0) * (1.0 + 0.3 * n), smoothstep(0.08, 0.35, k.u));
    // red scoria on the summit cone
    col = mixc(col, srgb(104.0, 58.0, 42.0), 0.35 * smoothstep(0.22, 0.06, k.u) * (0.3 + 0.7 * v.act));
    s.composite(Layer { cov, albedo: col, cls: cls::VOLCANIC_ASH, clear: 0.95, ..Default::default() });
    s.m.natural_ok *= 1.0 - 0.8 * cov;
}

/// Lava tongues down the flanks (2–6 per volcano by its activity; young where active).
fn flows(s: &mut Stack, v: &Vol, k: &Cone) {
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, s.l.fw);
    let nf = 2.0 + 4.0 * v.act;
    let scale = (v.r / 12000.0).clamp(0.7, 1.6);
    let e = smoothstep(1.2 * gsd, 3.0 * gsd, 200.0 * scale);
    let (mut cov, mut age, mut mean, mut mage) = (0.0f64, 0.0, 0.0f64, 0.0);
    let mut nzv: Option<f64> = None;
    for i in 0..6u64 {
        if i as f64 >= nf {
            break;
        }
        let hh = hash1(v.id, 50 + i as i64);
        let dir = std::f64::consts::TAU * u01k(hh, 1);
        // (a shield's flows are short and fat: its radius is tens of kilometres)
        let len = if v.kind == 2 { (v.r * (0.1 + 0.15 * u01k(hh, 2))).min(7000.0) } else { (v.r * (0.3 + 0.55 * u01k(hh, 2))).min(12_000.0) };
        let w0 = (90.0 + 200.0 * u01k(hh, 3)) * scale;
        let a = ((1.0 - v.act) * 0.8 + 0.6 * u01k(hh, 4) - 0.1).clamp(0.0, 1.0);
        let s0 = 1.3 * k.rce;
        let ph = std::f64::consts::TAU * u01k(hh, 5);
        // (only the tongues the sample is beside: the azimuth first)
        let d = (k.th - dir + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        let (c, _) = if e > 0.0 && d.abs() < 1.5 && k.rr * d.sin().abs() < 1.3 * w0 + 0.04 * len + 2.0 * fw {
            tongue(k.x, k.y, dir, s0, len, w0, ph, *nzv.get_or_insert_with(|| margin_noise(v.id, k.x, k.y, gsd)), fw)
        } else {
            (0.0, 0.0)
        };
        if c > cov {
            cov = c;
            age = a;
        }
        // the mean over the azimuth at this radius
        if e < 1.0 && k.rr > s0 && k.rr < len {
            let m = tongue_w((k.rr - s0) / (len - s0), w0) / (std::f64::consts::PI * k.rr);
            if m > mean {
                mean = m;
                mage = a;
            }
        }
    }
    let c = e * cov + (1.0 - e) * mean;
    let a = if e * cov >= (1.0 - e) * mean { age } else { mage };
    if c <= 0.0 {
        return;
    }
    let (col, cl, clear) = lava_look(a, p, gsd);
    s.composite(Layer { cov: c, albedo: col, cls: cl, clear, ..Default::default() });
    s.m.natural_ok *= 1.0 - c * clear;
}

/// Sulphur, rust and clay patches around the crater.
fn hydrothermal(s: &mut Stack, v: &Vol, k: &Cone) {
    let (p, gsd) = (s.ctx.p, s.ctx.gsd);
    let zone = 1.0 - smoothstep(v.rc, v.rc * 2.6 + 200.0, k.rr);
    if zone <= 0.0 {
        return;
    }
    let strength = 0.35 + 0.65 * v.act;
    let n = nz(SEED_HYD ^ v.id, p, 38.0, gsd) + 0.5 * nz((SEED_HYD ^ v.id) + 1, p, 11.0, gsd);
    let wd = 0.06 + 0.25 * (1.0 - band(38.0, gsd));
    let cov = smoothstep(-wd, wd, n - (1.3 - zone * strength));
    if cov <= 0.0 {
        return;
    }
    let pick = nz(SEED_HYD + 7, p, 90.0, gsd);
    let col = mixc(mixc(srgb(204.0, 178.0, 62.0), srgb(176.0, 92.0, 48.0), smoothstep(-0.15, 0.2, pick)), srgb(176.0, 158.0, 118.0), smoothstep(0.1, 0.5, n));
    s.composite(Layer { cov, albedo: col, cls: cls::BARE, clear: 1.0, ..Default::default() });
    s.m.natural_ok *= 1.0 - cov;
}

/// The crater or caldera lake: water at the instance's level where the ground is below it.
fn crater_lake(s: &mut Stack, v: &Vol, k: &Cone) {
    if v.lake < -1.0e5 || k.rr > 1.02 * v.rc.max(k.rce) {
        return;
    }
    let l = s.l;
    let depth = v.lake - l.ground;
    let cov = (depth / (l.slope * l.fw).max(0.03) + 0.5).clamp(0.0, 1.0);
    if cov <= 0.0 {
        return;
    }
    let acid = u01k(v.id, 14) < 0.5 || v.kind == 0;
    let deep = if acid { srgb(54.0, 150.0, 138.0) } else { srgb(18.0, 56.0, 90.0) };
    let col = mixc(deep, srgb(128.0, 176.0, 150.0), smoothstep(25.0, 0.0, depth));
    s.composite(Layer { cov, albedo: col, cls: cls::LAKE, hmode: hmode::ABS, dh: v.lake, relit: 1.0, water: true, clear: 1.0, ..Default::default() });
    s.m.natural_ok *= 1.0 - cov;
}

fn field_look(s: &mut Stack, it: &Instance) {
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, s.l.fw);
    let f = field(it, p, gsd, fw, true);
    if f.dens <= 0.0 && f.apron <= 0.0 {
        return;
    }
    // ash apron: a darkened ground, strongest at the field's heart
    if f.apron > 0.0 {
        let n = 0.5 + 0.5 * nz(SEED_ASH + 5, p, 120.0, gsd);
        let cov = 0.5 * f.apron * smoothstep(0.2, 0.8, n);
        s.composite(Layer { cov, albedo: ash_col(n) * 0.9, cls: cls::VOLCANIC_ASH, clear: 0.8, ..Default::default() });
    }
    if f.lava > 0.0 {
        let (col, cl, clear) = lava_look(f.age, p, gsd);
        s.composite(Layer { cov: f.lava, albedo: col, cls: cl, clear, ..Default::default() });
        s.m.natural_ok *= 1.0 - f.lava * clear;
    }
    if f.cinder > 0.0 {
        let t = u01k(f.chash, 14);
        let rho = f.crho;
        let mut col = mixc(srgb(44.0, 38.0, 38.0), srgb(118.0, 62.0, 42.0), 0.65 * t * t);
        // dark crater, pale rim, weathered to the ash at the foot
        col *= mix1(1.0, 0.72, smoothstep(0.22, 0.12, rho)) * (1.0 + 0.15 * smoothstep(0.15, 0.22, rho) * smoothstep(0.4, 0.22, rho));
        col = mixc(col, ash_col(u01k(f.chash, 17)) * 0.9, 0.5 * smoothstep(0.55, 0.95, rho));
        // (no light baked into the albedo: the cones are steeper instead, and their lower flanks
        // weather and grow over into the ground around them)
        col = mixc(col, s.col, 0.5 * smoothstep(0.55, 1.0, rho) + 0.4 * smoothstep(0.95, 1.4, rho));
        let col = col * (1.0 + 0.18 * nz(SEED_ASH + 6, p, 9.0, gsd));
        s.composite(Layer { cov: f.cinder, albedo: col, cls: cls::VOLCANIC_ASH, clear: 1.0, ..Default::default() });
        s.m.natural_ok *= 1.0 - f.cinder;
    }
    if f.ring > 0.0 {
        // (the tephra ring grown over: a bare pale ring drew a halo around every maar in the forest)
        s.composite(Layer { cov: f.ring, albedo: mixc(s.col, srgb(138.0, 118.0, 98.0), 0.4), cls: cls::VOLCANIC_ASH, clear: 0.4, ..Default::default() });
    }
    if f.lake > 0.0 {
        s.composite(Layer {
            cov: f.lake,
            albedo: srgb(30.0, 74.0, 80.0),
            cls: cls::LAKE,
            hmode: hmode::BLEND,
            dh: -0.2,
            relit: 1.0,
            water: true,
            clear: 1.0,
            ..Default::default()
        });
        s.m.natural_ok *= 1.0 - f.lake;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec2;

    #[test]
    fn registry_compiles_with_the_volcanic_biomes() {
        let r = crate::registry::Registry::builtin().unwrap();
        assert!(r.biomes.iter().any(|b| b.name == "volcanic_scrub"));
    }

    #[test]
    fn class_ids_are_the_v2_table() {
        use crate::registry::classes::id;
        assert_eq!(id("lake"), Some(cls::LAKE));
        assert_eq!(id("lava"), Some(cls::LAVA));
        assert_eq!(id("volcanic_ash"), Some(cls::VOLCANIC_ASH));
        assert_eq!(id("bare"), Some(cls::BARE));
    }

    fn test_vol(kind: u32) -> Vol {
        let center = DVec3::new(3_000_000.0, 3_500_000.0, 4_200_000.0).normalize() * 6_371_000.0;
        let (rc, d) = if kind == 1 { (2500.0, 500.0) } else { (400.0, 120.0) };
        Vol { id: 0xABCD_1234, center, r: 12_000.0, e: 300.0, h: 2800.0, rc, d, lake: -1.0e6, act: 0.6, kind }
    }

    fn at(v: &Vol, x: f64, y: f64) -> DVec3 {
        let (e, n) = frame(v.center);
        v.center + e * x + n * y
    }

    /// The cone rises out of the ground, its crater is below the rim, the caldera floor is flat.
    #[test]
    fn cone_shape() {
        for kind in [0u32, 1, 2] {
            let v = test_vol(kind);
            let z = |x: f64, y: f64, gsd: f64| cone(&v, at(&v, x, y), gsd, true).map(|c| c.z);
            assert!(z(1.15 * v.r, 0.0, 10.0).is_none() || kind == 99);
            let top = z(v.rc * 1.2, 0.0, 10.0).unwrap();
            let foot = z(0.9 * v.r, 0.0, 10.0).unwrap();
            assert!(top > foot + 0.3 * v.h, "kind {kind}: top {top} foot {foot}");
            let mid = z(0.0, 0.0, 10.0).unwrap();
            assert!(mid < top, "kind {kind}: crater {mid} rim {top}");
            if kind == 1 {
                let (a, b) = (z(0.0, 0.0, 10.0).unwrap(), z(0.4 * v.rc, 0.2 * v.rc, 10.0).unwrap());
                assert!((a - b).abs() < 1e-6, "caldera floor {a} {b}");
            }
        }
    }

    /// Widening the crater to the pixel keeps its volume (R-energy).
    #[test]
    fn crater_volume_is_kept_when_widened() {
        let v = test_vol(0);
        let vol = |gsd: f64| {
            let n = 400;
            let span = 2.0 * 1.5 * gsd.max(v.rc) * 1.2;
            let mut s = 0.0;
            for j in 0..n {
                for i in 0..n {
                    let (x, y) = ((i as f64 + 0.5) / n as f64 * span - span / 2.0, (j as f64 + 0.5) / n as f64 * span - span / 2.0);
                    let r = x.hypot(y);
                    if r >= span / 2.0 {
                        continue;
                    }
                    let c = cone(&v, at(&v, x, y), gsd, false).unwrap();
                    // depth below the flat rim
                    let rce = v.rc.max(1.5 * gsd);
                    let zrim = v.e + v.h * (1.0 - rce / v.r).powf(1.6);
                    if c.bowl {
                        s += (zrim - c.z) * (span / n as f64).powi(2);
                    }
                }
            }
            s
        };
        let (a, b) = (vol(10.0), vol(1000.0));
        assert!((a - b).abs() < 0.08 * a, "volume {a} vs {b}");
    }

    /// The cone profile and the gully term are what the explicit average says at coarse pixels:
    /// the gullies average out (zero-mean), the mean height over rings does not change.
    #[test]
    fn gullies_have_zero_mean() {
        let v = test_vol(0);
        let r = 0.3 * v.r;
        let n = 4000;
        let (mut fine, mut coarse) = (0.0, 0.0);
        for i in 0..n {
            let th = std::f64::consts::TAU * (i as f64 + 0.5) / n as f64;
            let p = at(&v, r * th.cos(), r * th.sin());
            fine += cone(&v, p, 1.0, true).unwrap().z;
            coarse += cone(&v, p, 2000.0, true).unwrap().z;
        }
        let (fine, coarse) = (fine / n as f64, coarse / n as f64);
        assert!((fine - coarse).abs() < 12.0, "ring mean fine {fine} coarse {coarse}");
    }

    /// Constants of the lattice's means against explicit averages (a different point set).
    #[test]
    fn field_means_match_explicit_averages() {
        let center = DVec3::new(3_000_000.0, 3_500_000.0, 4_200_000.0).normalize() * 6_371_000.0;
        let (e, n) = frame(center);
        for (cs, amp, ms) in [(700.0, 1.0, 0.0), (700.0, 1.0, 0.15), (900.0, 1.3, 0.0)] {
            let it = Instance { id: 0x1234_5678, center, v: [60000.0, cs as f32, amp as f32, 0.35, 0.3, 0.4, ms as f32, 0.0] };
            let span = 30000.0;
            let m = 100_000;
            let (mut zf, mut zc, mut cf, mut cc, mut lf, mut lc) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let mut dens = 0.0;
            for j in 0..m {
                let q = DVec2::new(crate::kernels::halton(j + 1, 2) - 0.5, crate::kernels::halton(j + 1, 3) - 0.5) * span;
                let p = center + e * q.x + n * q.y;
                let a = field(&it, p, 1.0, 1.0, true);
                // (the relief skips the cells that cannot reach the point: the same height)
                assert_eq!(a.z, field(&it, p, 1.0, 1.0, false).z, "cells skipped by the relief");
                let b = field(&it, p, 3000.0, 3000.0, true);
                zf += a.z;
                zc += b.z;
                cf += a.cinder;
                cc += b.cinder;
                lf += a.lava;
                lc += b.lava;
                dens += a.dens;
            }
            let k = m as f64;
            eprintln!(
                "cs {cs} ms {ms}: z {:.3} / {:.3} cinder {:.4} / {:.4} lava {:.4} / {:.4} dens {:.3}",
                zf / k,
                zc / k,
                cf / k,
                cc / k,
                lf / k,
                lc / k,
                dens / k
            );
            assert!((zf - zc).abs() / k < 0.04 * amp * cs / 700.0 + 0.1 * (zc / k).abs(), "height cs {cs}");
            assert!((cf - cc).abs() / k < 0.02 + 0.2 * cc / k, "cinder cs {cs}");
            assert!((lf - lc).abs() / k < 0.01 + 0.35 * lc / k, "lava cs {cs}");
        }
    }

    /// Places of the default world where the kit draws: (name, lat, lon), for the tests and stills.
    pub(crate) const PLACES: &[(&str, f64, f64)] = &[
        ("crater_lake", 11.850, -106.320),
        ("caldera_lake", -44.677, 9.339),
        ("active_boreal", 56.885, -101.647),
        ("cinder_field", 21.948, 71.408),
        ("island", 27.069, -174.181),
        ("shield_sea", 63.812, 4.905),
        ("caldera", -28.139, 42.765),
        ("shield_land", 3.806, -43.752),
        ("active", 14.069, -13.412),
    ];

    /// Debug: counts and places of the instances: `find_volcanic_places -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_volcanic_places() {
        use rayon::prelude::*;
        let w = crate::world::World::new(crate::Config::default());
        let ell = w.ell;
        let n = 400_000;
        let hits: Vec<(f64, f64, f64)> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let h = mix64(0x701C ^ i as u64);
                let lat = (2.0 * u01k(h, 1) - 1.0).asin().to_degrees();
                let lon = 360.0 * u01k(h, 2) - 180.0;
                let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &ell);
                let a = w.atlas().sample(p.normalize());
                (a.volcanism > 0.15).then_some((lat, lon, a.volcanism))
            })
            .collect();
        eprintln!("{} of {n} samples with volcanism > 0.15", hits.len());
        for (name, f) in [("volcanoes", fam_volcanoes()), ("shields", fam_shields()), ("fields", fam_fields())] {
            let mut seen = std::collections::BTreeMap::new();
            for &(lat, lon, _) in &hits {
                let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &ell);
                for i in crate::instances::near(&w, f, p, 0.0) {
                    seen.entry(i.id).or_insert(i);
                }
            }
            eprintln!("{name}: {} instances", seen.len());
            let mut groups: std::collections::BTreeMap<&str, Vec<String>> = Default::default();
            for it in seen.values() {
                let g = geodesy::ecef2geodetic(it.center, &ell);
                let v = it.v;
                let kind = match (name, v[7] as u32, v[1] > -150.0, v[5] > -1.0e5) {
                    ("fields", ..) => "field",
                    ("shields", _, true, _) => "shield_land",
                    ("shields", _, false, _) => "shield_sea",
                    (_, 1, _, true) => "caldera_lake",
                    (_, 1, _, false) => "caldera",
                    (_, _, true, true) => "crater_lake",
                    (_, _, true, false) => "strato_land",
                    (_, _, false, _) => "island",
                };
                let e = groups.entry(kind).or_default();
                if e.len() < 14 {
                    e.push(format!("({:.3}, {:.3}) R{:.0} E{:.0} H{:.0} act{:.1} T{:.0}", g.lat.to_degrees(), g.lon.to_degrees(), v[0], v[1], v[2], v[6], 0.0));
                }
            }
            for (k, v) in groups {
                eprintln!("  {k}: {}", v.join("; "));
            }
        }
    }

    fn tile_at(name: &str, z: u8) -> geodesy::tiles::TileId {
        let &(_, lat, lon) = PLACES.iter().find(|p| p.0 == name).unwrap();
        geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)
    }

    /// Share of samples differing by more than `tol`, and the mean difference.
    #[cfg(feature = "gpu")]
    fn off<T: Copy>(a: &[T], b: &[T], tol: f64, f: impl Fn(T) -> f64) -> (f64, f64) {
        let (mut bad, mut sum) = (0usize, 0.0);
        for (x, y) in a.iter().zip(b) {
            let e = (f(*x) - f(*y)).abs();
            sum += e;
            bad += (e > tol) as usize;
        }
        (bad as f64 / a.len() as f64, sum / a.len() as f64)
    }

    /// Parent ≈ mean of its children (elevation, albedo) where the kit draws.
    #[test]
    fn lod_consistent_at_volcanic_places() {
        let g = crate::NextGenerator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in [("crater_lake", 12u8), ("caldera_lake", 12), ("cinder_field", 13), ("island", 11)] {
            let parent = tile_at(name, z);
            let p = g.tile(parent);
            let kids = parent.children().map(|c| g.tile(c));
            let (mut eh, mut ea, mut cnt) = (0.0, 0.0, 0.0);
            for (k, kid) in kids.iter().enumerate() {
                let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
                for j in (0..n).step_by(2) {
                    for i in (0..n).step_by(2) {
                        let ix = [j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1];
                        let m: f32 = ix.iter().map(|&q| kid.elevation[q]).sum::<f32>() / 4.0;
                        let pi = (oy + j / 2) * n + ox + i / 2;
                        eh += (m - p.elevation[pi]).abs() as f64;
                        for c in 0..3 {
                            let ma: f64 = ix.iter().map(|&q| kid.albedo[3 * q + c] as f64).sum::<f64>() / 4.0;
                            ea += (ma - p.albedo[3 * pi + c] as f64).abs();
                        }
                        cnt += 1.0;
                    }
                }
            }
            let (eh, ea) = (eh / cnt, ea / (3.0 * cnt));
            eprintln!("{name} z{z}: |parent − mean(children)| elevation {eh:.2} m, albedo {ea:.2} DN");
            assert!(eh < 6.0, "{name}: elevation {eh}");
            assert!(ea < 12.0, "{name}: albedo {ea}");
        }
    }

    /// No seam between neighbouring tiles where the kit draws (E-W and N-S).
    #[test]
    fn seamless_at_volcanic_places() {
        let g = crate::NextGenerator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in [("crater_lake", 14u8), ("caldera_lake", 13), ("cinder_field", 15), ("active_boreal", 13)] {
            let id = tile_at(name, z);
            let a = g.tile(id);
            let (mut ea, mut ei, mut na, mut ni) = (0.0, 0.0, 0.0, 0.0);
            let e = g.tile(geodesy::tiles::TileId::new(z, id.x + 1, id.y));
            let s = g.tile(geodesy::tiles::TileId::new(z, id.x, id.y + 1));
            for j in 0..n {
                for c in 0..3 {
                    ea += (a.albedo[3 * (j * n + n - 1) + c] as f64 - e.albedo[3 * (j * n) + c] as f64).abs();
                    ei += (a.albedo[3 * (j * n + n - 2) + c] as f64 - a.albedo[3 * (j * n + n - 1) + c] as f64).abs();
                    na += (a.albedo[3 * ((n - 1) * n + j) + c] as f64 - s.albedo[3 * j + c] as f64).abs();
                    ni += (a.albedo[3 * ((n - 2) * n + j) + c] as f64 - a.albedo[3 * ((n - 1) * n + j) + c] as f64).abs();
                    // (and the neighbours' first two rows / columns: a ravine at the seam is no seam)
                    ei += (e.albedo[3 * (j * n) + c] as f64 - e.albedo[3 * (j * n + 1) + c] as f64).abs();
                    ni += (s.albedo[3 * j + c] as f64 - s.albedo[3 * (n + j) + c] as f64).abs();
                }
                // elevation: the step across the seam is no larger than the slope inside (+ a cinder cone: 12 m)
                let (h0, h1, h2) = (a.elevation[j * n + n - 2] as f64, a.elevation[j * n + n - 1] as f64, e.elevation[j * n] as f64);
                assert!((h2 - h1).abs() < 3.0 * (h1 - h0).abs() + 16.0, "{name}: E-W elevation seam {h0} {h1} {h2}");
                let (v0, v1, v2) = (a.elevation[(n - 2) * n + j] as f64, a.elevation[(n - 1) * n + j] as f64, s.elevation[j] as f64);
                assert!((v2 - v1).abs() < 3.0 * (v1 - v0).abs() + 16.0, "{name}: N-S elevation seam {v0} {v1} {v2}");
            }
            eprintln!("{name} z{z}: albedo E-W across {ea:.0} inside {ei:.0}; N-S across {na:.0} inside {ni:.0}");
            assert!(ea < 2.0 * ei + 2.0 * n as f64, "{name}: E-W across {ea} inside {ei}");
            assert!(na < 2.0 * ni + 2.0 * n as f64, "{name}: N-S across {na} inside {ni}");
        }
    }

    /// Tiles are a pure function of the world: the same tile twice is the same bits.
    #[test]
    fn deterministic_at_volcanic_places() {
        let cfg = crate::Config { tile_supersample: 1, ..crate::Config::default() };
        let a = crate::NextGenerator::new(cfg.clone());
        let b = crate::NextGenerator::new(cfg);
        for (name, z) in [("crater_lake", 14u8), ("cinder_field", 15)] {
            let id = tile_at(name, z);
            let (x, y) = (a.tile(id), b.tile(id));
            assert!(x.elevation == y.elevation && x.albedo == y.albedo && x.landcover == y.landcover, "{name}");
        }
    }

    /// The GPU's tiles are the CPU's where the kit draws.
    #[cfg(feature = "gpu")]
    #[test]
    fn tiles_match_the_cpu() {
        let cfg = crate::Config::default();
        let gen = match crate::gpu::GpuGenerator::new(cfg.clone()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("no GPU generator ({e}): skipped");
                return;
            }
        };
        let cpu = crate::NextGenerator::new(cfg);
        for (name, z) in [
            ("crater_lake", 14u8),
            ("crater_lake", 17),
            ("caldera_lake", 13),
            ("caldera_lake", 15),
            ("active_boreal", 14),
            ("cinder_field", 14),
            ("cinder_field", 16),
            ("island", 12),
            ("shield_sea", 12),
        ] {
            let id = tile_at(name, z);
            let g = gen.tiles(&[id]).unwrap().remove(0);
            let c = cpu.tile_cpu(id);
            let rgb = off(&g.rgb, &c.rgb, 2.0, |v| v as f64);
            let el = off(&g.elevation, &c.elevation, 0.05, |v| v as f64);
            let lc = off(&g.landcover, &c.landcover, 0.5, |v| v as f64);
            eprintln!("{name} z{z}: rgb off {:.3}% elevation off {:.3}% landcover off {:.3}%", rgb.0 * 100.0, el.0 * 100.0, lc.0 * 100.0);
            assert!(rgb.0 < 0.005 && el.0 < 0.005 && lc.0 < 0.005, "{name} z{z}");
        }
    }

    /// Every place generates on the GPU at all zooms (no CPU fallback: an overflowing family, a
    /// shader error): `Backend::Gpu` returns the error.
    #[cfg(feature = "gpu")]
    #[test]
    fn places_stay_on_the_gpu() {
        let Ok(g) = crate::gpu::shared() else { return };
        if g.check_generator().is_err() {
            return;
        }
        let gen = crate::NextGenerator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        for &(name, lat, lon) in PLACES {
            let ids: Vec<_> = [6u8, 8, 10, 12, 14, 16].iter().map(|&z| geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)).collect();
            if let Err(e) = gen.tiles(&ids) {
                panic!("{name}: GPU generation failed: {e:#}");
            }
        }
        assert!(gen.backend_name().starts_with("GPU"), "{}", gen.backend_name());
    }

    /// GPU tiles/s at the kit's places (8 x 8 tiles per place and zoom; performance budget):
    /// `cargo test -p terragen --release --lib kits::volcanic::tests::bench_places -- --ignored --nocapture`.
    /// For the baseline, run it with the kit's `KITS` entry commented out (the module stays).
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn bench_places() {
        let gen = crate::NextGenerator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        gen.tiles(&[geodesy::tiles::TileId::new(3, 4, 3)]).unwrap();
        for z in [8u8, 10, 12, 14, 16] {
            // three blocks of 8 x 8 tiles per place (neighbouring blocks: no cached lattice), the median
            let mut rates = Vec::new();
            for rep in 0..3i64 {
                let mut ids = Vec::new();
                for &(_, lat, lon) in PLACES {
                    let c = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
                    let n = 1i64 << z;
                    for dy in 0..8i64 {
                        for dx in 0..8i64 {
                            ids.push(geodesy::tiles::TileId::new(
                                z,
                                (c.x as i64 + dx - 4 + 8 * rep).rem_euclid(n) as u32,
                                (c.y as i64 + dy - 4).clamp(0, n - 1) as u32,
                            ));
                        }
                    }
                }
                let t0 = std::time::Instant::now();
                for chunk in ids.chunks(16) {
                    gen.tiles(chunk).unwrap();
                }
                rates.push(ids.len() as f64 / t0.elapsed().as_secs_f64());
            }
            rates.sort_by(|a, b| a.total_cmp(b));
            println!(
                "bench z{z}: {} tiles per block, median {:.1} tiles/s (blocks: {:.1} {:.1} {:.1})",
                64 * PLACES.len(),
                rates[1],
                rates[0],
                rates[1],
                rates[2]
            );
        }
        println!("backend: {}", gen.backend_name());
    }
}
