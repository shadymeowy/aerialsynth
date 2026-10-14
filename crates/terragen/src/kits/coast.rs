//! The coast and ocean kit (`docs/design/terrain-next.md` §5.5, `docs/next/kits/coast.md`): cliffs with
//! wave-cut platforms, rocky shores and skerries, tidal flats with salt marsh and dendritic
//! creeks, barrier islands and lagoons, fringing and barrier reefs, salt and aquaculture ponds,
//! archipelagos. The WGSL twins are `gpu/wgsl/kits/coast_relief.wgsl` (types, creeks, relief) and
//! `gpu/wgsl/kits/coast.wgsl` (the layers): the same functions in the same order of operations.
//!
//! **Coast types** are drawn per ~50 km stretch ([`types`]): three noises (55, 38 and 47 km) ×
//! the relief potential (tectonic uplift, mountains) × glaciation × the sea-level temperature
//! give the weights of cliff, skerry, tidal-flat, barrier and reef coasts. Every effect is a
//! smooth function of the pre-remap height `h0` near sea level (no distance to the coast), so it
//! costs no more than a few noises in a band of heights near the sea.
//!
//! **Relief (pass A, [`relief`])**, by `h0`:
//! * cliffs: `h' = h + C·smoothstep(0, ε, h)` fading out above 1.5..4 C (C 12–112 m, ε =
//!   max(2 m, 0.04 gsd)), and a wave-cut platform below: `h' → −1.3 + 0.1 h` in the first 10 m of
//!   depth;
//! * skerries: noise of 700, 230 and 75 m × a window around −8 m: islets and coves of a low rocky
//!   coast; archipelagos: the same at 2.6 km and a wider window;
//! * tidal flats: the intertidal band is compressed to a fifth of its slope (flat mud at
//!   0..1 m, marsh above), cut by dendritic creeks ([`creek`]: three levels of ridged noise);
//! * barrier islands: a contour bump at −6 m with inlets where a 3.2 km noise closes it;
//! * reefs: a contour bump at −7 m (fringing) or −28 m (barrier reef) up to −0.35 m with passes.
//!
//! **Pass B.** Slot 3 (azonal): cliff faces (strata by lithology), rocky shores, flat mud, marsh,
//! creeks, salt and aquaculture ponds in a rotated grid. Slot 8 (water): reef lagoons (turquoise
//! sand, coral patches), barrier lagoons, turbid flat water, kelp, the shallow platform.

use crate::instances::{Family, InstSite, Instance};
use crate::kits::{Kit, ReliefIn};
use crate::noise::*;
use crate::stack::{slot, Layer, Stack};
use crate::surface::{mixc, srgb};
use glam::DVec3;

pub static KIT: Kit = Kit {
    name: "coast",
    biomes_yaml: "",
    wgsl: include_str!("../gpu/wgsl/kits/coast.wgsl"),
    layers: &[(slot::AZONAL, azonal), (slot::AGRICULTURE, agriculture), (slot::WATER, water)],
    relief: Some(relief),
    relief_wgsl: include_str!("../gpu/wgsl/kits/coast_relief.wgsl"),
    families: &[Family { name: "coast_atolls", cell: 80_000.0, jitter: 0.7, reach: 24_000.0, relief: true, exists: atoll_exists }],
    astamps: Some(delta::astamps),
    astamp: Some(delta::relief),
    host: Some(delta::host),
    ..Kit::EMPTY
};

pub(crate) mod delta;

/// Land-cover v2 class ids (`registry::classes`; checked by a test).
mod cls {
    pub const OCEAN: u8 = 1;
    pub const ROCK: u8 = 6;
    pub const AQUACULTURE: u8 = 23;
    pub const TIDAL_FLAT: u8 = 24;
    pub const CORAL_REEF: u8 = 25;
    pub const CLIFF: u8 = 37;
    pub const MARSH: u8 = 59;
}

const SEED: u64 = 0xC0A5_7000;

/// Noise of wavelength `lam` at `p`, band-limited.
#[inline]
fn nz(seed: u64, p: DVec3, lam: f64, gsd: f64) -> f64 {
    let b = band(lam, gsd);
    if b > 0.0 {
        perlin3(seed, p / lam) * b
    } else {
        0.0
    }
}

#[inline]
fn cov(d: f64, hw: f64, fw: f64) -> f64 {
    ((hw - d.abs()) / fw + 0.5).clamp(0.0, 1.0)
}

// ============================================================================= coast types

/// The weights (0..1) of the coast types at a point.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Types {
    pub cliff: f64,
    pub skerry: f64,
    pub flat: f64,
    pub barrier: f64,
    pub reef: f64,
    pub arch: f64,
    /// 0: fringing reef (−7 m), 1: barrier reef (−28 m)
    pub far: f64,
}

/// The coast type at `p` from the atlas fields there: tectonic uplift, mountain mask,
/// glaciation, sea-level temperature (°C).
pub(crate) fn types(p: DVec3, uplift: f64, mountain: f64, glac: f64, temp: f64) -> Types {
    let u = perlin3(SEED + 1, p / 55_000.0);
    let v = perlin3(SEED + 2, p / 38_000.0);
    let w = perlin3(SEED + 3, p / 47_000.0);
    let relp = smoothstep(0.05, 0.4, uplift).max(mountain);
    let gl = smoothstep(0.3, 0.6, glac);
    let cliff_s = relp * smoothstep(-0.3, 0.3, u);
    let skerry_s = gl * (0.6 + 0.4 * smoothstep(-0.4, 0.3, v));
    let flat_s = (1.0 - relp) * (1.0 - gl) * smoothstep(0.05, 0.4, v);
    let barrier_s = (1.0 - relp) * (1.0 - gl) * smoothstep(-0.05, -0.4, v);
    let mx = cliff_s.max(skerry_s).max(flat_s).max(barrier_s).max(1e-4);
    let gate = smoothstep(0.0, 0.2, mx);
    let pick = |s: f64| smoothstep(0.7, 1.0, s / mx) * gate;
    let cliff = pick(cliff_s);
    Types {
        cliff,
        skerry: pick(skerry_s),
        flat: pick(flat_s),
        barrier: pick(barrier_s),
        reef: smoothstep(22.0, 25.0, temp) * smoothstep(-0.5, 0.0, w),
        arch: smoothstep(0.15, 0.5, uplift) * smoothstep(-0.1, 0.3, -u) * (1.0 - cliff),
        far: smoothstep(0.1, 0.3, perlin3(SEED + 4, p / 60_000.0)),
    }
}

// ============================================================================= creeks

/// Mean cover of the main channels (what a coarse pixel shows).
const CREEK_MEAN: f64 = 0.055;

/// Dendritic tidal creeks: coverage 0..1 at `p` (three levels of ridged noise: main channels
/// 20 m wide, branches 8 m, rills 3 m; the finer levels only near the coarser, each warped by
/// the level above), as crisp as the filter width `fw` allows.
pub(crate) fn creek(p: DVec3, gsd: f64, fw: f64) -> f64 {
    // (a fade by the pixel size, to the mean cover of the main channels: no tile-sized steps)
    let b1 = band(420.0, gsd);
    if b1 <= 0.0 {
        return CREEK_MEAN;
    }
    let n1 = perlin3(SEED + 10, p / 420.0);
    let d1 = n1.abs() * 420.0 / 1.3;
    let mut c = b1 * cov(d1, 7.0, fw) + (1.0 - b1) * CREEK_MEAN;
    if gsd < 37.0 {
        let q = p / 150.0 + DVec3::new(0.35, 0.21, -0.28) * n1;
        let n2 = perlin3(SEED + 11, q);
        let d2 = n2.abs() * 150.0 / 1.3;
        c = c.max(band(150.0, gsd) * cov(d2, 3.0, fw) * (1.0 - smoothstep(40.0, 130.0, d1)));
        if gsd < 14.0 {
            let q = p / 55.0 + DVec3::new(0.3, -0.2, 0.33) * n2;
            let n3 = perlin3(SEED + 12, q);
            let d3 = n3.abs() * 55.0 / 1.3;
            c = c.max(0.9 * band(55.0, gsd) * cov(d3, 1.2, fw) * (1.0 - smoothstep(20.0, 55.0, d1.min(d2))));
        }
    }
    c
}

// ============================================================================= relief

/// The change of the height `h0` (m) by the coast types, at `p` with pixel size `gsd`.
pub(crate) fn coast_dh(t: &Types, h0: f64, p: DVec3, gsd: f64) -> f64 {
    let mut dh = 0.0;
    // ---- cliffs and the wave-cut platform
    if t.cliff > 0.0 && h0 > -12.0 && h0 < 500.0 {
        let n = perlin3(SEED + 5, p / 30_000.0);
        let c = 12.0 + 100.0 * smoothstep(-0.4, 0.6, n);
        let eps = 2.0f64.max(0.04 * gsd);
        let lift = c * smoothstep(0.0, eps, h0) * (1.0 - smoothstep(1.5 * c, 4.0 * c, h0));
        let u = smoothstep(-10.0, -1.5, h0) * smoothstep(0.3, -0.3, h0);
        let plat = (-1.3 + 0.1 * h0 - h0) * u;
        dh += t.cliff * (lift + plat);
    }
    // ---- skerries and archipelagos
    if t.skerry > 0.0 && h0 > -60.0 && h0 < 40.0 {
        let g = 1.0 - smoothstep(18.0, 55.0, (h0 + 8.0).abs());
        if g > 0.0 {
            let n = 17.0 * nz(SEED + 20, p, 700.0, gsd) + 8.0 * nz(SEED + 21, p, 230.0, gsd) + 3.0 * nz(SEED + 22, p, 75.0, gsd);
            dh += t.skerry * g * (n + 4.0);
        }
    }
    if t.arch > 0.0 && h0 > -120.0 && h0 < 60.0 {
        let g = 1.0 - smoothstep(30.0, 90.0, (h0 + 25.0).abs());
        if g > 0.0 {
            let n = 38.0 * nz(SEED + 23, p, 2600.0, gsd) + 14.0 * nz(SEED + 24, p, 900.0, gsd) + 5.0 * nz(SEED + 25, p, 260.0, gsd);
            dh += t.arch * g * (n + 8.0);
        }
    }
    // ---- tidal flats: the intertidal band flattened, cut by creeks
    if t.flat > 0.0 && h0 > -16.0 && h0 < 16.0 {
        let e = 1.0 - smoothstep(6.0, 14.0, (h0 - 1.0).abs());
        let h1 = lerp(h0, 0.2 * h0 + 0.4, e);
        let win = (1.0 - smoothstep(1.6, 3.2, h1)) * smoothstep(-5.0, -2.0, h1);
        let carve = if win > 0.0 { 1.9 * creek(p, gsd, gsd) * win } else { 0.0 };
        dh += t.flat * (h1 - h0 - carve);
    }
    // ---- barrier islands: a bump at −6 m, open at inlets
    if t.barrier > 0.0 && h0 > -14.0 && h0 < 2.0 {
        let open = smoothstep(-0.25, 0.1, nz(SEED + 30, p, 3200.0, gsd));
        let x = (h0 + 6.0) / 2.8;
        dh += t.barrier * 8.7 * open * (-x * x).exp();
    }
    // ---- reefs: a bump on the −7 m (fringing) or −28 m (barrier reef) contour, up to −0.35 m
    if t.reef > 0.0 && h0 > -50.0 && h0 < 2.0 {
        let hb = lerp(-7.0, -28.0, t.far);
        let wd = lerp(2.4, 9.0, t.far);
        let gap = smoothstep(-0.35, 0.05, nz(SEED + 31, p, lerp(900.0, 1800.0, t.far), gsd));
        let x = (h0 - hb) / wd;
        dh += t.reef * (-hb - 0.9) * gap * (-x * x).exp();
    }
    dh
}

fn relief(w: &crate::world::World, r: &ReliefIn, h: &mut f64) {
    atolls_relief(w, r, h);
    let m = r.m;
    if m.coast_km > 60.0 || m.coast_km < -100.0 {
        return;
    }
    let h0 = *h;
    if !(-120.0..=500.0).contains(&h0) {
        return;
    }
    let t = types(r.ctx.p, m.uplift, r.mountain, m.glaciation, m.temp);
    *h += coast_dh(&t, h0, r.ctx.p, r.ctx.gsd);
}

// ============================================================================= atolls

fn fam_atolls() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| crate::instances::families().position(|f| f.name == "coast_atolls").unwrap())
}

/// Atolls: rings of reef around lagoons in the open tropical ocean (deep sea floor, far from any
/// coast, warmer than 25 °C; mostly on hotspot tracks). v: ring radius, rim half width, lagoon
/// depth, motu share.
fn atoll_exists(w: &crate::world::World, s: &InstSite) -> Option<[f32; 8]> {
    if u01k(s.id, 1) >= 0.25 {
        return None;
    }
    let a = w.atlas().sample(s.center.normalize());
    if !(-5500.0..=-1500.0).contains(&a.elevation_m) || a.coast_km > -150.0 || a.temp_c < 25.0 {
        return None;
    }
    let hot = a.hotspot || a.volcanism > 0.12;
    if u01k(s.id, 2) >= if hot { 0.9 } else { 0.35 } {
        return None;
    }
    let f = |x: f64| x as f32;
    Some([
        f(2000.0 + 12_000.0 * u01k(s.id, 3).powf(1.5)),
        f(180.0 + 220.0 * u01k(s.id, 4)),
        f(12.0 + 18.0 * u01k(s.id, 5)),
        f(0.1 + 0.35 * u01k(s.id, 6)),
        0.0,
        0.0,
        0.0,
        0.0,
    ])
}

/// The east and north axes of an instance's frame (at its centre).
fn frame(center: DVec3) -> (DVec3, DVec3) {
    let up = center.normalize();
    let e = if up.z.abs() > 0.999 { DVec3::X } else { DVec3::Z.cross(up).normalize() };
    (e, up.cross(e))
}

/// An atoll at a point: the profile height (m), the signed distance from the ring (m), the
/// rim half width after widening to the pixel, the motu cover.
#[derive(Clone, Copy, Debug)]
pub(crate) struct At {
    pub z: f64,
    pub d: f64,
    pub wre: f64,
}

pub(crate) fn atoll_at(it: &Instance, p: DVec3, gsd: f64) -> Option<At> {
    let v = it.v.map(|x| x as f64);
    let (r0, wr, l) = (v[0], v[1], v[2]);
    let (e, n) = frame(it.center);
    let rel = p - it.center;
    let (x, y) = (rel.dot(e), rel.dot(n));
    let r = x.hypot(y);
    if r > 1.2 * r0 + 8000.0 + 3.0 * gsd {
        return None;
    }
    let th = y.atan2(x);
    let tau = std::f64::consts::TAU;
    let edge = 1.0
        + 0.10 * (3.0 * th + tau * u01k(it.id, 10)).sin()
        + 0.06 * (5.0 * th + tau * u01k(it.id, 11)).sin()
        + 0.04 * (8.0 * th + tau * u01k(it.id, 12)).sin();
    let d = r / edge - r0;
    let wre = wr.max(1.5 * gsd);
    if d > wre + 7000.0 {
        return None;
    }
    let ad = d.abs();
    let z = if d < -wre {
        let t = smoothstep(0.0, 1.0, (-d - wre) / (350.0 + 0.04 * r0));
        -l + (l - 0.3) * (1.0 - t)
    } else if d <= wre {
        let thr = 0.45 - v[3];
        let motu = smoothstep(thr, thr + 0.12, nz(SEED ^ it.id ^ 0x80, p, 700.0, gsd));
        -0.3 + 2.2 * motu * (1.0 - smoothstep(0.4 * wre, wre, ad))
    } else {
        -0.3 - 0.5 * (d - wre)
    };
    Some(At { z, d, wre })
}

fn atolls_relief(w: &crate::world::World, r: &ReliefIn, h: &mut f64) {
    for it in r.instances(w, fam_atolls()).iter() {
        if let Some(a) = atoll_at(it, r.ctx.p, r.ctx.gsd) {
            let k = 1.0 - smoothstep(5500.0, 7000.0, a.d - a.wre);
            *h += (h.max(a.z) - *h) * k;
        }
    }
}

// ============================================================================= pass B

/// Slot 3: cliff faces, rocky shores, tidal flats, marsh, creeks, ponds.
pub fn azonal(s: &mut Stack) {
    let t = s.t;
    let ground = s.l.ground;
    if ground > -0.3 && ground < 4.0 && t.coast_km < -50.0 {
        motus(s);
    }
    if t.coast_km > 60.0 || t.coast_km < -40.0 || ground > 500.0 || ground < -1.0 {
        return;
    }
    let ty = types(s.ctx.p, t.uplift, t.mountain, t.glaciation, t.temp);
    if ty.cliff > 0.0 {
        cliff_face(s, &ty);
    }
    if ty.skerry > 0.0 {
        rocky_shore(s, &ty);
    }
    if ty.flat > 0.0 && ground < 3.6 {
        flats(s, &ty);
    }
}

/// Scrub on the motus of the atolls (the core's beach is the sand).
fn motus(s: &mut Stack) {
    let (p, gsd, ground) = (s.ctx.p, s.ctx.gsd, s.l.ground);
    let inst = s.instances(fam_atolls()).into_owned();
    for it in inst.iter() {
        if atoll_at(it, p, gsd).is_some_and(|a| a.d.abs() < a.wre + 20.0) {
            let n = nz(SEED + 45, p, 14.0, gsd);
            let sand = srgb(236.0, 226.0, 194.0) * (1.0 + 0.04 * n);
            s.composite(Layer { cov: 1.0, albedo: sand, cls: 4, clear: 1.0, ..Default::default() });
            let c = smoothstep(0.7, 1.4, ground + 0.5 * n) * 0.85;
            if c > 0.0 {
                let col = mixc(srgb(60.0, 104.0, 52.0), srgb(86.0, 120.0, 62.0), 0.5 + 0.5 * n);
                s.composite(Layer { cov: c, albedo: col, cls: 9, clear: 1.0, ..Default::default() });
            }
            return;
        }
    }
}

/// Rock faces of the cliff coasts: strata by lithology, streaked.
fn cliff_face(s: &mut Stack, ty: &Types) {
    let (p, gsd) = (s.ctx.p, s.ctx.gsd);
    let ground = s.l.ground;
    let slope = s.l.slope;
    let c = ty.cliff * smoothstep(0.4, 0.75, slope) * (1.0 - smoothstep(300.0, 500.0, ground));
    if c <= 0.0 {
        return;
    }
    let base = match s.bio.style.litho {
        1 => srgb(206.0, 200.0, 182.0),
        2 => srgb(112.0, 106.0, 100.0),
        3 => srgb(56.0, 50.0, 48.0),
        4 => srgb(176.0, 148.0, 112.0),
        _ => srgb(150.0, 98.0, 72.0),
    };
    let st = 0.5 + 0.5 * (ground * std::f64::consts::TAU / 3.4 + 2.0 * nz(SEED + 40, p, 60.0, gsd)).sin() * band(3.4 / (0.1 + slope), gsd);
    let streak = nz(SEED + 41, p, 18.0, gsd);
    let col = base * (0.86 + 0.2 * st + 0.1 * streak);
    s.composite(Layer { cov: c, albedo: col, cls: cls::CLIFF, clear: 1.0, ..Default::default() });
    s.m.natural_ok *= 1.0 - c;
}

/// Bare rock on the islets and low shores of glaciated coasts.
fn rocky_shore(s: &mut Stack, ty: &Types) {
    let (p, gsd) = (s.ctx.p, s.ctx.gsd);
    let ground = s.l.ground;
    let m = ty.skerry * (1.0 - smoothstep(4.0, 16.0, ground + 4.0 * nz(SEED + 42, p, 120.0, gsd)));
    if m <= 0.0 {
        return;
    }
    let rock = mixc(s.pal(crate::registry::pal::ROCK), s.pal(crate::registry::pal::ROCK + 1), 0.5 + 0.5 * nz(SEED + 43, p, 25.0, gsd));
    let lichen = 0.5 + 0.5 * nz(SEED + 44, p, 6.0, gsd);
    let col = mixc(rock, srgb(104.0, 112.0, 80.0), 0.25 * lichen * smoothstep(1.0, 6.0, ground))
        * (1.0 + 0.16 * nz(SEED + 46, p, 5.0, gsd) + 0.1 * nz(SEED + 47, p, 1.6, gsd));
    s.composite(Layer { cov: m, albedo: col, cls: cls::ROCK, clear: 1.0, ..Default::default() });
    s.m.natural_ok *= 1.0 - m;
}

/// Mud, marsh, creeks and ponds of the tidal-flat coasts (ground 0..3.6 m).
fn flats(s: &mut Stack, ty: &Types) {
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, s.l.fw);
    let ground = s.l.ground;
    let wet = 1.0 - smoothstep(0.15, 0.7, ground);
    let high = smoothstep(1.0, 1.9, ground + 0.25 * nz(SEED + 50, p, 30.0, gsd));
    let zone = ty.flat * (1.0 - smoothstep(2.8, 3.6, ground));
    if zone <= 0.0 {
        return;
    }
    s.no_town = true;
    let ripple = nz(SEED + 51, p, 14.0, gsd);
    let mud = mixc(srgb(122.0, 108.0, 90.0), srgb(86.0, 78.0, 68.0), wet) * (1.0 + 0.07 * ripple);
    let marsh = mixc(srgb(92.0, 104.0, 64.0), srgb(128.0, 96.0, 66.0), smoothstep(0.1, 0.6, nz(SEED + 52, p, 70.0, gsd)));
    let marsh = marsh * (1.0 + 0.1 * nz(SEED + 53, p, 7.0, gsd));
    let col = mixc(mud, marsh, high);
    s.composite(Layer { cov: zone, albedo: col, cls: if high > 0.5 { cls::MARSH } else { cls::TIDAL_FLAT }, clear: 1.0, ..Default::default() });
    s.m.natural_ok *= 1.0 - zone;
    // creeks: dark wet channels (the pixel's own carving shows as shallow water)
    let win = 1.0 - smoothstep(1.6, 3.2, ground);
    if win > 0.0 {
        let c = creek(p, gsd, fw.max(0.3 * gsd)) * zone * win;
        if c > 0.0 {
            let cc = mixc(srgb(62.0, 58.0, 48.0), srgb(52.0, 56.0, 40.0), high);
            s.composite(Layer { cov: c, albedo: cc, cls: cls::TIDAL_FLAT, clear: 1.0, ..Default::default() });
        }
    }
    ponds(s, zone);
}

/// Slot 5: the cover of the deltas' plains (`delta`): reed marsh, mud where it is arid or
/// lowest, with salt crusts, tidal creeks and scrub on the distributaries' levees, mottled and
/// pulled towards the land's own colour where the plain thins out (its margin breaks into
/// patches); fields stay where the land use draws them (before this layer), towns too, and the
/// tropical biomes' mangroves cover theirs (canopy slot).
pub fn agriculture(s: &mut Stack) {
    let ground = s.l.ground;
    if !(0.0..3.0).contains(&ground) || s.l.water > ground {
        return;
    }
    let t = s.t;
    if t.temp >= 20.0 && t.moist >= 0.4 {
        return;
    }
    let (p, gsd) = (s.ctx.p, s.ctx.gsd);
    let (mut plain, mut lev) = (0.0f32, 0.0f32);
    for st in s.stamps() {
        if st.template == delta::TEMPLATE {
            let id = st.v[2] as u64 | (st.v[3] as u64) << 24 | (st.v[4] as u64) << 48;
            let sh = delta::shape(st.center, st.v[0] as f64, st.v[1], id, st.h, p, gsd as f32);
            plain = plain.max(sh.plain);
            lev = lev.max(sh.lev);
        }
    }
    let plain = plain as f64;
    // (a ragged margin: patches of the plain's cover, not a line)
    let ragged = 0.6 * nz(SEED + 63, p, 170.0, gsd) + 0.4 * nz(SEED + 64, p, 52.0, gsd);
    // (over the core's beach, which the low land near the sea gets; not over fields)
    let zone = plain * smoothstep(-0.2, 0.2, plain - 0.5 + 0.5 * ragged) * (1.0 - smoothstep(2.4, 3.0, ground)) * (1.0 - s.m.field_cov);
    if zone <= 0.0 {
        return;
    }
    let arid = 1.0 - smoothstep(0.2, 0.34, t.moist);
    let low = 1.0 - smoothstep(0.6, 1.1, ground + 0.3 * nz(SEED + 60, p, 90.0, gsd));
    // grey silt drying out to salt crusts on the higher flats, darker where wet
    let silt = srgb(144.0, 134.0, 114.0) * (1.0 + 0.1 * nz(SEED + 65, p, 38.0, gsd));
    let salt = smoothstep(0.2, 0.45, nz(SEED + 66, p, 120.0, gsd) + 0.5 * nz(SEED + 67, p, 33.0, gsd)) * (1.0 - low);
    let dry = mixc(silt, srgb(192.0, 186.0, 170.0), 0.7 * salt);
    let mud = mixc(mixc(srgb(118.0, 106.0, 88.0), srgb(84.0, 76.0, 66.0), low), mixc(dry, srgb(96.0, 90.0, 78.0), low), arid);
    let reed = mixc(srgb(96.0, 106.0, 62.0), srgb(134.0, 112.0, 70.0), smoothstep(0.0, 0.7, nz(SEED + 61, p, 260.0, gsd)));
    let m = (1.0 - arid) * (1.0 - low);
    let mut col = mixc(mud, reed, m) * (1.0 + 0.08 * nz(SEED + 62, p, 9.0, gsd));
    // scrub and reeds on the levees (tamarisk in the arid deltas)
    let scrub = lev as f64 * (0.45 + 0.4 * smoothstep(-0.2, 0.3, nz(SEED + 68, p, 24.0, gsd)));
    col = mixc(col, mixc(srgb(70.0, 84.0, 48.0), srgb(84.0, 88.0, 58.0), arid), scrub);
    // the land's own colour where the plain thins out
    col = mixc(col, s.col, 0.3 + 0.45 * (1.0 - plain));
    s.composite(Layer { cov: zone, albedo: col, cls: if m > 0.5 { cls::MARSH } else { cls::TIDAL_FLAT }, clear: 1.0, ..Default::default() });
    s.m.natural_ok *= 1.0 - zone;
    // tidal creeks in the low mud
    let c = creek(p, gsd, s.l.fw.max(0.3 * gsd)) * zone * (0.3 + 0.7 * low) * (1.0 - scrub);
    if c > 0.0 {
        s.composite(Layer { cov: c, albedo: srgb(70.0, 66.0, 56.0), cls: cls::TIDAL_FLAT, clear: 1.0, ..Default::default() });
    }
}

/// Salt (arid) and aquaculture (humid) ponds in the field frame of the land-use region.
fn ponds(s: &mut Stack, zone: f64) {
    let t = s.t;
    let ground = s.l.ground;
    if t.temp < 14.0 || ground < 0.3 || ground > 3.2 || zone < 0.5 {
        return;
    }
    let (p, gsd, fw) = (s.ctx.p, s.ctx.gsd, s.l.fw);
    let area = smoothstep(-0.2, 0.1, perlin3(SEED + 60, p / 2400.0));
    if area <= 0.0 {
        return;
    }
    let q = if s.region.is_some() { s.q_rot } else { s.q_loc };
    let (ca, cb) = (150.0, 85.0);
    let (ix, iy) = ((q.x / ca).floor(), (q.y / cb).floor());
    let (fx, fy) = (q.x - (ix + 0.5) * ca, q.y - (iy + 0.5) * cb);
    let h = hash2(SEED + 61, ix as i64, iy as i64);
    let salt = t.moist < 0.45;
    let mean_col = if salt { srgb(184.0, 150.0, 140.0) } else { srgb(88.0, 130.0, 116.0) };
    // (ponds smaller than the pixel show their mean cover and colour)
    let e = smoothstep(1.2 * gsd, 3.0 * gsd, cb);
    let (mut exact, mut col) = (0.0, mean_col);
    if e > 0.0 && u01k(h, 1) < 0.7 {
        let f = fw.max(0.3 * gsd);
        let (hx, hy) = (0.5 * ca - 4.0, 0.5 * cb - 4.0);
        exact = ((hx - fx.abs()) / f + 0.5).clamp(0.0, 1.0).min(((hy - fy.abs()) / f + 0.5).clamp(0.0, 1.0));
        let pick = u01k(h, 2);
        col = if salt {
            if pick < 0.3 {
                srgb(206.0, 120.0, 132.0)
            } else if pick < 0.5 {
                srgb(232.0, 214.0, 208.0)
            } else if pick < 0.75 {
                srgb(78.0, 166.0, 168.0)
            } else {
                srgb(96.0, 150.0, 98.0)
            }
        } else if pick < 0.4 {
            srgb(70.0, 140.0, 130.0)
        } else if pick < 0.75 {
            srgb(52.0, 110.0, 100.0)
        } else {
            srgb(104.0, 112.0, 84.0)
        };
    }
    let c = area * zone * (e * exact + (1.0 - e) * 0.7 * 0.62);
    if c > 0.0 {
        s.composite(Layer { cov: c, albedo: mixc(mean_col, col, e), cls: cls::AQUACULTURE, clear: 1.0, relit: 0.6, ..Default::default() });
        s.m.natural_ok *= 1.0 - c;
    }
}

/// Slot 8: the colours of shallow coastal water.
pub fn water(s: &mut Stack) {
    let l = s.l;
    if l.water <= l.ground || l.water_kind != crate::world::water::OCEAN {
        return;
    }
    let depth = l.water - l.ground;
    let t = s.t;
    if depth > 45.0 {
        return;
    }
    let (p, gsd) = (s.ctx.p, s.ctx.gsd);
    let mut atoll = 0.0;
    let inst = s.instances(fam_atolls()).into_owned();
    for it in inst.iter() {
        if atoll_at(it, p, gsd).is_some_and(|a| a.d < a.wre + 300.0) {
            atoll = 1.0;
        }
    }
    let coastal = t.coast_km <= 40.0 && t.coast_km >= -120.0;
    if !coastal && atoll == 0.0 {
        return;
    }
    let ty = if coastal { types(p, t.uplift, t.mountain, t.glaciation, t.temp) } else { Types::default() };
    // ---- reef lagoons: turquoise sand, coral heads
    if ty.reef > 0.0 || atoll > 0.0 {
        let a = ty.reef.max(atoll) * (1.0 - smoothstep(18.0, 40.0, depth));
        if a > 0.0 {
            let sand = mixc(srgb(150.0, 214.0, 200.0), srgb(36.0, 168.0, 180.0), smoothstep(1.5, 14.0, depth));
            let n = 0.6 * nz(SEED + 70, p, 26.0, gsd) + 0.4 * nz(SEED + 71, p, 7.0, gsd);
            let wd = 0.06 + 0.3 * (1.0 - band(26.0, gsd));
            let coral = smoothstep(0.12 - wd, 0.12 + wd, n) * smoothstep(0.6, 1.8, depth) * (1.0 - smoothstep(10.0, 20.0, depth));
            let cc = mixc(srgb(88.0, 98.0, 62.0), srgb(122.0, 84.0, 60.0), 0.5 + 0.5 * nz(SEED + 72, p, 12.0, gsd));
            let col = mixc(sand, cc, 0.7 * coral);
            let c = if depth < 10.0 { cls::CORAL_REEF } else { cls::OCEAN };
            s.composite(Layer { cov: a, albedo: col, cls: c, relit: 1.0, ..Default::default() });
        }
    }
    // ---- barrier lagoons
    if ty.barrier > 0.0 && depth < 7.0 {
        let a = ty.barrier * (1.0 - smoothstep(5.0, 7.0, depth));
        let col = mixc(srgb(150.0, 186.0, 160.0), srgb(70.0, 130.0, 128.0), smoothstep(0.5, 6.0, depth));
        s.composite(Layer { cov: 0.85 * a, albedo: col, cls: cls::OCEAN, relit: 1.0, ..Default::default() });
    }
    // ---- turbid water over the flats
    if ty.flat > 0.0 && depth < 3.0 {
        let a = ty.flat * (1.0 - smoothstep(1.2, 3.0, depth));
        let col = mixc(srgb(122.0, 118.0, 92.0), srgb(86.0, 98.0, 84.0), smoothstep(0.2, 2.5, depth)) * (1.0 + 0.06 * nz(SEED + 73, p, 40.0, gsd));
        s.composite(Layer { cov: 0.9 * a, albedo: col, cls: cls::OCEAN, relit: 1.0, ..Default::default() });
    }
    // ---- kelp over the rocks of glaciated coasts
    if ty.skerry > 0.0 && depth > 0.8 && depth < 16.0 {
        let n = 0.6 * nz(SEED + 74, p, 45.0, gsd) + 0.4 * nz(SEED + 75, p, 12.0, gsd);
        let wd = 0.06 + 0.3 * (1.0 - band(45.0, gsd));
        let k = ty.skerry * smoothstep(0.05 - wd, 0.05 + wd, n) * smoothstep(0.8, 2.0, depth) * (1.0 - smoothstep(10.0, 16.0, depth));
        if k > 0.0 {
            s.composite(Layer { cov: 0.75 * k, albedo: srgb(22.0, 52.0, 42.0), cls: cls::OCEAN, relit: 1.0, ..Default::default() });
        }
    }
    // ---- the clear shallow platform below a cliff
    if ty.cliff > 0.0 && depth < 2.6 {
        let a = ty.cliff * (1.0 - smoothstep(0.8, 2.6, depth));
        s.composite(Layer { cov: 0.6 * a, albedo: srgb(112.0, 150.0, 140.0), cls: cls::OCEAN, relit: 1.0, ..Default::default() });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_compiles_with_the_coast_kit() {
        crate::registry::Registry::builtin().unwrap();
    }

    #[test]
    fn class_ids_are_the_v2_table() {
        use crate::registry::classes::id;
        assert_eq!(id("ocean"), Some(cls::OCEAN));
        assert_eq!(id("rock"), Some(cls::ROCK));
        assert_eq!(id("aquaculture"), Some(cls::AQUACULTURE));
        assert_eq!(id("tidal_flat"), Some(cls::TIDAL_FLAT));
        assert_eq!(id("coral_reef"), Some(cls::CORAL_REEF));
        assert_eq!(id("cliff"), Some(cls::CLIFF));
        assert_eq!(id("marsh"), Some(cls::MARSH));
    }

    fn pt(i: u64) -> DVec3 {
        let h = mix64(0xC0 ^ i);
        DVec3::new(u01k(h, 1) - 0.5, u01k(h, 2) - 0.5, u01k(h, 3) - 0.5).normalize() * 6_371_000.0
    }

    fn only(f: impl Fn(&mut Types)) -> Types {
        let mut t = Types::default();
        f(&mut t);
        t
    }

    /// The remaps are continuous in the height, and bounded.
    #[test]
    fn remaps_are_continuous() {
        let p = pt(1);
        for t in [
            only(|t| t.cliff = 1.0),
            only(|t| t.skerry = 1.0),
            only(|t| t.arch = 1.0),
            only(|t| t.flat = 1.0),
            only(|t| t.barrier = 1.0),
            only(|t| {
                t.reef = 1.0;
                t.far = 0.5
            }),
        ] {
            let mut prev = None;
            for i in -2400..10000 {
                let h0 = i as f64 * 0.05;
                let h1 = h0 + coast_dh(&t, h0, p, 20.0);
                if let Some((hp, q)) = prev {
                    let _: f64 = hp;
                    assert!((h1 - q) < 40.0 * 0.05 + 3.0 || t.cliff > 0.0, "jump at {h0}: {q} -> {h1} ({t:?})");
                }
                prev = Some((h0, h1));
            }
        }
    }

    /// The cliff is a step: C m of rise within the width eps.
    #[test]
    fn cliff_rises_over_a_few_metres() {
        let t = only(|t| t.cliff = 1.0);
        let p = pt(2);
        let a = coast_dh(&t, 0.0, p, 2.0);
        let b = coast_dh(&t, 6.0, p, 2.0);
        assert!(a < 0.0 && b > 12.0, "platform {a}, cliff {b}");
    }

    /// Tidal flats are flatter than the unmodified coast in the intertidal band.
    #[test]
    fn flats_are_flat() {
        let t = only(|t| t.flat = 1.0);
        let p = pt(3);
        let f = |h0: f64| h0 + coast_dh(&t, h0, p, 200.0);
        assert!((f(2.0) - f(-2.0)) < 0.35 * 4.0 + 0.1, "{} {}", f(2.0), f(-2.0));
    }

    /// Creeks cover a modest share of a patch, only at fine pixel sizes.
    #[test]
    fn creek_cover() {
        let p0 = pt(4);
        let (e, n) = (DVec3::Z.cross(p0.normalize()).normalize(), p0.normalize().cross(DVec3::Z.cross(p0.normalize()).normalize()));
        let mut sum = 0.0;
        let k = 4000;
        for i in 0..k {
            let q = (crate::kernels::halton(i + 1, 2) - 0.5) * 3000.0;
            let r = (crate::kernels::halton(i + 1, 3) - 0.5) * 3000.0;
            sum += creek(p0 + e * q + n * r, 2.0, 2.0);
        }
        let m = sum / k as f64;
        assert!(m > 0.02 && m < 0.25, "creek cover {m}");
        assert_eq!(creek(p0, 300.0, 300.0), CREEK_MEAN);
    }

    /// Types are weights in 0..1 that vary around the globe.
    #[test]
    fn types_vary() {
        let mut sums = [0.0f64; 5];
        for i in 0..2000 {
            let h = mix64(i ^ 0x77);
            let t = types(pt(i + 10), 0.6 * u01k(h, 1), 0.8 * u01k(h, 2).powi(3), u01k(h, 3), 30.0 * u01k(h, 4));
            for v in [t.cliff, t.skerry, t.flat, t.barrier, t.reef, t.arch] {
                assert!((0.0..=1.0).contains(&v));
            }
            for (s, v) in sums.iter_mut().zip([t.cliff, t.skerry, t.flat, t.barrier, t.reef]) {
                *s += v;
            }
        }
        assert!(sums.iter().all(|&s| s > 20.0), "{sums:?}");
    }

    /// Debug: coasts of each type by the atlas: `find_coast_places -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_coast_places() {
        use rayon::prelude::*;
        let w = crate::world::World::new(crate::Config::default());
        let ell = w.ell;
        let n = 600_000;
        let hits: Vec<(f64, f64, [f64; 6], f64, f64)> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let h = mix64(0xC0A5 ^ i as u64);
                let lat = (2.0 * u01k(h, 1) - 1.0).asin().to_degrees();
                let lon = 360.0 * u01k(h, 2) - 180.0;
                let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &ell);
                let a = w.atlas().sample(p.normalize());
                if a.coast_km.abs() > 4.0 {
                    return None;
                }
                let t = types(p, a.uplift, 0.0, a.glaciation, a.temp_c);
                Some((lat, lon, [t.cliff, t.skerry, t.flat, t.barrier, t.reef, t.arch], t.far, a.coast_km))
            })
            .collect();
        eprintln!("{} coastal samples", hits.len());
        let mut tot = [0.0f64; 6];
        for h in &hits {
            for (t, v) in tot.iter_mut().zip(h.2) {
                *t += v;
            }
        }
        eprintln!("mean weights cliff/skerry/flat/barrier/reef/arch: {:?}", tot.map(|x| x / hits.len() as f64));
        for (k, name) in ["cliff", "skerry", "flat", "barrier", "reef", "arch"].iter().enumerate() {
            let mut v: Vec<_> = hits.iter().filter(|h| h.2[k] > 0.95 && (k != 4 || h.2[0] + h.2[1] < 0.5)).collect();
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            let step = (v.len() / 12).max(1);
            let list: Vec<String> = v
                .iter()
                .step_by(step)
                .take(12)
                .map(|h| format!("({:.3}, {:.3}{})", h.0, h.1, if k == 4 { format!(" far{:.1}", h.3) } else { String::new() }))
                .collect();
            eprintln!("{name}: {} samples: {}", v.len(), list.join(" "));
        }
    }

    /// Places of the default world where the kit draws: (name, lat, lon), for the tests and stills.
    pub(crate) const PLACES: &[(&str, f64, f64)] = &[
        ("cliff", -21.804, -43.364),
        ("flat", 44.107, 11.756),
        ("flat_arid", 26.538, -5.414),
        ("barrier", 19.379, -14.658),
        ("reef", 11.859, 81.198),
        ("skerry", 61.882, 19.636),
        ("atoll", 7.131, 87.717),
        ("plain", -41.129, -152.831),
        // deltas (`delta`): their mouths
        ("delta_fan", 44.2101, 176.6552),
        ("delta_birdsfoot", 35.1942, 73.4256),
        ("delta_tropical", -1.4670, 173.6521),
    ];

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
            ("cliff", 14u8),
            ("cliff", 16),
            ("flat", 14),
            ("flat", 16),
            ("flat_arid", 14),
            ("barrier", 12),
            ("reef", 14),
            ("reef", 12),
            ("skerry", 14),
            ("atoll", 12),
            ("atoll", 14),
            ("delta_fan", 12),
            ("delta_fan", 14),
            ("delta_birdsfoot", 13),
            ("delta_tropical", 13),
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

    /// Debug: atolls of the default world.
    #[test]
    #[ignore]
    fn find_atolls() {
        let w = crate::world::World::new(crate::Config::default());
        let ell = w.ell;
        let mut seen = std::collections::BTreeMap::new();
        for i in 0..40_000u64 {
            let h = mix64(0xA7 ^ i);
            let lat = 50.0 * (2.0 * u01k(h, 1) - 1.0);
            let lon = 360.0 * u01k(h, 2) - 180.0;
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &ell);
            if w.atlas().sample(p.normalize()).temp_c < 25.0 {
                continue;
            }
            for it in crate::instances::near(&w, fam_atolls(), p, 0.0) {
                seen.entry(it.id).or_insert(it);
            }
        }
        eprintln!("{} atolls", seen.len());
        for it in seen.values().take(14) {
            let g = geodesy::ecef2geodetic(it.center, &ell);
            eprintln!("({:.3}, {:.3}) R{:.0} wr{:.0} L{:.0}", g.lat.to_degrees(), g.lon.to_degrees(), it.v[0], it.v[1], it.v[2]);
        }
    }

    /// Parent ≈ mean of its children (elevation, albedo) where the kit draws.
    #[test]
    fn lod_consistent_at_coast_places() {
        let g = crate::NextGenerator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in [
            ("cliff", 12u8),
            ("flat", 13),
            ("flat_arid", 12),
            ("flat_arid", 13),
            ("reef", 12),
            ("barrier", 12),
            ("atoll", 11),
            ("delta_fan", 11),
            ("delta_fan", 13),
            ("delta_tropical", 12),
        ] {
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
    fn seamless_at_coast_places() {
        let g = crate::NextGenerator::new(crate::Config { tile_supersample: 1, ..crate::Config::default() });
        let n = crate::TILE_SIZE;
        for (name, z) in [("cliff", 14u8), ("flat", 15), ("flat_arid", 14), ("reef", 14), ("skerry", 13), ("atoll", 12), ("delta_birdsfoot", 13)] {
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
                // elevation: the step across the seam is no larger than the slope inside
                let (h0, h1, h2) = (a.elevation[j * n + n - 2] as f64, a.elevation[j * n + n - 1] as f64, e.elevation[j * n] as f64);
                assert!((h2 - h1).abs() < 3.0 * (h1 - h0).abs() + 8.0, "{name}: E-W elevation seam {h0} {h1} {h2}");
                let (v0, v1, v2) = (a.elevation[(n - 2) * n + j] as f64, a.elevation[(n - 1) * n + j] as f64, s.elevation[j] as f64);
                assert!((v2 - v1).abs() < 3.0 * (v1 - v0).abs() + 8.0, "{name}: N-S elevation seam {v0} {v1} {v2}");
            }
            eprintln!("{name} z{z}: albedo E-W across {ea:.0} inside {ei:.0}; N-S across {na:.0} inside {ni:.0}");
            assert!(ea < 2.0 * ei + 2.0 * n as f64, "{name}: E-W across {ea} inside {ei}");
            assert!(na < 2.0 * ni + 2.0 * n as f64, "{name}: N-S across {na} inside {ni}");
        }
    }

    /// Tiles are a pure function of the world: the same tile twice is the same bits.
    #[test]
    fn deterministic_at_coast_places() {
        let cfg = crate::Config { tile_supersample: 1, ..crate::Config::default() };
        let a = crate::NextGenerator::new(cfg.clone());
        let b = crate::NextGenerator::new(cfg);
        for (name, z) in [("cliff", 14u8), ("flat", 15), ("reef", 15)] {
            let id = tile_at(name, z);
            let (x, y) = (a.tile(id), b.tile(id));
            assert!(x.elevation == y.elevation && x.albedo == y.albedo && x.landcover == y.landcover, "{name}");
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
    /// `cargo test -p terragen --release --lib kits::coast::tests::bench_places -- --ignored --nocapture`.
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

    /// Debug: real coast crossings (pass A) near the atlas' coastal samples, by type.
    #[test]
    #[ignore]
    fn find_coast_crossings() {
        use rayon::prelude::*;
        let w = crate::world::World::new(crate::Config::default());
        let ell = w.ell;
        let n = 150_000;
        let hits: Vec<(f64, f64, Types, f64)> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let h = mix64(0xC0A6 ^ i as u64);
                let lat = 70.0 * (2.0 * u01k(h, 1) - 1.0);
                let lon = 360.0 * u01k(h, 2) - 180.0;
                let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat.to_radians(), lon.to_radians(), 0.0), &ell);
                let a = w.atlas().sample(p.normalize());
                if a.coast_km.abs() > 8.0 {
                    return None;
                }
                let q = |dx: f64, dy: f64| {
                    let (la, lo) = (lat + dy / 111.0, lon + dx / (111.0 * lat.to_radians().cos()));
                    let ctx = crate::world::Ctx::new(la.to_radians(), lo.to_radians(), 150.0, &ell);
                    let t = w.terrain(&ctx);
                    (la, lo, t, ctx.p)
                };
                // a land pixel and a sea pixel within 3 km of each other along a ray
                for k in 0..6 {
                    let ang = std::f64::consts::TAU * (k as f64 / 6.0);
                    let (mut prev, mut pt) = (None, None);
                    for j in 0..14 {
                        let r = 12.0 * j as f64 / 13.0 - 6.0;
                        let (la, lo, t, pp) = q(r * ang.cos(), r * ang.sin());
                        let land = t.water_kind == 0 && t.ground > 0.0;
                        if let Some(pl) = prev {
                            if pl != land {
                                let ty = types(pp, t.uplift, t.mountain, t.glaciation, t.temp);
                                return Some((la, lo, ty, t.ground));
                            }
                        }
                        prev = Some(land);
                        pt = Some(t.ground);
                    }
                    let _ = pt;
                }
                None
            })
            .collect();
        eprintln!("{} crossings", hits.len());
        for (k, name) in ["cliff", "skerry", "flat", "barrier", "reef"].iter().enumerate() {
            let v: Vec<_> = hits.iter().filter(|h| [h.2.cliff, h.2.skerry, h.2.flat, h.2.barrier, h.2.reef][k] > 0.95).collect();
            let step = (v.len() / 10).max(1);
            let list: Vec<String> = v.iter().step_by(step).take(10).map(|h| format!("({:.3}, {:.3})", h.0, h.1)).collect();
            eprintln!("{name}: {} samples: {}", v.len(), list.join(" "));
        }
    }

    /// Debug: ground along a line at several pixel sizes.
    #[test]
    #[ignore]
    fn probe_line() {
        let w = crate::world::World::new(crate::Config::default());
        let (lat0, lon0): (f64, f64) = (26.538, -5.414);
        for j in 0..40 {
            let dx = (j as f64 - 20.0) * 0.4;
            let (la, lo) = (lat0, lon0 + dx / (111.0 * lat0.to_radians().cos()));
            let g = |gsd: f64| {
                let ctx = crate::world::Ctx::new(la.to_radians(), lo.to_radians(), gsd, &w.ell);
                let t = w.terrain(&ctx);
                format!("{:7.2}", t.ground)
            };
            eprintln!("{:6.1} km  gsd5 {} gsd40 {} gsd150 {} gsd600 {}", dx, g(5.0), g(40.0), g(150.0), g(600.0));
        }
    }
}
