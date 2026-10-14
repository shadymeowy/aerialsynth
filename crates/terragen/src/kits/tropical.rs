//! The tropical and savanna kit (`docs/design/terrain-next.md` §5.6, `docs/next/kits/tropical.md`).
//!
//! Biomes (`biomes/tropical.yaml`): lowland rainforest (a patch of the core `tropical_moist`)
//! with emergent and flowering crowns, treefall gaps, river-edge forest, meander scars and a
//! slash-and-burn frontier; flooded forest; monsoon and dry deciduous forest; acacia, baobab
//! and palm savannas with dry-season grass, burn scars, termite mounds, waterholes and gallery
//! forests; miombo woodland; cloud forest and páramo as altitudinal zones; mangroves on low
//! tropical coasts and estuaries.
//!
//! Everything is drawn by biome layers (data) over the kernels below (code); the kit has no
//! layer functions. Kernel kinds 144.. (the kit's range):
//!
//! | kernel | kind | draws |
//! |---|---|---|
//! | `tropical_crowns` | 144 | tree crowns on a jittered grid: groves, per-crown hue and flowering colours, crowns that shrink (never clip) where a mask thins them |
//! | `tropical_fire` | 145 | burn scars of several ages (charred, brown, green regrowth) with lobate edges and ash streaks (ECEF noise: seamless) |
//! | `tropical_swidden` | 146 | slash-and-burn plots in forest: irregular clearings, each burnt, cropped or in fallow |
//! | `tropical_waterhole` | 147 | waterholes: a pond (part `pond`), its mud rim, trampled ground and radiating game trails (part `ground`) |
//! | `tropical_meander` | 148 | meander scars of floodplains: oxbow lakes (part `lake`), silted arcs and scroll bars (part `scroll`) |
//! | `tropical_mangrove` | 149 | mangrove forest (part `canopy`) cut by a network of tidal creeks (part `channels`) |
//! | `tropical_grass` | 150 | savanna grass in dry-season colours: golden / grey tussocks, bare red patches |
//!
//! Waterholes and meander scars are instances of a jittered 3D lattice on a reference sphere
//! ([`sphere_frame`], [`sphere_site`]): seamless on the planet and independent of the land-use
//! regions whose frames the other kernels use. The WGSL twins are in
//! `gpu/wgsl/kits/tropical.wgsl`, in the same order with the same operations.

use crate::kernels::{band_cov, profile, shape, KIn, KOut, KParams, KernelSpec, C, P};
use crate::kits::Kit;
use crate::noise::*;
use glam::{DVec2, DVec3};

pub static KIT: Kit = Kit {
    name: "tropical",
    biomes_yaml: include_str!("../../biomes/tropical.yaml"),
    wgsl: include_str!("../gpu/wgsl/kits/tropical.wgsl"),
    kernels: KERNELS,
    layers: &[(crate::stack::slot::CANOPY, fishbone), (crate::stack::slot::WATER, rivers)],
    ..Kit::EMPTY
};

/// Side of the cubes (m) whose anchor's tangent plane frames the fishbone grid.
const FB_ANCHOR: f64 = 200_000.0;
/// Colonization projects (m, the plane's blocks), feeder road spacing, lot width along a
/// feeder, the deepest clearing from a road.
const FB_BLOCK: f64 = 22_000.0;
const FB_FEEDER: f64 = 4_400.0;
const FB_LOT: f64 = 250.0;
const FB_DEPTH: f64 = 3_000.0;

/// The fishbone plane at `p`: (q east, q north in the tangent plane of the anchor of `p`'s
/// 200 km cube, the anchor cube's hash). The anchor is the cube centre's direction (f32) on the
/// reference sphere, so both backends share it to f32 precision.
fn fishbone_plane(p: DVec3) -> (DVec2, u64) {
    let c = (p / FB_ANCHOR).floor();
    let cc = (c + 0.5) * FB_ANCHOR;
    let up = cc.as_vec3().normalize().as_dvec3();
    let o = up * R0;
    let el = (up.x * up.x + up.y * up.y).sqrt();
    let east = if el > 1e-6 { DVec3::new(-up.y / el, up.x / el, 0.0) } else { DVec3::X };
    let north = up.cross(east);
    let d = p - o;
    (DVec2::new(d.dot(east), d.dot(north)), hash3(0xF15B, c.x as i64, c.y as i64, c.z as i64))
}

/// Canopy slot: fishbone deforestation (`docs/design/terrain-next.md` §5.6, P3). Colonization
/// projects in the rainforest lowlands: a trunk road through a project (east–west in the plane
/// of its 200 km anchor), feeder roads every 4.4 km at right angles, and along every road lots
/// of 250 m, each cleared from the road to a depth of its own (deeper near the trunk road:
/// older) — pasture, fresh burns, red soil, crops, regrowth — so the clearings form fishbones
/// from altitude. Projects (22 km blocks of the plane) exist by a hash; the forest gate is the
/// sample's climate. Drawn over the canopy (the clearings reset its height and shadows).
pub fn fishbone(s: &mut crate::stack::Stack) {
    let t = s.t;
    let l = s.l;
    // (a frontier in the forest, not over farmland: projects drew km-long pale strips of pasture
    // over the plantation lowlands, v0.2.0's tropics scene)
    if s.is_water || t.temp < 21.0 || t.moist < 0.5 || l.ground > 700.0 || l.slope > 0.3 || t.development > 0.65 || t.agri > 0.55 {
        return;
    }
    let (q, ah) = fishbone_plane(s.ctx.p);
    let (bi, bj) = ((q.x / FB_BLOCK).floor(), (q.y / FB_BLOCK).floor());
    let bh = hash2(ah, bi as i64, bj as i64);
    if u01k(bh, 1) >= 0.3 {
        return;
    }
    let gsd = s.ctx.gsd;
    let fw = l.fw.max(0.5 * gsd);
    // the trunk road (y = y0 in the block), the nearest feeder (x = x0)
    let y0 = (bj + 0.3 + 0.4 * u01k(bh, 2)) * FB_BLOCK;
    let off = u01k(bh, 3) * FB_FEEDER;
    let fi = ((q.x - off) / FB_FEEDER).round();
    let x0 = fi * FB_FEEDER + off;
    let (dx, dy) = (q.x - x0, q.y - y0);
    // the lot along the feeder (index along it, side) and along the trunk road
    let lj = (dy / FB_LOT).floor();
    let hl = hash3(bh, fi as i64, lj as i64, (dx > 0.0) as i64);
    let age = 1.0 - (dy.abs() / (0.5 * FB_BLOCK)).min(1.0);
    let depth = FB_DEPTH * (0.1 + 0.9 * u01k(hl, 1)) * (0.25 + 0.75 * age.sqrt());
    let lk = (q.x / FB_LOT).floor();
    let ht = hash3(bh ^ 0x7B, lk as i64, 0, (dy > 0.0) as i64);
    let depth_t = FB_DEPTH * (0.2 + 0.6 * u01k(ht, 1));
    // a lot is cleared up to its depth, with a ragged back edge
    let n = 80.0 * perlin3(bh ^ 0x7C, s.ctx.p / 350.0);
    let reach = dy.abs() < 0.5 * FB_BLOCK - 300.0;
    let feeder_lot = reach && dx.abs() < depth + n;
    let trunk_lot = dy.abs() < depth_t + n;
    let (cleared, h) = if trunk_lot {
        (true, ht)
    } else if feeder_lot {
        (true, hl)
    } else {
        (false, 0)
    };
    // roads: red laterite tracks (the feeders within the project, the trunk road)
    let fwr = fw;
    let road = (crate::surface::band_cov(dx, 3.0, fwr) * if reach { 1.0 } else { 0.0 }).max(crate::surface::band_cov(dy, 5.0, fwr));
    if !cleared && road <= 0.0 {
        return;
    }
    let srgb = crate::surface::srgb;
    let laterite = s.pal(crate::registry::pal::LATERITE);
    let (mut col, mut cls) = (s.col, crate::landcover::TRACK);
    if cleared {
        let u = u01k(h, 2);
        let (c, k) = if u < 0.55 {
            (srgb(112.0, 116.0, 88.0), crate::landcover::PASTURE)
        } else if u < 0.62 {
            (srgb(62.0, 58.0, 52.0), crate::landcover::BURN_SCAR)
        } else if u < 0.7 {
            (crate::surface::mixc(laterite * 0.9, srgb(112.0, 100.0, 90.0), 0.4), crate::landcover::BARE)
        } else if u < 0.85 {
            (srgb(104.0, 112.0, 80.0), crate::landcover::CROP)
        } else {
            (srgb(68.0, 88.0, 58.0), crate::landcover::CLEAR_CUT)
        };
        col = c * (0.9 + 0.2 * u01k(h, 3)) * (1.0 + 0.08 * perlin3(h, s.ctx.p / 60.0) * band(60.0, gsd));
        cls = k;
    }
    col = crate::surface::mixc(col, laterite * 1.1, road);
    if road > 0.5 {
        cls = crate::landcover::TRACK;
    }
    let cov = if cleared { 1.0 } else { road };
    s.composite(crate::stack::Layer { cov, albedo: col, dh: 0.0, hmode: crate::stack::hmode::BLEND, cls, relit: 1.0, ..Default::default() });
}

/// Half width (m) above which a tropical river is a white-water river (a major river carrying
/// the sediment of its mountains).
const BIG_RIVER_HW: f64 = 25.0;

/// Slot 8: the water of tropical lowland rivers (`docs/design/terrain-next.md` §5.6, P0).
/// White-water rivers carry sediment (café au lait); black-water rivers of the wet rainforest
/// lowlands are tea-black (humic acids), so a black tributary meets a muddy main stem. Major
/// rivers (half width > 25 m) are white-water; the smaller ones are black-water in the very wet
/// lowlands of a regional style channel's provinces, else muddy. The colour takes over from the
/// core's river colour between 18 and 21 °C and below 600–900 m (a colour gradient: the river
/// stays crisp).
pub fn rivers(s: &mut crate::stack::Stack) {
    let l = s.l;
    // (not on standing water; `is_water` is also set by the river itself)
    if l.water > l.ground && l.water_kind != crate::world::water::NONE {
        return;
    }
    let t = s.t;
    if l.river_hw <= 0.0 || t.river_wet < 0.5 || t.temp < 18.0 || l.ground > 900.0 {
        return;
    }
    let fwr = l.fw.max(s.ctx.gsd * 0.35);
    let cov = band_cov(l.river_d, l.river_hw, fwr);
    if cov <= 0.0 {
        return;
    }
    let trop = smoothstep(18.0, 21.0, t.temp) * (1.0 - smoothstep(600.0, 900.0, l.ground));
    let black = l.river_hw < BIG_RIVER_HW && t.moist > 0.62 && t.style[3] > 0.55;
    let col = if black {
        crate::surface::srgb(36.0, 28.0, 21.0)
    } else {
        // (wide rivers a little lighter: more sediment in suspension at the surface)
        crate::surface::mixc(crate::surface::srgb(128.0, 100.0, 68.0), crate::surface::srgb(158.0, 128.0, 88.0), smoothstep(10.0, 120.0, l.river_hw))
    };
    s.composite(crate::stack::Layer { cov, albedo: crate::surface::mixc(s.col, col, trop), ..Default::default() });
}

/// Kernel kinds of the kit (from `kernels::KIT_BASE` + 80).
pub mod kind {
    pub const CROWNS: u32 = 144;
    pub const FIRE: u32 = 145;
    pub const SWIDDEN: u32 = 146;
    pub const WATERHOLE: u32 = 147;
    pub const MEANDER: u32 = 148;
    pub const MANGROVE: u32 = 149;
    pub const GRASS: u32 = 150;
}

const CROWNS_P: &[P] = &[
    P::Num("cell", 0, 0.5, 500.0, 10.0),
    P::Num("density", 1, 0.0, 1.0, 0.5),
    P::Enum("shape", 2, &shape::NAMES, 1),
    P::Range("radius", 3, 0.1, 500.0, [2.0, 3.0]),
    P::Range("height", 5, 0.0, 150.0, [6.0, 10.0]),
    P::Num("jitter", 7, 0.0, 1.0, 0.8),
    P::Num("clump", 8, 0.0, 1.0e4, 0.0),
    P::Num("clump_fraction", 9, 0.0, 1.0, 0.4),
    P::Num("second", 10, 0.0, 1.0, 0.0),
    P::Num("flowering", 11, 0.0, 1.0, 0.0),
    P::Num("colour_var", 12, 0.0, 1.0, 0.25),
    P::Num("leaf", 13, 0.0, 1.0, 0.6),
    P::Col("colour", 0, C::N("crown_tropic")),
    P::Col("colour2", 1, C::N("crown_dry")),
    P::Col("flower_colour", 2, C::Rgb([186.0, 92.0, 96.0])),
];
const FIRE_P: &[P] = &[
    P::Num("scale", 0, 10.0, 1.0e6, 1500.0),
    P::Num("fraction", 1, 0.0, 1.0, 0.2),
    P::Num("ages", 2, 1.0, 3.0, 3.0),
    P::Num("ash", 3, 0.0, 1.0, 0.5),
    P::Num("colour_var", 4, 0.0, 1.0, 0.15),
    P::Col("colour", 0, C::Rgb([38.0, 34.0, 32.0])),
    P::Col("colour2", 1, C::Rgb([108.0, 86.0, 64.0])),
    P::Col("regrowth_colour", 2, C::Rgb([118.0, 124.0, 64.0])),
];
const SWIDDEN_P: &[P] = &[
    P::Num("cell", 0, 10.0, 1.0e4, 160.0),
    P::Num("fill", 1, 0.0, 1.0, 0.5),
    P::Num("margin", 2, 0.0, 0.45, 0.12),
    P::Num("jitter", 3, 0.0, 1.0, 0.85),
    P::Num("roughness", 4, 0.0, 0.5, 0.18),
    P::Num("height", 5, 0.0, 30.0, 3.0),
    P::Num("colour_var", 6, 0.0, 1.0, 0.2),
    P::Col("colour", 0, C::Rgb([58.0, 50.0, 46.0])),
    P::Col("colour2", 1, C::Rgb([150.0, 118.0, 78.0])),
    P::Col("fallow_colour", 2, C::Rgb([104.0, 136.0, 56.0])),
];
const WATERHOLE_P: &[P] = &[
    P::Num("cell", 0, 50.0, 1.0e5, 2500.0),
    P::Num("density", 1, 0.0, 1.0, 0.5),
    P::Enum("part", 2, &["pond", "ground"], 0),
    P::Range("radius", 3, 1.0, 1.0e4, [15.0, 40.0]),
    P::Num("trails", 5, 0.0, 12.0, 8.0),
    P::Num("length", 6, 0.0, 1.0e4, 700.0),
    P::Num("trail_width", 7, 0.1, 100.0, 2.0),
    P::Num("ring", 8, 1.0, 20.0, 4.0),
    P::Num("jitter", 9, 0.0, 0.8, 0.6),
    P::Num("size", 15, 0.5, 1.0e5, 30.0),
    P::Col("colour", 0, C::Rgb([52.0, 58.0, 44.0])),
    P::Col("colour2", 1, C::Rgb([170.0, 140.0, 104.0])),
    P::Col("mud_colour", 2, C::Rgb([92.0, 76.0, 60.0])),
];
const MEANDER_P: &[P] = &[
    P::Num("cell", 0, 50.0, 1.0e5, 2000.0),
    P::Num("density", 1, 0.0, 1.0, 0.5),
    P::Enum("part", 2, &["lake", "scroll"], 0),
    P::Range("radius", 3, 10.0, 1.0e4, [180.0, 420.0]),
    P::Num("width", 5, 1.0, 1000.0, 30.0),
    P::Num("arc", 6, 0.5, 6.0, 4.4),
    P::Num("open", 7, 0.0, 1.0, 0.6),
    P::Num("bars", 8, 1.0, 1000.0, 45.0),
    P::Num("jitter", 9, 0.0, 0.8, 0.6),
    P::Num("height", 10, 0.0, 80.0, 0.0),
    P::Num("size", 15, 0.5, 1.0e5, 150.0),
    P::Col("colour", 0, C::Rgb([40.0, 52.0, 44.0])),
    P::Col("colour2", 1, C::Rgb([70.0, 84.0, 60.0])),
    P::Col("silt_colour", 2, C::Rgb([96.0, 126.0, 60.0])),
];
const MANGROVE_P: &[P] = &[
    P::Num("scale", 0, 20.0, 1.0e5, 420.0),
    P::Enum("part", 1, &["canopy", "channels"], 0),
    P::Num("width", 2, 0.5, 200.0, 9.0),
    P::Num("height", 3, 0.0, 50.0, 11.0),
    P::Num("crown", 4, 1.0, 50.0, 4.5),
    P::Num("colour_var", 5, 0.0, 1.0, 0.25),
    P::Col("colour", 0, C::Rgb([28.0, 56.0, 30.0])),
    P::Col("colour2", 1, C::Rgb([58.0, 74.0, 46.0])),
    P::Col("water_colour", 2, C::Rgb([62.0, 66.0, 52.0])),
];
const GRASS_P: &[P] = &[
    P::Num("scale", 0, 2.0, 1.0e5, 70.0),
    P::Num("bare", 1, 0.0, 1.0, 0.12),
    P::Num("tussock", 2, 0.0, 1.0, 0.6),
    P::Num("colour_var", 3, 0.0, 1.0, 0.15),
    P::Col("colour", 0, C::N("grass_dry")),
    P::Col("colour2", 1, C::Rgb([150.0, 140.0, 104.0])),
    P::Col("bare_colour", 2, C::N("laterite")),
];

static KERNELS: &[KernelSpec] = &[
    KernelSpec {
        name: "tropical_crowns",
        kind: kind::CROWNS,
        params: CROWNS_P,
        size_slot: 0,
        dyn_size: None,
        reach: Some((4, 0, 1.0)),
        default_hmode: "max",
        eval: crowns,
        wgsl: "tropical_crowns",
    },
    KernelSpec {
        name: "tropical_fire",
        kind: kind::FIRE,
        params: FIRE_P,
        size_slot: 0,
        dyn_size: None,
        reach: None,
        default_hmode: "none",
        eval: fire,
        wgsl: "tropical_fire",
    },
    KernelSpec {
        name: "tropical_swidden",
        kind: kind::SWIDDEN,
        params: SWIDDEN_P,
        size_slot: 0,
        dyn_size: None,
        reach: None,
        default_hmode: "add",
        eval: swidden,
        wgsl: "tropical_swidden",
    },
    KernelSpec {
        name: "tropical_waterhole",
        kind: kind::WATERHOLE,
        params: WATERHOLE_P,
        size_slot: 15,
        dyn_size: None,
        reach: Some((6, 0, SITE_REACH)),
        default_hmode: "none",
        eval: waterhole,
        wgsl: "tropical_waterhole",
    },
    KernelSpec {
        name: "tropical_meander",
        kind: kind::MEANDER,
        params: MEANDER_P,
        size_slot: 15,
        dyn_size: None,
        reach: None,
        default_hmode: "none",
        eval: meander,
        wgsl: "tropical_meander",
    },
    KernelSpec {
        name: "tropical_mangrove",
        kind: kind::MANGROVE,
        params: MANGROVE_P,
        size_slot: 0,
        dyn_size: None,
        reach: None,
        default_hmode: "max",
        eval: mangrove,
        wgsl: "tropical_mangrove",
    },
    KernelSpec {
        name: "tropical_grass",
        kind: kind::GRASS,
        params: GRASS_P,
        size_slot: 0,
        dyn_size: None,
        reach: None,
        default_hmode: "none",
        eval: grass,
        wgsl: "tropical_grass",
    },
];

// ============================================================================= helpers

/// Radius (m) of the reference sphere of the instance lattice.
const R0: f64 = 6_371_000.0;
/// How far (cells) an instance of the sphere lattice may reach from its site: with the jitter
/// at most 0.8 the 3 × 3 × 3 neighbourhood covers it.
const SITE_REACH: f64 = 0.45;

/// ~uniform 0..1 from a noise value of spread `sigma` (logistic).
#[inline]
fn logistic(n: f64, sigma: f64) -> f64 {
    1.0 / (1.0 + (-1.7 * n / sigma).exp())
}

/// An angle wrapped to (−π, π].
#[inline]
fn wrap(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    a - t * (a / t + 0.5).floor()
}

/// The sample on the reference sphere and the local east / north there (its own tangent
/// frame: instances are the same shape whatever land-use region the sample lies in).
fn sphere_frame(p: DVec3) -> (DVec3, DVec3, DVec3) {
    let r = p.length();
    let pu = p * (R0 / r);
    let up = (pu / R0).as_vec3().as_dvec3();
    let el = (up.x * up.x + up.y * up.y).sqrt();
    let east = if el > 1e-6 { DVec3::new(-up.y / el, up.x / el, 0.0) } else { DVec3::X };
    let north = up.cross(east);
    (pu, east, north)
}

/// The site of lattice cell `c` of the instance lattice (cell size `cell` m): its hash and its
/// centre on the reference sphere, if the site lies in the slab of half a cell around the
/// sphere (one site per cell² of surface on average; a decision of the site alone).
fn sphere_site(seed: u64, c: (i64, i64, i64), cell: f64, jit: f64) -> Option<(u64, DVec3)> {
    let h = hash3(seed, c.0, c.1, c.2);
    let j = DVec3::new(0.5 + jit * (u01k(h, 1) - 0.5), 0.5 + jit * (u01k(h, 2) - 0.5), 0.5 + jit * (u01k(h, 3) - 0.5));
    let s = (DVec3::new(c.0 as f64, c.1 as f64, c.2 as f64) + j.as_vec3().as_dvec3()) * cell;
    let rs = s.length();
    if (rs - R0).abs() >= 0.5 * cell {
        return None;
    }
    Some((h, s * (R0 / rs)))
}

/// Shrinking factor of an instance thinned by a mask: 1 well inside the density, → 0 at its
/// threshold (an instance shrinks away instead of being clipped by a per-pixel mask).
#[inline]
fn thin(u: f64, dens: f64) -> f64 {
    smoothstep(0.0, 0.3, (dens - u) / dens.max(1e-6)).sqrt()
}

// ============================================================================= kernels

/// `tropical_crowns`: tree crowns on a jittered grid (3 × 3 cells). Groves: a crown exists only
/// where the clump noise *at its centre* is below the fraction. Crowns near the density
/// threshold shrink. Colour: main / second / flowering by hash, per-crown hue, leaf mottling.
/// v: cell, density, shape, radius [3, 4], height [5, 6], jitter, clump, clump_fraction,
/// second, flowering, colour_var, leaf.
fn crowns(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let cell = v[0];
    let dens = v[1] * i.amount;
    let mut out = KOut::default();
    if dens <= 0.0 {
        return out;
    }
    let sh = v[2] as u32;
    let jit = v[7];
    let cf = (i.q / cell).floor();
    let (ix, iy) = (cf.x as i64, cf.y as i64);
    let mut best = f64::NEG_INFINITY;
    for dy in -1..=1i64 {
        for dx in -1..=1i64 {
            let h = hash2(k.seed, ix + dx, iy + dy);
            let u = u01k(h, 3);
            if u >= dens {
                continue;
            }
            let c = DVec2::new((ix + dx) as f64 + 0.5 + jit * (u01k(h, 1) - 0.5), (iy + dy) as f64 + 0.5 + jit * (u01k(h, 2) - 0.5)) * cell;
            let f = thin(u, dens);
            let r = (v[3] + (v[4] - v[3]) * u01k(h, 4)) * f;
            let rel = i.q - c;
            let d = rel.length();
            if d > r + i.fw {
                continue;
            }
            if v[8] > 0.0 {
                // groves: the clump noise at the crown's centre (crisp per crown)
                let cq = c / v[8];
                let g = perlin3(k.seed ^ 0x6C, DVec3::new(cq.x, cq.y, 0.37)) + 0.5 * perlin3(k.seed ^ 0x6D, DVec3::new(2.3 * cq.x, 2.3 * cq.y, 0.71));
                if logistic(g, 0.3) >= v[9] {
                    continue;
                }
            }
            let mut re = r;
            if sh == shape::STAR {
                let a = rel.y.atan2(rel.x) + u01k(h, 9) * 6.3;
                re *= 0.8 + 0.2 * (7.0 * a).cos();
            }
            let cov = band_cov(d, re, i.fw) * (d <= re + i.fw) as u8 as f64;
            if cov <= 0.0 {
                continue;
            }
            let x = (d / re.max(1e-6)).min(1.0);
            let hh = (v[5] + (v[6] - v[5]) * u01k(h, 5)) * (0.5 + 0.5 * f) * profile(sh, x);
            if hh > best {
                best = hh;
                let uc = u01k(h, 6);
                let base = if uc < v[11] {
                    k.col[2]
                } else if uc < v[11] + v[10] {
                    k.col[1]
                } else {
                    k.col[0]
                };
                let hv = u01k(h, 8) - 0.5;
                let var = v[12];
                let hue = DVec3::new(1.0 + 0.5 * var * hv, 1.0 + 0.15 * var * (u01k(h, 10) - 0.5), 1.0 - 0.45 * var * hv);
                let tint = 1.0 + var * (u01k(h, 7) - 0.5);
                let wl = band(1.2, i.gsd);
                let leaf = if wl > 0.0 { 1.0 + v[13] * 0.3 * perlin3(h, i.p / 1.2) * wl } else { 1.0 };
                out.albedo = base * hue * tint * leaf * (0.74 + 0.34 * (1.0 - x * x));
                out.dh = hh;
                out.id = h;
            }
            out.cov = out.cov.max(cov);
        }
    }
    out
}

/// `tropical_fire`: burn scars: a crisp threshold of lobate noise (3 octaves at ECEF `p`),
/// the age of each part from a second, smoother noise in `ages` crisp classes (a younger fire
/// over an older one): charred black with grey ash streaks and a dark fire front, brown, green
/// regrowth. v: scale, fraction, ages, ash, colour_var.
fn fire(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let sc = v[0];
    let mut n = 0.0;
    let mut a = 1.0;
    let mut norm = 0.0;
    let mut lam = sc;
    for o in 0..3u64 {
        n += a * perlin3(k.seed ^ (0xF1 + o), i.p / lam) * band(lam, i.gsd);
        norm += a * a;
        a *= 0.55;
        lam *= 0.43;
    }
    let u = logistic(n / norm.sqrt(), 0.27);
    let frac = v[1] * i.amount;
    let w = (2.0 * i.fw / sc).clamp(1e-3, 0.5);
    let m = frac - u;
    let cov = smoothstep(-w, w, m);
    if cov <= 0.0 {
        return KOut::default();
    }
    // age classes (crisp borders, antialiased over the filter width)
    let ua = logistic(perlin3(k.seed ^ 0xA6E, i.p / (1.7 * sc)), 0.27);
    let ages = v[2].round().max(1.0);
    let t = ua * ages;
    let bin = t.floor().min(ages - 1.0);
    let wa = (3.0 * i.fw / sc * ages).clamp(1e-3, 0.5);
    // the next younger class blends in over the antialiasing width at the bin border
    let fb = t - bin;
    let age_of = |b: f64| (b + 0.5) / ages;
    let colour = |age: f64| if age < 0.5 { k.col[0] + (k.col[1] - k.col[0]) * (2.0 * age) } else { k.col[1] + (k.col[2] - k.col[1]) * (2.0 * age - 1.0) };
    let mut col = colour(age_of(bin));
    if bin + 1.0 < ages {
        col = col + (colour(age_of(bin + 1.0)) - col) * smoothstep(1.0 - wa, 1.0, fb);
    }
    if bin > 0.0 {
        col = col + (colour(age_of(bin - 1.0)) - col) * (1.0 - smoothstep(0.0, wa, fb));
    }
    // fresh scars: grey ash streaks, a darker front along the edge
    let fresh = 1.0 - smoothstep(0.0, 0.5, age_of(bin) - 0.1);
    if fresh > 0.0 {
        let wl = band(0.06 * sc, i.gsd);
        if wl > 0.0 {
            let s = perlin3(k.seed ^ 0xA54, i.p / (0.06 * sc));
            col = col + (DVec3::new(0.24, 0.23, 0.21) - col) * (v[3] * smoothstep(0.15, 0.55, s) * wl * fresh);
        }
        col *= 1.0 - 0.35 * fresh * (1.0 - smoothstep(0.0, 0.04, m)) * band(0.04 * sc, i.gsd);
    }
    // (variation from the scar noise itself: no extra octave)
    col *= 1.0 + v[4] * (2.0 * u - 1.0);
    // patchy char and ash inside a scar (a fire does not burn evenly): mottling of ~0.2 scales
    let mw = band(0.2 * sc, i.gsd);
    if mw > 0.0 {
        col *= 1.0 + 0.22 * perlin3(k.seed ^ 0xB07, i.p / (0.2 * sc)) * mw;
    }
    KOut { cov, albedo: col, dh: 0.0, emit: 0.0, id: (bin as u64) + 1 }
}

/// `tropical_swidden`: slash-and-burn plots: a warped Worley partition (local frame); a share
/// `fill` of the cells holds a clearing inside a forest margin, each burnt, cropped or in
/// fallow (bushy regrowth). v: cell, fill, margin, jitter, roughness, height, colour_var.
fn swidden(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let cell = v[0];
    let wq = i.q / (0.4 * cell);
    let wn = DVec2::new(perlin3(k.seed ^ 0x51, DVec3::new(wq.x, wq.y, 0.5)), perlin3(k.seed ^ 0x52, DVec3::new(wq.x, wq.y, 1.5)));
    let qw = i.q + wn * (v[4] * cell);
    let wc = worley2(k.seed, qw, cell, v[3]);
    if u01k(wc.id, 1) >= v[1] * i.amount {
        return KOut::default();
    }
    let e = worley2_edge_dist(&wc, qw);
    let cov = ((e - v[2] * cell) / i.fw.max(1e-6) + 0.5).clamp(0.0, 1.0);
    if cov <= 0.0 {
        return KOut::default();
    }
    let st = u01k(wc.id, 2);
    let tint = 1.0 + v[6] * (u01k(wc.id, 3) - 0.5);
    let w3 = band(3.0, i.gsd);
    let n3 = if w3 > 0.0 { perlin3(wc.id, DVec3::new(i.q.x / 3.0, i.q.y / 3.0, 0.3)) * w3 } else { 0.0 };
    let (col, dh) = if st < 0.3 {
        // freshly burnt: charcoal with paler ash and logs
        (k.col[0] * tint * (1.0 + 0.5 * n3), 0.0)
    } else if st < 0.65 {
        // cropped: soil with crops in mottled rows
        (k.col[1] * tint * (1.0 + 0.3 * n3), 0.4)
    } else {
        // fallow: bushy regrowth, the older the taller
        let age = u01k(wc.id, 4);
        (k.col[2] * tint * (0.85 + 0.2 * age) * (1.0 + 0.35 * n3), v[5] * (0.4 + 0.6 * age) * (0.75 + 0.25 * n3))
    };
    KOut { cov, albedo: col, dh, emit: 0.0, id: wc.id }
}

/// `tropical_waterhole`: waterholes of the sphere lattice: a lobate pond (`pond`), and around
/// it a dark mud rim, trampled bare ground and game trails radiating out, wavy and tapering
/// (`ground`). v: cell, density, part, radius [3, 4], trails, length, trail_width, ring, jitter.
fn waterhole(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let cell = v[0];
    let dens = v[1] * i.amount;
    let mut out = KOut::default();
    if dens <= 0.0 {
        return out;
    }
    let ground = v[2] >= 0.5;
    let reach = if ground { v[6].max(v[4] * v[8] * 1.3) } else { v[4] * 1.3 };
    let (pu, east, north) = sphere_frame(i.p);
    let q = pu / cell;
    let qf = q.floor();
    let fq = q - qf;
    let half = 0.5 * v[9];
    let lim = ((reach + i.fw) / cell).powi(2) + 0.25;
    let mut best = f64::MAX;
    for dz in -1..=1i64 {
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let g = DVec3::new(nb_gap(fq.x, dx, half), nb_gap(fq.y, dy, half), nb_gap(fq.z, dz, half));
                if g.length_squared() > lim {
                    continue;
                }
                let Some((h, c)) = sphere_site(k.seed, (qf.x as i64 + dx, qf.y as i64 + dy, qf.z as i64 + dz), cell, v[9]) else { continue };
                let u = u01k(h, 4);
                if u >= dens {
                    continue;
                }
                let rel3 = (pu - c).as_vec3().as_dvec3();
                let rel = DVec2::new(rel3.dot(east), rel3.dot(north));
                let d = rel.length();
                let f = thin(u, dens);
                let r = (v[3] + (v[4] - v[3]) * u01k(h, 5)) * f;
                if d > reach * f + i.fw {
                    continue;
                }
                let th = rel.y.atan2(rel.x);
                let lobe = 1.0 + 0.12 * (3.0 * th + 6.3 * u01k(h, 6)).sin() + 0.07 * (5.0 * th + 6.3 * u01k(h, 7)).sin();
                let rp = r * lobe;
                if !ground {
                    let cov = band_cov(d, rp, i.fw);
                    if cov > out.cov {
                        let x = (d / rp.max(1e-6)).min(1.0);
                        out.cov = cov;
                        out.albedo = k.col[0] + (k.col[2] - k.col[0]) * smoothstep(0.6, 1.0, x);
                        out.dh = 0.0;
                        out.id = h;
                    }
                    continue;
                }
                // mud rim and trampled ground (inside the pond: the rim, under the water)
                let ring = rp * v[8] * (1.0 + 0.18 * (4.0 * th + 6.3 * u01k(h, 8)).sin());
                let mut cov = band_cov(d, ring, i.fw);
                let mut col = k.col[0] * (0.9 + 0.2 * smoothstep(rp, ring, d)) + (k.col[2] - k.col[0]) * (1.0 - smoothstep(rp, 1.35 * rp, d));
                // trails: n of them, nearest of the three around the sample's sector
                let n = v[5].round();
                let len = v[6] * f;
                if n >= 1.0 && d < len + i.fw && d > 0.5 * ring {
                    let sector = std::f64::consts::TAU / n;
                    let j0 = (wrap(th) / sector).round();
                    let mut tc = 0.0f64;
                    for dj in -1..=1 {
                        let j = j0 + dj as f64;
                        let ht = hash2(h, j.rem_euclid(n) as i64, 7);
                        let lk = len * (0.45 + 0.55 * u01k(ht, 1));
                        if d > lk + i.fw {
                            continue;
                        }
                        let a =
                            j * sector + 0.35 * sector * (u01k(ht, 2) - 0.5) + 0.25 * (d / (0.2 * len.max(1.0)) + 6.3 * u01k(ht, 3)).sin() * (d / len.max(1.0));
                        let dp = d * wrap(th - a).abs().min(1.5);
                        let hw = 0.5 * v[7] * (1.0 - smoothstep(0.5 * lk, lk, d)).sqrt();
                        tc = tc.max(band_cov(dp, hw, i.fw));
                    }
                    if tc > 0.0 {
                        col = if cov > 0.0 { col + (k.col[1] - col) * tc } else { k.col[1] };
                        cov = cov.max(tc);
                    }
                }
                if cov > 0.0 && d < best {
                    best = d;
                    out.cov = out.cov.max(cov);
                    out.albedo = col;
                    out.id = h;
                }
            }
        }
    }
    out
}

/// `tropical_meander`: meander scars of the sphere lattice: a horseshoe of the old channel
/// (radius R, half width `width`, spanning `arc`), open water over part of it (`lake`, a share
/// `open` of the scars) and silted elsewhere, with scroll bars (ridges and swales at `bars` m)
/// filling the loop (`scroll`). v: cell, density, part, radius [3, 4], width, arc, open, bars,
/// jitter, height.
fn meander(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let cell = v[0];
    let dens = v[1] * i.amount;
    let mut out = KOut::default();
    if dens <= 0.0 {
        return out;
    }
    let scroll = v[2] >= 0.5;
    let reach = v[4] * 1.2 + v[5];
    let (pu, east, north) = sphere_frame(i.p);
    let q = pu / cell;
    let qf = q.floor();
    let fq = q - qf;
    let half_j = 0.5 * v[9];
    let lim = ((reach + i.fw) / cell).powi(2) + 0.25;
    let hs = 0.5 * v[6];
    for dz in -1..=1i64 {
        for dy in -1..=1i64 {
            for dx in -1..=1i64 {
                let g = DVec3::new(nb_gap(fq.x, dx, half_j), nb_gap(fq.y, dy, half_j), nb_gap(fq.z, dz, half_j));
                if g.length_squared() > lim {
                    continue;
                }
                let Some((h, c)) = sphere_site(k.seed, (qf.x as i64 + dx, qf.y as i64 + dy, qf.z as i64 + dz), cell, v[9]) else { continue };
                let u = u01k(h, 4);
                if u >= dens {
                    continue;
                }
                let rel3 = (pu - c).as_vec3().as_dvec3();
                let rel = DVec2::new(rel3.dot(east), rel3.dot(north));
                let d = rel.length();
                let f = thin(u, dens);
                let rr = (v[3] + (v[4] - v[3]) * u01k(h, 5)) * f;
                if d > rr * 1.2 + v[5] + i.fw {
                    continue;
                }
                let a = wrap(rel.y.atan2(rel.x) - u01k(h, 6) * std::f64::consts::TAU);
                if a.abs() > hs {
                    continue;
                }
                let ra = rr * (1.0 + 0.16 * (u01k(h, 7) - 0.5) * (2.0 * a).cos());
                let end = 1.0 - a.abs() / hs;
                let w = v[5] * f * (3.0 * end).min(1.0).sqrt();
                let dr = d - ra;
                let open_half = if u01k(h, 8) < v[7] { hs * (0.45 + 0.55 * u01k(h, 9)) } else { 0.0 };
                let chan = band_cov(dr, w, i.fw);
                if !scroll {
                    if a.abs() < open_half && chan > out.cov {
                        out.cov = chan;
                        out.albedo = k.col[0] + (k.col[1] - k.col[0]) * smoothstep(0.4, 1.0, dr.abs() / w.max(1e-6));
                        out.dh = 0.0;
                        out.id = h;
                    }
                    continue;
                }
                // silted channel
                if a.abs() >= open_half && chan > 0.0 {
                    out.cov = out.cov.max(chan);
                    out.albedo = k.col[2];
                    out.dh = 0.3 * v[10];
                    out.id = h;
                    continue;
                }
                // scroll bars inside the loop
                let x = -dr - w;
                let xmax = 0.72 * ra * (1.0 - (a / hs).powi(2)).max(0.0).sqrt();
                let inside = ((x.min(xmax - x)) / i.fw.max(1e-6) + 0.5).clamp(0.0, 1.0);
                if inside <= 0.0 {
                    continue;
                }
                let lam = v[8];
                let ph = x / lam;
                let s = ph - ph.floor();
                let wfr = (i.fw / lam).max(1e-4);
                let crisp = smoothstep(0.0, wfr, s) * (1.0 - smoothstep(0.45, 0.45 + wfr, s));
                let ridge = 0.45 + (crisp - 0.45) * band(lam, i.gsd);
                let wl = band(8.0, i.gsd);
                let tex = if wl > 0.0 { 1.0 + 0.18 * perlin3(h ^ 0x5C, i.p / 8.0) * wl } else { 1.0 };
                let young = 1.0 - x / xmax.max(1.0);
                out.cov = out.cov.max(inside);
                out.albedo = (k.col[1] + (k.col[0] - k.col[1]) * ridge) * tex * (0.92 + 0.12 * young);
                out.dh = v[10] * (0.7 + 0.3 * ridge) * (0.75 + 0.25 * (1.0 - young)) * tex;
                out.id = h;
            }
        }
    }
    out
}

/// `tropical_mangrove`: mangrove forest: a crisp extent (lobate noise threshold by the
/// amount: the mask moves the edge), a dense dark canopy of small crown clumps, lower along
/// the creeks, open over the creeks (`canopy`); tidal creeks along the edges of two warped Worley lattices (ECEF:
/// seamless), each edge its own width, some absent (`channels`). v: scale, part, width,
/// height, crown, colour_var.
fn mangrove(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let sc = v[0];
    if i.amount <= 0.0 {
        return KOut::default();
    }
    let ue = logistic(perlin3(k.seed ^ 0x3A1, i.p / (0.9 * sc)) + 0.5 * perlin3(k.seed ^ 0x3A2, i.p / (0.33 * sc)) * band(0.33 * sc, i.gsd), 0.3);
    let w = (2.0 * i.fw / sc).clamp(1e-3, 0.5);
    let ext = smoothstep(-w, w, i.amount - 0.1 - 0.85 * ue);
    if ext <= 0.0 {
        return KOut::default();
    }
    // creeks
    let wl = 0.7 * sc;
    let pw = i.p + DVec3::new(perlin3(k.seed ^ 0x3B1, i.p / wl), perlin3(k.seed ^ 0x3B2, i.p / wl), perlin3(k.seed ^ 0x3B3, i.p / wl)) * (0.35 * sc);
    let c1 = worley3(k.seed ^ 0x3C1, pw, sc, 0.9);
    let e1 = worley_edge_dist(&c1, pw);
    let h1 = c1.id ^ c1.id2;
    let w1 = v[2] * (0.35 + 1.3 * u01k(h1, 1)) * (u01k(h1, 2) < 0.8) as u8 as f64;
    let c2 = worley3(k.seed ^ 0x3C2, pw, 0.36 * sc, 0.9);
    let e2 = worley_edge_dist(&c2, pw);
    let h2 = c2.id ^ c2.id2;
    let w2 = 0.4 * v[2] * (0.4 + 1.2 * u01k(h2, 1)) * (u01k(h2, 2) < 0.55) as u8 as f64;
    let chan = band_cov(e1, w1, i.fw).max(band_cov(e2, w2, i.fw));
    if v[1] >= 0.5 {
        let cov = chan * ext;
        if cov <= 0.0 {
            return KOut::default();
        }
        let wide = smoothstep(0.0, 2.0 * v[2], w1 * band_cov(e1, w1, i.fw));
        return KOut { cov, albedo: k.col[2] * (1.0 - 0.25 * wide), dh: 0.0, emit: 0.0, id: h1 };
    }
    // canopy: crown clumps (local frame), per-clump species tone, lower along the creeks
    let wc = worley2(k.seed ^ 0x3D1, i.q, v[4], 0.9);
    let x = (wc.f1 / (0.62 * v[4])).min(1.0);
    let bt = band(v[4], i.gsd);
    let base = if u01k(wc.id, 1) < 0.3 { k.col[1] } else { k.col[0] };
    let mean_c = k.col[0] * 0.7 + k.col[1] * 0.3;
    let tex = base * (1.0 + v[5] * (u01k(wc.id, 2) - 0.5)) * (0.76 + 0.34 * (1.0 - x * x));
    let col = mean_c * 0.93 + (tex - mean_c * 0.93) * bt;
    let near = smoothstep(0.0, 3.0 * v[2] + 2.0, e1 - w1).min(smoothstep(0.0, 1.5 * v[2] + 1.0, e2 - w2));
    let zone = 0.85 + 0.3 * (0.5 + 0.5 * perlin3(k.seed ^ 0x3D2, i.p / (0.5 * sc)));
    let dh = v[3] * (0.55 + 0.45 * near) * zone * (0.88 + 0.12 * (1.0 - x * x) * bt);
    // the creeks stay open (over standing water: the water shows through)
    KOut { cov: ext * (1.0 - chan), albedo: col * (0.9 + 0.1 * near), dh, emit: 0.0, id: wc.id }
}

/// `tropical_grass`: savanna grass in dry-season colours: golden and grey-straw tones at
/// `scale`, bare (laterite) patches, tussock mottling at metres (band-limited). Its extent is
/// a soft threshold of the tone noise by the amount (a crisp one drew golden patches cut out
/// of the green). v: scale, bare, tussock, colour_var.
fn grass(k: &KParams, i: &KIn) -> KOut {
    let v = &k.v;
    let sc = v[0];
    if i.amount <= 0.0 {
        return KOut::default();
    }
    let t = 0.5 + 0.5 * perlin3(k.seed ^ 0x6A1, i.p / sc);
    let w = (2.0 * i.fw / sc).clamp(0.15, 0.5);
    let cov = smoothstep(-w, w, i.amount - 0.05 - 0.9 * (1.0 - t));
    if cov <= 0.0 {
        return KOut::default();
    }
    let mut col = k.col[0] + (k.col[1] - k.col[0]) * smoothstep(0.3, 0.75, t);
    let wb = band(0.25 * sc, i.gsd);
    let nb = perlin3(k.seed ^ 0x6A2, i.p / (0.6 * sc)) + if wb > 0.0 { 0.5 * perlin3(k.seed ^ 0x6A3, i.p / (0.25 * sc)) * wb } else { 0.0 };
    let ub = logistic(nb, 0.3);
    let bw = (2.0 * i.fw / (0.6 * sc)).clamp(1e-3, 0.5);
    let bare = smoothstep(-bw, bw, v[1] - ub);
    col = col + (k.col[2] - col) * bare;
    let w1 = band(1.6, i.gsd);
    let mut m = 0.0;
    if w1 > 0.0 {
        m += 0.24 * perlin3(k.seed ^ 0x6A4, i.p / 1.6) * w1;
    }
    col *= (1.0 + v[2] * m * (1.0 - 0.6 * bare)) * (1.0 + v[3] * (t - 0.5));
    KOut { cov, albedo: col, dh: 0.0, emit: 0.0, id: 0 }
}

#[cfg(test)]
mod tests {
    /// Debug: fishbone projects and black-water / white-water rivers of the default world:
    /// `find_fishbone_and_rivers -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_fishbone_and_rivers() {
        let w = crate::world::World::new(crate::Config::default());
        let (mut nf, mut nb, mut nw) = (0, 0, 0);
        for i in 0..2_000_000u64 {
            let h = mix64(0xF15B ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            if lat.to_degrees().abs() > 25.0 {
                continue;
            }
            let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            let ctx = crate::world::Ctx::new(lat, lon, 20.0, &w.ell);
            let t = w.terrain(&ctx);
            if t.water_kind != 0 || t.temp < 21.0 || t.ground > 700.0 {
                continue;
            }
            let (la, lo) = (lat.to_degrees(), lon.to_degrees());
            if nf < 6 && t.moist >= 0.5 && t.development <= 0.65 {
                let (q, ah) = fishbone_plane(ctx.p);
                let bh = hash2(ah, (q.x / FB_BLOCK).floor() as i64, (q.y / FB_BLOCK).floor() as i64);
                if u01k(bh, 1) < 0.3 {
                    println!("fishbone ({la:.4}, {lo:.4}) moist {:.2} dev {:.2}", t.moist, t.development);
                    nf += 1;
                }
            }
            if t.river_hw > 0.0 && t.river_d.abs() < 3.0 * t.river_hw && t.river_wet > 0.5 {
                let black = t.river_hw < BIG_RIVER_HW && t.moist > 0.62 && t.style[3] > 0.55;
                if black && nb < 6 {
                    println!("black-water river ({la:.4}, {lo:.4}) hw {:.1}", t.river_hw);
                    nb += 1;
                } else if !black && nw < 6 && t.river_hw > 30.0 {
                    println!("white-water river ({la:.4}, {lo:.4}) hw {:.1} moist {:.2}", t.river_hw, t.moist);
                    nw += 1;
                }
            }
            if nf >= 6 && nb >= 6 && nw >= 6 {
                break;
            }
        }
    }

    use super::*;
    use crate::kernels;

    fn params(name: &str, y: &str) -> KParams {
        let m: std::collections::BTreeMap<String, serde_yaml::Value> = serde_yaml::from_str(y).unwrap();
        kernels::compile_params(kernels::spec(name).unwrap(), &m, 0x5EED, &crate::registry::BiomePal::default_pal()).unwrap()
    }

    /// Cases of every kit kernel (the parameters the biomes use, roughly).
    const CASES: &[(&str, &str)] = &[
        ("tropical_crowns", "{cell: 9, density: 0.7, radius: [3, 5], height: [8, 14], second: 0.3, flowering: 0.1}"),
        ("tropical_crowns", "{cell: 14, density: 0.8, shape: star, radius: [3, 5], height: [10, 16], clump: 120, clump_fraction: 0.4}"),
        ("tropical_crowns", "{cell: 12, density: 0.5, shape: umbrella, radius: [3, 6], height: [5, 9]}"),
        ("tropical_fire", "{scale: 600, fraction: 0.35, ash: 0.6}"),
        ("tropical_swidden", "{cell: 150, fill: 0.6, margin: 0.1, height: 4}"),
        ("tropical_waterhole", "{cell: 1500, density: 0.8, part: pond, radius: [20, 40], size: 30}"),
        ("tropical_waterhole", "{cell: 1500, density: 0.8, part: ground, radius: [20, 40], length: 600, trail_width: 3, ring: 4, size: 80}"),
        ("tropical_meander", "{cell: 1200, density: 0.8, part: lake, radius: [150, 350], width: 25, open: 0.7, size: 120}"),
        ("tropical_meander", "{cell: 1200, density: 0.8, part: scroll, radius: [150, 350], width: 25, bars: 40, height: 15, size: 120}"),
        ("tropical_mangrove", "{scale: 300, width: 8}"),
        ("tropical_mangrove", "{scale: 300, part: channels, width: 8}"),
        ("tropical_grass", "{scale: 60, bare: 0.2}"),
    ];

    #[test]
    fn registry_compiles_with_the_kit() {
        let r = crate::registry::Registry::builtin().unwrap();
        for b in [
            "tropical_moist",
            "tropical_dry",
            "savanna_acacia",
            "savanna_baobab",
            "palm_savanna",
            "miombo_woodland",
            "tropical_dry_forest",
            "monsoon_forest",
            "flooded_forest",
            "tropical_cloud_forest",
            "tropical_paramo",
        ] {
            assert!(r.index(b).is_some(), "biome {b}");
        }
    }

    /// The kernels' calibrated means are what their explicit evaluations average to (a
    /// different point set), and a coarse pixel shows the mean.
    #[test]
    fn means_match_explicit_averages() {
        for (name, y) in CASES {
            let k = params(name, y);
            let m = kernels::calibrate(&k);
            let sz = kernels::size(&k);
            // the lattices' cells, not the crossfade size, set the area to average over
            let span = match k.kind {
                kind::WATERHOLE | kind::MEANDER => 40.0 * k.v[0],
                _ => 90.0 * sz,
            };
            let n = 8000;
            let (mut c, mut a) = (0.0, DVec3::ZERO);
            let base = DVec3::new(6_371_000.0, 0.0, 0.0);
            for j in 0..n {
                let q = DVec2::new(kernels::halton(j + 7, 5) * span + 1e4, kernels::halton(j + 7, 7) * span - 3e3);
                let i =
                    KIn { q, p: base + DVec3::new(0.0, q.x, q.y), east: DVec3::Y, north: DVec3::Z, gsd: sz / 50.0, fw: sz / 50.0, amount: 1.0, aux: [0.0; 4] };
                let o = kernels::explicit(&k, &i);
                c += o.cov;
                a += o.albedo * o.cov;
            }
            let c = c / n as f64;
            assert!((c - m.cov).abs() < 0.03 + 0.15 * m.cov, "{name} {y}: mean coverage {} vs average {c}", m.cov);
            if c > 0.05 {
                assert!((a / (c * n as f64) - m.albedo).abs().max_element() < 0.03, "{name} {y}: mean albedo");
            }
            let i = KIn { q: DVec2::new(10.0, 20.0), p: base, east: DVec3::Y, north: DVec3::Z, gsd: 10.0 * sz, fw: 10.0 * sz, amount: 0.5, aux: [0.0; 4] };
            let o = kernels::eval(&k, &m, &i);
            assert!((o.cov - 0.5 * m.cov).abs() < 1e-9, "{name}: coarse coverage");
        }
    }

    /// Thinning by a mask shrinks instances: coverage falls continuously with the amount.
    #[test]
    fn masks_thin_instances_continuously() {
        let k = params("tropical_crowns", "{cell: 9, density: 0.8, radius: [3, 5], height: [8, 14]}");
        let mut prev: Option<f64> = None;
        for s in 0..=40 {
            let amount = s as f64 / 40.0;
            let mut acc = 0.0;
            for j in 0..3000 {
                let q = DVec2::new(kernels::halton(j + 1, 2) * 400.0, kernels::halton(j + 1, 3) * 400.0);
                let i = KIn { q, p: DVec3::new(6_371_000.0, q.x, q.y), east: DVec3::Y, north: DVec3::Z, gsd: 0.2, fw: 0.2, amount, aux: [0.0; 4] };
                acc += kernels::explicit(&k, &i).cov;
            }
            let c = acc / 3000.0;
            if let Some(p) = prev {
                assert!(c + 1e-9 >= p && c - p < 0.06, "coverage {p} -> {c} at amount {amount}");
            }
            prev = Some(c);
        }
    }

    /// The sphere lattice is seamless: a waterhole is the same shape whatever frame the sample
    /// comes with, and the kernels are deterministic.
    #[test]
    fn sphere_lattice_ignores_the_frame() {
        let k = params("tropical_waterhole", "{cell: 1500, density: 1, part: ground, radius: [20, 40], length: 600, trail_width: 3, ring: 4}");
        let ctx = crate::world::Ctx::new(5.0f64.to_radians(), -60.0f64.to_radians(), 1.0, &geodesy::Ellipsoid::WGS84);
        let mut hits = 0;
        for j in 0..4000 {
            let q = DVec2::new((kernels::halton(j + 1, 2) - 0.5) * 6000.0, (kernels::halton(j + 1, 3) - 0.5) * 6000.0);
            let p = ctx.p + ctx.east * q.x + ctx.north * q.y;
            let a = KIn { q, p, east: ctx.east, north: ctx.north, gsd: 1.0, fw: 1.0, amount: 1.0, aux: [0.0; 4] };
            // another frame (a land-use region's), another local position
            let b = KIn { q: q + DVec2::new(3210.0, -987.0), east: (ctx.east + ctx.north * 1e-3).normalize(), ..a };
            let (oa, ob) = (waterhole(&k, &a), waterhole(&k, &b));
            assert_eq!(oa.cov.to_bits(), ob.cov.to_bits());
            assert_eq!(oa.cov.to_bits(), waterhole(&k, &a).cov.to_bits());
            hits += (oa.cov > 0.5) as usize;
        }
        assert!(hits > 20, "{hits} samples on waterholes");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn kernels_match_the_cpu() {
        let Some(r) = crate::gpu::tests::kernel_parity(CASES) else { return };
        for (name, bad, mx) in &r {
            eprintln!("  {name:20} off {:.3}% max coverage difference {mx:.3}", bad * 100.0);
            assert!(*bad < 0.005, "{name}: {:.2}% of the samples differ", bad * 100.0);
        }
    }

    /// Places of the default world where the kit draws (rainforest with a floodplain river,
    /// savanna, burn scars, slash-and-burn, cloud forest, baobabs, mangrove coast).
    pub(crate) const PLACES: &[(&str, f64, f64)] = &[
        ("savanna_gallery_burns", 22.5041, -27.7881),
        ("rainforest_river_edge", 12.1997, -28.5894),
        ("meander_scars", 6.3939, 150.9967),
        ("monsoon_mosaic", -17.9542, -128.5902),
        ("mangrove_coast", 7.1584, -41.9711),
        ("savanna_river", 3.3011, -16.0396),
        ("baobab_savanna", 8.0571, 144.8028),
        ("white_water", 21.2115, -20.2621),
        ("fishbone", -5.5200, 111.7799),
        ("black_water", 4.6276, 5.3377),
    ];

    /// Tiles where the kit draws agree between the backends (at most 0.5 % of the pixels off).
    #[cfg(feature = "gpu")]
    #[test]
    fn tiles_match_the_cpu() {
        let Ok(g) = crate::gpu::shared() else { return };
        if g.check_generator().is_err() {
            return;
        }
        let cfg = crate::Config::default();
        let gen = crate::gpu::GpuGenerator::new(cfg.clone()).unwrap();
        let cpu = crate::NextGenerator::new(cfg);
        let mut worst = 0.0f64;
        for (i, &(name, lat, lon)) in PLACES.iter().enumerate() {
            let z = [16u8, 14, 12][i % 3];
            let id = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
            let gt = gen.tiles(&[id]).unwrap().remove(0);
            let ct = cpu.tile_cpu(id);
            for (layer, mean, bad, mx) in crate::gpu::tests::compare_tiles(&gt, &ct) {
                eprintln!("  {name:16} {id} {layer:10} mean {mean:8.4} off {:7.3}% max {mx:8.2}", bad * 100.0);
                if matches!(layer, "elevation" | "rgb" | "landcover") {
                    worst = worst.max(bad);
                }
            }
        }
        assert!(worst <= 0.005, "{:.2}% of the pixels differ", worst * 100.0);
    }

    /// The GPU generates the kit's places at every zoom without falling back to the CPU (an
    /// instance family overflowing a block, a shader error): `Backend::Gpu` returns the error.
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

    /// GPU tiles/s at the kit's places (8 × 8 tiles per place and zoom; performance budget):
    /// `cargo test -p terragen --release --lib kits::tropical::tests::bench_places -- --ignored --nocapture`.
    /// For the baseline, run it with the kit's `KITS` entry commented out (the module stays).
    #[cfg(feature = "gpu")]
    #[test]
    #[ignore]
    fn bench_places() {
        let gen = crate::NextGenerator::try_with_backend(crate::Config::default(), crate::Backend::Gpu).unwrap();
        gen.tiles(&[geodesy::tiles::TileId::new(3, 4, 3)]).unwrap();
        for z in [8u8, 10, 12, 14, 16] {
            let mut ids = Vec::new();
            for &(_, lat, lon) in PLACES {
                let c = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
                let n = 1i64 << z;
                for dy in 0..8i64 {
                    for dx in 0..8i64 {
                        ids.push(geodesy::tiles::TileId::new(z, (c.x as i64 + dx - 4).rem_euclid(n) as u32, (c.y as i64 + dy - 4).clamp(0, n - 1) as u32));
                    }
                }
            }
            let t0 = std::time::Instant::now();
            for chunk in ids.chunks(16) {
                gen.tiles(chunk).unwrap();
            }
            let dt = t0.elapsed().as_secs_f64();
            println!("bench z{z}: {} tiles {dt:.2} s {:.1} tiles/s", ids.len(), ids.len() as f64 / dt);
        }
    }

    /// Places of the default world where the kit's biomes and features are (to choose stills):
    /// `cargo test -p terragen --release --lib kits::tropical::tests::find_places -- --ignored --nocapture`.
    /// Per biome a few points, and per feature (meander floodplains, mangrove coasts, classes the
    /// kit draws) points where its masks hold or its class is probed.
    #[test]
    #[ignore]
    fn find_places() {
        let gen = crate::NextGenerator::new(crate::Config::default());
        let w = crate::world::World::new(crate::Config::default());
        let reg = crate::registry::Registry::builtin().unwrap();
        let mut found = std::collections::BTreeMap::<String, Vec<String>>::new();
        let mut add = |key: String, s: String| {
            let v = found.entry(key).or_default();
            if v.len() < 8 {
                v.push(s);
            }
        };
        let n: u64 = std::env::var("TROPICAL_FIND").ok().and_then(|s| s.parse().ok()).unwrap_or(30_000);
        let coast_only = std::env::var("TROPICAL_FIND_COAST").is_ok();
        for i in 0..n {
            let h = mix64(0x7F0 ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            if lat.abs() > 30f64.to_radians() {
                continue;
            }
            let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            if w.macro_at(p, 20_000.0).cont < 0.02 {
                continue;
            }
            if coast_only {
                // low tropical coasts only (mangrove places): pass A first, cheap
                let t = w.terrain(&crate::world::Ctx::new(lat, lon, 30.0, &w.ell));
                // shallow tropical sea near the coast (the mangroves' water-slot layer)
                if !(t.coast_km < 15.0 && t.water_kind == crate::world::water::OCEAN && t.water - t.ground < 2.5 && t.temp > 20.0) {
                    continue;
                }
            }
            let [(id, site), _] = crate::eco::sites(&w, p);
            let e = crate::eco::compute(&w, &reg, id, site);
            let name = reg.biomes[e.biome as usize].name.clone();
            if !(name.starts_with("tropical")
                || name.contains("savanna")
                || name.contains("forest") && !name.starts_with("temperate")
                || name.starts_with("miombo"))
            {
                continue;
            }
            let (t, _, cls) = gen.probe(lat, lon, 2.0);
            let s = format!(
                "{:.4},{:.4} {name} T {:.1} moist {:.2} h {:.0} coast {:.1} fp {:.2} river {:.0} cls {}",
                lat.to_degrees(),
                lon.to_degrees(),
                t.temp,
                t.moist,
                t.ground,
                t.coast_km,
                t.floodplain,
                if t.river_hw > 0.0 { t.river_d.abs() - t.river_hw } else { -1.0 },
                crate::registry::classes::name(cls)
            );
            if !coast_only && t.water_kind != crate::world::water::NONE && t.water > t.ground {
                continue;
            }
            add(format!("biome {name}"), s.clone());
            add(format!("class {}", crate::registry::classes::name(cls)), s.clone());
            if t.floodplain > 0.35 && t.river_hw > 0.0 && t.river_d.abs() < 2500.0 {
                add(format!("floodplain {name}"), s.clone());
            }
            if coast_only {
                add(format!("mangrove shore {name}"), s.clone());
            }
            if t.temp < 17.5 && t.temp > 10.0 {
                add(format!("cloud forest {name}"), s.clone());
            }
        }
        for (b, v) in found {
            println!("{b}:");
            for s in v {
                println!("    {s}");
            }
        }
    }
}
