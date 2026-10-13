//! The agriculture kit (`docs/design/terrain-next.md` §5.8, `docs/kits/agriculture.md`): field
//! systems by culture × climate × terrain.
//!
//! The core draws one field look everywhere (grid, irregular, pivot or strip fields of one crop
//! palette). This kit gives every land-use region one **field system** of a catalogue, drawn by
//! its ecoregion's culture and climate, and draws it in place of the core's fields:
//!
//! | system | look |
//! |---|---|
//! | section grid | 1-mile sections in the east / north frame, quarter sections, pivots where dry, section-line roads |
//! | shelterbelt steppe | huge fields bordered by tree belts |
//! | open-field strips | furlongs of long narrow strips, each its own crop, reverse-S |
//! | bocage | small irregular pastures and fields inside hedgerows |
//! | smallholder mosaic | tiny irregular plots, scattered trees, homestead compounds |
//! | rice paddies | bunded paddies, flooded / green / ripe by season; contour terraces on slopes; canals, fish ponds |
//! | dry terraces | stone-walled contour benches with grain, olives, orchards |
//! | vineyards | parcels of vine rows (field axis or contours) |
//! | groves | olive / almond / citrus groves (date palms in deserts, apples in cool climates) |
//! | plantations | oil palm, rubber, tea on contours, cane; estate roads |
//! | greenhouse seas | white plastic bays, irrigation ponds |
//! | pastures | paddocks, livestock trails to troughs, farm dams, hay meadows |
//! | irrigated pivots | centre pivots in the desert |
//! | ranches | huge fenced paddocks, stock trails to troughs and dams; feedlots (pens, a feed mill, manure lagoons) |
//!
//! **How a system is chosen.** Each ecoregion (its culture's archetype, its Köppen class,
//! development; [`EcoV`]) weighs the systems ([`weights`]), with a dominant one per ecoregion,
//! and each land-use region (~7 km) draws one by those weights ([`pick_system`]). The pixel's own
//! ecoregion is used (not the ecotone mosaic): systems change along ecoregion borders and
//! region borders, never per sample. The weights keep a share for the core's fields.
//!
//! **Replacing the core's fields.** Slot 4 (disturbance) lowers `Masks::flat_ok` 4× where the
//! kit draws (the core field layer of slot 5 then stands down), slot 5 restores it and draws the
//! system's parcels; slot 6 (linear) adds the section-line roads.
//!
//! Every parcel is a crisp unit with a hashed id (R-site: its kind, crop, border, state are
//! decided at the parcel); fine textures (rows, furrows, crowns, trails) are band-limited, and
//! parcels near the pixel size blend to the system's mean (as the core's fields do). The WGSL
//! twin is `gpu/wgsl/kits/agriculture.wgsl`; its tables are generated from this file
//! ([`wgsl_tables`], test `wgsl_tables_are_current`).

use crate::kits::Kit;
use crate::noise::*;
use crate::registry::pal;
use crate::stack::{hmode, slot, Layer, Stack};
use crate::surface::{band_cov, mixc, srgb};
use glam::{DVec2, DVec3};

pub static KIT: Kit = Kit {
    name: "agriculture",
    biomes_yaml: include_str!("../../biomes/agriculture.yaml"),
    wgsl: include_str!("../gpu/wgsl/kits/agriculture.wgsl"),
    layers: &[(slot::DISTURBANCE, claim), (slot::AGRICULTURE, fields), (slot::LINEAR, roads)],
    ..Kit::EMPTY
};

// ============================================================================= catalogue

/// Field systems.
pub mod sys {
    pub const CORE: u32 = 0;
    pub const SECTION: u32 = 1;
    pub const SHELTER: u32 = 2;
    pub const STRIPS: u32 = 3;
    pub const BOCAGE: u32 = 4;
    pub const SMALL: u32 = 5;
    pub const PADDY: u32 = 6;
    pub const TERRACE: u32 = 7;
    pub const VINE: u32 = 8;
    pub const GROVE: u32 = 9;
    pub const PLANT: u32 = 10;
    pub const GREENH: u32 = 11;
    pub const PASTURE: u32 = 12;
    pub const PIVOT: u32 = 13;
    pub const RANCH: u32 = 14;
    pub const N: usize = 15;
    #[allow(dead_code)]
    pub const NAMES: [&str; N] = [
        "core",
        "section",
        "shelterbelt",
        "strips",
        "bocage",
        "smallholder",
        "paddy",
        "terrace",
        "vineyard",
        "grove",
        "plantation",
        "greenhouse",
        "pasture",
        "pivot",
        "ranch",
    ];
}

/// What a parcel holds.
pub mod kind {
    pub const CROP: u32 = 0;
    pub const FALLOW: u32 = 1;
    pub const PASTURE: u32 = 2;
    pub const HAY: u32 = 3;
    pub const ORCHARD: u32 = 4;
    pub const OLIVE: u32 = 5;
    pub const VINE: u32 = 6;
    pub const PALM: u32 = 7;
    pub const TEA: u32 = 8;
    pub const PADDY: u32 = 9;
    pub const GREENH: u32 = 10;
    pub const POND: u32 = 11;
    pub const YARD: u32 = 12;
    pub const RUBBER: u32 = 13;
}

/// How a system's area is cut into parcels.
mod part {
    pub const GRID: u32 = 0;
    pub const VOR: u32 = 1; // cadastral blocks (`part_cad`)
    pub const STRIP: u32 = 2;
    pub const PIVOT: u32 = 3;
    pub const SECTION: u32 = 4;
}

/// Parcel borders.
mod border {
    pub const MARGIN: u32 = 0;
    pub const HEDGE: u32 = 1;
    pub const BELT: u32 = 2;
    pub const WALL: u32 = 3;
    pub const DITCH: u32 = 4;
    pub const TRACK: u32 = 5;
    pub const NONE: u32 = 6;
    pub const CANAL: u32 = 7;
}

/// A field system: partition, base parcel size (m), what the parcels hold, their borders, the
/// slopes it farms (smoothstep window of the tan where it stops), contour terraces, scattered
/// trees (density per cell).
#[derive(Clone, Copy, Debug)]
struct SysDef {
    part: u32,
    size: [f64; 2],
    kinds: [u32; 4],
    kw: [f64; 4],
    borders: [u32; 2],
    bp: [f64; 2],
    slope: [f64; 2],
    terraces: f64,
    trees: f64,
}

#[rustfmt::skip]
const SYS: [SysDef; sys::N] = {
    use border::*;
    use kind::*;
    use part::*;
    const D: SysDef = SysDef { part: GRID, size: [200.0, 200.0], kinds: [CROP, FALLOW, PASTURE, HAY], kw: [0.7, 0.15, 0.1, 0.05], borders: [MARGIN, MARGIN], bp: [0.0, 0.0], slope: [0.22, 0.32], terraces: 0.0, trees: 0.0 };
    [
        D,
        SysDef { part: SECTION, size: [402.336, 402.336], kinds: [CROP, PASTURE, FALLOW, POND], kw: [0.84, 0.07, 0.07, 0.02], borders: [BELT, DITCH], bp: [0.06, 0.08], ..D },
        SysDef { part: GRID, size: [1300.0, 900.0], kinds: [CROP, FALLOW, PASTURE, HAY], kw: [0.68, 0.22, 0.06, 0.04], borders: [BELT, TRACK], bp: [0.82, 0.1], ..D },
        SysDef { part: STRIP, size: [520.0, 24.0], kinds: [CROP, FALLOW, HAY, PASTURE], kw: [0.72, 0.12, 0.12, 0.04], borders: [MARGIN, MARGIN], bp: [1.0, 0.0], ..D },
        SysDef { part: VOR, size: [170.0, 1.5], kinds: [PASTURE, CROP, HAY, FALLOW], kw: [0.5, 0.3, 0.14, 0.06], borders: [HEDGE, WALL], bp: [0.55, 0.06], slope: [0.3, 0.42], ..D },
        SysDef { part: VOR, size: [48.0, 1.8], kinds: [CROP, FALLOW, PASTURE, YARD], kw: [0.6, 0.24, 0.1, 0.06], borders: [MARGIN, HEDGE], bp: [0.6, 0.08], slope: [0.3, 0.42], trees: 0.3, ..D },
        SysDef { part: GRID, size: [130.0, 90.0], kinds: [PADDY, POND, CROP, YARD], kw: [0.9, 0.025, 0.065, 0.01], borders: [NONE, CANAL], bp: [0.75, 0.25], slope: [0.7, 0.9], terraces: 1.0, ..D },
        SysDef { part: VOR, size: [110.0, 1.3], kinds: [CROP, OLIVE, FALLOW, ORCHARD], kw: [0.42, 0.28, 0.2, 0.1], borders: [WALL, MARGIN], bp: [0.75, 0.25], slope: [0.7, 0.9], terraces: 1.0, ..D },
        SysDef { part: GRID, size: [140.0, 70.0], kinds: [VINE, OLIVE, ORCHARD, FALLOW], kw: [0.7, 0.1, 0.1, 0.1], borders: [TRACK, MARGIN], bp: [0.15, 0.85], slope: [0.6, 0.8], ..D },
        SysDef { part: VOR, size: [190.0, 1.4], kinds: [OLIVE, ORCHARD, CROP, FALLOW], kw: [0.55, 0.2, 0.15, 0.1], borders: [WALL, MARGIN], bp: [0.5, 0.5], slope: [0.6, 0.8], ..D },
        SysDef { part: GRID, size: [480.0, 360.0], kinds: [PALM, RUBBER, TEA, CROP], kw: [0.45, 0.22, 0.15, 0.18], borders: [TRACK, MARGIN], bp: [0.9, 0.1], slope: [0.6, 0.8], ..D },
        SysDef { part: GRID, size: [95.0, 60.0], kinds: [GREENH, CROP, FALLOW, POND], kw: [0.76, 0.13, 0.09, 0.02], borders: [TRACK, MARGIN], bp: [0.25, 0.75], ..D },
        SysDef { part: GRID, size: [420.0, 300.0], kinds: [PASTURE, HAY, CROP, POND], kw: [0.6, 0.22, 0.16, 0.02], borders: [MARGIN, HEDGE], bp: [0.8, 0.2], slope: [0.38, 0.5], ..D },
        SysDef { part: PIVOT, size: [805.0, 805.0], kinds: [CROP, FALLOW, CROP, CROP], kw: [0.75, 0.25, 0.0, 0.0], borders: [NONE, NONE], bp: [1.0, 0.0], ..D },
        SysDef { part: GRID, size: [2200.0, 1500.0], kinds: [PASTURE, HAY, FALLOW, POND], kw: [0.86, 0.04, 0.08, 0.02], borders: [TRACK, MARGIN], bp: [0.4, 0.6], slope: [0.38, 0.5], ..D },
    ]
};

/// Base weights of the systems (`sys` order) by culture archetype (`atlas::Archetype` order).
#[rustfmt::skip]
const ARCH_W: [[f64; sys::N]; 12] = [
    //core  sect  shelt strip bocag small paddy terr  vine  grove plant green past  pivot ranch
    [0.08, 0.0,  0.0,  0.0,  0.0,  0.50, 0.12, 0.06, 0.0,  0.04, 0.08, 0.0,  0.12, 0.0,  0.02], // tropical smallholder
    [0.10, 0.0,  0.0,  0.0,  0.0,  0.15, 0.10, 0.04, 0.0,  0.02, 0.50, 0.0,  0.09, 0.0,  0.04], // tropical plantation
    [0.08, 0.0,  0.0,  0.0,  0.0,  0.32, 0.0,  0.02, 0.0,  0.0,  0.02, 0.0,  0.50, 0.06, 0.15], // savanna pastoral
    [0.08, 0.05, 0.0,  0.0,  0.0,  0.10, 0.0,  0.04, 0.02, 0.22, 0.0,  0.09, 0.05, 0.35, 0.06], // desert oasis
    [0.12, 0.04, 0.42, 0.10, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.25, 0.07, 0.18], // steppe
    [0.12, 0.0,  0.0,  0.04, 0.0,  0.06, 0.0,  0.16, 0.30, 0.25, 0.0,  0.08, 0.06, 0.03, 0.05], // mediterranean
    [0.28, 0.0,  0.02, 0.20, 0.20, 0.0,  0.0,  0.02, 0.08, 0.06, 0.0,  0.03, 0.14, 0.0,  0.0 ], // temperate village
    [0.10, 0.66, 0.02, 0.0,  0.0,  0.0,  0.0,  0.0,  0.02, 0.05, 0.0,  0.02, 0.09, 0.04, 0.16], // survey grid
    [0.10, 0.0,  0.0,  0.0,  0.0,  0.15, 0.60, 0.04, 0.0,  0.02, 0.05, 0.02, 0.02, 0.0,  0.0 ], // monsoon paddy
    [0.40, 0.0,  0.0,  0.08, 0.06, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.01, 0.45, 0.0,  0.0 ], // boreal
    [0.30, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.70, 0.0,  0.0 ], // arctic
    [0.06, 0.0,  0.0,  0.06, 0.0,  0.24, 0.08, 0.38, 0.0,  0.02, 0.02, 0.0,  0.14, 0.0,  0.04], // highland
];

/// Crops: colour (sRGB) in spring, summer, autumn; row spacing (m), row contrast, height (m).
#[rustfmt::skip]
const CROPS: [([[f64; 3]; 3], [f64; 3]); 12] = [
    ([[98.0, 122.0, 60.0], [196.0, 172.0, 104.0], [186.0, 170.0, 132.0]], [0.17, 0.06, 0.8]),   // wheat
    ([[112.0, 102.0, 76.0], [72.0, 108.0, 48.0], [178.0, 160.0, 110.0]], [0.76, 0.14, 2.2]),    // maize
    ([[112.0, 102.0, 80.0], [84.0, 118.0, 52.0], [172.0, 140.0, 78.0]], [0.5, 0.1, 0.8]),       // soy
    ([[214.0, 198.0, 72.0], [108.0, 110.0, 66.0], [140.0, 120.0, 90.0]], [0.2, 0.05, 1.2]),     // rapeseed
    ([[100.0, 116.0, 64.0], [214.0, 180.0, 48.0], [84.0, 70.0, 52.0]], [0.7, 0.1, 1.8]),        // sunflower
    ([[126.0, 104.0, 82.0], [60.0, 96.0, 48.0], [128.0, 106.0, 84.0]], [0.75, 0.2, 0.6]),       // potato / vegetables
    ([[86.0, 122.0, 56.0], [112.0, 136.0, 66.0], [98.0, 122.0, 62.0]], [6.5, 0.1, 0.5]),        // alfalfa / hay (windrows)
    ([[140.0, 112.0, 86.0], [90.0, 116.0, 62.0], [182.0, 170.0, 138.0]], [1.0, 0.12, 1.0]),     // cotton (open bolls: not white)
    ([[98.0, 132.0, 62.0], [86.0, 128.0, 58.0], [124.0, 136.0, 72.0]], [1.5, 0.08, 3.0]),       // sugarcane
    ([[112.0, 108.0, 96.0], [120.0, 104.0, 142.0], [104.0, 104.0, 92.0]], [1.6, 0.3, 0.6]),     // lavender
    ([[104.0, 118.0, 66.0], [156.0, 104.0, 66.0], [176.0, 150.0, 108.0]], [0.75, 0.1, 1.8]),    // sorghum / millet
    ([[88.0, 118.0, 58.0], [82.0, 114.0, 54.0], [100.0, 116.0, 64.0]], [1.0, 0.15, 1.5]),       // cassava
];

/// Crop mixes: four crops and their shares (temperate, corn belt, steppe, mediterranean,
/// tropical, arid irrigated).
#[rustfmt::skip]
const MIXES: [([u32; 4], [f64; 4]); 6] = [
    ([0, 1, 3, 5], [0.42, 0.24, 0.14, 0.20]),
    ([1, 2, 0, 6], [0.45, 0.35, 0.12, 0.08]),
    ([0, 4, 6, 3], [0.55, 0.30, 0.10, 0.05]),
    ([0, 4, 9, 5], [0.52, 0.22, 0.05, 0.21]),
    ([1, 11, 10, 8], [0.35, 0.30, 0.20, 0.15]),
    ([6, 0, 7, 1], [0.40, 0.30, 0.20, 0.10]),
];

/// Linear colour of an sRGB table entry.
fn lin(c: [f64; 3]) -> DVec3 {
    srgb(c[0], c[1], c[2])
}

// ============================================================================= choice

/// What the kit reads of an ecoregion (a proxy of the atlas' culture fields until they land:
/// archetype, development; climate at the site).
#[derive(Clone, Copy, Debug, Default)]
pub struct EcoV {
    pub id: u64,
    pub arch: u32,
    pub koppen: u32,
    pub temp: f64,
    pub precip: f64,
    pub dev: f64,
    pub scale: f64,
    pub hedges: f64,
}

impl EcoV {
    pub fn of(e: &crate::eco::EcoParams) -> EcoV {
        EcoV {
            id: e.id,
            arch: e.style.archetype as u32,
            koppen: e.climate.koppen,
            temp: e.climate.temp_c,
            precip: e.climate.precip_mm,
            dev: e.style.development,
            scale: e.style.field_scale,
            hedges: e.style.hedges,
        }
    }
}

/// The weights of the systems in an ecoregion: its archetype's, by its climate and development,
/// perturbed per ecoregion, with one dominant system.
pub fn weights(e: &EcoV) -> [f64; sys::N] {
    use sys::*;
    let mut w = ARCH_W[(e.arch as usize).min(11)];
    let k = e.koppen;
    let mut m = [1.0f64; N];
    if (7..=9).contains(&k) {
        // mediterranean summers: vines, groves, terraces, plastic
        m[VINE as usize] = 3.0;
        m[GROVE as usize] = 3.0;
        m[TERRACE as usize] = 1.5;
        m[GREENH as usize] = 2.0;
        m[PADDY as usize] = 0.2;
        m[BOCAGE as usize] = 0.2;
        m[SECTION as usize] = 0.6;
        m[PLANT as usize] = 0.0;
        m[RANCH as usize] = 0.6;
    } else if k == 14 || k == 15 {
        // oceanic: hedged pastures
        m[BOCAGE as usize] = 2.5;
        m[PASTURE as usize] = 1.6;
        m[VINE as usize] = 0.4;
        m[GROVE as usize] = 0.5;
        m[PADDY as usize] = 0.1;
        m[PIVOT as usize] = 0.0;
        m[RANCH as usize] = 0.0;
    } else if k == 10 || k == 13 {
        // humid subtropical: rice
        m[PADDY as usize] = 1.8;
        m[PLANT as usize] = 1.2;
        m[RANCH as usize] = 0.5;
    } else if k <= 2 {
        // tropical
        let wet = if k <= 1 { 1.5 } else { 1.0 };
        m[PLANT as usize] = wet;
        m[PADDY as usize] = 1.4;
        m[VINE as usize] = 0.05;
        m[GREENH as usize] = 0.2;
        m[SHELTER as usize] = 0.0;
        m[SECTION as usize] = 0.4;
        m[BOCAGE as usize] = 0.0;
        m[STRIPS as usize] = 0.2;
        // cattle ranches on the savannas (cerrado, llanos), few in the rainforest
        m[RANCH as usize] = if k <= 1 { 0.3 } else { 1.0 };
    } else if (3..=6).contains(&k) {
        // dry
        let desert = k <= 4;
        m[PIVOT as usize] = if desert { 4.0 } else { 2.0 };
        m[PASTURE as usize] = 1.5;
        m[PADDY as usize] = 0.05;
        m[BOCAGE as usize] = 0.0;
        m[PLANT as usize] = 0.0;
        m[VINE as usize] = 0.5;
        m[GREENH as usize] = if desert { 1.5 } else { 1.0 };
        // semi-arid grazing: ranches
        m[RANCH as usize] = if desert { 0.3 } else { 3.0 };
    } else if (16..=27).contains(&k) {
        // continental
        m[SHELTER as usize] = 1.6;
        m[PADDY as usize] = 0.15;
        m[PLANT as usize] = 0.0;
        m[GROVE as usize] = 0.4;
        m[VINE as usize] = 0.4;
        m[BOCAGE as usize] = 0.6;
        m[RANCH as usize] = 0.7;
    } else if k >= 28 {
        // polar: pastures and the core's clearings only
        for (i, mi) in m.iter_mut().enumerate() {
            if i != CORE as usize && i != PASTURE as usize {
                *mi = 0.0;
            }
        }
    }
    let dev = e.dev;
    m[GREENH as usize] *= 0.3 + 1.4 * dev;
    m[SMALL as usize] *= 1.6 - 1.0 * dev;
    m[SECTION as usize] *= 0.4 + 1.0 * dev;
    m[PIVOT as usize] *= (0.4 + 1.0 * dev) * (0.1 + 0.9 * (1.0 - smoothstep(700.0, 1200.0, e.precip)));
    let mut total = 0.0;
    for (i, wi) in w.iter_mut().enumerate() {
        *wi *= m[i] * (0.4 + 1.2 * u01k(e.id, 0xA600 + i as u64));
        total += *wi;
    }
    // one dominant system per ecoregion
    let v = u01k(e.id, 0xA6FF) * total;
    let mut acc = 0.0;
    for wi in w.iter_mut() {
        acc += *wi;
        if v < acc {
            *wi *= 3.0;
            break;
        }
    }
    w
}

/// The system of land-use region `rid` in ecoregion `e`.
pub fn pick_system(e: &EcoV, rid: u64) -> u32 {
    let w = weights(e);
    let total: f64 = w.iter().sum();
    let v = u01k(mix64(rid ^ e.id.rotate_left(29)), 0xA61) * total;
    let mut acc = 0.0;
    for (i, wi) in w.iter().enumerate() {
        acc += wi;
        if v < acc {
            return i as u32;
        }
    }
    sys::CORE
}

/// The crop mix of an ecoregion (by its climate and culture).
fn crop_mix(e: &EcoV) -> u32 {
    if e.temp > 20.0 && e.precip > 900.0 {
        4
    } else if e.precip < 400.0 {
        5
    } else if (7..=9).contains(&e.koppen) {
        3
    } else if e.precip < 650.0 && e.temp < 12.0 {
        2
    } else if e.arch == 7 {
        1
    } else {
        0
    }
}

/// Plan flags: climate variants of the kinds.
const F_MED: u32 = 1;
const F_DESERT: u32 = 2;
const F_TROPIC: u32 = 4;

/// The field system at a sample.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub sys: u32,
    pub e: EcoV,
    /// seed of the region's lattices, the crop mix, flags, season, size multiplier
    pub seed: u64,
    pub mix: u32,
    pub flags: u32,
    pub season: f64,
    pub scale: f64,
}

/// The pixel's own ecoregion (not the ecotone mosaic: systems follow the ecoregion border).
fn eco_of(s: &mut Stack) -> Option<EcoV> {
    let t = s.t;
    let (id, c) = if s.l.eco_edge < 0.0 { (t.eco.id2, t.eco.center2) } else { (t.eco.id, t.eco.center) };
    if id == 0 {
        return None;
    }
    let e = s.sm.eco.params(s.world, &s.sm.registry, s.cache, id, c);
    Some(EcoV::of(&e))
}

/// The kit's field system at the sample, None where the core's fields are drawn (or none).
pub fn plan(s: &mut Stack) -> Option<Plan> {
    let r = s.region?;
    if s.t.region.id == 0 || s.world.cfg.landuse.agriculture <= 0.0 || s.agri() <= 0.02 {
        return None;
    }
    let e = eco_of(s)?;
    let sys = pick_system(&e, s.t.region.id);
    if sys == sys::CORE {
        return None;
    }
    let seed = mix64(r.split.to_bits() ^ 0xA6A6);
    let mut flags = 0;
    if (7..=9).contains(&e.koppen) || (e.temp > 13.0 && e.precip < 900.0 && e.koppen > 6) {
        flags |= F_MED;
    }
    if e.temp > 18.0 && e.precip < 350.0 {
        flags |= F_DESERT;
    }
    if e.temp > 20.0 && e.precip > 900.0 {
        flags |= F_TROPIC;
    }
    Some(Plan { sys, e, seed, mix: crop_mix(&e), flags, season: r.season, scale: e.scale.sqrt() * (0.75 + 0.5 * u01k(seed, 3)) })
}

// ============================================================================= layers

/// Slot 4: where the kit draws the fields, the core's field layer stands down (`flat_ok` below
/// its threshold; restored ×4 in slot 5).
fn claim(s: &mut Stack) {
    if plan(s).is_some() {
        s.m.flat_ok *= 0.25;
    }
}

/// Slot 5: the system's parcels.
fn fields(s: &mut Stack) {
    let Some(pl) = plan(s) else { return };
    s.m.flat_ok *= 4.0;
    draw(s, &pl);
}

/// Slot 6: section-line farm tracks (every mile, east / north); the feedlots of ranches.
fn roads(s: &mut Stack) {
    let Some(pl) = plan(s) else { return };
    if pl.sys == sys::RANCH {
        feedlot(s, &pl);
        return;
    }
    if pl.sys != sys::SECTION {
        return;
    }
    let q = s.q_loc;
    let gsd = s.ctx.gsd;
    let ix = (q.x / MILE).round();
    let iy = (q.y / MILE).round();
    let d = (q.x - ix * MILE).abs().min((q.y - iy * MILE).abs());
    let steep = smoothstep(-0.02, 0.02, 0.45 - s.l.slope);
    let cov = band_cov(d, 4.0, s.l.fw.max(0.5 * gsd)) * steep * (1.0 - s.m.snow);
    // (no tracks across a kit's lakes: crater lakes, kettles)
    if cov <= 0.0 || s.class == crate::landcover::LAKE {
        return;
    }
    // gravel farm tracks (roads proper are the transport kit's)
    let (col, cls) = (srgb(168.0, 150.0, 122.0) * (1.0 + 0.05 * s.pf.detail), CLS_TRACK);
    s.m.road_cov = s.m.road_cov.max(cov);
    s.composite(Layer { cov, albedo: col, dh: 0.0, hmode: hmode::BLEND, cls, relit: 0.5, clear: 1.0, ..Default::default() });
}

const MILE: f64 = 1609.344;

/// The feedlot of a ranch region (seed `seed` of its plan, development `dev`): None or its
/// centre in the region frame (`q_rot`), half sizes of the pens' lot, the width of the lagoon
/// strip beyond its +x side (m).
pub fn feedlot_site(seed: u64, dev: f64) -> Option<(DVec2, f64, f64, f64)> {
    if u01k(seed, 0xFEE0) >= 0.2 + 0.45 * dev {
        return None;
    }
    let c = DVec2::new(1600.0 * (u01k(seed, 0xFEE1) - 0.5), 1000.0 * (u01k(seed, 0xFEE2) - 0.5));
    Some((c, 220.0 + 260.0 * u01k(seed, 0xFEE3), 140.0 + 160.0 * u01k(seed, 0xFEE4), 130.0 + 90.0 * u01k(seed, 0xFEE5)))
}

/// A feedlot (slot 6, in ranch regions; `docs/design/terrain-next.md` §5.8): pens of dark
/// manure in double rows along feed alleys with concrete bunks, cattle; a feed mill with silos
/// and silage bunkers at its −x end; manure lagoons in embankments beyond its +x end. No pivots.
fn feedlot(s: &mut Stack, pl: &Plan) {
    let Some((c, hx, hy, lag)) = feedlot_site(pl.seed, pl.e.dev) else { return };
    let d = s.q_rot - c;
    if d.x < -hx - 30.0 || d.x > hx + lag + 30.0 || d.y.abs() > hy + 30.0 || s.l.slope > 0.15 || s.class == crate::landcover::LAKE {
        return;
    }
    let (gsd, p) = (s.ctx.gsd, s.ctx.p);
    let fw = s.l.fw.max(0.5 * gsd);
    let seed = pl.seed;
    let soil = s.pal(pal::SOIL + 1);
    let (mut col, mut dh, mut cls, cov);
    if d.x < hx + 20.0 {
        // the lot: pens (40 × 60 m) in double rows along 12 m feed alleys
        cov = band_cov(d.x, hx, fw) * band_cov(d.y, hy, fw);
        let (pen_x, pen_y, alley) = (60.0, 40.0, 12.0);
        let per = 2.0 * pen_y + alley;
        let yy = (d.y + hy).rem_euclid(per);
        let manure = srgb(64.0, 52.0, 40.0)
            * (0.85 + 0.3 * u01k(hash2(seed ^ 0xFEE6, (d.x / pen_x).floor() as i64, ((d.y + hy) / per).floor() as i64 * 2 + (yy > pen_y) as i64), 1));
        let mut c0 = manure * (1.0 + 0.1 * perlin3(seed ^ 0xFEE7, p / 9.0) * band(9.0, gsd));
        // cattle: dark, brown and white dots
        let cell = 2.6;
        let (ci, cj) = ((d.x / cell).floor() as i64, (d.y / cell).floor() as i64);
        let hc = hash2(seed ^ 0xFEE8, ci, cj);
        if u01k(hc, 1) < 0.22 {
            let cc = [srgb(30.0, 26.0, 22.0), srgb(96.0, 60.0, 40.0), srgb(196.0, 186.0, 170.0)][(u01k(hc, 2) * 2.99) as usize];
            let off = DVec2::new(d.x - (ci as f64 + 0.5) * cell, d.y - (cj as f64 + 0.5) * cell);
            c0 = mixc(c0, cc, band_cov(off.length(), 0.8, fw) * band(cell, gsd));
        }
        // alleys and bunks (concrete), pen fences
        let alley_c = srgb(168.0, 160.0, 146.0);
        let in_alley = band_cov(yy - (pen_y + 0.5 * alley), 0.5 * alley, fw);
        let bunk = (band_cov(yy - (pen_y - 1.5), 1.5, fw) + band_cov(yy - (pen_y + alley + 1.5), 1.5, fw)).min(1.0);
        let fx = (d.x / pen_x - (d.x / pen_x).round()).abs() * pen_x;
        let fence = band_cov(fx, 0.25, fw).max(band_cov(yy.min(per - yy), 0.25, fw)) * band(1.0, gsd);
        col = mixc(c0, alley_c, in_alley * band(alley, gsd));
        col = mixc(col, srgb(196.0, 194.0, 186.0), bunk * band(3.0, gsd));
        col = mixc(col, srgb(150.0, 140.0, 126.0), fence * (1.0 - in_alley));
        cls = CLS_FARMYARD;
        dh = 0.3 * bunk;
        // the feed mill: a concrete pad, silage bunkers, a mill shed and silos
        if d.x < -hx + 80.0 {
            let m = DVec2::new(d.x + hx - 40.0, d.y - 0.45 * hy * (u01k(seed, 0xFEE9) - 0.5));
            let pad = band_cov(m.x, 40.0, fw) * band_cov(m.y, 60.0, fw);
            col = mixc(col, srgb(176.0, 172.0, 164.0), pad);
            // three bunkers (white plastic, black tyres)
            for k in 0..3 {
                let b = band_cov(m.x - 10.0, 22.0, fw) * band_cov(m.y - 14.0 * k as f64 - 6.0, 5.0, fw);
                col = mixc(col, if k == 1 { srgb(40.0, 40.0, 42.0) } else { srgb(222.0, 222.0, 216.0) }, b);
                dh += 2.5 * b;
            }
            let shed = band_cov(m.x + 18.0, 14.0, fw) * band_cov(m.y + 30.0, 22.0, fw);
            col = mixc(col, srgb(200.0, 204.0, 206.0), shed);
            dh += 9.0 * shed;
            for k in 0..2 {
                let o = m - DVec2::new(-24.0 + 12.0 * k as f64, -2.0);
                let silo = band_cov(o.length(), 4.5, fw);
                col = mixc(col, srgb(216.0, 218.0, 220.0), silo);
                dh += 22.0 * silo;
            }
            if pad > 0.5 {
                cls = CLS_BUILDING;
            }
        }
    } else {
        // manure lagoons: two basins in embankments
        let x0 = hx + 30.0 + 0.5 * (lag - 30.0);
        let yc = if d.y > 0.0 { 0.5 * hy } else { -0.5 * hy };
        let o = DVec2::new(d.x - x0, d.y - yc);
        let (ax, ay) = (0.5 * (lag - 30.0) - 6.0, 0.5 * hy - 14.0);
        // (a rounded rectangle's signed distance, m)
        let q = DVec2::new(o.x.abs() - ax + 12.0, o.y.abs() - ay + 12.0);
        let dist = q.max(DVec2::ZERO).length() + q.x.max(q.y).min(0.0) - 12.0;
        let water = (0.5 - dist / fw).clamp(0.0, 1.0);
        let bank = ((0.5 - (dist - 7.0) / fw).clamp(0.0, 1.0) - water).max(0.0);
        cov = (water + bank).min(1.0);
        col = mixc(soil * 0.9, srgb(58.0, 60.0, 42.0) * (1.0 + 0.08 * perlin3(seed ^ 0xFEEA, p / 30.0) * band(30.0, gsd)), water / cov.max(1e-6));
        dh = 1.2 * bank - 1.0 * water;
        cls = if water > 0.5 { crate::landcover::LAKE } else { crate::landcover::BARE };
    }
    if cov <= 0.0 {
        return;
    }
    s.composite(Layer { cov, albedo: col, dh, hmode: hmode::BLEND, cls, clear: 1.0, ..Default::default() });
}

const CLS_CROP: u8 = crate::landcover::CROP;
const CLS_RICE: u8 = crate::landcover::RICE_PADDY;
const CLS_ORCHARD: u8 = crate::landcover::ORCHARD;
const CLS_VINEYARD: u8 = crate::landcover::VINEYARD;
const CLS_PLANTATION: u8 = crate::landcover::PLANTATION;
const CLS_PASTURE: u8 = crate::landcover::PASTURE;
const CLS_GREENHOUSE: u8 = crate::landcover::GREENHOUSE;
const CLS_FALLOW: u8 = crate::landcover::FALLOW;
const CLS_HEDGEROW: u8 = crate::landcover::HEDGEROW;
const CLS_FARMYARD: u8 = crate::landcover::FARMYARD;
const CLS_BUILDING: u8 = crate::landcover::BUILDING;
const CLS_TRACK: u8 = crate::landcover::TRACK;
const CLS_CANAL: u8 = crate::landcover::CANAL;
const CLS_AQUACULTURE: u8 = crate::landcover::AQUACULTURE;
const CLS_RESERVOIR: u8 = crate::landcover::RESERVOIR;

/// The class of a kind (where nothing more specific is drawn).
fn kind_class(k: u32) -> u8 {
    match k {
        kind::CROP => CLS_CROP,
        kind::FALLOW => CLS_FALLOW,
        kind::PASTURE | kind::HAY => CLS_PASTURE,
        kind::ORCHARD | kind::OLIVE => CLS_ORCHARD,
        kind::VINE => CLS_VINEYARD,
        kind::PALM | kind::TEA | kind::RUBBER => CLS_PLANTATION,
        kind::PADDY => CLS_RICE,
        kind::GREENH => CLS_GREENHOUSE,
        kind::POND => CLS_AQUACULTURE,
        _ => CLS_FARMYARD,
    }
}

/// The sample, as the parcel shading sees it.
#[derive(Clone, Copy, Debug)]
struct Px {
    p: DVec3,
    gsd: f64,
    fw: f64,
    season: f64,
    moist: f64,
    slope: f64,
    ground: f64,
    soil: DVec3,
    grass: DVec3,
    grass_dry: DVec3,
    crown: DVec3,
    flags: u32,
}

/// A parcel: id, coordinates in its frame (along its rows' axis x), distance to its border,
/// centre (in the region frame), size, inside (pivots) and the distance to its block's border
/// (canals).
#[derive(Clone, Copy, Debug, Default)]
struct Parcel {
    id: u64,
    fx: f64,
    fy: f64,
    edge: f64,
    fc: DVec2,
    size: f64,
    inside: f64,
    bedge: f64,
    /// 1 for centre pivots (circular rows and wheel tracks)
    circ: f64,
}

fn draw(s: &mut Stack, pl: &Plan) {
    let d = SYS[pl.sys as usize];
    let m = s.m;
    let slope = s.l.slope;
    let slope_ok = 1.0 - smoothstep(d.slope[0], d.slope[1], slope);
    let rip = if pl.sys == sys::PADDY { 0.0 } else { m.riparian };
    let keep = m.natural_ok * slope_ok * (1.0 - rip) * (1.0 - m.woodlot) * m.shore_keep;
    let keep = smoothstep(0.45, 0.55, keep);
    if keep <= 0.0 {
        return;
    }
    let r = s.region.expect("a plan has a region");
    let agri = s.agri();
    let cult = smoothstep(0.02, 0.45, agri);
    let (gsd, fw, p) = (s.ctx.gsd, s.l.fw, s.ctx.p);
    let fwe = fw.max(0.6 * gsd);
    let loc = d.part == part::SECTION || d.part == part::PIVOT;
    let (q, ax, ay) = if loc { (s.q_loc, r.east, r.north) } else { (s.q_rot, r.ex, r.ey) };
    let px = Px {
        p,
        gsd,
        fw: fwe,
        season: pl.season,
        moist: s.t.moist,
        slope,
        ground: s.l.ground,
        // (a brown of the biome's soils: its second one alone, a red, drew maroon ploughland)
        soil: mixc(mixc(s.pal(pal::SOIL), s.pal(pal::SOIL + 2), 0.5), s.pal(pal::SOIL + 1), 0.3),
        grass: s.pal(pal::GRASS_WET),
        grass_dry: s.pal(pal::GRASS_DRY),
        crown: s.pal(pal::CROWN_DECID),
        flags: pl.flags,
    };
    let pc = parcel(pl, &d, q, p, fwe);
    // the parcel's size against the pixel: contrast towards the system's mean (as the core's
    // fields)
    let fsize = d.size[0].min(if d.part == part::VOR { d.size[0] } else { d.size[1] }) * if loc { 1.0 } else { pl.scale };
    let k = fsize / (fsize * fsize + 4.0 * gsd * gsd).sqrt();
    // (the parcels' colours fade to the mean sooner than their cover and height: small plots read as speckle)
    let kc = fsize / (fsize * fsize + 10.0 * gsd * gsd).sqrt();
    let mut col = DVec3::ZERO;
    let mut dh = 0.0;
    let mut cov = 0.0;
    let mut cls = kind_class(d.kinds[0]);
    if pc.inside > 0.0 {
        let c3 = r.center + ax * pc.fc.x + ay * pc.fc.y;
        let key = [c3.x.to_bits(), c3.y.to_bits(), c3.z.to_bits(), pl.seed ^ 0xA9];
        let sm = s.sm;
        let (mask, cluster) = *s.cache.fields.entry(key).or_insert_with(|| (sm.cultivated_mask(c3, gsd), worley2(pl.seed ^ 0xC1C, pc.fc, 700.0, 1.0).id));
        if mask < cult && u01k(pc.id, 5) >= 0.04 {
            let sh = explicit(pl, &d, &pc, &px, q, cluster);
            col = sh.0;
            dh = sh.1;
            cls = sh.2;
            cov = pc.inside;
        }
    }
    let inside_mean = if d.part == part::PIVOT {
        0.82 * 0.69
    } else if d.part == part::SECTION {
        1.0 - 0.21 * pivot_share(&pl.e)
    } else {
        1.0
    };
    let cov_m = lerp(cult * 0.9 * inside_mean, cov, k);
    if cov_m <= 0.0 {
        return;
    }
    let col = if kc < 0.999 {
        let mean = sys_mean(pl, &d, &px);
        if cov > 0.0 {
            mixc(mean, col, kc)
        } else {
            mean
        }
    } else {
        col
    };
    // the look of real fields: muted colours (a quarter towards their grey) and a broad variation
    // of soil and growth across the fields (as the core's)
    let fv = 1.0 + 0.09 * s.sm.pf_lazy(s.pf, crate::surface::PF_FIELD_VAR);
    // (and soil and moisture patches of tens of metres within every parcel)
    let mottle = if cov > 0.0 { 0.11 * perlin3(pc.id ^ 0x3077, p / 24.0) * band(24.0, gsd) } else { 0.0 };
    let col = field_tone(col) * (fv + mottle);
    let a = cov_m * keep;
    s.m.field_cov = a;
    // lit greenhouses (horticultural lamps; design §8 emission): an orange glow through the
    // roofs of some, the system's mean where the parcels are not resolved; gone at coarse pixels
    // (> ~300 m: the low zooms' town glow keeps its mean without them)
    let glow = if pl.sys == sys::GREENH && gsd < GH_GSD {
        let lit = if cls == CLS_GREENHOUSE && u01k(pc.id, 41) < GH_LIT { cov } else { 0.0 };
        lerp(cult * 0.9 * inside_mean * d.kw[0] * GH_LIT, lit, k) * keep * (1.0 - smoothstep(0.5 * GH_GSD, GH_GSD, gsd))
    } else {
        0.0
    };
    // the parcel is the instance (its crops, not its borders, yards and ponds)
    let inst = if cov > 0.0 && crate::landcover::group(cls) == crate::landcover::Group::Agriculture { crate::stack::instance_id(pc.id) } else { 0 };
    s.composite(Layer { cov: a, albedo: col, dh: dh * k - 0.6 * m.micro, hmode: hmode::ADD, cls, inst, emit: GH_GLOW * glow, ..Default::default() });
}

/// A field colour, muted: 25 % towards its luminance.
fn field_tone(c: DVec3) -> DVec3 {
    let y = c.dot(DVec3::new(0.2126, 0.7152, 0.0722));
    mixc(c, DVec3::splat(y), 0.25)
}

/// Share of the greenhouses lit at night, their glow (emission, linear: sodium lamps) and the
/// pixel size where it is gone.
const GH_LIT: f64 = 0.2;
const GH_GLOW: DVec3 = DVec3::new(0.25, 0.14, 0.05);
const GH_GSD: f64 = 300.0;

/// The share of quarter sections with a pivot (dry ecoregions).
fn pivot_share(e: &EcoV) -> f64 {
    0.08 + 0.72 * (1.0 - smoothstep(450.0, 800.0, e.precip))
}

// ============================================================================= parcels

fn parcel(pl: &Plan, d: &SysDef, q: DVec2, p: DVec3, fw: f64) -> Parcel {
    let seed = pl.seed;
    match d.part {
        part::GRID => {
            part_grid(seed, q, p, d.size[0] * pl.scale, d.size[1] * pl.scale, if pl.sys == sys::SHELTER || pl.sys == sys::PASTURE { 2.5 } else { 1.0 })
        }
        part::VOR => part_cad(seed, q, d.size[0] * pl.scale, d.size[1]),
        part::STRIP => part_strip(seed, q, d.size[0] * pl.scale, d.size[1] * pl.scale),
        part::PIVOT => part_pivot(seed, q, if u01k(seed, 4) < 0.5 { 805.0 } else { 402.0 }, fw),
        _ => part_section(seed, q, fw, pivot_share(&pl.e)),
    }
}

fn row_line(seed: u64, j: i64, h: f64) -> f64 {
    (j as f64 + 0.32 * (u01(hash2(seed ^ 0x40, 7, j)) - 0.5)) * h
}

fn col_line(seed: u64, j: i64, i: i64, w: f64) -> f64 {
    (i as f64 + 0.45 * (u01(hash2(seed ^ 0x41, j, i)) - 0.5)) * w
}

/// Jittered rows of jittered cells (the core's grid fields), some cut into up to `sub` + 1
/// strips.
fn part_grid(seed: u64, q0: DVec2, p: DVec3, w: f64, h: f64, sub: f64) -> Parcel {
    let wq = DVec2::new(perlin3(seed, p / 900.0), perlin3(seed ^ 1, p / 900.0));
    let q = q0 + wq * (0.05 * w.min(h));
    let mut j = (q.y / h).floor() as i64;
    if q.y < row_line(seed, j, h) {
        j -= 1;
    } else if q.y >= row_line(seed, j + 1, h) {
        j += 1;
    }
    let (y0, y1) = (row_line(seed, j, h), row_line(seed, j + 1, h));
    let shift = u01(hash1(seed, j)) * w;
    let x = q.x + shift;
    let mut i = (x / w).floor() as i64;
    if x < col_line(seed, j, i, w) {
        i -= 1;
    } else if x >= col_line(seed, j, i + 1, w) {
        i += 1;
    }
    let (x0, x1) = (col_line(seed, j, i, w), col_line(seed, j, i + 1, w));
    let (cw, ch) = (x1 - x0, y1 - y0);
    let fx = x - x0;
    let fy0 = q.y - y0;
    let hc = hash2(seed ^ 0x55, i, j);
    let nstrip = 1 + (u01k(hc, 1) * sub) as i64;
    let sh = ch / nstrip as f64;
    let k = (fy0 / sh).floor().clamp(0.0, nstrip as f64 - 1.0);
    let fy = fy0 - k * sh;
    let bedge = fx.min(cw - fx).min(fy0).min(ch - fy0);
    Parcel {
        id: mix64(hc ^ k as u64),
        fx,
        fy,
        edge: fx.min(cw - fx).min(fy).min(sh - fy),
        fc: DVec2::new(x0 + 0.5 * cw - shift, y0 + (k + 0.5) * sh),
        size: cw.min(sh),
        inside: 1.0,
        bedge,
        circ: 0.0,
    }
}

/// A block of a cadastral lattice: rows of jittered height `b`, each cut into blocks of
/// jittered width `b` with a per-row offset (T-junctions, no regular quilt). Returns the
/// block's id, centre (region frame), the sample's offset from it, its size, the distance to
/// its border.
fn cad_block(seed: u64, q: DVec2, b: f64) -> (u64, DVec2, DVec2, DVec2, f64) {
    let mut j = (q.y / b).floor() as i64;
    if q.y < row_line(seed, j, b) {
        j -= 1;
    } else if q.y >= row_line(seed, j + 1, b) {
        j += 1;
    }
    let (y0, y1) = (row_line(seed, j, b), row_line(seed, j + 1, b));
    let shift = u01(hash1(seed, j)) * b;
    let x = q.x + shift;
    let mut i = (x / b).floor() as i64;
    if x < col_line(seed, j, i, b) {
        i -= 1;
    } else if x >= col_line(seed, j, i + 1, b) {
        i += 1;
    }
    let (x0, x1) = (col_line(seed, j, i, b), col_line(seed, j, i + 1, b));
    let size = DVec2::new(x1 - x0, y1 - y0);
    let c = DVec2::new(x0 + 0.5 * size.x - shift, y0 + 0.5 * size.y);
    let rel = q - c;
    let edge = (0.5 * size.x - rel.x.abs()).min(0.5 * size.y - rel.y.abs());
    (hash2(seed ^ 0xCAD, i, j), c, rel, size, edge)
}

/// Cadastral parcels (bocage, smallholder plots, terraces, groves): blocks of about 2.6 ×
/// `cell` (`cad_block`), each in its own frame turned by up to ±13°, cut by up to five
/// recursive splits (the longer side at 0.3–0.7; with `aspect_max` > 1, sometimes the shorter
/// one: long strips) into fields of varied size around `cell`. Mostly four-sided fields of
/// different sizes and orientations, as real field maps, not a Voronoi honeycomb.
fn part_cad(seed: u64, q: DVec2, cell: f64, aspect_max: f64) -> Parcel {
    let b = 2.6 * cell;
    let (hb, c, rel, size, bedge) = cad_block(seed, q, b);
    let ang = 0.45 * (u01k(hb, 1) - 0.5);
    let (sa, ca) = ang.sin_cos();
    // the sample and the block's bounding box in the turned frame
    let u = rel.x * ca + rel.y * sa;
    let v = -rel.x * sa + rel.y * ca;
    let hx = 0.5 * (size.x * ca.abs() + size.y * sa.abs());
    let hy = 0.5 * (size.x * sa.abs() + size.y * ca.abs());
    let (mut ax0, mut ay0, mut ax1, mut ay1) = (-hx, -hy, hx, hy);
    let mut id = hb;
    let strip_p = 0.6 * (aspect_max - 1.0);
    for _ in 0..5 {
        let (w, h) = (ax1 - ax0, ay1 - ay0);
        let target = cell * (0.65 + 0.7 * u01k(id, 2));
        if w * h < target * target {
            break;
        }
        // split the longer side, or (strips) the shorter one while it stays wide enough
        let mut split_x = w >= h;
        if u01k(id, 4) < strip_p && w.min(h) > 0.9 * target {
            split_x = !split_x;
        }
        let t = 0.3 + 0.4 * u01k(id, 3);
        if split_x {
            let xs = ax0 + t * w;
            if u < xs {
                ax1 = xs;
                id = mix64(id ^ 0x11);
            } else {
                ax0 = xs;
                id = mix64(id ^ 0x22);
            }
        } else {
            let ys = ay0 + t * h;
            if v < ys {
                ay1 = ys;
                id = mix64(id ^ 0x33);
            } else {
                ay0 = ys;
                id = mix64(id ^ 0x44);
            }
        }
    }
    let lc = DVec2::new(0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1));
    let edge = (u - ax0).min(ax1 - u).min(v - ay0).min(ay1 - v).min(bedge);
    let fc = c + DVec2::new(lc.x * ca - lc.y * sa, lc.x * sa + lc.y * ca);
    Parcel { id, fx: u - lc.x, fy: v - lc.y, edge, fc, size: (ax1 - ax0).min(ay1 - ay0), inside: 1.0, bedge, circ: 0.0 }
}

/// `part_cad` for the core's irregular fields.
pub(crate) fn cadastral_parcel(seed: u64, q: DVec2, cell: f64, aspect: f64) -> (u64, f64, f64, f64, DVec2) {
    let pc = part_cad(seed, q, cell, aspect);
    (pc.id, pc.fx, pc.fy, pc.edge, pc.fc)
}

/// Furlongs (cadastral blocks of about `w`) of parallel strips `sw` wide, along a per-furlong
/// axis, with a reverse-S in some.
fn part_strip(seed: u64, q: DVec2, w: f64, sw0: f64) -> Parcel {
    let (fid, c0, rel, _, bedge) = cad_block(seed ^ 0x57, q, w);
    let ang = if u01k(fid, 1) < 0.5 { 0.0 } else { std::f64::consts::FRAC_PI_2 } + 0.3 * (u01k(fid, 2) - 0.5);
    let (sa, ca) = ang.sin_cos();
    let a = rel.x * ca + rel.y * sa;
    let c = -rel.x * sa + rel.y * ca;
    let amp = if u01k(fid, 4) < 0.6 { 0.03 * w } else { 0.0 };
    let c2 = c + amp * (std::f64::consts::TAU * a / w).sin();
    let sw = sw0 * (0.6 + 0.8 * u01k(fid, 3));
    let n = (c2 / sw).floor();
    let fy = c2 - n * sw;
    let fc = c0 + DVec2::new(-sa, ca) * ((n + 0.5) * sw);
    Parcel { id: hash2(fid ^ 0x5751, n as i64, 0), fx: a, fy, edge: fy.min(sw - fy).min(bedge), fc, size: sw, inside: 1.0, bedge, circ: 0.0 }
}

/// Centre pivots in a square grid of `s` (some cells empty).
fn part_pivot(seed: u64, q: DVec2, s: f64, fw: f64) -> Parcel {
    let cq = (q / s).floor();
    let c = (cq + 0.5) * s;
    let rel = q - c;
    let d = rel.length();
    let rad = 0.47 * s;
    let hc = hash2(seed ^ 0x9, cq.x as i64, cq.y as i64);
    let inside = if u01k(hc, 3) < 0.82 { band_cov(d, rad, fw) } else { 0.0 };
    let nsec = 1 + (u01k(hc, 1) * 3.0) as i64;
    let ang = rel.y.atan2(rel.x) + std::f64::consts::PI + u01k(hc, 2) * 6.0;
    let sec = ((ang / std::f64::consts::TAU * nsec as f64).floor() as i64).rem_euclid(nsec);
    let edge = (rad - d).abs();
    Parcel { id: mix64(hc ^ sec as u64), fx: rel.x, fy: rel.y, edge, fc: c, size: s, inside, bedge: edge, circ: 1.0 }
}

/// 1-mile sections of the east / north frame: quarter sections, halves or forties, a pivot in
/// some quarters (`pivot_p`; the corners stay unfarmed).
fn part_section(seed: u64, q: DVec2, fw: f64, pivot_p: f64) -> Parcel {
    let half = 0.5 * MILE;
    let sc = (q / MILE).floor();
    let hs = hash2(seed ^ 0x5EC, sc.x as i64, sc.y as i64);
    let l = q - sc * MILE;
    let qi = (l / half).floor().clamp(DVec2::ZERO, DVec2::ONE);
    let hq = hash2(hs, qi.x as i64, qi.y as i64);
    let lq = l - qi * half;
    let o = sc * MILE + qi * half;
    if u01k(hq, 7) < pivot_p {
        let rel = lq - DVec2::splat(0.5 * half);
        let d = rel.length();
        let rad = 0.48 * half;
        let edge = (rad - d).abs();
        return Parcel {
            id: hq,
            fx: rel.x,
            fy: rel.y,
            edge,
            fc: o + DVec2::splat(0.5 * half),
            size: half,
            inside: band_cov(d, rad, fw),
            bedge: edge,
            circ: 1.0,
        };
    }
    let mode = u01k(hq, 1);
    let n = if mode < 0.45 {
        1.0
    } else if mode < 0.8 {
        2.0
    } else {
        4.0
    };
    let along_x = u01k(hq, 2) < 0.5;
    // strips across the axis: (along, across)
    let (a, c) = if along_x { (lq.x, lq.y) } else { (lq.y, lq.x) };
    let sw = half / n;
    let k = (c / sw).floor().clamp(0.0, n - 1.0);
    let fy = c - k * sw;
    let cc = (k + 0.5) * sw;
    let fc = o + if along_x { DVec2::new(0.5 * half, cc) } else { DVec2::new(cc, 0.5 * half) };
    let bedge = lq.x.min(half - lq.x).min(lq.y).min(half - lq.y);
    Parcel { id: mix64(hq ^ k as u64), fx: a, fy, edge: a.min(half - a).min(fy).min(sw - fy), fc, size: sw, inside: 1.0, bedge, circ: 0.0 }
}

// ============================================================================= shading

/// A kind drawn from the system's kinds by `u` (fallow more common late in the season).
fn pick_kind(d: &SysDef, u: f64, season: f64) -> u32 {
    let mut w = d.kw;
    for (i, k) in d.kinds.iter().enumerate() {
        if *k == kind::FALLOW {
            w[i] *= 0.5 + season;
        }
    }
    let total = w[0] + w[1] + w[2] + w[3];
    let v = u * total;
    let mut acc = 0.0;
    for (i, wi) in w.iter().enumerate().take(4) {
        acc += *wi;
        if v < acc {
            return d.kinds[i];
        }
    }
    d.kinds[0]
}

/// A crop of the mix by `u`.
fn pick_crop(mix: u32, u: f64) -> u32 {
    let (c, w) = MIXES[mix as usize];
    let mut acc = 0.0;
    for i in 0..4 {
        acc += w[i];
        if u < acc {
            return c[i];
        }
    }
    c[3]
}

/// The colour of a stage table at phase `ph` (0 spring, 0.5 summer, 1 autumn).
fn stage(c: &[[f64; 3]; 3], ph: f64) -> DVec3 {
    if ph < 0.5 {
        mixc(lin(c[0]), lin(c[1]), ph * 2.0)
    } else {
        mixc(lin(c[1]), lin(c[2]), ph * 2.0 - 1.0)
    }
}

/// The mean colour of the system (what its parcels average to).
fn sys_mean(pl: &Plan, d: &SysDef, px: &Px) -> DVec3 {
    let mut acc = DVec3::ZERO;
    let mut wt = 0.0;
    for i in 0..4 {
        let w = d.kw[i] * if d.kinds[i] == kind::FALLOW { 0.5 + px.season } else { 1.0 };
        acc += kind_mean(pl, d.kinds[i], px) * w;
        wt += w;
    }
    acc / wt.max(1e-6)
}

fn kind_mean(pl: &Plan, k: u32, px: &Px) -> DVec3 {
    match k {
        kind::CROP => {
            let (c, w) = MIXES[pl.mix as usize];
            let mut acc = DVec3::ZERO;
            for i in 0..4 {
                acc += stage(&CROPS[c[i] as usize].0, px.season) * w[i];
            }
            acc
        }
        kind::FALLOW => mixc(px.soil * 0.8, mixc(px.soil, px.grass_dry, 0.5), 0.5 - 0.4 * px.season),
        kind::PASTURE => pasture_col(px) * 0.97,
        kind::HAY => pasture_col(px) * 1.05,
        kind::ORCHARD => mixc(mixc(px.grass, px.soil, 0.3), px.crown * 1.1, 0.4),
        kind::OLIVE => mixc(px.soil * 1.15, srgb(104.0, 112.0, 84.0), 0.3),
        kind::VINE => mixc(px.soil * 1.2, vine_col(px.season, 0.5), 0.35),
        kind::PALM => mixc(srgb(92.0, 100.0, 60.0), srgb(58.0, 92.0, 44.0), 0.55),
        kind::TEA => srgb(118.0, 156.0, 58.0) * 0.92,
        kind::PADDY => paddy_mean(px.season),
        kind::GREENH => srgb(214.0, 216.0, 212.0) * 0.9,
        kind::POND => srgb(52.0, 74.0, 76.0),
        kind::YARD => srgb(172.0, 140.0, 108.0),
        _ => srgb(40.0, 70.0, 38.0) * 0.9,
    }
}

fn pasture_col(px: &Px) -> DVec3 {
    mixc(px.grass_dry, px.grass, smoothstep(0.25, 0.65, px.moist))
}

fn vine_col(season: f64, u: f64) -> DVec3 {
    let autumn = if u < 0.5 { srgb(156.0, 118.0, 52.0) } else { srgb(132.0, 64.0, 48.0) };
    if season < 0.5 {
        mixc(srgb(104.0, 124.0, 60.0), srgb(70.0, 98.0, 46.0), season * 2.0)
    } else {
        mixc(srgb(70.0, 98.0, 46.0), autumn, season * 2.0 - 1.0)
    }
}

/// Paddy colour by state: 0 flooded, 1 transplanted, 2 green, 3 ripe, 4 harvested.
fn paddy_col(st: u32) -> DVec3 {
    match st {
        0 => srgb(64.0, 84.0, 94.0),
        1 => srgb(76.0, 100.0, 86.0),
        2 => srgb(80.0, 116.0, 50.0),
        3 => srgb(188.0, 172.0, 92.0),
        _ => srgb(150.0, 128.0, 88.0),
    }
}

/// The paddy state at phase `ph`.
fn paddy_state(ph: f64) -> u32 {
    if ph < 0.18 {
        0
    } else if ph < 0.32 {
        1
    } else if ph < 0.66 {
        2
    } else if ph < 0.84 {
        3
    } else {
        4
    }
}

fn paddy_mean(season: f64) -> DVec3 {
    // phases spread ±0.15 around the season: the mean of five draws
    let mut acc = DVec3::ZERO;
    for i in 0..5 {
        acc += paddy_col(paddy_state((season + 0.075 * (i as f64 - 2.0)).clamp(0.0, 1.0)));
    }
    acc / 5.0
}

/// Contour terraces: (bench index, position across the bench 0 (lip) .. 1, bench width (m),
/// explicit weight).
fn terrace(px: &Px, dhs: f64, phase: f64) -> (f64, f64, f64, f64) {
    let t = px.ground / dhs + phase;
    let i = t.floor();
    let f = t - i;
    let wb = dhs / px.slope.max(0.02);
    let ex = smoothstep(1.2 * px.gsd, 3.0 * px.gsd, wb);
    (i, f, wb, ex)
}

/// The explicit parcel: colour, height, class.
fn explicit(pl: &Plan, d: &SysDef, pc: &Parcel, px: &Px, q: DVec2, cluster: u64) -> (DVec3, f64, u8) {
    let (p, gsd, fw) = (px.p, px.gsd, px.fw);
    let mut id = pc.id;
    let mut k = pick_kind(d, u01k(id, 40), px.season);
    // climate variants: olives in mediterranean climates, dates in deserts, apples elsewhere
    if k == kind::OLIVE && px.flags & F_MED == 0 {
        k = if px.flags & F_DESERT != 0 { kind::PALM } else { kind::ORCHARD };
    }
    let mut col;
    let mut dh = 0.0;
    let mut cls = kind_class(k);
    // terraces on slopes (benches: a parcel within a parcel)
    let mut lip = 1e9;
    let mut t_dh = 0.0;
    if d.terraces > 0.0 && px.slope > 0.07 {
        let dhs = if pl.sys == sys::PADDY { 1.2 + 1.0 * u01k(pl.seed, 40) } else { 1.8 + 1.6 * u01k(pl.seed, 40) };
        let (i, f, wb, ex) = terrace(px, dhs, u01k(pl.seed, 41));
        lip = f * wb / ex.max(1e-3) + (1.0 - ex) * 1e9;
        t_dh = (0.5 - f) * dhs * ex;
        id = mix64(id ^ (i as i64 as u64));
    }
    // rows across the parcel's long axis or along it
    let rot = u01k(id, 9) < 0.3;
    let (ru, rv) = if rot { (pc.fy, pc.fx) } else { (pc.fx, pc.fy) };
    match k {
        kind::CROP | kind::HAY => {
            let u_crop = if u01k(id, 16) < 0.5 { u01k(cluster, 6) } else { u01k(id, 6) };
            let c = if k == kind::HAY { 6 } else { pick_crop(pl.mix, u_crop) };
            let (cc, row) = CROPS[c as usize];
            let ph = (px.season + 0.3 * (u01k(id, 20) - 0.5)).clamp(0.0, 1.0);
            col = stage(&cc, ph) * (0.92 + 0.16 * u01k(id, 8));
            col *= 1.0 + 0.2 * perlin3(id, p / 35.0) * band(35.0, gsd) + 0.11 * perlin3(id ^ 1, p / 120.0) * band(120.0, gsd);
            // soil and growth texture (band-limited)
            col *= 1.0 + 0.15 * perlin3(id ^ 0x7E1, p / 7.0) * band(7.0, gsd) + 0.10 * perlin3(id ^ 0x7E2, p / 2.5) * band(2.5, gsd);
            let gz = perlin3(id ^ 0x6A0, p / 55.0) * band(55.0, gsd);
            // growth zones (soil, moisture): yellower and greener patches of tens of metres
            col = mixc(col, col * DVec3::new(1.12, 1.04, 0.82), 0.55 * smoothstep(0.0, 0.6, gz));
            col = mixc(col, col * DVec3::new(0.84, 0.92, 0.86), 0.5 * smoothstep(0.0, 0.6, -gz));
            // headland: a strip along the border driven more often (slightly darker)
            let hw = 8.0 + 10.0 * u01k(id, 11);
            if pc.edge < hw {
                col *= 0.96;
            }
            // pivots: circular rows
            let rc = (pc.fx * pc.fx + pc.fy * pc.fy).sqrt();
            let u = if pc.circ > 0.0 { rc } else { ru };
            col *= 1.0 + row[1] * (std::f64::consts::TAU * u / row[0]).sin() * band(row[0], gsd);
            if pc.edge >= hw && (c == 0 || c == 2 || c == 3) {
                let sp = 18.0 + 6.0 * (u01k(id, 15) * 3.0).floor();
                let mt = u.rem_euclid(sp);
                let tl = band_cov(mt - 0.9, 0.22, fw) + band_cov(mt - 2.7, 0.22, fw);
                let vis = band(1.2, gsd).max(0.35 * band(sp, gsd));
                col = mixc(col, col * 0.78, tl * vis);
            }
            dh = row[2] * (1.0 - (2.0 * ph - 1.0).abs()).max(0.25);
            if pc.circ > 0.0 {
                // the wheel tracks of the pivot's towers
                let sp = 48.0 + 8.0 * u01k(id, 17);
                let mt = (rc + 0.5 * sp).rem_euclid(sp) - 0.5 * sp;
                let vis = band(1.5, gsd).max(0.3 * band(sp, gsd));
                col = mixc(col, px.soil * 0.9, band_cov(mt, 0.5, fw) * vis * smoothstep(20.0, 30.0, rc));
            }
            if k == kind::HAY {
                cls = CLS_PASTURE;
            }
        }
        kind::FALLOW => {
            if u01k(id, 21) < 0.5 + 0.4 * px.season {
                // ploughed: furrows, plough lands, clods
                col = px.soil * (0.74 + 0.12 * u01k(id, 22));
                let sp = 0.35 + 0.15 * u01k(id, 23);
                let fur = (std::f64::consts::TAU * ru / sp).sin() * band(sp, gsd);
                col *= 1.0 + 0.16 * fur + 0.10 * perlin3(id ^ 0xC10, p / 2.5) * band(2.5, gsd);
                let land = 22.0 + 18.0 * u01k(id, 24);
                col *= 1.0 + 0.05 * (std::f64::consts::TAU * ru / (2.0 * land)).sin().signum() * band(land, gsd);
                dh = 0.06 * fur;
            } else {
                // weedy fallow
                let n = perlin3(id ^ 0xF0A, p / 18.0) * band(18.0, gsd);
                col = mixc(px.soil, px.grass_dry, 0.5 + 0.3 * n) * (1.0 + 0.08 * perlin3(id ^ 0xF0B, p / 4.0) * band(4.0, gsd));
                dh = 0.2;
            }
        }
        kind::PASTURE => {
            let sh = pasture(pl, pc, px, id, q);
            col = sh.0;
            dh = sh.1;
            cls = sh.2;
        }
        kind::ORCHARD | kind::OLIVE => {
            let olive = k == kind::OLIVE;
            let (sx, sy, r) = if olive {
                let s0 = 7.0 + 3.0 * u01k(id, 10);
                (s0, s0, 2.4 + 0.8 * u01k(id, 12))
            } else {
                (5.0 + 2.0 * u01k(id, 10), 3.6 + 1.0 * u01k(id, 12), 1.6 + 0.5 * u01k(id, 13))
            };
            let ground = if olive { px.soil * (1.12 + 0.08 * (std::f64::consts::TAU * ru / 0.5).sin() * band(0.5, gsd)) } else { mixc(px.grass, px.soil, 0.3) };
            let crown_c = if olive { srgb(104.0, 112.0, 84.0) } else { px.crown * 1.1 };
            let cv = tree_grid(ru, rv, sx, sy, r, gsd, fw);
            let tex = 1.0 + 0.12 * perlin3(id ^ 0x0C1, p / 1.3) * band(1.3, gsd);
            col = mixc(ground, crown_c * tex, cv);
            dh = (if olive { 5.0 } else { 4.0 }) * cv;
        }
        kind::VINE => {
            let sp = 2.2 + 0.6 * u01k(id, 10);
            let hw = 0.45 + 0.15 * u01k(id, 12);
            // on slopes, half the vineyards follow the contours
            let (v, sp_act) = if px.slope > 0.12 && u01k(id, 13) < 0.5 {
                let dz = sp * 0.25;
                (px.ground / dz, dz / px.slope.max(0.05))
            } else {
                (ru / sp, sp)
            };
            let mrow = (v - v.floor() - 0.5).abs() * sp_act;
            let ex = band(sp_act, gsd);
            let cv = band_cov(mrow, hw, fw) * ex + (1.0 - ex) * (2.0 * hw / sp_act).min(1.0);
            let ground = if u01k(id, 14) < 0.4 { mixc(px.grass, px.grass_dry, 0.5) } else { px.soil * 1.25 };
            col = mixc(ground, vine_col(px.season, u01k(id, 15)), cv);
            dh = 1.6 * cv;
        }
        kind::PALM => {
            let fine = 1.0 + 0.08 * perlin3(id ^ 0x7A1, p / 6.0) * band(6.0, gsd);
            if px.flags & F_DESERT != 0 {
                // date palms in a square grid on sand
                let cv = tree_grid(ru, rv, 8.0, 8.0, 3.4, gsd, fw);
                col = mixc(srgb(186.0, 164.0, 124.0) * fine, srgb(98.0, 110.0, 66.0), cv);
                dh = 9.0 * cv;
                cls = CLS_ORCHARD;
            } else {
                // oil palm: star crowns on a triangular grid, stands of different age
                let age = 0.35 + 0.65 * u01k(id, 34);
                let (sx, sy) = (9.0, 7.8);
                let j = (rv / sy).round();
                let off = if (j as i64).rem_euclid(2) == 0 { 0.0 } else { 0.5 * sx };
                let i = ((ru - off) / sx).round();
                let rel = DVec2::new(ru - off - i * sx, rv - j * sy);
                let ang = rel.y.atan2(rel.x) + u01k(mix64(id ^ (i as i64 as u64) ^ ((j as i64 as u64) << 20)), 1) * 6.3;
                let r_eff = 4.4 * age * (0.82 + 0.18 * (8.0 * ang).cos());
                let dd = rel.length();
                let ex = band(sx, gsd);
                let crown = band_cov(dd, r_eff, fw) * ex + (1.0 - ex) * (0.75 * age);
                let shade = 0.85 + 0.25 * (1.0 - (dd / r_eff.max(0.1)).min(1.0)) * ex;
                col = mixc(srgb(92.0, 100.0, 60.0) * fine, srgb(58.0, 92.0, 44.0) * shade * fine, crown);
                dh = (3.0 + 9.0 * age) * crown;
            }
        }
        kind::RUBBER => {
            // closed rows of crowns, narrow gaps between the rows
            let sp = 7.0;
            let m = (rv / sp - (rv / sp).floor() - 0.5).abs() * sp;
            let ex = band(sp, gsd);
            let cv = band_cov(m, 2.9, fw) * ex + (1.0 - ex) * (5.8 / sp);
            let tex = 1.0 + 0.14 * perlin3(id ^ 0x2BB, p / 3.0) * band(3.0, gsd);
            col = mixc(srgb(70.0, 66.0, 50.0), srgb(40.0, 70.0, 38.0) * tex, cv);
            dh = 16.0 * cv;
        }
        kind::TEA => {
            // clipped hedges along the contours (straight rows on flat ground)
            let (v, sp_act) = if px.slope > 0.06 {
                let dz = 1.5 * 0.2;
                (px.ground / dz, dz / px.slope.max(0.05))
            } else {
                (rv / 1.5, 1.5)
            };
            let m = (v - v.floor() - 0.5).abs() * sp_act;
            let ex = band(sp_act, gsd);
            let cv = band_cov(m, 0.42 * sp_act, fw) * ex + (1.0 - ex) * 0.84;
            let tex = 1.0 + 0.08 * perlin3(id ^ 0x7EA, p / 2.0) * band(2.0, gsd);
            col = mixc(srgb(96.0, 82.0, 62.0), srgb(118.0, 156.0, 58.0) * tex, cv);
            dh = 1.0 * cv;
        }
        kind::PADDY => {
            let ph = (px.season + 0.3 * (u01k(id, 31) - 0.5)).clamp(0.0, 1.0);
            let st = paddy_state(ph);
            col = paddy_col(st) * (0.94 + 0.12 * u01k(id, 32));
            let fine = perlin3(id ^ 0x9AD, p / 7.0) * band(7.0, gsd);
            col *= 1.0 + 0.06 * fine;
            if st == 1 {
                // seedling rows over the water
                let r = (std::f64::consts::TAU * ru / 0.3).sin() * band(0.3, gsd);
                col = mixc(col, paddy_col(2), 0.25 + 0.2 * r);
            }
            // plots cut by bunds (on terraces: the benches)
            let mut bd = pc.edge.min(lip);
            if lip > 1e8 {
                let sx = 18.0 + 22.0 * u01k(id, 33);
                let sy = sx * (1.2 + 0.6 * u01k(id, 35));
                let (mx, my) = ((pc.fx + 1000.0).rem_euclid(sx), (pc.fy + 1000.0).rem_euclid(sy));
                bd = bd.min(mx.min(sx - mx).min(my).min(sy - my) / band(sx, gsd).max(1e-3));
            }
            let bund = band_cov(bd, 0.4, fw);
            col = mixc(col, srgb(122.0, 116.0, 82.0), bund);
            dh = 0.3 * bund - 0.05;
        }
        kind::GREENH => {
            let inside = ((pc.edge - 1.8) / fw + 0.5).clamp(0.0, 1.0);
            let bw = 8.0 + 1.6 * u01k(id, 10);
            let mbay = ru / bw - (ru / bw).floor();
            let ex = band(bw, gsd);
            let roof = 1.0 + 0.10 * (std::f64::consts::TAU * mbay).cos() * ex;
            let gutter = band_cov(mbay.min(1.0 - mbay) * bw, 0.35, fw) * ex;
            let u = u01k(id, 11);
            let base = if u < 0.15 {
                srgb(150.0, 156.0, 150.0)
            } else if u < 0.45 {
                srgb(232.0, 234.0, 230.0)
            } else if u < 0.8 {
                srgb(212.0, 216.0, 214.0)
            } else {
                srgb(240.0, 238.0, 226.0)
            };
            let dirt = 1.0 + 0.06 * perlin3(id ^ 0x6E1, p / 10.0) * band(10.0, gsd);
            let gh = mixc(base * roof * dirt, srgb(118.0, 122.0, 124.0), gutter);
            col = mixc(px.soil * 1.1, gh, inside);
            dh = 4.0 * inside;
            if inside < 0.5 {
                cls = CLS_CROP;
            }
        }
        kind::POND => {
            let m_in = 3.0 + 3.0 * u01k(id, 10);
            let rp = 18.0 + 30.0 * u01k(id, 12);
            let dc = (q - pc.fc).abs() - DVec2::splat(rp);
            let w = ((pc.edge - m_in) / fw + 0.5).clamp(0.0, 1.0) * ((-dc.x.max(dc.y)) / fw + 0.5).clamp(0.0, 1.0);
            let water = if pl.sys == sys::PADDY || pl.sys == sys::GREENH { srgb(62.0, 90.0, 72.0) } else { srgb(48.0, 70.0, 74.0) };
            col = mixc(srgb(140.0, 128.0, 100.0), water * (0.9 + 0.2 * u01k(id, 11)), w);
            dh = -0.6 * w;
            cls = if w > 0.5 {
                if pl.sys == sys::PADDY || pl.sys == sys::GREENH {
                    CLS_AQUACULTURE
                } else {
                    CLS_RESERVOIR
                }
            } else {
                CLS_FALLOW
            };
        }
        kind::YARD => {
            let sh = yard(pc, px, id, q);
            col = sh.0;
            dh = sh.1;
            cls = sh.2;
        }
        _ => {
            col = px.soil;
        }
    }
    // stone risers of dry terraces, the bench as a whole lowered / raised to its level
    if lip < 1e8 {
        if pl.sys == sys::TERRACE {
            let w = band_cov(lip, 0.5, fw);
            col = mixc(col, srgb(150.0, 144.0, 132.0) * (0.9 + 0.2 * perlin3(id ^ 0x5701, p / 1.5) * band(1.5, gsd)), w);
        }
        dh += t_dh;
    }
    // scattered trees (farmed parkland)
    // (and in warm pastures: savanna trees)
    let (trees, tcell) = if (pl.sys == sys::PASTURE || pl.sys == sys::RANCH) && pl.e.temp > 18.0 { (0.12, 60.0) } else { (d.trees, 30.0) };
    if trees > 0.0 {
        let (tc, th) = scatter_tree(pl.seed, q, tcell, trees, gsd, fw);
        if tc > 0.0 {
            col = mixc(col, px.crown * (0.85 + 0.25 * u01k(pc.id, 50)) * (0.72 + 0.4 * (th / 10.0).min(1.0)), tc);
            dh = lerp(dh, th, tc);
        }
    }
    // the border
    let bsel = u01k(pc.id, 0xB0);
    let hedge_m = 0.3 + 0.7 * pl.e.hedges;
    let p0 = d.bp[0] * if d.borders[0] == border::HEDGE || d.borders[0] == border::BELT { hedge_m } else { 1.0 };
    let b = if bsel < p0 {
        d.borders[0]
    } else if bsel < p0 + d.bp[1] {
        d.borders[1]
    } else {
        border::MARGIN
    };
    // canals along the blocks of paddies, the border of the block
    let edge = if b == border::CANAL { pc.bedge } else { pc.edge };
    let (col, dh, cls) = draw_border(b, edge, col, dh, cls, pc, px);
    (col, dh, cls)
}

/// Coverage of a grid of round crowns (spacing `sx` × `sy`, radius `r`) at `(u, v)`: explicit or
/// its mean.
fn tree_grid(u: f64, v: f64, sx: f64, sy: f64, r: f64, gsd: f64, fw: f64) -> f64 {
    let gx = (u / sx).round() * sx;
    let gy = (v / sy).round() * sy;
    let d = DVec2::new(u - gx, v - gy).length();
    let ex = band(sx.min(sy), gsd);
    let mean = (std::f64::consts::PI * r * r / (sx * sy)).min(1.0);
    band_cov(d, r, fw) * ex + (1.0 - ex) * mean
}

/// Scattered trees on a jittered lattice of `cell` (density `dens`): coverage and crown height.
fn scatter_tree(seed: u64, q: DVec2, cell: f64, dens: f64, gsd: f64, fw: f64) -> (f64, f64) {
    let ci = (q / cell).floor();
    let h = hash2(seed ^ 0x7EE, ci.x as i64, ci.y as i64);
    let ex = band(cell, gsd);
    let mean = dens * std::f64::consts::PI * 0.0147;
    let mut cv = 0.0;
    let mut th = 0.0;
    if u01k(h, 1) < dens && ex > 0.0 {
        let c = (ci + DVec2::new(0.25 + 0.5 * u01k(h, 2), 0.25 + 0.5 * u01k(h, 3))) * cell;
        let r = (0.09 + 0.06 * u01k(h, 4)) * cell;
        let d = (q - c).length();
        cv = band_cov(d, r, fw);
        th = (5.0 + 5.0 * u01k(h, 5)) * (1.0 - (d / r).min(1.0).powi(2)).sqrt();
    }
    (cv * ex + (1.0 - ex) * mean, th * ex + (1.0 - ex) * 7.5)
}

/// A pasture: grass grazed in patches, livestock trails converging on a trough, a farm dam in
/// some paddocks.
fn pasture(pl: &Plan, pc: &Parcel, px: &Px, id: u64, q: DVec2) -> (DVec3, f64, u8) {
    let (p, gsd, fw) = (px.p, px.gsd, px.fw);
    let mut col = pasture_col(px) * (0.86 + 0.28 * u01k(id, 8));
    col *= 1.0 + 0.08 * perlin3(id ^ 0x9A5, p / 14.0) * band(14.0, gsd) + 0.05 * perlin3(id ^ 0x9A6, p / 4.0) * band(4.0, gsd);
    // tussocks and grazed patches
    let tus = perlin3(id ^ 0x9A7, p / 1.7) * band(1.7, gsd);
    col *= 1.0 + 0.10 * tus + 0.12 * perlin3(id ^ 0x9A8, p / 40.0) * band(40.0, gsd);
    let mut cls = CLS_PASTURE;
    let mut dh = 0.1;
    // only paddocks big enough for trails (the big systems)
    if pl.sys == sys::PASTURE || pl.sys == sys::SHELTER || pl.sys == sys::SECTION || pl.sys == sys::RANCH {
        let half = 0.5 * pc.size;
        let t = DVec2::new(0.4 * half * (u01k(id, 60) - 0.5), 0.4 * half * (u01k(id, 61) - 0.5));
        let rel = q - pc.fc - t;
        let r = rel.length();
        let n = 5.0 + (u01k(id, 62) * 4.0).floor();
        let th = rel.y.atan2(rel.x) / std::f64::consts::TAU * n + n;
        let mut dmin: f64 = 1e9;
        for o in 0..2 {
            let kk = th.floor() + o as f64;
            let kh = mix64(id ^ (kk.rem_euclid(n) as u64) ^ 0x7A11);
            let a = kk + 0.3 * (u01k(kh, 1) - 0.5) + 0.12 * (r / 45.0 + 6.0 * u01k(kh, 2)).sin();
            let len = (0.3 + 0.7 * u01k(kh, 3)) * half;
            let dd = (th - a).abs() / n * std::f64::consts::TAU * r;
            if r < len {
                dmin = dmin.min(dd);
            }
        }
        let vis = band(1.5, gsd).max(0.3 * band(30.0, gsd));
        let trail = band_cov(dmin, 0.45, fw) * vis * smoothstep(8.0, 20.0, r);
        let bare = px.soil * 0.95;
        col = mixc(col, bare, trail * 0.8);
        // trampled ground and the trough
        // (a third of the paddocks: bare discs in every one dotted the land like a pattern)
        let tramp = (1.0 - smoothstep(5.0, 13.0 + 5.0 * perlin3(id ^ 0x7A12, p / 9.0), r)) * if u01k(id, 65) < 0.35 { 1.0 } else { 0.0 };
        col = mixc(col, bare * 0.9, 0.5 * tramp);
        let trough = band_cov(r, 1.4, fw);
        col = mixc(col, srgb(112.0, 118.0, 122.0), trough);
        dh += 0.8 * trough;
        // farm dam
        if u01k(id, 63) < 0.15 {
            let dc = rel - DVec2::new(30.0, 0.0);
            let rd = 12.0 + 14.0 * u01k(id, 64);
            let w = band_cov(dc.length(), rd, fw);
            let ring = band_cov(dc.length(), rd + 3.0, fw) - w;
            col = mixc(col, bare * 0.85, ring.max(0.0));
            col = mixc(col, srgb(56.0, 74.0, 70.0), w);
            if w > 0.5 {
                cls = CLS_RESERVOIR;
            }
            dh -= 0.6 * w;
        }
    }
    (col, dh, cls)
}

/// A homestead compound: swept earth inside a thorn fence, round huts.
fn yard(pc: &Parcel, px: &Px, id: u64, q: DVec2) -> (DVec3, f64, u8) {
    let (p, gsd, fw) = (px.p, px.gsd, px.fw);
    let rel = q - pc.fc;
    let r = rel.length();
    let ry = 14.0 + 10.0 * u01k(id, 70);
    let mut col = px.soil * 1.15 * (1.0 + 0.05 * perlin3(id ^ 0x7A8, p / 3.0) * band(3.0, gsd));
    let mut dh = 0.0;
    let mut cls = CLS_FARMYARD;
    if r > ry {
        // a garden around the compound
        col = mixc(px.soil, srgb(82.0, 114.0, 54.0), 0.5 + 0.3 * perlin3(id ^ 0x7A9, p / 5.0) * band(5.0, gsd));
        return (col, 0.6, CLS_CROP);
    }
    let fence = band_cov(r - ry, 0.8, fw);
    col = mixc(col, srgb(92.0, 90.0, 60.0), fence);
    dh += 1.5 * fence;
    let nh = 2.0 + (u01k(id, 71) * 4.0).floor();
    for i in 0..5 {
        if (i as f64) < nh {
            let a = std::f64::consts::TAU * (i as f64 / nh + u01k(id, 72));
            let c = DVec2::new(a.cos(), a.sin()) * (0.5 * ry);
            let hr = 2.2 + 1.3 * u01k(id, 73 + i as u64);
            let dd = (rel - c).length();
            let hut = band_cov(dd, hr, fw);
            if hut > 0.0 {
                let tin = u01k(id, 80 + i as u64) < 0.35;
                let roof = if tin { srgb(176.0, 180.0, 182.0) } else { srgb(146.0, 122.0, 86.0) * (0.9 + 0.2 * (1.0 - dd / hr)) };
                col = mixc(col, roof, hut);
                dh = lerp(dh, 2.0 + 1.5 * (1.0 - dd / hr), hut);
                if hut > 0.5 {
                    cls = CLS_BUILDING;
                }
            }
        }
    }
    (col, dh, cls)
}

/// A parcel's border over its colour.
fn draw_border(b: u32, edge: f64, col: DVec3, dh: f64, cls: u8, pc: &Parcel, px: &Px) -> (DVec3, f64, u8) {
    let (p, gsd, fw) = (px.p, px.gsd, px.fw);
    let fwb = fw.max(0.4 * gsd);
    let hb = mix64(pc.id ^ 0xED6E);
    match b {
        border::MARGIN => {
            let c = band_cov(edge, 0.9, fwb);
            (mixc(col, mixc(px.grass_dry, px.grass, px.moist), 0.45 * c), dh, cls)
        }
        border::HEDGE | border::BELT => {
            let belt = b == border::BELT;
            let hw0 = if belt { 4.0 + 3.0 * u01k(hb, 4) } else { 1.0 + 1.0 * u01k(hb, 4) };
            if edge.abs() > 4.5 * hw0 + fwb {
                return (col, dh, cls);
            }
            let lam = if belt { 7.0 } else { 5.0 };
            let clump = perlin3(hb, p / lam);
            // hedgerow trees: crowns bulging out of the hedge every few tens of metres (a hedge of
            // constant width outlined every field like a drawing)
            let b11 = band(11.0, gsd);
            let tb = if belt { 0.0 } else { smoothstep(0.3, 0.6, perlin3(hb ^ 3, p / 11.0)) * b11 + (1.0 - b11) * 0.08 };
            // crowns of different size along the hedge: its width varies (the mean below the pixel)
            let hw = hw0 * (0.875 + 0.4 * clump * band(lam, gsd)) * (1.0 + 2.0 * tb);
            // (the band box-filtered exactly: a band narrower than the filter keeps its area
            // instead of widening to the filter)
            // gaps: gateways and missing stretches
            let b18 = band(18.0, gsd);
            let gap = if belt { 1.0 } else { smoothstep(-0.5, -0.2, perlin3(hb ^ 5, p / 18.0)) * b18 + (1.0 - b18) * 0.85 };
            let c = gap * ((edge + 0.5 * fwb).min(hw) - (edge - 0.5 * fwb).max(-hw)).max(0.0) / fwb;
            if c <= 0.0 {
                return (col, dh, cls);
            }
            let tex = 1.0 + 0.18 * perlin3(hb ^ 1, p / 2.2) * band(2.2, gsd);
            let across = (1.0 - (edge.abs() / hw).powi(2)).max(0.0).sqrt();
            let hc = px.crown * (if belt { 0.82 } else { 0.95 }) * (0.85 + 0.3 * clump) * tex * (0.82 + 0.3 * across) * (1.0 - 0.1 * tb);
            let along = 0.55 + 0.45 * (0.5 + 0.5 * clump);
            let h = if belt { 9.0 + 6.0 * u01k(hb, 2) } else { 3.0 + 4.0 * u01k(hb, 2) + 7.0 * tb };
            (mixc(col, hc, c), lerp(dh, h * across * along, c), if c > 0.5 { CLS_HEDGEROW } else { cls })
        }
        border::WALL => {
            let c = band_cov(edge, 0.45, fwb);
            let sc = srgb(150.0, 146.0, 136.0) * (0.88 + 0.24 * (0.5 + 0.5 * perlin3(hb, p / 1.5)));
            (mixc(col, sc, c), lerp(dh, 1.0, c), cls)
        }
        border::DITCH => {
            let c = band_cov(edge, 1.0, fwb);
            (mixc(col, srgb(66.0, 76.0, 58.0), c), dh - 0.5 * c, cls)
        }
        border::TRACK => {
            let c = band_cov(edge, 2.5, fwb);
            (
                mixc(col, srgb(162.0, 146.0, 120.0) * (1.0 + 0.06 * perlin3(hb, p / 3.0) * band(3.0, gsd)), c),
                lerp(dh, 0.0, c),
                if c > 0.5 { CLS_TRACK } else { cls },
            )
        }
        border::CANAL => {
            let bank = band_cov(edge, 3.0, fwb);
            let w = band_cov(edge, 1.6, fwb);
            let c1 = mixc(col, srgb(128.0, 120.0, 92.0), bank);
            (mixc(c1, srgb(50.0, 74.0, 78.0), w), dh + 0.3 * (bank - w) - 0.6 * w, if w > 0.5 { CLS_CANAL } else { cls })
        }
        _ => (col, dh, cls),
    }
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ============================================================================= WGSL tables

/// The WGSL of the catalogue's tables (between the markers of `agriculture.wgsl`).
#[cfg(test)]
pub fn wgsl_tables() -> String {
    let f = |x: f64| {
        let s = format!("{:?}", x as f32);
        if s.contains('.') || s.contains('e') {
            s
        } else {
            format!("{s}.0")
        }
    };
    let mut s = String::from("// BEGIN GENERATED (kits::agriculture::wgsl_tables; AERIALSYNTH_BLESS=1 cargo test -p terragen --lib agriculture)\n");
    let n = sys::N;
    s.push_str(&format!("fn agri_arch_w(a: u32) -> array<f32, {n}> {{\n    switch a {{\n"));
    for (i, w) in ARCH_W.iter().enumerate() {
        let v: Vec<String> = w.iter().map(|x| f(*x)).collect();
        if i == 11 {
            s.push_str(&format!("        default: {{ return array<f32, {n}>({}); }}\n", v.join(", ")));
        } else {
            s.push_str(&format!("        case {i}u: {{ return array<f32, {n}>({}); }}\n", v.join(", ")));
        }
    }
    s.push_str("    }\n}\n");
    s.push_str("fn agri_sys(i: u32) -> AgSys {\n    switch i {\n");
    for (i, d) in SYS.iter().enumerate() {
        let body = format!(
            "AgSys({}u, vec2<f32>({}, {}), vec4<u32>({}u, {}u, {}u, {}u), vec4<f32>({}, {}, {}, {}), vec2<u32>({}u, {}u), vec2<f32>({}, {}), vec2<f32>({}, {}), {}, {})",
            d.part,
            f(d.size[0]),
            f(d.size[1]),
            d.kinds[0],
            d.kinds[1],
            d.kinds[2],
            d.kinds[3],
            f(d.kw[0]),
            f(d.kw[1]),
            f(d.kw[2]),
            f(d.kw[3]),
            d.borders[0],
            d.borders[1],
            f(d.bp[0]),
            f(d.bp[1]),
            f(d.slope[0]),
            f(d.slope[1]),
            f(d.terraces),
            f(d.trees)
        );
        if i == sys::N - 1 {
            s.push_str(&format!("        default: {{ return {body}; }}\n"));
        } else {
            s.push_str(&format!("        case {i}u: {{ return {body}; }}\n"));
        }
    }
    s.push_str("    }\n}\n");
    s.push_str("/// A crop: linear colours in spring, summer, autumn; (row spacing, contrast, height).\nfn agri_crop(i: u32) -> array<vec3<f32>, 4> {\n    switch i {\n");
    for (i, (c, r)) in CROPS.iter().enumerate() {
        let v = |c: [f64; 3]| {
            let l = lin(c);
            format!("vec3<f32>({}, {}, {})", f(l.x), f(l.y), f(l.z))
        };
        let body = format!("array<vec3<f32>, 4>({}, {}, {}, vec3<f32>({}, {}, {}))", v(c[0]), v(c[1]), v(c[2]), f(r[0]), f(r[1]), f(r[2]));
        if i == CROPS.len() - 1 {
            s.push_str(&format!("        default: {{ return {body}; }}\n"));
        } else {
            s.push_str(&format!("        case {i}u: {{ return {body}; }}\n"));
        }
    }
    s.push_str("    }\n}\n");
    s.push_str("/// A crop mix: crops and shares.\nfn agri_mix(i: u32) -> AgMix {\n    switch i {\n");
    for (i, (c, w)) in MIXES.iter().enumerate() {
        let body = format!("AgMix(vec4<u32>({}u, {}u, {}u, {}u), vec4<f32>({}, {}, {}, {}))", c[0], c[1], c[2], c[3], f(w[0]), f(w[1]), f(w[2]), f(w[3]));
        if i == MIXES.len() - 1 {
            s.push_str(&format!("        default: {{ return {body}; }}\n"));
        } else {
            s.push_str(&format!("        case {i}u: {{ return {body}; }}\n"));
        }
    }
    s.push_str("    }\n}\n// END GENERATED\n");
    s
}

// ============================================================================= tests

#[cfg(test)]
mod tests {
    use super::*;

    const WGSL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/gpu/wgsl/kits/agriculture.wgsl");

    /// The WGSL tables are those of this file (`AERIALSYNTH_BLESS=1` rewrites them).
    #[test]
    fn wgsl_tables_are_current() {
        let src = std::fs::read_to_string(WGSL_PATH).unwrap();
        let (a, b) = (src.find("// BEGIN GENERATED").unwrap(), src.find("// END GENERATED").unwrap() + "// END GENERATED\n".len());
        let want = wgsl_tables();
        if src[a..b] != want {
            if std::env::var("AERIALSYNTH_BLESS").is_ok() {
                std::fs::write(WGSL_PATH, format!("{}{}{}", &src[..a], want, &src[b..])).unwrap();
            } else {
                panic!("agriculture.wgsl tables are stale: AERIALSYNTH_BLESS=1 cargo test -p terragen --lib agriculture");
            }
        }
    }

    #[test]
    fn registry_compiles() {
        crate::registry::Registry::builtin().unwrap();
    }

    /// Every archetype draws several systems, and over many ecoregions every system appears.
    #[test]
    fn systems_vary() {
        let mut seen = [0usize; sys::N];
        for a in 0..12u32 {
            let mut here = std::collections::BTreeSet::new();
            for i in 0..400u64 {
                let h = mix64(0xA6 ^ (i << 8) ^ a as u64);
                let k = if a == 10 { 28 } else { (u01k(h, 1) * 28.0) as u32 };
                let e = EcoV {
                    id: h,
                    arch: a,
                    koppen: k,
                    temp: 5.0 + 20.0 * u01k(h, 2),
                    precip: 200.0 + 1500.0 * u01k(h, 3),
                    dev: u01k(h, 4),
                    scale: 1.0,
                    hedges: 0.4,
                };
                for r in 0..8u64 {
                    let s = pick_system(&e, mix64(h ^ r));
                    seen[s as usize] += 1;
                    here.insert(s);
                }
            }
            assert!(here.len() >= 2, "archetype {a}: {here:?}");
        }
        for (i, n) in seen.iter().enumerate() {
            assert!(*n > 50, "{}: {n} ({seen:?})", sys::NAMES[i]);
        }
    }

    /// Places of the systems in the default world (seed 1): name, lat, lon (deg).
    pub(super) const PLACES: &[(&str, f64, f64)] = &[
        ("section", 35.8711, 51.8475),
        ("shelterbelt", 54.4992, -104.7994),
        ("strips", 55.6454, 28.0690),
        ("bocage", -51.8166, -103.9835),
        ("smallholder", 15.2294, 58.0429),
        ("paddy", 18.4204, 160.8040),
        ("terrace", -38.4956, 176.2920),
        ("vineyard", 30.5227, 163.7190),
        ("grove", 29.0000, 107.2354),
        ("plantation", -11.5620, 165.4861),
        ("greenhouse", 27.7830, 172.0161),
        ("pasture", 34.4425, 147.2698),
        ("pivot", 16.3893, 120.3407),
        ("feedlot", 16.2941, 149.3331),
    ];

    /// Per-pixel comparison of two tiles: (layer, share of pixels off) (rgb > 2 DN, elevation
    /// > 5 cm, class differs; `gpu::tests::compare_tiles` is private to the gpu module).
    #[cfg(feature = "gpu")]
    fn off(g: &crate::TileData, c: &crate::TileData) -> [(&'static str, f64); 5] {
        let n = g.elevation.len() as f64;
        let e = g.elevation.iter().zip(&c.elevation).filter(|(a, b)| (**a - **b).abs() > 0.05).count() as f64 / n;
        let r = g.rgb.iter().zip(&c.rgb).filter(|(a, b)| (**a as i32 - **b as i32).abs() > 2).count() as f64 / g.rgb.len() as f64;
        let l = g.landcover.iter().zip(&c.landcover).filter(|(a, b)| a != b).count() as f64 / n;
        // (the optional layers: the emission of lit greenhouses, the parcels as instances)
        let em = g.emission.iter().zip(&c.emission).filter(|(a, b)| (**a as i32 - **b as i32).abs() > 2).count() as f64 / g.emission.len() as f64;
        let i = g.instance.iter().zip(&c.instance).filter(|(a, b)| a != b).count() as f64 / n;
        [("elevation", e), ("rgb", r), ("landcover", l), ("emission", em), ("instance", i)]
    }

    /// The systems' tiles on the GPU are the CPU's (both backends: at most 0.5 % of the pixels
    /// off).
    #[cfg(feature = "gpu")]
    #[test]
    fn tiles_match_the_cpu() {
        match crate::gpu::device::shared() {
            Ok(g) if g.check_generator().is_ok() => {}
            _ => {
                eprintln!("no GPU that runs the tile generator: skipped");
                return;
            }
        }
        let cfg = crate::Config::default();
        let gen = crate::gpu::GpuGenerator::new(cfg.clone()).unwrap();
        let cpu = crate::NextGenerator::new(cfg);
        let mut worst = 0.0f64;
        for &(name, lat, lon) in PLACES {
            for z in [12u8, 14, 16] {
                let id = geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z);
                let g = gen.tiles(&[id]).unwrap().remove(0);
                let c = cpu.tile_cpu(id);
                let o = off(&g, &c);
                eprintln!("{name:12} z{z}: {}", o.iter().map(|(k, v)| format!("{k} {:.3}%", v * 100.0)).collect::<Vec<_>>().join(", "));
                for (_, v) in o {
                    worst = worst.max(v);
                }
                // AGRI_DIFF=DIR: both tiles and the mask of the pixels off, where many are
                if let (Ok(dir), true) = (std::env::var("AGRI_DIFF"), o[1].1 > 0.005) {
                    let n = crate::TILE_SIZE as u32;
                    let img = |t: &crate::TileData| image::RgbImage::from_raw(n, n, t.rgb.clone()).unwrap();
                    img(&g).save(format!("{dir}/{name}_z{z}_gpu.png")).unwrap();
                    img(&c).save(format!("{dir}/{name}_z{z}_cpu.png")).unwrap();
                    let mut m = image::GrayImage::new(n, n);
                    for (k, px) in m.pixels_mut().enumerate() {
                        let d = (0..3).map(|j| (g.rgb[3 * k + j] as i32 - c.rgb[3 * k + j] as i32).abs()).max().unwrap();
                        px.0 = [(d * 8).min(255) as u8];
                    }
                    m.save(format!("{dir}/{name}_z{z}_diff.png")).unwrap();
                }
            }
        }
        assert!(worst < 0.005, "worst {:.3}%", worst * 100.0);
    }

    /// The places the LOD / seam / determinism test visits (CPU tiles are slow).
    const LOD_PLACES: [&str; 5] = ["section", "strips", "paddy", "vineyard", "plantation"];

    fn tile_at(name: &str, z: u8) -> geodesy::tiles::TileId {
        let &(_, lat, lon) = PLACES.iter().find(|p| p.0 == name).unwrap();
        geodesy::tiles::tile_for_latlon(lat.to_radians(), lon.to_radians(), z)
    }

    /// Parent ≈ mean of the children (albedo and class histogram) at the kit's places, and the
    /// tiles are deterministic and seamless there.
    #[test]
    fn lod_seams_determinism() {
        let cfg = crate::Config { tile_supersample: 1, ..crate::Config::default() };
        let g = crate::NextGenerator::new(cfg.clone());
        let n = crate::TILE_SIZE;
        for (name, _, _) in PLACES.iter().filter(|p| LOD_PLACES.contains(&p.0)) {
            for z in [12u8, 15] {
                let parent = tile_at(name, z);
                let p = g.tile(parent);
                let kids = parent.children().map(|c| g.tile(c));
                let (mut err, mut cnt) = (0.0, 0.0);
                let mut hp = vec![0f64; 256];
                let mut hk = vec![0f64; 256];
                for (k, kid) in kids.iter().enumerate() {
                    let (ox, oy) = ((k % 2) * n / 2, (k / 2) * n / 2);
                    for j in (0..n).step_by(2) {
                        for i in (0..n).step_by(2) {
                            for c in 0..3 {
                                let a = |ii: usize, jj: usize| kid.albedo[3 * (jj * n + ii) + c] as f64;
                                let m = (a(i, j) + a(i + 1, j) + a(i, j + 1) + a(i + 1, j + 1)) / 4.0;
                                err += (m - p.albedo[3 * ((oy + j / 2) * n + ox + i / 2) + c] as f64).abs();
                                cnt += 1.0;
                            }
                            for (jj, ii) in [(j, i), (j, i + 1), (j + 1, i), (j + 1, i + 1)] {
                                hk[kid.landcover[jj * n + ii] as usize] += 0.25;
                            }
                            hp[p.landcover[(oy + j / 2) * n + ox + i / 2] as usize] += 1.0;
                        }
                    }
                }
                let hist: f64 = hp.iter().zip(&hk).map(|(a, b)| (a - b).abs()).sum::<f64>() / (2.0 * (n * n) as f64);
                eprintln!("{name} z{z}: albedo |parent - mean(children)| {:.2} DN, class histogram {:.3}", err / cnt, hist);
                // (the core alone, without the kit, at these places: up to 14.5 DN and 0.17 at
                // z15 — the canopy's crowns at the crossfade)
                assert!(err / cnt < 12.0, "{name} z{z}: albedo {}", err / cnt);
                assert!(hist < 0.1, "{name} z{z}: class histogram {hist}");
            }
            // determinism (a fresh generator) and the seam to the eastern neighbour
            let id = tile_at(name, 15);
            let a = g.tile(id);
            let b = crate::NextGenerator::new(cfg.clone()).tile(id);
            assert_eq!(a.rgb, b.rgb, "{name}: not deterministic");
            let r = g.tile(geodesy::tiles::TileId::new(id.z, id.x + 1, id.y));
            let (mut across, mut inside) = (0.0, 0.0);
            for j in 0..n {
                for c in 0..3 {
                    across += (a.rgb[3 * (j * n + n - 1) + c] as f64 - r.rgb[3 * (j * n) + c] as f64).abs();
                    inside += (a.rgb[3 * (j * n + n - 2) + c] as f64 - a.rgb[3 * (j * n + n - 1) + c] as f64).abs();
                }
            }
            eprintln!("{name}: seam {across:.0} vs inside {inside:.0}");
            assert!(across < 1.5 * inside + 3.0 * n as f64, "{name}: seam {across} inside {inside}");
        }
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

    /// GPU tiles/s at the kit's places (8 x 8 tiles per place and zoom; performance budget):
    /// `cargo test -p terragen --release --lib kits::agriculture::tests::bench_places -- --ignored --nocapture`.
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

    /// Places of every system in the default world (for stills):
    /// `cargo test --release -p terragen --lib find_places -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_places() {
        use rayon::prelude::*;
        let w = crate::world::World::new(crate::Config::default());
        let reg = crate::registry::Registry::builtin().unwrap();
        let n = std::env::var("AGRI_FIND_N").ok().and_then(|v| v.parse().ok()).unwrap_or(40_000u64);
        let hits: Vec<(u32, f64, f64, String)> = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let h = mix64(0xF1EDA ^ i);
                let lat = (2.0 * u01k(h, 1) - 1.0).asin().to_degrees();
                if lat.abs() > 70.0 {
                    return None;
                }
                let lon = 360.0 * u01k(h, 2) - 180.0;
                let ctx = crate::world::Ctx::new(lat.to_radians(), lon.to_radians(), 30.0, &w.ell);
                let t = w.terrain(&ctx);
                if t.water_kind != 0 || t.region.id == 0 || t.agri < 0.35 || t.eco.id == 0 {
                    return None;
                }
                let e = EcoV::of(&crate::eco::compute(&w, &reg, t.eco.id, t.eco.center));
                let s = pick_system(&e, t.region.id);
                Some((
                    s,
                    lat,
                    lon,
                    format!("agri {:.2} T {:.0} M {:.2} h {:.0} arch {} K {} P {:.0}", t.agri, t.temp, t.moist, t.ground, e.arch, e.koppen, e.precip),
                ))
            })
            .collect();
        // ranch regions with a feedlot: the feedlot's centre
        let g = crate::NextGenerator::new(crate::Config::default());
        let mut nf = 0;
        for x in hits.iter().filter(|x| x.0 == sys::RANCH) {
            let ctx = crate::world::Ctx::new(x.1.to_radians(), x.2.to_radians(), 30.0, &w.ell);
            let t = w.terrain(&ctx);
            let r = g.surface.region_info(&g.world, &mut crate::surface::Caches::default(), &t);
            let e = EcoV::of(&crate::eco::compute(&w, &reg, t.eco.id, t.eco.center));
            let seed = mix64(r.split.to_bits() ^ 0xA6A6);
            if let Some((c, ..)) = feedlot_site(seed, e.dev) {
                let pc = r.center + r.ex * c.x + r.ey * c.y;
                let gd = geodesy::ecef2geodetic(pc, &w.ell);
                println!("feedlot ({:.4}, {:.4}) {}", gd.lat.to_degrees(), gd.lon.to_degrees(), x.3);
                nf += 1;
                if nf >= 8 {
                    break;
                }
            }
        }
        for sy in 0..sys::N as u32 {
            let v: Vec<_> = hits.iter().filter(|x| x.0 == sy).collect();
            println!("{} ({}):", sys::NAMES[sy as usize], v.len());
            for x in v.iter().take(6) {
                println!("  {:.4},{:.4}  {}", x.1, x.2, x.3);
            }
        }
    }
}
