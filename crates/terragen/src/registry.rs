//! The biome registry (`docs/design/terrain-next.md` §3.7): biomes are data. YAML definitions
//! (`crates/terragen/biomes/*.yaml`, one file per kit) are compiled at start-up into Rust
//! structs ([`Registry`]) and packed GPU tables ([`Registry::gpu`]), checked with keyed error
//! messages, and stored resolved in the world config (`biomes.resolved`), so a tile store stays
//! self-describing.
//!
//! A biome has
//! * a climate envelope and Köppen classes (picked per ecoregion, [`crate::eco`]),
//! * a natural palette (soil, grass, rock, sand, crowns …) and vegetation parameters that drive
//!   the zonal ground (slot 1) and the canopy,
//! * crown layers (the canopy: scatter kernels of tree / shrub crowns),
//! * a zonation (other biomes below given temperatures: montane → alpine → nival),
//! * up to [`MAX_LAYERS`] kernel layers at any stack slot, each with a mask of ≤ 4 smoothstep
//!   windows over named fields ([`field`]) and a calibrated mean (§3.7 "Calibrated means").
//!
//! The mean of every kernel layer is integrated once here, so a layer whose features are below
//! the pixel blends to exactly that mean (parent ≈ mean of children by construction), and the
//! per-GSD-band layer lists ([`Biome::bands`]) leave out what cannot matter in a band.

use crate::kernels::{self, KMean, KParams};
use crate::noise::*;
use anyhow::{bail, Context, Result};
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// At most this many kernel layers per biome, crown layers per biome, windows per mask.
pub const MAX_LAYERS: usize = 8;
pub const MAX_CROWNS: usize = 6;
pub const MAX_WINDOWS: usize = 4;
pub const MAX_ZONES: usize = 4;
/// GSD bands of the per-band layer lists (m): < 1, 1–4, 4–16, 16–64, 64–256, ≥ 256.
pub const BANDS: usize = 6;

/// The band of a pixel size.
pub fn band_of(gsd: f64) -> usize {
    let mut b = 0;
    let mut g = 1.0;
    while b < BANDS - 1 && gsd >= g {
        b += 1;
        g *= 4.0;
    }
    b
}

/// Lower and upper pixel size of band `b`.
pub fn band_range(b: usize) -> (f64, f64) {
    let lo = if b == 0 { 0.0 } else { 4f64.powi(b as i32 - 1) };
    let hi = if b + 1 >= BANDS { f64::INFINITY } else { 4f64.powi(b as i32) };
    (lo, hi)
}

/// Named fields of masks and kernels (the same ids on the GPU: `FIELD_*` in `registry.wgsl`).
pub mod field {
    pub const TEMP: usize = 0;
    pub const MOIST: usize = 1;
    pub const SLOPE: usize = 2;
    pub const GULLY: usize = 3;
    pub const HEIGHT: usize = 4;
    pub const RIVER_DIST: usize = 5;
    pub const PATCH: usize = 6;
    pub const DETAIL: usize = 7;
    pub const LAND: usize = 8;
    pub const FOREST: usize = 9;
    pub const SNOW_N: usize = 10;
    pub const MOUNTAIN: usize = 11;
    pub const FLOODPLAIN: usize = 12;
    pub const AGRI: usize = 13;
    pub const HABIT: usize = 14;
    pub const SAND: usize = 15;
    pub const ROCK_EXPECT: usize = 16;
    pub const MESA: usize = 17;
    pub const CONT: usize = 18;
    pub const LAT: usize = 19;
    pub const URBAN: usize = 20;
    pub const FIELD: usize = 21;
    pub const NATURAL: usize = 22;
    pub const VEG: usize = 23;
    pub const STYLE0: usize = 24;
    pub const STYLE1: usize = 25;
    pub const STYLE2: usize = 26;
    pub const STYLE3: usize = 27;
    pub const ECO_EDGE: usize = 28;
    pub const PRECIP: usize = 29;
    pub const TEMP_RANGE: usize = 30;
    pub const DRY_MONTHS: usize = 31;
    pub const N: usize = 32;
    /// YAML names (index = id)
    pub const NAMES: [&str; N] = [
        "temp",
        "moist",
        "slope",
        "gully",
        "height",
        "river_dist",
        "patch",
        "detail",
        "land",
        "forest",
        "snow_noise",
        "mountain",
        "floodplain",
        "agri",
        "habit",
        "sand",
        "rock_expect",
        "mesa",
        "cont",
        "lat",
        "urban",
        "field",
        "natural",
        "veg",
        "style0",
        "style1",
        "style2",
        "style3",
        "eco_edge",
        "precip_mm",
        "temp_range",
        "dry_months",
    ];
    pub fn id(name: &str) -> Option<usize> {
        NAMES.iter().position(|n| *n == name)
    }
}

/// Land-cover class names of the registry (v2 ids, `docs/design/terrain-next.md` §7.2).
pub mod classes {
    #[rustfmt::skip]
    pub const TABLE: &[(&str, u8)] = &[
        ("unknown", 0), ("ocean", 1), ("lake", 2), ("river", 3), ("beach", 4), ("sand", 5), ("rock", 6), ("snow", 7),
        ("grass", 8), ("shrub", 9), ("forest", 10), ("crop", 11), ("building", 12), ("road", 13), ("wetland", 14),
        ("tundra", 15), ("bare", 16), ("urban", 17),
        ("reservoir", 20), ("lagoon", 21), ("canal", 22), ("aquaculture", 23), ("tidal_flat", 24), ("coral_reef", 25),
        ("sea_ice", 26), ("glacier", 27), ("frozen_water", 28), ("dry_riverbed", 29), ("salt_flat", 30), ("lava", 31),
        ("volcanic_ash", 32), ("gravel", 33), ("badlands", 34), ("scree", 35), ("moraine", 36), ("cliff", 37),
        ("tropical_rainforest", 40), ("mangrove", 41), ("broadleaf_forest", 42), ("needleleaf_forest", 43),
        ("mixed_forest", 44), ("woodland", 45),
        ("savanna", 52), ("steppe", 53), ("desert_scrub", 54), ("maquis", 55), ("alpine_meadow", 56),
        ("polygon_tundra", 57), ("bog", 58), ("marsh", 59), ("burn_scar", 60), ("clear_cut", 61),
        ("rice_paddy", 70), ("orchard", 71), ("vineyard", 72), ("plantation", 73), ("pasture", 74),
        ("greenhouse", 75), ("fallow", 76), ("hedgerow", 77), ("farmyard", 78),
        ("residential", 80), ("commercial", 81), ("industrial", 82), ("building_tall", 83), ("park", 84),
        ("sports", 85), ("paved", 86), ("solar_farm", 87), ("port", 88), ("cemetery", 89), ("quarry", 90),
        ("motorway", 100), ("road_major", 101), ("road_minor", 102), ("track", 103), ("railway", 104),
        ("runway", 105), ("taxiway", 106), ("bridge", 107), ("dam", 108), ("seasonal_snow", 110),
    ];
    pub fn id(name: &str) -> Option<u8> {
        TABLE.iter().find(|(n, _)| *n == name).map(|e| e.1)
    }
    pub fn name(id: u8) -> &'static str {
        TABLE.iter().find(|e| e.1 == id).map_or("?", |e| e.0)
    }
}

// ============================================================================= YAML schema

/// A colour: `[r, g, b]` (sRGB 0..255) or the name of a default palette entry.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(untagged)]
enum ColourYaml {
    Rgb([f64; 3]),
    Name(String),
}

/// A palette entry: one colour, or a list (soil 4, rock 3, sand 3).
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(untagged)]
enum PalYaml {
    One(ColourYaml),
    Many(Vec<ColourYaml>),
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
struct EnvelopeYaml {
    temp: Option<[f64; 2]>,
    precip_mm: Option<[f64; 2]>,
    dry_months: Option<[f64; 2]>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(untagged)]
enum LithoYaml {
    Any(String),
    List(Vec<String>),
}

/// Multipliers / offsets of the canopy and ground model (1 / 0: today's world).
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
struct VegYaml {
    trees: f64,
    conifer: f64,
    tropic: f64,
    shrubs: f64,
    savanna: f64,
    groves: f64,
    woodlots: f64,
    tall: f64,
    laterite: f64,
    meadow: f64,
    crown_scale: f64,
    gallery: f64,
}

impl Default for VegYaml {
    fn default() -> Self {
        VegYaml { trees: 1.0, conifer: 0.0, tropic: 0.0, shrubs: 1.0, savanna: 1.0, groves: 1.0, woodlots: 1.0, tall: 0.0, laterite: 1.0, meadow: 1.0, crown_scale: 1.0, gallery: 0.0 }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct CrownYaml {
    /// which share of the canopy model: conifer, broadleaf, tropical, shrub, savanna, all
    share: String,
    #[serde(default = "default_shape")]
    shape: String,
    cell: f64,
    /// [base, + per unit of `tall`] (m)
    height: [f64; 2],
    #[serde(default = "one")]
    open_height: f64,
    colour: ColourYaml,
    colour_dry: Option<ColourYaml>,
    #[serde(default = "one")]
    density: f64,
    #[serde(default = "yes")]
    closure: bool,
    #[serde(default = "yes")]
    stand: bool,
    seed: Option<u64>,
}

fn default_shape() -> String {
    "dome".into()
}
fn one() -> f64 {
    1.0
}
fn yes() -> bool {
    true
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct ZoneYaml {
    below_c: f64,
    biome: String,
}

/// A smoothstep window: `[lo, hi]` (soft edges of 10 % of the width) or `[a0, a1, b0, b1]`
/// (in over a0..a1, out over b0..b1); `null` for an open end.
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(transparent)]
struct WindowYaml(Vec<Option<f64>>);

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct LayerYaml {
    slot: String,
    kernel: String,
    #[serde(default)]
    mask: BTreeMap<String, WindowYaml>,
    #[serde(default)]
    params: BTreeMap<String, serde_yaml::Value>,
    class: Option<String>,
    #[serde(default)]
    clear: f64,
    height: Option<String>,
    #[serde(default)]
    material: u8,
    seed: Option<u64>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
struct LanduseYaml {
    agriculture: f64,
    towns: f64,
}

impl Default for LanduseYaml {
    fn default() -> Self {
        LanduseYaml { agriculture: 1.0, towns: 1.0 }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
struct BiomeYaml {
    id: String,
    /// change an existing biome of this id (only the keys given)
    #[serde(default)]
    patch: bool,
    group: Option<String>,
    koppen: Option<Vec<String>>,
    envelope: Option<EnvelopeYaml>,
    lithology: Option<LithoYaml>,
    weight: Option<f64>,
    min_share: Option<f64>,
    palette: Option<BTreeMap<String, PalYaml>>,
    vegetation: Option<VegYaml>,
    canopy: Option<Vec<CrownYaml>>,
    zonation: Option<Vec<ZoneYaml>>,
    layers: Option<Vec<LayerYaml>>,
    landuse: Option<LanduseYaml>,
}

/// The registry files: (name, YAML text) of the core and every kit, in order.
pub fn builtin_sources() -> Vec<(&'static str, &'static str)> {
    let mut v = vec![("core", include_str!("../biomes/core.yaml"))];
    for k in crate::kits::KITS {
        if !k.biomes_yaml.trim().is_empty() {
            v.push((k.name, k.biomes_yaml));
        }
    }
    v
}

// ============================================================================= compiled

/// Indices into [`BiomePal::c`].
pub mod pal {
    pub const SOIL: usize = 0; // 4
    pub const GRASS_WET: usize = 4;
    pub const GRASS_DRY: usize = 5;
    pub const GRASS_COLD: usize = 6;
    pub const TUNDRA: usize = 7;
    pub const MARSH: usize = 8;
    pub const ROCK: usize = 9; // 3
    pub const SAND: usize = 12; // 3
    pub const BEACH: usize = 15;
    pub const WET_SAND: usize = 16;
    pub const CROWN_CONIFER: usize = 17;
    pub const CROWN_DECID: usize = 18;
    pub const CROWN_TROPIC: usize = 19;
    pub const CROWN_DRY: usize = 20;
    pub const SHRUB: usize = 21;
    pub const LATERITE: usize = 22;
    pub const FLOOR: usize = 23;
    pub const N: usize = 24;
    /// YAML keys: (name, first index, count)
    pub const KEYS: [(&str, usize, usize); 18] = [
        ("soil", SOIL, 4),
        ("grass_wet", GRASS_WET, 1),
        ("grass_dry", GRASS_DRY, 1),
        ("grass_cold", GRASS_COLD, 1),
        ("tundra", TUNDRA, 1),
        ("marsh", MARSH, 1),
        ("rock", ROCK, 3),
        ("sand", SAND, 3),
        ("beach", BEACH, 1),
        ("wet_sand", WET_SAND, 1),
        ("crown_conifer", CROWN_CONIFER, 1),
        ("crown_decid", CROWN_DECID, 1),
        ("crown_tropic", CROWN_TROPIC, 1),
        ("crown_dry", CROWN_DRY, 1),
        ("shrub", SHRUB, 1),
        ("laterite", LATERITE, 1),
        ("floor", FLOOR, 1),
        ("crop_mean", usize::MAX, 0),
    ];
}

/// A biome's natural palette (linear RGB).
#[derive(Clone, Copy, Debug)]
pub struct BiomePal {
    pub c: [DVec3; pal::N],
}

impl BiomePal {
    /// Today's palette (`surface::Palette`).
    pub fn default_pal() -> BiomePal {
        let p = crate::surface::Palette::new();
        let mut c = [DVec3::ZERO; pal::N];
        c[pal::SOIL..pal::SOIL + 4].copy_from_slice(&p.soil);
        c[pal::GRASS_WET] = p.grass_wet;
        c[pal::GRASS_DRY] = p.grass_dry;
        c[pal::GRASS_COLD] = p.grass_cold;
        c[pal::TUNDRA] = p.tundra;
        c[pal::MARSH] = p.marsh;
        c[pal::ROCK..pal::ROCK + 3].copy_from_slice(&p.rock);
        c[pal::SAND..pal::SAND + 3].copy_from_slice(&p.sand);
        c[pal::BEACH] = p.beach;
        c[pal::WET_SAND] = p.wet_sand;
        c[pal::CROWN_CONIFER] = p.crown_conifer;
        c[pal::CROWN_DECID] = p.crown_decid;
        c[pal::CROWN_TROPIC] = p.crown_tropic;
        c[pal::CROWN_DRY] = p.crown_dry;
        c[pal::SHRUB] = p.shrub;
        c[pal::LATERITE] = crate::surface::srgb(146.0, 82.0, 54.0);
        c[pal::FLOOR] = p.crown_conifer + (p.soil[0] - p.crown_conifer) * 0.45;
        BiomePal { c }
    }
}

/// Vegetation / ground parameters of a biome (`vegetation:`).
#[derive(Clone, Copy, Debug)]
pub struct Veg {
    pub trees: f64,
    pub conifer: f64,
    pub tropic: f64,
    pub shrubs: f64,
    pub savanna: f64,
    pub groves: f64,
    pub woodlots: f64,
    pub tall: f64,
    pub laterite: f64,
    pub meadow: f64,
    pub crown_scale: f64,
    pub gallery: f64,
}

/// Canopy model shares a crown layer takes its density from.
pub mod share {
    pub const CONIFER: u32 = 0;
    pub const BROADLEAF: u32 = 1;
    pub const TROPICAL: u32 = 2;
    pub const SHRUB: u32 = 3;
    pub const SAVANNA: u32 = 4;
    pub const ALL: u32 = 5;
    pub const NAMES: [&str; 6] = ["conifer", "broadleaf", "tropical", "shrub", "savanna", "all"];
}

/// One crown layer of the canopy.
#[derive(Clone, Copy, Debug)]
pub struct Crown {
    pub share: u32,
    pub shape: u32,
    pub cell: f64,
    pub h0: f64,
    pub h_tall: f64,
    pub open_height: f64,
    pub colour: DVec3,
    pub colour_dry: Option<DVec3>,
    pub density: f64,
    pub closure: bool,
    pub stand: bool,
    pub seed: u64,
}

/// A mask window over a field.
#[derive(Clone, Copy, Debug, Default)]
pub struct Window {
    pub field: usize,
    pub a0: f64,
    pub a1: f64,
    pub b0: f64,
    pub b1: f64,
}

impl Window {
    #[inline]
    pub fn eval(&self, x: f64) -> f64 {
        let up = if self.a1 > self.a0 { smoothstep(self.a0, self.a1, x) } else if x >= self.a0 { 1.0 } else { 0.0 };
        let down = if self.b1 > self.b0 { 1.0 - smoothstep(self.b0, self.b1, x) } else if x <= self.b0 { 1.0 } else { 0.0 };
        up * down
    }
}

/// One kernel layer of a biome.
#[derive(Clone, Debug)]
pub struct LayerInst {
    pub slot: usize,
    pub k: KParams,
    pub mean: KMean,
    /// feature size (m) of the explicit / mean crossfade
    pub size: f64,
    pub win: Vec<Window>,
    pub cls: u8,
    pub clear: f64,
    pub hmode: u8,
    pub mat: u8,
}

impl LayerInst {
    pub fn mask(&self, s: &crate::stack::Stack) -> f64 {
        let mut m = 1.0;
        for w in &self.win {
            m *= w.eval(s.field(w.field));
            if m <= 0.0 {
                return 0.0;
            }
        }
        m
    }
}

#[derive(Clone, Debug)]
pub struct Biome {
    pub name: String,
    pub group: String,
    /// bit k: Köppen class k (`atlas::Koppen as usize`); 0: any
    pub koppen: u32,
    /// [lo, hi] of temperature (°C at the ecoregion site), precipitation (mm/yr), dry months
    pub env: [[f64; 2]; 3],
    /// bit k: lithology k; 0: any
    pub litho: u8,
    pub weight: f64,
    pub min_share: f64,
    pub pal: BiomePal,
    pub veg: Veg,
    pub crowns: Vec<Crown>,
    /// (upper temperature °C, biome index), warmest first
    pub zones: Vec<(f64, u16)>,
    pub layers: Vec<LayerInst>,
    /// per GSD band: indices into `layers` that matter there
    pub bands: [Vec<u16>; BANDS],
    pub agriculture: f64,
    pub towns: f64,
}

pub struct Registry {
    pub biomes: Vec<Biome>,
    pub resolved: serde_yaml::Value,
}

/// Errors of one source, prefixed with its key path.
struct Errs {
    v: Vec<String>,
}

impl Errs {
    fn push(&mut self, key: &str, msg: impl std::fmt::Display) {
        self.v.push(format!("{key}: {msg}"));
    }
}

fn colour(c: &ColourYaml, def: &BiomePal) -> Result<DVec3, String> {
    match c {
        ColourYaml::Rgb([r, g, b]) => {
            if [r, g, b].iter().any(|v| !(0.0..=255.0).contains(*v)) {
                return Err(format!("colour [{r}, {g}, {b}]: components must be 0..255 (sRGB)"));
            }
            Ok(crate::surface::srgb(*r, *g, *b))
        }
        ColourYaml::Name(n) => match pal::KEYS.iter().find(|k| k.0 == n.as_str() && k.2 == 1) {
            Some(k) => Ok(def.c[k.1]),
            None => match n.as_str() {
                "crop_mean" => Ok(crate::surface::Palette::new().crop_mean),
                _ => Err(format!("unknown colour name {n:?} (a palette entry such as crown_decid, or [r, g, b])")),
            },
        },
    }
}

fn window(key: &str, w: &WindowYaml, e: &mut Errs) -> Option<(f64, f64, f64, f64)> {
    let v = &w.0;
    let lo = f64::NEG_INFINITY;
    let hi = f64::INFINITY;
    let get = |i: usize, d: f64| v.get(i).copied().flatten().unwrap_or(d);
    let r = match v.len() {
        2 => {
            let (a, b) = (get(0, lo), get(1, hi));
            let soft = if a.is_finite() && b.is_finite() { 0.1 * (b - a) } else { 0.0 };
            (a - soft, a + soft, b - soft, b + soft)
        }
        4 => (get(0, lo), get(1, lo), get(2, hi), get(3, hi)),
        _ => {
            e.push(key, "a window is [lo, hi] or [a0, a1, b0, b1] (null: open)");
            return None;
        }
    };
    if r.0.is_nan() || r.1.is_nan() || r.2.is_nan() || r.3.is_nan() || r.1 < r.0 || r.3 < r.2 || r.2 < r.0 {
        e.push(key, format!("window {v:?} must be increasing"));
        return None;
    }
    Some(r)
}

impl Registry {
    /// The builtin registry (core + kits).
    pub fn builtin() -> Result<Registry> {
        Self::from_sources(&builtin_sources())
    }

    /// The registry of a world: the builtin one with the config's overrides (compiled once per
    /// distinct set of overrides in a process).
    pub fn for_config(cfg: &crate::Config) -> Result<std::sync::Arc<Registry>> {
        use std::sync::{Arc, Mutex};
        static CACHE: Mutex<Vec<(String, Arc<Registry>)>> = Mutex::new(Vec::new());
        let key = serde_yaml::to_string(&cfg.biomes.overrides).unwrap_or_default();
        if let Some(r) = CACHE.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|e| e.0 == key) {
            return Ok(r.1.clone());
        }
        let mut src: Vec<(&str, String)> = builtin_sources().into_iter().map(|(n, t)| (n, t.to_string())).collect();
        if !cfg.biomes.overrides.is_empty() {
            let base = Self::from_sources(&builtin_sources())?;
            let mut y = String::new();
            for (id, o) in &cfg.biomes.overrides {
                if base.index(id).is_none() {
                    bail!("overrides.{id}: unknown biome (one of {})", base.biomes.iter().map(|b| b.name.as_str()).collect::<Vec<_>>().join(", "));
                }
                if let Some(w) = o.weight {
                    y.push_str(&format!("- {{id: {id}, patch: true, weight: {w}}}\n"));
                }
            }
            src.push(("overrides", y));
        }
        let refs: Vec<(&str, &str)> = src.iter().map(|(n, t)| (*n, t.as_str())).collect();
        let r = Arc::new(Self::from_sources(&refs)?);
        let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if c.len() > 16 {
            c.clear();
        }
        c.push((key, r.clone()));
        Ok(r)
    }

    /// The registry from YAML sources (name, text), later sources patching or adding biomes.
    pub fn from_sources(sources: &[(&str, &str)]) -> Result<Registry> {
        // ---- parse and merge by id (a later `patch: true` entry changes an earlier biome)
        let mut defs: Vec<BiomeYaml> = Vec::new();
        let mut origin: Vec<String> = Vec::new();
        for (name, text) in sources {
            let list: Vec<BiomeYaml> = serde_yaml::from_str(text).with_context(|| format!("biome registry {name}"))?;
            for b in list {
                match defs.iter().position(|d| d.id == b.id) {
                    Some(i) if b.patch => {
                        let d = &mut defs[i];
                        macro_rules! take {
                            ($($f:ident),*) => { $( if b.$f.is_some() { d.$f = b.$f.clone(); } )* };
                        }
                        take!(group, koppen, envelope, lithology, weight, min_share, vegetation, canopy, zonation, layers, landuse);
                        if let Some(p) = b.palette {
                            d.palette.get_or_insert_with(BTreeMap::new).extend(p);
                        }
                        origin[i] = format!("{} (patched by {name})", origin[i]);
                    }
                    Some(_) => bail!("biome registry {name}: biome {:?} is defined twice (use `patch: true` to change it)", b.id),
                    None if b.patch => bail!("biome registry {name}: `patch: true` for unknown biome {:?}", b.id),
                    None => {
                        origin.push(name.to_string());
                        defs.push(b);
                    }
                }
            }
        }
        Self::compile(defs, &origin)
    }

    fn compile(defs: Vec<BiomeYaml>, origin: &[String]) -> Result<Registry> {
        let mut e = Errs { v: Vec::new() };
        let def_pal = BiomePal::default_pal();
        let names: Vec<String> = defs.iter().map(|d| d.id.clone()).collect();
        let mut biomes = Vec::new();
        if defs.is_empty() {
            bail!("biome registry: no biomes");
        }
        for (bi, d) in defs.iter().enumerate() {
            let key = format!("biomes.{}", d.id);
            if d.id.is_empty() || !d.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
                e.push(&key, "ids are lower_snake_case");
            }
            // koppen, envelope, lithology
            let mut koppen = 0u32;
            for k in d.koppen.iter().flatten() {
                match crate::atlas_stub::Koppen::from_code(k) {
                    Some(c) => koppen |= 1 << c as u32,
                    None => e.push(&format!("{key}.koppen"), format!("unknown Köppen class {k:?}")),
                }
            }
            let env_y = d.envelope.clone().unwrap_or_default();
            let mut env = [[-100.0, 100.0], [0.0, 1e5], [0.0, 12.0]];
            for (i, (nm, v)) in [("temp", env_y.temp), ("precip_mm", env_y.precip_mm), ("dry_months", env_y.dry_months)].iter().enumerate() {
                if let Some([a, b]) = v {
                    if !(a.is_finite() && b.is_finite() && a <= b) {
                        e.push(&format!("{key}.envelope.{nm}"), format!("[{a}, {b}] must be finite and increasing"));
                    }
                    env[i] = [*a, *b];
                }
            }
            let litho = match &d.lithology {
                None => 0,
                Some(LithoYaml::Any(s)) if s == "any" => 0,
                Some(LithoYaml::Any(s)) => {
                    e.push(&format!("{key}.lithology"), format!("{s:?}: `any` or a list of {:?}", crate::atlas_stub::Lithology::ALL.map(|l| l.name())));
                    0
                }
                Some(LithoYaml::List(l)) => {
                    let mut m = 0u8;
                    for n in l {
                        match crate::atlas_stub::Lithology::ALL.iter().find(|x| x.name() == n) {
                            Some(x) => m |= 1 << *x as u8,
                            None => e.push(&format!("{key}.lithology"), format!("unknown lithology {n:?}")),
                        }
                    }
                    m
                }
            };
            let weight = d.weight.unwrap_or(1.0);
            if !(weight.is_finite() && weight >= 0.0) {
                e.push(&format!("{key}.weight"), "must be finite and >= 0");
            }
            // palette
            let mut p = def_pal;
            for (k, v) in d.palette.iter().flatten() {
                let Some(&(_, first, n)) = pal::KEYS.iter().find(|x| x.0 == k.as_str() && x.2 > 0) else {
                    e.push(&format!("{key}.palette.{k}"), format!("unknown entry (one of {})", pal::KEYS.iter().filter(|x| x.2 > 0).map(|x| x.0).collect::<Vec<_>>().join(", ")));
                    continue;
                };
                let list: Vec<ColourYaml> = match v {
                    PalYaml::One(c) => vec![c.clone(); n],
                    PalYaml::Many(l) => l.clone(),
                };
                if list.len() != n {
                    e.push(&format!("{key}.palette.{k}"), format!("{n} colours expected"));
                    continue;
                }
                for (i, c) in list.iter().enumerate() {
                    match colour(c, &def_pal) {
                        Ok(c) => p.c[first + i] = c,
                        Err(m) => e.push(&format!("{key}.palette.{k}"), m),
                    }
                }
            }
            let vy = d.vegetation.clone().unwrap_or_default();
            for (nm, v, lo, hi) in [
                ("trees", vy.trees, 0.0, 4.0),
                ("conifer", vy.conifer, -1.0, 1.0),
                ("tropic", vy.tropic, -1.0, 1.0),
                ("shrubs", vy.shrubs, 0.0, 4.0),
                ("savanna", vy.savanna, 0.0, 10.0),
                ("groves", vy.groves, 0.0, 10.0),
                ("woodlots", vy.woodlots, 0.0, 3.0),
                ("tall", vy.tall, -1.0, 1.0),
                ("laterite", vy.laterite, 0.0, 2.0),
                ("meadow", vy.meadow, 0.0, 3.0),
                ("crown_scale", vy.crown_scale, 0.3, 2.0),
                ("gallery", vy.gallery, 0.0, 1.0),
            ] {
                if !(v.is_finite() && (lo..=hi).contains(&v)) {
                    e.push(&format!("{key}.vegetation.{nm}"), format!("must be in [{lo}, {hi}] (is {v})"));
                }
            }
            let veg = Veg {
                trees: vy.trees,
                conifer: vy.conifer,
                tropic: vy.tropic,
                shrubs: vy.shrubs,
                savanna: vy.savanna,
                groves: vy.groves,
                woodlots: vy.woodlots,
                tall: vy.tall,
                laterite: vy.laterite,
                meadow: vy.meadow,
                crown_scale: vy.crown_scale,
                gallery: vy.gallery,
            };
            // canopy
            let crowns_y = d.canopy.clone().unwrap_or_else(default_crowns);
            if crowns_y.len() > MAX_CROWNS {
                e.push(&format!("{key}.canopy"), format!("at most {MAX_CROWNS} crown layers"));
            }
            let mut crowns = Vec::new();
            for (ci, c) in crowns_y.iter().enumerate() {
                let ck = format!("{key}.canopy[{ci}]");
                let sh = share::NAMES.iter().position(|n| *n == c.share);
                if sh.is_none() {
                    e.push(&format!("{ck}.share"), format!("{:?}: one of {:?}", c.share, share::NAMES));
                }
                let shape = kernels::shape::id(&c.shape);
                if shape.is_none() {
                    e.push(&format!("{ck}.shape"), format!("{:?}: one of {:?}", c.shape, kernels::shape::NAMES));
                }
                if !(c.cell.is_finite() && (0.5..=200.0).contains(&c.cell)) {
                    e.push(&format!("{ck}.cell"), "must be in [0.5, 200] m");
                }
                if !(c.height[0].is_finite() && c.height[0] >= 0.0 && c.height[1].is_finite()) {
                    e.push(&format!("{ck}.height"), "must be [base >= 0, per unit of tall]");
                }
                let col = colour(&c.colour, &p).unwrap_or_else(|m| {
                    e.push(&format!("{ck}.colour"), m);
                    DVec3::ZERO
                });
                let col_dry = c.colour_dry.as_ref().map(|x| {
                    colour(x, &p).unwrap_or_else(|m| {
                        e.push(&format!("{ck}.colour_dry"), m);
                        DVec3::ZERO
                    })
                });
                crowns.push(Crown {
                    share: sh.unwrap_or(0) as u32,
                    shape: shape.unwrap_or(0),
                    cell: c.cell,
                    h0: c.height[0],
                    h_tall: c.height[1],
                    open_height: c.open_height,
                    colour: col,
                    colour_dry: col_dry,
                    density: c.density,
                    closure: c.closure,
                    stand: c.stand,
                    seed: c.seed.unwrap_or(0x7EE1 + ci as u64),
                });
            }
            // zonation
            let mut zones = Vec::new();
            for (zi, z) in d.zonation.iter().flatten().enumerate() {
                match names.iter().position(|n| *n == z.biome) {
                    Some(i) if i != bi => zones.push((z.below_c, i as u16)),
                    Some(_) => e.push(&format!("{key}.zonation[{zi}]"), "a biome cannot be its own zone"),
                    None => e.push(&format!("{key}.zonation[{zi}].biome"), format!("unknown biome {:?}", z.biome)),
                }
            }
            if zones.len() > MAX_ZONES {
                e.push(&format!("{key}.zonation"), format!("at most {MAX_ZONES} zones"));
            }
            zones.sort_by(|a, b| b.0.total_cmp(&a.0));
            // kernel layers
            let ly = d.layers.clone().unwrap_or_default();
            if ly.len() > MAX_LAYERS {
                e.push(&format!("{key}.layers"), format!("at most {MAX_LAYERS} layers (is {})", ly.len()));
            }
            let mut layers = Vec::new();
            for (li, l) in ly.iter().enumerate() {
                let lk = format!("{key}.layers[{li}]");
                let slot = crate::stack::slot::NAMES.iter().position(|n| *n == l.slot);
                match slot {
                    None | Some(0) => {
                        e.push(&format!("{lk}.slot"), format!("{:?}: one of {:?}", l.slot, &crate::stack::slot::NAMES[1..]));
                        continue;
                    }
                    _ => {}
                }
                let Some(spec) = kernels::spec(&l.kernel) else {
                    e.push(&format!("{lk}.kernel"), format!("unknown kernel {:?} (one of {})", l.kernel, kernels::spec_names().join(", ")));
                    continue;
                };
                let seed = mix64(l.seed.unwrap_or_else(|| hash2(0xB10E, bi as i64, li as i64)));
                let k = match kernels::compile_params(spec, &l.params, seed, &p) {
                    Ok(k) => k,
                    Err(m) => {
                        e.push(&format!("{lk}.params"), m);
                        continue;
                    }
                };
                let mut win = Vec::new();
                for (fname, w) in &l.mask {
                    let Some(f) = field::id(fname) else {
                        e.push(&format!("{lk}.mask.{fname}"), format!("unknown field (one of {})", field::NAMES.join(", ")));
                        continue;
                    };
                    if let Some((a0, a1, b0, b1)) = window(&format!("{lk}.mask.{fname}"), w, &mut e) {
                        win.push(Window { field: f, a0, a1, b0, b1 });
                    }
                }
                if win.len() > MAX_WINDOWS {
                    e.push(&format!("{lk}.mask"), format!("at most {MAX_WINDOWS} windows"));
                }
                let cls = match &l.class {
                    None => 0,
                    Some(c) => classes::id(c).unwrap_or_else(|| {
                        e.push(&format!("{lk}.class"), format!("unknown class {c:?}"));
                        0
                    }),
                };
                let hmode = match l.height.as_deref().unwrap_or(spec.default_hmode) {
                    "none" => crate::stack::hmode::NONE,
                    "blend" => crate::stack::hmode::BLEND,
                    "max" => crate::stack::hmode::MAX,
                    "add" => crate::stack::hmode::ADD,
                    h => {
                        e.push(&format!("{lk}.height"), format!("{h:?}: none, blend, max or add"));
                        0
                    }
                };
                if !(0.0..=1.0).contains(&l.clear) {
                    e.push(&format!("{lk}.clear"), "must be in [0, 1]");
                }
                let size = kernels::size(&k);
                if let Err(m) = kernels::check_reach(&k) {
                    e.push(&format!("{lk}.params"), m);
                }
                let mean = kernels::calibrate(&k);
                layers.push(LayerInst { slot: slot.unwrap(), k, mean, size, win, cls, clear: l.clear, hmode, mat: l.material });
            }
            let bands: [Vec<u16>; BANDS] = std::array::from_fn(|b| {
                let (lo, _) = band_range(b);
                (0..layers.len() as u16)
                    .filter(|&i| {
                        let l = &layers[i as usize];
                        // explicit somewhere in the band, or a mean that changes the look
                        l.size > 1.2 * lo || l.mean.cov > 0.002 || l.mean.emit > 1e-4
                    })
                    .collect()
            });
            let lu = d.landuse.clone().unwrap_or_default();
            biomes.push(Biome {
                name: d.id.clone(),
                group: d.group.clone().unwrap_or_else(|| "vegetation".into()),
                koppen,
                env,
                litho,
                weight,
                min_share: d.min_share.unwrap_or(0.0),
                pal: p,
                veg,
                crowns,
                zones,
                layers,
                bands,
                agriculture: lu.agriculture,
                towns: lu.towns,
            });
            let _ = &origin[bi];
        }
        if biomes.len() > 1024 {
            e.push("biomes", "at most 1024 biomes");
        }
        if !e.v.is_empty() {
            let n = e.v.len();
            bail!("biome registry: {}{}", e.v[..n.min(8)].join("; "), if n > 8 { format!(" (and {} more)", n - 8) } else { String::new() });
        }
        let resolved = serde_yaml::to_value(&defs).unwrap_or(serde_yaml::Value::Null);
        Ok(Registry { biomes, resolved })
    }

    pub fn index(&self, name: &str) -> Option<u16> {
        self.biomes.iter().position(|b| b.name == name).map(|i| i as u16)
    }

    /// The biome picked for an ecoregion: those whose Köppen classes, envelope and lithology
    /// fit, drawn by weight (× `pref`) with `u` ∈ [0, 1); none fitting: the nearest envelope.
    pub fn pick(&self, k: u32, temp: f64, precip: f64, dry: f64, litho: u8, u: f64, pref: impl Fn(&Biome) -> f64) -> u16 {
        let fits = |b: &Biome| {
            (b.koppen == 0 || b.koppen & (1 << k) != 0)
                && (b.litho == 0 || b.litho & (1 << litho) != 0)
                && (b.env[0][0]..=b.env[0][1]).contains(&temp)
                && (b.env[1][0]..=b.env[1][1]).contains(&precip)
                && (b.env[2][0]..=b.env[2][1]).contains(&dry)
        };
        let w: Vec<f64> = self.biomes.iter().map(|b| if fits(b) { b.weight * pref(b) } else { 0.0 }).collect();
        let total: f64 = w.iter().sum();
        if total > 0.0 {
            let mut acc = 0.0;
            for (i, wi) in w.iter().enumerate() {
                acc += wi / total;
                if u < acc && *wi > 0.0 {
                    return i as u16;
                }
            }
            return w.iter().rposition(|x| *x > 0.0).unwrap_or(0) as u16;
        }
        // nearest envelope (normalized distance), Köppen-matching first
        let dist = |b: &Biome| {
            let d = |x: f64, r: [f64; 2], s: f64| ((r[0] - x).max(0.0) + (x - r[1]).max(0.0)) / s;
            let kp = if b.koppen == 0 || b.koppen & (1 << k) != 0 { 0.0 } else { 10.0 };
            kp + d(temp, b.env[0], 5.0) + d(precip, b.env[1], 400.0) + d(dry, b.env[2], 3.0)
        };
        (0..self.biomes.len()).min_by(|&a, &b| dist(&self.biomes[a]).total_cmp(&dist(&self.biomes[b]))).unwrap_or(0) as u16
    }
}

/// Today's four crown layers (conifers, broadleaves, tropical crowns, shrubs).
fn default_crowns() -> Vec<CrownYaml> {
    let c = |share: &str, shape: &str, cell: f64, h: [f64; 2], open: f64, col: &str, dry: Option<&str>, closure: bool, stand: bool| CrownYaml {
        share: share.into(),
        shape: shape.into(),
        cell,
        height: h,
        open_height: open,
        colour: ColourYaml::Name(col.into()),
        colour_dry: dry.map(|d| ColourYaml::Name(d.into())),
        density: 1.0,
        closure,
        stand,
        seed: None,
    };
    vec![
        c("conifer", "cone", 5.5, [14.0, 10.0], 1.0, "crown_conifer", None, true, true),
        c("broadleaf", "dome", 8.5, [10.0, 10.0], 0.65, "crown_decid", Some("crown_dry"), true, true),
        c("tropical", "dome", 13.0, [22.0, 14.0], 1.0, "crown_tropic", None, true, true),
        c("shrub", "dome", 3.2, [1.6, 0.0], 1.0, "shrub", None, false, false),
    ]
}

// ============================================================================= per sample

/// Climate of an ecoregion site (from the atlas).
#[derive(Clone, Copy, Debug, Default)]
pub struct SiteClimate {
    pub temp_c: f64,
    pub precip_mm: f64,
    pub temp_range_c: f64,
    pub dry_months: f64,
    pub koppen: u32,
}

/// The biome(s) of a sample and the resolved style of its ecoregion.
#[derive(Clone, Copy, Debug)]
pub struct BioSample {
    /// the biome (index into `Registry::biomes`)
    pub a: u16,
    /// the biome blended in (coarse pixels at ecotones / zone borders) and its weight 0..1
    pub b: u16,
    pub w: f64,
    /// style of the ecoregion (blended as the biomes are)
    pub style: crate::eco::EcoStyle,
    pub site: SiteClimate,
    /// distance to the ecoregion border (km)
    pub edge_km: f64,
}

impl BioSample {
    /// A sample's biomes: the ecoregion (A or B, dithered within the ecotone, blended where the
    /// dither is below the pixel), then the zone of that biome by the sample's temperature.
    pub fn at(world: &crate::world::World, sm: &crate::surface::SurfaceModel, cache: &mut crate::surface::Caches, ctx: &crate::world::Ctx, l: &crate::surface::Local, _pf: &crate::surface::PixFields) -> BioSample {
        let t = l.t;
        let reg = &sm.registry;
        // the pixel's pair of ecoregions; a negative border distance: the sample lies on the
        // other one's side
        let (mut ia, mut ib, mut ca, mut cb) = (t.eco.id, t.eco.id2, t.eco.center, t.eco.center2);
        let mut edge = l.eco_edge;
        if edge < 0.0 {
            std::mem::swap(&mut ia, &mut ib);
            std::mem::swap(&mut ca, &mut cb);
            edge = -edge;
        }
        let ea = sm.eco.params(world, reg, cache, ia, ca);
        let wb = if ib != 0 && ib != ia { crate::eco::ecotone_pick(world, ctx.p, ctx.gsd, edge) } else { 0.0 };
        let eb = if wb > 0.0 { sm.eco.params(world, reg, cache, ib, cb) } else { ea };
        // the sample's ecoregion and the one blended in (where the mosaic is below the pixel)
        let (e1, e2, w) = if wb > 0.5 { (eb, ea, 1.0 - wb) } else { (ea, eb, wb) };
        let z1 = zone(reg, e1.biome, t.temp, ctx.p, ctx.gsd);
        let (a, b, wz) = if w > 0.0 {
            let z2 = zone(reg, e2.biome, t.temp, ctx.p, ctx.gsd);
            (z1.0, z2.0, w)
        } else {
            z1
        };
        let style = if w > 0.0 { e1.style.mix(&e2.style, w) } else { e1.style };
        BioSample { a, b, w: if a == b { 0.0 } else { wz }, style, site: e1.climate, edge_km: edge / 1000.0 }
    }

    /// A palette colour of the sample's biome(s).
    #[inline]
    pub fn pal(&self, reg: &Registry, i: usize) -> DVec3 {
        let a = reg.biomes[self.a as usize].pal.c[i];
        if self.w <= 0.0 {
            return a;
        }
        let b = reg.biomes[self.b as usize].pal.c[i];
        a + (b - a) * self.w
    }

    /// A vegetation parameter of the sample's biome(s).
    #[inline]
    pub fn veg(&self, reg: &Registry, f: impl Fn(&Veg) -> f64) -> f64 {
        let a = f(&reg.biomes[self.a as usize].veg);
        if self.w <= 0.0 {
            return a;
        }
        a + (f(&reg.biomes[self.b as usize].veg) - a) * self.w
    }
}

/// The zone of biome `b` at temperature `temp`: (zone, partner across the nearest zone
/// border, partner weight); dithered over ±1 °C where the dither is resolved.
fn zone(reg: &Registry, b: u16, temp: f64, p: DVec3, gsd: f64) -> (u16, u16, f64) {
    let bz = &reg.biomes[b as usize];
    if bz.zones.is_empty() {
        return (b, b, 0.0);
    }
    // zones sorted warmest first: the sample's zone is the last whose upper limit is above temp
    let n = perlin3(0x20E5, p / 260.0) * 0.8 + 0.4 * perlin3(0x20E6, p / 90.0);
    let ex = band(260.0, gsd);
    let tt = temp + 1.2 * n * ex;
    let mut cur = b;
    let mut partner = (b, 0.0);
    for &(below, zb) in &bz.zones {
        // weight of the colder zone across this border (blend where the dither is unresolved)
        let wz = (1.0 - ex) * (1.0 - smoothstep(below - 1.0, below + 1.0, temp));
        if tt < below {
            cur = zb;
        } else if wz > partner.1 {
            partner = (zb, wz);
        }
    }
    (cur, partner.0, if cur == b { partner.1 } else { 0.0 })
}

// ============================================================================= GPU tables

/// The registry as GPU tables (`registry.wgsl`).
#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    /// `Biome` of registry.wgsl (fixed size: palette, vegetation, ranges).
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub struct GBiome {
        pub pal: [[f32; 4]; pal::N],
        /// trees, conifer, shrubs, savanna
        pub veg0: [f32; 4],
        /// groves, woodlots, tall, laterite
        pub veg1: [f32; 4],
        /// meadow, tropic, crown_scale, gallery
        pub veg2: [f32; 4],
        /// agriculture, towns, -, -
        pub land: [f32; 4],
        /// first crown, crowns, first zone, zones
        pub r0: [u32; 4],
        /// first layer, layers, first band entry (6 bands × (first, count) in `band_ranges`)
        pub r1: [u32; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    pub struct GCrown {
        pub share: u32,
        pub shape: u32,
        pub seed_lo: u32,
        pub seed_hi: u32,
        pub cell: f32,
        pub h0: f32,
        pub h_tall: f32,
        pub open_height: f32,
        pub density: f32,
        /// bit 0: closure, bit 1: stand, bit 2: has a dry colour
        pub flags: u32,
        pub _p: [u32; 2],
        pub colour: [f32; 4],
        pub colour_dry: [f32; 4],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    pub struct GZone {
        pub below: f32,
        pub biome: u32,
    }

    /// `KLayer` of registry.wgsl: a kernel instance with its mask and mean.
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    pub struct GLayer {
        /// slot, kernel kind, class, hmode | mat << 8
        pub head: [u32; 4],
        /// seed (lo, hi), windows, -
        pub head2: [u32; 4],
        /// size, clear, -, -
        pub sz: [f32; 4],
        /// window fields
        pub wf: [u32; 4],
        /// windows (a0, a1, b0, b1)
        pub win: [[f32; 4]; MAX_WINDOWS],
        pub v: [[f32; 4]; 4],
        pub col: [[f32; 4]; 3],
        /// mean coverage, height, emission, -
        pub mean: [f32; 4],
        pub mean_albedo: [f32; 4],
    }

    fn v4(c: DVec3) -> [f32; 4] {
        [c.x as f32, c.y as f32, c.z as f32, 0.0]
    }

    /// WGSL constants of the registry, kernels and stack (the same values as here).
    pub fn wgsl_consts() -> String {
        let mut o = String::from("// ---------------------------------------------------------------- registry constants (generated)\n");
        let mut c = |n: &str, v: u32| o.push_str(&format!("const {n}: u32 = {v}u;\n"));
        for (i, n) in field::NAMES.iter().enumerate() {
            c(&format!("FIELD_{}", n.to_uppercase()), i as u32);
        }
        for (n, i, _) in pal::KEYS.iter().filter(|k| k.2 > 0) {
            c(&format!("BP_{}", n.to_uppercase()), *i as u32);
        }
        for (i, n) in share::NAMES.iter().enumerate() {
            c(&format!("SHARE_{}", n.to_uppercase()), i as u32);
        }
        for (i, n) in crate::kernels::shape::NAMES.iter().enumerate() {
            c(&format!("SHAPE_{}", n.to_uppercase()), i as u32);
        }
        for k in crate::kernels::CORE {
            c(&format!("KIND_{}", k.name.to_uppercase()), k.kind);
        }
        c("KIT_BASE", crate::kernels::KIT_BASE);
        for (i, n) in crate::stack::slot::NAMES.iter().enumerate() {
            c(&format!("SLOT_{}", n.to_uppercase()), i as u32);
        }
        use crate::stack::hmode;
        c("HM_NONE", hmode::NONE as u32);
        c("HM_BLEND", hmode::BLEND as u32);
        c("HM_MAX", hmode::MAX as u32);
        c("HM_ADD", hmode::ADD as u32);
        c("HM_ABS", hmode::ABS as u32);
        c("MAX_CROWNS", MAX_CROWNS as u32);
        c("NBANDS", BANDS as u32);
        for (n, id) in classes::TABLE.iter().filter(|e| e.1 > 17) {
            c(&format!("LC_{}", n.to_uppercase()), *id as u32);
        }
        o
    }

    pub struct Tables {
        pub biomes: Vec<GBiome>,
        pub crowns: Vec<GCrown>,
        pub zones: Vec<GZone>,
        pub layers: Vec<GLayer>,
        pub band_ranges: Vec<[u32; 2]>,
        pub band_idx: Vec<u32>,
    }

    pub fn klayer(l: &LayerInst) -> GLayer {
        let k = &l.k;
        let fin = |x: f64| if x.is_finite() { x as f32 } else if x > 0.0 { 3.0e38 } else { -3.0e38 };
        let mut g = GLayer {
            head: [l.slot as u32, k.kind, l.cls as u32, l.hmode as u32 | (l.mat as u32) << 8],
            head2: [k.seed as u32, (k.seed >> 32) as u32, l.win.len() as u32, 0],
            sz: [l.size as f32, l.clear as f32, 0.0, 0.0],
            ..Default::default()
        };
        for (i, w) in l.win.iter().enumerate().take(MAX_WINDOWS) {
            g.wf[i] = w.field as u32;
            g.win[i] = [fin(w.a0), fin(w.a1), fin(w.b0), fin(w.b1)];
        }
        for i in 0..16 {
            g.v[i / 4][i % 4] = k.v[i] as f32;
        }
        for i in 0..3 {
            g.col[i] = v4(k.col[i]);
        }
        g.mean = [l.mean.cov as f32, l.mean.dh as f32, l.mean.emit as f32, 0.0];
        g.mean_albedo = v4(l.mean.albedo);
        g
    }

    impl Registry {
        pub fn gpu(&self) -> Tables {
            let mut t = Tables { biomes: vec![], crowns: vec![], zones: vec![], layers: vec![], band_ranges: vec![], band_idx: vec![] };
            for b in &self.biomes {
                let c0 = t.crowns.len() as u32;
                for c in &b.crowns {
                    t.crowns.push(GCrown {
                        share: c.share,
                        shape: c.shape,
                        seed_lo: c.seed as u32,
                        seed_hi: (c.seed >> 32) as u32,
                        cell: c.cell as f32,
                        h0: c.h0 as f32,
                        h_tall: c.h_tall as f32,
                        open_height: c.open_height as f32,
                        density: c.density as f32,
                        flags: c.closure as u32 | (c.stand as u32) << 1 | (c.colour_dry.is_some() as u32) << 2,
                        _p: [0; 2],
                        colour: v4(c.colour),
                        colour_dry: v4(c.colour_dry.unwrap_or(c.colour)),
                    });
                }
                let z0 = t.zones.len() as u32;
                for &(below, zb) in &b.zones {
                    t.zones.push(GZone { below: below as f32, biome: zb as u32 });
                }
                let l0 = t.layers.len() as u32;
                t.layers.extend(b.layers.iter().map(klayer));
                let br0 = t.band_ranges.len() as u32;
                for band in &b.bands {
                    t.band_ranges.push([t.band_idx.len() as u32, band.len() as u32]);
                    t.band_idx.extend(band.iter().map(|&i| l0 + i as u32));
                }
                let v = &b.veg;
                t.biomes.push(GBiome {
                    pal: b.pal.c.map(v4),
                    veg0: [v.trees as f32, v.conifer as f32, v.shrubs as f32, v.savanna as f32],
                    veg1: [v.groves as f32, v.woodlots as f32, v.tall as f32, v.laterite as f32],
                    veg2: [v.meadow as f32, v.tropic as f32, v.crown_scale as f32, v.gallery as f32],
                    land: [b.agriculture as f32, b.towns as f32, 0.0, 0.0],
                    r0: [c0, b.crowns.len() as u32, z0, b.zones.len() as u32],
                    r1: [l0, b.layers.len() as u32, br0, 0],
                });
            }
            // (never empty: storage buffers need a size)
            if t.crowns.is_empty() {
                t.crowns.push(GCrown::default());
            }
            if t.zones.is_empty() {
                t.zones.push(GZone::default());
            }
            if t.layers.is_empty() {
                t.layers.push(GLayer::default());
            }
            if t.band_idx.is_empty() {
                t.band_idx.push(0);
            }
            t
        }
    }

    #[cfg(test)]
    #[test]
    fn sizes() {
        assert_eq!(std::mem::size_of::<GBiome>(), 16 * (pal::N + 6));
        assert_eq!(std::mem::size_of::<GCrown>(), 80);
        assert_eq!(std::mem::size_of::<GLayer>(), 16 * (4 + 4 + 4 + 3 + 2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(yaml: &str) -> String {
        match Registry::from_sources(&[("core", include_str!("../biomes/core.yaml")), ("test", yaml)]) {
            Ok(_) => panic!("accepted:\n{yaml}"),
            Err(e) => format!("{e:#}"),
        }
    }

    #[test]
    fn builtin_registry_compiles() {
        let r = Registry::builtin().unwrap();
        assert!(r.biomes.len() >= 6);
        assert!(r.index("temperate").is_some());
        for b in &r.biomes {
            assert!(b.layers.len() <= MAX_LAYERS && b.crowns.len() <= MAX_CROWNS, "{}", b.name);
        }
    }

    #[test]
    fn bad_definitions_name_their_key() {
        let e = err("- {id: x, koppen: [Qq]}");
        assert!(e.contains("biomes.x.koppen") && e.contains("Qq"), "{e}");
        let e = err("- {id: x, palette: {soil: [[1, 2, 3], [4, 5, 6]]}}");
        assert!(e.contains("biomes.x.palette.soil"), "{e}");
        let e = err("- {id: x, palette: {grass_wet: [300, 0, 0]}}");
        assert!(e.contains("biomes.x.palette.grass_wet") && e.contains("0..255"), "{e}");
        let e = err("- {id: x, layers: [{slot: azonal, kernel: nope}]}");
        assert!(e.contains("biomes.x.layers[0].kernel") && e.contains("nope"), "{e}");
        let e = err("- {id: x, layers: [{slot: roof, kernel: scatter}]}");
        assert!(e.contains("biomes.x.layers[0].slot"), "{e}");
        let e = err("- {id: x, layers: [{slot: azonal, kernel: scatter, mask: {wetness: [0, 1]}}]}");
        assert!(e.contains("biomes.x.layers[0].mask.wetness"), "{e}");
        let e = err("- {id: x, layers: [{slot: azonal, kernel: scatter, mask: {moist: [0.5, 0.1]}}]}");
        assert!(e.contains("increasing"), "{e}");
        let e = err("- {id: x, layers: [{slot: azonal, kernel: scatter, params: {cell: 10, colur: [1, 2, 3]}}]}");
        assert!(e.contains("biomes.x.layers[0].params") && e.contains("colur"), "{e}");
        let e = err("- {id: x, zonation: [{below_c: 0, biome: nowhere}]}");
        assert!(e.contains("biomes.x.zonation[0].biome"), "{e}");
        let e = err("- {id: temperate, weight: 2}");
        assert!(e.contains("defined twice"), "{e}");
        let e = err("- {id: x, unknown_key: 1}");
        assert!(e.contains("unknown field"), "{e}");
        let many = (0..9).map(|_| "{slot: azonal, kernel: patches}").collect::<Vec<_>>().join(", ");
        let e = err(&format!("- {{id: x, layers: [{many}]}}"));
        assert!(e.contains("at most 8 layers"), "{e}");
        // a patch changes only what it names
        let r = Registry::from_sources(&[("core", include_str!("../biomes/core.yaml")), ("t", "- {id: temperate, patch: true, weight: 3}")]).unwrap();
        assert_eq!(r.biomes[r.index("temperate").unwrap() as usize].weight, 3.0);
    }

    #[test]
    fn bands_cover_the_scales() {
        assert_eq!(band_of(0.3), 0);
        assert_eq!(band_of(1.0), 1);
        assert_eq!(band_of(3.9), 1);
        assert_eq!(band_of(4.0), 2);
        assert_eq!(band_of(300.0), 5);
        assert_eq!(band_range(2), (4.0, 16.0));
    }
}
