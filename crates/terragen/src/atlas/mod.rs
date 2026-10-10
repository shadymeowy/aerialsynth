//! The planetary atlas: world-scale fields that cannot be computed locally, precomputed once per
//! world on a cube map (design `docs/design/terrain-next.md` §3.6).
//!
//! * **Geometry:** a cube map with a tangent warp (near-uniform texels), 6 × r² texels
//!   (`world.atlas.resolution`, 512 by default: ~20 km), and a 2-texel apron per face filled from
//!   the neighbouring faces, so the bicubic sampler never reads across a face edge.
//! * **Channels:** 18 16-bit slots per texel (9 `u32` words): 15 continuous fields as f16, sampled
//!   with a cubic B-spline (smooth, C², no overshoot), and 3 discrete ones as bit fields, taken
//!   from the nearest texel (see [`AtlasSample`] for the fields and their units).
//! * **Determinism:** every texel is an f64 function of the config, or the result of iterations
//!   that read only the previous buffer (jump flooding, moisture advection), so the bytes do
//!   not depend on the thread count. Both backends sample the same buffer: the CPU with
//!   [`Atlas::sample`], WGSL with `atlas_sample` (`gpu/wgsl/atlas.wgsl`, bound by the GPU generator).
//! * **Caching:** [`Atlas::for_world`] keeps the atlases of the last few worlds in the process and
//!   stores them on disk under `$XDG_CACHE_HOME/terrain/atlas-<key>.bin` (or
//!   `~/.cache/terrain`; `TERRAGEN_ATLAS_CACHE=off` disables it, `TERRAGEN_ATLAS_CACHE=DIR`
//!   moves it).
//!
//! Use it only for smooth quantities: per-pixel detail stays analytic.

mod build;
pub mod geom;
mod koppen;
#[cfg(test)]
mod tests;

pub use koppen::{koppen, koppen_with_lapse, Koppen, KoppenGroup};

use crate::world::World;
use crate::Config;
use anyhow::{bail, Context, Result};
use geom::Grid;
use glam::DVec3;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// Version of the atlas' contents: part of the cache key (bump it whenever the fields change).
pub const ATLAS_VERSION: u32 = 4;
/// Apron texels on each side of a face.
pub const APRON: usize = 2;
/// 32-bit words per texel (two 16-bit slots each).
pub const WORDS: usize = 9;
/// Words of the buffer header (before the texel data).
pub const HEADER: usize = 16;
/// Magic number of the buffer header ("ATLS").
pub const MAGIC: u32 = 0x534C_5441;

/// The 16-bit slots of a texel (slot `s` is in word `s / 2`, low half for even `s`).
pub mod slot {
    pub const ELEVATION: usize = 0;
    pub const COAST: usize = 1;
    pub const WIND_E: usize = 2;
    pub const WIND_N: usize = 3;
    pub const PRECIP: usize = 4;
    pub const TEMP: usize = 5;
    pub const TEMP_RANGE: usize = 6;
    pub const REGIME: usize = 7;
    pub const PLATE_DIST: usize = 8;
    pub const CONVERGENCE: usize = 9;
    pub const UPLIFT: usize = 10;
    pub const VOLCANISM: usize = 11;
    pub const GLACIATION: usize = 12;
    pub const DEVELOPMENT: usize = 13;
    pub const POPULATION: usize = 14;
    /// u16: plate id (bits 0–5), boundary type (6–8), continental crust (9), hotspot track (10)
    pub const PLATE: usize = 15;
    /// u16: dominant lithology (bits 0–2), secondary (3–5), secondary fraction × 510 (8–15)
    pub const LITHO: usize = 16;
    /// u16: culture id (bits 0–11), archetype (12–15)
    pub const CULTURE: usize = 17;
    /// The continuous (f16) slots: 0..CONTINUOUS.
    pub const CONTINUOUS: usize = 15;
    pub const COUNT: usize = 18;
}

/// Kind of the nearest plate boundary, seen from the texel's plate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Boundary {
    /// (no boundary: a single-plate world)
    #[default]
    None = 0,
    /// convergent, this side overrides an oceanic plate: coastal range and volcanic arc inland
    Overriding = 1,
    /// convergent, this side subducts under a continent: trench
    Subducting = 2,
    /// convergent continent–continent: collision belt and plateau
    Collision = 3,
    /// convergent ocean–ocean: island arc
    IslandArc = 4,
    /// divergent through continental crust: rift valley
    Rift = 5,
    /// divergent through oceanic crust: mid-ocean ridge
    Ridge = 6,
    /// transform (strike-slip)
    Transform = 7,
}

impl Boundary {
    pub fn from_bits(b: u32) -> Boundary {
        match b & 7 {
            1 => Boundary::Overriding,
            2 => Boundary::Subducting,
            3 => Boundary::Collision,
            4 => Boundary::IslandArc,
            5 => Boundary::Rift,
            6 => Boundary::Ridge,
            7 => Boundary::Transform,
            _ => Boundary::None,
        }
    }
    pub fn is_convergent(self) -> bool {
        matches!(self, Boundary::Overriding | Boundary::Subducting | Boundary::Collision | Boundary::IslandArc)
    }
    pub fn name(self) -> &'static str {
        match self {
            Boundary::None => "none",
            Boundary::Overriding => "overriding",
            Boundary::Subducting => "subducting",
            Boundary::Collision => "collision",
            Boundary::IslandArc => "island_arc",
            Boundary::Rift => "rift",
            Boundary::Ridge => "ridge",
            Boundary::Transform => "transform",
        }
    }
}

/// Rock class at the surface.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Lithology {
    /// clastic sedimentary rock (sandstone, shale)
    #[default]
    Sedimentary = 0,
    /// limestone, dolomite, chalk (karst)
    Carbonate = 1,
    /// granite, gneiss, schist (shields, orogen cores)
    Crystalline = 2,
    /// basalt, andesite, tuff
    Volcanic = 3,
    /// alluvium, loess, till, sand
    Unconsolidated = 4,
}

impl Lithology {
    pub const ALL: [Lithology; 5] = [Lithology::Sedimentary, Lithology::Carbonate, Lithology::Crystalline, Lithology::Volcanic, Lithology::Unconsolidated];
    pub fn from_bits(b: u32) -> Lithology {
        Self::ALL.get((b & 7) as usize).copied().unwrap_or_default()
    }
    pub fn name(self) -> &'static str {
        ["sedimentary", "carbonate", "crystalline", "volcanic", "unconsolidated"][self as usize]
    }
}

/// Culture archetype (drawn per culture area from the climate at its site).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Archetype {
    #[default]
    TropicalSmallholder = 0,
    TropicalPlantation = 1,
    SavannaPastoral = 2,
    DesertOasis = 3,
    SteppeNomadic = 4,
    Mediterranean = 5,
    TemperateVillage = 6,
    SurveyGrid = 7,
    MonsoonPaddy = 8,
    Boreal = 9,
    Arctic = 10,
    Highland = 11,
}

impl Archetype {
    pub const ALL: [Archetype; 12] = [
        Archetype::TropicalSmallholder,
        Archetype::TropicalPlantation,
        Archetype::SavannaPastoral,
        Archetype::DesertOasis,
        Archetype::SteppeNomadic,
        Archetype::Mediterranean,
        Archetype::TemperateVillage,
        Archetype::SurveyGrid,
        Archetype::MonsoonPaddy,
        Archetype::Boreal,
        Archetype::Arctic,
        Archetype::Highland,
    ];
    pub fn from_bits(b: u32) -> Archetype {
        Self::ALL.get(b as usize).copied().unwrap_or_default()
    }
    pub fn name(self) -> &'static str {
        [
            "tropical_smallholder",
            "tropical_plantation",
            "savanna_pastoral",
            "desert_oasis",
            "steppe_nomadic",
            "mediterranean",
            "temperate_village",
            "survey_grid",
            "monsoon_paddy",
            "boreal",
            "arctic",
            "highland",
        ][self as usize]
    }
}

/// The atlas fields at a point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AtlasSample {
    /// Smooth elevation (m), band-limited at the texel size (~40 km features).
    pub elevation_m: f64,
    /// Signed distance to the coast (km): > 0 on land, < 0 at sea.
    pub coast_km: f64,
    /// Annual-mean prevailing surface wind (m/s), components toward east and north.
    pub wind_e: f64,
    pub wind_n: f64,
    /// Annual precipitation (mm/year).
    pub precip_mm: f64,
    /// Annual mean temperature at sea level (°C); subtract the lapse rate for the elevation.
    pub temp_c: f64,
    /// Seasonality: warmest minus coldest monthly mean temperature (°C).
    pub temp_range_c: f64,
    /// Precipitation regime: (summer − winter) / (summer + winter) of the half-year
    /// precipitation: −1 dry summers (Mediterranean) … +1 summer rains (monsoon).
    pub regime: f64,
    /// Distance to the nearest plate boundary (km).
    pub plate_dist_km: f64,
    /// Closing speed of the nearest plate boundary (mm/year; < 0: diverging).
    pub convergence_mm_yr: f64,
    /// Tectonic relief potential −1..1 from the boundary type, distance and speed: + mountain
    /// belts, arcs, rift shoulders, swells; − trenches and rift grabens.
    pub uplift: f64,
    /// Volcanism 0..1: volcanic arcs, rifts, ridges and hotspot tracks.
    pub volcanism: f64,
    /// Glaciation index 0..1 (ice at the last glacial maximum: fjords, drumlins, cirques …).
    pub glaciation: f64,
    /// Development index 0..1 (per culture, ± regional noise).
    pub development: f64,
    /// Population potential 0..1 (flatness × water × climate comfort × fertility).
    pub population: f64,
    /// Plate id (nearest texel).
    pub plate: u32,
    /// Kind of the nearest plate boundary, seen from this plate.
    pub boundary: Boundary,
    /// Continental crust (else oceanic).
    pub continental: bool,
    /// On a hotspot track.
    pub hotspot: bool,
    /// Dominant and secondary lithology, and the secondary's share (0..0.5).
    pub litho: Lithology,
    pub litho2: Lithology,
    pub litho2_frac: f64,
    /// Culture area id (dense, 0..`Atlas::cultures().len()`) and its archetype.
    pub culture: u32,
    pub archetype: Archetype,
}

impl AtlasSample {
    /// Decode the discrete slots.
    fn set_discrete(&mut self, plate: u32, litho: u32, culture: u32) {
        self.plate = plate & 63;
        self.boundary = Boundary::from_bits(plate >> 6);
        self.continental = plate & (1 << 9) != 0;
        self.hotspot = plate & (1 << 10) != 0;
        self.litho = Lithology::from_bits(litho);
        self.litho2 = Lithology::from_bits(litho >> 3);
        self.litho2_frac = ((litho >> 8) & 255) as f64 / 510.0;
        self.culture = culture & 0xFFF;
        self.archetype = Archetype::from_bits(culture >> 12);
    }
    fn set_continuous(&mut self, v: &[f64; slot::CONTINUOUS]) {
        self.elevation_m = v[slot::ELEVATION];
        self.coast_km = v[slot::COAST];
        self.wind_e = v[slot::WIND_E];
        self.wind_n = v[slot::WIND_N];
        self.precip_mm = v[slot::PRECIP];
        self.temp_c = v[slot::TEMP];
        self.temp_range_c = v[slot::TEMP_RANGE];
        self.regime = v[slot::REGIME];
        self.plate_dist_km = v[slot::PLATE_DIST];
        self.convergence_mm_yr = v[slot::CONVERGENCE];
        self.uplift = v[slot::UPLIFT];
        self.volcanism = v[slot::VOLCANISM];
        self.glaciation = v[slot::GLACIATION];
        self.development = v[slot::DEVELOPMENT];
        self.population = v[slot::POPULATION];
    }
}

/// A culture area: its site and archetype.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Culture {
    /// Hash of the site's lattice cell (stable across atlas resolutions).
    pub hash: u64,
    /// Site direction (unit).
    pub site: DVec3,
    pub archetype: Archetype,
    /// Development of the culture (before the regional noise), 0..1.
    pub development: f64,
}

/// The planetary atlas of one world.
pub struct Atlas {
    /// header + texel data (+ the culture table): the GPU buffer
    words: Vec<u32>,
    cultures: Vec<Culture>,
    key: u64,
}

impl std::fmt::Debug for Atlas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Atlas {{ resolution: {}, cultures: {}, key: {:016x} }}", self.resolution(), self.cultures.len(), self.key)
    }
}

/// f32 → IEEE half (round to nearest even; saturating at ±65504; |x| < 2⁻¹⁴ and NaN → 0, so
/// no subnormals reach a GPU that may flush them).
pub fn f16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return 0;
    }
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a < 6.103_515_6e-5 {
        return 0;
    }
    if a >= 65504.0 {
        return sign | 0x7BFF;
    }
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7f_ffff;
    let mut h = ((e as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h.min(0x7BFF) as u16
}

/// IEEE half → f32 (exact).
pub fn f16_value(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    match e {
        0 => {
            let v = m as f32 * (1.0 / 16_777_216.0);
            if sign != 0 {
                -v
            } else {
                v
            }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (m << 13)),
        _ => f32::from_bits(sign | ((e + 112) << 23) | (m << 13)),
    }
}

/// Cubic B-spline weights at fraction `t`.
#[inline]
fn bspline(t: f64) -> [f64; 4] {
    let (t2, s) = (t * t, 1.0 - t);
    let t3 = t2 * t;
    [s * s * s / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0]
}

/// FNV-1a 64 of bytes.
fn fnv(bytes: impl IntoIterator<Item = u8>, h0: u64) -> u64 {
    bytes.into_iter().fold(h0, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

impl Atlas {
    /// The key of a world's atlas: a hash of the settings it depends on (seed, planet, home,
    /// continents, relief, climate, atlas) and [`ATLAS_VERSION`].
    pub fn key(cfg: &Config) -> u64 {
        #[derive(serde::Serialize)]
        struct Inputs<'a> {
            version: u32,
            seed: u64,
            planet: &'a crate::config::Planet,
            home: &'a Option<crate::config::Home>,
            continents: &'a crate::config::Continents,
            relief: &'a crate::config::Relief,
            climate: &'a crate::config::Climate,
            atlas: &'a crate::config::AtlasConfig,
        }
        let s = serde_yaml::to_string(&Inputs {
            version: ATLAS_VERSION,
            seed: cfg.seed,
            planet: &cfg.planet,
            home: &cfg.home,
            continents: &cfg.continents,
            relief: &cfg.relief,
            climate: &cfg.climate,
            atlas: &cfg.atlas,
        })
        .unwrap_or_default();
        fnv(s.bytes(), 0xcbf2_9ce4_8422_2325)
    }

    /// Compute the atlas of a world (no caches; ~1–2 s at the default resolution on 8 threads).
    /// Runs on the current rayon pool; the result does not depend on its size.
    pub fn build(world: &World) -> Atlas {
        let t = std::time::Instant::now();
        let a = build::build(world);
        if std::env::var_os("TERRAGEN_PROFILE").is_some() {
            eprintln!("atlas {}² × 6 built in {:.2} s", a.resolution(), t.elapsed().as_secs_f64());
        }
        a
    }

    /// The atlas of a world: from the process' cache (the last few worlds), else from the disk
    /// cache, else built (and stored on disk).
    pub fn for_world(world: &World) -> Arc<Atlas> {
        type Slot = Arc<OnceLock<Arc<Atlas>>>;
        static CACHE: Mutex<Vec<(u64, Slot)>> = Mutex::new(Vec::new());
        const KEEP: usize = 4;
        let key = Self::key(&world.cfg);
        let slot = {
            let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
            match c.iter().position(|(k, _)| *k == key) {
                Some(i) => {
                    let e = c.remove(i);
                    c.push(e.clone());
                    e.1
                }
                None => {
                    let s: Slot = Default::default();
                    c.push((key, s.clone()));
                    if c.len() > KEEP {
                        c.remove(0);
                    }
                    s
                }
            }
        };
        slot.get_or_init(|| {
            let path = cache_dir().map(|d| d.join(format!("atlas-{key:016x}.bin")));
            if let Some(a) = path.as_deref().and_then(|p| Atlas::load(p, key).ok()) {
                return Arc::new(a);
            }
            // built on a plain thread with a pool of its own: callers may be rayon workers
            // inside a `OnceLock` initialization (the GPU generator's); waiting on parallel work
            // there would let them steal jobs that re-enter that initialization (a deadlock),
            // and the global pool's workers may all be blocked on it
            let a = std::thread::scope(|s| {
                s.spawn(|| match rayon::ThreadPoolBuilder::new().num_threads(rayon::current_num_threads()).build() {
                    Ok(pool) => pool.install(|| Atlas::build(world)),
                    Err(_) => Atlas::build(world),
                })
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e))
            });
            if let Some(p) = &path {
                if let Err(e) = a.save(p) {
                    eprintln!("warning: could not cache the planetary atlas in {}: {e:#}", p.display());
                }
            }
            Arc::new(a)
        })
        .clone()
    }

    pub(crate) fn from_parts(words: Vec<u32>, cultures: Vec<Culture>, key: u64) -> Atlas {
        Atlas { words, cultures, key }
    }

    /// Texels per face edge.
    pub fn resolution(&self) -> usize {
        self.words[2] as usize
    }
    /// [`Atlas::key`] of the world it was built for.
    pub fn cache_key(&self) -> u64 {
        self.key
    }
    /// The buffer the GPU binds: header (16 words: magic, version, resolution, apron, words per
    /// texel, offset of the texel data, number of cultures, offset of the culture table), the
    /// padded faces (6 × (r + 4)² texels of 9 words), the culture table (4 words per culture:
    /// site hash low, high, archetype, development as f32 bits).
    pub fn words(&self) -> &[u32] {
        &self.words
    }
    /// The culture areas (index = [`AtlasSample::culture`]).
    pub fn cultures(&self) -> &[Culture] {
        &self.cultures
    }

    #[inline]
    fn texel(&self, f: usize, i: usize, j: usize) -> usize {
        let p = self.resolution() + 2 * APRON;
        HEADER + ((f * p + j) * p + i) * WORDS
    }

    /// The 16-bit slot `s` of padded texel (f, i, j) (apron included: i, j in 0..r + 4).
    pub fn raw(&self, f: usize, i: usize, j: usize, s: usize) -> u16 {
        let w = self.words[self.texel(f, i, j) + s / 2];
        if s.is_multiple_of(2) {
            w as u16
        } else {
            (w >> 16) as u16
        }
    }

    /// The fields at a direction (any length; ECEF directions are fine): continuous fields by a
    /// cubic B-spline over 4 × 4 texels, discrete ones from the nearest texel.
    ///
    /// Within half a texel of a face edge, the B-splines of the faces meeting there (each
    /// continued over the edge by its apron) are blended with weights that are ½ on the edge
    /// (¼ at a corner, normalized): the result is the same from either side, a continuous (C¹)
    /// function of the direction with no seams.
    pub fn sample(&self, dir: DVec3) -> AtlasSample {
        let r = self.resolution();
        let rf = r as f64;
        let g = Grid::new(r);
        let ramp = |x: f64| {
            let (lo, hi) = ((x + 1.0).clamp(0.0, 1.0), (rf - x).clamp(0.0, 1.0));
            lo * lo * (3.0 - 2.0 * lo) * hi * hi * (3.0 - 2.0 * hi)
        };
        let mut acc = [0.0f64; 16];
        let mut wsum = 0.0;
        for axis in 0..3 {
            let c = dir[axis];
            if c == 0.0 {
                continue;
            }
            let f = 2 * axis + (c < 0.0) as usize;
            let [n, u, v] = geom::FACES[f];
            let m = dir.dot(n);
            let (x, y) = (g.x_of((dir.dot(u) / m).atan() / std::f64::consts::FRAC_PI_4), g.x_of((dir.dot(v) / m).atan() / std::f64::consts::FRAC_PI_4));
            let w = ramp(x) * ramp(y);
            if w <= 0.0 {
                continue;
            }
            self.accumulate(f, x, y, w, &mut acc);
            wsum += w;
        }
        let mut s = AtlasSample::default();
        let mut c = [0.0; slot::CONTINUOUS];
        for (ci, a) in c.iter_mut().zip(acc) {
            *ci = a / wsum;
        }
        s.set_continuous(&c);
        let (f, x, y) = g.locate(dir);
        let near = |x: f64| ((x.clamp(-0.5, rf - 0.5) + 0.5).floor() as usize).min(r - 1);
        let t = self.texel(f, near(x) + APRON, near(y) + APRON);
        s.set_discrete(self.words[t + 7] >> 16, self.words[t + 8] & 0xFFFF, self.words[t + 8] >> 16);
        s
    }

    /// Add `w` × the B-spline of face `f` at texel coordinates (x, y) ∈ (−1, r)² to `acc`.
    fn accumulate(&self, f: usize, x: f64, y: f64, w: f64, acc: &mut [f64; 16]) {
        let (cx, cy) = (x.floor(), y.floor());
        let (wx, wy) = (bspline(x - cx), bspline(y - cy));
        // first tap, in padded indices (cx − 1 + APRON >= 0)
        let (i0, j0) = ((cx as i64 + 1) as usize, (cy as i64 + 1) as usize);
        for (b, wyb) in wy.iter().enumerate() {
            for (a, wxa) in wx.iter().enumerate() {
                let wt = w * wxa * wyb;
                let t = self.texel(f, i0 + a, j0 + b);
                for k in 0..8 {
                    let v = self.words[t + k];
                    acc[2 * k] += wt * f16_value(v as u16) as f64;
                    acc[2 * k + 1] += wt * f16_value((v >> 16) as u16) as f64;
                }
            }
        }
    }

    /// Distance (texels) from a direction to the nearest boundary between texels (where the
    /// discrete fields switch): GPU and CPU may disagree on the discrete fields closer than f32
    /// rounding.
    pub fn texel_edge_distance(&self, dir: DVec3) -> f64 {
        let r = self.resolution() as f64;
        let (_, x, y) = Grid::new(self.resolution()).locate(dir);
        let e = |x: f64| {
            let x = x.clamp(-0.5, r - 0.5);
            let fr = x + 0.5 - (x + 0.5).floor();
            fr.min(1.0 - fr)
        };
        e(x).min(e(y))
    }

    /// Write the atlas to `path` (via a temporary file and a rename: safe with concurrent
    /// processes writing the same atlas).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let mut bytes = Vec::with_capacity(32 + self.words.len() * 4 + self.cultures.len() * 40);
        bytes.extend_from_slice(FILE_MAGIC);
        bytes.extend_from_slice(&self.key.to_le_bytes());
        bytes.extend_from_slice(&(self.words.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.cultures.len() as u64).to_le_bytes());
        for w in &self.words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        for c in &self.cultures {
            bytes.extend_from_slice(&c.hash.to_le_bytes());
            for v in [c.site.x, c.site.y, c.site.z, c.development] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.push(c.archetype as u8);
        }
        let sum = fnv(bytes.iter().copied(), 0xcbf2_9ce4_8422_2325);
        bytes.extend_from_slice(&sum.to_le_bytes());
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let tmp = path.with_extension(format!("tmp{}-{}", std::process::id(), SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
        if let Err(e) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e).with_context(|| format!("renaming to {}", path.display()));
        }
        Ok(())
    }

    /// Read an atlas written by [`Atlas::save`]; fails unless it is intact, of this format and
    /// built for `key`.
    pub fn load(path: &Path, key: u64) -> Result<Atlas> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let n = bytes.len();
        if n < 40 || &bytes[..8] != FILE_MAGIC {
            bail!("{}: not an atlas file of this version", path.display());
        }
        let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        if u64_at(n - 8) != fnv(bytes[..n - 8].iter().copied(), 0xcbf2_9ce4_8422_2325) {
            bail!("{}: checksum mismatch", path.display());
        }
        if u64_at(8) != key {
            bail!("{}: built for another world", path.display());
        }
        let (nw, nc) = (u64_at(16) as usize, u64_at(24) as usize);
        if 32 + nw * 4 + nc * 41 + 8 != n || nw < HEADER {
            bail!("{}: truncated", path.display());
        }
        let words: Vec<u32> = bytes[32..32 + nw * 4].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        let f64_at = |o: usize| f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let cultures = (0..nc)
            .map(|i| {
                let o = 32 + nw * 4 + i * 41;
                Culture {
                    hash: u64_at(o),
                    site: DVec3::new(f64_at(o + 8), f64_at(o + 16), f64_at(o + 24)),
                    development: f64_at(o + 32),
                    archetype: Archetype::from_bits(bytes[o + 40] as u32),
                }
            })
            .collect();
        let r = words[2] as usize;
        if words[0] != MAGIC || words[1] != ATLAS_VERSION || words[3] as usize != APRON || words[4] as usize != WORDS || words[5] as usize != HEADER {
            bail!("{}: unexpected header", path.display());
        }
        if HEADER + 6 * (r + 2 * APRON).pow(2) * WORDS + 4 * words[6] as usize != nw {
            bail!("{}: size does not match its resolution", path.display());
        }
        Ok(Atlas { words, cultures, key })
    }
}

const FILE_MAGIC: &[u8; 8] = b"TGATLAS1";

/// The directory of the disk cache: `TERRAGEN_ATLAS_CACHE` (a directory; `off`, `0` or empty
/// disable the cache), else `$XDG_CACHE_HOME/terrain`, else `~/.cache/terrain`.
pub fn cache_dir() -> Option<PathBuf> {
    if let Some(v) = std::env::var_os("TERRAGEN_ATLAS_CACHE") {
        let s = v.to_string_lossy();
        if s.is_empty() || s == "off" || s == "0" {
            return None;
        }
        return Some(PathBuf::from(v));
    }
    std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".cache"))).map(|d| d.join("terrain"))
}
