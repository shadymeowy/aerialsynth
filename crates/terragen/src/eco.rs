//! Ecoregions and cultures (`docs/design/terrain-next.md` §3.8): the main source of regional
//! variety.
//!
//! * **Ecoregions:** a warped 3D jittered lattice (~100 km, `world.ecoregions.cell_km`). Each
//!   site picks a biome from the registry by the climate at the site (Köppen class, envelope,
//!   lithology, weights × its culture's preferences) and draws a style: soil and rock colours
//!   by lithology, grass and crown tints, tree density and species bias, season offset.
//! * **Cultures:** areas of ~1000 km (the atlas' culture field; the stand-in's lattice until the
//!   atlas lands), each with one of 12 archetypes. An archetype is a *distribution*: every
//!   ecoregion draws its field systems, field size, roof palette, building height, block size,
//!   lamp mix, agriculture and town density from its culture's archetype. Cultures are assigned
//!   per ecoregion (from its site), so culture borders follow ecoregion borders.
//! * **Ecotones:** within `ecotone_km` of a border each ~220 m patch belongs to one side or the
//!   other, with a probability by the distance (a mosaic, not a gradient); where the patches are
//!   below the pixel the two sides blend with exactly those probabilities.
//!
//! Ecoregion parameters are computed here, on the host, for both backends (a CPU tile computes
//! them per id on demand; the GPU generator for the ids its tiles report, in the same round as
//! the land-use regions): every discrete choice (biome, archetype, field systems) is the same
//! on both. Results are cached; they are pure functions of (world, id).

use crate::atlas;
use crate::noise::*;
use crate::registry::{Registry, SiteClimate};
use crate::world::World;
use glam::DVec3;

const KM: f64 = 1000.0;

/// Lattice key of the ecoregion lattice.
pub const ECO_KEY: u64 = 0xEC0E;

/// The style an ecoregion draws (blended across ecotones where unresolved).
#[derive(Clone, Copy, Debug)]
pub struct EcoStyle {
    /// soil and rock colours of the lithology (linear) and how strongly they replace the
    /// biome's
    pub soil: DVec3,
    pub soil_w: f64,
    pub rock: DVec3,
    pub rock_w: f64,
    /// multiplicative tints of grass and crowns
    pub grass: DVec3,
    pub crown: DVec3,
    /// tree density multiplier, conifer bias (−1..1), shrub multiplier, woodlot share multiplier
    pub trees: f64,
    pub conifer: f64,
    pub shrubs: f64,
    pub woodlots: f64,
    /// crop calendar 0 spring .. 1 autumn
    pub season: f64,
    /// agriculture and town multipliers
    pub agri: f64,
    pub towns: f64,
    /// field-system weights: grid, irregular, pivots, strips
    pub fields: [f64; 4],
    /// field size multiplier
    pub field_scale: f64,
    /// share of fields bordered by hedges
    pub hedges: f64,
    /// roof palette offset 0..1, building height, block size multiplier, share of sodium lamps
    pub roof: f64,
    pub height: f64,
    pub block: f64,
    pub sodium: f64,
    pub development: f64,
    pub archetype: u8,
    pub litho: u8,
}

impl Default for EcoStyle {
    fn default() -> Self {
        EcoStyle {
            soil: DVec3::splat(0.2),
            soil_w: 0.0,
            rock: DVec3::splat(0.2),
            rock_w: 0.0,
            grass: DVec3::ONE,
            crown: DVec3::ONE,
            trees: 1.0,
            conifer: 0.0,
            shrubs: 1.0,
            woodlots: 1.0,
            season: 0.5,
            agri: 1.0,
            towns: 1.0,
            fields: [0.5, 0.38, 0.0, 0.12],
            field_scale: 1.0,
            hedges: 0.4,
            roof: 0.5,
            height: 0.5,
            block: 1.0,
            sodium: 0.45,
            development: 0.5,
            archetype: atlas::Archetype::TemperateVillage as u8,
            litho: 0,
        }
    }
}

impl EcoStyle {
    /// `self` blended with `o` by `w` (continuous parameters; discrete ones from `self`).
    pub fn mix(&self, o: &EcoStyle, w: f64) -> EcoStyle {
        let m = |a: f64, b: f64| a + (b - a) * w;
        let mv = |a: DVec3, b: DVec3| a + (b - a) * w;
        EcoStyle {
            soil: mv(self.soil, o.soil),
            soil_w: m(self.soil_w, o.soil_w),
            rock: mv(self.rock, o.rock),
            rock_w: m(self.rock_w, o.rock_w),
            grass: mv(self.grass, o.grass),
            crown: mv(self.crown, o.crown),
            trees: m(self.trees, o.trees),
            conifer: m(self.conifer, o.conifer),
            shrubs: m(self.shrubs, o.shrubs),
            woodlots: m(self.woodlots, o.woodlots),
            season: m(self.season, o.season),
            agri: m(self.agri, o.agri),
            towns: m(self.towns, o.towns),
            fields: std::array::from_fn(|i| m(self.fields[i], o.fields[i])),
            field_scale: m(self.field_scale, o.field_scale),
            hedges: m(self.hedges, o.hedges),
            roof: m(self.roof, o.roof),
            height: m(self.height, o.height),
            block: m(self.block, o.block),
            sodium: m(self.sodium, o.sodium),
            development: m(self.development, o.development),
            ..*self
        }
    }
}

/// An ecoregion: its site, climate, biome and style.
#[derive(Clone, Copy, Debug)]
pub struct EcoParams {
    pub id: u64,
    /// the surface point below the site
    pub site: DVec3,
    pub climate: SiteClimate,
    pub biome: u16,
    pub culture: u32,
    pub style: EcoStyle,
    /// the agriculture kit's per-system weight multipliers (0.4..1.6) and its draw of the
    /// dominant system (`[15]`), varying smoothly across the ecoregions (`coherent_u`)
    pub agri_w: [f64; 16],
    /// the expected fields of its land-use regions (`kits::agriculture::expected_mean`): mean
    /// colour (linear) and cover share
    pub agri_mean: [f64; 4],
}

/// Culture archetypes as distributions of parameters.
struct Arche {
    /// field-system weights (grid, irregular, pivots, strips)
    fields: [f64; 4],
    field_scale: [f64; 2],
    hedges: [f64; 2],
    roof: [f64; 2],
    height: [f64; 2],
    block: [f64; 2],
    sodium: [f64; 2],
    agri: [f64; 2],
    towns: [f64; 2],
    trees: [f64; 2],
    season: [f64; 2],
}

#[rustfmt::skip]
const ARCHES: [Arche; 12] = [
    // tropical smallholder: small irregular plots, corrugated / tile roofs, low
    Arche { fields: [0.15, 0.7, 0.0, 0.15], field_scale: [0.45, 0.7], hedges: [0.0, 0.2], roof: [0.3, 0.7], height: [0.0, 0.25], block: [0.7, 0.9], sodium: [0.5, 0.8], agri: [0.9, 1.2], towns: [0.9, 1.3], trees: [0.9, 1.1], season: [0.0, 0.4] },
    // tropical plantation: big regular blocks
    Arche { fields: [0.75, 0.25, 0.0, 0.0], field_scale: [1.3, 2.0], hedges: [0.0, 0.1], roof: [0.4, 0.8], height: [0.1, 0.4], block: [0.9, 1.2], sodium: [0.4, 0.7], agri: [1.0, 1.3], towns: [0.7, 1.0], trees: [0.8, 1.0], season: [0.0, 0.3] },
    // savanna pastoral: few fields, kraals, dispersed
    Arche { fields: [0.15, 0.65, 0.2, 0.0], field_scale: [0.6, 1.0], hedges: [0.0, 0.2], roof: [0.2, 0.6], height: [0.0, 0.2], block: [0.8, 1.1], sodium: [0.5, 0.9], agri: [0.4, 0.7], towns: [0.5, 0.9], trees: [0.9, 1.2], season: [0.3, 0.8] },
    // desert oasis: irrigated circles and grids, flat roofs
    Arche { fields: [0.3, 0.1, 0.6, 0.0], field_scale: [0.8, 1.3], hedges: [0.0, 0.1], roof: [0.55, 0.75], height: [0.1, 0.35], block: [0.8, 1.0], sodium: [0.6, 0.9], agri: [0.6, 0.9], towns: [0.7, 1.0], trees: [0.8, 1.0], season: [0.4, 0.9] },
    // steppe nomadic: huge fields and strips, sparse towns
    Arche { fields: [0.45, 0.05, 0.15, 0.35], field_scale: [1.6, 2.4], hedges: [0.3, 0.7], roof: [0.4, 0.8], height: [0.2, 0.5], block: [1.0, 1.3], sodium: [0.7, 0.95], agri: [0.8, 1.1], towns: [0.5, 0.8], trees: [0.7, 1.0], season: [0.4, 0.8] },
    // mediterranean: small irregular fields, terracotta roofs, dense towns
    Arche { fields: [0.15, 0.65, 0.05, 0.15], field_scale: [0.5, 0.8], hedges: [0.1, 0.4], roof: [0.0, 0.15], height: [0.2, 0.5], block: [0.7, 0.9], sodium: [0.5, 0.8], agri: [0.9, 1.1], towns: [1.1, 1.4], trees: [0.7, 0.95], season: [0.5, 0.9] },
    // temperate village: mixed fields, hedgerows, varied roofs
    Arche { fields: [0.25, 0.45, 0.0, 0.3], field_scale: [0.8, 1.1], hedges: [0.3, 0.8], roof: [0.0, 1.0], height: [0.2, 0.6], block: [0.9, 1.1], sodium: [0.3, 0.6], agri: [0.9, 1.1], towns: [0.9, 1.2], trees: [0.9, 1.15], season: [0.2, 0.7] },
    // survey grid: big square fields, grey roofs, wide blocks
    Arche { fields: [0.92, 0.0, 0.08, 0.0], field_scale: [1.8, 2.6], hedges: [0.0, 0.3], roof: [0.3, 0.6], height: [0.1, 0.5], block: [1.2, 1.5], sodium: [0.6, 0.9], agri: [1.0, 1.25], towns: [0.7, 1.0], trees: [0.8, 1.0], season: [0.3, 0.8] },
    // monsoon paddy: small plots, dense villages
    Arche { fields: [0.45, 0.5, 0.0, 0.05], field_scale: [0.4, 0.65], hedges: [0.0, 0.15], roof: [0.3, 0.8], height: [0.0, 0.3], block: [0.6, 0.85], sodium: [0.4, 0.7], agri: [1.1, 1.3], towns: [1.1, 1.5], trees: [0.85, 1.05], season: [0.0, 0.4] },
    // boreal: clearings in the forest, few towns
    Arche { fields: [0.4, 0.55, 0.0, 0.05], field_scale: [0.6, 1.0], hedges: [0.0, 0.3], roof: [0.4, 1.0], height: [0.1, 0.4], block: [1.0, 1.3], sodium: [0.3, 0.7], agri: [0.4, 0.7], towns: [0.5, 0.8], trees: [1.0, 1.3], season: [0.3, 0.7] },
    // arctic: almost nothing
    Arche { fields: [0.5, 0.5, 0.0, 0.0], field_scale: [0.6, 1.0], hedges: [0.0, 0.1], roof: [0.5, 1.0], height: [0.0, 0.2], block: [1.0, 1.3], sodium: [0.3, 0.6], agri: [0.1, 0.3], towns: [0.2, 0.4], trees: [0.9, 1.1], season: [0.3, 0.6] },
    // highland: terraced strips and small irregular fields
    Arche { fields: [0.05, 0.55, 0.0, 0.4], field_scale: [0.4, 0.7], hedges: [0.2, 0.6], roof: [0.0, 0.6], height: [0.0, 0.3], block: [0.6, 0.9], sodium: [0.4, 0.8], agri: [0.7, 1.0], towns: [0.7, 1.0], trees: [0.8, 1.1], season: [0.3, 0.7] },
];

/// Soil and rock colours (sRGB) by lithology (two variants each).
const LITHO_COL: [[[f64; 3]; 4]; 5] = [
    // sedimentary: brown loams; buff / red sandstone
    [[150.0, 118.0, 84.0], [138.0, 98.0, 70.0], [172.0, 146.0, 112.0], [168.0, 112.0, 84.0]],
    // carbonate: pale rendzina; white-grey limestone
    [[170.0, 152.0, 124.0], [184.0, 170.0, 146.0], [196.0, 190.0, 176.0], [182.0, 176.0, 160.0]],
    // crystalline: grey-brown podzolic; grey / pink granite
    [[122.0, 108.0, 92.0], [134.0, 114.0, 96.0], [126.0, 124.0, 120.0], [156.0, 132.0, 120.0]],
    // volcanic: dark andosols; basalt
    [[84.0, 72.0, 62.0], [96.0, 78.0, 64.0], [74.0, 72.0, 70.0], [92.0, 84.0, 80.0]],
    // unconsolidated: loess / alluvium; gravel
    [[160.0, 136.0, 96.0], [146.0, 124.0, 92.0], [158.0, 148.0, 130.0], [150.0, 140.0, 122.0]],
];

/// The ecoregion lattice settings.
#[derive(Clone, Copy, Debug)]
pub struct EcoCfg {
    pub cell: f64,
    pub ecotone: f64,
}

impl EcoCfg {
    pub fn of(w: &World) -> EcoCfg {
        EcoCfg { cell: w.cfg.ecoregions.cell_km * KM, ecotone: w.cfg.ecoregions.ecotone_km * KM }
    }
}

/// Warp (m) of the lookup in the ecoregion lattice: curvy, irregular borders.
pub fn warp(w: &World, p: DVec3) -> DVec3 {
    let c = w.cfg.ecoregions.cell_km * KM;
    let s = w.seed();
    let a = |k: u64, lam: f64| perlin3(s ^ k, p / lam);
    DVec3::new(a(0xEC1, 0.8 * c), a(0xEC2, 0.8 * c), a(0xEC3, 0.8 * c)) * (0.22 * c)
        + DVec3::new(a(0xEC4, 0.17 * c), a(0xEC5, 0.17 * c), a(0xEC6, 0.17 * c)) * (0.035 * c)
}

/// The shortest wavelength of [`warp`] (m): it is interpolated from the tile grid when at least
/// the grid's cut.
pub fn warp_min_wavelength(w: &World) -> f64 {
    0.17 * w.cfg.ecoregions.cell_km * KM
}

/// The width of the wide blend in differences of site distance, per cell size: the
/// ecoregions' styles, palettes and vegetation blend over ~¼ cell on each side of a border
/// (`d₂ − d₁` is about twice the distance to the border).
pub const BLEND_CELLS: f64 = 0.5;

/// Weights (sum 1) of the three nearest ecoregion sites at distances `d1 <= d2 <= d3` (m, in the
/// warped lattice): ½ / ½ on a border, the nearest alone from ~¼ cell inside it, continuous
/// everywhere (where the second nearest changes, the third's weight is the same). Their
/// continuous parameters (styles, palettes, vegetation densities, site climate) blend by them
/// at every zoom (round 16: the 10 km ecotones left each ecoregion a polygon of its own tone,
/// a camouflage seen from orbit); the crowns, landforms and kernel layers stay the ecotone
/// mosaic's ([`ecotone_pick`]).
pub fn wide_weights(w: &World, d1: f64, d2: f64, d3: f64) -> [f64; 3] {
    let wd = BLEND_CELLS * w.cfg.ecoregions.cell_km * KM;
    if wd <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let phi = |x: f64| 1.0 - smoothstep(0.0, wd, x);
    let (w2, w3) = (phi(d2 - d1), phi(d3 - d1));
    let s = 1.0 + w2 + w3;
    [1.0 / s, w2 / s, w3 / s]
}

/// Probability (0..½) of the ecoregion across the border at `edge` m from it: ½ at the border,
/// 0 at the ecotone's edge (the mean of [`ecotone_pick`]; the ecoregions' styles blend by it).
pub fn ecotone_prob(w: &World, gsd: f64, edge: f64) -> f64 {
    let wd = w.cfg.ecoregions.ecotone_km * KM;
    let wd_eff = wd.max(1.5 * gsd);
    if edge >= wd_eff || wd_eff <= 0.0 {
        return 0.0;
    }
    0.5 * (1.0 - smoothstep(0.0, wd_eff, edge))
}

/// Weight (0..1) of the ecoregion across the border at `edge` m from it: within the ecotone a
/// crisp mosaic of ragged ~220 m patches (0 or 1), blended to the probability where the patches are
/// below the pixel.
pub fn ecotone_pick(w: &World, p: DVec3, gsd: f64, edge: f64) -> f64 {
    let pb = ecotone_prob(w, gsd, edge);
    if pb <= 0.0 {
        return 0.0;
    }
    let ex = band(220.0, gsd);
    if ex <= 0.0 {
        return pb;
    }
    let s = w.seed();
    // (ragged patches: a strong 80 m octave; with a weak one the patches of a strongly
    // different neighbour read as smooth, hard-edged discs. A blended edge was tried: it broke
    // the LOD of the DSM, crowns and vegetation of two biomes do not blend linearly)
    let b80 = band(80.0, gsd);
    let n = 0.75 * perlin3(s ^ 0xEC7, p / 220.0) + 0.35 * perlin3(s ^ 0xEC8, p / 80.0) * b80;
    // ~uniform 0..1 (logistic of the noise over its spread)
    // (normalized by the spread of the octaves present: u stays uniform, so the patches cover
    // the probability at every pixel size and a parent matches its children)
    let u = 1.0 / (1.0 + (-1.7 * n / (0.3504 * (0.5625 + 0.1225 * b80 * b80).sqrt())).exp());
    let crisp = if u < pb { 1.0 } else { 0.0 };
    pb + (crisp - pb) * ex
}

/// Shared ecoregion parameters (all threads; per-thread caches in front, `surface::Caches`).
#[derive(Default)]
pub struct Ecoregions {
    shared: std::sync::RwLock<FxHashMap<u64, EcoParams>>,
}

impl Ecoregions {
    /// The parameters of ecoregion `id` whose lattice site is `site` (cached).
    pub fn params(&self, w: &World, reg: &Registry, cache: &mut crate::surface::Caches, id: u64, site: DVec3) -> EcoParams {
        if let Some(e) = cache.eco.get(&id) {
            return *e;
        }
        if let Some(e) = self.shared.read().unwrap().get(&id).copied() {
            cache.eco.insert(id, e);
            return e;
        }
        let e = compute(w, reg, id, site);
        cache.eco.insert(id, e);
        let mut g = self.shared.write().unwrap();
        if g.len() > 200_000 {
            g.clear();
        }
        g.insert(id, e);
        e
    }
}

/// Months (0..12) with less than 60 mm of rain, from the annual sum and the regime.
pub fn dry_months(s: &atlas::AtlasSample) -> f64 {
    let pm = s.precip_mm.max(0.0) / 12.0;
    let a = s.regime.abs().min(1.0);
    // monthly rain p(m) = pm (1 + a cos θ): the share of months below 60 mm
    if pm * (1.0 + a) < 60.0 {
        return 12.0;
    }
    if pm * (1.0 - a) >= 60.0 {
        return 0.0;
    }
    let c = ((60.0 / pm - 1.0) / a.max(1e-6)).clamp(-1.0, 1.0);
    12.0 * (1.0 - c.acos() / std::f64::consts::PI)
}

/// An ecoregion's parameters (pure function of the world and the site): the atlas at the site
/// (Köppen class at its smooth elevation, precipitation, dry months, lithology, culture).
pub fn compute(w: &World, reg: &Registry, id: u64, site: DVec3) -> EcoParams {
    let dir = site.normalize_or(DVec3::X);
    let s = w.atlas().sample(dir);
    let g = geodesy::ecef2geodetic(dir * w.ell.a, &w.ell);
    let surface = geodesy::geodetic2ecef(geodesy::Geodetic::new(g.lat, g.lon, 0.0), &w.ell);
    let elev = s.elevation_m.max(0.0);
    let lapse = w.cfg.climate.lapse_rate_c_per_km;
    let k = atlas::koppen_with_lapse(&s, elev, lapse);
    let temp = s.temp_c - lapse * elev / KM;
    let dry = dry_months(&s);
    let climate = SiteClimate { temp_c: temp, precip_mm: s.precip_mm, temp_range_c: s.temp_range_c, dry_months: dry, koppen: k as u32 };
    let arch = s.archetype as usize;
    // cold steppes are open taiga (Köppen's dry threshold makes the larch and spruce forests of
    // cold continental interiors, Yakutia, interior Alaska, "arid"): their biome is the
    // subarctic forest's
    let kb = if matches!(k, atlas::Koppen::BSk | atlas::Koppen::BWk) && temp < 3.0 && s.precip_mm > 150.0 { atlas::Koppen::Dfc } else { k };
    // (the ecoregion's own draw, as before round 16: a smooth draw across the ecoregions
    // made neighbours in a similar climate alike, but changed the biome of two thirds of the
    // gallery's places; the styles and the field systems' weights are the smooth draws)
    let biome = reg.pick(kb as u32, temp, s.precip_mm, dry, s.regime, s.temp_range_c, s.litho as u8, u01k(id, 0xB10), |_| 1.0);
    let style = draw_style(w.seed(), id, site, &s, arch, temp);
    // (the ecoregion's own draws, as in v0.3.0: the field systems a place shows)
    let agri_w = std::array::from_fn(|i| if i < 15 { 0.4 + 1.2 * u01k(id, 0xA600 + i as u64) } else { u01k(id, 0xA6FF) });
    let mut e = EcoParams { id, site: surface, climate, biome, culture: s.culture, style, agri_w, agri_mean: [0.0; 4] };
    e.agri_mean = crate::kits::agriculture::expected_mean(w, reg, &e);
    e
}

/// A draw 0..1 (about uniform) of the ecoregion at `site` that varies smoothly across the
/// ecoregions (noise of the site's position over ~600 and ~250 km), with a share `own` of the
/// ecoregion's own hash: neighbouring ecoregions draw alike, so their biomes and styles do not
/// jump from cell to cell (round 16: a camouflage of polygonal cells seen from orbit).
pub fn coherent_u(seed: u64, site: DVec3, key: u64, own: f64, id: u64) -> f64 {
    let n = coherent_n(seed, site, key);
    // uniform by the noise's own distribution (its spread is narrower and its tails lighter than
    // a normal's: a logistic map drew the draws to the middle, so the biomes of small weight
    // lost to the large ones everywhere and vanished)
    let cdf = noise_cdf();
    let k = cdf.partition_point(|&x| x < n);
    let u = if k == 0 {
        0.5 / cdf.len() as f64
    } else if k >= cdf.len() {
        1.0 - 0.5 / cdf.len() as f64
    } else {
        let (a, b) = (cdf[k - 1], cdf[k]);
        (k as f64 - 0.5 + if b > a { (n - a) / (b - a) } else { 0.5 }) / cdf.len() as f64
    };
    (1.0 - own) * u + own * u01k(id, key)
}

/// The noise behind [`coherent_u`] (two octaves of the site's position).
fn coherent_n(seed: u64, site: DVec3, key: u64) -> f64 {
    perlin3(seed ^ 0xC0E0 ^ key.rotate_left(17), site / 600_000.0) + 0.35 * perlin3(seed ^ 0xC0E1 ^ key.rotate_left(23), site / 250_000.0)
}

/// The sorted values of [`coherent_n`] over 8192 fixed points of the globe and keys (its
/// distribution, the same for every seed).
fn noise_cdf() -> &'static [f64] {
    static CDF: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    CDF.get_or_init(|| {
        let mut v: Vec<f64> = (0..8192u64)
            .map(|i| {
                let h = mix64(0xCDF0 ^ i);
                let d = DVec3::new(u01k(h, 1) - 0.5, u01k(h, 2) - 0.5, u01k(h, 3) - 0.5).normalize_or(DVec3::X);
                coherent_n(0x05EE_DCDF, d * 6_371_000.0, h)
            })
            .collect();
        v.sort_by(|a, b| a.total_cmp(b));
        v
    })
}

/// The style of an ecoregion: lithology colours, tints and its culture's draws.
fn draw_style(seed: u64, id: u64, site: DVec3, s: &atlas::AtlasSample, arch: usize, temp: f64) -> EcoStyle {
    // colours: draws that vary smoothly across the ecoregions (round 16: drawn per ecoregion,
    // neighbouring soils and tints jumped from polygon to polygon seen from orbit); what the
    // land holds (tree density, agriculture, field systems, towns, season) stays the
    // ecoregion's own draw, as in v0.3.0 (smooth draws there turned farmland into forest at a
    // third of the A/B places; the wide blend softens their borders)
    let u = |k: u64| coherent_u(seed, site, 0x5700 + k, 0.15, id);
    let own = |k: u64| u01k(id, 0x5700 + k);
    let r = |k: u64, a: [f64; 2]| a[0] + (a[1] - a[0]) * own(k);
    let lit = s.litho as usize;
    let lc = &LITHO_COL[lit];
    let pick = |a: [f64; 3], b: [f64; 3], t: f64| crate::surface::srgb(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t);
    let mut soil = pick(lc[0], lc[1], u(1));
    // black steppe soils (chernozem) in temperate grasslands on soft rock
    let steppe = smoothstep(250.0, 400.0, s.precip_mm)
        * (1.0 - smoothstep(550.0, 800.0, s.precip_mm))
        * smoothstep(0.0, 6.0, temp)
        * (1.0 - smoothstep(14.0, 20.0, temp));
    if steppe > 0.3 && lit != 2 && u(9) < 0.7 {
        soil = soil + (crate::surface::srgb(76.0, 62.0, 50.0) - soil) * steppe;
    }
    let rock = pick(lc[2], lc[3], u(2));
    let ar = &ARCHES[arch.min(11)];
    let tint = |k: u64, amp: f64| DVec3::new(1.0 + amp * (u(k) - 0.5), 1.0 + 0.6 * amp * (u(k + 1) - 0.5), 1.0 + amp * (u(k + 2) - 0.5));
    // field-system weights: the archetype's, perturbed per ecoregion (each region of it then
    // draws by them)
    let mut fields = ar.fields;
    for (i, f) in fields.iter_mut().enumerate() {
        *f *= 0.6 + 0.8 * own(20 + i as u64);
    }
    let ft: f64 = fields.iter().sum();
    let fields = if ft > 0.0 { fields.map(|f| f / ft) } else { [0.5, 0.38, 0.0, 0.12] };
    EcoStyle {
        soil,
        soil_w: 0.35 + 0.4 * u(3),
        rock,
        rock_w: 0.45 + 0.4 * u(4),
        grass: tint(5, 0.14),
        crown: tint(10, 0.10),
        trees: r(13, ar.trees) * (0.75 + 0.5 * own(14)),
        conifer: 0.5 * (own(15) - 0.5),
        shrubs: 0.6 + 0.8 * own(16),
        woodlots: 0.6 + 0.9 * own(17),
        season: r(18, ar.season),
        agri: r(19, ar.agri),
        towns: r(25, ar.towns) * (0.7 + 0.6 * s.development),
        fields,
        field_scale: r(26, ar.field_scale),
        hedges: r(27, ar.hedges),
        roof: r(28, ar.roof),
        height: r(29, ar.height) * (0.6 + 0.8 * s.development),
        block: r(30, ar.block),
        sodium: r(31, ar.sodium) * (1.2 - 0.6 * s.development),
        development: s.development,
        archetype: arch as u8,
        litho: lit as u8,
    }
}

/// The field system of a land-use region (`surface::RegionInfo`, the GPU's `Region`).
#[derive(Clone, Copy, Debug)]
pub struct RegionStyle {
    /// 0 grid, 1 irregular (Voronoi), 2 centre pivots, 3 long strips
    pub style: u8,
    pub fw: f64,
    pub fh: f64,
    pub hedge: f64,
    pub season: f64,
}

/// The field system of land-use region `id`: from the moisture and the style channel 3 at its
/// centre and its ecoregion's culture (field-system weights, field size, hedges, season).
/// Irrigated pivots wherever it is dry; elsewhere the culture's systems.
pub fn region_style(id: u64, moist: f64, style3: f64, e: &EcoStyle) -> RegionStyle {
    let dry = 1.0 - smoothstep(0.2, 0.4, moist);
    let u = u01k(id, 2);
    let style = if dry > 0.5 && u < 0.6 * dry {
        2
    } else {
        let w = e.fields;
        let total: f64 = w.iter().sum::<f64>().max(1e-9);
        let v = u01k(id, 22) * total;
        if v < w[0] {
            0
        } else if v < w[0] + w[1] {
            1
        } else if v < w[0] + w[1] + w[2] {
            2
        } else {
            3
        }
    };
    let scale = (0.6 + 1.1 * u01k(id, 3)) * e.field_scale;
    let (fw, fh) = match style {
        0 => (220.0 * scale, 220.0 * scale * (1.0 + 2.0 * u01k(id, 4))),
        1 => (300.0 * scale, 0.0),
        2 => (if u01k(id, 4) < 0.5 { 805.0 } else { 402.0 }, 0.0),
        _ => ((60.0 + 90.0 * u01k(id, 4)) * scale.sqrt(), (400.0 + 600.0 * u01k(id, 5)) * scale.sqrt()),
    };
    RegionStyle {
        style,
        fw,
        fh,
        hedge: if u01k(id, 7) < e.hedges { u01k(id, 8) } else { 0.0 },
        season: (0.5 * (style3 * 0.7 + 0.3 * u01k(id, 12)) + 0.5 * e.season).clamp(0.0, 1.0),
    }
}

/// A town's culture: (existence multiplier, block size multiplier, roof palette offset 0..1,
/// height 0..1).
pub fn town_style(id: u64, e: &EcoStyle) -> (f64, f64, f64, f64) {
    let roof = (e.roof + 0.25 * (u01k(id, 8) - 0.5)).rem_euclid(1.0);
    let height = (0.5 * u01k(id, 9) + e.height).clamp(0.0, 1.0);
    (e.towns, e.block, roof, height)
}

/// The parameters of ecoregion `id` near `p` (its site found from the lattice around `p`).
pub fn params_near(w: &World, reg: &Registry, id: u64, p: DVec3) -> Option<EcoParams> {
    let sites = find_sites(w, &[id], &[(p, 0.0)]);
    sites.get(&id).map(|&site| compute(w, reg, id, site))
}

/// The lattice sites of ecoregion `ids` that can be among the two nearest of a point within
/// the `areas` (centre, radius m): the lattice cells around them are enumerated (an id is the
/// hash of its cell). The GPU generator reports ids only; this finds their exact sites, so its
/// ecoregions are computed from the same sites as the CPU's.
pub fn find_sites(w: &World, ids: &[u64], areas: &[(DVec3, f64)]) -> FxHashMap<u64, DVec3> {
    let cell = w.cfg.ecoregions.cell_km * KM;
    let seed = w.seed() ^ ECO_KEY;
    // |warp| < √3 · 1.1 · (0.22 + 0.035) cell; the second-nearest site lies within ~1.7 cells
    let reach = 0.5 * cell + 2.0 * cell;
    let want: std::collections::HashSet<u64> = ids.iter().copied().collect();
    let mut out: FxHashMap<u64, DVec3> = FxHashMap::default();
    let (r_lo, r_hi) = (w.ell.b - 12_000.0 - reach - cell, w.ell.a + 9_000.0 + reach + cell);
    for &(c, r) in areas {
        if out.len() == want.len() {
            break;
        }
        let rr = r + reach;
        let lo = ((c - DVec3::splat(rr)) / cell).floor();
        let hi = ((c + DVec3::splat(rr)) / cell).floor();
        for z in lo.z as i64..=hi.z as i64 {
            for y in lo.y as i64..=hi.y as i64 {
                for x in lo.x as i64..=hi.x as i64 {
                    let cc = (DVec3::new(x as f64, y as f64, z as f64) + 0.5) * cell;
                    let d0 = cc.length();
                    if d0 < r_lo || d0 > r_hi || (cc - c).length() > rr + 0.87 * cell {
                        continue;
                    }
                    let (id, site) = worley3_site(seed, (x, y, z), cell, 0.9);
                    if want.contains(&id) {
                        out.insert(id, site);
                    }
                }
            }
        }
    }
    out
}

/// The ecoregion sites around `p` (the two nearest of the warped lattice).
pub fn sites(w: &World, p: DVec3) -> [(u64, DVec3); 2] {
    worley3_sites(w.seed() ^ ECO_KEY, p + warp(w, p), w.cfg.ecoregions.cell_km * KM, 0.9)
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    /// `Eco` of registry.wgsl: an ecoregion's resolved parameters (pass B).
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable, Default)]
    pub struct GEco {
        pub id: u64,
        /// biome, archetype, litho, culture
        pub biome: u32,
        pub arch: u32,
        pub litho: u32,
        pub culture: u32,
        pub _p: [u32; 2],
        /// soil (rgb, weight), rock (rgb, weight), grass tint, crown tint
        pub soil: [f32; 4],
        pub rock: [f32; 4],
        pub grass: [f32; 4],
        pub crown: [f32; 4],
        /// trees, conifer, shrubs, woodlots
        pub veg: [f32; 4],
        /// season, agri, towns, field scale
        pub land: [f32; 4],
        pub fields: [f32; 4],
        /// hedges, roof, height, block
        pub town: [f32; 4],
        /// sodium, development, site temp, site precip
        pub misc: [f32; 4],
        /// temp range, dry months, koppen, -
        pub clim: [f32; 4],
        /// `EcoParams::agri_w`
        pub aw: [f32; 16],
        /// `EcoParams::agri_mean`
        pub am: [f32; 4],
    }

    impl GEco {
        pub fn of(e: &EcoParams) -> GEco {
            let s = &e.style;
            let v = |c: DVec3, w: f64| [c.x as f32, c.y as f32, c.z as f32, w as f32];
            GEco {
                id: e.id,
                biome: e.biome as u32,
                arch: s.archetype as u32,
                litho: s.litho as u32,
                culture: e.culture,
                _p: [0; 2],
                soil: v(s.soil, s.soil_w),
                rock: v(s.rock, s.rock_w),
                grass: v(s.grass, 0.0),
                crown: v(s.crown, 0.0),
                veg: [s.trees as f32, s.conifer as f32, s.shrubs as f32, s.woodlots as f32],
                land: [s.season as f32, s.agri as f32, s.towns as f32, s.field_scale as f32],
                fields: s.fields.map(|f| f as f32),
                town: [s.hedges as f32, s.roof as f32, s.height as f32, s.block as f32],
                misc: [s.sodium as f32, s.development as f32, e.climate.temp_c as f32, e.climate.precip_mm as f32],
                clim: [e.climate.temp_range_c as f32, e.climate.dry_months as f32, e.climate.koppen as f32, 0.0],
                aw: e.agri_w.map(|v| v as f32),
                am: e.agri_mean.map(|v| v as f32),
            }
        }
    }

    #[cfg(test)]
    #[test]
    fn size() {
        assert_eq!(std::mem::size_of::<GEco>(), 32 + 15 * 16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ecoregions pick many biomes and archetypes: their variety does not collapse.
    #[test]
    fn ecoregions_vary() {
        let w = World::new(crate::Config::default());
        let reg = Registry::builtin().unwrap();
        let mut biomes = std::collections::BTreeMap::<String, usize>::new();
        let mut arch = std::collections::BTreeSet::new();
        let mut litho = std::collections::BTreeSet::new();
        let mut n = 0;
        for i in 0..1500u64 {
            let h = mix64(0xEC0 ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            if w.macro_at(p, 20_000.0).cont < 0.05 {
                continue;
            }
            let [(id, site), _] = sites(&w, p);
            let e = compute(&w, &reg, id, site);
            *biomes.entry(reg.biomes[e.biome as usize].name.clone()).or_default() += 1;
            arch.insert(e.style.archetype);
            litho.insert(e.style.litho);
            n += 1;
        }
        assert!(n > 200, "{n} land samples");
        assert!(biomes.len() >= 6, "biomes: {biomes:?}");
        assert!(arch.len() >= 8, "archetypes: {arch:?}");
        assert_eq!(litho.len(), 5);
        // no biome takes more than 45 % of the land
        let max = biomes.values().max().copied().unwrap_or(0);
        assert!(max * 100 <= 45 * n, "biomes: {biomes:?}");
    }
}
