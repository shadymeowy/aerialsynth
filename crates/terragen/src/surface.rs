//! Fine-scale surface ("pass B"): evaluated per sub-sample on top of interpolated pass-A values.
//! Produces albedo (linear RGB), DSM height, land cover and a cast-shadow factor.
//!
//! [`SurfaceModel`] holds what the layers share (noise fields, the default palette, the biome
//! registry, the land-use site caches); the surface itself is the composite stack of
//! [`crate::stack`] over the layers of [`crate::layers`] and the kits.
//!
//! Everything is analytic and prefiltered: a feature of size `s` is drawn explicitly only when
//! `s` is resolvable at the pixel GSD; otherwise its expected (mean) contribution is used, so a coarse
//! zoom level looks like the average of the finer one.

use crate::noise::*;
use crate::world::{water, Ctx, Terrain, World};
use glam::{DVec2, DVec3};

/// Linear-light colour from sRGB 0..255.
#[inline]
pub fn srgb(r: f64, g: f64, b: f64) -> DVec3 {
    DVec3::new(s2l(r / 255.0), s2l(g / 255.0), s2l(b / 255.0))
}
#[inline]
pub fn s2l(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
#[inline]
pub fn l2s(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}
#[inline]
pub fn mixc(a: DVec3, b: DVec3, t: f64) -> DVec3 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Inputs to pass B at one sub-sample.
#[derive(Clone, Copy, Debug)]
pub struct Local<'a> {
    /// Pass-A data of the nearest pixel (discrete info, sites, climate).
    pub t: &'a Terrain,
    /// Bilinearly interpolated bare-earth elevation.
    pub ground: f64,
    /// Standing water level (max over neighbouring water pixels) or -inf.
    pub water: f64,
    pub water_kind: u8,
    pub river_d: f64,
    pub river_hw: f64,
    pub river_level: f64,
    /// distance to the ecoregion border (m), interpolated like the region edge
    pub eco_edge: f64,
    pub road_major: f64,
    pub road_minor: f64,
    /// Terrain slope (tan) at pixel scale.
    pub slope: f64,
    /// Gradient of the ground (east, north; m/m) at pixel scale: aspect, flow direction.
    pub grad: [f64; 2],
    /// Sub-sample filter width (m) for analytic edge antialiasing.
    pub fw: f64,
    /// the instance lists of the pixel's block (per family; empty: not known)
    pub inst: &'a [Vec<crate::instances::Instance>],
    /// the tile's binned features and the pixel's bin
    pub feat: Option<(&'a crate::features::Binned, usize)>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Surface {
    pub albedo: DVec3,
    /// DSM elevation (m above ellipsoid).
    pub height: f64,
    pub class: u8,
    /// 1 = fully sunlit, 0 = fully in cast shadow.
    pub lit: f64,
    pub is_water: bool,
    /// Night-time artificial light (linear radiance).
    pub emission: DVec3,
}

/// Smooth noise fields evaluated once per pixel and shared by its sub-samples. The fields that
/// only some surfaces need (rock strata, forest stands, fields, water) are completed on first use
/// (see `SurfaceModel::pf_lazy`).
#[derive(Clone, Debug, Default)]
pub struct PixFields {
    pub detail: f64,
    pub patch: f64,
    pub land: f64,
    pub snow: f64,
    pub forest: f64,
    pub warp2: f64,
    /// domain warp of the forest-stand lattice (×70 m)
    pub stand_warp: [f64; 3],
    /// the forest stand, where known for the whole pixel
    pub stand_id: Option<u64>,
    /// the fields `LAZY`: the part known so far, and which are still to be completed (their short
    /// octaves at `p`, `gsd`, below `cut`, or all octaves without a cut)
    lazy: [std::cell::Cell<f64>; 7],
    pending: std::cell::Cell<u8>,
    p: DVec3,
    gsd: f64,
    cut: Option<f64>,
}

/// Indices of the pixel fields in [`SurfaceModel::pixel_fields_part`].
pub(crate) const PF_STRATA: usize = 3;
pub(crate) const PF_STRATA2: usize = 4;
pub(crate) const PF_STAND: usize = 7;
pub(crate) const PF_FIELD_VAR: usize = 8;
pub(crate) const PF_FIELD_VAR2: usize = 9;
pub(crate) const PF_FIELD_VAR3: usize = 10;
pub(crate) const PF_WATER: usize = 12;
const LAZY: [usize; 7] = [PF_STRATA, PF_STRATA2, PF_STAND, PF_FIELD_VAR, PF_FIELD_VAR2, PF_FIELD_VAR3, PF_WATER];

impl PixFields {
    pub const N: usize = 16;
    /// From the values of [`SurfaceModel::pixel_fields_part`] at `p`; the lazy fields hold the
    /// part that is known (the long octaves below `cut`, or nothing without a cut).
    pub fn from_parts(a: [f64; Self::N], p: DVec3, gsd: f64, cut: Option<f64>) -> Self {
        PixFields {
            detail: a[0],
            patch: a[1],
            land: a[2],
            snow: a[5],
            forest: a[6],
            warp2: a[11],
            stand_warp: [a[13], a[14], a[15]],
            stand_id: None,
            lazy: LAZY.map(|i| std::cell::Cell::new(a[i])),
            pending: std::cell::Cell::new((1 << LAZY.len()) - 1),
            p,
            gsd,
            cut,
        }
    }
}

/// Field system of a land-use region.
#[derive(Clone, Copy, Debug)]
pub struct RegionInfo {
    pub center: DVec3,
    pub ex: DVec3,
    pub ey: DVec3,
    /// Unrotated local east/north at the centre (trees, shadows).
    pub east: DVec3,
    pub north: DVec3,
    pub style: u8,
    pub fw: f64,
    pub fh: f64,
    pub split: f64,
    pub hedge: f64,
    pub track: f64,
    pub border_w: f64,
    pub palette: f64,
    pub agri: f64,
    /// 0 = spring (green), 0.5 = summer (golden), 1 = autumn (ploughed)
    pub season: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct TownInfo {
    pub exists: bool,
    pub center: DVec3,
    pub ex: DVec3,
    pub ey: DVec3,
    pub radius: f64,
    pub block: f64,
    pub street: f64,
    pub organic: f64,
    pub roof_style: f64,
    pub height: f64,
    pub lot: f64,
    /// footprint elongation (≥ 1) and a per-town seed
    pub elong: f64,
    pub seed: u64,
    /// Horizontal unit vector towards the sun (ECEF).
    pub sun: DVec3,
}

/// Per-thread caches for site-level properties (pure functions of the site; caching is only an
/// optimization, so results are deterministic regardless of evaluation order).
#[derive(Default)]
pub struct Caches {
    regions: FxHashMap<u64, RegionInfo>,
    towns: FxHashMap<u64, TownInfo>,
    /// towns before resolving overlaps
    towns_base: FxHashMap<u64, TownInfo>,
    /// existing towns around each cell of the town lattice
    town_cands: FxHashMap<(i64, i64, i64), Vec<TownInfo>>,
    /// per field (by its centre): cultivation mask value and crop-cluster id
    pub(crate) fields: FxHashMap<[u64; 4], (f64, u64)>,
    /// ecoregions by id
    pub(crate) eco: FxHashMap<u64, crate::eco::EcoParams>,
}

impl Caches {
    /// Bound the memory of a long-lived cache.
    pub fn trim(&mut self) {
        if self.regions.len() + self.towns.len() + self.towns_base.len() + self.town_cands.len() + self.fields.len() + self.eco.len() > 200_000 {
            *self = Caches::default();
        }
    }
}

/// A small light source (lamp head) of peak `amp` and radius `sigma` at squared distance `d2`,
/// widened to the sample spacing `fw` with its energy kept: at coarse zooms it becomes one bright
/// texel instead of vanishing (band-limiting it away left only the soft pools, every lamp a disc).
pub fn point_light(d2: f64, amp: f64, sigma: f64, fw: f64) -> f64 {
    let s = sigma.max(0.6 * fw);
    amp * (sigma * sigma) / (s * s) * (-d2 / (2.0 * s * s)).exp()
}

/// Sampling context at the surface point below a 3D Worley site.
pub(crate) fn site_ctx(world: &World, pt: DVec3, gsd: f64) -> Ctx {
    let g = geodesy::ecef2geodetic(pt, &world.ell);
    Ctx::new(g.lat, g.lon, gsd, &world.ell)
}

pub struct Palette {
    pub(crate) soil: [DVec3; 4],
    pub(crate) grass_wet: DVec3,
    pub(crate) grass_dry: DVec3,
    pub(crate) grass_cold: DVec3,
    pub(crate) rock: [DVec3; 3],
    pub(crate) snow: DVec3,
    pub(crate) sand: [DVec3; 3],
    pub(crate) beach: DVec3,
    pub(crate) wet_sand: DVec3,
    pub(crate) tundra: DVec3,
    pub(crate) marsh: DVec3,
    pub(crate) crown_conifer: DVec3,
    pub(crate) crown_decid: DVec3,
    pub(crate) crown_tropic: DVec3,
    pub(crate) crown_dry: DVec3,
    pub(crate) shrub: DVec3,
    pub(crate) crop: [DVec3; 9],
    pub crop_mean: DVec3,
    pub(crate) asphalt: DVec3,
    pub(crate) gravel: DVec3,
    pub(crate) concrete: DVec3,
    pub(crate) roofs: [DVec3; 7],
    pub(crate) ocean_deep: DVec3,
    pub(crate) ocean_shallow: DVec3,
    pub(crate) lake_deep: DVec3,
    pub(crate) river: DVec3,
}

impl Palette {
    pub fn new() -> Self {
        let mut p = Self::base();
        p.crop_mean = p.crop.iter().take(7).fold(DVec3::ZERO, |a, c| a + *c) / 7.0;
        p
    }

    fn base() -> Self {
        Palette {
            soil: [srgb(128.0, 100.0, 72.0), srgb(150.0, 92.0, 60.0), srgb(118.0, 108.0, 94.0), srgb(165.0, 140.0, 105.0)],
            grass_wet: srgb(84.0, 112.0, 50.0),
            grass_dry: srgb(172.0, 156.0, 104.0),
            grass_cold: srgb(112.0, 116.0, 84.0),
            rock: [srgb(128.0, 122.0, 114.0), srgb(150.0, 128.0, 104.0), srgb(92.0, 90.0, 88.0)],
            snow: srgb(238.0, 242.0, 248.0),
            sand: [srgb(212.0, 190.0, 150.0), srgb(198.0, 162.0, 118.0), srgb(224.0, 208.0, 176.0)],
            beach: srgb(222.0, 206.0, 165.0),
            wet_sand: srgb(150.0, 140.0, 118.0),
            tundra: srgb(128.0, 124.0, 98.0),
            marsh: srgb(84.0, 98.0, 66.0),
            crown_conifer: srgb(34.0, 54.0, 38.0),
            crown_decid: srgb(52.0, 80.0, 36.0),
            crown_tropic: srgb(36.0, 76.0, 34.0),
            crown_dry: srgb(88.0, 96.0, 58.0),
            shrub: srgb(78.0, 82.0, 56.0),
            crop: [
                srgb(92.0, 108.0, 66.0),   // green crop
                srgb(124.0, 132.0, 86.0),  // light green
                srgb(184.0, 168.0, 124.0), // ripe cereal
                srgb(178.0, 166.0, 138.0), // stubble
                srgb(124.0, 108.0, 88.0),  // ploughed
                srgb(152.0, 136.0, 110.0), // dry soil
                srgb(120.0, 126.0, 88.0),  // pasture
                srgb(196.0, 186.0, 104.0), // rapeseed
                srgb(104.0, 106.0, 76.0),  // orchard ground
            ],
            crop_mean: DVec3::ZERO,
            asphalt: srgb(78.0, 78.0, 82.0),
            gravel: srgb(162.0, 146.0, 120.0),
            concrete: srgb(176.0, 174.0, 168.0),
            roofs: [
                srgb(158.0, 74.0, 54.0),
                srgb(128.0, 62.0, 50.0),
                srgb(120.0, 120.0, 124.0),
                srgb(80.0, 80.0, 86.0),
                srgb(205.0, 204.0, 198.0),
                srgb(150.0, 120.0, 96.0),
                srgb(96.0, 110.0, 120.0),
            ],
            ocean_deep: srgb(14.0, 36.0, 66.0),
            ocean_shallow: srgb(48.0, 104.0, 110.0),
            lake_deep: srgb(22.0, 44.0, 58.0),
            river: srgb(54.0, 72.0, 70.0),
        }
    }
}

impl Default for Palette {
    fn default() -> Self {
        Self::new()
    }
}

/// Crop kind for a field, from the region's season (0 spring .. 1 autumn) and a random draw.
/// kinds: 0 green, 1 light green, 2 ripe cereal, 3 stubble, 4 ploughed, 5 dry soil, 6 pasture,
/// 7 rapeseed, 8 orchard
pub(crate) fn crop_kind(season: f64, u: f64, dry: bool) -> usize {
    const SPRING: [f64; 9] = [0.35, 0.25, 0.0, 0.0, 0.15, 0.04, 0.13, 0.02, 0.06];
    const SUMMER: [f64; 9] = [0.14, 0.10, 0.30, 0.20, 0.08, 0.03, 0.10, 0.0, 0.05];
    const AUTUMN: [f64; 9] = [0.10, 0.03, 0.02, 0.20, 0.33, 0.12, 0.12, 0.0, 0.08];
    let (a, b, f) = if season < 0.5 { (&SPRING, &SUMMER, season * 2.0) } else { (&SUMMER, &AUTUMN, season * 2.0 - 1.0) };
    let mut acc = 0.0;
    let mut kind = 8;
    for k in 0..9 {
        acc += a[k] + (b[k] - a[k]) * f;
        if u < acc {
            kind = k;
            break;
        }
    }
    if dry && kind <= 1 {
        kind = if u < 0.5 { 5 } else { 3 };
    }
    kind
}

/// Coverage of a band of half-width `hw` at distance `d`, box-filtered with width `fw`.
#[inline]
pub fn band_cov(d: f64, hw: f64, fw: f64) -> f64 {
    ((hw - d.abs()) / fw + 0.5).clamp(0.0, 1.0)
}

pub struct SurfaceModel {
    pub pal: Palette,
    pub(crate) detail: Fbm,
    pub(crate) patch: Fbm,
    pub(crate) forest: Fbm,
    pub(crate) field_var: Fbm,
    pub(crate) warp2: Fbm,
    pub(crate) strata: Fbm,
    pub(crate) snow_n: Fbm,
    pub(crate) cult_n: Fbm,
    pub(crate) land_n: Fbm,
    /// Horizontal direction to the sun (ENU) and tan(elevation).
    pub(crate) sun_h: DVec2,
    pub(crate) sun_tan: f64,
    /// Site data shared by all worker threads (each site is computed once; the per-thread
    /// `Caches` in front of it avoid the lock on most lookups).
    shared: std::sync::RwLock<SharedSites>,
    /// the biome registry (core + kits, with the world's overrides)
    pub registry: std::sync::Arc<crate::registry::Registry>,
    /// ecoregion parameters (shared cache)
    pub eco: crate::eco::Ecoregions,
}

#[derive(Default)]
struct SharedSites {
    regions: FxHashMap<u64, RegionInfo>,
    towns: FxHashMap<u64, TownInfo>,
    towns_base: FxHashMap<u64, TownInfo>,
}

impl SurfaceModel {
    pub fn new(world: &World) -> Self {
        let s = world.seed();
        let k = |i: u64| mix64(s ^ (0xB0B0 + i * 31337));
        let look = &world.cfg.satellite;
        let az = look.sun_azimuth_deg.to_radians();
        let el = look.sun_elevation_deg.to_radians().max(0.05);
        SurfaceModel {
            pal: Palette::new(),
            detail: Fbm::new(k(1), 60.0, 7, 2.0, 0.6),
            patch: Fbm::new(k(2), 450.0, 5, 2.0, 0.55),
            forest: Fbm::new(k(3), 2600.0, 7, 2.0, 0.55),
            field_var: Fbm::new(k(4), 140.0, 4, 2.0, 0.5),
            warp2: Fbm::new(k(5), 380.0, 2, 2.0, 0.5),
            strata: Fbm::new(k(6), 900.0, 3, 2.0, 0.5),
            snow_n: Fbm::new(k(7), 1800.0, 6, 2.0, 0.55),
            cult_n: Fbm::new(k(8), 3500.0, 4, 2.0, 0.5),
            land_n: Fbm::new(k(9), 9000.0, 5, 2.0, 0.55),
            sun_h: DVec2::new(az.sin(), az.cos()),
            sun_tan: el.tan(),
            shared: Default::default(),
            registry: crate::registry::Registry::for_config(&world.cfg).expect("biome registry (checked by Config::validate)"),
            eco: Default::default(),
        }
    }

    /// Evaluate the surface at one sub-sample (the composite stack, [`crate::stack`]).
    pub fn eval(&self, world: &World, cache: &mut Caches, ctx: &Ctx, l: &Local, pf: &PixFields) -> Surface {
        crate::stack::eval(self, world, cache, ctx, l, pf)
    }

    fn shared_get<T: Copy>(&self, f: impl Fn(&SharedSites) -> Option<T>) -> Option<T> {
        f(&self.shared.read().unwrap())
    }

    fn shared_put(&self, f: impl FnOnce(&mut SharedSites)) {
        let mut g = self.shared.write().unwrap();
        if g.regions.len() + g.towns.len() + g.towns_base.len() > 500_000 {
            *g = SharedSites::default();
        }
        f(&mut g);
    }

    /// Forest stand at `p` (~240 m, irregular borders) given the stand warp (else evaluated at `p`;
    /// it is smooth at the scale of a pixel, so the pixel's is used for its samples).
    pub fn stand_id(&self, p: DVec3, warp: Option<[f64; 3]>) -> u64 {
        let w = warp.unwrap_or_else(|| [0x57A1, 0x57A2, 0x57A3].map(|k| perlin3(k, p / 180.0)));
        let sp = p + DVec3::from_array(w) * 70.0;
        worley3(0x57A4, sp, 240.0, 0.9).id
    }

    /// Per-pixel smooth fields (band-limited at the pixel GSD).
    pub fn pixel_fields(&self, p: DVec3, gsd: f64) -> PixFields {
        PixFields::from_parts(self.pixel_fields_part(p, gsd, None, false), p, gsd, None)
    }

    /// The pixel fields, all octaves (`split` None) or only those of wavelength >= `cut` (`split`
    /// Some((cut, true))) or < `cut` (Some((cut, false))); the two parts sum to the whole. The
    /// smooth part is interpolated from a coarse grid by the tile generator. The `LAZY` fields
    /// are 0 unless `lazy_too`.
    pub fn pixel_fields_part(&self, p: DVec3, gsd: f64, split: Option<(f64, bool)>, lazy_too: bool) -> [f64; PixFields::N] {
        std::array::from_fn(|i| if lazy_too || !LAZY.contains(&i) { self.pixel_field(i, p, gsd, split) } else { 0.0 })
    }

    /// Pixel field `i` (or a part of it, see [`SurfaceModel::pixel_fields_part`]).
    pub(crate) fn pixel_field(&self, i: usize, p: DVec3, gsd: f64, split: Option<(f64, bool)>) -> f64 {
        // a field evaluated at p·k sees a sample spacing of gsd·k (and wavelengths ·k) in its own
        // domain
        let f = |n: &Fbm, k: f64| -> f64 {
            match split {
                None => n.eval(p * k, gsd * k),
                Some((cut, low)) => n.eval_part(p * k, gsd * k, cut * k, low),
            }
        };
        // single-octave terms of wavelength `lam`
        let single = |lam: f64, f: &dyn Fn() -> f64| -> f64 {
            match split {
                Some((cut, low)) if (lam >= cut) != low => 0.0,
                _ => f(),
            }
        };
        let stand_warp = |seed: u64| single(180.0, &|| perlin3(seed, p / 180.0));
        match i {
            0 => f(&self.detail, 1.0),
            1 => f(&self.patch, 1.0),
            2 => f(&self.land_n, 1.0) * self.land_n.norm() * 1.8,
            PF_STRATA => f(&self.strata, 1.0),
            PF_STRATA2 => f(&self.strata, 1.7),
            5 => f(&self.snow_n, 1.0),
            6 => f(&self.forest, 1.0) * self.forest.norm() * 1.8,
            PF_STAND => f(&self.patch, 0.3) + single(1200.0, &|| 0.5 * perlin3(0x57A, p / 1200.0) * crate::noise::band(1200.0, gsd)),
            PF_FIELD_VAR => f(&self.field_var, 1.0),
            PF_FIELD_VAR2 => f(&self.field_var, 1.7),
            PF_FIELD_VAR3 => f(&self.field_var, 3.0),
            11 => f(&self.warp2, 1.0),
            PF_WATER => f(&self.patch, 0.37),
            13 => stand_warp(0x57A1),
            14 => stand_warp(0x57A2),
            15 => stand_warp(0x57A3),
            _ => unreachable!(),
        }
    }

    /// Lazy pixel field `i` (one of `LAZY`), completed on first use.
    pub(crate) fn pf_lazy(&self, pf: &PixFields, i: usize) -> f64 {
        let k = LAZY.iter().position(|&l| l == i).expect("a lazy pixel field");
        if pf.pending.get() & (1 << k) != 0 {
            let rest = self.pixel_field(i, pf.p, pf.gsd, pf.cut.map(|c| (c, false)));
            pf.lazy[k].set(pf.lazy[k].get() + rest);
            pf.pending.set(pf.pending.get() & !(1 << k));
        }
        pf.lazy[k].get()
    }

    pub(crate) fn region_info(&self, world: &World, cache: &mut Caches, t: &Terrain) -> RegionInfo {
        if let Some(r) = cache.regions.get(&t.region.id) {
            return *r;
        }
        if let Some(r) = self.shared_get(|sh| sh.regions.get(&t.region.id).copied()) {
            cache.regions.insert(t.region.id, r);
            return r;
        }
        let id = t.region.id;
        let cctx = site_ctx(world, t.region.center, 400.0);
        let (c, east, north) = (cctx.p, cctx.east, cctx.north);
        let ang = u01k(id, 1) * std::f64::consts::PI;
        let (sa, ca) = ang.sin_cos();
        let ex = east * ca + north * sa;
        let ey = north * ca - east * sa;
        // climate at the region centre and its ecoregion's culture decide the field system
        let tc = world.terrain(&cctx);
        let eco = self.eco.params(world, &self.registry, cache, tc.eco.id, tc.eco.center);
        let rs = crate::eco::region_style(id, tc.moist, tc.style[3], &eco.style);
        let (style, fw, fh) = (rs.style, rs.fw, rs.fh);
        let info = RegionInfo {
            center: c,
            ex,
            ey,
            east,
            north,
            style,
            fw,
            fh,
            split: u01k(id, 6),
            hedge: rs.hedge,
            track: 0.2 + 0.6 * u01k(id, 9),
            border_w: 1.5 + 3.0 * u01k(id, 10),
            palette: u01k(id, 11),
            agri: tc.agri,
            season: rs.season,
        };
        cache.regions.insert(id, info);
        self.shared_put(|sh| {
            sh.regions.insert(id, info);
        });
        info
    }

    /// The most built-up town at `p` among the existing towns of the lattice cells within
    /// ±`range` cells of p's cell (cached per cell). Complete for `range` 2: a town reaches at
    /// most ~1.16 cells from its centre, which is within 0.8 cells of its site.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn select_town(
        &self,
        world: &World,
        cache: &mut Caches,
        p: DVec3,
        gsd: f64,
        slope: f64,
        clear: f64,
        pf: &PixFields,
        range: i64,
    ) -> Option<(TownInfo, f64)> {
        let cell = world.cfg.landuse.town_cell_km * 1000.0;
        let qf = (p / cell).floor();
        let key = (qf.x as i64, qf.y as i64, qf.z as i64);
        if !cache.town_cands.contains_key(&key) {
            let mut v = Vec::new();
            for dz in -range..=range {
                for dy in -range..=range {
                    for dx in -range..=range {
                        let (id, c) = worley3_site(world.seed ^ 0x70E1, (key.0 + dx, key.1 + dy, key.2 + dz), cell, 0.8);
                        let info = self.town_info(world, cache, id, c);
                        if info.exists {
                            v.push(info);
                        }
                    }
                }
            }
            cache.town_cands.insert(key, v);
        }
        let mut sel: Option<(TownInfo, f64)> = None;
        for info in &cache.town_cands[&key] {
            let u = self.town_urban(info, p, gsd, slope, clear, pf).0;
            if u > sel.map_or(0.0, |b| b.1) {
                sel = Some((*info, u));
            }
        }
        sel
    }

    /// A town site, with overlaps resolved: of two towns whose footprints would overlap only the
    /// larger exists (overlapping towns with different street grids met along seams that cut
    /// streets and houses).
    fn town_info(&self, world: &World, cache: &mut Caches, id: u64, center: DVec3) -> TownInfo {
        if let Some(r) = cache.towns.get(&id) {
            return *r;
        }
        if let Some(r) = self.shared_get(|sh| sh.towns.get(&id).copied()) {
            cache.towns.insert(id, r);
            return r;
        }
        let mut info = self.town_base(world, cache, id, center);
        if info.exists {
            let cell = world.cfg.landuse.town_cell_km * 1000.0;
            let extent = |t: &TownInfo| 1.6 * t.radius * t.elong.sqrt();
            let k = (center / cell).floor();
            'search: for dz in -2..=2i64 {
                for dy in -2..=2i64 {
                    for dx in -2..=2i64 {
                        let (nid, nc) = worley3_site(world.seed ^ 0x70E1, (k.x as i64 + dx, k.y as i64 + dy, k.z as i64 + dz), cell, 0.8);
                        if nid == id {
                            continue;
                        }
                        let n = self.town_base(world, cache, nid, nc);
                        if !n.exists || (n.center - info.center).length() > extent(&n) + extent(&info) {
                            continue;
                        }
                        if n.radius > info.radius || (n.radius == info.radius && nid > id) {
                            info.exists = false;
                            break 'search;
                        }
                    }
                }
            }
        }
        cache.towns.insert(id, info);
        self.shared_put(|sh| {
            sh.towns.insert(id, info);
        });
        info
    }

    fn town_base(&self, world: &World, cache: &mut Caches, id: u64, center: DVec3) -> TownInfo {
        if let Some(r) = cache.towns_base.get(&id) {
            return *r;
        }
        if let Some(r) = self.shared_get(|sh| sh.towns_base.get(&id).copied()) {
            cache.towns_base.insert(id, r);
            return r;
        }
        let ctx = site_ctx(world, center, 300.0);
        let (east, north) = (ctx.east, ctx.north);
        // only lattice sites within 0.8 cells of the surface make towns: a town sits below /
        // above its site, and a site farther away lay outside the ±2-cell candidate search of the
        // pixels its town covers (the town was cut along lattice-cell planes); see `select_town`
        let near_surface = (center.length() - ctx.p.length()).abs() < 0.8 * world.cfg.landuse.town_cell_km * 1000.0;
        // the terrain under the site (expensive: drainage) only where a town can still exist
        // (p_exist <= 0.95)
        // (the culture of the town's ecoregion: density, blocks, roofs, heights)
        let mut culture = (1.0, 1.0, u01k(id, 8), u01k(id, 9));
        let exists = near_surface && u01k(id, 1) < 0.95 && {
            let tc = world.terrain(&ctx);
            let eco = self.eco.params(world, &self.registry, cache, tc.eco.id, tc.eco.center);
            culture = crate::eco::town_style(id, &eco.style);
            // (the atlas' population potential sets the density around the culture's)
            let p_exist = (tc.habit * 1.1 * world.cfg.landuse.towns * culture.0 * (0.45 + 1.1 * tc.population)).min(0.95);
            u01k(id, 1) < p_exist && tc.water_kind == water::NONE && tc.ground > 2.0 && tc.ground < 4000.0
        };
        let ang = u01k(id, 2) * std::f64::consts::FRAC_PI_2;
        let (sa, ca) = ang.sin_cos();
        let mut radius = 160.0 * (u01k(id, 3).powf(1.6) * 2.4).exp();
        if u01k(id, 4) < 0.03 {
            radius *= 4.0; // occasional city
        }
        let elong = 1.0 + 1.6 * u01k(id, 11) * u01k(id, 12);
        let info = TownInfo {
            exists,
            center: ctx.p,
            ex: east * ca + north * sa,
            ey: north * ca - east * sa,
            radius: radius.min(world.cfg.landuse.town_cell_km * 1000.0 * 0.45),
            block: (70.0 + 70.0 * u01k(id, 5)) * culture.1,
            street: 6.5 + 6.0 * u01k(id, 6),
            organic: u01k(id, 7),
            roof_style: culture.2,
            height: culture.3,
            lot: 13.0 + 12.0 * u01k(id, 10),
            elong,
            seed: mix64(id ^ 0x70E5),
            sun: east * self.sun_h.x + north * self.sun_h.y,
        };
        cache.towns_base.insert(id, info);
        self.shared_put(|sh| {
            sh.towns_base.insert(id, info);
        });
        info
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ±2-cell town candidate search finds the same town as a much wider (±4) search.
    #[test]
    fn town_search_is_complete() {
        let world = World::new(crate::config::Config::default());
        let sm = SurfaceModel::new(&world);
        let mut rng = 0x1234_5678_u64;
        let mut next = || {
            rng = mix64(rng);
            u01(rng)
        };
        let (mut towns, mut n) = (0, 0);
        let (mut ca, mut cb) = (Caches::default(), Caches::default());
        for &(lat0, lon0) in &[(41.58, 33.18), (16.69, -15.77), (39.6, 33.6), (41.573, 32.984), (39.894, 32.93)] {
            for _ in 0..400 {
                let (lat, lon) = (lat0 + 0.12 * (next() - 0.5), lon0 + 0.16 * (next() - 0.5));
                let ctx = Ctx::new(lat.to_radians(), lon.to_radians(), 2.0, &world.ell);
                let pf = sm.pixel_fields(ctx.p, 2.0);
                let a = sm.select_town(&world, &mut ca, ctx.p, 2.0, 0.2, 1.0, &pf, 2);
                let b = sm.select_town(&world, &mut cb, ctx.p, 2.0, 0.2, 1.0, &pf, 4);
                assert_eq!(a.map(|x| x.0.seed), b.map(|x| x.0.seed), "different town at {lat},{lon}");
                towns += a.is_some() as usize;
                n += 1;
            }
        }
        // (enough towns that the comparison means something; their density varies by culture)
        assert!(towns > n / 50, "too few town samples ({towns}/{n})");
    }
}
