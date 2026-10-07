//! Fine-scale surface ("pass B"): evaluated per sub-sample on top of interpolated pass-A values.
//! Produces albedo (linear RGB), DSM height, land cover and a cast-shadow factor.
//!
//! Everything is analytic and prefiltered: a feature of size `s` is drawn explicitly only when
//! `s` is resolvable at the pixel GSD; otherwise its expected (mean) contribution is used, so a coarse
//! zoom level looks like the average of the finer one.

use crate::landcover as lc;
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
fn mixc(a: DVec3, b: DVec3, t: f64) -> DVec3 {
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
    pub road_major: f64,
    pub road_minor: f64,
    /// Terrain slope (tan) at pixel scale.
    pub slope: f64,
    /// Sub-sample filter width (m) for analytic edge antialiasing.
    pub fw: f64,
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

/// Smooth noise fields evaluated once per pixel and shared by its sub-samples.
#[derive(Clone, Copy, Debug, Default)]
pub struct PixFields {
    pub detail: f64,
    pub patch: f64,
    pub land: f64,
    pub strata: f64,
    pub strata2: f64,
    pub snow: f64,
    pub forest: f64,
    pub stand: f64,
    pub field_var: f64,
    pub field_var2: f64,
    pub field_var3: f64,
    pub warp2: f64,
    pub water: f64,
    /// domain warp of the forest-stand lattice (×70 m)
    pub stand_warp: [f64; 3],
    /// the forest stand, where known for the whole pixel
    pub stand_id: Option<u64>,
}

impl PixFields {
    pub const N: usize = 16;
    pub fn from_array(a: [f64; Self::N]) -> Self {
        let [detail, patch, land, strata, strata2, snow, forest, stand, field_var, field_var2, field_var3, warp2, water, w0, w1, w2] = a;
        PixFields { detail, patch, land, strata, strata2, snow, forest, stand, field_var, field_var2, field_var3, warp2, water, stand_warp: [w0, w1, w2], stand_id: None }
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
    fields: FxHashMap<[u64; 4], (f64, u64)>,
}

impl Caches {
    /// Bound the memory of a long-lived cache.
    pub fn trim(&mut self) {
        if self.regions.len() + self.towns.len() + self.towns_base.len() + self.town_cands.len() + self.fields.len() > 200_000 {
            *self = Caches::default();
        }
    }
}

/// A small light source (lamp head) of peak `amp` and radius `sigma` at squared distance `d2`,
/// widened to the sample spacing `fw` with its energy kept: at coarse zooms it becomes one bright
/// texel instead of vanishing (band-limiting it away left only the soft pools, every lamp a disc).
fn point_light(d2: f64, amp: f64, sigma: f64, fw: f64) -> f64 {
    let s = sigma.max(0.6 * fw);
    amp * (sigma * sigma) / (s * s) * (-d2 / (2.0 * s * s)).exp()
}

/// Sampling context at the surface point below a 3D Worley site.
fn site_ctx(world: &World, pt: DVec3, gsd: f64) -> Ctx {
    let g = geodesy::ecef2geodetic(pt, &world.ell);
    Ctx::new(g.lat, g.lon, gsd, &world.ell)
}

#[allow(dead_code)]
fn tangent_frame(c: DVec3) -> (DVec3, DVec3, DVec3) {
    let up = c.normalize();
    let mut east = DVec3::Z.cross(up);
    if east.length_squared() < 1e-12 {
        east = DVec3::X;
    }
    let east = east.normalize();
    let north = up.cross(east);
    (up, east, north)
}

pub struct Palette {
    soil: [DVec3; 4],
    grass_wet: DVec3,
    grass_dry: DVec3,
    grass_cold: DVec3,
    rock: [DVec3; 3],
    snow: DVec3,
    sand: [DVec3; 3],
    beach: DVec3,
    wet_sand: DVec3,
    tundra: DVec3,
    marsh: DVec3,
    crown_conifer: DVec3,
    crown_decid: DVec3,
    crown_tropic: DVec3,
    crown_dry: DVec3,
    shrub: DVec3,
    crop: [DVec3; 9],
    pub crop_mean: DVec3,
    asphalt: DVec3,
    gravel: DVec3,
    concrete: DVec3,
    roofs: [DVec3; 7],
    ocean_deep: DVec3,
    ocean_shallow: DVec3,
    lake_deep: DVec3,
    river: DVec3,
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
fn crop_kind(season: f64, u: f64, dry: bool) -> usize {
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
fn band_cov(d: f64, hw: f64, fw: f64) -> f64 {
    ((hw - d.abs()) / fw + 0.5).clamp(0.0, 1.0)
}

pub struct SurfaceModel {
    pub pal: Palette,
    detail: Fbm,
    patch: Fbm,
    forest: Fbm,
    field_var: Fbm,
    warp2: Fbm,
    strata: Fbm,
    snow_n: Fbm,
    cult_n: Fbm,
    land_n: Fbm,
    /// Horizontal direction to the sun (ENU) and tan(elevation).
    sun_h: DVec2,
    sun_tan: f64,
    /// Site data shared by all worker threads (each site is computed once; the per-thread
    /// `Caches` in front of it avoid the lock on most lookups).
    shared: std::sync::RwLock<SharedSites>,
}

#[derive(Default)]
struct SharedSites {
    regions: FxHashMap<u64, RegionInfo>,
    towns: FxHashMap<u64, TownInfo>,
    towns_base: FxHashMap<u64, TownInfo>,
}

/// One tree layer (crowns on a jittered grid).
#[derive(Clone, Copy)]
struct TreeLayer {
    cell: f64,
    seed: u64,
    density: f64,
    /// density of the whole stand (all layers): crowns grow to close the canopy
    closure: f64,
    height: f64,
    color: DVec3,
    conifer: f64,
    /// crown size of the stand (age): ~0.7 young … ~1.25 old
    scale: f64,
    /// colour of the stand (species, age, health)
    tone: DVec3,
}

impl SurfaceModel {
    pub fn new(world: &World) -> Self {
        let s = world.seed();
        let k = |i: u64| mix64(s ^ (0xB0B0 + i * 31337));
        let look = &world.cfg.look;
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
        }
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
        PixFields::from_array(self.pixel_fields_part(p, gsd, None))
    }

    /// The pixel fields, all octaves (`split` None) or only those of wavelength >= `cut` (`split`
    /// Some((cut, true))) or < `cut` (Some((cut, false))); the two parts sum to the whole. The
    /// smooth part is interpolated from a coarse grid by the tile generator.
    pub fn pixel_fields_part(&self, p: DVec3, gsd: f64, split: Option<(f64, bool)>) -> [f64; PixFields::N] {
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
        let stand_lf = single(1200.0, &|| 0.5 * perlin3(0x57A, p / 1200.0) * crate::noise::band(1200.0, gsd));
        let stand_warp = |seed: u64| single(180.0, &|| perlin3(seed, p / 180.0));
        [
            f(&self.detail, 1.0),
            f(&self.patch, 1.0),
            f(&self.land_n, 1.0) * self.land_n.norm() * 1.8,
            f(&self.strata, 1.0),
            f(&self.strata, 1.7),
            f(&self.snow_n, 1.0),
            f(&self.forest, 1.0) * self.forest.norm() * 1.8,
            f(&self.patch, 0.3) + stand_lf,
            f(&self.field_var, 1.0),
            f(&self.field_var, 1.7),
            f(&self.field_var, 3.0),
            f(&self.warp2, 1.0),
            f(&self.patch, 0.37),
            stand_warp(0x57A1),
            stand_warp(0x57A2),
            stand_warp(0x57A3),
        ]
    }

    fn region_info(&self, world: &World, cache: &mut Caches, t: &Terrain) -> RegionInfo {
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
        // climate at the region centre decides the field style
        let tc = world.terrain(&cctx);
        let dry = 1.0 - smoothstep(0.2, 0.4, tc.moist);
        let u = u01k(id, 2);
        let style = if dry > 0.5 && u < 0.6 * dry {
            2 // centre pivots
        } else if u < 0.5 {
            0 // rectangular grid
        } else if u < 0.88 {
            1 // irregular voronoi fields
        } else {
            3 // long strips
        };
        let scale = 0.6 + 1.1 * u01k(id, 3);
        let (fw, fh) = match style {
            0 => (220.0 * scale, 220.0 * scale * (1.0 + 2.0 * u01k(id, 4))),
            1 => (300.0 * scale, 0.0),
            2 => (if u01k(id, 4) < 0.5 { 805.0 } else { 402.0 }, 0.0),
            _ => (60.0 + 90.0 * u01k(id, 4), 400.0 + 600.0 * u01k(id, 5)),
        };
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
            hedge: if u01k(id, 7) < 0.4 { u01k(id, 8) } else { 0.0 },
            track: 0.2 + 0.6 * u01k(id, 9),
            border_w: 1.5 + 3.0 * u01k(id, 10),
            palette: u01k(id, 11),
            agri: tc.agri,
            season: (tc.style[3] * 0.7 + 0.3 * u01k(id, 12)).clamp(0.0, 1.0),
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
    fn select_town(&self, world: &World, cache: &mut Caches, p: DVec3, gsd: f64, slope: f64, clear: f64, pf: &PixFields, range: i64) -> Option<(TownInfo, f64)> {
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
        let tc = world.terrain(&ctx);
        let p_exist = (tc.habit * 1.1 * world.cfg.landuse.towns).min(0.95);
        // only lattice sites within 0.8 cells of the surface make towns: a town sits below /
        // above its site, and a site farther away lay outside the ±2-cell candidate search of the
        // pixels its town covers (the town was cut along lattice-cell planes); see `select_town`
        let near_surface = (center.length() - ctx.p.length()).abs() < 0.8 * world.cfg.landuse.town_cell_km * 1000.0;
        let exists = near_surface && u01k(id, 1) < p_exist && tc.water_kind == water::NONE && tc.ground > 2.0 && tc.ground < 4000.0;
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
            block: 70.0 + 70.0 * u01k(id, 5),
            street: 6.5 + 6.0 * u01k(id, 6),
            organic: u01k(id, 7),
            roof_style: u01k(id, 8),
            height: u01k(id, 9),
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

    /// Evaluate the surface at one sub-sample.
    pub fn eval(&self, world: &World, cache: &mut Caches, ctx: &Ctx, l: &Local, pf: &PixFields) -> Surface {
        let pal = &self.pal;
        let t = l.t;
        let p = ctx.p;
        let gsd = ctx.gsd;
        let fw = l.fw;

        // ------------------------------------------------------------- standing water
        if l.water > l.ground && l.water_kind != water::NONE {
            let depth = l.water - l.ground;
            let mut col = match l.water_kind {
                water::OCEAN => mixc(pal.ocean_shallow, pal.ocean_deep, smoothstep(0.0, 28.0, depth)),
                _ => mixc(mixc(pal.river, pal.ocean_shallow, 0.25), pal.lake_deep, smoothstep(0.0, 5.0, depth)),
            };
            // sediment / plankton variation
            let v = pf.water;
            col *= 1.0 + 0.06 * v;
            if l.water_kind == water::OCEAN && depth < 3.0 {
                // the sandy bottom shows through clear shallow water
                let sh = 1.0 - smoothstep(0.0, 3.0, depth);
                col = mixc(col, mixc(pal.ocean_shallow, pal.beach, 0.55) * 1.05, 0.75 * sh * sh);
                // surf: thin broken lines of foam along the shore (swash at the water's edge,
                // breakers a little further out), not a blotchy band
                let wave = depth + 0.18 * perlin3(0x5F1, p / 40.0);
                let broken = smoothstep(-0.25, 0.35, perlin3(0x5F2, p / 22.0) + 0.5 * perlin3(0x5F3, p / 7.0));
                let line = |c: f64, w: f64| (-((wave - c) / w).powi(2)).exp();
                let foam = (0.85 * line(0.06, 0.05) + 0.6 * broken * line(0.45, 0.05) + 0.4 * broken * line(1.0, 0.06))
                    * (0.75 + 0.25 * perlin3(7, p / 3.0))
                    * band(3.0, gsd);
                col = mixc(col, srgb(225.0, 232.0, 230.0), foam.min(1.0));
            }
            let class = match l.water_kind {
                water::OCEAN => lc::OCEAN,
                _ => lc::LAKE,
            };
            return Surface { albedo: col, height: l.water, class, lit: 1.0, is_water: true, emission: DVec3::ZERO };
        }

        let slope = l.slope;
        let st = &t.style;
        // ------------------------------------------------------------- natural ground
        let detail = pf.detail; // ~[-0.7, 0.7]
        let patch = pf.patch;
        let wet = t.moist;
        let temp = t.temp;
        let soil_i = st[0] * 3.0;
        let i0 = (soil_i.floor() as usize).min(2);
        let soil = mixc(pal.soil[i0], pal.soil[i0 + 1], soil_i - i0 as f64);
        // red laterite soils in hot, wet climates
        let soil = mixc(soil, srgb(146.0, 82.0, 54.0), 0.75 * smoothstep(19.0, 25.0, temp) * smoothstep(0.45, 0.7, wet));
        let grass_green = mixc(pal.grass_dry, pal.grass_wet, smoothstep(0.25, 0.75, wet + 0.15 * patch));
        let grass = mixc(pal.grass_cold, grass_green, smoothstep(-2.0, 8.0, temp));
        // hue drift per region so neighbouring areas differ
        let grass = grass * DVec3::new(1.0 + 0.10 * (st[1] - 0.5), 1.0 + 0.06 * (st[3] - 0.5), 1.0 - 0.08 * (st[1] - 0.5));
        let land_n = pf.land;
        let cover = smoothstep(0.08, 0.45, wet + 0.25 * patch + 0.2 * land_n) * smoothstep(-9.0, -1.0, temp);
        let mut col = mixc(soil, grass, cover) * (1.0 + 0.16 * land_n);
        let mut class = if cover > 0.5 { lc::GRASS } else { lc::BARE };
        if temp < 0.0 && cover > 0.3 {
            col = mixc(col, pal.tundra, smoothstep(0.0, -6.0, temp));
            class = lc::TUNDRA;
        }
        if t.floodplain > 0.3 && wet > 0.55 {
            let m = smoothstep(0.3, 0.9, t.floodplain) * smoothstep(0.55, 0.8, wet) * smoothstep(-0.1, 0.3, patch);
            col = mixc(col, pal.marsh, m);
            if m > 0.5 {
                class = lc::WETLAND;
            }
        }
        col *= 1.0 + 0.22 * detail;
        // meadow texture: dry straw-coloured patches and mottling (tussocks, growth) at a few
        // metres to tens of metres, band-limited (the mean is unchanged at coarse zooms)
        {
            let dry_p = smoothstep(0.05, 0.55, perlin3(0x3EAD, p / 60.0) + 0.5 * perlin3(0x3EAE, p / 22.0)) * band(30.0, gsd) * cover;
            col = mixc(col, col * DVec3::new(1.16, 1.06, 0.80), 0.45 * dry_p);
            col *= 1.0 + (0.10 * perlin3(0x3EB1, p / 12.0) * band(12.0, gsd) + 0.08 * perlin3(0x3EB2, p / 4.0) * band(4.0, gsd)
                + 0.06 * perlin3(0x3EB3, p / 1.3) * band(1.3, gsd)) * cover;
        }
        // drainage lines: moister, greener, darker channels; dry bright spurs
        if t.gully != 0.0 {
            let ch = smoothstep(0.1, 0.8, -t.gully);
            col = mixc(col, mixc(col * 0.8, pal.grass_wet * 0.85, 0.5 * smoothstep(-6.0, 4.0, temp)), 0.6 * ch);
            col *= 1.0 + 0.06 * smoothstep(0.2, 1.0, t.gully);
        }

        // ------------------------------------------------------------- rock (slope + expected)
        // effective slope: at coarse zooms the resolved slope underestimates the true one, so blend
        // in the slope expected from the relief type (keeps rock cover consistent across zooms)
        let resolve = 1.0 - smoothstep(8.0, 80.0, gsd);
        let exp_slope = 0.12 + 0.75 * t.rock_expect;
        let slope_eff = lerp(exp_slope, slope.max(exp_slope * 0.6), resolve);
        let rock_n = 0.7 * patch + 0.3 * pf.land;
        let rock = smoothstep(0.55, 0.85, slope_eff + 0.25 * detail + 0.25 * rock_n + 0.2 * (t.rock_expect - 0.4) + 0.15 * t.gully)
            * (1.0 - 0.5 * smoothstep(0.3, 0.8, cover) * (1.0 - t.mountain));
        if rock > 0.0 {
            let ri = st[2] * 2.0;
            let j = (ri.floor() as usize).min(1);
            let mut rc = mixc(pal.rock[j], pal.rock[j + 1], ri - j as f64);
            // bands along the contours: their horizontal wavelength shrinks with the slope (a
            // fixed band limit aliased them into hairlines on steep valley walls)
            let strata_h = 6.0 + 10.0 * st[3];
            let strata = (l.ground / strata_h + 3.0 * pf.strata).sin();
            let strata_w = std::f64::consts::TAU * strata_h / slope.max(0.05);
            rc *= 1.0 + 0.06 * strata * band(strata_w, 1.5 * gsd) + 0.25 * detail + 0.12 * pf.strata2;
            col = mixc(col, rc, rock);
            if rock > 0.5 {
                class = lc::ROCK;
            }
        }

        // ------------------------------------------------------------- sand seas, beaches
        if t.sand > 0.0 {
            let si = st[1] * 2.0;
            let j = (si.floor() as usize).min(1);
            let sc = mixc(pal.sand[j], pal.sand[j + 1], si - j as f64) * (1.0 + 0.06 * detail);
            let s = smoothstep(0.2, 0.6, t.sand + 0.2 * patch);
            col = mixc(col, sc, s);
            if s > 0.5 {
                class = lc::SAND;
            }
        }
        // land within a few metres of sea level is coastal plain (the continent field does not
        // mark the shoreline: coasts are shaped by the relief on top of it); river floodplains
        // near sea level (deltas) are not beaches
        let coastal = 1.0 - smoothstep(0.3, 0.7, t.floodplain);
        // beaches up to ~4 m above the sea (a band tens of metres wide on flat shores) with a
        // ragged inland edge; nothing is farmed or wooded on them (fields and trees ran into the
        // water) and fields keep back from the shore behind them
        let mut beach = 0.0;
        let mut shore_keep = 1.0;
        if coastal > 0.0 {
            shore_keep = 1.0 - coastal * (1.0 - smoothstep(4.0, 8.0, l.ground + 2.0 * patch));
        }
        if l.ground < 6.0 && coastal > 0.0 && slope < 0.3 {
            let b = (1.0 - smoothstep(2.6, 4.2, l.ground + 1.0 * detail + 0.8 * patch)) * coastal * (1.0 - smoothstep(0.15, 0.3, slope));
            beach = b;
            // dry sand mottled by wind and footprints, darker wet sand only at the water's edge
            let bc = pal.beach * (1.0 + 0.05 * perlin3(0xBE1, p / 9.0) * band(9.0, gsd) + 0.04 * perlin3(0xBE2, p / 2.5) * band(2.5, gsd));
            let bc = mixc(bc, mixc(pal.beach, pal.wet_sand, 0.7), 1.0 - smoothstep(0.05, 0.3, l.ground + 0.08 * perlin3(0xBE3, p / 15.0)));
            col = mixc(col, bc, b);
            if b > 0.5 {
                class = lc::BEACH;
            }
        }

        // ------------------------------------------------------------- snow
        let snow_n = pf.snow;
        // snow cover with crisp, ragged edges (a transition over several degrees looked like
        // cloud or fog lying on the land)
        // follows the terrain: a snow line in altitude (temperature), lingering longer in gullies
        // and hollows; noise only roughens it (a strong noise term drew blobs unrelated to the land)
        let snow_t = temp + 1.0 * snow_n - 1.6 * smoothstep(0.1, 0.8, -t.gully) + 0.6 * smoothstep(0.2, 0.8, t.gully)
            + 0.5 * perlin3(0x5E0, p / 60.0) * band(60.0, gsd) + 0.25 * perlin3(0x5E1, p / 18.0) * band(18.0, gsd);
        let snow = smoothstep(-2.6, -2.8, snow_t) * (1.0 - 0.75 * smoothstep(0.9, 1.6, slope));
        if snow > 0.0 {
            col = mixc(col, pal.snow * (1.0 + 0.03 * detail), snow);
            if snow > 0.5 {
                class = lc::SNOW;
            }
        }

        // micro-relief of the ground: hummocks, tussocks and bumps (decimetres over metres),
        // band-limited; fields get a smoother, tilled surface (applied below)
        let micro_relief = (0.30 * perlin3(0x9A01, p / 9.0) * band(9.0, gsd)
            + 0.14 * perlin3(0x9A02, p / 3.2) * band(3.2, gsd)
            + 0.06 * perlin3(0x9A03, p / 1.1) * band(1.1, gsd))
            * (1.0 - 0.5 * rock)
            * (1.0 - 0.7 * beach);
        let mut height = l.ground + micro_relief;
        let mut lit: f64 = 1.0;
        let mut emission = DVec3::ZERO;
        let natural_ok = (1.0 - rock) * (1.0 - snow) * (1.0 - t.sand) * (1.0 - beach);

        // local planar frame of the land-use region
        let have_region = t.region.id != 0;
        let region = if have_region { Some(self.region_info(world, cache, t)) } else { None };
        let (q_loc, q_rot) = if let Some(r) = &region {
            let d = p - r.center;
            (DVec2::new(d.dot(r.east), d.dot(r.north)), DVec2::new(d.dot(r.ex), d.dot(r.ey)))
        } else {
            (DVec2::new(p.dot(ctx.east), p.dot(ctx.north)), DVec2::ZERO)
        };

        // riparian belt along rivers
        let riparian = if l.river_hw > 0.0 && t.river_wet > 0.3 {
            let ad = l.river_d.abs();
            let belt = 4.0 + 0.6 * l.river_hw.min(60.0);
            (1.0 - smoothstep(l.river_hw + 0.3 * belt, l.river_hw + belt, ad))
                * smoothstep(0.35, 0.6, t.moist)
                * smoothstep(0.3, 0.7, t.river_wet)
                * natural_ok
                * (0.55 + 0.45 * smoothstep(-0.3, 0.3, pf.patch))
        } else {
            0.0
        };

        // ------------------------------------------------------------- woodlots
        // farmland keeps forest on its most forest-prone patches (low values of the forest
        // pattern, up to ~1/7 of the land, less in dry climates): areas of woodland between the
        // fields, not only the riparian strips and hedgerows
        let woodlot = {
            let fpu = 0.5 + 0.5 * pf.forest;
            let cover = smoothstep(0.12, 0.45, t.moist.max(0.0)) * smoothstep(-6.0, 2.0, t.temp) * world.cfg.vegetation.tree_density;
            smoothstep(-0.02, 0.02, 0.34 * cover - fpu) * natural_ok
        };

        // ------------------------------------------------------------- fields
        // fields on gentle to moderate slopes (up to ~17°): a lower limit turned every scarp,
        // terrace edge and gully wall in farmland into a thin strip of forest
        let flat_ok = 1.0 - smoothstep(0.22, 0.32, slope);
        let mut field_cov = 0.0;
        if let Some(r) = &region {
            if t.agri > 0.02 && natural_ok * flat_ok > 0.3 && world.cfg.landuse.agriculture > 0.0 {
                if let Some((fcol, fh, cov, edge_kind)) = self.field(cache, r, t, q_rot, p, gsd, fw, pf) {
                    // a field is there or not: a crisp (noisy) cutoff instead of fading fields out
                    // over gentle valley sides, which left washed, half-transparent bands
                    let keep = natural_ok * flat_ok * (1.0 - riparian) * (1.0 - woodlot) * shore_keep;
                    // (constant threshold: a noisy one left specks of open ground inside fields,
                    // where trees clipped to a pixel or two became spikes)
                    let a = cov * smoothstep(0.45, 0.55, keep);
                    col = mixc(col, fcol, a);
                    // tilled fields are smoother than natural ground
                    height += fh * a - 0.6 * micro_relief * a;
                    field_cov = a;
                    if a > 0.5 {
                        class = if edge_kind == 1 { lc::FOREST } else if edge_kind == 2 { lc::ROAD } else { lc::CROP };
                    }
                }
            }
        }

        // no trees anywhere in a town's footprint, only a short fade at its outer edge (forest
        // bands ran through towns, and trees between the lots of the sparse outskirts cut into
        // roofs and streets). The town is evaluated here (drawn below, over the roads) so that its
        // lots and streets also mask the trees in that fade.
        // Every existing town of the surrounding lattice cells is tried and the most built-up one
        // wins: a large town reaches beyond its own cell, where it was cut off along straight
        // lines (taking only the nearest site, or the two nearest, did that)
        // towns end at the bank of a river (with a riverside strip), not under the water
        let river_clear = if l.river_hw > 0.0 {
            // a narrow bank (a few metres plus a tenth of the width), not a wide green strip
            let bank = 2.0 + 0.1 * l.river_hw;
            1.0 - (1.0 - smoothstep(l.river_hw + bank, l.river_hw + 2.0 * bank + 2.0, l.river_d.abs())) * smoothstep(0.3, 0.6, t.river_wet)
        } else {
            1.0
        };
        // towns avoid steep relief, judged from the relief type (smooth), not the slope of each
        // pixel: that cut houses in half along every terrace edge and gully wall
        let town_slope = 0.12 + 0.75 * t.rock_expect;
        let town_sel = if t.town.id != 0 && world.cfg.landuse.towns > 0.0 {
            self.select_town(world, cache, p, gsd, town_slope, river_clear, pf, 2)
        } else {
            None
        };
        let town_urban = town_sel.map_or(0.0, |x| x.1);
        let town_px = town_sel.and_then(|(town, _)| self.town(&town, p, gsd, fw, town_slope, river_clear, world.cfg.look.shadows, pf));
        let town_cov = town_px.map_or(0.0, |x| x.2);
        let not_urban = (1.0 - smoothstep(0.0, 0.08, town_urban)) * (1.0 - town_cov);

        // ------------------------------------------------------------- trees
        let veg = &world.cfg.vegetation;
        if veg.tree_density > 0.0 && natural_ok * not_urban > 0.05 {
            let base_cover = smoothstep(0.3, 0.68, wet) * smoothstep(-6.0, 2.0, temp) * veg.tree_density;
            let fpat = pf.forest; // ~[-1,1]
            let fpu = 0.5 + 0.5 * fpat;
            // forests where the patch field is below the cover fraction (crisp but noisy edges)
            let edge = 0.03 + 0.6 * band(30.0, gsd).min(1.0) * 0.0;
            let forest = smoothstep(-edge, edge, base_cover - fpu) ;
            // savanna / steppe scattered trees
            let savanna = smoothstep(0.18, 0.35, wet) * (1.0 - smoothstep(0.55, 0.7, wet)) * smoothstep(12.0, 20.0, temp) * 0.12;
            let groves = 0.04 * smoothstep(0.15, 0.3, wet);
            // farmland is cleared except on steep ground (and its woodlots): what cannot be a
            // field there is meadow, not forest
            let clear = 1.0 - 0.85 * smoothstep(0.05, 0.4, t.agri) * (1.0 - smoothstep(0.45, 0.7, slope)) * (1.0 - woodlot);
            let mut dens = (forest * 0.9 * clear + savanna + groves) * natural_ok * (1.0 - field_cov) * veg.tree_density;
            dens = dens.max(0.8 * riparian * veg.tree_density);
            // drainage lines carry scrub, not trees: the lines are only metres wide, so tree
            // crowns there were clipped into slivers (spikes in the DSM) and drew curving tree
            // lines across the land
            let gully_scrub = 0.55 * smoothstep(0.2, 0.9, -t.gully) * smoothstep(0.2, 0.5, wet) * natural_ok * (1.0 - field_cov);
            dens *= 1.0 - smoothstep(0.9, 1.4, slope);
            dens *= 1.0 - smoothstep(0.0, 0.6, t.mountain * smoothstep(-2.0, -6.0, temp)); // tree line
            // no trees standing in the snow: they end below the snow line
            dens *= 1.0 - smoothstep(-1.2, -2.4, temp + 1.0 * snow_n);
            dens *= not_urban;
            // shrubs / bushes in steppe, maquis and rocky slopes (texture of natural ground)
            let shrub_clim = smoothstep(0.15, 0.3, wet) * (1.0 - smoothstep(0.6, 0.8, wet)) * smoothstep(2.0, 10.0, temp);
            let shrub_patch = smoothstep(-0.2, 0.5, patch + 0.4 * pf.land);
            let shrub = ((0.45 * shrub_clim * shrub_patch * (1.0 - forest) * natural_ok.max(0.4 * rock) * (1.0 - field_cov) + gully_scrub) * veg.tree_density * not_urban)
                .clamp(0.0, 0.7);
            // forest stands (~240 m, irregular borders): each of its own age (crown size, height),
            // tone and conifer / broadleaf mix, with small canopy gaps; one lattice of identical
            // crowns read as a uniform camouflage texture
            let stand_id = pf.stand_id.unwrap_or_else(|| self.stand_id(p, Some(pf.stand_warp)));
            let age = u01k(stand_id, 1);
            let tone_u = u01k(stand_id, 2);
            let stand_tone = mixc(DVec3::new(0.86, 0.93, 0.92), DVec3::new(1.12, 1.08, 0.88), tone_u) * (0.92 + 0.12 * age);
            let gap = smoothstep(0.3, 0.6, perlin3(0x6A9, p / 30.0) + 0.5 * perlin3(0x6AA, p / 11.0)) * (0.2 + 0.8 * u01k(stand_id, 4));
            dens *= 1.0 - 0.9 * gap * smoothstep(0.3, 0.7, dens);
            if dens > 0.0 || shrub > 0.01 {
                // conifers in proper stands; lone and scattered trees in open land are broadleaf
                // (lower, wider crowns), not needle-thin spruces
                let stand_d = smoothstep(0.3, 0.75, dens);
                let conifer = (1.0 - smoothstep(4.0, 13.0, temp)) * stand_d.max(1.0 - smoothstep(-5.0, 1.0, temp));
                // mixed forests: stands lean conifer or broadleaf
                let conifer = (conifer + 0.9 * (u01k(stand_id, 3) - 0.5) * (1.0 - (2.0 * conifer - 1.0).abs())).clamp(0.0, 1.0);
                let scale = 0.7 + 0.55 * age;
                let tropic = smoothstep(19.0, 25.0, temp) * smoothstep(0.55, 0.75, wet);
                let dry = 1.0 - smoothstep(0.3, 0.5, wet);
                let tall = 0.5 + 0.5 * st[3];
                let layers = [
                    TreeLayer { cell: 5.5, seed: 0x7EE1, density: dens * conifer, closure: dens, height: 14.0 + 10.0 * tall, color: pal.crown_conifer, conifer: 1.0, scale, tone: stand_tone },
                    TreeLayer {
                        cell: 8.5,
                        seed: 0x7EE2,
                        density: dens * (1.0 - conifer) * (1.0 - tropic),
                        closure: dens,
                        height: (10.0 + 10.0 * tall) * (0.65 + 0.35 * stand_d),
                        color: mixc(pal.crown_decid, pal.crown_dry, dry),
                        conifer: 0.0,
                        scale,
                        tone: stand_tone,
                    },
                    TreeLayer { cell: 13.0, seed: 0x7EE3, density: dens * tropic, closure: dens, height: 22.0 + 14.0 * tall, color: pal.crown_tropic, conifer: 0.0, scale, tone: stand_tone },
                    TreeLayer { cell: 3.2, seed: 0x7EE4, density: shrub, closure: 0.0, height: 1.6, color: pal.shrub, conifer: 0.0, scale: 1.0, tone: DVec3::ONE },
                ];
                // forest floor: shaded litter and understory, not sunlit grass, between the crowns
                let floor = smoothstep(0.25, 0.8, dens);
                col = mixc(col, mixc(pal.crown_conifer, pal.soil[0], 0.45) * 0.7, floor * 0.85);
                let (mut tc, th, tcov) = self.trees(&layers, q_loc, gsd, fw, p);
                if tcov > 0.0 {
                    // forest stands of different age / species composition
                    let stand = pf.stand;
                    tc *= DVec3::new(1.0 + 0.10 * stand, 1.0 + 0.14 * stand, 1.0 + 0.05 * stand);
                    col = mixc(col, tc, tcov);
                    if veg.trees_in_dsm {
                        height = height.max(l.ground + th);
                    }
                    if tcov > 0.5 {
                        class = lc::FOREST;
                    }
                }
                // cast shadows from neighbouring trees onto the ground/other crowns
                if world.cfg.look.shadows && tcov < 0.99 {
                    let mut shadow = 0.0;
                    for layer in &layers {
                        if layer.density <= 0.0 || layer.cell < 2.0 * gsd {
                            continue;
                        }
                        let off = self.sun_h * (0.6 * layer.height / self.sun_tan);
                        let (_, _, sc) = self.trees(std::slice::from_ref(layer), q_loc + off, gsd, fw, p);
                        shadow = f64::max(shadow, sc);
                    }
                    lit = lit.min(1.0 - 0.9 * shadow * (1.0 - tcov));
                }
                // dark understory under sparse prefiltered forest
                if tcov < 0.01 && dens > 0.0 {
                    col *= 1.0 - 0.15 * dens;
                }
            }
        }

        // ------------------------------------------------------------- roads
        let roads = world.cfg.landuse.roads;
        let mut road_major_cov: f64 = 0.0;
        if roads > 0.0 {
            // a road is there or not (crisp cutoffs): roads fading with habitation or slope were
            // half-transparent ghosts with the trees showing through
            let steep = smoothstep(-0.02, 0.02, 0.5 - slope);
            let habit = smoothstep(-0.005, 0.005, t.habit - 0.03) * steep * (1.0 - snow) * (1.0 - t.sand * 0.7);
            let mut road_cov: f64 = 0.0;
            let mut road_col = pal.asphalt;
            if habit > 0.0 {
                let w_major = 12.0;
                let c1 = band_cov(l.road_major, w_major * 0.5, fw.max(gsd * 0.5));
                if c1 > 0.0 {
                    road_cov = c1 * habit;
                    road_major_cov = road_cov;
                    // lighter shoulders
                    let sh = band_cov(l.road_major, w_major * 0.5 + 1.5, fw) - band_cov(l.road_major, w_major * 0.5, fw);
                    road_col = mixc(pal.asphalt, pal.concrete, sh.max(0.0) * 0.6);
                }
                let w_minor = 6.0;
                let c2 = band_cov(l.road_minor, w_minor * 0.5, fw.max(gsd * 0.5)) * habit;
                if c2 > road_cov {
                    road_cov = c2;
                    road_col = mixc(pal.asphalt, pal.gravel, smoothstep(0.4, 0.7, st[0]));
                }
            }
            // farm tracks along land-use region borders
            if let Some(r) = &region {
                if r.agri > 0.1 && t.agri > 0.05 {
                    let pair = t.region.id ^ t.region.id2;
                    if u01k(pair, 3) < 0.7 {
                        let c3 = band_cov(t.region.edge, 3.0, fw.max(gsd * 0.5)) * steep;
                        if c3 > road_cov {
                            road_cov = c3;
                            road_col = if u01k(pair, 4) < 0.5 { pal.gravel } else { pal.asphalt };
                        }
                    }
                }
            }
            if road_cov > 0.0 {
                let rc = road_col * (1.0 + 0.05 * detail);
                col = mixc(col, rc, road_cov);
                height = lerp(height, l.ground, road_cov);
                lit = lerp(lit, 1.0, road_cov * 0.5);
                if road_cov > 0.5 {
                    class = lc::ROAD;
                }
            }
        }

        // ------------------------------------------------------------- farmsteads
        if let Some(r) = &region {
            if t.agri > 0.06 && flat_ok > 0.3 && natural_ok > 0.3 && world.cfg.landuse.towns > 0.0 {
                if let Some((fcol, fh, fcov, fcls, fem)) = self.farmstead(r, t, q_rot, gsd, fw) {
                    col = mixc(col, fcol, fcov);
                    if world.cfg.landuse.buildings_in_dsm {
                        height = lerp(height, l.ground + fh, fcov);
                    }
                    emission += fem;
                    if fcov > 0.5 {
                        class = fcls;
                    }
                }
            }
        }

        // ------------------------------------------------------------- lit main roads near towns
        if road_major_cov > 0.0 {
            if let Some((town, _)) = town_sel {
                let dist = (p - town.center).length();
                let near = 1.0 - smoothstep(1.2 * town.radius, 2.2 * town.radius, dist);
                let sp = 38.0;
                let res = band(sp, gsd);
                if near > 0.0 {
                    let ql = (q_loc / sp).round() * sp;
                    let d2 = (q_loc - ql).length_squared();
                    let lh = hash2(town.seed ^ 0x40AD, (ql.x / sp) as i64, (ql.y / sp) as i64);
                    let lamp_col = if u01k(lh, 1) < 0.5 { DVec3::new(1.0, 0.48, 0.12) } else { DVec3::new(0.86, 0.92, 1.0) };
                    let pool = (0.04 * (-d2 / (2.0 * 6.0 * 6.0)).exp() + point_light(d2, 6.0, 0.4, fw)) * res + 0.04 * (1.0 - res);
                    emission += lamp_col * pool * near * road_major_cov;
                }
            }
        }

        // ------------------------------------------------------------- towns
        // embankment lamps: a string of lights along the banks of rivers through towns (the
        // river was a long dark gap in the lit town)
        if town_urban > 0.25 && l.river_hw > 0.0 && t.river_wet > 0.5 {
            let bank = 2.0 + 0.1 * l.river_hw;
            let dl = l.river_d.abs() - (l.river_hw + 0.5 * bank);
            let dots = smoothstep(0.35, 0.6, perlin3(0xE3B, p / 4.0)) * band(4.0, gsd) + 0.3 * (1.0 - band(4.0, gsd));
            emission += DVec3::new(1.0, 0.80, 0.55) * (4.0 * (-(dl * dl) / (2.0 * 0.5 * 0.5)).exp() * dots * smoothstep(0.25, 0.45, town_urban));
        }
        if let Some((tcol, th, cov, cls, shadow, em)) = town_px {
            emission += em * cov;
            col = mixc(col, tcol, cov);
            if world.cfg.landuse.buildings_in_dsm {
                height = lerp(height, l.ground + th, cov);
            }
            lit = lit.min(1.0 - shadow);
            if cov > 0.5 {
                class = cls;
            }
        }

        // ------------------------------------------------------------- rivers (on top)
        if l.river_hw > 0.0 {
            let fwr = fw.max(gsd * 0.35);
            let cov = band_cov(l.river_d, l.river_hw, fwr);
            if cov > 0.0 {
                let wet_r = t.river_wet;
                let mut wcol = mixc(pal.river, pal.lake_deep, smoothstep(30.0, 200.0, l.river_hw * 2.0));
                // mountain rivers carry glacial flour: milky turquoise, not deep dark water
                wcol = mixc(wcol, srgb(96.0, 138.0, 140.0), 0.7 * t.mountain * smoothstep(8.0, 0.0, temp));
                // in the cold the river freezes over and is snowed on (a dark channel through the
                // snowfields read as a crack)
                let ice = smoothstep(-1.5, -4.0, temp + 1.5 * snow_n);
                wcol = mixc(wcol, mixc(pal.snow * 0.9, srgb(170.0, 190.0, 200.0), 0.35 * (0.5 + 0.5 * detail)), ice);
                // dry beds are only a subtle pale line (gravel / sand with some vegetation)
                let dry_col = mixc(col, mixc(pal.gravel, pal.sand[2], 0.5) * (1.0 + 0.1 * detail), 0.55);
                let rc = mixc(dry_col, wcol, wet_r);
                col = mixc(col, rc, cov);
                height = lerp(height, l.river_level, cov);
                lit = lerp(lit, 1.0, cov);
                if cov > 0.5 && ice > 0.5 && wet_r > 0.5 {
                    class = lc::SNOW;
                } else if cov > 0.5 {
                    if wet_r > 0.5 {
                        return Surface { albedo: col, height, class: lc::RIVER, lit, is_water: true, emission };
                    }
                    class = lc::SAND;
                }
            }
        }

        Surface { albedo: col.max(DVec3::ZERO), height, class, lit, is_water: false, emission }
    }

    /// Farmstead: a small cluster (house, barn, gravel yard, yard lamp) on a sparse lattice in the
    /// agricultural regions. Returns (colour, height, coverage, class, emission).
    fn farmstead(&self, r: &RegionInfo, t: &Terrain, q: DVec2, gsd: f64, fw: f64) -> Option<(DVec3, f64, f64, u8, DVec3)> {
        let pal = &self.pal;
        let wc = worley2(r.split.to_bits() ^ 0xFA4, q, 650.0, 0.8);
        let fid = wc.id;
        if u01k(fid, 1) > 0.55 * smoothstep(0.05, 0.4, t.agri) {
            return None;
        }
        let rel0 = q - wc.point;
        if rel0.length() > 60.0 {
            return None;
        }
        let ang = (u01k(fid, 2) - 0.5) * 0.4;
        let (sa, ca) = ang.sin_cos();
        let rel = DVec2::new(rel0.x * ca + rel0.y * sa, -rel0.x * sa + rel0.y * ca);
        let fwe = fw.max(0.3 * gsd);
        let boxc = |c: DVec2, half: DVec2| -> f64 {
            let d = (half - (rel - c).abs()).min_element();
            (d / fwe + 0.5).clamp(0.0, 1.0)
        };
        let yard_half = DVec2::new(20.0 + 10.0 * u01k(fid, 3), 14.0 + 8.0 * u01k(fid, 4));
        let yard = boxc(DVec2::ZERO, yard_half);
        // lamp light pool (also lights the surrounding field a little)
        let lamp = rel - DVec2::new(2.0, 6.0);
        let d2 = lamp.length_squared();
        let lamp_col = if u01k(fid, 9) < 0.6 { DVec3::new(1.0, 0.72, 0.38) } else { DVec3::new(0.86, 0.92, 1.0) };
        let res = band(10.0, gsd);
        let emission = lamp_col * (0.04 * (-d2 / (2.0 * 7.0 * 7.0)).exp() + point_light(d2, 5.0, 0.4, fw)) * res
            + lamp_col * 0.02 * (1.0 - res) * (1.0 - smoothstep(20.0, 60.0, rel.length()));
        if yard <= 0.0 {
            return if emission.max_element() > 1e-4 { Some((DVec3::ZERO, 0.0, 0.0, lc::CROP, emission)) } else { None };
        }
        let mut col = mixc(pal.gravel, pal.concrete, 0.3 * u01k(fid, 5)) * (0.9 + 0.2 * u01k(fid, 6));
        let mut h = 0.0;
        let mut class = lc::URBAN;
        // house (pitched roof) and barn (long, low-pitched metal roof)
        let hc = DVec2::new(-yard_half.x * 0.45, -yard_half.y * 0.3);
        let hh = DVec2::new(5.5, 4.5);
        let house = boxc(hc, hh);
        if house > 0.0 {
            let roof = pal.roofs[(u01k(fid, 7) * 2.99) as usize] * (0.9 + 0.2 * u01k(fid, 8));
            let ridge = ((hh.y - (rel.y - hc.y).abs()) / hh.y).clamp(0.0, 1.0);
            col = mixc(col, roof, house);
            h = (5.5 + 2.0 * ridge) * house;
            class = lc::BUILDING;
        }
        let bc = DVec2::new(yard_half.x * 0.3, yard_half.y * 0.35);
        let bh = DVec2::new(11.0 + 5.0 * u01k(fid, 10), 6.0);
        let barn = boxc(bc, bh);
        if barn > 0.0 {
            let roof = mixc(pal.roofs[4], pal.roofs[2], u01k(fid, 11));
            let ridge = ((bh.y - (rel.y - bc.y).abs()) / bh.y).clamp(0.0, 1.0);
            col = mixc(col, roof, barn);
            h = h.max((6.0 + 1.5 * ridge) * barn);
            class = lc::BUILDING;
        }
        let cov = yard * smoothstep(1.0 * gsd, 3.0 * gsd, 30.0).max(0.25);
        Some((col, h, cov, class, emission))
    }

    /// Tree crowns of several layers at local position `q`. Returns (colour, canopy height, coverage).
    fn trees(&self, layers: &[TreeLayer], q: DVec2, gsd: f64, fw: f64, p: DVec3) -> (DVec3, f64, f64) {
        let mut best_h = 0.0f64;
        let mut best_col = DVec3::ZERO;
        let mut cov_total = 0.0f64;
        let mut mean_col = DVec3::ZERO;
        let mut mean_w = 0.0;
        let mut mean_h = 0.0;
        for layer in layers {
            if layer.density <= 1e-4 {
                continue;
            }
            let explicit = smoothstep(1.2 * gsd, 3.0 * gsd, layer.cell);
            // expected coverage when unresolved
            let mean_r = layer.cell * 0.48;
            let cov_mean = (layer.density * std::f64::consts::PI * mean_r * mean_r / (layer.cell * layer.cell)).min(1.0);
            if explicit < 1.0 {
                let w = cov_mean * (1.0 - explicit);
                // stochastic canopy texture (gaps, crown clusters) at scales just above the pixel
                let tex = perlin3(layer.seed, p / (layer.cell * 3.0)) * band(layer.cell * 3.0, gsd)
                    + 0.6 * perlin3(layer.seed ^ 5, p / (layer.cell * 9.0)) * band(layer.cell * 9.0, gsd);
                mean_col += layer.color * layer.tone * (0.85 + 0.3 * tex) * w;
                mean_w += w;
                mean_h += layer.height * 0.6 * cov_mean * (1.0 - explicit);
            }
            if explicit <= 0.0 {
                continue;
            }
            let cq = q / layer.cell;
            let cf = cq.floor();
            let (ix, iy) = (cf.x as i64, cf.y as i64);
            let fq = cq - cf;
            // crowns of cells farther than the largest crown radius (+ the filter width) cannot
            // reach the sample: skipped before hashing
            let reach = (0.6 * 1.55 * layer.scale.max(0.0) + fw / layer.cell) * (1.0 + 1e-9) + 1e-9;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (gx, gy) = (crate::noise::nb_gap(fq.x, dx, 0.4), crate::noise::nb_gap(fq.y, dy, 0.4));
                    if gx * gx + gy * gy > reach * reach {
                        continue;
                    }
                    let h = hash2(layer.seed, ix + dx, iy + dy);
                    let u = u01k(h, 3);
                    if u >= layer.density {
                        continue;
                    }
                    // density is evaluated per pixel: where it falls (stand / field edges) a crown
                    // would be clipped into a wall or a spike; fade it out with the margin instead
                    let fade = smoothstep(0.0, 0.3, (layer.density - u) / layer.density.max(1e-6));
                    let c = DVec2::new(
                        (ix + dx) as f64 + 0.5 + 0.8 * (u01k(h, 1) - 0.5),
                        (iy + dy) as f64 + 0.5 + 0.8 * (u01k(h, 2) - 0.5),
                    ) * layer.cell;
                    // crowns grow with the stand density: dense forest closes its canopy
                    let r = layer.cell * (0.36 + 0.24 * u01k(h, 4)) * (1.0 + 0.55 * smoothstep(0.35, 0.9, layer.closure)) * layer.scale;
                    let d = (q - c).length();
                    if d > r + fw {
                        continue;
                    }
                    let cov = ((r - d) / fw + 0.5).clamp(0.0, 1.0) * explicit * fade;
                    let x = (d / r).min(1.0);
                    let hh = layer.height * (0.55 + 0.9 * u01k(h, 5)) * (0.45 + 0.55 * layer.scale);
                    // crown surfaces taper towards the ground (cone / dome): a tall vertical wall at
                    // the crown rim made every tree a column, seen as a spike from low angles
                    let prof = if layer.conifer > 0.5 { 0.08 + 0.92 * (1.0 - x).powf(1.15) } else { 0.12 + 0.88 * (1.0 - x * x).sqrt() };
                    // foliage clumps: a bumpy crown surface gives the shading texture (a smooth cone or
                    // dome shades as a glossy blob)
                    let clump = perlin3(h ^ 0xC1, p / 1.6) * band(1.6, gsd) + 0.5 * perlin3(h ^ 0xC2, p / 0.8) * band(0.8, gsd);
                    let th = hh * prof * (1.0 + 0.10 * clump * (0.3 + 0.7 * (1.0 - x))) * fade;
                    if th > best_h {
                        best_h = th;
                        // neighbouring crowns of a stand look alike (strong per-tree tints made
                        // the canopy a camouflage pattern); the stand tone carries the variety
                        let tint = 0.86 + 0.24 * u01k(h, 6);
                        let hv = u01k(h, 7) - 0.5;
                        let hue = DVec3::new(1.0 + 0.16 * hv, 1.0 + 0.04 * (u01k(h, 9) - 0.5), 1.0 - 0.12 * hv);
                        // leafy mottling inside the crown (clumps of foliage, ~1 m)
                        let leaf = 0.82 + 0.36 * (0.5 + 0.5 * perlin3(h, p / 1.1)) * band(1.1, gsd);
                        // crowns are a bit darker at the rim (self-shading inside the crown)
                        best_col = layer.color * layer.tone * tint * hue * leaf * (0.75 + 0.35 * (1.0 - x));
                    }
                    cov_total = cov_total.max(cov);
                }
            }
        }
        // combine explicit and mean contributions
        let mean_cov = mean_w.min(1.0);
        let mc = if mean_w > 0.0 { mean_col / mean_w } else { DVec3::ZERO };
        let cov = (cov_total + mean_cov * (1.0 - cov_total)).min(1.0);
        if cov <= 0.0 {
            return (DVec3::ZERO, 0.0, 0.0);
        }
        let col = if cov_total > 0.0 { mixc(mc, best_col, cov_total / cov) } else { mc };
        let h = best_h.max(mean_h);
        (col, h, cov)
    }

    /// Agricultural field at rotated local coords. Returns (colour, extra height, coverage, kind)
    /// where kind 0 = crop, 1 = hedge, 2 = track.
    #[allow(clippy::too_many_arguments)]
    fn field(&self, cache: &mut Caches, r: &RegionInfo, t: &Terrain, q: DVec2, p: DVec3, gsd: f64, fw: f64, pf: &PixFields) -> Option<(DVec3, f64, f64, u8)> {
        let pal = &self.pal;
        let cult = smoothstep(0.02, 0.45, t.agri);
        let tint = DVec3::new(1.0 + 0.08 * (r.palette - 0.5), 1.0, 1.0 - 0.06 * (r.palette - 0.5));
        // Always evaluate the actual field; when fields approach the pixel size, their contrast is
        // reduced towards the mean, mimicking the variance reduction of box-filtering a mosaic.
        let fsize = if r.fh > 0.0 { r.fw.min(r.fh) } else { r.fw };
        let k = fsize / (fsize * fsize + 4.0 * gsd * gsd).sqrt();
        let tropic = smoothstep(19.0, 25.0, t.temp) * smoothstep(0.5, 0.7, t.moist);
        let mean = mixc(pal.crop_mean, srgb(78.0, 100.0, 58.0), tropic) * tint * (1.0 + 0.10 * pf.field_var);
        let fwe = fw.max(0.6 * gsd); // edge filter never sharper than ~half a pixel
        let (c, h, cov, kind) = self.field_explicit(cache, r, t, q, p, gsd, fwe, cult, tint, pf).unwrap_or((mean, 0.0, 0.0, 0));
        let cult_mean = cult * 0.9;
        let cov_m = lerp(cult_mean, cov, k);
        if cov_m <= 0.0 {
            return None;
        }
        let col = if cov > 0.0 { mixc(mean, c, k) } else { mean };
        Some((col, h * k, cov_m, kind))
    }

    /// Contiguous cultivation zones: a smooth mask (compared with the cultivated fraction).
    fn cultivated_mask(&self, c3: DVec3, gsd: f64) -> f64 {
        0.5 + 0.5 * self.cult_n.eval(c3, gsd.min(100.0)) * self.cult_n.norm() * 2.2
    }

    #[allow(clippy::too_many_arguments)]
    fn field_explicit(&self, cache: &mut Caches, r: &RegionInfo, t: &Terrain, q: DVec2, p: DVec3, gsd: f64, fw: f64, cult: f64, tint: DVec3, pf: &PixFields) -> Option<(DVec3, f64, f64, u8)> {
        let pal = &self.pal;
        // field id, within-field coords (along, across), distance to boundary
        let (id, fx, fy, edge, inside, fc) = match r.style {
            0 | 3 => {
                let wq = DVec2::new(perlin3(r.split.to_bits(), p / 900.0), perlin3(r.split.to_bits() ^ 1, p / 900.0));
                let q = q + wq * (0.08 * r.fw);
                let (w, h) = (r.fw, r.fh.max(r.fw));
                let seed = r.split.to_bits();
                // rows with jittered heights; within a row, field boundaries are jittered lines
                // (widths vary ~0.55-1.45 w) so the grid does not look like a regular quilt
                let row_line = |j: i64| (j as f64 + 0.32 * (u01(hash2(seed ^ 0x40, 7, j)) - 0.5)) * h;
                let mut j = (q.y / h).floor() as i64;
                if q.y < row_line(j) {
                    j -= 1;
                } else if q.y >= row_line(j + 1) {
                    j += 1;
                }
                let (y0, y1) = (row_line(j), row_line(j + 1));
                let shift = u01(hash1(seed, j)) * w;
                let x = q.x + shift;
                let col_line = |i: i64| (i as f64 + 0.45 * (u01(hash2(seed ^ 0x41, j, i)) - 0.5)) * w;
                let mut i = (x / w).floor() as i64;
                if x < col_line(i) {
                    i -= 1;
                } else if x >= col_line(i + 1) {
                    i += 1;
                }
                let (x0, x1) = (col_line(i), col_line(i + 1));
                let (cw, ch) = (x1 - x0, y1 - y0);
                let fx = x - x0;
                let fy = q.y - y0;
                // split some cells into strips
                let hc = hash2(seed ^ 0x55, i, j);
                let nstrip = 1 + (u01k(hc, 1) * if r.style == 3 { 1.0 } else { 3.5 }) as i64;
                let sh = ch / nstrip as f64;
                let k = (fy / sh).floor().min(nstrip as f64 - 1.0);
                let fy2 = fy - k * sh;
                let edge = fx.min(cw - fx).min(fy2).min(sh - fy2);
                let fc = DVec2::new(x0 + 0.5 * cw - shift, y0 + k * sh + 0.5 * sh);
                (mix64(hc ^ k as u64), fx, fy2, edge, 1.0, fc)
            }
            1 => {
                // irregular fields: Voronoi cells in a stretched frame (elongated fields), cell
                // size modulated across the region
                let aspect = 1.0 + 2.2 * u01k(r.split.to_bits(), 21);
                let qs = DVec2::new(q.x / aspect, q.y);
                let scale = 0.7 + 0.6 * (0.5 + 0.5 * perlin3(r.split.to_bits() ^ 0x5CA1, p / 2500.0));
                let cell = r.fw / aspect.sqrt() * scale;
                let wc = worley2(r.split.to_bits(), qs, cell, 0.85);
                // edge distance back in metres: the bisector normal is stretched by the aspect
                let dn = wc.point2 - wc.point;
                let nrm = DVec2::new(dn.x / aspect, dn.y).length() / dn.length().max(1e-9);
                let e = worley2_edge_dist(&wc, qs) / nrm.max(1e-6) * 1.0;
                let rel = q - DVec2::new(wc.point.x * aspect, wc.point.y);
                (wc.id, rel.x, rel.y, e, 1.0, DVec2::new(wc.point.x * aspect, wc.point.y))
            }
            _ => {
                let s = r.fw;
                let cq = (q / s).floor();
                let c = (cq + 0.5) * s;
                let rel = q - c;
                let d = rel.length();
                let rad = s * 0.48;
                let hc = hash2(r.split.to_bits(), cq.x as i64, cq.y as i64);
                let ins = band_cov(d, rad, fw);
                // pie sectors
                let nsec = 1 + (u01k(hc, 1) * 3.0) as i64;
                let ang = rel.y.atan2(rel.x) + std::f64::consts::PI + u01k(hc, 2) * 6.0;
                let sec = ((ang / std::f64::consts::TAU * nsec as f64).floor() as i64).rem_euclid(nsec);
                let edge_c = (rad - d).abs();
                (mix64(hc ^ sec as u64), rel.x, rel.y, edge_c, ins, c)
            }
        };
        if inside <= 0.0 {
            return None;
        }
        // is this field cultivated?
        let c3 = r.center + r.ex * fc.x + r.ey * fc.y;
        // both depend on the field only (the mask's shortest octave is resolved at any gsd ≤
        // 100 m), so they are kept per field instead of being evaluated for every sample;
        // neighbouring fields often grow the same crop: drawn from a coarse crop-cluster cell
        let (mask, cluster) = *cache.fields.entry([c3.x.to_bits(), c3.y.to_bits(), c3.z.to_bits(), r.split.to_bits()]).or_insert_with(|| {
            (self.cultivated_mask(c3, gsd), worley2(r.split.to_bits() ^ 0xC1C, fc, 700.0, 1.0).id)
        });
        if mask >= cult || u01k(id, 5) < 0.06 {
            return None;
        }
        let u_crop = if u01k(id, 16) < 0.5 { u01k(cluster, 6) } else { u01k(id, 6) };
        let kind = crop_kind(r.season, u_crop, t.moist < 0.33 && r.style != 2);
        // hot, wet climates grow other crops: rice paddies, oil-palm plantations, sugarcane and
        // bare red laterite between plantings (not the golden cereals of temperate farmland)
        let tropic = smoothstep(19.0, 25.0, t.temp) * smoothstep(0.5, 0.7, t.moist);
        let mut col;
        let mut extra_h = 0.0;
        if u01k(id, 30) < tropic {
            let tk = u_crop;
            let fine = 1.0 + 0.08 * perlin3(id ^ 0x7A1, p / 6.0) * band(6.0, gsd) + 0.06 * perlin3(id ^ 0x7A2, p / 1.5) * band(1.5, gsd);
            if tk < 0.35 {
                // rice: young green or flooded paddies, cut into small plots by earth bunds
                col = mixc(srgb(74.0, 106.0, 62.0), srgb(58.0, 82.0, 74.0), u01k(id, 31)) * fine;
                let sx = 18.0 + 22.0 * u01k(id, 32);
                let sy = sx * (1.2 + 0.8 * u01k(id, 33));
                let (mx, my) = (fx.rem_euclid(sx), fy.rem_euclid(sy));
                let d = mx.min(sx - mx).min(my).min(sy - my);
                let bund = band_cov(d, 0.6, fw) * band(sx, gsd);
                col = mixc(col, srgb(108.0, 106.0, 74.0), bund);
                extra_h = 0.3 * bund;
            } else if tk < 0.75 {
                // oil palm: star-shaped crowns on a triangular grid; stands of different age
                let age = 0.35 + 0.65 * u01k(id, 34);
                let (sx, sy) = (9.0, 7.8);
                let j = (fy / sy).round();
                let off = if (j as i64).rem_euclid(2) == 0 { 0.0 } else { 0.5 * sx };
                let i = ((fx - off) / sx).round();
                let rel = DVec2::new(fx - off - i * sx, fy - j * sy);
                let ang = rel.y.atan2(rel.x) + u01k(mix64(id ^ (i as i64 as u64) ^ ((j as i64 as u64) << 20)), 1) * 6.3;
                let r_eff = 4.4 * age * (0.82 + 0.18 * (8.0 * ang).cos());
                let d = rel.length();
                let explicit = band(sx, gsd);
                let crown = band_cov(d, r_eff, fw) * explicit + (1.0 - explicit) * (0.75 * age);
                let ground = srgb(92.0, 100.0, 60.0) * fine;
                let shade = 0.85 + 0.25 * (1.0 - (d / r_eff.max(0.1)).min(1.0));
                col = mixc(ground, srgb(58.0, 92.0, 44.0) * shade * fine, crown);
                extra_h = (3.0 + 9.0 * age) * crown;
            } else if tk < 0.92 {
                // sugarcane / banana: dense vivid green in rows
                col = srgb(86.0, 116.0, 60.0) * fine;
                let along = fx;
                col *= 1.0 + 0.08 * (along * std::f64::consts::TAU / 1.5).sin() * band(1.5, gsd);
                extra_h = 2.5;
            } else {
                // freshly ploughed red laterite
                col = srgb(152.0, 90.0, 62.0) * fine;
                col *= 1.0 + 0.1 * (fx * std::f64::consts::TAU / 0.9).sin() * band(0.9, gsd);
            }
            col *= tint;
        } else {
            col = pal.crop[kind];
            col *= 0.94 + 0.12 * u01k(id, 8);
            col *= tint;
            // within-field variation (soil moisture, growth, management) at several scales
            col *= 1.0 + 0.10 * pf.field_var
                + 0.12 * perlin3(id, p / 35.0) * band(35.0, gsd)
                + 0.08 * perlin3(id ^ 1, p / (0.8 * r.fw));
            // growth zones (soil, moisture): greener / yellower patches of tens of metres
            let gz = perlin3(id ^ 0x6A0, p / 55.0) * band(55.0, gsd);
            col = mixc(col, col * DVec3::new(1.12, 1.04, 0.82), 0.5 * smoothstep(0.0, 0.6, gz));
            col = mixc(col, col * DVec3::new(0.86, 0.93, 0.88), 0.5 * smoothstep(0.0, 0.6, -gz));
            // management direction: rows / tramlines along one field axis; in the headland (strip
            // along the field edge where the tractor turns) the pattern runs parallel to the edge
            let row_ang = if u01k(id, 9) < 0.7 { 0.0 } else { std::f64::consts::FRAC_PI_2 };
            let headland_w = 8.0 + 10.0 * u01k(id, 11);
            let in_headland = edge < headland_w && matches!(kind, 0..=4);
            let row_ang = if in_headland { row_ang + std::f64::consts::FRAC_PI_2 } else { row_ang };
            let (sa, ca) = row_ang.sin_cos();
            let along = fx * ca + fy * sa;
            // soil / growth texture at several scales (band-limited)
            let tex = 0.18 * perlin3(id ^ 0x7E1, p / 7.0) * band(7.0, gsd)
                + 0.13 * perlin3(id ^ 0x7E2, p / 2.5) * band(2.5, gsd)
                + 0.07 * perlin3(id ^ 0x7E5, p / 0.9) * band(0.9, gsd)
                + 0.12 * perlin3(id ^ 0x7E4, p / 22.0) * band(22.0, gsd)
                + 0.10 * perlin3(id ^ 0x7E3, p / 90.0) * band(90.0, gsd);
            col *= 1.0 + tex;
            // wet hollows / bare patches inside some fields
            if u01k(id, 12) < 0.35 {
                let wp = perlin3(id ^ 0x5A7, p / (40.0 + 60.0 * u01k(id, 13)));
                let m = smoothstep(0.25, 0.45, wp) * band(30.0, gsd).max(0.3);
                col = mixc(col, col * DVec3::new(0.82, 0.86, 0.80), m);
            }
            if in_headland {
                col *= 0.96 + 0.03 * u01k(id, 14);
            }
            match kind {
                0 | 1 => {
                    let sp = 0.8 + 0.8 * u01k(id, 10);
                    col *= 1.0 + 0.12 * (along * std::f64::consts::TAU / sp).sin() * band(sp, gsd);
                }
                3 => {
                    let sp = 6.0 + 4.0 * u01k(id, 10);
                    col *= 1.0 + 0.07 * ((along / sp * std::f64::consts::TAU).sin()).signum() * band(sp, gsd);
                }
                4 => {
                    let sp = 0.45;
                    col *= 1.0 + 0.15 * (along * std::f64::consts::TAU / sp).sin() * band(sp, gsd);
                    col *= 1.0 - 0.15 * smoothstep(0.2, 0.6, pf.field_var2);
                }
                8 => {
                    // orchard: rows of small trees
                    let (sx, sy) = (5.0 + 2.0 * u01k(id, 10), 4.0);
                    let gx = (fx / sx).round() * sx;
                    let gy = (fy / sy).round() * sy;
                    let d = DVec2::new(fx - gx, fy - gy).length();
                    let explicit = band(sx, gsd);
                    let cov = band_cov(d, 1.7, fw) * explicit + (1.0 - explicit) * 0.4;
                    col = mixc(col, pal.crown_decid * 1.1, cov);
                    extra_h = 4.0 * cov;
                }
                _ => {}
            }
            // tramlines (wheel tracks every ~18-36 m) in cereals / green crops / stubble
            if matches!(kind, 0 | 1 | 2 | 3 | 7) && !in_headland {
                let sp = 18.0 + 18.0 * (u01k(id, 15) * 2.0).floor() / 2.0;
                let m = along.rem_euclid(sp);
                let tl = band_cov(m - 0.9, 0.22, fw) + band_cov(m - 2.7, 0.22, fw);
                let vis = band(1.2, gsd).max(0.35 * band(sp, gsd));
                col = mixc(col, col * DVec3::new(0.78, 0.76, 0.74), tl * vis);
            }
        }
        // field borders: hedges (trees) or tracks or simply a thin margin
        let bw = r.border_w;
        let bcov = band_cov(edge, bw, fw.max(gsd * 0.4));
        let mut kind_out = 0;
        if bcov > 0.0 {
            let hb = mix64(id ^ 0xED6E);
            if u01k(hb, 1) < r.hedge {
                let hc = pal.crown_decid * (0.8 + 0.3 * pf.field_var3);
                col = mixc(col, hc, bcov);
                // rounded cross-section and a height varying along the hedge (a row of shrubs and
                // small trees, not a flat-topped wall)
                let across = (1.0 - (edge.abs() / bw.max(0.1)).powi(2)).max(0.0).sqrt();
                let along = 0.45 + 0.55 * (0.5 + 0.5 * perlin3(hb, p / 6.0));
                extra_h = lerp(extra_h, (2.2 + 2.3 * u01k(hb, 2)) * across * along, bcov);
                if bcov > 0.5 {
                    kind_out = 1;
                }
            } else if u01k(hb, 3) < r.track * 0.4 {
                col = mixc(col, pal.gravel, bcov);
                if bcov > 0.5 {
                    kind_out = 2;
                }
            } else {
                col = mixc(col, mixc(pal.grass_dry, pal.grass_wet, t.moist), bcov * 0.7);
            }
        }
        Some((col, extra_h, inside, kind_out))
    }

    /// Town at point p. Returns (colour, height above ground, coverage, class, shadow, emission).
    /// Coverage is crisp: streets and built lots cover the ground fully, open land in the
    /// outskirts shows the underlying fields / nature.
    #[allow(clippy::too_many_arguments)]
    /// How built-up the town is at `p` (1 in the centre, 0 outside its irregular footprint), and
    /// the relative distance from the centre.
    /// `clear`: 0 where the town must not be (rivers), 1 elsewhere.
    fn town_urban(&self, town: &TownInfo, p: DVec3, gsd: f64, slope: f64, clear: f64, pf: &PixFields) -> (f64, f64) {
        let d = p - town.center;
        let q0 = DVec2::new(d.dot(town.ex), d.dot(town.ey));
        let r = town.radius;
        // far outside: the test below without its square roots (|qa|² >= |q0|² min(e, 1/e))
        let (e, q2, r2) = (town.elong, q0.length_squared(), 4.0 * r * r * (1.0 + 1e-9));
        if if e >= 1.0 { q2 > r2 * e } else { e * q2 > r2 } {
            return (0.0, 2.0);
        }
        // elongated, irregular footprint (noise relative to the town size)
        let qa = DVec2::new(q0.x / town.elong.sqrt(), q0.y * town.elong.sqrt());
        if qa.length() > r * 2.0 {
            return (0.0, 2.0);
        }
        let n1 = 0.32 * perlin3(town.seed ^ 0x71, p / (0.9 * r)) + 0.18 * perlin3(town.seed ^ 0x72, p / (0.35 * r)) * band(0.35 * r, gsd)
            + 0.08 * pf.warp2;
        let rel = qa.length() / (r * (1.0 + n1)).max(1.0);
        ((1.0 - smoothstep(0.3, 1.0, rel)) * (1.0 - smoothstep(0.45, 0.8, slope)) * clear, rel)
    }

    #[allow(clippy::too_many_arguments)]
    fn town(&self, town: &TownInfo, p: DVec3, gsd: f64, fw: f64, slope: f64, clear: f64, shadows: bool, pf: &PixFields) -> Option<(DVec3, f64, f64, u8, f64, DVec3)> {
        let pal = &self.pal;
        let d = p - town.center;
        let q0 = DVec2::new(d.dot(town.ex), d.dot(town.ey));
        let (urban, rel) = self.town_urban(town, p, gsd, slope, clear, pf);
        if urban <= 0.0 {
            return None;
        }
        // organic (curved) streets in old towns
        let warp = DVec2::new(perlin3(town.seed, p / 350.0), perlin3(town.seed ^ 9, p / 350.0)) * (35.0 * town.organic);
        let q = q0 + warp;
        // street grid with jittered (irregularly spaced) street lines
        let b = town.block;
        let line = |axis: u64, i: i64| (i as f64 + 0.36 * (u01(hash2(town.seed ^ 0x5EE7, axis as i64, i)) - 0.5)) * b;
        let cell = |axis: u64, x: f64| -> (i64, f64, f64) {
            let mut i = (x / b).floor() as i64;
            if x < line(axis, i) {
                i -= 1;
            } else if x >= line(axis, i + 1) {
                i += 1;
            }
            let (a0, a1) = (line(axis, i), line(axis, i + 1));
            (i, x - a0, a1 - a0)
        };
        let (bix, bqx, bsx) = cell(0, q.x);
        let (biy, bqy, bsy) = cell(1, q.y);
        let sw = town.street * 0.7; // lots start beyond the widest street (a verge along narrower ones)
        // streets exist where the town is dense enough; outskirts keep only some of them. Decided
        // per street segment (axis, line, and the segment along it), so the blocks on both sides
        // agree, and per axis, so streets end square at a crossing; crisp: a street is there or
        // not (a fading street left half-transparent asphalt with half-masked trees on it)
        let near_x = bix + (bqx > bsx - bqx) as i64; // nearest line across x and across y
        let near_y = biy + (bqy > bsy - bqy) as i64;
        // each street segment exists or not as a whole: decided with the town density at its
        // midpoint (per-pixel density faded streets in and out along a density contour, and the
        // houses of the blocks beside them were cut along it)
        let urban_at = |dq: DVec2| self.town_urban(town, p + town.ex * dq.x + town.ey * dq.y, gsd, slope, clear, pf).0;
        let seg_here = |h: u64, u: f64| -> f64 { if u > 0.15 + 0.12 * u01k(h, 4) && !(u < 0.45 && u01k(h, 3) < 0.4) { 1.0 } else { 0.0 } };
        let ymid = 0.5 * (line(1, biy) + line(1, biy + 1));
        let xmid = 0.5 * (line(0, bix) + line(0, bix + 1));
        let seg_x = |i: i64| seg_here(hash2(town.seed ^ 0x57, i, biy), urban_at(DVec2::new(line(0, i), ymid) - q));
        let seg_y = |i: i64| seg_here(hash2(town.seed ^ 0x58, i, bix), urban_at(DVec2::new(xmid, line(1, i)) - q));
        let (x0, x1, y0, y1) = (seg_x(bix), seg_x(bix + 1), seg_y(biy), seg_y(biy + 1));
        let here_x = if near_x == bix { x0 } else { x1 };
        let here_y = if near_y == biy { y0 } else { y1 };
        // a block is built on only if a street runs along at least one of its sides (houses stood
        // in the fields of the outskirts with no street anywhere near)
        let access = x0.max(x1).max(y0).max(y1);
        let sw_of = |line: i64| if line.rem_euclid(4) == 0 { town.street * 1.4 } else { town.street } * 0.5;
        let (sw_x, sw_y) = (sw_of(near_x), sw_of(near_y));
        let fws = fw.max(gsd * 0.5);
        let street = f64::max(band_cov(bqx.min(bsx - bqx), sw_x, fws) * here_x, band_cov(bqy.min(bsy - bqy), sw_y, fws) * here_y);
        // sidewalks between the carriageway and the lots (that verge was left as bare ground,
        // unlit at night: dark lines along every street)
        let walk_w = 0.2 * town.street + 0.5;
        let walk = f64::max(band_cov(bqx.min(bsx - bqx), sw_x + walk_w, fws) * here_x, band_cov(bqy.min(bsy - bqy), sw_y + walk_w, fws) * here_y);
        let bh = hash2(town.seed ^ 0xB10C, bix, biy);
        let block_kind = u01k(bh, 1);
        let mut col = pal.asphalt;
        let mut height = 0.0;
        let mut class = lc::URBAN;
        let mut shadow = 0.0;
        let mut cov_lot = 0.0;
        let mut porch = 0.0;
        let mut roof_frac = 0.0; // building roof coverage of this sample
        let mut win_col = DVec3::new(1.0, 0.74, 0.44);
        let mut windows = 0.0; // lit windows on the walls (the footprint rim, which the renderer
                               // stretches into the facades)
        let inner = DVec2::new(bqx - sw, bqy - sw);
        let (bw, bd) = (bsx - 2.0 * sw, bsy - 2.0 * sw);
        let central = (1.0 - rel).max(0.0);
        let buildings_resolved = band(town.lot, gsd);
        if urban > 0.5 && block_kind < 0.07 {
            // park
            col = pal.grass_wet * (1.0 + 0.15 * pf.detail);
            class = lc::GRASS;
            cov_lot = 1.0;
        } else if urban > 0.45 && block_kind < 0.12 {
            col = pal.concrete * (0.82 + 0.1 * u01k(bh, 4)); // parking / plaza
            cov_lot = 1.0;
        } else if inner.x >= 0.0 && inner.y >= 0.0 && inner.x < bw && inner.y < bd {
            let industrial = block_kind > 0.92 && central < 0.5;
            let lot_w = if industrial { bw } else { town.lot * (0.7 + 0.6 * u01k(bh, 5)) };
            let rows = if industrial || bd < 2.6 * town.lot { 1.0 } else { 2.0 };
            let li = (inner.x / lot_w).floor();
            let lj = (inner.y / (bd / rows)).floor();
            let lx = inner.x - li * lot_w;
            let ly = inner.y - lj * (bd / rows);
            let lh = hash2(bh, li as i64, lj as i64);
            // decided with the town density at the lot's centre, not per pixel: houses were cut
            // in half where the density contour crossed them
            let lot_c = DVec2::new(li * lot_w + 0.5 * lot_w - inner.x, (lj + 0.5) * (bd / rows) - inner.y);
            let urban_lot = self.town_urban(town, p + town.ex * lot_c.x + town.ey * lot_c.y, gsd, slope, clear, pf).0;
            let built = u01k(lh, 3) < urban_lot.powf(0.7) * 1.05 && access > 0.5;
            if built || urban_lot > 0.65 {
                cov_lot = 1.0;
                let yard = mixc(mixc(pal.grass_wet, pal.soil[0], 0.3 + 0.4 * u01k(lh, 10)), pal.concrete, 0.25 * central)
                    * (1.0 + 0.2 * pf.detail);
                col = yard;
                if built {
                    // building footprint inside the lot (sometimes L-shaped)
                    let setb = if industrial { 6.0 } else { 1.5 + 3.5 * u01k(lh, 1) };
                    let fwid = lot_w - 2.0 * setb.min(lot_w * 0.3);
                    let fdep = (bd / rows - setb - 2.0 - 5.0 * u01k(lh, 2)).max(0.0);
                    if fwid > 3.0 && fdep > 3.0 {
                        let cx = lx - lot_w * 0.5;
                        let cy = if (lj as i64) % 2 == 0 { ly - setb - fdep * 0.5 } else { ly - (bd / rows - setb - fdep * 0.5) };
                        let ex = fwid * 0.5 - cx.abs();
                        let ey = fdep * 0.5 - cy.abs();
                        let mut inside = (ex.min(ey) / fw + 0.5).clamp(0.0, 1.0);
                        if !industrial && u01k(lh, 11) < 0.3 {
                            // L-shape: remove a corner quadrant
                            let qx = if u01k(lh, 12) < 0.5 { cx } else { -cx };
                            let qy = if u01k(lh, 13) < 0.5 { cy } else { -cy };
                            let cut = (qx - fwid * 0.1).min(qy - fdep * 0.1);
                            inside *= 1.0 - (cut / fw + 0.5).clamp(0.0, 1.0);
                        }
                        let tall = central.powf(2.0) * town.height;
                        let hb = if industrial { 7.0 + 7.0 * u01k(lh, 4) } else { 3.5 + 4.0 * u01k(lh, 4) + 40.0 * tall * u01k(lh, 5) };
                        let flat_roof = industrial || hb > 12.0 || u01k(lh, 6) < 0.2 + 0.3 * central;
                        let ri = ((town.roof_style * 3.0 + u01k(lh, 7) * 4.0) as usize) % 7;
                        let mut roof = if industrial { mixc(pal.roofs[4], pal.roofs[2], u01k(lh, 8)) } else { pal.roofs[ri] };
                        roof *= 0.82 + 0.36 * u01k(lh, 9);
                        let mut h_here = hb;
                        if !flat_roof {
                            // pitched roof along the longer axis
                            let (half, dperp) = if fwid > fdep { (fdep * 0.5, ey) } else { (fwid * 0.5, ex) };
                            h_here = hb + 0.35 * half * (dperp / half).clamp(0.0, 1.0);
                        } else {
                            roof *= 1.0 - 0.15 * band_cov(ex.min(ey), 0.6, fw); // parapet edge
                        }
                        if inside > 0.0 {
                            col = mixc(col, roof, inside);
                            height = h_here * inside;
                            roof_frac = inside * buildings_resolved;
                            // windows every ~2.6 m along the walls, a share of them lit (more in
                            // the centre, offices and flats included)
                            let rim = (1.0 - smoothstep(0.0, 0.9, ex.min(ey))) * inside;
                            if rim > 0.0 {
                                let along = if ex < ey { cy } else { cx };
                                let wsp = 2.6;
                                let wi = (along / wsp).floor();
                                let wf = along / wsp - wi;
                                let lit_frac = 0.08 + 0.2 * central + 0.12 * u01k(lh, 15);
                                // incandescent / warm LED in most homes, some neutral and cool
                                // (offices, screens)
                                let wc = u01k(lh, 16);
                                win_col = if wc < 0.6 {
                                    DVec3::new(1.0, 0.70, 0.40)
                                } else if wc < 0.85 {
                                    DVec3::new(1.0, 0.86, 0.66)
                                } else {
                                    DVec3::new(0.82, 0.90, 1.0)
                                };
                                let lit = u01k(hash2(lh ^ 0x3D0, wi as i64, (ex < ey) as i64), 1) < lit_frac;
                                let explicit = band(wsp, gsd);
                                let pane = if lit && (0.3..0.65).contains(&wf) { 1.0 } else { 0.0 };
                                windows = rim * (pane * explicit + 0.55 * lit_frac * (1.0 - explicit));
                            }
                            if inside > 0.5 {
                                class = lc::BUILDING;
                            }
                        }
                        // porch / yard light in front of some houses
                        if !industrial && u01k(lh, 14) < 0.6 {
                            let front_y = if (lj as i64) % 2 == 0 { setb * 0.5 } else { bd / rows - setb * 0.5 };
                            let d2 = (lx - lot_w * 0.5).powi(2) + (ly - front_y).powi(2);
                            // a small bright lamp by the door, not a soft blob over the yard
                            porch = point_light(d2, 2.0, 0.4, fw);
                        }
                    }
                }
            }
            // mean appearance when lots are unresolved
            if buildings_resolved < 1.0 {
                let mean_roof = mixc(pal.roofs[(town.roof_style * 6.99) as usize], pal.concrete, 0.3);
                let mean = mixc(mixc(pal.grass_wet, pal.soil[0], 0.4), mean_roof, 0.5);
                col = mixc(mean, col, buildings_resolved);
                height = lerp(3.0 * urban, height, buildings_resolved);
                cov_lot = lerp(urban.powf(0.7), cov_lot, buildings_resolved);
            }
            // cast shadows of buildings onto the ground (march toward the sun)
            if shadows && buildings_resolved > 0.0 && height < 0.5 && cov_lot > 0.0 {
                for k in 1..=4 {
                    let dist_s = k as f64 * 3.5;
                    let need = dist_s * self.sun_tan;
                    let sp = p + town.sun * dist_s;
                    let hh = self.building_height_at(town, sp, gsd, pf);
                    if hh > need {
                        shadow = 0.75 * buildings_resolved;
                        break;
                    }
                }
            }
        }
        let cov = cov_lot.max(walk);
        if cov <= 0.0 {
            return None;
        }
        col = mixc(col, pal.concrete * 0.9, (walk - street).max(0.0) * (1.0 - cov_lot) / cov);
        col = mixc(col, pal.asphalt * (1.0 + 0.05 * pf.detail), street / cov);
        if street > 0.5 {
            class = lc::ROAD;
        }
        height *= 1.0 - street;

        // ---- night lights: sharp lamp heads with soft pools under them along every street,
        // lit plazas / industrial yards, windows and porch lights
        let lamp_sp = 18.0 + 8.0 * town.organic;
        // lamp types: high-pressure sodium (amber), warm LED (3000 K) and neutral LED (4000 K),
        // chosen per street with a town-specific mix (main streets lean to neutral LED)
        const LAMPS: [DVec3; 3] = [DVec3::new(1.0, 0.48, 0.12), DVec3::new(1.0, 0.72, 0.38), DVec3::new(0.86, 0.92, 1.0)];
        let mix_sodium = 0.25 + 0.4 * u01k(town.seed, 20);
        let dominant = if mix_sodium > 0.45 { 0 } else { 1 };
        let lamp_type = |axis: u64, line: i64| -> DVec3 {
            let u = u01k(hash2(town.seed ^ 0x1A3C ^ (axis << 40), line, 0), 1);
            let main = line.rem_euclid(4) == 0;
            let t = if main && u < 0.45 {
                2
            } else if u < mix_sodium {
                0
            } else if u < 0.85 {
                1
            } else {
                2
            };
            LAMPS[t]
        };
        let lamp_col = LAMPS[dominant];
        // lamps stay points down to the scale of the street grid: where they would merge (sample
        // spacing above half their spacing) only every m-th lamp is drawn, m times brighter. An
        // area mean there was clipped by the camera very differently from the points it averaged:
        // the far part of a town glowed as a flat slab next to the point-lit near part.
        let lamp_res = band(town.block, gsd);
        let fwl = fw.max(0.5 * gsd);
        let m = (2.0 * fwl / lamp_sp).max(1.0).log2().ceil().exp2();
        let sp_eff = lamp_sp * m;
        let mut emission = DVec3::ZERO;
        if lamp_res > 0.0 && here_x.max(here_y) > 0.0 {
            // lamps every lamp_sp along each street, alternating sides of the carriageway
            for axis in 0..2u64 {
                let (bi, bq, bs, along) = if axis == 0 { (bix, bqx, bsx, q.y) } else { (biy, bqy, bsy, q.x) };
                // signed offset of the pixel from the nearest street centreline, and that line's index
                let (off, li) = if bq < bs - bq { (bq, bi) } else { (-(bs - bq), bi + 1) };
                let (here, swa) = if axis == 0 { (here_x, sw_x) } else { (here_y, sw_y) };
                if here <= 0.0 || off.abs() > swa + 22.0 + 3.0 * fwl {
                    continue;
                }
                // index of the lamp (in units of lamp_sp; a multiple of m when thinned)
                let k = (along / sp_eff).round() * m;
                let side = if (k as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
                let lamp_off = side * swa * 0.85;
                let (dp, da) = (off - lamp_off, along - k * lamp_sp);
                let d2 = dp * dp + da * da;
                let lh = hash2(town.seed ^ 0x1A3B ^ (axis << 40), li, k as i64);
                if u01k(lh, 1) < 0.93 {
                    let sa = 0.28 * lamp_sp;
                    // a soft pool on the street and a sharp, bright head (a point from the air)
                    // (the head carries most of the light seen from the air: with a bright pool
                    // every lamp was a soft disc)
                    let pool = 0.03 * (-(dp * dp) / (2.0 * 3.5 * 3.5) - (da * da) / (2.0 * sa * sa)).exp();
                    let core = point_light(d2, 6.0, 0.4, fw.max(0.5 * gsd));
                    emission += lamp_type(axis, li) * ((pool + core) * m) * (0.75 + 0.5 * u01k(lh, 2));
                }
            }
        }
        // the lamps light the ground below them, not the roofs (lit roofs read as glowing spikes)
        let ground_lit = 1.0 - roof_frac * (1.0 - street);
        emission *= ground_lit;
        porch *= ground_lit;
        // prefiltered mean when lamps are unresolved (town glow)
        // unresolved lamps: their mean over the area (lamp energy: head 2π·6·0.4² + pool
        // 2π·0.03·3.5·0.28·sp, 93% present, every sp along streets every ~block on both axes); a
        // fixed 0.07 was ~7x the mean of the resolved lamps, so the far part of a town glowed as a
        // flat orange slab next to the dark-roofed near part
        let lamp_e = 0.93 * (2.0 * std::f64::consts::PI * (6.0 * 0.16 + 0.03 * 3.5 * 0.28 * lamp_sp));
        let lamp_mean = lamp_e * 2.0 / (town.block * lamp_sp);
        emission = emission * lamp_res + lamp_col * lamp_mean * (1.0 - lamp_res) * smoothstep(0.15, 0.4, urban);
        // windows: only where the buildings are resolved (a rim smeared over coarse pixels lit
        // whole blocks; the lamps carry the town's light at coarse zooms)
        let windows = windows * buildings_resolved;
        emission += win_col * (0.55 * windows);
        // industrial yards: a dim base (plazas and parking have only the street lamps around
        // them: a lit base drew uniform grey slabs)

        if block_kind > 0.92 && central < 0.5 {
            emission += DVec3::new(1.0, 0.88, 0.7) * 0.02 * band(lamp_sp, gsd);
        }
        emission += DVec3::new(1.0, 0.72, 0.42) * porch;
        Some((col, height, cov, class, shadow * (1.0 - street * 0.5), emission / cov.max(0.05)))
    }

    fn building_height_at(&self, town: &TownInfo, p: DVec3, gsd: f64, pf: &PixFields) -> f64 {
        match self.town(town, p, gsd, 0.01, 0.0, 1.0, false, pf) {
            Some((_, h, cov, _, _, _)) => h * cov,
            None => 0.0,
        }
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
        assert!(towns > n / 20, "too few town samples ({towns}/{n})");
    }
}
