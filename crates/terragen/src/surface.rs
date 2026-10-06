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
use std::collections::HashMap;

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
    /// Horizontal unit vector towards the sun (ECEF).
    pub sun: DVec3,
}

/// Per-thread caches for site-level properties (pure functions of the site; caching is only an
/// optimization, so results are deterministic regardless of evaluation order).
#[derive(Default)]
pub struct Caches {
    regions: HashMap<u64, RegionInfo>,
    towns: HashMap<u64, TownInfo>,
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
            sand: [srgb(218.0, 190.0, 142.0), srgb(204.0, 150.0, 96.0), srgb(228.0, 210.0, 170.0)],
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
}

/// One tree layer (crowns on a jittered grid).
#[derive(Clone, Copy)]
struct TreeLayer {
    cell: f64,
    seed: u64,
    density: f64,
    height: f64,
    color: DVec3,
    conifer: f64,
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
        }
    }

    /// Per-pixel smooth fields (band-limited at the pixel GSD).
    pub fn pixel_fields(&self, p: DVec3, gsd: f64) -> PixFields {
        PixFields {
            detail: self.detail.eval(p, gsd),
            patch: self.patch.eval(p, gsd),
            land: self.land_n.eval(p, gsd) * self.land_n.norm() * 1.8,
            strata: self.strata.eval(p, gsd),
            strata2: self.strata.eval(p * 1.7, gsd),
            snow: self.snow_n.eval(p, gsd),
            forest: self.forest.eval(p, gsd) * self.forest.norm() * 1.8,
            stand: self.patch.eval(p * 0.3, gsd) + 0.5 * perlin3(0x57A, p / 1200.0),
            field_var: self.field_var.eval(p, gsd),
            field_var2: self.field_var.eval(p * 1.7, gsd),
            field_var3: self.field_var.eval(p * 3.0, gsd),
            warp2: self.warp2.eval(p, gsd),
            water: self.patch.eval(p * 0.37, gsd),
        }
    }

    fn region_info(&self, world: &World, cache: &mut Caches, t: &Terrain) -> RegionInfo {
        if let Some(r) = cache.regions.get(&t.region.id) {
            return *r;
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
        info
    }

    fn town_info(&self, world: &World, cache: &mut Caches, t: &Terrain) -> TownInfo {
        if let Some(r) = cache.towns.get(&t.town.id) {
            return *r;
        }
        let id = t.town.id;
        let ctx = site_ctx(world, t.town.center, 300.0);
        let (east, north) = (ctx.east, ctx.north);
        let tc = world.terrain(&ctx);
        let p_exist = (tc.habit * 1.1 * world.cfg.landuse.towns).min(0.95);
        let exists = u01k(id, 1) < p_exist && tc.water_kind == water::NONE && tc.ground > 2.0 && tc.ground < 4000.0;
        let ang = u01k(id, 2) * std::f64::consts::FRAC_PI_2;
        let (sa, ca) = ang.sin_cos();
        let mut radius = 160.0 * (u01k(id, 3).powf(1.6) * 2.4).exp();
        if u01k(id, 4) < 0.03 {
            radius *= 4.0; // occasional city
        }
        let info = TownInfo {
            exists,
            center: ctx.p,
            ex: east * ca + north * sa,
            ey: north * ca - east * sa,
            radius: radius.min(world.cfg.landuse.town_cell_km * 1000.0 * 0.45),
            block: 70.0 + 70.0 * u01k(id, 5),
            street: 9.0 + 7.0 * u01k(id, 6),
            organic: u01k(id, 7),
            roof_style: u01k(id, 8),
            height: u01k(id, 9),
            lot: 13.0 + 12.0 * u01k(id, 10),
            sun: east * self.sun_h.x + north * self.sun_h.y,
        };
        cache.towns.insert(id, info);
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
            col *= 1.0 + 0.12 * v;
            // surf/foam close to the ocean shore
            if l.water_kind == water::OCEAN && depth < 1.2 {
                let foam = (1.0 - depth / 1.2) * 0.5 * (0.5 + 0.5 * perlin3(7, p / 6.0)) * band(6.0, gsd);
                col = mixc(col, srgb(200.0, 210.0, 210.0), foam);
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
            let strata = (l.ground / (6.0 + 10.0 * st[3]) + 3.0 * pf.strata).sin();
            rc *= 1.0 + 0.06 * strata * band(8.0, gsd) + 0.25 * detail + 0.12 * pf.strata2;
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
        let coastal = 1.0 - smoothstep(0.0, 0.03, t.cont.abs());
        if l.ground < 3.0 && coastal > 0.0 && slope < 0.25 {
            let b = (1.0 - smoothstep(1.2, 3.0, l.ground + 0.8 * detail)) * coastal * (1.0 - smoothstep(0.12, 0.25, slope));
            let bc = mixc(pal.beach, pal.wet_sand, 1.0 - smoothstep(0.0, 0.6, l.ground));
            col = mixc(col, bc, b);
            if b > 0.5 {
                class = lc::BEACH;
            }
        }

        // ------------------------------------------------------------- snow
        let snow_n = pf.snow;
        let snow = smoothstep(-1.0, -5.0, temp + 4.0 * snow_n) * (1.0 - 0.75 * smoothstep(0.9, 1.6, slope));
        if snow > 0.0 {
            col = mixc(col, pal.snow * (1.0 + 0.03 * detail), snow);
            if snow > 0.5 {
                class = lc::SNOW;
            }
        }

        let mut height = l.ground;
        let mut lit: f64 = 1.0;
        let mut emission = DVec3::ZERO;
        let natural_ok = (1.0 - rock) * (1.0 - snow) * (1.0 - t.sand);

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
        let riparian = if l.river_hw > 0.0 {
            let ad = l.river_d.abs();
            (1.0 - smoothstep(l.river_hw + 4.0, l.river_hw * 2.0 + 30.0, ad)) * smoothstep(0.3, 0.6, t.moist) * natural_ok
        } else {
            0.0
        };

        // ------------------------------------------------------------- fields
        let flat_ok = 1.0 - smoothstep(0.12, 0.22, slope);
        let mut field_cov = 0.0;
        if let Some(r) = &region {
            if t.agri > 0.02 && natural_ok * flat_ok > 0.3 && world.cfg.landuse.agriculture > 0.0 {
                if let Some((fcol, fh, cov, edge_kind)) = self.field(r, t, q_rot, p, gsd, fw, pf) {
                    let a = cov * natural_ok * flat_ok * (1.0 - riparian);
                    col = mixc(col, fcol, a);
                    height += fh * a;
                    field_cov = a;
                    if a > 0.5 {
                        class = if edge_kind == 1 { lc::FOREST } else if edge_kind == 2 { lc::ROAD } else { lc::CROP };
                    }
                }
            }
        }

        // ------------------------------------------------------------- trees
        let veg = &world.cfg.vegetation;
        if veg.tree_density > 0.0 && natural_ok > 0.05 {
            let base_cover = smoothstep(0.3, 0.68, wet) * smoothstep(-6.0, 2.0, temp) * veg.tree_density;
            let fpat = pf.forest; // ~[-1,1]
            let fpu = 0.5 + 0.5 * fpat;
            // forests where the patch field is below the cover fraction (crisp but noisy edges)
            let edge = 0.03 + 0.6 * band(30.0, gsd).min(1.0) * 0.0;
            let forest = smoothstep(-edge, edge, base_cover - fpu) ;
            // savanna / steppe scattered trees
            let savanna = smoothstep(0.18, 0.35, wet) * (1.0 - smoothstep(0.55, 0.7, wet)) * smoothstep(12.0, 20.0, temp) * 0.12;
            let groves = 0.04 * smoothstep(0.15, 0.3, wet);
            let clear = 1.0 - 0.85 * smoothstep(0.05, 0.4, t.agri) * flat_ok;
            let mut dens = (forest * 0.9 * clear + savanna + groves) * natural_ok * (1.0 - field_cov) * veg.tree_density;
            dens = dens.max(0.8 * riparian * veg.tree_density);
            dens = (dens + 0.35 * smoothstep(0.2, 0.9, -t.gully) * smoothstep(0.2, 0.5, wet) * natural_ok).min(1.0);
            dens *= 1.0 - smoothstep(0.9, 1.4, slope);
            dens *= 1.0 - smoothstep(0.0, 0.6, t.mountain * smoothstep(-2.0, -6.0, temp)); // tree line
            // shrubs / bushes in steppe, maquis and rocky slopes (texture of natural ground)
            let shrub_clim = smoothstep(0.15, 0.3, wet) * (1.0 - smoothstep(0.6, 0.8, wet)) * smoothstep(2.0, 10.0, temp);
            let shrub_patch = smoothstep(-0.2, 0.5, patch + 0.4 * pf.land);
            let shrub = (0.45 * shrub_clim * shrub_patch * (1.0 - forest) * natural_ok.max(0.4 * rock) * (1.0 - field_cov) * veg.tree_density)
                .clamp(0.0, 0.6);
            if dens > 0.0 || shrub > 0.01 {
                let conifer = 1.0 - smoothstep(4.0, 13.0, temp);
                let tropic = smoothstep(19.0, 25.0, temp) * smoothstep(0.55, 0.75, wet);
                let dry = 1.0 - smoothstep(0.3, 0.5, wet);
                let tall = 0.5 + 0.5 * st[3];
                let layers = [
                    TreeLayer { cell: 5.5, seed: 0x7EE1, density: dens * conifer, height: 14.0 + 10.0 * tall, color: pal.crown_conifer, conifer: 1.0 },
                    TreeLayer {
                        cell: 8.5,
                        seed: 0x7EE2,
                        density: dens * (1.0 - conifer) * (1.0 - tropic),
                        height: 10.0 + 10.0 * tall,
                        color: mixc(pal.crown_decid, pal.crown_dry, dry),
                        conifer: 0.0,
                    },
                    TreeLayer { cell: 13.0, seed: 0x7EE3, density: dens * tropic, height: 22.0 + 14.0 * tall, color: pal.crown_tropic, conifer: 0.0 },
                    TreeLayer { cell: 3.2, seed: 0x7EE4, density: shrub, height: 1.6, color: pal.shrub, conifer: 0.0 },
                ];
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
        if roads > 0.0 {
            let steep = 1.0 - smoothstep(0.25, 0.45, slope);
            let habit = smoothstep(0.02, 0.25, t.habit) * steep * (1.0 - snow) * (1.0 - t.sand * 0.7);
            let mut road_cov: f64 = 0.0;
            let mut road_col = pal.asphalt;
            if habit > 0.0 {
                let w_major = 12.0;
                let c1 = band_cov(l.road_major, w_major * 0.5, fw.max(gsd * 0.5));
                if c1 > 0.0 {
                    road_cov = c1 * habit;
                    // lighter shoulders
                    let sh = band_cov(l.road_major, w_major * 0.5 + 1.5, fw) - band_cov(l.road_major, w_major * 0.5, fw);
                    road_col = mixc(pal.asphalt, pal.concrete, sh.max(0.0) * 0.6);
                }
                let w_minor = 6.0;
                let c2 = band_cov(l.road_minor, w_minor * 0.5, fw.max(gsd * 0.5)) * habit * 0.9;
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

        // ------------------------------------------------------------- towns
        if t.town.id != 0 && world.cfg.landuse.towns > 0.0 {
            let town = self.town_info(world, cache, t);
            if town.exists {
                if let Some((tcol, th, cov, cls, shadow, em)) = self.town(&town, p, gsd, fw, slope, world.cfg.look.shadows, pf) {
                    emission = em;
                    col = mixc(col, tcol, cov);
                    if world.cfg.landuse.buildings_in_dsm {
                        height = lerp(height, l.ground + th, cov);
                    }
                    lit = lit.min(1.0 - shadow);
                    if cov > 0.5 {
                        class = cls;
                    }
                }
            }
        }

        // ------------------------------------------------------------- rivers (on top)
        if l.river_hw > 0.0 {
            let fwr = fw.max(gsd * 0.35);
            let cov = band_cov(l.river_d, l.river_hw, fwr);
            if cov > 0.0 {
                let wet_r = t.river_wet;
                let wcol = mixc(pal.river, pal.lake_deep, smoothstep(30.0, 200.0, l.river_hw * 2.0));
                let dry_col = mixc(pal.gravel, pal.sand[2], 0.5) * (1.0 + 0.1 * detail);
                let rc = mixc(dry_col, wcol, wet_r);
                col = mixc(col, rc, cov);
                height = lerp(height, l.river_level, cov);
                lit = lerp(lit, 1.0, cov);
                if cov > 0.5 {
                    if wet_r > 0.5 {
                        return Surface { albedo: col, height, class: lc::RIVER, lit, is_water: true, emission };
                    }
                    class = lc::SAND;
                }
            }
        }

        Surface { albedo: col.max(DVec3::ZERO), height, class, lit, is_water: false, emission }
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
                mean_col += layer.color * (0.85 + 0.3 * tex) * w;
                mean_w += w;
                mean_h += layer.height * 0.6 * cov_mean * (1.0 - explicit);
            }
            if explicit <= 0.0 {
                continue;
            }
            let cq = q / layer.cell;
            let cf = cq.floor();
            let (ix, iy) = (cf.x as i64, cf.y as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let h = hash2(layer.seed, ix + dx, iy + dy);
                    if u01k(h, 3) >= layer.density {
                        continue;
                    }
                    let c = DVec2::new(
                        (ix + dx) as f64 + 0.5 + 0.8 * (u01k(h, 1) - 0.5),
                        (iy + dy) as f64 + 0.5 + 0.8 * (u01k(h, 2) - 0.5),
                    ) * layer.cell;
                    let r = layer.cell * (0.36 + 0.24 * u01k(h, 4));
                    let d = (q - c).length();
                    if d > r + fw {
                        continue;
                    }
                    let cov = ((r - d) / fw + 0.5).clamp(0.0, 1.0) * explicit;
                    let x = (d / r).min(1.0);
                    let hh = layer.height * (0.7 + 0.6 * u01k(h, 5));
                    let prof = if layer.conifer > 0.5 { 0.35 + 0.65 * (1.0 - x) } else { 0.55 + 0.45 * (1.0 - x * x).sqrt() };
                    let th = hh * prof;
                    if th > best_h {
                        best_h = th;
                        let tint = 0.8 + 0.4 * u01k(h, 6);
                        let hue = DVec3::new(1.0 + 0.15 * (u01k(h, 7) - 0.5), 1.0, 1.0 - 0.1 * (u01k(h, 7) - 0.5));
                        // crowns are a bit darker at the rim (self-shading inside the crown)
                        best_col = layer.color * tint * hue * (0.75 + 0.35 * (1.0 - x));
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
    fn field(&self, r: &RegionInfo, t: &Terrain, q: DVec2, p: DVec3, gsd: f64, fw: f64, pf: &PixFields) -> Option<(DVec3, f64, f64, u8)> {
        let pal = &self.pal;
        let cult = smoothstep(0.02, 0.45, t.agri);
        let tint = DVec3::new(1.0 + 0.08 * (r.palette - 0.5), 1.0, 1.0 - 0.06 * (r.palette - 0.5));
        // Always evaluate the actual field; when fields approach the pixel size, their contrast is
        // reduced towards the mean, mimicking the variance reduction of box-filtering a mosaic.
        let fsize = if r.fh > 0.0 { r.fw.min(r.fh) } else { r.fw };
        let k = fsize / (fsize * fsize + 4.0 * gsd * gsd).sqrt();
        let mean = pal.crop_mean * tint * (1.0 + 0.10 * pf.field_var);
        let fwe = fw.max(0.6 * gsd); // edge filter never sharper than ~half a pixel
        let (c, h, cov, kind) = self.field_explicit(r, t, q, p, gsd, fwe, cult, tint, pf).unwrap_or((mean, 0.0, 0.0, 0));
        let cult_mean = cult * 0.9;
        let cov_m = lerp(cult_mean, cov, k);
        if cov_m <= 0.0 {
            return None;
        }
        let col = if cov > 0.0 { mixc(mean, c, k) } else { mean };
        Some((col, h * k, cov_m, kind))
    }

    /// Contiguous cultivation zones: a smooth mask compared with the cultivated fraction.
    fn cultivated(&self, c3: DVec3, frac: f64, gsd: f64) -> bool {
        let m = 0.5 + 0.5 * self.cult_n.eval(c3, gsd.min(100.0)) * self.cult_n.norm() * 2.2;
        m < frac
    }

    #[allow(clippy::too_many_arguments)]
    fn field_explicit(&self, r: &RegionInfo, t: &Terrain, q: DVec2, p: DVec3, gsd: f64, fw: f64, cult: f64, tint: DVec3, pf: &PixFields) -> Option<(DVec3, f64, f64, u8)> {
        let pal = &self.pal;
        // field id, within-field coords (along, across), distance to boundary
        let (id, fx, fy, edge, inside, fc) = match r.style {
            0 | 3 => {
                let wq = DVec2::new(perlin3(r.split.to_bits(), p / 900.0), perlin3(r.split.to_bits() ^ 1, p / 900.0));
                let q = q + wq * (0.08 * r.fw);
                let (w, h) = (r.fw, r.fh.max(r.fw));
                let j = (q.y / h).floor();
                let shift = u01(hash1(r.split.to_bits(), j as i64)) * w;
                let x = q.x + shift;
                let i = (x / w).floor();
                let fx = x - i * w;
                let fy = q.y - j * h;
                // split some cells into strips
                let hc = hash2(r.split.to_bits() ^ 0x55, i as i64, j as i64);
                let nstrip = 1 + (u01k(hc, 1) * if r.style == 3 { 1.0 } else { 3.5 }) as i64;
                let sh = h / nstrip as f64;
                let k = (fy / sh).floor().min(nstrip as f64 - 1.0);
                let fy2 = fy - k * sh;
                let edge = fx.min(w - fx).min(fy2).min(sh - fy2);
                let fc = DVec2::new(i * w + 0.5 * w - shift, j * h + k * sh + 0.5 * sh);
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
        if !self.cultivated(c3, cult, gsd) || u01k(id, 5) < 0.06 {
            return None;
        }
        let kind = crop_kind(r.season, u01k(id, 6), t.moist < 0.33 && r.style != 2);
        let mut col = pal.crop[kind];
        col *= 0.92 + 0.16 * u01k(id, 8);
        col *= tint;
        // within-field variation (soil moisture, growth, management) at several scales
        col *= 1.0 + 0.10 * pf.field_var
            + 0.07 * perlin3(id, p / 35.0) * band(35.0, gsd)
            + 0.08 * perlin3(id ^ 1, p / (0.8 * r.fw));
        // management direction: rows / tramlines along one field axis; in the headland (strip
        // along the field edge where the tractor turns) the pattern runs parallel to the edge
        let row_ang = if u01k(id, 9) < 0.7 { 0.0 } else { std::f64::consts::FRAC_PI_2 };
        let headland_w = 8.0 + 10.0 * u01k(id, 11);
        let in_headland = edge < headland_w && matches!(kind, 0..=4);
        let row_ang = if in_headland { row_ang + std::f64::consts::FRAC_PI_2 } else { row_ang };
        let (sa, ca) = row_ang.sin_cos();
        let along = fx * ca + fy * sa;
        let mut extra_h = 0.0;
        // soil / growth texture at several scales (band-limited)
        let tex = 0.08 * perlin3(id ^ 0x7E1, p / 7.0) * band(7.0, gsd)
            + 0.06 * perlin3(id ^ 0x7E2, p / 2.5) * band(2.5, gsd)
            + 0.05 * perlin3(id ^ 0x7E4, p / 22.0) * band(22.0, gsd)
            + 0.08 * perlin3(id ^ 0x7E3, p / 90.0) * band(90.0, gsd);
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
        // field borders: hedges (trees) or tracks or simply a thin margin
        let bw = r.border_w;
        let bcov = band_cov(edge, bw, fw.max(gsd * 0.4));
        let mut kind_out = 0;
        if bcov > 0.0 {
            let hb = mix64(id ^ 0xED6E);
            if u01k(hb, 1) < r.hedge {
                let hc = pal.crown_decid * (0.8 + 0.3 * pf.field_var3);
                col = mixc(col, hc, bcov);
                extra_h = lerp(extra_h, 4.0 + 3.0 * u01k(hb, 2), bcov);
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

    /// Town at point p. Returns (colour, height above ground, coverage, class, shadow amount).
    #[allow(clippy::too_many_arguments)]
    fn town(&self, town: &TownInfo, p: DVec3, gsd: f64, fw: f64, slope: f64, shadows: bool, pf: &PixFields) -> Option<(DVec3, f64, f64, u8, f64, DVec3)> {
        let pal = &self.pal;
        let d = p - town.center;
        let q0 = DVec2::new(d.dot(town.ex), d.dot(town.ey));
        let dist = q0.length();
        if dist > town.radius * 1.6 {
            return None;
        }
        let edge_n = pf.warp2;
        let urban = 1.0 - smoothstep(0.45, 1.0, dist / (town.radius * (1.0 + 0.45 * edge_n)));
        let urban = urban * (1.0 - smoothstep(0.2, 0.4, slope));
        if urban <= 0.0 {
            return None;
        }
        // organic (curved) streets in old towns
        let warp = DVec2::new(perlin3(town.lot.to_bits(), p / 350.0), perlin3(town.lot.to_bits() ^ 9, p / 350.0)) * (35.0 * town.organic);
        let q = q0 + warp;
        let b = town.block;
        let s = town.street;
        let bi = (q / b).floor();
        let bq = q - bi * b; // within block cell [0,b)
        // streets on the cell borders
        let ds = bq.x.min(b - bq.x).min(bq.y).min(b - bq.y);
        let big_street = (bi.x as i64).rem_euclid(4) == 0 && bq.x < b * 0.5 || (bi.y as i64).rem_euclid(4) == 0 && bq.y < b * 0.5;
        let sw = if big_street { s * 1.4 } else { s } * 0.5;
        let street = band_cov(ds, sw, fw.max(gsd * 0.5));
        let bh = hash2(town.lot.to_bits() ^ 0xB10C, bi.x as i64, bi.y as i64);
        let block_kind = u01k(bh, 1);
        let mut col;
        let mut height = 0.0;
        let mut class = lc::URBAN;
        let mut shadow = 0.0;
        // block content
        let inner = DVec2::new(bq.x - sw, bq.y - sw);
        let bsz = b - 2.0 * sw;
        let central = 1.0 - dist / town.radius;
        let buildings_resolved = band(town.lot, gsd);
        if block_kind < 0.08 + 0.1 * (1.0 - urban) {
            // park / green
            col = pal.grass_wet * (1.0 + 0.15 * pf.detail);
            class = lc::GRASS;
        } else if block_kind < 0.14 {
            col = pal.concrete * 0.85; // parking / plaza
        } else {
            let industrial = block_kind > 0.93 && central < 0.4;
            let (lot_w, rows) = if industrial { (bsz, 1.0) } else { (town.lot, 2.0) };
            let li = (inner.x / lot_w).floor();
            let lj = (inner.y / (bsz / rows)).floor();
            let lx = inner.x - li * lot_w;
            let ly = inner.y - lj * (bsz / rows);
            let lh = hash2(bh, li as i64, lj as i64);
            let yard = mixc(pal.grass_wet, pal.soil[0], 0.4) * (1.0 + 0.2 * pf.detail);
            col = yard;
            // building footprint inside the lot
            let setb = if industrial { 6.0 } else { 2.0 + 3.0 * u01k(lh, 1) };
            let fwid = lot_w - 2.0 * setb.min(lot_w * 0.3);
            let fdep = bsz / rows - setb - 2.0 - 4.0 * u01k(lh, 2);
            let exists = u01k(lh, 3) < (0.25 + 0.75 * urban) && inner.x >= 0.0 && inner.y >= 0.0 && inner.x < bsz && inner.y < bsz;
            if exists && fwid > 3.0 && fdep > 3.0 {
                let cx = lx - lot_w * 0.5;
                let cy = if lj as i64 % 2 == 0 { ly - setb - fdep * 0.5 } else { ly - (bsz / rows - setb - fdep * 0.5) };
                let ex = fwid * 0.5 - cx.abs();
                let ey = fdep * 0.5 - cy.abs();
                let inside = (ex.min(ey) / fw + 0.5).clamp(0.0, 1.0);
                let tall = (central.max(0.0)).powf(2.0) * town.height;
                let hb = if industrial { 8.0 + 6.0 * u01k(lh, 4) } else { 3.5 + 4.0 * u01k(lh, 4) + 40.0 * tall * u01k(lh, 5) };
                let flat_roof = industrial || hb > 12.0 || u01k(lh, 6) < 0.25;
                let ri = ((town.roof_style * 3.0 + u01k(lh, 7) * 4.0) as usize) % 7;
                let mut roof = if industrial { mixc(pal.roofs[4], pal.roofs[2], u01k(lh, 8)) } else { pal.roofs[ri] };
                roof *= 0.85 + 0.3 * u01k(lh, 9);
                let mut h_here = hb;
                if !flat_roof {
                    // pitched roof along the longer axis
                    let (half, dperp) = if fwid > fdep { (fdep * 0.5, ey) } else { (fwid * 0.5, ex) };
                    let ridge = 0.35 * half;
                    h_here = hb + ridge * (dperp / half).clamp(0.0, 1.0);
                } else {
                    roof *= 1.0 - 0.15 * band_cov(ex.min(ey), 0.6, fw); // parapet edge
                }
                if inside > 0.0 {
                    col = mixc(col, roof, inside * buildings_resolved + (1.0 - buildings_resolved) * 0.45);
                    height = h_here * inside;
                    if inside > 0.5 {
                        class = lc::BUILDING;
                    }
                }
                let _ = slope;
            }
            // mean appearance when lots are unresolved
            if buildings_resolved < 1.0 {
                let mean_roof = mixc(pal.roofs[(town.roof_style * 6.99) as usize], pal.concrete, 0.3);
                col = mixc(mixc(yard, mean_roof, 0.45 * urban), col, buildings_resolved);
                height = lerp(5.0 * urban * 0.4, height, buildings_resolved);
            }
            // cast shadows of buildings onto the ground (march toward the sun)
            if shadows && buildings_resolved > 0.0 && height < 0.5 {
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
        col = mixc(col, pal.asphalt * (1.0 + 0.05 * pf.detail), street);
        if street > 0.5 {
            class = lc::ROAD;
        }
        height *= 1.0 - street;

        // ---- night lights: pools of light under street lamps, lit plazas / industrial yards
        let lamp_sp = 26.0 + 12.0 * town.organic;
        let lamp_col = if town.roof_style < 0.55 { DVec3::new(1.0, 0.58, 0.24) } else { DVec3::new(0.86, 0.9, 1.0) };
        let lamp_res = band(lamp_sp, gsd);
        let mut emission = DVec3::ZERO;
        if lamp_res > 0.0 {
            let ql = (q / lamp_sp).round() * lamp_sp;
            let lq = ql - (ql / b).floor() * b;
            let dsl = lq.x.min(b - lq.x).min(lq.y).min(b - lq.y);
            let lh = hash2(town.lot.to_bits() ^ 0x1A3B, (ql.x / lamp_sp) as i64, (ql.y / lamp_sp) as i64);
            if dsl < sw * 1.6 && u01k(lh, 1) < 0.9 {
                let d2 = (q - ql).length_squared();
                let pool = 0.35 * (-d2 / (2.0 * 7.0 * 7.0)).exp();
                let core = 4.0 * (-d2 / (2.0 * 0.7 * 0.7)).exp() * band(1.5, gsd);
                emission += lamp_col * (pool + core) * (0.7 + 0.6 * u01k(lh, 2));
            }
        }
        // prefiltered mean when lamps are unresolved (town glow)
        emission = emission * lamp_res + lamp_col * 0.07 * (1.0 - lamp_res);
        if block_kind < 0.14 && block_kind >= 0.08 + 0.1 * (1.0 - urban) {
            emission += lamp_col * 0.12; // lit plaza / parking
        }
        if block_kind > 0.93 && central < 0.4 {
            emission += DVec3::new(0.9, 0.95, 1.0) * 0.08; // industrial yard floodlights
        }
        emission *= urban;
        Some((col, height, urban, class, shadow * (1.0 - street * 0.5), emission))
    }

    fn building_height_at(&self, town: &TownInfo, p: DVec3, gsd: f64, pf: &PixFields) -> f64 {
        match self.town(town, p, gsd, 0.01, 0.0, false, pf) {
            Some((_, h, cov, _, _, _)) => h * cov,
            None => 0.0,
        }
    }
}
