//! Macro-scale world model ("pass A"): continents, relief, hydrology and climate.
//!
//! Evaluated once per pixel centre. All functions are smooth at sub-pixel scale (band-limited by the
//! pixel GSD), so the fine detail pass ("pass B", `surface.rs`) can interpolate these values.

use crate::config::Config;
use crate::noise::*;
use geodesy::{Ellipsoid, Geodetic};
use glam::{DVec2, DVec3};

#[path = "hydro.rs"]
mod hydro;
pub use hydro::{RiverHit, Seg};

/// Drainage data precomputed for an area (a tile, a chunk of a row): exact shortcuts for
/// `terrain_impl`.
pub struct NearSegs<'a> {
    /// pieces that can reach the area where the ground before carving is ≤ `h_max`
    /// (`World::local_segments`)
    pub local: &'a [Seg],
    pub h_max: f64,
    /// `World::sink_lakes` of the full piece list
    pub sinks: &'a [(u64, DVec3, f64)],
    /// the instances of the relief families that can reach the area
    pub inst: &'a [Vec<crate::instances::Instance>],
}

/// What `terrain_impl` evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Full,
    /// everything but lakes (used to find lake spill levels)
    NoLakes,
    /// relief only: no water, no land-use sites or roads (drainage heights)
    Relief,
}

/// Sampling context of a point on the ellipsoid surface.
#[derive(Clone, Copy, Debug)]
pub struct Ctx {
    /// ECEF position on the ellipsoid (h = 0), meters. Noise domain.
    pub p: DVec3,
    pub up: DVec3,
    pub east: DVec3,
    pub north: DVec3,
    pub lat: f64,
    pub lon: f64,
    /// Pixel ground sample distance (m): features smaller than ~2 gsd are averaged out.
    pub gsd: f64,
}

impl Ctx {
    pub fn new(lat: f64, lon: f64, gsd: f64, ell: &Ellipsoid) -> Self {
        let p = geodesy::geodetic2ecef(Geodetic::new(lat, lon, 0.0), ell);
        let (sl, cl) = lat.sin_cos();
        let (so, co) = lon.sin_cos();
        Ctx { p, up: DVec3::new(cl * co, cl * so, sl), east: DVec3::new(-so, co, 0.0), north: DVec3::new(-sl * co, -sl * so, cl), lat, lon, gsd }
    }
    /// Point offset by (east, north) meters in the tangent plane (good for small offsets).
    #[inline]
    pub fn offset(&self, de: f64, dn: f64) -> DVec3 {
        self.p + self.east * de + self.north * dn
    }
}

/// Water body kinds.
pub mod water {
    pub const NONE: u8 = 0;
    pub const OCEAN: u8 = 1;
    pub const LAKE: u8 = 2;
    pub const RIVER: u8 = 3;
}

/// Pass-A result at a pixel centre.
#[derive(Clone, Copy, Debug, Default)]
pub struct Terrain {
    /// Bare-earth elevation (m above ellipsoid). Under water this is the bed.
    pub ground: f64,
    /// Standing water level (ocean/lake) or `NEG_INFINITY`.
    pub water: f64,
    pub water_kind: u8,
    /// Signed distance-like value to the river centre line (m), and river half width (m).
    pub river_d: f64,
    pub river_hw: f64,
    pub river_level: f64,
    /// River is perennial (1) or a dry wash (0).
    pub river_wet: f64,
    /// Climate: mean annual temperature (°C) at this elevation, moisture 0..1.
    pub temp: f64,
    pub moist: f64,
    /// 0..1 masks
    pub mountain: f64,
    pub rock_expect: f64,
    pub sand: f64,
    pub floodplain: f64,
    pub mesa: f64,
    /// signed continent value (>0 land)
    pub cont: f64,
    /// Agricultural intensity 0..1 (before slope test).
    pub agri: f64,
    /// Settlement suitability 0..1.
    pub habit: f64,
    /// Erosion gully signal (−: channel / valley floor, +: spur / ridge), ~[-1.5, 1.5].
    pub gully: f64,
    /// Regional style parameters 0..1 (colour variations etc.)
    pub style: [f64; 4],
    /// Signed distance-like values to the major / minor road centre lines (m).
    pub road_major: f64,
    pub road_minor: f64,
    /// Land-use region (field system) Worley site.
    pub region: Site,
    /// Nearest potential town site.
    pub town: Site,
    /// Ecoregion (`crate::eco`): the two nearest sites of the warped lattice, the distance to
    /// their border.
    pub eco: Site,
}

/// A Worley site: id hash, centre (ECEF, on the surface), distance to its Voronoi border (m).
#[derive(Clone, Copy, Debug, Default)]
pub struct Site {
    pub id: u64,
    /// id of the neighbouring site across the nearest border
    pub id2: u64,
    pub center: DVec3,
    /// the neighbouring site's point
    pub center2: DVec3,
    pub dist: f64,
    pub edge: f64,
}

/// Large-scale (≥ ~25 km) fields. Smooth enough to be sampled on a coarse grid per tile and
/// interpolated (see `tile.rs`); all non-linear mappings are applied after interpolation.
#[derive(Clone, Copy, Debug, Default)]
pub struct Macro {
    pub cont: f64,
    pub plateau: f64,
    pub belt: f64,
    pub belt2: f64,
    pub belt_var: f64,
    pub hill_amp: f64,
    pub rough: f64,
    pub temp: f64,
    pub moist: f64,
    pub mesa: f64,
    pub sand: f64,
    pub agri: f64,
    pub style: [f64; 4],
    pub river_width: f64,
    pub mtn_warp: [f64; 2],
    /// Smooth per-pixel inputs interpolated from a coarse grid (tile generator), or None (exact).
    pub pre: Option<Pre>,
}

/// Smooth inputs of `terrain_impl` precomputed on a coarse grid: the gradient (east, north) of
/// the low-passed relief that steers the erosion gullies, and the road networks' noise value
/// and gradient (east, north) after the domain warp.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pre {
    pub gully: Option<[f64; 2]>,
    pub road_major: Option<[f64; 3]>,
    pub road_minor: Option<[f64; 3]>,
    /// the long octaves of the relief: ridged (sum, low-passed sum, weight) and hills (sum,
    /// low-passed sum), split at the wavelengths `relief_cut` (ridged, hills)
    pub relief: Option<[f64; 5]>,
    pub relief_cut: [f64; 2],
    /// meander warps of the drainage levels, the warp of the region lattice
    pub river_warp: [Option<[f64; 2]>; 4],
    pub region_warp: Option<[f64; 3]>,
    /// the warp of the ecoregion lattice
    pub eco_warp: Option<[f64; 3]>,
    /// the long gully octaves (state of `gullies_part`, split at `relief_cut[1]`)
    pub gully_oct: Option<[f64; 4]>,
    /// floodplain-edge noise per drainage level
    pub floodplain: [Option<f64>; 4],
    /// the two nearest sites of the lake, region, town and ecoregion lattices where known (the
    /// same for a whole block of the tile's coarse grid)
    pub sites: [Option<[(u64, DVec3); 2]>; 4],
}

/// Indices of [`Pre::sites`].
pub const SITE_LAKE: usize = 0;
pub const SITE_REGION: usize = 1;
pub const SITE_TOWN: usize = 2;
pub const SITE_ECO: usize = 3;

impl Macro {
    /// Bilinear interpolation of four corner values (a b / c d).
    pub fn bilerp(a: &Macro, b: &Macro, c: &Macro, d: &Macro, fx: f64, fy: f64) -> Macro {
        let l = |x: f64, y: f64, z: f64, w: f64| {
            let t = x + (y - x) * fx;
            let u = z + (w - z) * fx;
            t + (u - t) * fy
        };
        let l4 = |f: &dyn Fn(&Macro) -> f64| l(f(a), f(b), f(c), f(d));
        Macro {
            cont: l4(&|m| m.cont),
            plateau: l4(&|m| m.plateau),
            belt: l4(&|m| m.belt),
            belt2: l4(&|m| m.belt2),
            belt_var: l4(&|m| m.belt_var),
            hill_amp: l4(&|m| m.hill_amp),
            rough: l4(&|m| m.rough),
            temp: l4(&|m| m.temp),
            moist: l4(&|m| m.moist),
            mesa: l4(&|m| m.mesa),
            sand: l4(&|m| m.sand),
            agri: l4(&|m| m.agri),
            style: [l4(&|m| m.style[0]), l4(&|m| m.style[1]), l4(&|m| m.style[2]), l4(&|m| m.style[3])],
            river_width: l4(&|m| m.river_width),
            mtn_warp: [l4(&|m| m.mtn_warp[0]), l4(&|m| m.mtn_warp[1])],
            pre: None,
        }
    }
}

pub struct World {
    pub cfg: Config,
    pub ell: Ellipsoid,
    pub(crate) seed: u64,
    pub(crate) cont: Fbm,
    pub(crate) cont_warp: [Fbm; 3],
    pub(crate) belt: Fbm,
    pub(crate) belt2: Fbm,
    pub(crate) belt_var: Fbm,
    pub(crate) mtn_frames: OctaveFrames,
    pub(crate) mtn_warp: [Fbm; 2],
    pub(crate) plateau: Fbm,
    pub(crate) hills: OctaveFrames,
    pub(crate) hill_amp: Fbm,
    pub(crate) rough: Fbm,
    pub(crate) micro: Fbm,
    pub(crate) temp_n: Fbm,
    pub(crate) moist_n: Fbm,
    pub(crate) river_warp: [Fbm; 2],
    pub(crate) river_width_n: Fbm,
    pub(crate) mesa_n: Fbm,
    pub(crate) dune_frames: OctaveFrames,
    pub(crate) sand_n: Fbm,
    pub(crate) agri_n: Fbm,
    pub(crate) style_n: [Fbm; 4],
    pub(crate) road_major: Fbm,
    pub(crate) road_minor: Fbm,
    pub(crate) home: Option<(DVec3, f64, f64)>,
    /// Hash of the whole config: key of the thread-local caches, so generators with the same
    /// seed but different settings in one process do not share cached hydrology / lakes.
    pub(crate) cache_key: u64,
}

const KM: f64 = 1000.0;

impl World {
    pub fn new(cfg: Config) -> Self {
        let ell = Ellipsoid::from_a_invf(cfg.planet.a, cfg.planet.inv_f);
        let s = cfg.seed;
        let k = |i: u64| mix64(s.wrapping_mul(0x9E37_79B9).wrapping_add(i * 7919));
        let c = &cfg.continents;
        let r = &cfg.relief;

        let cw = c.wavelength_km * KM;
        let home = cfg.home.as_ref().map(|h| {
            let p = geodesy::geodetic2ecef(Geodetic::from_deg(h.lat, h.lon, 0.0), &ell);
            (p, h.radius_km * KM, h.strength)
        });
        let cache_key =
            serde_yaml::to_string(&cfg).unwrap_or_default().bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
        World {
            cache_key,
            seed: s,
            cont: Fbm::new(k(1), cw, 7, 2.0, 0.52),
            cont_warp: [Fbm::new(k(2), cw * 0.8, 3, 2.0, 0.5), Fbm::new(k(3), cw * 0.8, 3, 2.0, 0.5), Fbm::new(k(4), cw * 0.8, 3, 2.0, 0.5)],
            belt: Fbm::new(k(5), r.belt_wavelength_km * KM, 3, 2.0, 0.45),
            belt2: Fbm::new(k(6), r.belt_wavelength_km * KM * 0.43, 3, 2.0, 0.5),
            belt_var: Fbm::new(k(7), r.belt_wavelength_km * KM * 0.3, 3, 2.0, 0.5),
            mtn_frames: OctaveFrames::new(k(8), 24),
            mtn_warp: [Fbm::new(k(9), 30.0 * KM, 3, 2.0, 0.5), Fbm::new(k(10), 30.0 * KM, 3, 2.0, 0.5)],
            plateau: Fbm::new(k(11), 1100.0 * KM, 3, 2.0, 0.5),
            hills: OctaveFrames::new(k(12), 24),
            hill_amp: Fbm::new(k(13), 180.0 * KM, 3, 2.0, 0.5),
            rough: Fbm::new(k(14), 90.0 * KM, 3, 2.0, 0.5),
            micro: Fbm::new(k(15), 160.0, 9, 2.0, 0.55),
            temp_n: Fbm::new(k(16), 900.0 * KM, 3, 2.0, 0.5),
            moist_n: Fbm::new(k(17), 1400.0 * KM, 5, 2.0, 0.55),
            river_warp: [Fbm::new(k(20), 12.0 * KM, 3, 2.0, 0.5), Fbm::new(k(21), 12.0 * KM, 3, 2.0, 0.5)],
            river_width_n: Fbm::new(k(22), 300.0 * KM, 2, 2.0, 0.5),
            mesa_n: Fbm::new(k(23), 250.0 * KM, 3, 2.0, 0.5),
            dune_frames: OctaveFrames::new(k(24), 8),
            sand_n: Fbm::new(k(25), 350.0 * KM, 3, 2.0, 0.5),
            agri_n: Fbm::new(k(26), 70.0 * KM, 3, 2.0, 0.5),
            style_n: [
                Fbm::new(k(27), 120.0 * KM, 3, 2.0, 0.5),
                Fbm::new(k(28), 60.0 * KM, 3, 2.0, 0.5),
                Fbm::new(k(29), 200.0 * KM, 3, 2.0, 0.5),
                Fbm::new(k(30), 25.0 * KM, 3, 2.0, 0.5),
            ],
            road_major: Fbm::new(k(31), 34.0 * KM, 2, 2.2, 0.4),
            road_minor: Fbm::new(k(32), 7.5 * KM, 2, 2.2, 0.4),
            home,
            ell,
            cfg,
        }
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Continent field (>0 land), smooth at ≥100 km scales.
    pub fn continent(&self, p: DVec3, gsd: f64) -> f64 {
        let cw = self.cont.wavelength;
        let warp = DVec3::new(self.cont_warp[0].eval(p, gsd), self.cont_warp[1].eval(p, gsd), self.cont_warp[2].eval(p, gsd)) * (self.cfg.continents.warp * cw);
        // keep the continental field coarse: octaves down to ~30 km; finer coast detail comes
        // from the hill/micro relief crossing sea level (fractal coastlines at every zoom).
        let gl = gsd.max(15.0 * KM);
        let mut c = self.cont.eval(p + warp, gl) * self.cont.norm() * 1.6 - self.cfg.continents.threshold;
        if let Some((hp, r, st)) = self.home {
            let d = (p - hp).length();
            let w = 1.0 - smoothstep(0.4 * r, r, d);
            c = lerp(c, c.max(0.18 + 0.1 * c), w * st);
        }
        c
    }

    /// Ridged multifractal mountains, 0..~1.
    fn ridged(&self, p: DVec3, gsd: f64, sharp: f64) -> (f64, f64) {
        let st = self.ridged_part(p, gsd, sharp, 0.0, true, [0.0, 0.0, 1.0]);
        (st[0] * 0.5, st[1] * 0.5)
    }

    /// The octaves of [`World::ridged`] of wavelength >= `cut` (`low`, from the start) or < `cut`
    /// (not `low`, continuing from the state `st` of the long ones). State: sum, low-passed sum,
    /// weight of the next octave (before the final scaling).
    #[allow(clippy::too_many_arguments)]
    fn ridged_part(&self, p: DVec3, gsd: f64, sharp: f64, cut: f64, low: bool, st: [f64; 3]) -> [f64; 3] {
        let [mut sum, mut low_sum, mut weight] = st;
        let mut lam = 14.0 * KM;
        let mut amp = 1.0;
        for i in 0..self.mtn_frames.rot.len() {
            if low && lam < cut {
                break;
            }
            // ridges contain harmonics above the octave frequency: band-limit more strictly
            let wb = band(lam, 1.6 * gsd);
            if wb <= 0.0 {
                break;
            }
            if low || lam < cut {
                let q = self.mtn_frames.rot[i] * (p / lam) + self.mtn_frames.off[i];
                let n = perlin3(self.mtn_frames.seeds[i], q);
                let mut r = (1.0 - n.abs()).max(0.0);
                r = r.powf(sharp);
                r *= weight;
                weight = (r * 1.8).clamp(0.0, 1.0);
                sum += r * amp * wb;
                if lam > 6.0 * KM {
                    low_sum += r * amp * wb;
                }
            }
            lam *= 0.5;
            amp *= if lam > 1.0 * KM { 0.58 } else { 0.42 };
        }
        [sum, low_sum, weight]
    }

    /// Band-limited hill fBm with regionally varying roughness. Returns (full, lowpass>=15km).
    fn hills(&self, p: DVec3, gsd: f64, gain: f64) -> (f64, f64) {
        let [sum, low] = self.hills_part(p, gsd, gain, 0.0, true);
        (sum * 0.7, low * 0.7)
    }

    /// The octaves of [`World::hills`] of wavelength >= `cut` (`low`) or < `cut` (not `low`):
    /// sum and low-passed sum before the final scaling (the two parts add up).
    fn hills_part(&self, p: DVec3, gsd: f64, gain: f64, cut: f64, low: bool) -> [f64; 2] {
        let mut lam = 9.0 * KM;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut low_sum = 0.0;
        for i in 0..self.hills.rot.len() {
            if low && lam < cut {
                break;
            }
            let wb = band(lam, gsd);
            if wb <= 0.0 || lam < 20.0 {
                break;
            }
            if low || lam < cut {
                let q = self.hills.rot[i] * (p / lam) + self.hills.off[i];
                let n = perlin3(self.hills.seeds[i], q);
                sum += n * amp * wb;
                if lam >= 4.0 * KM {
                    low_sum += n * amp * wb;
                }
            }
            lam *= 0.5;
            amp *= gain;
        }
        [sum, low_sum]
    }

    /// One octave of gradient-aligned gully noise in a 3D jittered lattice (point `q` in lattice
    /// units). Returns the stripe value and its derivative (lattice units).
    pub(crate) fn gully_octave(seed: u64, q: DVec3, dir: DVec3) -> (f64, DVec3) {
        let qf = q.floor();
        let (ix, iy, iz) = (qf.x as i64, qf.y as i64, qf.z as i64);
        let f = q - qf;
        let mut v = 0.0;
        let mut d = DVec3::ZERO;
        let mut wt = 0.0;
        // kernel (1 − d²/1.5²)³: close to the Gaussian e^(−2d²) it replaces but exactly zero
        // beyond 1.5 cells, so the ±2-cell window holds every contributing point (points of cells
        // 3 away are ≥ 2 cells off). The truncated Gaussian creased the terrain along the lattice
        // planes; a support of 1 cell made the blend switch abruptly between neighbouring points,
        // cutting straight cliffs and grooves along their bisectors. Every point lies within
        // √3/2 of some point, so the weight sum stays well above zero.
        // a point of cell offset `o` lies at f − o − [0, 0.5) per axis: cells whose box is 1.5
        // cells away or more contribute nothing and are skipped before hashing (same result)
        let gap = |f: f64, o: i64| -> f64 {
            let t = f - o as f64;
            if t < 0.0 {
                t * t
            } else if t > 0.5 {
                (t - 0.5) * (t - 0.5)
            } else {
                0.0
            }
        };
        const R2: f64 = 2.25 + 1e-9;
        for dz in -2..=2i64 {
            let gz = gap(f.z, dz);
            if gz > R2 {
                continue;
            }
            for dy in -2..=2i64 {
                let gzy = gz + gap(f.y, dy);
                if gzy > R2 {
                    continue;
                }
                for dx in -2..=2i64 {
                    if gzy + gap(f.x, dx) > R2 {
                        continue;
                    }
                    let h = hash3(seed, ix + dx, iy + dy, iz + dz);
                    let jit = DVec3::new(u01k(h, 1), u01k(h, 2), u01k(h, 3)) * 0.5;
                    let pp = f - DVec3::new(dx as f64, dy as f64, dz as f64) - jit;
                    let d2 = pp.length_squared();
                    if d2 >= 2.25 {
                        continue;
                    }
                    let k = 1.0 - d2 / 2.25;
                    let w = k * k * k;
                    wt += w;
                    let mag = pp.dot(dir) * std::f64::consts::TAU;
                    let (s, c) = mag.sin_cos();
                    v += c * w;
                    d -= dir * (s * w);
                }
            }
        }
        (v / wt, d / wt)
    }

    /// Erosion-like gullies: stripes running down the large-scale slope `grad` (m/m, tangent),
    /// bent by the gullies of previous octaves → dendritic patterns. Returns a height offset in
    /// units of the first octave amplitude (roughly within [-1.5, 1.5]).
    fn gullies(&self, p: DVec3, up: DVec3, grad: DVec3, gsd: f64, lam0: f64) -> f64 {
        self.gullies_part(p, up, grad, gsd, lam0, 0.0, true, [0.0; 4])[0]
    }

    /// The octaves of [`World::gullies`] of wavelength >= `cut` (`low`, from the start) or < `cut`
    /// (not `low`, continuing from the state `st` of the long ones). State: height and the
    /// accumulated derivative that bends the next octaves.
    #[allow(clippy::too_many_arguments)]
    fn gullies_part(&self, p: DVec3, up: DVec3, grad: DVec3, gsd: f64, lam0: f64, cut: f64, low: bool, st: [f64; 4]) -> [f64; 4] {
        let dir0 = up.cross(grad).normalize_or_zero();
        let mut a = 1.0;
        let mut lam = lam0;
        let mut h = st[0];
        let mut hd = DVec3::new(st[1], st[2], st[3]);
        for i in 0..5u64 {
            if low && lam < cut {
                break;
            }
            let wb = band(lam, gsd);
            if wb <= 0.0 {
                break;
            }
            if low || lam < cut {
                // bending rotates the stripe direction; keep its length (= stripe frequency) at 1
                let dir = (dir0 + up.cross(hd) * 0.7).normalize_or_zero();
                let (v, d) = Self::gully_octave(self.seed ^ (0xE205 + i * 0x9E37), p / lam, dir);
                h += v * a * wb;
                hd += d * (a * wb);
            }
            a *= 0.45;
            lam *= 0.5;
        }
        [h, hd.x, hd.y, hd.z]
    }

    /// Sand dunes: main ridges perpendicular to a slowly varying, locally bent wind direction
    /// (oriented stripe kernel → curved transverse / barchanoid crests), plus isotropic ridged
    /// secondary dunes.
    pub fn dunes(&self, ctx: &Ctx, m: &Macro, gsd: f64) -> f64 {
        // NB: never modulate a noise wavelength with a spatially varying field — with absolute
        // (ECEF) coordinates that shears the noise into streaks. Wavelengths are constants.
        let lam0 = 520.0;
        let mut sum = 0.0;
        if band(lam0, gsd) > 0.0 {
            let bend = 0.7 * perlin3(self.seed ^ 0xB3D, ctx.p / (4.0 * lam0)) + 0.3 * perlin3(self.seed ^ 0xB3E, ctx.p / (1.5 * lam0));
            let th = 2.5 * m.style[0] + 0.8 * m.style[2] + bend;
            let wind = ctx.east * th.cos() + ctx.north * th.sin();
            let (v, _) = Self::gully_octave(self.seed ^ 0xD0E, ctx.p / lam0, wind);
            sum += (0.5 + 0.5 * v).max(0.0).powf(1.25) * band(lam0, gsd);
        }
        let mut lam = lam0 * 0.33;
        let mut amp = 0.3;
        for i in 0..3usize {
            let wb = band(lam, 1.5 * gsd);
            if wb <= 0.0 {
                break;
            }
            let q = self.dune_frames.rot[i] * (ctx.p / lam) + self.dune_frames.off[i];
            let r = 1.0 - perlin3(self.dune_frames.seeds[i], q).abs();
            sum += r * r * amp * wb;
            amp *= 0.35;
            lam *= 0.4;
        }
        sum
    }

    /// Mountain domain warp (exact). Short wavelength x 9 km gain: too fine for the 16-px macro
    /// grid (bilinear error ~100 m in ridge position), so tiles evaluate it per pixel.
    pub fn mtn_warp_at(&self, p: DVec3, gsd: f64) -> [f64; 2] {
        [self.mtn_warp[0].eval(p, gsd), self.mtn_warp[1].eval(p, gsd)]
    }

    /// Evaluate all large-scale fields at a point.
    pub fn macro_at(&self, p: DVec3, gsd: f64) -> Macro {
        Macro {
            cont: self.continent(p, gsd),
            plateau: self.plateau.eval(p, gsd) * self.plateau.norm() * 1.8,
            belt: self.belt.eval(p, gsd) * self.belt.norm() * 2.0,
            belt2: self.belt2.eval(p, gsd) * self.belt2.norm() * 2.0,
            belt_var: self.belt_var.eval(p, gsd) * self.belt_var.norm() * 1.6,
            hill_amp: self.hill_amp.eval(p, gsd) * self.hill_amp.norm() * 1.6,
            rough: self.rough.eval(p, gsd) * self.rough.norm() * 1.6,
            temp: 5.0 * self.temp_n.eval(p, 50.0 * KM),
            moist: self.moist_n.eval(p, 20.0 * KM) * self.moist_n.norm() * 1.6,
            mesa: self.mesa_n.eval(p, gsd) * self.mesa_n.norm() * 1.8,
            sand: self.sand_n.eval(p, gsd) * self.sand_n.norm() * 1.8,
            agri: self.agri_n.eval(p, gsd) * self.agri_n.norm() * 1.8,
            style: [self.style_n[0].eval(p, gsd), self.style_n[1].eval(p, gsd), self.style_n[2].eval(p, gsd), self.style_n[3].eval(p, gsd)],
            river_width: self.river_width_n.eval(p, gsd),
            mtn_warp: self.mtn_warp_at(p, gsd),
            pre: None,
        }
    }

    /// Worley cell of the lake / region / town lattice at `p`, from the sites known for the pixel
    /// when there.
    fn site_cell(&self, m: &Macro, which: usize, key: u64, p: DVec3, cell: f64, jitter: f64) -> Cell3 {
        match m.pre.and_then(|pre| pre.sites[which]) {
            Some(s) => worley3_from(p, cell, s),
            None => worley3(self.seed ^ key, p, cell, jitter),
        }
    }

    /// Wavelength (m) of the floodplain-edge noise of drainage level `li`.
    pub(crate) fn floodplain_wavelength(&self, li: usize) -> f64 {
        let lc = &self.cfg.hydro.levels[li];
        400.0 + 0.75 * (lc.width_m[0] + lc.width_m[1])
    }

    /// Noise of the floodplain edge of drainage level `li` (a constant wavelength per level: one
    /// following the channel width sheared the noise into streaks).
    fn floodplain_noise(&self, li: usize, p: DVec3) -> f64 {
        perlin3(0xF10D ^ li as u64, p / self.floodplain_wavelength(li))
    }

    /// Warp (m) of the lookup in the region lattice: curvy borders between field systems.
    fn region_warp(&self, p: DVec3) -> DVec3 {
        let region_cell = self.cfg.landuse.region_km * KM;
        DVec3::new(
            perlin3(self.seed ^ 0xA1, p / (0.9 * region_cell)),
            perlin3(self.seed ^ 0xA2, p / (0.9 * region_cell)),
            perlin3(self.seed ^ 0xA3, p / (0.9 * region_cell)),
        ) * (0.18 * region_cell)
            + DVec3::new(perlin3(self.seed ^ 0xA4, p / 1500.0), perlin3(self.seed ^ 0xA5, p / 1500.0), perlin3(self.seed ^ 0xA6, p / 1500.0)) * 120.0
    }

    /// The smooth inputs `Pre` at a point (for the coarse grid): the low-passed relief gradient
    /// for the gullies (`gully`), the road networks (`roads`).
    pub fn pre_at(&self, ctx: &Ctx, m: &Macro, gully: bool, roads: bool, relief_cut: Option<[f64; 2]>) -> Pre {
        let mut pre = Pre::default();
        if let Some(cut) = relief_cut {
            let wp = DVec2::new(m.mtn_warp[0], m.mtn_warp[1]) * 9.0 * KM;
            let pw = ctx.p + ctx.east * wp.x + ctx.north * wp.y;
            let [rs, rl, rw] = self.ridged_part(pw, ctx.gsd, 1.6 + 0.8 * m.style[2], cut[0], true, [0.0, 0.0, 1.0]);
            let [hs, hl] = self.hills_part(ctx.p, ctx.gsd, 0.47 + 0.08 * m.rough, cut[1], true);
            pre.relief = Some([rs, rl, rw, hs, hl]);
            pre.relief_cut = cut;
            // smooth warps whose shorter wavelength is >= the cut
            for li in 0..self.cfg.hydro.levels.len().min(4) {
                if 0.37 * self.meander_wavelength(li) >= cut[1] {
                    pre.river_warp[li] = Some(self.meander_warp(li, ctx.p));
                }
                if self.floodplain_wavelength(li) >= cut[1] {
                    pre.floodplain[li] = Some(self.floodplain_noise(li, ctx.p));
                }
            }
            let region_cell = self.cfg.landuse.region_km * KM;
            if 1500.0f64.min(0.9 * region_cell) >= cut[1] {
                pre.region_warp = Some(self.region_warp(ctx.p).to_array());
            }
            if crate::eco::warp_min_wavelength(self) >= cut[1] {
                pre.eco_warp = Some(crate::eco::warp(self, ctx.p).to_array());
            }
        }
        if gully {
            let lam_e = self.cfg.relief.gully_wavelength_m;
            let (mountain, amp_m) = self.mountain_mask(m);
            let hill_amp = self.hill_amplitude(m);
            let gain = 0.47 + 0.08 * m.rough;
            pre.gully = Some(self.low_relief_gradient(ctx, m, mountain, amp_m, hill_amp, gain, lam_e));
        }
        if roads {
            let warp = self.network_warp(ctx);
            let field = |f: &Fbm, amp: f64| -> [f64; 3] {
                let ([w0, w1], [g0, g1]) = warp;
                let qw = ctx.p + (ctx.east * w0 + ctx.north * w1) * amp;
                let (n, g) = f.eval_d(qw, ctx.gsd.max(200.0), 99);
                let grad = g + (g0 * ctx.east.dot(g) + g1 * ctx.north.dot(g)) * amp;
                [n, grad.dot(ctx.east), grad.dot(ctx.north)]
            };
            pre.road_major = Some(field(&self.road_major, 2500.0));
            pre.road_minor = Some(field(&self.road_minor, 700.0));
        }
        // the nearest sites of the lake, region and town lattices
        if relief_cut.is_some() {
            pre.sites[SITE_LAKE] = Some(worley3_sites(self.seed ^ 0x1A4E, ctx.p, self.cfg.hydro.lake_cell_km * KM, 0.85));
            let region_cell = self.cfg.landuse.region_km * KM;
            let pw = ctx.p + self.region_warp(ctx.p);
            pre.sites[SITE_REGION] = Some(worley3_sites(self.seed ^ 0x5E61, pw, region_cell, 0.9));
            pre.sites[SITE_TOWN] = Some(worley3_sites(self.seed ^ 0x70E1, ctx.p, self.cfg.landuse.town_cell_km * KM, 0.8));
            pre.sites[SITE_ECO] = Some(crate::eco::sites(self, ctx.p));
        }
        // the long gully octaves (they follow the low-passed relief gradient above)
        if let (Some(cut), Some([ge, gn])) = (relief_cut, pre.gully) {
            let grad = ctx.east * ge + ctx.north * gn;
            let lam_e = self.cfg.relief.gully_wavelength_m;
            pre.gully_oct = Some(self.gullies_part(ctx.p, ctx.up, grad, ctx.gsd, lam_e, cut[1], true, [0.0; 4]));
        }
        pre
    }

    /// Gradient (east, north, per meter) of the relief low-passed at half the gully wavelength,
    /// by finite differences.
    #[allow(clippy::too_many_arguments)]
    fn low_relief_gradient(&self, ctx: &Ctx, m: &Macro, mountain: f64, amp_m: f64, hill_amp: f64, gain: f64, lam_e: f64) -> [f64; 2] {
        let p = ctx.p;
        let gl = lam_e * 0.5;
        let low = |q: DVec3| -> f64 {
            let mut v = hill_amp * self.hills(q, gl, gain).0;
            if mountain > 1e-3 {
                let wp = DVec2::new(m.mtn_warp[0], m.mtn_warp[1]) * 9.0 * KM;
                v += amp_m * self.ridged(q + ctx.east * wp.x + ctx.north * wp.y, gl, 1.6 + 0.8 * m.style[2]).0;
            }
            v
        };
        let e = lam_e * 0.15;
        let h0 = low(p);
        [(low(p + ctx.east * e) - h0) / e, (low(p + ctx.north * e) - h0) / e]
    }

    /// Shared domain warp of the river / road networks: values and gradients (per meter).
    fn network_warp(&self, ctx: &Ctx) -> ([f64; 2], [DVec3; 2]) {
        let (w0, g0) = self.river_warp[0].eval_d(ctx.p, ctx.gsd, 99);
        let (w1, g1) = self.river_warp[1].eval_d(ctx.p, ctx.gsd, 99);
        ([w0, w1], [g0, g1])
    }

    /// Signed distance (m) to the zero iso-line of a warped noise network, from the noise value
    /// and its analytic gradient (chain rule through the warp).
    fn network_dist(&self, f: &Fbm, ctx: &Ctx, warp: &([f64; 2], [DVec3; 2]), amp: f64) -> f64 {
        self.network_dist_g(f, ctx, warp, amp).0
    }

    /// Like `network_dist`, also returning the gradient magnitude normalized by the network
    /// wavelength (small near noise extrema, where iso-lines form small closed loops).
    fn network_dist_g(&self, f: &Fbm, ctx: &Ctx, warp: &([f64; 2], [DVec3; 2]), amp: f64) -> (f64, f64) {
        let ([w0, w1], [g0, g1]) = *warp;
        let qw = ctx.p + (ctx.east * w0 + ctx.north * w1) * amp;
        let (n, g) = f.eval_d(qw, ctx.gsd.max(200.0), 99);
        // d n / d p = J^T g with J = I + amp (east ⊗ ∇w0 + north ⊗ ∇w1)
        let grad = g + (g0 * ctx.east.dot(g) + g1 * ctx.north.dot(g)) * amp;
        let gl = DVec2::new(grad.dot(ctx.east), grad.dot(ctx.north)).length().max(1e-12);
        (n / gl, gl * f.wavelength)
    }

    /// Climate (temperature °C, moisture 0..1) from macro fields at a given elevation.
    pub fn climate(&self, m: &Macro, lat: f64, elev: f64) -> (f64, f64) {
        let c = &self.cfg.climate;
        let la = lat.abs() / std::f64::consts::FRAC_PI_2;
        let mut t = c.equator_temp_c - c.pole_drop_c * la.powf(1.6) + m.temp;
        t -= c.lapse_rate_c_per_km * elev.max(0.0) / KM;
        let latd = lat.abs().to_degrees();
        let hadley = (-((latd - 24.0) / 9.0).powi(2)).exp();
        let mut w = 0.56 + 0.62 * m.moist;
        w -= 0.40 * hadley;
        w += 0.12 * (1.0 - smoothstep(0.0, 0.25, m.cont)); // coastal
        w -= 0.22 * smoothstep(0.15, 0.55, m.cont); // continental interiors
        w -= 0.10 * smoothstep(1500.0, 3500.0, elev); // high plateaus drier
        w += c.moisture_bias;
        (t, w.clamp(0.0, 1.0))
    }

    fn base_elevation(s: f64) -> f64 {
        if s > 0.0 {
            20.0 + 900.0 * s.powf(1.3)
        } else {
            20.0 - 120.0 * smoothstep(0.0, 0.04, -s) - 3800.0 * smoothstep(0.03, 0.35, -s)
        }
    }

    fn mountain_mask(&self, m: &Macro) -> (f64, f64) {
        let b1 = 1.0 - m.belt.abs();
        let b2 = 1.0 - m.belt2.abs();
        let belt = (b1 * 0.75 + b2 * 0.45 + 0.35 * m.belt_var).max(0.0);
        let mountain = smoothstep(0.62, 0.92, belt) * smoothstep(-0.04, 0.08, m.cont);
        let amp_m = self.cfg.relief.mountain_height_m * (0.55 + 0.45 * smoothstep(-0.4, 0.6, m.belt_var)) * mountain;
        (mountain, amp_m)
    }

    fn hill_amplitude(&self, m: &Macro) -> f64 {
        let land = smoothstep(-0.06, 0.05, m.cont);
        self.cfg.relief.hill_height_m * (0.15 + 0.85 * smoothstep(-0.5, 0.6, m.hill_amp)) * (0.25 + 0.75 * land)
    }

    /// Smooth (≥ ~5 km) elevation at a point; used for water levels.
    pub fn smooth_elevation(&self, p: DVec3) -> f64 {
        let gsd = 1500.0;
        let m = self.macro_at(p, gsd);
        let s = m.cont;
        let plateau = smoothstep(0.15, 0.55, m.plateau) * 900.0 * smoothstep(0.02, 0.15, s);
        let (_, amp_m) = self.mountain_mask(&m);
        let (_, hl_low) = self.hills(p, gsd, 0.47 + 0.08 * m.rough);
        Self::base_elevation(s) + plateau + 0.22 * amp_m + amp_m * 0.12 + self.hill_amplitude(&m) * hl_low
    }

    /// Lake surface level: the spill height of the basin (lowest rim sample), or None when the
    /// site is on a slope / in the sea. Cached per thread (pure function of the lake).
    /// Level of a lake that must exist (a river's closed basin): the spill height when the
    /// basin holds water, else just above the basin floor (a low levee holds it).
    fn lake_level_forced(&self, id: u64, center: DVec3, rad: f64) -> Option<f64> {
        thread_local! {
            static CACHE: std::cell::RefCell<FxHashMap<u64, Option<f64>>> = Default::default();
        }
        // the level also depends on `rad` (inflow widths): part of the key, so a cached value
        // never stands in for a different basin size (thread-order dependence)
        let key = id ^ self.cache_key.rotate_left(29) ^ rad.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
        if let Some(v) = CACHE.with(|c| c.borrow().get(&key).copied()) {
            return v;
        }
        let v = self.lake_level(id, center, rad).or_else(|| {
            let g = geodesy::ecef2geodetic(center, &self.ell);
            let cctx = Ctx::new(g.lat, g.lon, 20.0, &self.ell);
            let tc = self.terrain_impl(&cctx, &self.macro_at(cctx.p, cctx.gsd), Mode::NoLakes, None, None);
            (tc.water_kind == water::NONE && tc.ground > 1.0).then_some(tc.ground + 1.0)
        });
        CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() > 100_000 {
                c.clear();
            }
            c.insert(key, v);
        });
        v
    }

    fn lake_level(&self, id: u64, center: DVec3, rad: f64) -> Option<f64> {
        thread_local! {
            static CACHE: std::cell::RefCell<FxHashMap<u64, Option<f64>>> = Default::default();
        }
        // the level also depends on `rad` (inflow widths): part of the key, so a cached value
        // never stands in for a different basin size (thread-order dependence)
        let key = id ^ self.cache_key.rotate_left(17) ^ rad.to_bits().wrapping_mul(0x9E37_79B9_7F4A_7C15);
        if let Some(v) = CACHE.with(|c| c.borrow().get(&key).copied()) {
            return v;
        }
        let g = geodesy::ecef2geodetic(center, &self.ell);
        let cctx = Ctx::new(g.lat, g.lon, 20.0, &self.ell);
        let tc = self.terrain_impl(&cctx, &self.macro_at(cctx.p, cctx.gsd), Mode::NoLakes, None, None);
        let v = if tc.water_kind != water::NONE || tc.ground < 1.0 {
            None
        } else {
            let mut rim = f64::MAX;
            for k in 0..10 {
                let a = k as f64 * std::f64::consts::TAU / 10.0;
                let q = cctx.offset(rad * a.cos(), rad * a.sin());
                let gq = geodesy::ecef2geodetic(q, &self.ell);
                let qctx = Ctx::new(gq.lat, gq.lon, 20.0, &self.ell);
                let tq = self.terrain_impl(&qctx, &self.macro_at(qctx.p, qctx.gsd), Mode::NoLakes, None, None);
                rim = rim.min(tq.ground);
            }
            if rim < tc.ground - 4.0 {
                None
            } else {
                Some((rim - 0.7).max(tc.ground + 1.0))
            }
        };
        CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.len() > 100_000 {
                c.clear();
            }
            c.insert(key, v);
        });
        v
    }

    /// Pass A at one point (macro fields and drainage evaluated exactly).
    pub fn terrain(&self, ctx: &Ctx) -> Terrain {
        let m = self.macro_at(ctx.p, ctx.gsd);
        self.terrain_impl(ctx, &m, Mode::Full, None, None)
    }

    /// Pass A with given (e.g. interpolated) macro fields and the drainage segments near the
    /// point (see `river_segments`).
    pub fn terrain_with(&self, ctx: &Ctx, m: &Macro, segs: &[Seg]) -> Terrain {
        self.terrain_impl(ctx, m, Mode::Full, Some(segs), None)
    }

    /// [`Self::terrain_with`] with precomputed per-area drainage data (identical results).
    pub fn terrain_with_near(&self, ctx: &Ctx, m: &Macro, segs: &[Seg], near: &NearSegs) -> Terrain {
        self.terrain_impl(ctx, m, Mode::Full, Some(segs), Some(near))
    }

    /// Closed basins fed by the channels in `segs`: (id, centre, radius), sized by the total
    /// inflow (the sum of the inflowing channel widths), in the order of `segs`.
    pub fn sink_lakes(segs: &[Seg]) -> Vec<(u64, DVec3, f64)> {
        let mut lakes: Vec<(u64, DVec3, f64)> = Vec::new();
        for sg in segs.iter().filter(|sg| sg.sink) {
            let hh = hash3(0x51A7, (sg.b.x / 10.0) as i64, (sg.b.y / 10.0) as i64, (sg.b.z / 10.0) as i64);
            match lakes.iter_mut().find(|l| l.0 == hh) {
                Some(l) => l.2 += sg.hw,
                None => lakes.push((hh, sg.b, sg.hw)),
            }
        }
        for l in lakes.iter_mut() {
            l.2 = (6.0 * l.2).clamp(200.0, 2500.0) * (0.8 + 0.4 * u01k(l.0, 1));
        }
        lakes
    }

    /// Ground height before rivers, lakes and the sea (cheap; for bounds).
    pub fn relief_height(&self, ctx: &Ctx, m: &Macro) -> f64 {
        self.terrain_impl(ctx, m, Mode::Relief, None, None).ground
    }

    /// The pieces of `segs` that can reach any point within `radius` of `center` whose ground
    /// before carving is at most `h_max`: rejected only where the distance minus the radius and
    /// the largest meander warp exceeds the largest query reach (see `river_query`). Keeps the
    /// order of `segs`.
    pub fn local_segments(&self, segs: &[Seg], center: DVec3, radius: f64, h_max: f64) -> Vec<Seg> {
        segs.iter()
            .filter(|s| {
                let lc = &self.cfg.hydro.levels[s.level as usize];
                // |perlin3| ≤ 1.1·√3: warp ≤ 0.22·λ·2.76·√2 < 0.9·λ
                let warp = 0.9 * lc.meander.max(1e-3) * lc.cell_km * KM;
                let floor_min = s.ha.min(s.hb);
                let reach = (s.valley * 1.5 + s.hw + 200.0).max(11.3 * s.hw + 6.0 * (h_max - floor_min + 2.0 + 0.04 * s.hw) + 50.0);
                let ab = s.b - s.a;
                let u = ((center - s.a).dot(ab) / ab.length_squared().max(1e-9)).clamp(0.0, 1.0);
                (center - (s.a + ab * u)).length() - radius - warp <= reach + 1.0
            })
            .copied()
            .collect()
    }

    fn terrain_impl(&self, ctx: &Ctx, m: &Macro, mode: Mode, segs: Option<&[Seg]>, near_segs: Option<&NearSegs>) -> Terrain {
        let with_lakes = mode == Mode::Full;
        let p = ctx.p;
        let gsd = ctx.gsd;
        let r = &self.cfg.relief;
        let s = m.cont;
        let land = smoothstep(-0.06, 0.05, s);

        // ---- base elevation from the continent field
        let base = Self::base_elevation(s);

        // ---- high plateaus
        let plateau = smoothstep(0.15, 0.55, m.plateau) * 900.0 * smoothstep(0.02, 0.15, s);

        // ---- mountain belts
        let (mountain, amp_m) = self.mountain_mask(m);
        let (ridged, ridged_low) = if mountain > 1e-3 {
            let wp = DVec2::new(m.mtn_warp[0], m.mtn_warp[1]) * 9.0 * KM;
            let pw = p + ctx.east * wp.x + ctx.north * wp.y;
            let sharp = 1.6 + 0.8 * m.style[2];
            match m.pre.and_then(|p| p.relief.map(|r| (r, p.relief_cut[0]))) {
                // the long octaves from the tile's coarse grid
                Some((r, cut)) => {
                    let st = self.ridged_part(pw, gsd, sharp, cut, false, [r[0], r[1], r[2].clamp(0.0, 1.0)]);
                    (st[0] * 0.5, st[1] * 0.5)
                }
                None => self.ridged(pw, gsd, sharp),
            }
        } else {
            (0.0, 0.0)
        };
        let uplift = 0.22 * amp_m;
        let mtn = amp_m * ridged;

        // ---- hills
        let rough = m.rough;
        let hill_amp = self.hill_amplitude(m);
        let gain = 0.47 + 0.08 * rough;
        let (hl, hl_low) = match m.pre.and_then(|p| p.relief.map(|r| (r, p.relief_cut[1]))) {
            Some((r, cut)) => {
                let [hs, hlow] = self.hills_part(p, gsd, gain, cut, false);
                ((r[3] + hs) * 0.7, (r[4] + hlow) * 0.7)
            }
            None => self.hills(p, gsd, gain),
        };
        let hills = hill_amp * hl;

        // ---- erosion gullies on mountain and hill slopes
        let relief_amp = amp_m + 0.8 * hill_amp;
        let lam_e = self.cfg.relief.gully_wavelength_m;
        let mut gully = 0.0;
        let mut gully_n = 0.0;
        if self.cfg.relief.erosion > 0.0 && relief_amp > 40.0 && gsd < lam_e * 0.5 {
            // large-scale gradient by finite differences of the low-passed relief (from the
            // tile's coarse grid when available)
            let [ge, gn] = match m.pre.and_then(|p| p.gully) {
                Some(g) => g,
                None => self.low_relief_gradient(ctx, m, mountain, amp_m, hill_amp, gain, lam_e),
            };
            let grad = ctx.east * ge + ctx.north * gn;
            let slope_l = (ge * ge + gn * gn).sqrt();
            let mask = smoothstep(0.03, 0.25, slope_l) * smoothstep(40.0, 140.0, relief_amp);
            if mask > 0.0 {
                let g = match m.pre.and_then(|p| p.gully_oct.map(|st| (st, p.relief_cut[1]))) {
                    // the long octaves from the tile's coarse grid
                    Some((st, cut)) => self.gullies_part(p, ctx.up, grad, gsd, lam_e, cut, false, st)[0],
                    None => self.gullies(p, ctx.up, grad, gsd, lam_e),
                };
                gully_n = g * mask;
                gully = gully_n * self.cfg.relief.erosion * (0.05 * amp_m + 0.12 * hill_amp);
            }
        }

        // ---- micro relief
        let micro =
            if r.micro_height_m > 0.0 { r.micro_height_m * (0.4 + 0.6 * smoothstep(-0.3, 0.6, rough) + mountain) * self.micro.eval(p, gsd) } else { 0.0 };

        let mut h = base + plateau + uplift + mtn + hills + micro + gully;
        let smooth = base + plateau + uplift + amp_m * ridged_low * 0.6 + hill_amp * hl_low;

        // ---- climate (from smooth elevation, so it does not alias)
        let (temp0, moist) = self.climate(m, ctx.lat, smooth.max(0.0));

        // ---- mesas (arid terraces)
        let arid = 1.0 - smoothstep(0.18, 0.42, moist);
        let mesa_noise = m.mesa;
        let mesa = arid * smoothstep(0.1, 0.45, mesa_noise) * (1.0 - mountain) * smoothstep(0.02, 0.1, s) * r.mesas;
        if mesa > 1e-3 {
            // terrace height: hashed per band of the mesa field, blended between neighbouring
            // bands (a hard switch made cliffs along the band borders)
            let xb = mesa_noise * 7.0;
            let kb = xb.floor();
            let sh = |k: f64| 35.0 + 90.0 * u01(hash1(self.seed, k as i64));
            let step = lerp(sh(kb), sh(kb + 1.0), smoothstep(0.0, 1.0, xb - kb));
            let x = h / step;
            let k = x.floor();
            let f = x - k;
            let ft = smoothstep(0.72, 0.97, f);
            let ht = (k + ft) * step;
            h = lerp(h, ht, mesa * band(step, gsd).max(0.35));
        }

        // ---- sand seas with dunes
        let sand_n = m.sand;
        let sand = (1.0 - smoothstep(0.1, 0.28, moist))
            * smoothstep(6.0, 14.0, temp0)
            * smoothstep(-0.25, 0.15, sand_n)
            * (1.0 - mountain)
            * smoothstep(0.01, 0.06, s);
        if sand > 1e-3 && r.dune_height_m > 0.0 {
            h += r.dune_height_m * sand * self.dunes(ctx, m, gsd);
        }

        // ---- the kits' relief operators (volcanoes, karst, dunes …)
        if !crate::kits::KITS.is_empty() {
            let rin = crate::kits::ReliefIn { ctx, m, temp: temp0, moist, mountain, sand, mesa, smooth, inst: near_segs.map(|n| n.inst) };
            crate::kits::relief(self, &rin, &mut h);
        }

        // ---- rivers: major + minor networks carve valleys, set water level
        let mut t = Terrain { river_d: f64::MAX, river_hw: 0.0, water: f64::NEG_INFINITY, ..Default::default() };
        let mut floodplain: f64 = 0.0;
        let mut sink_lakes: Vec<(u64, DVec3, f64)> = Vec::new();
        // (no hard land cutoff: switching rivers off at a contour of the smooth continent field
        // left straight cliffs along the coast; the carve fades out towards the open sea instead)
        let land_fade = smoothstep(0.0, 0.1, land);
        if mode != Mode::Relief && self.cfg.hydro.rivers && land > 0.0 {
            let local;
            let segs = match segs {
                Some(s) => s,
                None => {
                    local = self.river_segments(p, 0.0, gsd);
                    &local[..]
                }
            };
            // closed basins fed by rivers near p: (id, centre, radius)
            match near_segs {
                Some(n) => sink_lakes.extend(n.sinks.iter().filter(|l| (p - l.1).length() < 1.6 * l.2)),
                None => {
                    sink_lakes = Self::sink_lakes(segs);
                    sink_lakes.retain(|l| (p - l.1).length() < 1.6 * l.2);
                }
            }
            let qsegs = match near_segs {
                Some(n) if h <= n.h_max => n.local,
                _ => segs,
            };
            let hits = self.river_query(ctx, qsegs, h, m.pre.as_ref().map_or(&[][..], |p| &p.river_warp[..]));
            let h0 = h;
            let wn = smoothstep(-0.6, 0.6, m.river_width);
            // the floodplain-edge noise depends on the level only: once per level (from the
            // tile's coarse grid when there)
            let mut fp_noise: [Option<f64>; 4] = m.pre.map_or([None; 4], |p| p.floodplain);
            for rh in &hits {
                let lc = &self.cfg.hydro.levels[rh.level as usize];
                let ad = rh.d.abs();
                let hw = rh.hw;
                let width = 2.0 * hw;
                // irregular floodplain edge (a constant width drew the edge as a straight line
                // along straight reaches)
                // (a constant wavelength per level: one following the channel width sheared the
                // noise into streaks — absolute ECEF coordinates turn a tiny wavelength change into
                // a large phase shift)
                let li = rh.level as usize;
                let fpn = match fp_noise.get(li).copied().flatten() {
                    Some(v) => v,
                    None => {
                        let v = self.floodplain_noise(li, p);
                        if li < 4 {
                            fp_noise[li] = Some(v);
                        }
                        v
                    }
                };
                let fp_w = (hw + width * (1.0 + 3.0 * wn)) * (1.0 + 0.25 * fpn);
                let incision = 1.0 + 0.02 * width;
                let floor = rh.floor.max(1.0) - incision;
                // valley profile: bed, floodplain, walls kept below ~30° (V shape); every channel
                // carves from the uncarved height and the lowest result wins (continuous where
                // the nearest channel changes)
                // walls ~24° in mountains, ~9° in lowlands (steep walls drew long straight scarps
                // along the floodplains of big lowland rivers)
                let wall_k = lerp(6.0, 2.2, mountain);
                let valley = rh.valley.max(fp_w + wall_k * (h0 - floor));
                // no carving of the deeper seabed; floodplains stay above the sea (near the coast they were
                // carved below it and flooded: straight "coastlines" along the valley walls and
                // channels drawn into the sea) — only the channel itself forms an estuary
                // (the channel runs on across the shallow shelf, so it reaches open water instead
                // of stopping square at the 0 m contour of a flat coast)
                if h0 > floor && h0 > -4.0 && ad < valley {
                    let wall = smoothstep(fp_w, valley, ad);
                    let wall = wall * wall * (3.0 - 2.0 * wall);
                    let fp = (floor + 0.8).max(0.5) + 0.4 * micro.abs();
                    let target = if ad < hw { floor - 0.8 - 0.02 * width } else { fp };
                    let carved = h0.min(lerp(target, h0, wall));
                    // limit the carve depth (small streams only notch the terrain)
                    let carved = carved.max(h0 - lc.max_depth_m * (1.0 - 0.3 * wall));
                    // valleys narrower than a pixel fade out per pixel (no hard level cutoff)
                    let fade = if width < 0.3 * ctx.gsd { smoothstep(0.25, 0.5, rh.valley / ctx.gsd) } else { 1.0 };
                    // the cut fades out across the shallow shelf (a channel continuing into the sea
                    // drew a dark band that ended square)
                    let shelf = smoothstep(-4.0, 0.5, h0);
                    h = h.min(h0 + (carved - h0) * fade * land_fade * shelf);
                    floodplain = floodplain.max((1.0 - wall) * land_fade * smoothstep(20.0, 120.0, width) * fade);
                }
            }
            // channel attributes from the nearest channel
            if let Some(rh) = hits.iter().min_by(|a, b| (a.d.abs() - a.hw).total_cmp(&(b.d.abs() - b.hw))) {
                let lc = &self.cfg.hydro.levels[rh.level as usize];
                let ad = rh.d.abs();
                let floor = rh.floor.max(1.0) - (1.0 + 0.04 * rh.hw);
                t.river_d = rh.d;
                t.river_hw = rh.hw;
                // water surface: the drainage floor, but never buried below the (notched) ground
                // nor above its banks: a tributary's floor (from the uncarved relief) can lie far
                // above the floor of a bigger valley it crosses, and that perched water surface,
                // blended into wide coarse-zoom pixels, raised walls tens of metres high along the
                // rivers. Outside the channel the level is the ground.
                t.river_level = if ad < rh.hw { floor.min(h + 1.0 + 0.06 * rh.hw).max(h + 0.6) } else { h };
                t.river_wet = smoothstep(lc.wet_moisture, lc.wet_moisture + 0.12, moist);
            }
        }

        // ---- lakes (Worley cells; flat surface at the basin spill height)
        let lake_cell = self.cfg.hydro.lake_cell_km * KM;
        // (resolution cutoffs below are placed where the feature covers at most a few percent of a
        // pixel, so switching it off along a row of constant GSD is invisible)
        if with_lakes && self.cfg.hydro.lake_density > 0.0 && land > 0.3 && lake_cell > gsd {
            let wc = self.site_cell(m, SITE_LAKE, 0x1A4E, p, lake_cell, 0.85);
            for (id, pt) in [(wc.id, wc.point), (wc.id2, wc.point2)] {
                let prob = self.cfg.hydro.lake_density * (0.3 + 0.9 * moist) * (1.0 - 0.8 * mountain);
                if u01k(id, 1) > prob {
                    continue;
                }
                let rad = (300.0 * (u01k(id, 2) * 2.7).exp()).min(lake_cell * 0.3);
                // centre on the surface below the 3D feature point
                let pc = pt.normalize() * p.length();
                let d = (p - pc).length();
                if d > rad * 1.5 {
                    continue;
                }
                let Some(level) = self.lake_level(id, pt, rad) else { continue };
                let lw = rad * 0.6;
                let warpn = perlin3(id, p / lw) * 0.3 + perlin3(id ^ 7, p / (lw * 0.3)) * 0.12 * band(lw * 0.3, gsd);
                let de = d / rad * (1.0 + warpn);
                let depth = 3.0 + 0.01 * rad;
                if de < 1.0 {
                    let bowl = level - depth * (1.0 - de * de);
                    h = h.min(lerp(h, bowl, smoothstep(1.0, 0.7, de)));
                }
                if de < 1.0 && h < level {
                    t.water = t.water.max(level);
                    t.water_kind = water::LAKE;
                } else if de < 1.3 && h < level + 0.8 {
                    // low natural levee keeps the water inside
                    h = lerp(level + 0.8, h, smoothstep(1.0, 1.3, de));
                }
            }
        }
        // lakes at the end of rivers that drain into a closed basin
        if with_lakes {
            for &(id, c, rad) in &sink_lakes {
                let Some(level) = self.lake_level_forced(id, c, rad) else { continue };
                let pc = c.normalize() * p.length();
                // elongated along a random axis, with lobes, bays and a ragged shore at several
                // scales (a circle with a fine scalloped edge looked artificial)
                let up = pc.normalize();
                let ax = up.cross(DVec3::new(u01k(id, 2) - 0.5, u01k(id, 3) - 0.5, u01k(id, 4) - 0.5)).normalize_or_zero();
                let el = 1.0 + 1.2 * u01k(id, 5);
                let dv = p - pc;
                let (da, db) = (dv.dot(ax), (dv - ax * dv.dot(ax)).length());
                let d = (da * da / el + db * db * el).sqrt();
                let lw = rad * 0.6;
                let warpn = perlin3(id, p / (1.4 * lw)) * 0.45
                    + perlin3(id ^ 7, p / (lw * 0.45)) * 0.18 * band(lw * 0.45, gsd)
                    + perlin3(id ^ 9, p / (lw * 0.12)) * 0.06 * band(lw * 0.12, gsd);
                let de = d / rad * (1.0 + warpn);
                // the basin is filled up to the level: ground a little above it is flooded too
                // (partly flooded basins drew thin crescents of water), but higher ground is not
                // dug away (that carved pits into the hillsides of sloping basins); the shore
                // follows the terrain's contours
                let depth = 3.0 + 0.01 * rad;
                let near = 1.0 - smoothstep(1.5, 4.0, h - level);
                if de < 1.0 && near > 0.0 {
                    let bowl = level - depth * (1.0 - de * de) - 0.3;
                    h = h.min(lerp(h, bowl, near));
                    if h < level {
                        t.water = t.water.max(level);
                        t.water_kind = water::LAKE;
                    }
                } else if de < 1.5 && h < level + 4.0 {
                    let shore = level + 0.4 + 12.0 * (de - 1.0) * (de - 1.0);
                    h = h.min(lerp(shore, h, smoothstep(1.0, 1.5, de)));
                }
            }
        }

        // ---- ocean
        if h < 0.0 && t.water_kind == water::NONE {
            t.water = 0.0;
            t.water_kind = water::OCEAN;
        }

        // ---- climate at actual elevation (for snow etc.)
        let temp = temp0 - self.cfg.climate.lapse_rate_c_per_km * (h.max(0.0) - smooth.max(0.0)) / KM;

        // ---- land use suitability
        let an = m.agri;
        let climate_ok = smoothstep(2.0, 8.0, temp) * (1.0 - smoothstep(27.0, 31.0, temp));
        let wet_ok = smoothstep(0.22, 0.42, moist);
        let irrig = (1.0 - wet_ok) * smoothstep(0.15, 0.6, an) * smoothstep(14.0, 20.0, temp); // dry: pivots
        let agri = (climate_ok
            * (wet_ok + 0.7 * irrig)
            * (0.35 + 0.65 * smoothstep(-0.6, 0.2, an))
            * (1.0 - mountain * 0.9)
            * (1.0 - 0.75 * smoothstep(120.0, 320.0, hill_amp * (0.6 + 0.8 * smoothstep(-0.3, 0.6, rough))))
            * self.cfg.landuse.agriculture)
            .clamp(0.0, 1.0);
        let habit = climate_ok * (0.4 + 0.6 * wet_ok) * (1.0 - mountain) * land;

        let style = [0.5 + 0.5 * m.style[0] * 1.4, 0.5 + 0.5 * m.style[1] * 1.4, 0.5 + 0.5 * m.style[2] * 1.4, 0.5 + 0.5 * m.style[3] * 1.4].map(saturate);

        // ---- land-use sites (only relevant when such features can be resolved)
        let region_cell = self.cfg.landuse.region_km * KM;
        if mode != Mode::Relief && gsd < region_cell * 0.5 {
            // warped lookup → curvy (not straight) borders between field systems
            let wq = match m.pre.and_then(|p| p.region_warp) {
                Some(w) => DVec3::from_array(w),
                None => self.region_warp(p),
            };
            let pw = p + wq;
            let wc = self.site_cell(m, SITE_REGION, 0x5E61, pw, region_cell, 0.9);
            t.region = Site { id: wc.id, id2: wc.id2, center: wc.point, center2: wc.point2, dist: wc.f1, edge: worley_edge_dist(&wc, pw) };
        }
        // the ecoregion (every zoom: the biomes' look)
        if mode != Mode::Relief {
            let wq = match m.pre.and_then(|p| p.eco_warp) {
                Some(w) => DVec3::from_array(w),
                None => crate::eco::warp(self, p),
            };
            let pw = p + wq;
            let wc = self.site_cell(m, SITE_ECO, crate::eco::ECO_KEY, pw, self.cfg.ecoregions.cell_km * KM, 0.9);
            t.eco = Site { id: wc.id, id2: wc.id2, center: wc.point, center2: wc.point2, dist: wc.f1, edge: worley_edge_dist(&wc, pw) };
        }
        let town_cell = self.cfg.landuse.town_cell_km * KM;
        if mode != Mode::Relief && gsd < town_cell * 0.25 && self.cfg.landuse.towns > 0.0 {
            let wc = self.site_cell(m, SITE_TOWN, 0x70E1, p, town_cell, 0.8);
            t.town = Site { id: wc.id, id2: wc.id2, center: wc.point, center2: wc.point2, dist: wc.f1, edge: worley_edge_dist(&wc, p) };
        }

        // ---- road networks (iso-lines of warped noise), only where people live
        t.road_major = f64::MAX;
        t.road_minor = f64::MAX;
        // (roads are drawn by pixel coverage: 12 m at 400 m/px or 6 m at 200 m/px is ~3%)
        if mode != Mode::Relief && self.cfg.landuse.roads > 0.0 && habit > 0.02 && gsd < 400.0 {
            let dist = |v: [f64; 3]| v[0] / DVec2::new(v[1], v[2]).length().max(1e-12);
            match m.pre.and_then(|p| p.road_major.zip(p.road_minor)) {
                Some((major, minor)) => {
                    t.road_major = dist(major);
                    if gsd < 200.0 {
                        t.road_minor = dist(minor);
                    }
                }
                None => {
                    let warp = self.network_warp(ctx);
                    t.road_major = self.network_dist(&self.road_major, ctx, &warp, 2500.0);
                    if gsd < 200.0 {
                        t.road_minor = self.network_dist(&self.road_minor, ctx, &warp, 700.0);
                    }
                }
            }
        }

        let rock_expect = (mountain * smoothstep(0.15, 0.6, ridged) + 0.25 * mesa).clamp(0.0, 1.0);

        t.ground = h;
        t.gully = gully_n;
        t.temp = temp;
        t.moist = moist;
        t.mountain = mountain;
        t.rock_expect = rock_expect;
        t.sand = sand;
        t.floodplain = floodplain;
        t.mesa = mesa;
        t.cont = s;
        t.agri = agri;
        t.habit = habit;
        t.style = style;
        t
    }
}
