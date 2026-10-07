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
        Ctx {
            p,
            up: DVec3::new(cl * co, cl * so, sl),
            east: DVec3::new(-so, co, 0.0),
            north: DVec3::new(-sl * co, -sl * so, cl),
            lat,
            lon,
            gsd,
        }
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
}

/// A Worley site: id hash, centre (ECEF, on the surface), distance to its Voronoi border (m).
#[derive(Clone, Copy, Debug, Default)]
pub struct Site {
    pub id: u64,
    /// id of the neighbouring site across the nearest border
    pub id2: u64,
    pub center: DVec3,
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
}

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
        }
    }
}

pub struct World {
    pub cfg: Config,
    pub ell: Ellipsoid,
    pub(crate) seed: u64,
    cont: Fbm,
    cont_warp: [Fbm; 3],
    belt: Fbm,
    belt2: Fbm,
    belt_var: Fbm,
    mtn_frames: OctaveFrames,
    mtn_warp: [Fbm; 2],
    plateau: Fbm,
    hills: OctaveFrames,
    hill_amp: Fbm,
    rough: Fbm,
    micro: Fbm,
    temp_n: Fbm,
    moist_n: Fbm,
    river_warp: [Fbm; 2],
    river_width_n: Fbm,
    mesa_n: Fbm,
    dune_frames: OctaveFrames,
    sand_n: Fbm,
    agri_n: Fbm,
    style_n: [Fbm; 4],
    road_major: Fbm,
    road_minor: Fbm,
    home: Option<(DVec3, f64, f64)>,
    /// Hash of the whole config: key of the thread-local caches, so generators with the same
    /// seed but different settings in one process do not share cached hydrology / lakes.
    cache_key: u64,
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
        let cache_key = serde_yaml::to_string(&cfg).unwrap_or_default().bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3));
        World {
            cache_key,
            seed: s,
            cont: Fbm::new(k(1), cw, 7, 2.0, 0.52),
            cont_warp: [
                Fbm::new(k(2), cw * 0.8, 3, 2.0, 0.5),
                Fbm::new(k(3), cw * 0.8, 3, 2.0, 0.5),
                Fbm::new(k(4), cw * 0.8, 3, 2.0, 0.5),
            ],
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
        let warp = DVec3::new(
            self.cont_warp[0].eval(p, gsd),
            self.cont_warp[1].eval(p, gsd),
            self.cont_warp[2].eval(p, gsd),
        ) * (self.cfg.continents.warp * cw);
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
        let mut lam = 14.0 * KM;
        let mut amp = 1.0;
        let mut weight = 1.0;
        let mut sum = 0.0;
        let mut low = 0.0;
        let mut norm = 0.0;
        for i in 0..self.mtn_frames.rot.len() {
            // ridges contain harmonics above the octave frequency: band-limit more strictly
            let wb = band(lam, 1.6 * gsd);
            if wb <= 0.0 {
                break;
            }
            let q = self.mtn_frames.rot[i] * (p / lam) + self.mtn_frames.off[i];
            let n = perlin3(self.mtn_frames.seeds[i], q);
            let mut r = (1.0 - n.abs()).max(0.0);
            r = r.powf(sharp);
            r *= weight;
            weight = (r * 1.8).clamp(0.0, 1.0);
            sum += r * amp * wb;
            if lam > 6.0 * KM {
                low += r * amp * wb;
            }
            norm += amp;
            lam *= 0.5;
            amp *= if lam > 1.0 * KM { 0.58 } else { 0.42 };
        }
        let _ = norm;
        (sum * 0.5, low * 0.5)
    }

    /// Band-limited hill fBm with regionally varying roughness. Returns (full, lowpass>=15km).
    fn hills(&self, p: DVec3, gsd: f64, gain: f64) -> (f64, f64) {
        let mut lam = 9.0 * KM;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut low = 0.0;
        for i in 0..self.hills.rot.len() {
            let wb = band(lam, gsd);
            if wb <= 0.0 || lam < 20.0 {
                break;
            }
            let q = self.hills.rot[i] * (p / lam) + self.hills.off[i];
            let n = perlin3(self.hills.seeds[i], q);
            sum += n * amp * wb;
            if lam >= 4.0 * KM {
                low += n * amp * wb;
            }
            lam *= 0.5;
            amp *= gain;
        }
        (sum * 0.7, low * 0.7)
    }

    /// One octave of gradient-aligned gully noise in a 3D jittered lattice (point `q` in lattice
    /// units). Returns the stripe value and its derivative (lattice units).
    fn gully_octave(seed: u64, q: DVec3, dir: DVec3) -> (f64, DVec3) {
        let qf = q.floor();
        let (ix, iy, iz) = (qf.x as i64, qf.y as i64, qf.z as i64);
        let f = q - qf;
        let mut v = 0.0;
        let mut d = DVec3::ZERO;
        let mut wt = 0.0;
        for dz in -1..=1i64 {
            for dy in -1..=1i64 {
                for dx in -1..=1i64 {
                    let h = hash3(seed, ix + dx, iy + dy, iz + dz);
                    let jit = DVec3::new(u01k(h, 1), u01k(h, 2), u01k(h, 3)) * 0.5;
                    let pp = f - DVec3::new(dx as f64, dy as f64, dz as f64) - jit;
                    let w = (-2.0 * pp.length_squared()).exp();
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
        let dir0 = up.cross(grad).normalize_or_zero();
        let mut a = 1.0;
        let mut lam = lam0;
        let mut h = 0.0;
        let mut hd = DVec3::ZERO;
        for i in 0..5u64 {
            let wb = band(lam, gsd);
            if wb <= 0.0 {
                break;
            }
            let freq = lam0 / lam;
            // bending rotates the stripe direction; keep its length (= stripe frequency) at 1
            let dir = (dir0 + up.cross(hd) * 0.7).normalize_or_zero();
            let (v, d) = Self::gully_octave(self.seed ^ (0xE205 + i * 0x9E37), p / lam, dir);
            h += v * a * wb;
            hd += d * (a * wb);
            let _ = freq;
            a *= 0.45;
            lam *= 0.5;
        }
        h
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
        }
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
        let mut t = c.equator_temp - c.pole_drop * la.powf(1.6) + m.temp;
        t -= c.lapse_rate * elev.max(0.0) / KM;
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
        let amp_m = self.cfg.relief.mountain_height * (0.55 + 0.45 * smoothstep(-0.4, 0.6, m.belt_var)) * mountain;
        (mountain, amp_m)
    }

    fn hill_amplitude(&self, m: &Macro) -> f64 {
        let land = smoothstep(-0.06, 0.05, m.cont);
        self.cfg.relief.hill_height * (0.15 + 0.85 * smoothstep(-0.5, 0.6, m.hill_amp)) * (0.25 + 0.75 * land)
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
    fn lake_level(&self, id: u64, center: DVec3, rad: f64) -> Option<f64> {
        thread_local! {
            static CACHE: std::cell::RefCell<std::collections::HashMap<u64, Option<f64>>> = Default::default();
        }
        let key = id ^ self.cache_key.rotate_left(17);
        if let Some(v) = CACHE.with(|c| c.borrow().get(&key).copied()) {
            return v;
        }
        let g = geodesy::ecef2geodetic(center, &self.ell);
        let cctx = Ctx::new(g.lat, g.lon, 20.0, &self.ell);
        let tc = self.terrain_impl(&cctx, &self.macro_at(cctx.p, cctx.gsd), Mode::NoLakes, None);
        let v = if tc.water_kind != water::NONE || tc.ground < 1.0 {
            None
        } else {
            let mut rim = f64::MAX;
            for k in 0..10 {
                let a = k as f64 * std::f64::consts::TAU / 10.0;
                let q = cctx.offset(rad * a.cos(), rad * a.sin());
                let gq = geodesy::ecef2geodetic(q, &self.ell);
                let qctx = Ctx::new(gq.lat, gq.lon, 20.0, &self.ell);
                let tq = self.terrain_impl(&qctx, &self.macro_at(qctx.p, qctx.gsd), Mode::NoLakes, None);
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
        self.terrain_impl(ctx, &m, Mode::Full, None)
    }

    /// Pass A with given (e.g. interpolated) macro fields and the drainage segments near the
    /// point (see `river_segments`).
    pub fn terrain_with(&self, ctx: &Ctx, m: &Macro, segs: &[Seg]) -> Terrain {
        self.terrain_impl(ctx, m, Mode::Full, Some(segs))
    }

    fn terrain_impl(&self, ctx: &Ctx, m: &Macro, mode: Mode, segs: Option<&[Seg]>) -> Terrain {
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
            self.ridged(pw, gsd, sharp)
        } else {
            (0.0, 0.0)
        };
        let uplift = 0.22 * amp_m;
        let mtn = amp_m * ridged;

        // ---- hills
        let rough = m.rough;
        let hill_amp = self.hill_amplitude(m);
        let gain = 0.47 + 0.08 * rough;
        let (hl, hl_low) = self.hills(p, gsd, gain);
        let hills = hill_amp * hl;

        // ---- erosion gullies on mountain and hill slopes
        let relief_amp = amp_m + 0.8 * hill_amp;
        let lam_e = self.cfg.relief.gully_wavelength;
        let mut gully = 0.0;
        let mut gully_n = 0.0;
        if self.cfg.relief.erosion > 0.0 && relief_amp > 40.0 && gsd < lam_e * 0.5 {
            // large-scale gradient by finite differences of the low-passed relief
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
            let ge = (low(p + ctx.east * e) - h0) / e;
            let gn = (low(p + ctx.north * e) - h0) / e;
            let grad = ctx.east * ge + ctx.north * gn;
            let slope_l = (ge * ge + gn * gn).sqrt();
            let mask = smoothstep(0.03, 0.25, slope_l) * smoothstep(40.0, 140.0, relief_amp);
            if mask > 0.0 {
                gully_n = self.gullies(p, ctx.up, grad, gsd, lam_e) * mask;
                gully = gully_n * self.cfg.relief.erosion * (0.05 * amp_m + 0.12 * hill_amp);
            }
        }

        // ---- micro relief
        let micro = if r.micro_height > 0.0 {
            r.micro_height * (0.4 + 0.6 * smoothstep(-0.3, 0.6, rough) + mountain) * self.micro.eval(p, gsd)
        } else {
            0.0
        };

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
        if sand > 1e-3 && r.dune_height > 0.0 {
            h += r.dune_height * sand * self.dunes(ctx, m, gsd);
        }

        // ---- rivers: major + minor networks carve valleys, set water level
        let mut t = Terrain {
            river_d: f64::MAX,
            river_hw: 0.0,
            water: f64::NEG_INFINITY,
            ..Default::default()
        };
        let mut floodplain: f64 = 0.0;
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
            let hits = self.river_query(ctx, segs, h);
            let h0 = h;
            let wn = smoothstep(-0.6, 0.6, m.river_width);
            for rh in &hits {
                let lc = &self.cfg.hydro.levels[rh.level as usize];
                let ad = rh.d.abs();
                let hw = rh.hw;
                let width = 2.0 * hw;
                // irregular floodplain edge (a constant width drew the edge as a straight line
                // along straight reaches)
                let fp_w = (hw + width * (1.0 + 3.0 * wn)) * (1.0 + 0.25 * perlin3(0xF10D ^ rh.level as u64, p / (1.5 * width + 400.0)));
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
                t.river_level = if ad < rh.hw { floor.max(h + 0.6) } else { floor.max(h) };
                t.river_wet = smoothstep(lc.wet_moisture, lc.wet_moisture + 0.12, moist);
            }
        }

        // ---- lakes (Worley cells; flat surface at the basin spill height)
        let lake_cell = self.cfg.hydro.lake_cell_km * KM;
        // (resolution cutoffs below are placed where the feature covers at most a few percent of a
        // pixel, so switching it off along a row of constant GSD is invisible)
        if with_lakes && self.cfg.hydro.lake_density > 0.0 && land > 0.3 && lake_cell > gsd {
            let wc = worley3(self.seed ^ 0x1A4E, p, lake_cell, 0.85);
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

        // ---- ocean
        if h < 0.0 && t.water_kind == water::NONE {
            t.water = 0.0;
            t.water_kind = water::OCEAN;
        }

        // ---- climate at actual elevation (for snow etc.)
        let temp = temp0 - self.cfg.climate.lapse_rate * (h.max(0.0) - smooth.max(0.0)) / KM;

        // ---- land use suitability
        let an = m.agri;
        let climate_ok = smoothstep(2.0, 8.0, temp) * (1.0 - smoothstep(27.0, 31.0, temp));
        let wet_ok = smoothstep(0.22, 0.42, moist);
        let irrig = (1.0 - wet_ok) * smoothstep(0.15, 0.6, an) * smoothstep(14.0, 20.0, temp); // dry: pivots
        let agri = (climate_ok * (wet_ok + 0.7 * irrig) * (0.35 + 0.65 * smoothstep(-0.6, 0.2, an))
            * (1.0 - mountain * 0.9)
            * (1.0 - 0.75 * smoothstep(120.0, 320.0, hill_amp * (0.6 + 0.8 * smoothstep(-0.3, 0.6, rough))))
            * self.cfg.landuse.agriculture)
            .clamp(0.0, 1.0);
        let habit = climate_ok * (0.4 + 0.6 * wet_ok) * (1.0 - mountain) * land;

        let style = [
            0.5 + 0.5 * m.style[0] * 1.4,
            0.5 + 0.5 * m.style[1] * 1.4,
            0.5 + 0.5 * m.style[2] * 1.4,
            0.5 + 0.5 * m.style[3] * 1.4,
        ]
        .map(saturate);

        // ---- land-use sites (only relevant when such features can be resolved)
        let region_cell = self.cfg.landuse.region_km * KM;
        if mode != Mode::Relief && gsd < region_cell * 0.5 {
            // warped lookup → curvy (not straight) borders between field systems
            let wq = DVec3::new(
                perlin3(self.seed ^ 0xA1, p / (0.9 * region_cell)),
                perlin3(self.seed ^ 0xA2, p / (0.9 * region_cell)),
                perlin3(self.seed ^ 0xA3, p / (0.9 * region_cell)),
            ) * (0.18 * region_cell)
                + DVec3::new(perlin3(self.seed ^ 0xA4, p / 1500.0), perlin3(self.seed ^ 0xA5, p / 1500.0), perlin3(self.seed ^ 0xA6, p / 1500.0)) * 120.0;
            let pw = p + wq;
            let wc = worley3(self.seed ^ 0x5E61, pw, region_cell, 0.9);
            t.region = Site { id: wc.id, id2: wc.id2, center: wc.point, dist: wc.f1, edge: worley_edge_dist(&wc, pw) };
        }
        let town_cell = self.cfg.landuse.town_cell_km * KM;
        if mode != Mode::Relief && gsd < town_cell * 0.25 && self.cfg.landuse.towns > 0.0 {
            let wc = worley3(self.seed ^ 0x70E1, p, town_cell, 0.8);
            t.town = Site { id: wc.id, id2: wc.id2, center: wc.point, dist: wc.f1, edge: worley_edge_dist(&wc, p) };
        }

        // ---- road networks (iso-lines of warped noise), only where people live
        t.road_major = f64::MAX;
        t.road_minor = f64::MAX;
        // (roads are drawn by pixel coverage: 12 m at 400 m/px or 6 m at 200 m/px is ~3%)
        if mode != Mode::Relief && self.cfg.landuse.roads > 0.0 && habit > 0.02 && gsd < 400.0 {
            let warp = self.network_warp(ctx);
            t.road_major = self.network_dist(&self.road_major, ctx, &warp, 2500.0);
            if gsd < 200.0 {
                t.road_minor = self.network_dist(&self.road_minor, ctx, &warp, 700.0);
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
