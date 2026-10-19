//! Settlement placement: the central-place hierarchy (`docs/design/terrain-next.md` §6.1).
//!
//! Five levels, each a 3D jittered lattice (seamless on the sphere): metropolis (300 km), city
//! (90 km), town (25 km), village (6 km), hamlet (1.6 km). Every lattice cell has K hashed
//! candidate positions; their score is the atlas' population potential × climate × coast access
//! (all candidates), then × flatness (a ring of smooth-elevation samples; Mediterranean and
//! highland villages prefer hilltops instead) × water for the best three (villages, hamlets): a
//! candidate whose centre lies in open water (pass A at the point: the sea, a lake, a level 0–1
//! river channel) is dropped, water on a ring around it lowers its score, a river within 2.5 km
//! raises it; the best town or larger gives way to the next where a lake can reach its centre
//! (`lake_free`). Scores are quantized to 1/64 with the candidate hash as the
//! tie-breaker. The best candidate exists with a probability by
//! level, population potential, culture (archetype) and `landuse.towns`.
//!
//! Suppression is deterministic and bounded, top-down: a site is removed when a site of a
//! higher level (already final) or of the same level with a higher priority (level, score, id;
//! base existence only) is too close. A town or city inside a larger settlement's footprint
//! becomes a **sub-centre** of it (`Place::parent`): the conurbation is polycentric.
//!
//! Everything here is host-side Rust over analytic world fields (atlas, smooth relief): the same
//! for the CPU and GPU tile paths, a pure function of the world (memoized per lattice cell), and
//! cheap enough for the transport kit's graph ([`settlements_near`]).

use crate::noise::*;
use crate::world::World;
use glam::DVec3;

/// A settlement (the transport kit's view).
#[derive(Clone, Debug, PartialEq)]
pub struct Settlement {
    pub id: u64,
    /// centre on the ellipsoid surface (ECEF)
    pub ecef: DVec3,
    pub lat_deg: f64,
    pub lon_deg: f64,
    /// 0 hamlet, 1 village, 2 town, 3 city, 4 metropolis
    pub level: u8,
    pub population: f64,
    /// radius of the built-up area (m)
    pub radius_m: f64,
    /// the atlas' culture area and its archetype (`atlas::Archetype`), development 0..1
    pub culture: u32,
    pub archetype: u8,
    pub development: f64,
    /// on the coast (atlas)
    pub coastal: bool,
    /// may have an airport / a port (the kit places them where the terrain allows)
    pub airport: bool,
    pub port: bool,
    /// the larger settlement this one is a sub-centre of (0: none)
    pub parent: u64,
    /// smooth elevation at the centre (m)
    pub elevation_m: f64,
}

pub const HAMLET: u8 = 0;
pub const VILLAGE: u8 = 1;
pub const TOWN: u8 = 2;
pub const CITY: u8 = 3;
pub const METRO: u8 = 4;
pub const NLEVELS: usize = 5;

/// Village / town forms (`Place::form`; the stamp's form bits).
pub mod form {
    /// districts, blocks and lots (towns and larger, survey-grid villages)
    pub const CITY: u8 = 0;
    pub const NUCLEATED: u8 = 1;
    /// a street village (Strassendorf)
    pub const LINEAR: u8 = 2;
    /// a forest-strip village: a street village whose farms each own a strip up the valley side
    pub const WALDHUFEN: u8 = 3;
    pub const DISPERSED: u8 = 4;
    /// a ring of huts around a cattle enclosure (savanna pastoral)
    pub const KRAAL: u8 = 5;
    /// a compact hilltop village (Mediterranean, highland)
    pub const PERCHED: u8 = 6;
}

/// The parameters of a level.
#[derive(Clone, Copy, Debug)]
pub struct LevelSpec {
    /// lattice cell (m)
    pub cell: f64,
    /// candidates per cell
    pub k: u32,
    /// built-up radius range (m) and population range
    pub radius: [f64; 2],
    pub pop: [f64; 2],
    /// existence probability at full population potential
    pub base: f64,
    /// population potential where existence starts / saturates
    pub pop_window: [f64; 2],
    /// radius of the flatness ring (m)
    pub ring: f64,
    /// minimum distance between two sites of this level, in cells
    pub spacing: f64,
}

#[rustfmt::skip]
pub const LEVELS: [LevelSpec; NLEVELS] = [
    // hamlet
    LevelSpec { cell: 1_600.0, k: 1, radius: [45.0, 150.0], pop: [20.0, 200.0], base: 0.42, pop_window: [-0.05, 0.3], ring: 250.0, spacing: 0.3 },
    // village
    LevelSpec { cell: 6_000.0, k: 3, radius: [180.0, 800.0], pop: [200.0, 5_000.0], base: 0.62, pop_window: [-0.02, 0.35], ring: 600.0, spacing: 0.4 },
    // town
    LevelSpec { cell: 25_000.0, k: 6, radius: [1_000.0, 3_600.0], pop: [5_000.0, 100_000.0], base: 0.72, pop_window: [0.02, 0.45], ring: 2_000.0, spacing: 0.45 },
    // city
    LevelSpec { cell: 90_000.0, k: 8, radius: [3_500.0, 13_000.0], pop: [100_000.0, 2_000_000.0], base: 0.72, pop_window: [0.06, 0.55], ring: 5_000.0, spacing: 0.45 },
    // metropolis
    LevelSpec { cell: 300_000.0, k: 8, radius: [13_000.0, 32_000.0], pop: [2_000_000.0, 25_000_000.0], base: 0.75, pop_window: [0.15, 0.75], ring: 10_000.0, spacing: 0.5 },
];

/// Town density by archetype (a settlement's existence multiplier), by level group: hamlets
/// (dispersed cultures have many), villages, towns and larger.
pub(crate) const ARCH_HAMLETS: [f64; 12] = [1.5, 0.8, 1.3, 0.3, 0.7, 0.5, 0.8, 1.2, 0.6, 1.0, 0.3, 1.2];
pub(crate) const ARCH_VILLAGES: [f64; 12] = [1.0, 0.9, 0.7, 0.8, 0.5, 1.25, 1.15, 0.6, 1.4, 0.55, 0.25, 0.9];
pub(crate) const ARCH_TOWNS: [f64; 12] = [1.0, 0.85, 0.7, 0.8, 0.6, 1.2, 1.05, 0.9, 1.25, 0.6, 0.25, 0.85];
/// Built-up radius multiplier by archetype (dense cores vs sprawl).
pub(crate) const ARCH_SPRAWL: [f64; 12] = [0.95, 0.9, 0.9, 0.8, 0.95, 0.82, 0.92, 1.3, 0.85, 1.05, 0.85, 0.85];

/// A settlement with what the kit draws from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub id: u64,
    pub level: u8,
    pub center: DVec3,
    /// radians
    pub lat: f64,
    pub lon: f64,
    pub population: f64,
    pub radius: f64,
    pub culture: u32,
    pub arch: u8,
    pub dev: f64,
    pub coastal: bool,
    pub airport: bool,
    pub port: bool,
    /// the larger settlement this one is a sub-centre of (0: none)
    pub parent: u64,
    pub form: u8,
    /// main axis (radians from east, counter-clockwise), elongation (≥ 1, along the axis)
    pub rot: f64,
    pub elong: f64,
    /// quantized score (1/64) of the site
    pub score: u32,
    /// annual wind (east, north m/s) at the site: industry lies downwind
    pub wind: [f64; 2],
    /// sea-level temperature, annual precipitation (atlas) at the site
    pub temp: f64,
    pub precip: f64,
    /// smooth elevation (m) at the site
    pub elev: f64,
    /// the atlas' lithology at the site (`atlas::Lithology`)
    pub litho: u8,
}

impl Place {
    pub fn settlement(&self) -> Settlement {
        Settlement {
            id: self.id,
            ecef: self.center,
            lat_deg: self.lat.to_degrees(),
            lon_deg: self.lon.to_degrees(),
            level: self.level,
            population: self.population,
            radius_m: self.radius,
            culture: self.culture,
            archetype: self.arch,
            development: self.dev,
            coastal: self.coastal,
            airport: self.airport,
            port: self.port,
            parent: self.parent,
            elevation_m: self.elev,
        }
    }

    /// Priority order: higher level, then score, then lower id first.
    fn outranks(&self, o: &Place) -> bool {
        (self.level, self.score, std::cmp::Reverse(self.id)) > (o.level, o.score, std::cmp::Reverse(o.id))
    }

    /// The largest distance from the centre the place draws at (m; its stamp's reach).
    pub fn extent(&self) -> f64 {
        let r = 1.62 * self.radius * self.elong.sqrt();
        if self.form == form::WALDHUFEN {
            r.max(self.radius * 0.6 + WALDHUFEN_STRIP)
        } else {
            r
        }
    }
}

/// The longest strip of a forest-strip village (m).
pub const WALDHUFEN_STRIP: f64 = 1_400.0;

/// All settlements whose centre lies within `radius_m` of `center_ecef`, sorted by id;
/// deterministic, analytic (atlas and smooth relief only) and the same for both backends.
pub fn settlements_near(world: &World, center_ecef: DVec3, radius_m: f64) -> Vec<Settlement> {
    settlements_near_min_level(world, center_ecef, radius_m, HAMLET)
}

/// [`settlements_near`] for the levels ≥ `min_level` only (cheaper over large radii: villages
/// and hamlets are dense).
pub fn settlements_near_min_level(world: &World, center: DVec3, radius_m: f64, min_level: u8) -> Vec<Settlement> {
    let mut v: Vec<Settlement> = Vec::new();
    for level in min_level as usize..NLEVELS {
        v.extend(places_near(world, level, center, radius_m).iter().map(Place::settlement));
    }
    v.sort_unstable_by_key(|s| s.id);
    v
}

/// The places of `level` whose centre lies within `radius` of `center`, by id.
pub fn places_near(w: &World, level: usize, center: DVec3, radius: f64) -> Vec<Place> {
    let cells = cells_near(w, level, center, radius);
    let within = |c: &Cell| final_place(w, level, *c).filter(|p| (p.center - center).length() <= radius);
    // (wide discs, e.g. a transport network's nodes at a low zoom: their cells in parallel)
    let mut out: Vec<Place> = if cells.len() > 64 {
        use rayon::prelude::*;
        cells.par_iter().filter_map(within).collect()
    } else {
        cells.iter().filter_map(within).collect()
    };
    out.sort_unstable_by_key(|p| p.id);
    out
}

/// The lattice cells of `level` whose candidates (projected to the surface) can lie within
/// `radius` of `center`.
fn for_cells(w: &World, level: usize, center: DVec3, radius: f64, mut f: impl FnMut((i64, i64, i64))) {
    let cell = LEVELS[level].cell;
    // a candidate lies in its cell; its surface point within half a cell of it radially
    let rr = radius + 0.5 * cell;
    let lo = ((center - DVec3::splat(rr)) / cell).floor();
    let hi = ((center + DVec3::splat(rr)) / cell).floor();
    let (r_lo, r_hi) = (w.ell.b - 0.6 * cell, w.ell.a + 0.6 * cell);
    for z in lo.z as i64..=hi.z as i64 {
        for y in lo.y as i64..=hi.y as i64 {
            for x in lo.x as i64..=hi.x as i64 {
                // skip cells far from the surface or from the disc
                let cc = (DVec3::new(x as f64, y as f64, z as f64) + 0.5) * cell;
                let d0 = cc.length();
                if d0 < r_lo - 0.87 * cell || d0 > r_hi + 0.87 * cell || (cc - center).length() > rr + 0.87 * cell {
                    continue;
                }
                f((x, y, z));
            }
        }
    }
}

/// A lattice cell of a level.
pub type Cell = (i64, i64, i64);

/// The lattice cells of `level` whose places can lie within `radius` of `center`.
pub fn cells_near(w: &World, level: usize, center: DVec3, radius: f64) -> Vec<Cell> {
    let mut v = Vec::new();
    for_cells(w, level, center, radius, |c| v.push(c));
    v
}

/// The cells whose base places the resolution of cell `c` may read (its own level's
/// neighbours, `resolve`), `c` included.
pub fn resolve_cells(w: &World, level: usize, c: Cell) -> Vec<Cell> {
    let spec = &LEVELS[level];
    let reach = (spec.spacing * spec.cell).max(2.4 * spec.radius[1]);
    // (around any candidate of the cell: its centre ± the cell's half diagonal)
    let cc = (DVec3::new(c.0 as f64, c.1 as f64, c.2 as f64) + 0.5) * spec.cell;
    cells_near(w, level, cc, reach + 0.87 * spec.cell)
}

/// Compute (memoize) the base place of a cell.
pub fn warm_base(w: &World, level: usize, c: Cell) {
    base(w, level, c);
}

fn level_seed(w: &World, level: usize) -> u64 {
    mix64(w.seed() ^ 0x5E77_1E00 ^ (level as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// A candidate: its hash and surface point (None: its lattice point is too far from the
/// surface; each surface point sees one lattice column).
fn candidate(w: &World, level: usize, c: (i64, i64, i64), k: u32) -> Option<(u64, DVec3)> {
    let cell = LEVELS[level].cell;
    let h0 = hash3(level_seed(w, level), c.0, c.1, c.2);
    let h = mix64(h0 ^ (k as u64 + 1).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let pt = DVec3::new(c.0 as f64 + 0.5 + 0.9 * (u01k(h, 1) - 0.5), c.1 as f64 + 0.5 + 0.9 * (u01k(h, 2) - 0.5), c.2 as f64 + 0.5 + 0.9 * (u01k(h, 3) - 0.5))
        * cell;
    let (a, b) = (w.ell.a, w.ell.b);
    let len = pt.length().max(1.0);
    let d = pt / len;
    let re = a * b / (b * b * (d.x * d.x + d.y * d.y) + a * a * d.z * d.z).sqrt();
    if (len - re).abs() >= 0.5 * cell {
        return None;
    }
    Some((h, d * re))
}

/// Geodetic latitude, longitude (radians) and the east / north axes of a surface point.
pub fn frame_at(w: &World, p: DVec3) -> (f64, f64, DVec3, DVec3) {
    let (a, b) = (w.ell.a, w.ell.b);
    let e2 = 1.0 - (b * b) / (a * a);
    let wxy = (p.x * p.x + p.y * p.y).sqrt();
    let lat = p.z.atan2((1.0 - e2) * wxy);
    let lon = p.y.atan2(p.x);
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    (lat, lon, DVec3::new(-so, co, 0.0), DVec3::new(-sl * co, -sl * so, cl))
}

/// The village form of a place of `level` in archetype `arch` (hashed by its id), on a ring
/// slope `slope`: towns and larger are cities (districts), so are survey-grid villages.
fn pick_form(id: u64, level: u8, arch: usize, slope: f64) -> u8 {
    use form::*;
    if level >= TOWN {
        return CITY;
    }
    let u = u01k(id, 30);
    // weights by archetype: nucleated, linear, waldhufen, dispersed, kraal, perched, grid
    let w: [f64; 7] = match arch {
        0 => [0.3, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0],
        1 => [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        2 => [0.0, 0.0, 0.0, 0.35, 0.65, 0.0, 0.0],
        3 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        4 => [0.3, 0.5, 0.0, 0.2, 0.0, 0.0, 0.0],
        5 => [0.45, 0.05, 0.0, 0.0, 0.0, 0.5, 0.0],
        // temperate villages: forest-strip villages in the hills
        6 => [0.35, 0.3, 0.25 * smoothstep(0.01, 0.04, slope) * 2.0, 0.1, 0.0, 0.0, 0.0],
        7 => [0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.6],
        8 => [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        9 => [0.0, 0.6, 0.0, 0.4, 0.0, 0.0, 0.0],
        10 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        _ => [0.35, 0.0, 0.0, 0.4, 0.0, 0.25, 0.0],
    };
    let tot: f64 = w.iter().sum();
    let mut acc = 0.0;
    for (i, wi) in w.iter().enumerate() {
        acc += wi / tot;
        if u < acc {
            return [NUCLEATED, LINEAR, WALDHUFEN, DISPERSED, KRAAL, PERCHED, CITY][i];
        }
    }
    NUCLEATED
}

/// The best candidate of a cell and its (base) existence: None without a scoring candidate or
/// when the existence draw fails.
fn base_place(w: &World, level: usize, c: (i64, i64, i64)) -> Option<Place> {
    let spec = &LEVELS[level];
    let atlas = w.atlas();
    let towns = w.cfg.landuse.towns;
    if towns <= 0.0 {
        return None;
    }
    // stage 1: atlas only (population, climate, coast) for every candidate
    let mut cands: Vec<(u32, u64, DVec3, crate::atlas::AtlasSample)> = Vec::with_capacity(spec.k as usize);
    for k in 0..spec.k {
        let Some((h, p)) = candidate(w, level, c, k) else { continue };
        let a = atlas.sample(p / p.length());
        if a.coast_km < 0.3 || a.glaciation > 0.6 {
            continue;
        }
        let climate = smoothstep(-14.0, 4.0, a.temp_c) * (0.3 + 0.7 * smoothstep(60.0, 450.0, a.precip_mm)) * (1.0 - smoothstep(0.3, 0.6, a.glaciation));
        let coast = 1.0 + 0.45 * (1.0 - smoothstep(2.0, 25.0, a.coast_km));
        let s1 = (0.04 + a.population) * climate * coast;
        if s1 <= 0.0 {
            continue;
        }
        cands.push((quant(s1), h, p, a));
    }
    if cands.is_empty() {
        return None;
    }
    cands.sort_unstable_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
    cands.truncate(3);
    // existence first (a cheap draw on the best candidate's atlas values: no relief samples for
    // the cells that have no settlement)
    let a0 = cands[0].3;
    let arch0 = (a0.archetype as usize).min(11);
    let culture_w = match level as u8 {
        HAMLET => ARCH_HAMLETS[arch0],
        VILLAGE => ARCH_VILLAGES[arch0],
        _ => ARCH_TOWNS[arch0],
    };
    let pe = (spec.base * smoothstep(spec.pop_window[0], spec.pop_window[1], a0.population) * culture_w * towns).min(0.97);
    let cell_h = hash3(level_seed(w, level) ^ 0xE715, c.0, c.1, c.2);
    if u01k(cell_h, 7) >= pe {
        return None;
    }
    // stage 2: flatness (hilltops for perched villages), elevation for the best three
    let lapse = w.cfg.climate.lapse_rate_c_per_km;
    let perch = matches!(arch0, 5 | 11) && level as u8 <= VILLAGE;
    let mut ranked: Vec<(u32, u64, DVec3, crate::atlas::AtlasSample, f64, f64, f64)> = Vec::with_capacity(3);
    for (s1q, h, p, a) in cands {
        let (h0, amp) = w.smooth_relief(p, 0.5 * spec.ring);
        if h0 < 1.0 {
            continue;
        }
        let (_, _, east, north) = frame_at(w, p);
        let mut lo = h0;
        let mut hi = h0;
        let mut sum = 0.0;
        let mut ge = 0.0;
        let mut gn = 0.0;
        for (de, dn) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            let q = p + (east * de + north * dn) * spec.ring;
            let hq = w.smooth_relief(q, 0.5 * spec.ring).0;
            lo = lo.min(hq);
            hi = hi.max(hq);
            sum += hq;
            ge += de * hq;
            gn += dn * hq;
        }
        let slope = (hi - lo) / (2.0 * spec.ring);
        let flat = if perch {
            // a hilltop: higher than its ring
            0.3 + 0.7 * smoothstep(-5.0, 30.0, h0 - 0.25 * sum) * (1.0 - smoothstep(0.08, 0.2, slope))
        } else {
            0.04 + 0.96 * (1.0 - smoothstep(0.012, 0.09, slope))
        };
        if hi < 0.5 && lo < 0.5 {
            continue;
        }
        let t_here = a.temp_c - lapse * h0 / 1000.0;
        let elev = smoothstep(-12.0, 2.0, t_here) * (1.0 - 0.8 * smoothstep(2_500.0, 4_500.0, h0)) * (1.0 - 0.6 * smoothstep(400.0, 1_800.0, amp));
        // water: no centre in open water; water in the footprint lowers the score, a river
        // nearby raises it (water access: crossings, mills, river ports)
        // (hamlets and villages: larger places are visited over thousands of kilometres by the
        // transport kit's trunk networks, where pass A at points costs too much; their
        // smooth-relief test keeps them off the sea)
        let wa = if level as u8 <= VILLAGE { water_access(w, level as u8, p, 0.8 * spec.radius[0]) } else { Some((1.0, f64::MAX)) };
        let Some((dry, river_d)) = wa else { continue };
        let water = (0.3 + 0.7 * dry) * (1.0 + RIVER_BONUS * (1.0 - smoothstep(150.0, 2_500.0, river_d)));
        let s = s1q as f64 / 64.0 * flat * elev * water;
        let q = quant(s);
        if q == 0 {
            continue;
        }
        // the downhill direction (valley axis) for street villages
        let axis = (-gn).atan2(-ge);
        ranked.push((q, h, p, a, h0, slope, axis));
    }
    // the best (then the first by hash); towns and larger: the best whose centre no lake can
    // reach (`lake_free`; hamlets and villages tested theirs above)
    ranked.sort_unstable_by(|x, y| y.0.cmp(&x.0).then(x.1.cmp(&y.1)));
    let (score, id, p, a, h0, slope, axis) = ranked.into_iter().find(|c| level as u8 <= VILLAGE || lake_free(w, c.2))?;
    let arch = (a.archetype as usize).min(11);
    // population (Zipf-like within the level: few large ones), by potential and development
    let t = (0.55 * u01k(id, 11).powf(2.2) + 0.45 * smoothstep(spec.pop_window[0], 1.0, a.population) * (0.6 + 0.4 * u01k(id, 12))).clamp(0.0, 1.0);
    let population = spec.pop[0] * (spec.pop[1] / spec.pop[0]).powf(t);
    let e = (spec.radius[1] / spec.radius[0]).ln() / (spec.pop[1] / spec.pop[0]).ln();
    let sprawl = ARCH_SPRAWL[arch] * (0.85 + 0.3 * a.development) * (0.9 + 0.2 * u01k(id, 13));
    let mut radius = (spec.radius[0] * (population / spec.pop[0]).powf(e) * sprawl).clamp(0.7 * spec.radius[0], 1.15 * spec.radius[1]);
    let (lat, lon, _, _) = frame_at(w, p);
    let lvl = level as u8;
    let fm = pick_form(id, lvl, arch, slope);
    if fm == form::PERCHED {
        radius *= 0.7;
    }
    let coastal = a.coast_km < radius / 1000.0 + 3.0;
    let airport = match lvl {
        METRO => true,
        CITY => a.development > 0.12 || population > 600_000.0,
        TOWN => population > 30_000.0 && a.development > 0.5,
        _ => false,
    };
    let port = coastal && lvl >= TOWN && (population > 15_000.0 || lvl >= CITY);
    // the street frame: survey-grid cultures keep north-south streets; street villages run
    // along their valley
    let (rot, elong) = match fm {
        form::LINEAR | form::WALDHUFEN => (axis, 2.6 + 2.0 * u01k(id, 21)),
        _ if arch == crate::atlas::Archetype::SurveyGrid as usize => (0.04 * (u01k(id, 20) - 0.5), 1.0 + 0.5 * u01k(id, 21).powi(2)),
        // (round 12: the streets of a town on sloping ground follow its relief: one axis down the
        // smooth relief's slope over the site's ring (the valley's axis its river runs along), the
        // other along the contours, the town stretched along the first; a random grid on flat land)
        _ => {
            let r0 = u01k(id, 20) * std::f64::consts::FRAC_PI_2;
            let k = smoothstep(0.003, 0.012, slope);
            // (the town stretched along that axis)
            let d = (axis - r0).rem_euclid(std::f64::consts::PI);
            let d = if d > std::f64::consts::FRAC_PI_2 { d - std::f64::consts::PI } else { d };
            (r0 + k * (d + 0.12 * (u01k(id, 22) - 0.5)), 1.0 + 0.7 * u01k(id, 21).powi(2))
        }
    };
    let elong = if fm == form::KRAAL || fm == form::PERCHED { 1.0 } else { elong };
    Some(Place {
        id,
        level: lvl,
        center: p,
        lat,
        lon,
        population,
        radius,
        culture: a.culture,
        arch: arch as u8,
        dev: a.development,
        coastal,
        airport,
        port,
        parent: 0,
        form: fm,
        rot,
        elong,
        score,
        wind: [a.wind_e, a.wind_n],
        temp: a.temp_c,
        precip: a.precip_mm,
        elev: h0,
        litho: a.litho as u8,
    })
}

/// Is the centre `p` of a town or larger clear of lakes? Not where a lattice lake or a
/// closed-basin lake of the major rivers can reach it (cheap conservative screens without
/// drainage pieces, see `World::lattice_lake_near`, `World::sink_near_level`): such a candidate
/// gives way to the next one. (Pass A at a point costs milliseconds, tens where a lake's level
/// must be found, and the transport kit's networks visit every town of a continent at low
/// zooms: the screens cost ~0.05 ms per town. The tributaries' and streams' closed basins,
/// ponds of a few hundred metres, would cost 0.4 ms: not screened.) The smooth-relief test of
/// the ring keeps towns off the sea; the built-up area may still reach a lake's shore.
fn lake_free(w: &World, p: DVec3) -> bool {
    !w.lattice_lake_near(p) && !w.sink_near_level(p, 0)
}

/// Score bonus of a site at a river's bank (level 0–1; fading out by 2.5 km).
const RIVER_BONUS: f64 = 0.3;

/// Open water at a candidate site and its access to rivers, by pass A at points
/// (`World::terrain`: the sea, the core's lattice and closed-basin lakes, the kits' lakes, the
/// channels): None where the centre is in water (a river of levels 0–1, half width > 5.5 m);
/// else the dry share of a ring of 4 points at `ring` m (pass A there for hamlets and villages,
/// the relief's sea for larger levels) and the distance (m) from the centre to the nearest level 0–1 channel's bank
/// within the carve's reach. Analytic (host Rust), the same for both backends.
fn water_access(w: &World, level: u8, p: DVec3, ring: f64) -> Option<(f64, f64)> {
    let at = |q: DVec3| {
        let (lat, lon, _, _) = frame_at(w, q);
        crate::world::Ctx::new(lat, lon, 20.0, &w.ell)
    };
    let wet =
        |t: &crate::world::Terrain| (t.water_kind != crate::world::water::NONE && t.water > t.ground) || (t.river_hw > 5.5 && t.river_d.abs() < t.river_hw);
    let tc = w.terrain(&at(p));
    if wet(&tc) {
        return None;
    }
    let (_, _, east, north) = frame_at(w, p);
    let mut dry = 0.0;
    for k in 0..4 {
        let (s, co) = (k as f64 * std::f64::consts::FRAC_PI_2 + 0.4).sin_cos();
        let c = at(p + (east * co + north * s) * ring);
        let ok = if level <= VILLAGE { !wet(&w.terrain(&c)) } else { w.relief_height(&c, &w.macro_at(c.p, 20.0)) >= 0.5 };
        if ok {
            dry += 0.25;
        }
    }
    let river = if tc.river_hw > 5.5 { (tc.river_d.abs() - tc.river_hw).max(0.0) } else { f64::MAX };
    Some((dry, river))
}

// ---------------------------------------------------------------- expected places (glow)

/// Measured ratio of the final places' Σ r² to the expectation's (`expected_places`) per level:
/// the existence draws' survival (suppression by neighbours and higher levels, cells without a
/// scoring candidate, water) and the radius draw's clamp; `calibrate_glow` (ignored test).
pub const GLOW_CAL: [f64; NLEVELS] = [0.76, 1.19, 1.24, 1.28, 1.43];

/// The places of `level` expected per m² at a point of population potential `pop`, culture
/// archetype `arch` (towns: `landuse.towns`), and their mean squared built-up radius (m²) by
/// the development `dev`: the existence and population draws of `base_place` integrated over
/// their hashes. The settlements kit's low-zoom glow where a level's places are not listed.
pub fn expected_places(level: usize, pop: f64, arch: usize, dev: f64, towns: f64) -> (f64, f64) {
    let spec = &LEVELS[level];
    let arch = arch.min(11);
    let culture_w = match level as u8 {
        HAMLET => ARCH_HAMLETS[arch],
        VILLAGE => ARCH_VILLAGES[arch],
        _ => ARCH_TOWNS[arch],
    };
    let pe = (spec.base * smoothstep(spec.pop_window[0], spec.pop_window[1], pop) * culture_w * towns).min(0.97);
    let n = pe / (spec.cell * spec.cell);
    (n, mean_r2(level, pop, arch, dev))
}

/// E[r²] of a place of `level` (`base_place`: t = 0.55 u₁^2.2 + 0.45 s (0.6 + 0.4 u₂),
/// r = r₀ (r₁/r₀)^t × sprawl; the clamp left out): r₀² · C(level) · e^{0.27 a s} ·
/// (e^{0.18 a s} − 1) / (0.18 a s) · E[sprawl²], a = 2 ln(r₁/r₀).
pub fn mean_r2(level: usize, pop: f64, arch: usize, dev: f64) -> f64 {
    let spec = &LEVELS[level];
    let a = 2.0 * (spec.radius[1] / spec.radius[0]).ln();
    let s = smoothstep(spec.pop_window[0], 1.0, pop);
    let x = 0.18 * a * s;
    let g = if x > 1e-6 { (x.exp() - 1.0) / x } else { 1.0 };
    let sprawl2 = (ARCH_SPRAWL[arch.min(11)] * (0.85 + 0.3 * dev)).powi(2) * (1.0 + 0.04 / 12.0);
    spec.radius[0] * spec.radius[0] * GLOW_C1[level] * (0.27 * a * s).exp() * g * sprawl2
}

/// E[e^{0.55 a u^2.2}] over u in [0, 1] per level (test `glow_constants`).
pub const GLOW_C1: [f64; NLEVELS] = [1.6437, 1.9004, 1.707, 1.7338, 1.4268];

fn quant(s: f64) -> u32 {
    (s * 64.0).floor().clamp(0.0, 1e9) as u32
}

type Key = (u64, u8, bool, i64, i64, i64);

/// Shared cache of base and final places (pure functions of the world and the cell).
static CACHE: crate::memo::Memo<Key, Option<Place>> = crate::memo::Memo::new(1 << 20);

fn cached(w: &World, level: usize, fin: bool, c: (i64, i64, i64), f: impl FnOnce() -> Option<Place>) -> Option<Place> {
    CACHE.get_or((w.cache_key, level as u8, fin, c.0, c.1, c.2), f)
}

fn base(w: &World, level: usize, c: (i64, i64, i64)) -> Option<Place> {
    cached(w, level, false, c, || base_place(w, level, c))
}

/// The place of a cell after suppression (None: no settlement there).
pub fn final_place(w: &World, level: usize, c: (i64, i64, i64)) -> Option<Place> {
    cached(w, level, true, c, || resolve(w, level, c))
}

fn resolve(w: &World, level: usize, c: (i64, i64, i64)) -> Option<Place> {
    let mut p = base(w, level, c)?;
    let spec = &LEVELS[level];
    // the same level: a higher-priority base site within the spacing (or overlapping) wins
    let reach = (spec.spacing * spec.cell).max(2.4 * spec.radius[1]);
    let mut lost = false;
    for_cells(w, level, p.center, reach, |cc| {
        if lost || cc == c {
            return;
        }
        if let Some(o) = base(w, level, cc) {
            let d = (o.center - p.center).length();
            if (d < spec.spacing * spec.cell || d < 1.05 * (o.extent() + p.extent())) && o.outranks(&p) {
                lost = true;
            }
        }
    });
    if lost {
        return None;
    }
    // higher levels (final): sub-centre inside a larger footprint (towns, cities), removed
    // where the footprints would overlap
    let mut parent: Option<Place> = None;
    for (hl, spec_h) in LEVELS.iter().enumerate().skip(level + 1) {
        let r = spec_h.radius[1] * 1.15 * 1.62 * 1.3 + p.extent() + 500.0;
        for_cells(w, hl, p.center, r, |cc| {
            if lost {
                return;
            }
            if let Some(h) = final_place(w, hl, cc) {
                let d = (h.center - p.center).length();
                if level >= TOWN as usize && d < 0.8 * h.radius && h.parent == 0 {
                    if parent.is_none_or(|q| h.outranks(&q)) {
                        parent = Some(h);
                    }
                } else if d < h.extent() + p.extent() + 200.0 {
                    lost = true;
                }
            }
        });
    }
    if lost {
        return None;
    }
    if let Some(h) = parent {
        p.parent = h.id;
        // a sub-centre: its own core inside the parent's footprint
        p.radius = p.radius.min(0.45 * h.radius);
        p.airport = false;
        p.port = p.port && h.port;
    }
    Some(p)
}

/// Clear the shared cache (tests).
#[cfg(test)]
pub fn clear_cache() {
    CACHE.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> World {
        World::new(crate::Config::default())
    }

    fn land_points(w: &World, n: u64, pop: f64) -> Vec<DVec3> {
        let mut v = Vec::new();
        for i in 0..n * 400 {
            let h = mix64(0x5E77 ^ i);
            let lat = (2.0 * u01k(h, 1) - 1.0).asin();
            let lon = (u01k(h, 2) - 0.5) * std::f64::consts::TAU;
            let p = geodesy::geodetic2ecef(geodesy::Geodetic::new(lat, lon, 0.0), &w.ell);
            let a = w.atlas().sample(p.normalize());
            if a.coast_km > 50.0 && a.population > pop {
                v.push(p);
                if v.len() as u64 == n {
                    break;
                }
            }
        }
        v
    }

    /// The same query twice (also after clearing the cache, and from a shifted centre) gives
    /// the same settlements: a pure function of the world.
    #[test]
    fn deterministic_and_tile_independent() {
        let w = world();
        for p in land_points(&w, 3, 0.2) {
            let a = settlements_near_min_level(&w, p, 60_000.0, TOWN);
            clear_cache();
            let b = settlements_near_min_level(&w, p, 60_000.0, TOWN);
            assert_eq!(a, b);
            // a neighbouring query's view of the same area agrees
            let (_, _, east, _) = frame_at(&w, p);
            let c = settlements_near_min_level(&w, p + east * 30_000.0, 100_000.0, TOWN);
            for s in &a {
                let o = c.iter().find(|x| x.id == s.id).expect("seen from the neighbour");
                assert_eq!(s, o);
            }
            assert!(a.windows(2).all(|x| x[0].id < x[1].id));
        }
    }

    /// Counts per level in populated land: a hierarchy (many villages, fewer towns, few
    /// cities), fields in range, sub-centres inside their parents.
    #[test]
    fn hierarchy() {
        let w = world();
        let mut n = [0usize; NLEVELS];
        let pts = land_points(&w, 8, 0.15);
        let mut forms = [0usize; 7];
        for p in &pts {
            let all = settlements_near(&w, *p, 25_000.0);
            for s in &all {
                n[s.level as usize] += 1;
                assert!(s.radius_m > 0.0 && s.population > 0.0);
                assert!((s.ecef.length() - 6.37e6).abs() < 30_000.0);
                assert!((0.0..=1.0).contains(&s.development));
            }
            for pl in places_near(&w, VILLAGE as usize, *p, 25_000.0) {
                forms[pl.form as usize] += 1;
            }
        }
        eprintln!("levels in {} discs of 25 km: {n:?}; village forms {forms:?}", pts.len());
        assert!(n[HAMLET as usize] > n[TOWN as usize], "{n:?}");
        assert!(n[VILLAGE as usize] > n[TOWN as usize], "{n:?}");
        assert!(n[VILLAGE as usize] > 0, "{n:?}");
        // cities exist within 300 km of populated land
        let cities: usize = pts.iter().map(|p| settlements_near_min_level(&w, *p, 300_000.0, CITY).len()).sum();
        assert!(cities > 0);
    }

    /// No two final places of one level overlap, and a lower-level place overlaps a larger
    /// one only as its sub-centre.
    #[test]
    fn no_overlaps() {
        let w = world();
        for p in land_points(&w, 4, 0.25) {
            let mut all: Vec<Place> = Vec::new();
            for l in 0..NLEVELS {
                all.extend(places_near(&w, l, p, 40_000.0));
            }
            for a in &all {
                for b in &all {
                    if a.id >= b.id || a.parent == b.id || b.parent == a.id {
                        continue;
                    }
                    let d = (a.center - b.center).length();
                    assert!(d > 0.9 * (a.radius + b.radius), "{} (level {}) and {} (level {}) overlap: {d:.0} m", a.id, a.level, b.id, b.level);
                }
            }
        }
    }

    /// The glow's integral constants are current (`GLOW_C1`).
    #[test]
    fn glow_constants() {
        for (l, spec) in LEVELS.iter().enumerate() {
            let a = 2.0 * (spec.radius[1] / spec.radius[0]).ln();
            let m = 20_000;
            let c: f64 = (0..m).map(|i| (0.55 * a * ((i as f64 + 0.5) / m as f64).powf(2.2)).exp()).sum::<f64>() / m as f64;
            assert!((c - GLOW_C1[l]).abs() < 1e-3, "GLOW_C1[{l}] = {c:.4}");
        }
    }

    /// Debug: calibrate the low-zoom glow (`GLOW_CAL`): the final places per level within
    /// discs of populated land against `expected_places` integrated over the discs (count and
    /// Σ r²): `calibrate_glow -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn calibrate_glow() {
        let w = world();
        let radius = [5_000.0, 20_000.0, 70_000.0, 250_000.0, 900_000.0];
        let pts = land_points(&w, 24, 0.05);
        for (l, &r) in radius.iter().enumerate() {
            let (mut n_act, mut r2_act, mut n_exp, mut r2_exp) = (0.0, 0.0, 0.0, 0.0);
            for (k, p) in pts.iter().enumerate() {
                if l >= 3 && k >= 8 {
                    break;
                }
                for pl in places_near(&w, l, *p, r) {
                    n_act += 1.0;
                    r2_act += pl.radius * pl.radius;
                }
                // the expectation over the disc (a 24 × 24 grid of atlas samples)
                let (_, _, east, north) = frame_at(&w, *p);
                let g = 24;
                let da = (2.0 * r / g as f64).powi(2);
                for i in 0..g {
                    for j in 0..g {
                        let (u, v) = ((i as f64 + 0.5) / g as f64 * 2.0 - 1.0, (j as f64 + 0.5) / g as f64 * 2.0 - 1.0);
                        if u * u + v * v > 1.0 {
                            continue;
                        }
                        let q = crate::kits::transport::route::surface(&w, *p + (east * u + north * v) * r);
                        let a = w.atlas().sample(q.normalize());
                        if a.coast_km < 0.3 {
                            continue;
                        }
                        let (n, r2) = expected_places(l, a.population, a.archetype as usize, a.development, w.cfg.landuse.towns);
                        n_exp += n * da;
                        r2_exp += n * r2 * da;
                    }
                }
            }
            eprintln!(
                "level {l}: {n_act} places (expected {n_exp:.1}: survival {:.3}), Σr² {r2_act:.3e} (expected {r2_exp:.3e}: ratio {:.3})",
                n_act / n_exp.max(1e-9),
                r2_act / r2_exp.max(1e-9)
            );
        }
    }

    /// Places keep their centres out of open water (pass A at the centre: the sea, lakes,
    /// rivers), and their scores prefer river banks.
    #[test]
    fn places_avoid_open_water() {
        let w = world();
        let (mut n, mut wet) = (0, 0);
        for p in land_points(&w, 6, 0.15) {
            for s in settlements_near(&w, p, 12_000.0) {
                let g = geodesy::ecef2geodetic(s.ecef, &w.ell);
                let t = w.terrain(&crate::world::Ctx::new(g.lat, g.lon, 20.0, &w.ell));
                n += 1;
                // (rivers of levels 0–1: half widths from 6 m)
                if (t.water_kind != crate::world::water::NONE && t.water > t.ground) || (t.river_hw > 5.5 && t.river_d.abs() < t.river_hw) {
                    wet += 1;
                    eprintln!("in water: level {} at ({:.4}, {:.4}) kind {}", s.level, s.lat_deg, s.lon_deg, t.water_kind);
                }
            }
        }
        eprintln!("{wet} of {n} place centres in water");
        assert!(n > 50 && wet * 100 <= n, "{wet} of {n}");
    }

    /// Towns and cities keep their centres out of lakes and the sea (pass A at the centre; their
    /// candidates avoid the reach of possible lattice and closed-basin lakes).
    #[test]
    fn towns_avoid_lakes() {
        let w = world();
        let t0 = std::time::Instant::now();
        let mut towns = Vec::new();
        for p in land_points(&w, 4, 0.1) {
            towns.extend(settlements_near_min_level(&w, p, 1_500_000.0, TOWN));
        }
        eprintln!("{} towns and cities in {:.2} s", towns.len(), t0.elapsed().as_secs_f64());
        let n = towns.len().min(3000);
        let mut wet = 0;
        for s in &towns[..n] {
            let t = w.terrain(&crate::world::Ctx::new(s.lat_deg.to_radians(), s.lon_deg.to_radians(), 20.0, &w.ell));
            if t.water_kind != crate::world::water::NONE && t.water > t.ground {
                wet += 1;
                eprintln!("in water: level {} at ({:.4}, {:.4}) kind {}", s.level, s.lat_deg, s.lon_deg, t.water_kind);
            }
        }
        // (7 of these 3000 were in lakes before the screens: 5 in lattice and major rivers'
        // closed-basin lakes, 2 in streams' ponds of ~200 m, which are not screened)
        assert!(n >= 2000 && wet * 1000 <= n, "{wet} of {n} town centres in water");
    }

    /// Debug: places of each level and form in the default world:
    /// `find_settlement_places -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn find_settlement_places() {
        let w = world();
        let pts = land_points(&w, 60, 0.1);
        let mut seen = std::collections::BTreeMap::new();
        for p in &pts {
            for l in 0..NLEVELS {
                for pl in places_near(&w, l, *p, if l >= 3 { 400_000.0 } else { 30_000.0 }) {
                    seen.entry((pl.level, pl.form, pl.arch)).or_insert(pl);
                }
            }
        }
        for ((l, f, a), pl) in &seen {
            eprintln!(
                "level {l} form {f} arch {a}: ({:.4}, {:.4}) R {:.0} pop {:.0} dev {:.2} coast {}",
                pl.lat.to_degrees(),
                pl.lon.to_degrees(),
                pl.radius,
                pl.population,
                pl.dev,
                pl.coastal
            );
        }
    }
}
