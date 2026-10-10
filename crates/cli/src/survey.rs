//! `terrain survey`: find diverse places of the world quickly and look at them.
//!
//! 1. **Candidates:** random points (area-uniform, |lat| <= 80°, deterministic from
//!    `--seed-places`) on land or near a coast, each classified by cheap point queries of the
//!    generator ([`Sampler`]): a 5 × 5 stencil 500 m apart (land cover, DSM height, water) gives
//!    the class mix, elevation, relief, slope, coast / river / town flags and climate.
//! 2. **Selection** ([`select`]): one place per *theme* found (coast, mountains, town, river,
//!    snow and ice, desert, forest, farmland, wetland, plateau, lake, tundra; at most half of the
//!    places), then farthest-point sampling in a feature space (class mix, elevation, relief,
//!    climate, flags, distance on the globe), at least `MIN_SEPARATION_KM` apart where possible.
//! 3. **Stills** of every place with the dataset renderer and sensor (GPU when there is one):
//!    the views of `--views` (oblique 1.5 km above the ground looking 25° down, nadir from
//!    800 m, optionally high from 10 km), the sun at `--local-time` local solar time on
//!    `--date`, coming from the side. The tiles a view needs (dry runs of the renderer's tile
//!    selection) are generated into one store (`--tiles`), one place at a time (memory stays
//!    bounded: one renderer and tile cache for all places).
//! 4. **Outputs:** `OUT/NN_<lat>_<lon>_<view>.png`, a labelled contact sheet `OUT/sheet.jpg`
//!    and `OUT/places.csv`; `--places FILE.csv` renders the places of a CSV again (regression
//!    stills, before / after comparisons at the same spots).

use crate::Common;
use anyhow::{bail, Context, Result};
use clap::Args;
use geodesy::Geodetic;
use image::RgbImage;
use rayon::prelude::*;
use render::camera::{CameraConfig, Extrinsics, Mount};
use render::scenario::{CameraSpec, Scenario};
use render::sensor::Sensor;
use render::{pipeline, Pose};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use terragen::landcover::{self as lc, Group};
use terragen::Generator;

/// The most places of one survey (thumbnails of all of them are kept for the contact sheet).
pub const MAX_PLACES: usize = 128;
/// The most candidate points (each costs 26 point queries).
pub const MAX_SAMPLES: usize = 100_000;
/// The most views per place.
pub const MAX_VIEWS: usize = 4;
/// The largest still (pixels per side).
pub const MAX_SIDE: u32 = 4096;
/// Places are at least this far apart (km) while candidates that far away remain.
pub const MIN_SEPARATION_KM: f64 = 150.0;
/// Spacing (m) and side (points) of the classification stencil.
const STENCIL_M: f64 = 500.0;
const STENCIL_N: usize = 5;
/// Pixel size (m) of the stencil's point queries.
const PROBE_GSD: f64 = 30.0;
const EARTH_R: f64 = 6_371_000.0;

#[derive(Args)]
pub struct SurveyArgs {
    #[command(flatten)]
    pub common: Common,
    /// Output directory: NN_<lat>_<lon>_<view>.png, sheet.jpg, places.csv.
    #[arg(long, short, default_value = "out/survey")]
    pub out: PathBuf,
    /// Number of places (at most 128).
    #[arg(long, default_value_t = 24)]
    pub count: usize,
    /// Seed of the random candidate points (the world is world.seed / --seed).
    #[arg(long, default_value_t = 1)]
    pub seed_places: u64,
    /// Candidate points on land and coasts to choose from (at least --count, at most 100000).
    #[arg(long, default_value_t = 2000)]
    pub samples: usize,
    /// Render the places of this CSV (`lat`, `lon` columns; a places.csv of an earlier survey)
    /// instead of searching.
    #[arg(long)]
    pub places: Option<PathBuf>,
    /// Views per place (comma separated, at most 4): oblique (1500 m above the ground, 25°
    /// down), nadir (800 m), high (10 km, 30° down), or NAME=AGL_M@PITCH_DEG.
    #[arg(long, value_delimiter = ',', default_value = "oblique,nadir")]
    pub views: Vec<String>,
    /// Still size WxH (pixels).
    #[arg(long, default_value = "960x540")]
    pub size: String,
    /// Horizontal field of view (deg).
    #[arg(long, default_value_t = 70.0)]
    pub hfov: f64,
    /// Tile store the views' tiles are generated into (created if missing; reused, so a second
    /// run renders right away; ~300 MB per place). Default: OUT/tiles.h5.
    #[arg(long)]
    pub tiles: Option<PathBuf>,
    /// Local solar time of the stills (hours).
    #[arg(long, default_value_t = 10.5)]
    pub local_time: f64,
    /// Date of the sun position (YYYY-MM-DD; the equinox lights both hemispheres alike).
    #[arg(long, default_value = "2026-03-20")]
    pub date: String,
    /// Only choose the places (places.csv); render nothing.
    #[arg(long)]
    pub no_render: bool,
}

// ============================================================================= views

/// A camera placement relative to a place: height above the ground and pitch.
#[derive(Clone, Debug, PartialEq)]
pub struct ViewSpec {
    pub name: String,
    /// metres above the ground (the highest stencil point gets 40 % of it at least)
    pub agl: f64,
    /// degrees, negative = down (−90: nadir)
    pub pitch: f64,
}

impl std::str::FromStr for ViewSpec {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        let preset = |name: &str, agl: f64, pitch: f64| Ok(ViewSpec { name: name.into(), agl, pitch });
        match s.trim() {
            "oblique" => return preset("oblique", 1500.0, -25.0),
            "nadir" => return preset("nadir", 800.0, -90.0),
            "high" => return preset("high", 10_000.0, -30.0),
            _ => {}
        }
        const FORM: &str = "a view is oblique, nadir, high or NAME=AGL_M@PITCH_DEG (e.g. low=300@-20)";
        let (name, rest) = s.trim().split_once('=').with_context(|| format!("{s:?}: {FORM}"))?;
        let (agl, pitch) = rest.split_once('@').with_context(|| format!("{s:?}: {FORM}"))?;
        let (agl, pitch): (f64, f64) =
            (agl.trim().parse().with_context(|| format!("{s:?}: {FORM}"))?, pitch.trim().parse().with_context(|| format!("{s:?}: {FORM}"))?);
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            bail!("{s:?}: the view name must be lower-case letters, digits or _");
        }
        if !(agl.is_finite() && (10.0..=100_000.0).contains(&agl)) {
            bail!("{s:?}: the height above the ground must be 10 .. 100000 m");
        }
        if !(pitch.is_finite() && (-90.0..=0.0).contains(&pitch)) {
            bail!("{s:?}: the pitch must be -90 .. 0 deg");
        }
        Ok(ViewSpec { name: name.into(), agl, pitch })
    }
}

/// The checked arguments.
#[derive(Debug)]
pub struct Plan {
    pub views: Vec<ViewSpec>,
    pub size: (u32, u32),
    /// Unix time (s) of `--date` 00:00 UTC
    pub date_unix: f64,
    /// local solar time of the stills (h)
    pub local_time: f64,
}

/// Check the arguments (bounded counts and sizes: the survey's memory stays small).
pub fn check_args(a: &SurveyArgs) -> Result<Plan> {
    if a.places.is_none() {
        if !(1..=MAX_PLACES).contains(&a.count) {
            bail!("--count {}: 1 ..= {MAX_PLACES} places", a.count);
        }
        if !(a.count..=MAX_SAMPLES).contains(&a.samples) {
            bail!("--samples {}: at least --count ({}) and at most {MAX_SAMPLES}", a.samples, a.count);
        }
    }
    let views: Vec<ViewSpec> = a.views.iter().map(|v| v.parse()).collect::<Result<_>>()?;
    if views.is_empty() || views.len() > MAX_VIEWS {
        bail!("--views: 1 ..= {MAX_VIEWS} views");
    }
    for (i, v) in views.iter().enumerate() {
        if views[..i].iter().any(|w| w.name == v.name) {
            bail!("--views: {} twice", v.name);
        }
    }
    let size = a.size.split_once('x').and_then(|(w, h)| Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?)));
    let Some(size) = size.filter(|&(w, h)| (16..=MAX_SIDE).contains(&w) && (16..=MAX_SIDE).contains(&h)) else {
        bail!("--size {:?}: WxH with 16 ..= {MAX_SIDE} pixels per side", a.size);
    };
    if !(a.hfov.is_finite() && a.hfov > 5.0 && a.hfov < 150.0) {
        bail!("--hfov {}: 5 .. 150 deg", a.hfov);
    }
    if !(a.local_time.is_finite() && (0.0..=24.0).contains(&a.local_time)) {
        bail!("--local-time {}: 0 ..= 24 h", a.local_time);
    }
    let date_unix = parse_date(&a.date).with_context(|| format!("--date {:?}: YYYY-MM-DD", a.date))?;
    Ok(Plan { views, size, date_unix, local_time: a.local_time })
}

/// Unix time of a date's midnight UTC.
fn parse_date(s: &str) -> Result<f64> {
    let f: Vec<i64> = s.trim().split('-').map(|x| x.parse()).collect::<Result<_, _>>()?;
    let [y, m, d] = f[..] else { bail!("three fields expected") };
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || !(1900..=2200).contains(&y) {
        bail!("out of range");
    }
    // days from civil (H. Hinnant)
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Ok((era * 146_097 + doe - 719_468) as f64 * 86_400.0)
}

// ============================================================================= sampling

/// The world at one point (a cheap query of the generator).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Probe {
    /// surface height (DSM or water level, m above the ellipsoid)
    pub height: f64,
    pub class: u8,
    pub ocean: bool,
    /// a perennial river within ~150 m of its bank
    pub river: bool,
    /// mean annual temperature (°C) and moisture (0..1)
    pub temp: f64,
    pub moist: f64,
    /// The biome of the ecoregion (registry index).
    pub biome: Option<u32>,
}

/// Point queries of a world (the generator, or a mock in tests).
pub trait Sampler: Sync {
    /// The world at (lat, lon) (radians) seen with pixels of `gsd` metres.
    fn probe(&self, lat: f64, lon: f64, gsd: f64) -> Probe;

    /// Is each point (lat, lon in radians) in the sea (seen with 250 m pixels)? A batch: the
    /// generator answers it on the GPU.
    fn in_sea(&self, pts: &[(f64, f64)]) -> Vec<bool> {
        pts.par_iter().map(|&(lat, lon)| self.probe(lat, lon, 250.0).ocean).collect()
    }
}

impl Sampler for Generator {
    fn probe(&self, lat: f64, lon: f64, gsd: f64) -> Probe {
        let (t, h, class) = Generator::probe(self, lat, lon, gsd);
        Probe {
            height: if t.water.is_finite() { h.max(t.water) } else { h },
            class,
            ocean: t.water_kind == terragen::world::water::OCEAN,
            river: t.river_wet > 0.5 && t.river_hw > 3.0 && t.river_d.abs() < t.river_hw + 150.0,
            temp: t.temp,
            moist: t.moist,
            biome: Some(self.biome(&t).0 as u32),
        }
    }

    fn in_sea(&self, pts: &[(f64, f64)]) -> Vec<bool> {
        let mut out = Vec::with_capacity(pts.len());
        for c in pts.chunks(8192) {
            let q: Vec<(f64, f64, f64)> = c.iter().map(|&(lat, lon)| (lat, lon, 250.0)).collect();
            match self.terrain_points(&q) {
                Ok(t) => out.extend(t.iter().map(|t| t.water_kind == terragen::world::water::OCEAN)),
                Err(_) => out.par_extend(c.par_iter().map(|&(lat, lon)| Sampler::probe(self, lat, lon, 250.0).ocean)),
            }
        }
        out
    }
}

/// What the survey knows about a place.
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    /// degrees
    pub lat: f64,
    pub lon: f64,
    /// surface height at the centre and the highest stencil point (m)
    pub elevation: f64,
    pub top: f64,
    /// highest − lowest stencil point (m)
    pub relief: f64,
    /// mean slope between neighbouring stencil points (m/m)
    pub slope: f64,
    /// fraction of water classes in the stencil
    pub water: f64,
    pub coast: bool,
    pub river: bool,
    pub town: bool,
    pub temp: f64,
    pub moist: f64,
    /// stencil fractions of the legacy classes 0–17 and of the groups
    pub legacy: [f64; 18],
    pub groups: [f64; 11],
    /// the most frequent classes (id, fraction), most frequent first (at most 3)
    pub dominant: Vec<(u8, f64)>,
    pub biome: Option<u32>,
    /// why it was chosen (a theme, or "spread" for farthest-point picks)
    pub theme: String,
}

impl Place {
    /// "forest 52% grass 30%" (the `dominant` column).
    pub fn dominant_str(&self) -> String {
        self.dominant.iter().map(|(c, f)| format!("{} {:.0}%", lc::name(*c), f * 100.0)).collect::<Vec<_>>().join(" ")
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// (lat, lon) (radians) at `east`, `north` metres from (lat, lon) (local spherical offset).
fn offset(lat: f64, lon: f64, east: f64, north: f64) -> (f64, f64) {
    (lat + north / EARTH_R, lon + east / (EARTH_R * lat.cos().max(0.05)))
}

/// Great-circle distance (km) between two points in degrees.
pub fn distance_km(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (la1, lo1, la2, lo2) = (a.0.to_radians(), a.1.to_radians(), b.0.to_radians(), b.1.to_radians());
    let h = ((la2 - la1) * 0.5).sin().powi(2) + la1.cos() * la2.cos() * ((lo2 - lo1) * 0.5).sin().powi(2);
    2.0 * EARTH_R / 1000.0 * h.sqrt().min(1.0).asin()
}

/// Random point `i` of the seed (area-uniform; None beyond 80° of latitude).
fn random_point(seed: u64, i: u64) -> Option<(f64, f64)> {
    let h = splitmix(seed ^ splitmix(i.wrapping_add(0x5u64 << 60)));
    let lat = (2.0 * unit(h) - 1.0).asin();
    let lon = (2.0 * unit(splitmix(h)) - 1.0) * std::f64::consts::PI;
    (lat.abs() <= 80f64.to_radians()).then_some((lat, lon))
}

/// Directions (east, north unit offsets) of the coast search around a point in the sea.
fn ring() -> [(f64, f64); 8] {
    std::array::from_fn(|k| {
        let a = k as f64 * std::f64::consts::FRAC_PI_4;
        (a.sin(), a.cos())
    })
}

/// Candidates among the random points `range` of the seed: points on land, and points in the
/// sea with land within 4 km, moved 3 km towards it (the stencil then spans the coast).
fn candidate_points(s: &dyn Sampler, seed: u64, range: std::ops::Range<u64>) -> Vec<(f64, f64)> {
    let pts: Vec<(f64, f64)> = range.filter_map(|i| random_point(seed, i)).collect();
    let sea = s.in_sea(&pts);
    let probe_ring: Vec<(f64, f64)> =
        pts.iter().zip(&sea).filter(|(_, &w)| w).flat_map(|(&(lat, lon), _)| ring().map(|(e, n)| offset(lat, lon, 4000.0 * e, 4000.0 * n))).collect();
    let ring_sea = s.in_sea(&probe_ring);
    let mut out = Vec::new();
    let mut r = 0;
    for (&(lat, lon), &w) in pts.iter().zip(&sea) {
        if !w {
            out.push((lat, lon));
            continue;
        }
        if let Some(k) = (0..8).find(|k| !ring_sea[r + k]) {
            let (e, n) = ring()[k];
            out.push(offset(lat, lon, 3000.0 * e, 3000.0 * n));
        }
        r += 8;
    }
    out
}

/// Classify the place at (lat, lon) (radians) from its stencil.
pub fn classify(s: &dyn Sampler, lat: f64, lon: f64) -> Place {
    let n = STENCIL_N;
    let half = (n / 2) as f64;
    let probes: Vec<Probe> = (0..n * n)
        .map(|k| {
            let (la, lo) = offset(lat, lon, ((k % n) as f64 - half) * STENCIL_M, ((k / n) as f64 - half) * STENCIL_M);
            s.probe(la, lo, PROBE_GSD)
        })
        .collect();
    let c = &probes[n * n / 2];
    let hs: Vec<f64> = probes.iter().map(|p| p.height).collect();
    let (lo, hi) = hs.iter().fold((f64::MAX, f64::MIN), |(a, b), &h| (a.min(h), b.max(h)));
    let mut grad = 0.0;
    for j in 0..n {
        for i in 0..n {
            if i + 1 < n {
                grad += (hs[j * n + i + 1] - hs[j * n + i]).abs();
            }
            if j + 1 < n {
                grad += (hs[(j + 1) * n + i] - hs[j * n + i]).abs();
            }
        }
    }
    let inv = 1.0 / (n * n) as f64;
    let mut counts = [0usize; 256];
    let mut legacy = [0.0; 18];
    let mut groups = [0.0; 11];
    for p in &probes {
        counts[p.class as usize] += 1;
        legacy[lc::legacy(p.class) as usize] += inv;
        groups[lc::group(p.class) as usize] += inv;
    }
    let mut dominant: Vec<(u8, f64)> = (0..256).filter(|&k| counts[k] > 0).map(|k| (k as u8, counts[k] as f64 * inv)).collect();
    dominant.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    dominant.truncate(3);
    let water = groups[Group::Water as usize];
    let ocean = probes.iter().any(|p| p.ocean);
    Place {
        lat: lat.to_degrees(),
        lon: lon.to_degrees(),
        elevation: c.height,
        top: hi,
        relief: hi - lo,
        slope: grad / (2 * n * (n - 1)) as f64 / STENCIL_M,
        water,
        coast: ocean && probes.iter().any(|p| !p.ocean && lc::group(p.class) != Group::Water),
        river: probes.iter().any(|p| p.class == lc::RIVER) || c.river,
        town: groups[Group::Built as usize] + groups[Group::Transport as usize] >= 0.08,
        temp: c.temp,
        moist: c.moist,
        legacy,
        groups,
        dominant,
        biome: c.biome,
        theme: String::new(),
    }
}

/// `samples` classified candidates of the seed (deterministic: independent of the threads).
pub fn candidates(s: &dyn Sampler, seed: u64, samples: usize) -> Vec<Place> {
    const CHUNK: u64 = 4096;
    let max_attempts = (samples as u64).saturating_mul(200).max(CHUNK);
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(samples);
    let mut next = 0u64;
    while pts.len() < samples && next < max_attempts {
        pts.extend(candidate_points(s, seed, next..next + CHUNK));
        next += CHUNK;
    }
    pts.truncate(samples);
    pts.par_iter().map(|&(lat, lon)| classify(s, lat, lon)).collect()
}

// ============================================================================= selection

/// A theme: a kind of place the survey wants one of (if the world has it).
struct Theme {
    name: &'static str,
    /// how well a place shows it (None: not at all)
    score: fn(&Place) -> Option<f64>,
}

fn g(p: &Place, grp: Group) -> f64 {
    p.groups[grp as usize]
}

const THEMES: &[Theme] = &[
    Theme { name: "coast", score: |p| p.coast.then(|| 0.5 - (p.water - 0.5).abs()) },
    Theme { name: "mountains", score: |p| (p.relief > 500.0).then(|| p.relief / 1000.0 + p.elevation / 4000.0) },
    Theme { name: "town", score: |p| p.town.then(|| g(p, Group::Built) + g(p, Group::Transport)) },
    Theme { name: "river", score: |p| (p.river && g(p, Group::SnowIce) < 0.5).then(|| 1.0 - p.water + 4.0 * p.legacy[lc::RIVER as usize]) },
    Theme { name: "snow-ice", score: |p| (g(p, Group::SnowIce) > 0.3).then(|| g(p, Group::SnowIce)) },
    Theme { name: "desert", score: |p| (g(p, Group::Bare) > 0.5 && p.moist < 0.35).then(|| g(p, Group::Bare) - p.moist) },
    Theme { name: "forest", score: |p| (g(p, Group::Forest) > 0.6).then(|| g(p, Group::Forest)) },
    Theme { name: "farmland", score: |p| (g(p, Group::Agriculture) > 0.4).then(|| g(p, Group::Agriculture)) },
    Theme { name: "wetland", score: |p| (g(p, Group::Wetland) > 0.15).then(|| g(p, Group::Wetland)) },
    Theme { name: "plateau", score: |p| (p.elevation > 2000.0 && p.relief < 400.0).then(|| p.elevation / 4000.0) },
    Theme { name: "lake", score: |p| (p.legacy[lc::LAKE as usize] > 0.1).then(|| p.legacy[lc::LAKE as usize]) },
    Theme { name: "tundra", score: |p| (p.legacy[lc::TUNDRA as usize] > 0.4).then(|| p.legacy[lc::TUNDRA as usize]) },
];

const NF: usize = 18 + 7;

/// The feature vector of a place (selection distances).
fn features(p: &Place) -> [f64; NF] {
    let mut f = [0.0; NF];
    for (k, v) in p.legacy.iter().enumerate() {
        f[k] = 0.8 * v;
    }
    let b = |x: bool| if x { 0.4 } else { 0.0 };
    f[18] = 0.35 * (p.elevation / 1500.0).clamp(-1.0, 4.0);
    f[19] = 0.35 * (p.relief / 600.0).clamp(0.0, 3.0);
    f[20] = 0.4 * p.temp / 25.0;
    f[21] = 0.4 * p.moist;
    f[22] = b(p.coast);
    f[23] = b(p.river);
    f[24] = b(p.town);
    f
}

/// Distance between two places: features, biome (when known) and the globe.
fn distance(a: &Place, fa: &[f64; NF], b: &Place, fb: &[f64; NF]) -> f64 {
    let d2: f64 = fa.iter().zip(fb).map(|(x, y)| (x - y) * (x - y)).sum();
    let biome = match (a.biome, b.biome) {
        (Some(x), Some(y)) if x != y => 0.5,
        _ => 0.0,
    };
    d2.sqrt() + biome + 0.25 * (distance_km((a.lat, a.lon), (b.lat, b.lon)) / 3000.0).min(1.0)
}

/// Choose `count` diverse places among the candidates: themes first, then farthest-point
/// sampling. Deterministic (ties go to the lower candidate index).
pub fn select(cands: &[Place], count: usize) -> Vec<Place> {
    let feats: Vec<[f64; NF]> = cands.iter().map(features).collect();
    let mut chosen: Vec<usize> = Vec::new();
    let mut themes: Vec<&'static str> = Vec::new();
    // candidates far enough from every chosen place (all unchosen ones when none is)
    let pool = |chosen: &[usize]| -> Vec<usize> {
        let far: Vec<usize> = (0..cands.len())
            .filter(|i| {
                !chosen.contains(i) && chosen.iter().all(|&j| distance_km((cands[*i].lat, cands[*i].lon), (cands[j].lat, cands[j].lon)) >= MIN_SEPARATION_KM)
            })
            .collect();
        if far.is_empty() {
            (0..cands.len()).filter(|i| !chosen.contains(i)).collect()
        } else {
            far
        }
    };
    let spread =
        |i: usize, chosen: &[usize]| -> f64 { chosen.iter().map(|&j| distance(&cands[i], &feats[i], &cands[j], &feats[j])).fold(f64::INFINITY, f64::min) };
    let argmax = |v: &[usize], f: &dyn Fn(usize) -> f64| -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for &i in v {
            let s = f(i);
            if best.is_none_or(|(_, b)| s > b) {
                best = Some((i, s));
            }
        }
        best.map(|b| b.0)
    };
    for t in THEMES {
        if chosen.len() >= count.div_ceil(2) || chosen.len() >= count {
            break;
        }
        let v: Vec<usize> = pool(&chosen).into_iter().filter(|&i| (t.score)(&cands[i]).is_some()).collect();
        // the best example, nudged away from the places chosen so far
        let pick = argmax(&v, &|i| (t.score)(&cands[i]).unwrap_or(0.0) + if chosen.is_empty() { 0.0 } else { 0.3 * spread(i, &chosen).min(2.0) });
        if let Some(i) = pick {
            chosen.push(i);
            themes.push(t.name);
        }
    }
    while chosen.len() < count.min(cands.len()) {
        let v = pool(&chosen);
        let pick = if chosen.is_empty() {
            // the candidate nearest the mean: a typical place first
            let n = cands.len() as f64;
            let mean: Vec<f64> = (0..NF).map(|k| feats.iter().map(|f| f[k]).sum::<f64>() / n).collect();
            argmax(&v, &|i| -feats[i].iter().zip(&mean).map(|(a, b)| (a - b) * (a - b)).sum::<f64>())
        } else {
            argmax(&v, &|i| spread(i, &chosen))
        };
        let Some(i) = pick else { break };
        chosen.push(i);
        themes.push("spread");
    }
    chosen.iter().zip(themes).map(|(&i, t)| Place { theme: t.into(), ..cands[i].clone() }).collect()
}

// ============================================================================= CSV

const CSV_HEADER: &str = "index,lat,lon,theme,elevation_m,relief_m,slope,water,coast,river,town,temp_c,moist,biome,dominant";

pub fn write_csv(path: &Path, places: &[Place]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?);
    writeln!(f, "{CSV_HEADER}")?;
    for (k, p) in places.iter().enumerate() {
        writeln!(
            f,
            "{k},{:.6},{:.6},{},{:.1},{:.1},{:.4},{:.3},{},{},{},{:.1},{:.3},{},{}",
            p.lat,
            p.lon,
            p.theme,
            p.elevation,
            p.relief,
            p.slope,
            p.water,
            p.coast as u8,
            p.river as u8,
            p.town as u8,
            p.temp,
            p.moist,
            p.biome.map(|b| b.to_string()).unwrap_or_default(),
            p.dominant_str()
        )?;
    }
    f.flush()?;
    Ok(())
}

/// (lat, lon, theme) of the rows of a places CSV (columns `lat` and `lon` in degrees, `theme`
/// optional; other columns are ignored, `#` lines are comments).
pub fn read_csv(path: &Path) -> Result<Vec<(f64, f64, String)>> {
    let txt = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut lines = txt.lines().enumerate().filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with('#'));
    let Some((_, head)) = lines.next() else { bail!("{}: empty", path.display()) };
    let cols: Vec<&str> = head.split(',').map(|c| c.trim()).collect();
    let col = |n: &str| cols.iter().position(|c| *c == n);
    let (Some(ilat), Some(ilon)) = (col("lat"), col("lon")) else { bail!("{}: needs `lat` and `lon` columns", path.display()) };
    let itheme = col("theme");
    let mut v = Vec::new();
    for (ln, l) in lines {
        let f: Vec<&str> = l.split(',').map(|c| c.trim()).collect();
        let num = |i: usize| -> Result<f64> {
            f.get(i).and_then(|x| x.parse().ok()).with_context(|| format!("{}:{}: a number expected in column {}", path.display(), ln + 1, i + 1))
        };
        let (lat, lon) = (num(ilat)?, num(ilon)?);
        if !(lat.abs() <= 85.0 && lon.abs() <= 180.0) {
            bail!("{}:{}: lat {lat}, lon {lon} out of range (|lat| <= 85)", path.display(), ln + 1);
        }
        v.push((lat, lon, itheme.and_then(|i| f.get(i)).map(|s| s.to_string()).unwrap_or_default()));
        if v.len() > MAX_PLACES {
            bail!("{}: more than {MAX_PLACES} places", path.display());
        }
    }
    if v.is_empty() {
        bail!("{}: no places", path.display());
    }
    Ok(v)
}

// ============================================================================= rendering

fn file_stem(k: usize, p: &Place) -> String {
    format!("{k:02}_{:+07.3}_{:+08.3}", p.lat, p.lon)
}

/// Work of one view: tiles generated, seconds planning, generating and rendering.
#[derive(Clone, Copy, Debug, Default)]
struct ViewStats {
    tiles: usize,
    plan_s: f64,
    gen_s: f64,
    render_s: f64,
}

/// The surveyed places' stills (one renderer, one tile store, one place at a time).
struct Stills {
    scn: Scenario,
    gen: Arc<Generator>,
    store: Arc<tilestore::TileStore>,
    renderer: render::Renderer,
    spec: CameraSpec,
    sensor: render::sensor::SensorSettings,
    size: (u32, u32),
    date_unix: f64,
    local_time: f64,
}

impl Stills {
    fn new(scn: Scenario, gen: Arc<Generator>, tiles: &Path, plan: &Plan, hfov: f64) -> Result<Self> {
        let store = Arc::new(gen.open_store_rw(tiles).with_context(|| format!("opening the tile store {}", tiles.display()))?);
        let mut spec = CameraSpec::example();
        spec.path = "/survey".into();
        spec.intrinsics = CameraConfig::pinhole_hfov(plan.size.0, plan.size.1, hfov);
        (spec.depth, spec.flow, spec.landcover) = (None, None, None);
        let rgb = spec.rgb.clone().unwrap_or_default();
        let mut sensor = rgb.sensor.clone();
        // clean stills for comparisons: no noise, no motion
        sensor.motion_blur.enabled = false;
        sensor.noise.enabled = false;
        let model = spec.intrinsics.build()?;
        let cache = pipeline::tile_cache(&scn, store.clone(), Some(gen.clone()));
        let mut renderer = pipeline::renderer(&scn, model, spec.supersample(&scn.render), store.meta().ellipsoid(), cache);
        renderer.radiance_only = true;
        Ok(Stills { scn, gen, store, renderer, spec, sensor, size: plan.size, date_unix: plan.date_unix, local_time: plan.local_time })
    }

    /// Generate the tiles of a view: the renderer's own selection, one zoom level more per dry
    /// run, until nothing is missing.
    fn prepare(&self, spec: &CameraSpec, pose: &Pose, st: &mut ViewStats) -> Result<()> {
        let mut ps = self.scn.clone();
        ps.cameras = vec![spec.clone()];
        (ps.output.start, ps.output.end, ps.output.max_frames) = (0.0, None, None);
        // (the neighbours the renderer filters across are generated as it gathers them)
        ps.tiles.margin = 0;
        let poses = [*pose];
        let t = std::time::Instant::now();
        // dry runs of the renderer's selection with the elevation ranges of the stored tiles
        // (fewer tiles and quicker than a plan with estimated ranges)
        let mut todo: Vec<_> = pipeline::plan_missing(&ps, &poses, &self.store)?.into_iter().collect();
        st.plan_s += t.elapsed().as_secs_f64();
        for _ in 0..24 {
            let t = std::time::Instant::now();
            st.tiles += pipeline::generate(&self.gen, &self.store, &todo, false, &|_, _| {})?;
            st.gen_s += t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            todo = pipeline::plan_missing(&ps, &poses, &self.store)?.into_iter().collect();
            st.plan_s += t.elapsed().as_secs_f64();
            if todo.is_empty() {
                break;
            }
        }
        Ok(())
    }

    /// Render the views of a place.
    fn render(&self, p: &Place, views: &[ViewSpec]) -> Result<Vec<(RgbImage, ViewStats)>> {
        let (lat, lon) = (p.lat.to_radians(), p.lon.to_radians());
        let unix = self.date_unix + (self.local_time - p.lon / 15.0) * 3600.0;
        let sun = self.scn.render.lighting.sun_at_utc(unix, lat, lon);
        // looking across the sunlight (the sun behind, to the side): relief shows, no glare
        let heading = sun.azimuth + 150f64.to_radians();
        let ell = self.renderer.ell;
        let (w, h) = self.size;
        let mut out = Vec::new();
        for v in views {
            let mut spec = self.spec.clone();
            spec.extrinsics = Extrinsics { mount: Mount::Forward, pitch_deg: v.pitch, ..Extrinsics::default() };
            let alt = (p.elevation + v.agl).max(p.top + 0.4 * v.agl);
            let pose = Pose { t: 0.0, geo: Geodetic::new(lat, lon, alt), q_ned_body: geodesy::euler_zyx_to_quat(heading, 0.0, 0.0) };
            let mut st = ViewStats::default();
            self.prepare(&spec, &pose, &mut st)?;
            let cam = pose.camera(&spec.extrinsics, &ell);
            let t = std::time::Instant::now();
            let frame = self.renderer.try_render(&cam, &sun)?;
            st.render_s = t.elapsed().as_secs_f64();
            let mut sensor = Sensor::new(self.sensor.clone(), w as usize, h as usize);
            sensor.meter(&frame.radiance);
            let ex = sensor.exposure_for(0.0);
            let rgb = sensor.develop(&frame.radiance, &ex, 0);
            out.push((RgbImage::from_raw(w, h, rgb).context("developed image size")?, st));
        }
        Ok(out)
    }
}

// ============================================================================= the command

pub fn survey(a: SurveyArgs) -> Result<()> {
    let plan = check_args(&a)?;
    let s = crate::commands::setup(&a.common)?;
    std::fs::create_dir_all(&a.out).with_context(|| format!("creating {}", a.out.display()))?;
    let gen = Arc::new(pipeline::generator(&s)?);
    let t0 = std::time::Instant::now();
    let places: Vec<Place> = match &a.places {
        Some(f) => {
            let rows = read_csv(f)?;
            eprintln!("{} places from {}", rows.len(), f.display());
            rows.par_iter()
                .map(|(lat, lon, theme)| Place {
                    theme: if theme.is_empty() { "listed".into() } else { theme.clone() },
                    ..classify(gen.as_ref(), lat.to_radians(), lon.to_radians())
                })
                .collect()
        }
        None => {
            let c = candidates(gen.as_ref(), a.seed_places, a.samples);
            eprintln!("classified {} candidate points (seed {}) in {:.1}s", c.len(), a.seed_places, t0.elapsed().as_secs_f64());
            if c.len() < a.count {
                eprintln!("warning: only {} candidates on land (is the world all ocean?)", c.len());
            }
            select(&c, a.count)
        }
    };
    let csv = a.out.join("places.csv");
    write_csv(&csv, &places)?;
    eprintln!("wrote {} ({} places)", csv.display(), places.len());
    for (k, p) in places.iter().enumerate() {
        eprintln!("  {k:2} {:+8.3} {:+9.3} {:>9} {:5.0} m relief {:4.0} m  {}", p.lat, p.lon, p.theme, p.elevation, p.relief, p.dominant_str());
    }
    if a.no_render {
        return Ok(());
    }
    let tiles = a.tiles.clone().unwrap_or_else(|| a.out.join("tiles.h5"));
    let mut scn = s.clone();
    scn.tiles.lazy = true;
    // a place's views use ~500 tiles: keep a few places' worth in memory
    scn.tiles.cache_tiles = scn.tiles.cache_tiles.min(800);
    let stills = Stills::new(scn, gen.clone(), &tiles, &plan, a.hfov)?;
    eprintln!("rendering on the {} (tiles: {}, generated on the {})", backend_name(&stills), tiles.display(), gen.backend_name());
    let mut thumbs: Vec<Vec<RgbImage>> = Vec::with_capacity(places.len());
    for (k, p) in places.iter().enumerate() {
        let t = std::time::Instant::now();
        let imgs = stills.render(p, &plan.views).with_context(|| format!("place {k} ({:.4}, {:.4})", p.lat, p.lon))?;
        let mut row = Vec::new();
        let mut work = Vec::new();
        for ((img, st), v) in imgs.into_iter().zip(&plan.views) {
            let path = a.out.join(format!("{}_{}.png", file_stem(k, p), v.name));
            img.save(&path).with_context(|| format!("writing {}", path.display()))?;
            row.push(thumbnail(&img));
            work.push(format!("{} {} tiles (plan {:.1}s, generate {:.1}s, render {:.2}s)", v.name, st.tiles, st.plan_s, st.gen_s, st.render_s));
        }
        thumbs.push(row);
        eprintln!("[{}/{}] {:+.3} {:+.3} {} {:.1}s: {}", k + 1, places.len(), p.lat, p.lon, p.theme, t.elapsed().as_secs_f64(), work.join("; "));
    }
    stills.renderer.cache.flush_generated()?;
    stills.store.flush()?;
    let title = match &a.places {
        Some(f) => format!("WORLD SEED {}  PLACES {}  {}  {:.1}H LOCAL", s.world.seed, f.display(), a.date, a.local_time),
        None => format!("WORLD SEED {}  PLACES SEED {}  {}  {:.1}H LOCAL", s.world.seed, a.seed_places, a.date, a.local_time),
    };
    let sheet = contact_sheet(&title, &places, &plan.views, &thumbs);
    let path = a.out.join("sheet.jpg");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut f, 88).encode_image(&sheet)?;
    f.flush()?;
    eprintln!(
        "wrote {} ({}x{}) and {} stills in {:.1}s",
        path.display(),
        sheet.width(),
        sheet.height(),
        places.len() * plan.views.len(),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn backend_name(s: &Stills) -> &'static str {
    match s.renderer.settings.backend {
        render::raster::Backend::Gpu => "GPU",
        _ => "CPU",
    }
}

// ============================================================================= contact sheet

const THUMB_W: u32 = 320;
const LABEL_W: u32 = 380;
const GAP: u32 = 6;
const SCALE: u32 = 2;
const LINE_H: u32 = 9 * SCALE;
const CHAR_W: u32 = 6 * SCALE;

fn thumbnail(img: &RgbImage) -> RgbImage {
    let h = (THUMB_W as f64 * img.height() as f64 / img.width() as f64).round().max(1.0) as u32;
    image::imageops::resize(img, THUMB_W, h, image::imageops::FilterType::Triangle)
}

/// The label lines of a place.
fn label(k: usize, p: &Place) -> Vec<String> {
    let mut flags: Vec<&str> = Vec::new();
    for (on, n) in [(p.coast, "coast"), (p.river, "river"), (p.town, "town")] {
        if on {
            flags.push(n);
        }
    }
    let mut lines = vec![
        format!("{k:02}  {:+.3} {:+.3}", p.lat, p.lon),
        format!("{}  elev {:.0} m", p.theme, p.elevation),
        format!("relief {:.0} m  {:.0} c", p.relief, p.temp),
    ];
    for (c, f) in &p.dominant {
        lines.push(format!("{} {:.0}%", lc::name(*c).replace('_', " "), f * 100.0));
    }
    if !flags.is_empty() {
        lines.push(flags.join(" "));
    }
    lines
}

/// Rows of places: the label, then the views' thumbnails.
fn contact_sheet(title: &str, places: &[Place], views: &[ViewSpec], thumbs: &[Vec<RgbImage>]) -> RgbImage {
    let th = thumbs.iter().flatten().map(|t| t.height()).max().unwrap_or(180);
    let label_h = 7 * LINE_H + GAP;
    let row_h = th.max(label_h) + GAP;
    let head_h = 2 * LINE_H + 2 * GAP;
    let w = LABEL_W + views.len() as u32 * (THUMB_W + GAP) + GAP;
    let h = head_h + places.len() as u32 * row_h + GAP;
    let mut img = RgbImage::from_pixel(w, h, image::Rgb([28, 28, 30]));
    draw_text(&mut img, GAP, GAP, title, [255, 210, 120], w - 2 * GAP);
    for (j, v) in views.iter().enumerate() {
        let x = LABEL_W + j as u32 * (THUMB_W + GAP);
        draw_text(&mut img, x, GAP + LINE_H + GAP / 2, &format!("{} {:.0} m {:.0} deg", v.name, v.agl, v.pitch), [200, 200, 200], THUMB_W);
    }
    for (k, (p, row)) in places.iter().zip(thumbs).enumerate() {
        let y = head_h + k as u32 * row_h;
        for (i, l) in label(k, p).iter().enumerate() {
            let col = if i == 0 { [255, 255, 255] } else { [190, 190, 190] };
            draw_text(&mut img, GAP, y + i as u32 * LINE_H, l, col, LABEL_W - 2 * GAP);
        }
        for (j, t) in row.iter().enumerate() {
            image::imageops::replace(&mut img, t, (LABEL_W + j as u32 * (THUMB_W + GAP)) as i64, y as i64);
        }
    }
    img
}

/// Draw `text` (upper-cased; 5 × 7 pixel font at `SCALE`) at (x, y), clipped to `max_w`.
fn draw_text(img: &mut RgbImage, x: u32, y: u32, text: &str, rgb: [u8; 3], max_w: u32) {
    for (i, ch) in text.chars().enumerate() {
        let x0 = x + i as u32 * CHAR_W;
        if x0 + CHAR_W > x + max_w {
            break;
        }
        let rows = glyph(ch);
        for (r, row) in rows.iter().enumerate() {
            for (c, b) in row.bytes().enumerate() {
                if b != b'#' {
                    continue;
                }
                for dy in 0..SCALE {
                    for dx in 0..SCALE {
                        let (px, py) = (x0 + c as u32 * SCALE + dx, y + r as u32 * SCALE + dy);
                        if px < img.width() && py < img.height() {
                            img.put_pixel(px, py, image::Rgb(rgb));
                        }
                    }
                }
            }
        }
    }
}

/// A 5 × 7 glyph (`#` = ink); letters are drawn upper case, unknown characters as `?`.
fn glyph(c: char) -> [&'static str; 7] {
    let c = c.to_ascii_uppercase();
    FONT.iter().find(|(k, _)| *k == c).or_else(|| FONT.iter().find(|(k, _)| *k == '?')).map(|g| g.1).unwrap_or(["....."; 7])
}

const FONT: &[(char, [&str; 7])] = &[
    ('A', [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"]),
    ('B', ["####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."]),
    ('C', [".###.", "#...#", "#....", "#....", "#....", "#...#", ".###."]),
    ('D', ["####.", "#...#", "#...#", "#...#", "#...#", "#...#", "####."]),
    ('E', ["#####", "#....", "#....", "####.", "#....", "#....", "#####"]),
    ('F', ["#####", "#....", "#....", "####.", "#....", "#....", "#...."]),
    ('G', [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".####"]),
    ('H', ["#...#", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"]),
    ('I', [".###.", "..#..", "..#..", "..#..", "..#..", "..#..", ".###."]),
    ('J', ["..###", "...#.", "...#.", "...#.", "...#.", "#..#.", ".##.."]),
    ('K', ["#...#", "#..#.", "#.#..", "##...", "#.#..", "#..#.", "#...#"]),
    ('L', ["#....", "#....", "#....", "#....", "#....", "#....", "#####"]),
    ('M', ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"]),
    ('N', ["#...#", "#...#", "##..#", "#.#.#", "#..##", "#...#", "#...#"]),
    ('O', [".###.", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."]),
    ('P', ["####.", "#...#", "#...#", "####.", "#....", "#....", "#...."]),
    ('Q', [".###.", "#...#", "#...#", "#...#", "#.#.#", "#..#.", ".##.#"]),
    ('R', ["####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"]),
    ('S', [".####", "#....", "#....", ".###.", "....#", "....#", "####."]),
    ('T', ["#####", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."]),
    ('U', ["#...#", "#...#", "#...#", "#...#", "#...#", "#...#", ".###."]),
    ('V', ["#...#", "#...#", "#...#", "#...#", "#...#", ".#.#.", "..#.."]),
    ('W', ["#...#", "#...#", "#...#", "#.#.#", "#.#.#", "#.#.#", ".#.#."]),
    ('X', ["#...#", "#...#", ".#.#.", "..#..", ".#.#.", "#...#", "#...#"]),
    ('Y', ["#...#", "#...#", ".#.#.", "..#..", "..#..", "..#..", "..#.."]),
    ('Z', ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"]),
    ('0', [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."]),
    ('1', ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."]),
    ('2', [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"]),
    ('3', ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."]),
    ('4', ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."]),
    ('5', ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."]),
    ('6', ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."]),
    ('7', ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."]),
    ('8', [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."]),
    ('9', [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."]),
    (' ', [".....", ".....", ".....", ".....", ".....", ".....", "....."]),
    ('.', [".....", ".....", ".....", ".....", ".....", ".##..", ".##.."]),
    (',', [".....", ".....", ".....", ".....", ".##..", ".##..", ".#..."]),
    ('-', [".....", ".....", ".....", ".###.", ".....", ".....", "....."]),
    ('+', [".....", "..#..", "..#..", "#####", "..#..", "..#..", "....."]),
    (':', [".....", ".##..", ".##..", ".....", ".##..", ".##..", "....."]),
    ('%', ["##...", "##..#", "...#.", "..#..", ".#...", "#..##", "...##"]),
    ('/', [".....", "....#", "...#.", "..#..", ".#...", "#....", "....."]),
    ('_', [".....", ".....", ".....", ".....", ".....", ".....", "#####"]),
    ('(', ["...#.", "..#..", ".#...", ".#...", ".#...", "..#..", "...#."]),
    (')', [".#...", "..#..", "...#.", "...#.", "...#.", "..#..", ".#..."]),
    ('=', [".....", ".....", "#####", ".....", "#####", ".....", "....."]),
    ('?', [".###.", "#...#", "....#", "...#.", "..#..", ".....", "..#.."]),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// A small world: ocean west of 0° longitude and in narrow straits between 120° and 160°
    /// east, a mountain range at (30°, 60°), a town at (10°, 20°), a river along 25° north,
    /// snow north of 60°, desert around (20°, 100°), forest between 40° and 60°, farmland
    /// elsewhere.
    struct Mock;
    impl Sampler for Mock {
        fn probe(&self, lat: f64, lon: f64, _gsd: f64) -> Probe {
            let (la, lo) = (lat.to_degrees(), lon.to_degrees());
            let ocean = lo < 0.0 || (lo > 120.0 && lo < 160.0 && (lo * 20.0).sin() > 0.9);
            let d2 = |a: f64, b: f64| (la - a).powi(2) + (lo - b).powi(2);
            let mtn = 4000.0 * (-d2(30.0, 60.0) / 200.0).exp() * (1.0 + 0.3 * (la * 2000.0).sin());
            let class = if ocean {
                lc::OCEAN
            } else if d2(10.0, 20.0) < 1.0 {
                lc::URBAN
            } else if (la - 25.0).abs() < 0.004 {
                lc::RIVER
            } else if la > 60.0 {
                lc::SNOW
            } else if d2(20.0, 100.0) < 300.0 {
                lc::SAND
            } else if la > 40.0 {
                lc::FOREST
            } else if mtn > 1500.0 {
                lc::ROCK
            } else {
                lc::CROP
            };
            Probe {
                height: if ocean { 0.0 } else { 50.0 + mtn },
                class,
                ocean,
                river: false,
                temp: 25.0 - la.abs() / 3.0,
                moist: if class == lc::SAND { 0.1 } else { 0.6 },
                biome: None,
            }
        }
    }

    #[test]
    fn selection_is_deterministic_and_diverse() {
        let c = candidates(&Mock, 7, 400);
        assert_eq!(c.len(), 400);
        assert!(c.iter().all(|p| p.lat.abs() <= 80.5));
        let a = select(&c, 12);
        assert_eq!(a.len(), 12);
        assert_eq!(a, select(&candidates(&Mock, 7, 400), 12), "same seed, same places");
        let b = select(&candidates(&Mock, 8, 400), 12);
        assert_ne!(a, b, "another seed, other places");
        // the themes the world has are found (half of the places at most)
        let themes: Vec<&str> = a.iter().map(|p| p.theme.as_str()).collect();
        for t in ["coast", "mountains", "snow-ice", "desert"] {
            assert!(themes.contains(&t), "no {t} place: {themes:?}");
        }
        assert_eq!(themes.iter().filter(|t| **t != "spread").count(), 6, "{themes:?}");
        assert!(a.iter().any(|p| p.groups[Group::Agriculture as usize] > 0.4), "farmland");
        assert!(a.iter().any(|p| p.groups[Group::Forest as usize] > 0.6), "forest");
        // every candidate shows land
        assert!(c.iter().all(|p| p.water < 1.0));
        assert!(a.iter().any(|p| p.coast && p.water > 0.0 && p.water < 1.0));
        // spread over the globe
        for (i, p) in a.iter().enumerate() {
            for q in &a[..i] {
                assert!(distance_km((p.lat, p.lon), (q.lat, q.lon)) >= MIN_SEPARATION_KM, "{p:?} near {q:?}");
            }
        }
        // fewer candidates than places: all of them
        assert_eq!(select(&c[..5], 12).len(), 5);
    }

    #[test]
    fn classification() {
        let p = classify(&Mock, 30f64.to_radians(), 60f64.to_radians());
        assert!(p.relief > 100.0 && p.elevation > 2500.0 && p.slope > 0.0);
        let coast = classify(&Mock, 45f64.to_radians(), 0.0);
        assert!(coast.coast && coast.water > 0.2 && coast.water < 0.8);
        assert_eq!(coast.dominant.len(), 2);
        let town = classify(&Mock, 10f64.to_radians(), 20f64.to_radians());
        assert!(town.town && !town.coast && town.dominant[0] == (lc::URBAN, 1.0));
        assert_eq!(town.dominant_str(), "urban 100%");
        let river = classify(&Mock, 25f64.to_radians(), 30f64.to_radians());
        assert!(river.river && !classify(&Mock, 26f64.to_radians(), 30f64.to_radians()).river);
        // the open ocean has no candidates
        let c = candidate_points(&Mock, 0, 0..400);
        assert!(c.len() > 100 && c.iter().all(|(_, lon)| lon.to_degrees() > -0.05));
    }

    #[test]
    fn places_csv_round_trip() {
        let c = select(&candidates(&Mock, 3, 100), 5);
        let path = std::env::temp_dir().join(format!("survey-{}.csv", std::process::id()));
        write_csv(&path, &c).unwrap();
        let back = read_csv(&path).unwrap();
        assert_eq!(back.len(), 5);
        for (p, (lat, lon, theme)) in c.iter().zip(&back) {
            assert!((p.lat - lat).abs() < 1e-6 && (p.lon - lon).abs() < 1e-6);
            assert_eq!(&p.theme, theme);
        }
        std::fs::write(&path, "# a list\nname,lon,lat\nhome,32.5,39.9\n").unwrap();
        assert_eq!(read_csv(&path).unwrap(), vec![(39.9, 32.5, String::new())]);
        std::fs::write(&path, "lat,lon\n95,0\n").unwrap();
        assert!(read_csv(&path).is_err());
        std::fs::write(&path, "x,y\n1,2\n").unwrap();
        assert!(read_csv(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    fn args(extra: &[&str]) -> Result<Plan> {
        use clap::Parser;
        #[derive(Parser)]
        struct T {
            #[command(flatten)]
            a: SurveyArgs,
        }
        let t = T::try_parse_from(std::iter::once("survey").chain(extra.iter().copied()))?;
        check_args(&t.a)
    }

    #[test]
    fn arguments_are_checked() {
        let p = args(&[]).unwrap();
        assert_eq!(p.size, (960, 540));
        assert_eq!(p.views.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(), ["oblique", "nadir"]);
        assert_eq!(p.date_unix, 1_773_964_800.0, "2026-03-20 00:00 UTC");
        assert_eq!(args(&["--views", "oblique,nadir,high,low=300@-20"]).unwrap().views[3], ViewSpec { name: "low".into(), agl: 300.0, pitch: -20.0 });
        for bad in [
            &["--count", "0"][..],
            &["--count", "129"],
            &["--count", "50", "--samples", "10"],
            &["--samples", "100001"],
            &["--size", "0x0"],
            &["--size", "5000x100"],
            &["--size", "640"],
            &["--views", "sideways"],
            &["--views", "x=5@-20"],
            &["--views", "x=500@20"],
            &["--views", "nadir,nadir"],
            &["--views", "oblique,nadir,high,a=100@-10,b=100@-10"],
            &["--local-time", "25"],
            &["--date", "2026-13-01"],
            &["--hfov", "200"],
        ] {
            assert!(args(bad).is_err(), "{bad:?} accepted");
        }
        // a places list needs no search bounds
        assert!(args(&["--places", "x.csv", "--count", "0"]).is_ok());
    }

    #[test]
    fn font_covers_the_labels() {
        for (c, g) in FONT {
            assert!(g.iter().all(|r| r.len() == 5 && r.bytes().all(|b| b == b'#' || b == b'.')), "{c}");
        }
        let p = classify(&Mock, 30f64.to_radians(), 60f64.to_radians());
        let text: String = label(3, &p).concat() + "abcdefghijklmnopqrstuvwxyz0123456789 .,-+:%/_()=";
        for ch in text.chars() {
            assert!(FONT.iter().any(|(k, _)| *k == ch.to_ascii_uppercase()), "no glyph for {ch:?}");
        }
        let mut img = RgbImage::new(200, 40);
        draw_text(&mut img, 0, 0, "Ab1", [255, 255, 255], 200);
        assert!(img.pixels().any(|p| p.0 == [255, 255, 255]));
    }
}
