//! Generator configuration (YAML). Every field has a default, so an empty file is valid.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Master seed. Everything is a deterministic function of (seed, config, position).
    pub seed: u64,
    pub planet: Planet,
    /// Region that is guaranteed to be (mostly) land; trajectories usually live here.
    pub home: Option<Home>,
    pub continents: Continents,
    pub relief: Relief,
    pub hydro: Hydro,
    pub climate: Climate,
    pub vegetation: Vegetation,
    pub landuse: Landuse,
    /// Ecoregions: the lattice of regions with one biome and style each.
    pub ecoregions: Ecoregions,
    /// The biome registry: weight overrides; the resolved registry (written by the generator,
    /// stored with the tiles).
    pub biomes: Biomes,
    /// Colour of the surface (the `albedo` layer, and so every rendering of it).
    pub albedo: AlbedoLook,
    /// Lighting of the baked `rgb` layer of the tiles (a satellite-style image). Camera images
    /// are lit by the scenario's `render.lighting` instead.
    pub satellite: SatelliteLook,
    /// Supersampling per axis of the tile pixels (1 = one sample per pixel).
    pub tile_supersample: u32,
    /// With tile_supersample 2: evaluate the two diagonal samples first and the other two only
    /// where those differ (class, colour, height, light); flat areas cost half.
    pub tile_supersample_adaptive: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            seed: 1,
            planet: Planet::default(),
            home: Some(Home::default()),
            continents: Continents::default(),
            relief: Relief::default(),
            hydro: Hydro::default(),
            climate: Climate::default(),
            vegetation: Vegetation::default(),
            landuse: Landuse::default(),
            ecoregions: Ecoregions::default(),
            biomes: Biomes::default(),
            albedo: AlbedoLook::default(),
            satellite: SatelliteLook::default(),
            tile_supersample: 2,
            tile_supersample_adaptive: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Ecoregions {
    /// Lattice cell size (km): ecoregions are ~this wide.
    pub cell_km: f64,
    /// Width (km) of the mosaic of both sides along their borders.
    pub ecotone_km: f64,
}
impl Default for Ecoregions {
    fn default() -> Self {
        Ecoregions { cell_km: 100.0, ecotone_km: 5.0 }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Biomes {
    /// Per biome id: a pick weight replacing the registry's (0: never picked).
    pub overrides: std::collections::BTreeMap<String, BiomeOverride>,
    /// The resolved registry this world was generated with (set by the generator; a store's
    /// world records it, so a changed registry is a different world).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved: Option<serde_yaml::Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BiomeOverride {
    pub weight: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Planet {
    /// Semi-major axis (m).
    pub a: f64,
    /// Inverse flattening (0 = sphere).
    pub inv_f: f64,
}
impl Default for Planet {
    fn default() -> Self {
        Planet { a: 6378137.0, inv_f: 298.257223563 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Home {
    pub lat: f64,
    pub lon: f64,
    pub radius_km: f64,
    /// How strongly land is enforced (0..1).
    pub strength: f64,
}
impl Default for Home {
    fn default() -> Self {
        Home { lat: 39.9, lon: 32.8, radius_km: 600.0, strength: 0.6 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Continents {
    pub wavelength_km: f64,
    /// Threshold on the continent field; higher = less land.
    pub threshold: f64,
    pub warp: f64,
}
impl Default for Continents {
    fn default() -> Self {
        Continents { wavelength_km: 4000.0, threshold: 0.02, warp: 0.35 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Relief {
    /// Wavelength of mountain belts (km).
    pub belt_wavelength_km: f64,
    /// Typical maximum mountain height above the surrounding base (m).
    pub mountain_height_m: f64,
    /// Typical hill amplitude (m).
    pub hill_height_m: f64,
    /// Max amplitude of sub-100m micro relief (m).
    pub micro_height_m: f64,
    /// Fraction of arid uplands that are terraced into mesas.
    pub mesas: f64,
    /// Sand dunes amplitude in sand seas (m).
    pub dune_height_m: f64,
    /// Strength of the erosion-gully filter on mountain / hill slopes (0 disables).
    pub erosion: f64,
    /// Wavelength of the coarsest gully octave (m).
    pub gully_wavelength_m: f64,
}
impl Default for Relief {
    fn default() -> Self {
        Relief {
            belt_wavelength_km: 700.0,
            mountain_height_m: 3200.0,
            hill_height_m: 260.0,
            micro_height_m: 3.0,
            mesas: 0.5,
            dune_height_m: 35.0,
            erosion: 1.0,
            gully_wavelength_m: 1400.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Hydro {
    pub rivers: bool,
    /// Drainage network levels (coarse → fine): major rivers, tributaries, streams.
    pub levels: Vec<RiverLevel>,
    /// Probability of a lake per lake cell, scaled by moisture/flatness.
    pub lake_density: f64,
    pub lake_cell_km: f64,
}
impl Default for Hydro {
    fn default() -> Self {
        Hydro {
            rivers: true,
            levels: vec![
                RiverLevel { cell_km: 30.0, width_m: [60.0, 260.0], valley_m: 2500.0, wet_moisture: 0.08, meander: 0.35, max_depth_m: 400.0 },
                RiverLevel { cell_km: 8.0, width_m: [12.0, 45.0], valley_m: 700.0, wet_moisture: 0.25, meander: 0.4, max_depth_m: 35.0 },
                RiverLevel { cell_km: 2.2, width_m: [2.5, 8.0], valley_m: 140.0, wet_moisture: 0.5, meander: 0.45, max_depth_m: 6.0 },
            ],
            lake_density: 0.55,
            lake_cell_km: 22.0,
        }
    }
}

/// One level of the drainage network.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RiverLevel {
    /// Lattice cell size: typical distance between drainage nodes (km).
    pub cell_km: f64,
    /// Channel width range (m).
    pub width_m: [f64; 2],
    /// Valley half width (m).
    pub valley_m: f64,
    /// Moisture above which the channel carries water (dry bed otherwise).
    pub wet_moisture: f64,
    /// Meander wavelength as a fraction of the cell size.
    pub meander: f64,
    /// Maximum depth the channel/valley is carved into the local terrain (m).
    pub max_depth_m: f64,
}
impl Default for RiverLevel {
    fn default() -> Self {
        RiverLevel { cell_km: 8.0, width_m: [12.0, 45.0], valley_m: 700.0, wet_moisture: 0.25, meander: 0.4, max_depth_m: 35.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Climate {
    /// Mean annual temperature at the equator at sea level (°C).
    pub equator_temp_c: f64,
    /// Temperature drop from equator to pole (°C).
    pub pole_drop_c: f64,
    /// Lapse rate (°C per km).
    pub lapse_rate_c_per_km: f64,
    /// Added to moisture (−1..1).
    pub moisture_bias: f64,
}
impl Default for Climate {
    fn default() -> Self {
        Climate { equator_temp_c: 28.0, pole_drop_c: 46.0, lapse_rate_c_per_km: 6.0, moisture_bias: 0.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Vegetation {
    /// Multiplier on tree cover.
    pub tree_density: f64,
    /// Put tree canopies into the elevation layer (DSM).
    pub trees_in_dsm: bool,
}
impl Default for Vegetation {
    fn default() -> Self {
        Vegetation { tree_density: 1.0, trees_in_dsm: true }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Landuse {
    /// Agricultural intensity multiplier (0 disables fields).
    pub agriculture: f64,
    /// Settlement density multiplier (0 disables towns).
    pub towns: f64,
    /// Road density multiplier (0 disables roads).
    pub roads: f64,
    pub buildings_in_dsm: bool,
    /// Size of land-use regions (field systems), km.
    pub region_km: f64,
    /// Spacing of potential town sites, km.
    pub town_cell_km: f64,
}
impl Default for Landuse {
    fn default() -> Self {
        Landuse { agriculture: 1.0, towns: 1.0, roads: 1.0, buildings_in_dsm: true, region_km: 7.0, town_cell_km: 6.0 }
    }
}

/// Lighting of the baked `rgb` layer (a satellite-style image of each tile).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SatelliteLook {
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
    pub ambient: f64,
    pub direct: f64,
    pub exposure: f64,
    /// Atmospheric haze mixed into the image (0..1).
    pub haze: f64,
    /// Cast shadows of trees and buildings.
    pub shadows: bool,
}
impl Default for SatelliteLook {
    fn default() -> Self {
        SatelliteLook { sun_azimuth_deg: 145.0, sun_elevation_deg: 52.0, ambient: 0.30, direct: 0.95, exposure: 1.0, haze: 0.04, shadows: true }
    }
}

/// Global colour knobs applied to the generated surface albedo (and therefore to both the
/// `albedo` and `rgb` layers).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AlbedoLook {
    /// 1 = as designed, > 1 more vivid.
    pub saturation: f64,
    pub brightness: f64,
}
impl Default for AlbedoLook {
    fn default() -> Self {
        AlbedoLook { saturation: 1.0, brightness: 1.0 }
    }
}

impl Config {
    pub fn from_yaml_str(s: &str) -> anyhow::Result<Self> {
        if s.trim().is_empty() {
            return Ok(Config::default());
        }
        Ok(serde_yaml::from_str(s)?)
    }
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_yaml_str(&s)
    }
    pub fn to_yaml(&self) -> String {
        serde_yaml::to_string(self).unwrap_or_default()
    }
}

/// Largest `tile_supersample`: the cost grows with its square (4: 16 samples per pixel).
pub const MAX_TILE_SUPERSAMPLE: u32 = 4;

impl Config {
    /// Check that every value is usable: lengths and wavelengths finite and > 0, amplitudes and
    /// densities finite and >= 0, the planet a real ellipsoid. Errors name the key, e.g.
    /// `world.hydro.levels[0].cell_km must be > 0`.
    pub fn validate(&self) -> anyhow::Result<()> {
        let mut errs: Vec<String> = Vec::new();
        let mut check = |key: &str, v: f64, ok: bool, need: &str| {
            if !v.is_finite() || !ok {
                errs.push(format!("world.{key} must be {need} (is {v})"));
            }
        };
        // lengths, wavelengths, cell sizes: > 0
        let mut pos = |key: &str, v: f64| check(key, v, v > 0.0, "finite and > 0");
        let p = &self.planet;
        pos("planet.a", p.a);
        pos("continents.wavelength_km", self.continents.wavelength_km);
        pos("relief.belt_wavelength_km", self.relief.belt_wavelength_km);
        pos("relief.gully_wavelength_m", self.relief.gully_wavelength_m);
        for (i, l) in self.hydro.levels.iter().enumerate() {
            pos(&format!("hydro.levels[{i}].cell_km"), l.cell_km);
        }
        pos("hydro.lake_cell_km", self.hydro.lake_cell_km);
        pos("landuse.region_km", self.landuse.region_km);
        pos("landuse.town_cell_km", self.landuse.town_cell_km);
        pos("ecoregions.cell_km", self.ecoregions.cell_km);
        if let Some(h) = &self.home {
            pos("home.radius_km", h.radius_km);
        }
        // amplitudes, densities, multipliers: >= 0
        let mut nonneg = |key: &str, v: f64| check(key, v, v >= 0.0, "finite and >= 0");
        let r = &self.relief;
        nonneg("continents.warp", self.continents.warp);
        nonneg("relief.mountain_height_m", r.mountain_height_m);
        nonneg("relief.hill_height_m", r.hill_height_m);
        nonneg("relief.micro_height_m", r.micro_height_m);
        nonneg("relief.mesas", r.mesas);
        nonneg("relief.dune_height_m", r.dune_height_m);
        nonneg("relief.erosion", r.erosion);
        for (i, l) in self.hydro.levels.iter().enumerate() {
            nonneg(&format!("hydro.levels[{i}].width_m[0]"), l.width_m[0]);
            nonneg(&format!("hydro.levels[{i}].width_m[1]"), l.width_m[1]);
            nonneg(&format!("hydro.levels[{i}].valley_m"), l.valley_m);
            nonneg(&format!("hydro.levels[{i}].meander"), l.meander);
            nonneg(&format!("hydro.levels[{i}].max_depth_m"), l.max_depth_m);
        }
        nonneg("hydro.lake_density", self.hydro.lake_density);
        nonneg("vegetation.tree_density", self.vegetation.tree_density);
        nonneg("landuse.agriculture", self.landuse.agriculture);
        nonneg("landuse.towns", self.landuse.towns);
        nonneg("landuse.roads", self.landuse.roads);
        nonneg("ecoregions.ecotone_km", self.ecoregions.ecotone_km);
        for (id, o) in &self.biomes.overrides {
            if let Some(w) = o.weight {
                nonneg(&format!("biomes.overrides.{id}.weight"), w);
            }
        }
        nonneg("albedo.saturation", self.albedo.saturation);
        nonneg("albedo.brightness", self.albedo.brightness);
        let s = &self.satellite;
        nonneg("satellite.ambient", s.ambient);
        nonneg("satellite.direct", s.direct);
        nonneg("satellite.exposure", s.exposure);
        // ranges and plain finite values
        let mut within = |key: &str, v: f64, lo: f64, hi: f64| check(key, v, (lo..=hi).contains(&v), &format!("in [{lo}, {hi}]"));
        if let Some(h) = &self.home {
            within("home.lat", h.lat, -90.0, 90.0);
            within("home.strength", h.strength, 0.0, 1.0);
        }
        within("satellite.sun_elevation_deg", s.sun_elevation_deg, -90.0, 90.0);
        within("satellite.haze", s.haze, 0.0, 1.0);
        let mut finite = |key: &str, v: f64| check(key, v, true, "finite");
        finite("continents.threshold", self.continents.threshold);
        if let Some(h) = &self.home {
            finite("home.lon", h.lon);
        }
        for (i, l) in self.hydro.levels.iter().enumerate() {
            finite(&format!("hydro.levels[{i}].wet_moisture"), l.wet_moisture);
        }
        let c = &self.climate;
        finite("climate.equator_temp_c", c.equator_temp_c);
        finite("climate.pole_drop_c", c.pole_drop_c);
        finite("climate.lapse_rate_c_per_km", c.lapse_rate_c_per_km);
        finite("climate.moisture_bias", c.moisture_bias);
        finite("satellite.sun_azimuth_deg", s.sun_azimuth_deg);
        // inverse flattening: 0 or inf (a sphere) or > 1 (b = a (1 - 1/inv_f) > 0)
        if !(p.inv_f == 0.0 || p.inv_f == f64::INFINITY || (p.inv_f.is_finite() && p.inv_f > 1.0)) {
            errs.push(format!("world.planet.inv_f must be 0 (a sphere) or > 1 (is {})", p.inv_f));
        }
        for (i, l) in self.hydro.levels.iter().enumerate() {
            if l.width_m[1] < l.width_m[0] {
                errs.push(format!("world.hydro.levels[{i}].width_m must be [min, max] with min <= max (is {:?})", l.width_m));
            }
        }
        if !(1..=MAX_TILE_SUPERSAMPLE).contains(&self.tile_supersample) {
            errs.push(format!("world.tile_supersample must be 1..={MAX_TILE_SUPERSAMPLE} (is {})", self.tile_supersample));
        }
        if errs.is_empty() {
            if let Err(e) = crate::registry::Registry::for_config(self) {
                errs.push(format!("world.biomes: {e:#}"));
            }
        }
        match errs.len() {
            0 => Ok(()),
            1 => anyhow::bail!("{}", errs[0]),
            _ => anyhow::bail!("{} (and {} more: {})", errs[0], errs.len() - 1, errs[1..].join("; ")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(f: impl FnOnce(&mut Config)) -> String {
        let mut c = Config::default();
        f(&mut c);
        match c.validate() {
            Ok(()) => panic!("accepted"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn defaults_are_valid() {
        Config::default().validate().unwrap();
        Config { home: None, tile_supersample: 1, ..Config::default() }.validate().unwrap();
        let mut sphere = Config::default();
        sphere.planet.inv_f = 0.0;
        sphere.validate().unwrap();
        sphere.planet.inv_f = f64::INFINITY;
        sphere.validate().unwrap();
    }

    #[test]
    fn bad_values_name_their_key() {
        assert!(err(|c| c.planet.a = 0.0).starts_with("world.planet.a must be finite and > 0"));
        assert!(err(|c| c.planet.a = f64::NAN).starts_with("world.planet.a"));
        assert!(err(|c| c.planet.inv_f = 0.5).starts_with("world.planet.inv_f"));
        assert!(err(|c| c.planet.inv_f = -300.0).starts_with("world.planet.inv_f"));
        assert!(err(|c| c.tile_supersample = 64).starts_with("world.tile_supersample must be 1..=4"));
        assert!(err(|c| c.tile_supersample = 0).starts_with("world.tile_supersample"));
        assert!(err(|c| c.hydro.levels[0].cell_km = 0.0).starts_with("world.hydro.levels[0].cell_km must be finite and > 0"));
        assert!(err(|c| c.hydro.levels[2].cell_km = f64::INFINITY).starts_with("world.hydro.levels[2].cell_km"));
        assert!(err(|c| c.continents.wavelength_km = 0.0).starts_with("world.continents.wavelength_km"));
        assert!(err(|c| c.landuse.town_cell_km = 0.0).starts_with("world.landuse.town_cell_km"));
        assert!(err(|c| c.hydro.lake_cell_km = 0.0).starts_with("world.hydro.lake_cell_km"));
        assert!(err(|c| c.relief.mountain_height_m = -1.0).starts_with("world.relief.mountain_height_m must be finite and >= 0"));
        assert!(err(|c| c.satellite.haze = 2.0).starts_with("world.satellite.haze"));
        let two = err(|c| {
            c.planet.a = 0.0;
            c.landuse.region_km = -1.0;
        });
        assert!(two.contains("and 1 more") && two.contains("world.landuse.region_km"), "{two}");
    }
}
