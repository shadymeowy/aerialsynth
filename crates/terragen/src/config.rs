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
            albedo: AlbedoLook::default(),
            satellite: SatelliteLook::default(),
            tile_supersample: 2,
            tile_supersample_adaptive: true,
        }
    }
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
        Landuse {
            agriculture: 1.0,
            towns: 1.0,
            roads: 1.0,
            buildings_in_dsm: true,
            region_km: 7.0,
            town_cell_km: 6.0,
        }
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
