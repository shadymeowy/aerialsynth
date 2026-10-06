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
    pub look: SatelliteLook,
    /// Supersampling per axis for colour (1 = one sample per pixel).
    pub supersample: u32,
    /// Finest zoom level the generator is tuned for (features below its GSD are not modelled).
    pub max_zoom: u8,
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
            look: SatelliteLook::default(),
            supersample: 2,
            max_zoom: 19,
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
    /// Wavelength of mountain belts.
    pub belt_wavelength_km: f64,
    /// Typical maximum mountain height above the surrounding base (m).
    pub mountain_height: f64,
    /// Typical hill amplitude (m).
    pub hill_height: f64,
    /// Max amplitude of sub-100m micro relief (m).
    pub micro_height: f64,
    /// Fraction of arid uplands that are terraced into mesas.
    pub mesas: f64,
    /// Sand dunes amplitude in sand seas (m).
    pub dune_height: f64,
    /// Strength of the erosion-gully filter on mountain / hill slopes (0 disables).
    pub erosion: f64,
    /// Wavelength of the coarsest gully octave (m).
    pub gully_wavelength: f64,
}
impl Default for Relief {
    fn default() -> Self {
        Relief {
            belt_wavelength_km: 700.0,
            mountain_height: 3200.0,
            hill_height: 260.0,
            micro_height: 3.0,
            mesas: 0.5,
            dune_height: 25.0,
            erosion: 1.0,
            gully_wavelength: 1400.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Hydro {
    pub rivers: bool,
    /// Wavelength of the major river network (km).
    pub major_river_wavelength_km: f64,
    pub minor_river_wavelength_km: f64,
    /// Probability of a lake per lake cell, scaled by moisture/flatness.
    pub lake_density: f64,
    pub lake_cell_km: f64,
}
impl Default for Hydro {
    fn default() -> Self {
        Hydro {
            rivers: true,
            major_river_wavelength_km: 160.0,
            minor_river_wavelength_km: 28.0,
            lake_density: 0.55,
            lake_cell_km: 22.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Climate {
    /// Mean annual temperature at the equator at sea level (°C).
    pub equator_temp: f64,
    /// Temperature drop from equator to pole (°C).
    pub pole_drop: f64,
    /// Lapse rate (°C per km).
    pub lapse_rate: f64,
    /// Added to moisture (−1..1).
    pub moisture_bias: f64,
}
impl Default for Climate {
    fn default() -> Self {
        Climate { equator_temp: 28.0, pole_drop: 46.0, lapse_rate: 6.0, moisture_bias: 0.0 }
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

/// How the baked "satellite" rgb layer is lit.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SatelliteLook {
    pub sun_azimuth_deg: f64,
    pub sun_elevation_deg: f64,
    pub ambient: f64,
    pub direct: f64,
    pub exposure: f64,
    /// Atmospheric haze mixed into the satellite image (0..1).
    pub haze: f64,
    /// Cast shadows of trees/buildings.
    pub shadows: bool,
}
impl Default for SatelliteLook {
    fn default() -> Self {
        SatelliteLook {
            sun_azimuth_deg: 145.0,
            sun_elevation_deg: 52.0,
            ambient: 0.30,
            direct: 0.95,
            exposure: 1.0,
            haze: 0.04,
            shadows: true,
        }
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
