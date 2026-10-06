//! One scenario YAML drives every subcommand. All sections are optional (defaults apply).
//!
//! ```yaml
//! world:      { seed: 7, ... }          # terragen::Config — procedural terrain
//! tiles:      { file: out/world.h5, max_zoom: 18, ... }
//! camera:     { model: pinhole_radtan, width: 640, height: 512, fx: ..., ... }
//! extrinsics: { mount: nadir, pitch_deg: 0, ... }
//! trajectory: { file: out/traj.csv, synth: { kind: random, ... } }
//! render:     { supersample: 3, shading: relit, lighting: {...}, ... }
//! sensor:     { exposure: {...}, motion_blur: {...}, noise: {...}, ... }
//! output:     { dir: out/seq, frame_rate: 10, png: true, h5: out/seq.h5, ... }
//! ```
//! Relative paths are relative to the current working directory.

use crate::camera::{CameraConfig, Extrinsics};
use crate::raster::RenderSettings;
use crate::sensor::SensorSettings;
use crate::dynamics::SynthConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scenario {
    pub world: terragen::Config,
    pub tiles: TilesConfig,
    pub camera: CameraConfig,
    pub extrinsics: Extrinsics,
    pub trajectory: TrajectoryConfig,
    pub render: RenderSettings,
    pub sensor: SensorSettings,
    pub output: OutputConfig,
}

impl Default for Scenario {
    fn default() -> Self {
        Scenario {
            world: terragen::Config::default(),
            tiles: TilesConfig::default(),
            camera: CameraConfig::pinhole_hfov(640, 512, 70.0),
            extrinsics: Extrinsics::default(),
            trajectory: TrajectoryConfig::default(),
            render: RenderSettings::default(),
            sensor: SensorSettings::default(),
            output: OutputConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TilesConfig {
    /// HDF5 tile store.
    pub file: PathBuf,
    /// Coarsest zoom planned/rendered (the whole visible area is covered from here).
    pub min_zoom: u8,
    /// Finest zoom generated.
    pub max_zoom: u8,
    /// Planning refines until a texel covers at most this many pixels (smaller = finer, safer).
    pub plan_texel_px: f64,
    /// Neighbour tiles added around every planned tile (per zoom).
    pub margin: u32,
    /// Plan with every n-th rendered frame.
    pub plan_every: usize,
    /// Generate tiles missing from the store while rendering (and write them back).
    pub lazy: bool,
    /// Tile cache size (tiles in memory, ~0.7 MB each) for rendering.
    pub cache_tiles: usize,
}

impl Default for TilesConfig {
    fn default() -> Self {
        TilesConfig {
            file: PathBuf::from("out/world.h5"),
            min_zoom: 2,
            max_zoom: 18,
            plan_texel_px: 0.8,
            margin: 1,
            plan_every: 1,
            lazy: false,
            cache_tiles: 2000,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrajectoryConfig {
    /// Trajectory CSV (input of plan/render, output of `traj`).
    pub file: PathBuf,
    /// Synthetic flight recorder settings (`terrain traj`).
    pub synth: SynthConfig,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        TrajectoryConfig { file: PathBuf::from("out/traj.csv"), synth: SynthConfig::default() }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    /// Directory for PNG/NPY outputs.
    pub dir: PathBuf,
    /// Camera frame rate (Hz); frames are rendered at t0 + k / frame_rate.
    pub frame_rate: f64,
    /// Time window (s, relative to the trajectory start); end = None renders to the end.
    pub start: f64,
    pub end: Option<f64>,
    /// Max number of frames (None = all).
    pub max_frames: Option<usize>,
    pub png: bool,
    /// HDF5 dataset with images + GT (None = not written).
    pub h5: Option<PathBuf>,
    pub depth: bool,
    pub flow: bool,
    pub landcover: bool,
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            dir: PathBuf::from("out/seq"),
            frame_rate: 10.0,
            start: 0.0,
            end: None,
            max_frames: None,
            png: true,
            h5: Some(PathBuf::from("out/seq.h5")),
            depth: true,
            flow: true,
            landcover: true,
        }
    }
}

impl Scenario {
    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        if s.trim().is_empty() {
            return Ok(Scenario::default());
        }
        serde_yaml::from_str(&s).with_context(|| format!("parsing {}", path.display()))
    }
    pub fn load_or_default(path: Option<&Path>) -> Result<Self> {
        match path {
            Some(p) => Self::load(p),
            None => Ok(Scenario::default()),
        }
    }
    pub fn to_yaml(&self) -> String {
        serde_yaml::to_string(self).unwrap_or_default()
    }
}
