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
//! events:     { enabled: false, contrast_pos: 0.25, ... }   # written into output.h5
//! imu:        { enabled: true, rate_hz: 200, extrinsics: {...}, gyro: {...}, accel: {...} }
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
    /// Event camera simulation (`terrain events`, or `run` when enabled).
    pub events: crate::events::EventConfig,
    /// Synthetic IMU (written with the frames into `output.h5`).
    pub imu: crate::imu::ImuConfig,
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
            events: crate::events::EventConfig::default(),
            imu: crate::imu::ImuConfig::default(),
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
    /// Forward optical flow. Note: flow is exactly recomputable from depth + poses
    /// (scripts/check_gt.py does so); disable it to save ~half of the file size.
    pub flow: bool,
    pub landcover: bool,
    pub compression: Compression,
    /// Group / dataset names of the per-sensor view in `output.h5` (frames, IMU, events).
    pub layout: Layout,
}

/// Names of the per-sensor groups and datasets written into `output.h5`. The defaults follow
/// the M3ED layout (what camodocal's H5 readers expect), but nothing is assumed: every path and
/// name can be changed here.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Layout {
    /// Write the per-sensor view at all.
    pub enabled: bool,
    /// Frame camera group; images at `<frames_group>/<frames_data>`, timestamps (µs) at
    /// `frames_ts`, calibration under `<frames_group>/<calib_group>`.
    pub frames_group: String,
    pub frames_data: String,
    pub frames_ts: String,
    /// Images as grayscale u8 [N,H,W] (true) or RGB u8 [N,H,W,3].
    pub frames_gray: bool,
    /// IMU group and dataset names.
    pub imu_group: String,
    pub imu_ts: String,
    pub imu_accel: String,
    pub imu_gyro: String,
    /// Event camera group and dataset names (x, y, t, p, per-ms index).
    pub events_group: String,
    pub events_x: String,
    pub events_y: String,
    pub events_t: String,
    pub events_p: String,
    pub events_ms_map: String,
    /// Per-sensor calibration: group name (relative to the sensor group) and dataset names.
    pub calib_group: String,
    pub calib_intrinsics: String,
    pub calib_distortion: String,
    pub calib_resolution: String,
    /// 4x4 row-major sensor → rig transform (rig = body FRD frame).
    pub calib_transform: String,
}

impl Default for Layout {
    fn default() -> Self {
        Layout {
            enabled: true,
            frames_group: "/ovc/left".into(),
            frames_data: "data".into(),
            frames_ts: "/ovc/ts".into(),
            frames_gray: true,
            imu_group: "/ovc/imu".into(),
            imu_ts: "ts".into(),
            imu_accel: "accel".into(),
            imu_gyro: "omega".into(),
            events_group: "/prophesee/left".into(),
            events_x: "x".into(),
            events_y: "y".into(),
            events_t: "t".into(),
            events_p: "p".into(),
            events_ms_map: "ms_map_idx".into(),
            calib_group: "calib".into(),
            calib_intrinsics: "intrinsics".into(),
            calib_distortion: "distortion_coeffs".into(),
            calib_resolution: "resolution".into(),
            calib_transform: "T_to_prophesee_left".into(),
        }
    }
}

/// HDF5 path without the leading slash (our wrapper resolves paths from the root).
pub fn h5path(p: &str) -> &str {
    p.trim_start_matches('/')
}

/// HDF5 compression of the sequence file (shuffle + deflate, i.e. h5py `compression="gzip",
/// shuffle=True`), with optional lossy mantissa rounding of the float GT (depth, flow).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Compression {
    /// deflate level 0..9
    pub level: u8,
    /// Keep this many of the 23 f32 mantissa bits in depth / flow (None = lossless).
    /// 16 bits: max relative error 7.6e-6 (7.6 mm at 1 km, 1e-4 px on 10 px of flow), files
    /// ~35-40% smaller; 12 bits: 1.2e-4, ~55% smaller. Applied to the NPY outputs as well.
    pub float_keep_bits: Option<u8>,
}

impl Default for Compression {
    fn default() -> Self {
        Compression { level: 4, float_keep_bits: None }
    }
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
            compression: Compression::default(),
            layout: Layout::default(),
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
