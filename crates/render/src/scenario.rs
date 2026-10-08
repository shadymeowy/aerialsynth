//! One scenario YAML drives every subcommand:
//!
//! ```yaml
//! world:      { seed: 7, ... }                  # terragen::Config — procedural terrain
//! tiles:      { file: out/world.h5, max_zoom: 18, ... }
//! trajectory: { file: out/traj.csv, synth: { kind: random, ... } }
//! render:     { supersample: 3, shading: relit, lighting: {...}, atmosphere: {...} }
//! cameras:                                      # any number of cameras
//!   - path: /cam0                               # HDF5 group of this camera
//!     intrinsics: { model: pinhole, width: 640, height: 480, intrinsics: [...], distortion: [...] }
//!     extrinsics: { mount: forward, pitch_deg: -30, translation: [0.4, 0, 0.1] }
//!     frame_rate: 10
//!     rgb: {}                                   # modalities: omitted = not produced
//!     depth: {}
//!     flow: {}
//!     landcover: {}
//!     events: {}
//! imu: { path: /imu, rate_hz: 200, ... }        # omitted = no IMU
//! output: { file: out/seq.h5, pose: { path: /pose, rate_hz: 200 }, ... }
//! ```
//! Everything goes into one HDF5 file (layout in `output.rs`). Only group paths are
//! configurable; dataset names inside a group are fixed. Relative file paths are relative to
//! the current working directory.

use crate::camera::{CameraConfig, Extrinsics};
use crate::dynamics::SynthConfig;
use crate::events::EventConfig;
use crate::imu::ImuConfig;
use crate::raster::RenderSettings;
use crate::sensor::SensorSettings;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Container defaults fill `world`, `tiles`, `trajectory`, `render` and `output`; `cameras`
/// and `imu` are empty when omitted (the defaults below only serve as the `terrain config`
/// template).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Scenario {
    pub world: terragen::Config,
    pub tiles: TilesConfig,
    pub trajectory: TrajectoryConfig,
    pub render: RenderSettings,
    #[serde(default)]
    pub cameras: Vec<CameraSpec>,
    #[serde(default)]
    pub imu: Option<ImuConfig>,
    pub output: OutputConfig,
}

impl Default for Scenario {
    fn default() -> Self {
        Scenario {
            world: terragen::Config::default(),
            tiles: TilesConfig::default(),
            trajectory: TrajectoryConfig::default(),
            render: RenderSettings::default(),
            cameras: vec![CameraSpec::example()],
            imu: Some(ImuConfig::default()),
            output: OutputConfig::default(),
        }
    }
}

/// A camera: intrinsics, mounting on the body, HDF5 group, and the modalities it produces.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraSpec {
    /// HDF5 group; calibration and every modality of this camera are written under it.
    pub path: String,
    /// Camera model in camodocal's schema (pinhole | pinhole_full | kannala_brandt | mei | scaramuzza).
    pub intrinsics: CameraConfig,
    /// Mounting on the body (FRD).
    #[serde(default)]
    pub extrinsics: Extrinsics,
    /// Rate (Hz) of the frame modalities (rgb, depth, flow, landcover).
    #[serde(default = "default_frame_rate")]
    pub frame_rate: f64,
    /// Delay (s) of this camera's first frame after the sequence start.
    #[serde(default)]
    pub time_offset: f64,
    #[serde(default)]
    pub rgb: Option<RgbModality>,
    #[serde(default)]
    pub depth: Option<DepthModality>,
    #[serde(default)]
    pub flow: Option<FlowModality>,
    #[serde(default)]
    pub landcover: Option<LandcoverModality>,
    #[serde(default)]
    pub events: Option<EventConfig>,
    /// Star ground truth (needs `rgb` and `lighting.stars`).
    #[serde(default)]
    pub stars: Option<StarsModality>,
}

fn default_frame_rate() -> f64 {
    10.0
}

impl CameraSpec {
    pub fn example() -> Self {
        CameraSpec {
            path: "/cam0".into(),
            intrinsics: CameraConfig::pinhole_hfov(640, 512, 70.0),
            extrinsics: Extrinsics::default(),
            frame_rate: 10.0,
            time_offset: 0.0,
            rgb: Some(RgbModality::default()),
            depth: Some(DepthModality::default()),
            flow: Some(FlowModality {}),
            landcover: None,
            events: None,
            stars: None,
        }
    }

    /// Does this camera produce frames (rgb, depth, flow or landcover)?
    pub fn has_frames(&self) -> bool {
        self.rgb.is_some() || self.depth.is_some() || self.flow.is_some() || self.landcover.is_some()
    }

    /// Seed offset of this camera (FNV-1a of the path), mixed into its noise seeds so that two
    /// cameras with the same settings do not get identical noise.
    pub fn seed_mix(&self) -> u64 {
        self.path.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
    }

    /// Supersampling of this camera's frames.
    pub fn supersample(&self, render: &RenderSettings) -> u32 {
        match &self.rgb {
            Some(r) => r.supersample.unwrap_or(render.supersample).max(1),
            None => 1, // geometry only: sampled at the pixel centres
        }
    }

    /// File-name friendly version of the path ("/ovc/left" → "ovc_left").
    pub fn slug(&self) -> String {
        let s: String = self.path.trim_matches('/').chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        if s.is_empty() { "camera".into() } else { s }
    }
}

/// HDF5 path without the leading slash (the h5 wrapper resolves paths from the root).
pub fn h5path(p: &str) -> &str {
    p.trim_start_matches('/')
}

/// Developed camera images: `rgb` u8 [N,H,W,3] (or [N,H,W] when `gray`) + `exposure`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RgbModality {
    /// Store luma (Rec. 601) instead of RGB.
    pub gray: bool,
    /// Supersampling override for this camera (default: render.supersample).
    pub supersample: Option<u32>,
    /// Sensor model: auto exposure, motion blur, optics, noise, tone curve.
    pub sensor: SensorSettings,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DepthKind {
    /// z along the optical axis (OpenCV convention; negative beyond 90° on wide-angle models)
    #[default]
    Z,
    /// distance along the pixel ray
    Range,
}

/// `depth` f32 [N,H,W] in metres, +inf = sky.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DepthModality {
    pub kind: DepthKind,
}

/// `flow` f32 [N,H,W,2]: forward flow to the next frame of the same camera (dx, dy) in px, and
/// `flow_valid` u8 [N,H,W] (target visible). The last frame has zero, invalid flow.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowModality {}

/// `landcover` u8 [N,H,W] class ids (255 = sky).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LandcoverModality {}

/// `stars/`: the catalogue stars in each frame (id, sub-pixel position, magnitude, irradiance,
/// visibility); see output.rs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StarsModality {
    /// Faintest V magnitude recorded (None = every rendered star; stars fainter than
    /// `render.stars.mag_limit` are never rendered).
    pub mag_limit: Option<f64>,
}

/// The largest `tiles.min_zoom`: the tile selection tests every tile of `min_zoom` (4^min_zoom)
/// for every frame.
pub const MAX_MIN_ZOOM: u8 = 6;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TilesConfig {
    /// HDF5 tile store.
    pub file: PathBuf,
    /// Zoom range of the tiles: planned, generated and rendered (the whole visible area is
    /// covered from `min_zoom`; detail is limited to `max_zoom`). The level of detail within it
    /// is `render.texel_px`. `min_zoom` is at most [`MAX_MIN_ZOOM`].
    pub min_zoom: u8,
    pub max_zoom: u8,
    /// Rings of neighbour tiles added around every planned tile (per zoom), so that consumers
    /// of the tile store (e.g. an odometry / SLAM system whose pose estimate is slightly off)
    /// find the neighbourhood of every viewed tile.
    pub margin: u32,
    /// Plan with every n-th frame of each camera.
    pub plan_every: usize,
    /// Generate tiles missing from the store while rendering (and write them back).
    pub lazy: bool,
    /// Tile cache size (tiles in memory, ~0.7 MB each) for rendering.
    pub cache_tiles: usize,
    /// Where tiles are generated: `auto` (the GPU when it has 64-bit shaders, else the CPU),
    /// `gpu` or `cpu`. Both build the same world (tiles agree to f32 precision).
    pub generator: terragen::Backend,
}

impl Default for TilesConfig {
    fn default() -> Self {
        TilesConfig {
            file: PathBuf::from("out/world.h5"),
            min_zoom: 2,
            max_zoom: 18,
            margin: 2,
            plan_every: 1,
            lazy: false,
            cache_tiles: 2000,
            generator: terragen::Backend::Auto,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrajectoryConfig {
    /// Trajectory CSV (input of plan/render, output of `traj`).
    pub file: PathBuf,
    /// Synthetic flight recorder settings (`terrain run --step traj`).
    pub synth: SynthConfig,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        TrajectoryConfig { file: PathBuf::from("out/traj.csv"), synth: SynthConfig::default() }
    }
}

/// Body ground truth, written once for all sensors (camera pose = body pose ∘ T_body_cam).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PoseOutput {
    pub path: String,
    pub rate_hz: f64,
}

impl Default for PoseOutput {
    fn default() -> Self {
        PoseOutput { path: "/pose".into(), rate_hz: 200.0 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputConfig {
    /// The sequence file (all cameras, IMU, body ground truth).
    pub file: PathBuf,
    /// Optional PNG / NPY export per camera (`<png_dir>/<camera>/rgb/000000.png`, ...).
    pub png_dir: Option<PathBuf>,
    /// Time window (s, relative to the trajectory start); end = None runs to the end.
    pub start: f64,
    pub end: Option<f64>,
    /// Max frames per camera (None = all).
    pub max_frames: Option<usize>,
    pub pose: PoseOutput,
    pub compression: Compression,
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            file: PathBuf::from("out/seq.h5"),
            png_dir: None,
            start: 0.0,
            end: None,
            max_frames: None,
            pose: PoseOutput::default(),
            compression: Compression::default(),
        }
    }
}

/// HDF5 compression (shuffle + deflate, i.e. h5py `compression="gzip", shuffle=True`), with
/// optional lossy mantissa rounding of the float ground truth (depth, flow).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Compression {
    /// deflate level 0..9
    pub level: u8,
    /// Keep this many of the 23 f32 mantissa bits in depth / flow (None = lossless).
    /// 16 bits: max relative error 7.6e-6 (7.6 mm at 1 km), ~35-40% smaller float datasets;
    /// 12 bits: 1.2e-4, ~55% smaller. Applied to the NPY export as well.
    pub float_keep_bits: Option<u8>,
}

impl Default for Compression {
    fn default() -> Self {
        Compression { level: 4, float_keep_bits: None }
    }
}

impl Scenario {
    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let scn: Scenario = if s.trim().is_empty() { Scenario::default() } else { serde_yaml::from_str(&s).with_context(|| format!("parsing {}", path.display()))? };
        scn.validate().with_context(|| format!("checking {}", path.display()))?;
        Ok(scn)
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

    /// Paths must be absolute and must not collide; models must build.
    pub fn validate(&self) -> Result<()> {
        let mut groups: Vec<(&str, String)> = vec![("output.pose", self.output.pose.path.clone())];
        if let Some(imu) = &self.imu {
            groups.push(("imu", imu.path.clone()));
        }
        if self.output.pose.rate_hz <= 0.0 {
            bail!("output.pose.rate_hz must be > 0");
        }
        let (z0, z1) = (self.tiles.min_zoom, self.tiles.max_zoom);
        if z0 > z1 || z1 > geodesy::MAX_ZOOM {
            bail!("tiles: min_zoom {z0} .. max_zoom {z1} is not a zoom range (0 <= min_zoom <= max_zoom <= {})", geodesy::MAX_ZOOM);
        }
        if z0 > MAX_MIN_ZOOM {
            bail!("tiles.min_zoom {z0} > {MAX_MIN_ZOOM}: the tile selection tests every tile of min_zoom (4^{z0}) for every frame");
        }
        if let Some(imu) = &self.imu {
            if imu.rate_hz <= 0.0 {
                bail!("imu.rate_hz must be > 0");
            }
        }
        for c in &self.cameras {
            groups.push(("camera", c.path.clone()));
            if c.has_frames() && c.frame_rate <= 0.0 {
                bail!("camera {}: frame_rate must be > 0", c.path);
            }
            if c.time_offset < 0.0 {
                bail!("camera {}: time_offset must be >= 0 (frames before the sequence start)", c.path);
            }
            // geometry GT is taken at the central sub-sample, which is the pixel centre only for
            // odd supersampling
            let geometry = c.depth.is_some() || c.flow.is_some() || c.landcover.is_some();
            if geometry && c.supersample(&self.render) % 2 == 0 {
                bail!("camera {}: supersample {} is even; depth / flow / landcover need an odd supersample (pixel-centre sample)", c.path, c.supersample(&self.render));
            }
            let ss = c.supersample(&self.render);
            if !(1..=9).contains(&ss) || c.events.as_ref().is_some_and(|e| !(1..=9).contains(&e.supersample)) {
                bail!("camera {}: supersample must be 1..=9", c.path);
            }
            if c.rgb.as_ref().is_some_and(|r| r.sensor.motion_blur.max_samples == 0) {
                bail!("camera {}: motion_blur.max_samples must be >= 1", c.path);
            }
            if let Some(e) = &c.events {
                if e.max_px_per_step <= 0.0 || e.min_rate_hz <= 0.0 || e.max_rate_hz < e.min_rate_hz {
                    bail!("camera {}: events need max_px_per_step > 0 and 0 < min_rate_hz <= max_rate_hz", c.path);
                }
            }
            let model = c.intrinsics.build().with_context(|| format!("camera {}", c.path))?;
            // `backend: gpu` must be able to render the camera (`auto` falls back to the CPU)
            #[cfg(feature = "gpu")]
            if self.render.backend == crate::raster::Backend::Gpu {
                let frames = std::iter::once(ss).filter(|_| c.has_frames());
                for s in frames.chain(c.events.as_ref().map(|e| e.supersample)) {
                    crate::gpu::supports(&*model, s).map_err(|e| anyhow::anyhow!("camera {}: render.backend gpu: {e} (backend auto renders it on the CPU)", c.path))?;
                }
            }
        }
        for (i, (what, p)) in groups.iter().enumerate() {
            if !p.starts_with('/') || p.len() < 2 {
                bail!("{what} path '{p}' must be an absolute HDF5 group such as /cam0");
            }
            for (_, q) in &groups[..i] {
                let (a, b) = (p.trim_end_matches('/'), q.trim_end_matches('/'));
                if a == b || a.starts_with(&format!("{b}/")) || b.starts_with(&format!("{a}/")) {
                    bail!("HDF5 groups '{p}' and '{q}' overlap");
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_modalities_are_off() {
        let s: Scenario = serde_yaml::from_str(
            "cameras:\n  - path: /ev\n    intrinsics: { model: pinhole, width: 64, height: 48, intrinsics: [50, 50, 32, 24] }\n    events: {}\n    flow: {}\n",
        )
        .unwrap();
        s.validate().unwrap();
        let c = &s.cameras[0];
        assert!(c.events.is_some() && c.flow.is_some());
        assert!(c.rgb.is_none() && c.depth.is_none() && c.landcover.is_none());
        assert!(s.imu.is_none());
        assert!(serde_yaml::from_str::<Scenario>("{}").unwrap().cameras.is_empty());
    }

    #[test]
    fn overlapping_groups_are_rejected() {
        let mut s = Scenario::default();
        s.cameras.push(CameraSpec { path: "/cam0/sub".into(), ..CameraSpec::example() });
        assert!(s.validate().is_err());
    }

    #[test]
    fn zoom_ranges_are_checked() {
        let with = |z0: u8, z1: u8| {
            let mut s = Scenario::default();
            (s.tiles.min_zoom, s.tiles.max_zoom) = (z0, z1);
            s.validate()
        };
        assert!(with(2, 18).is_ok() && with(MAX_MIN_ZOOM, MAX_MIN_ZOOM).is_ok());
        assert!(with(15, 10).is_err() && with(2, geodesy::MAX_ZOOM + 1).is_err() && with(MAX_MIN_ZOOM + 1, 18).is_err());
    }
}
