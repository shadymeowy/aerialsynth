//! Onboard sensor simulation over the generated planet: the stage that turns a scenario
//! ([`scenario`]) into a dataset.
//!
//! * camera models and rig ([`camera`]), trajectories and synthetic flights ([`trajectory`],
//!   [`dynamics`]);
//! * LOD tile selection ([`lod`]) over a tile cache with lazy generation ([`cache`]);
//! * the renderer ([`raster`]: CPU reference backend; `gpu`: headless wgpu backend, feature
//!   `gpu`) with sky, sun / moon lighting and stars ([`atmo`], [`lighting`], [`stars`]);
//! * the camera sensor ([`sensor`]), ground truth (depth, optical flow, land cover), event
//!   camera ([`events`]) and IMU ([`imu`]);
//! * the HDF5 sequence writer ([`output`]) and the high-level steps the CLI runs ([`pipeline`]).

pub mod atmo;
pub mod cache;
pub mod camera;
pub mod dynamics;
pub mod events;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod imu;
pub mod lighting;
pub mod lod;
pub mod output;
pub mod pipeline;
pub mod raster;
pub mod scenario;
pub mod sensor;
pub mod stars;
pub mod trajectory;

pub use camera::{CameraConfig, CameraModel, Extrinsics};
pub use raster::{FrameOut, RenderSettings, Renderer, Shading};
pub use trajectory::{CamPose, Pose};
