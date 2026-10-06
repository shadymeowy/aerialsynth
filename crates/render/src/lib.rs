//! Onboard camera simulation: camera models, trajectories, LOD tile selection, CPU renderer,
//! ground truth (depth / optical flow) and dataset writers.

pub mod atmo;
pub mod cache;
pub mod camera;
pub mod dynamics;
pub mod lighting;
pub mod lod;
pub mod output;
pub mod pipeline;
pub mod scenario;
pub mod sensor;
pub mod raster;
pub mod trajectory;

pub use camera::{CameraConfig, CameraModel, Extrinsics, RigConfig};
pub use raster::{FrameOut, RenderSettings, Renderer, Shading};
pub use trajectory::{CamPose, Pose};
