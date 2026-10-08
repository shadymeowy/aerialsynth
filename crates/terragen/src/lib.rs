//! Procedural, deterministic, lazily evaluable terrain for XYZ tiles.
//!
//! `Generator::tile(TileId)` is a pure function of (config, tile id): any subset of tiles at any
//! zoom can be generated independently, and coarse zooms approximate the average of finer ones.

pub mod config;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod landcover;
pub mod noise;
pub mod store;
pub mod surface;
pub mod tile;
pub mod world;

pub use config::Config;
pub use store::GENERATOR_VERSION;
pub use tile::{Backend, Generator, PointTerrain, TileData, TILE_SIZE};
