//! Procedural, deterministic, lazily evaluable terrain for XYZ tiles.
//!
//! `Generator::tile(TileId)` is a pure function of (config, tile id): any subset of tiles at any
//! zoom can be generated independently, and coarse zooms approximate the average of finer ones.
//! Point queries ([`Generator::terrain_points`], [`Generator::probe`]) evaluate the same world at
//! single locations.
//!
//! * [`atlas`]: the planetary atlas: world-scale fields (climate with rain shadows and
//!   seasons, plates, lithology, glaciation, cultures) precomputed once per world on a cube map;
//! * [`world`]: the macro-scale world model ("pass A"): continents, relief, hydrology, climate;
//! * [`surface`], [`stack`], [`layers`]: the fine-scale surface ("pass B"): albedo, DSM height,
//!   land cover, as a fixed stack of composited layers;
//! * [`registry`], [`kernels`], [`eco`], [`kits`]: biomes as data over a kernel library,
//!   ecoregions and cultures, and the kits that add biomes, kernels and layers;
//! * [`noise`], [`landcover`], [`config`]: noise primitives, land-cover classes, the world config;
//! * [`tile`]: the [`Generator`] producing [`TileData`] (elevation, albedo, satellite rgb,
//!   normals, land cover, night lights);
//! * `gpu` (feature `gpu`, default): the same generator as wgpu compute shaders, used when a
//!   device with 64-bit float / integer shader support is available;
//! * [`store`]: the generator's [`tilestore`] binding (config and version stored with the tiles).

pub mod atlas;
pub mod config;
pub mod eco;
pub mod features;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod instances;
pub mod kernels;
pub mod kits;
pub mod landcover;
pub mod layers;
pub mod noise;
pub mod registry;
pub mod stack;
pub mod store;
pub mod surface;
pub mod tile;
pub mod world;

pub use config::Config;
pub use store::GENERATOR_VERSION;
pub use tile::{Backend, Generator, PointTerrain, TileData, TILE_SIZE};
