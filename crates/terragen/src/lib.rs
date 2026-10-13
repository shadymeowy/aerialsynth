//! Procedural, deterministic, lazily evaluable terrain for XYZ tiles.
//!
//! `Generator::tile(TileId)` is a pure function of (config, tile id): any subset of tiles at any
//! zoom can be generated independently, and coarse zooms approximate the average of finer ones.
//! Point queries ([`Generator::terrain_points`], [`Generator::probe`]) evaluate the same world at
//! single locations.
//!
//! Two generators build worlds, chosen by the config's `generator` ([`GeneratorKind`]):
//!
//! * [`classic`] (the default): the generator of v0.2.0, unchanged (bit-identical tiles);
//! * next: the modular generator with the planetary atlas, biome registry and kits
//!   ([`NextGenerator`] and the modules below), a preview.
//!
//! [`Generator`] holds either; the GPU device, its error handling and the compile session are
//! shared ([`gpu`]). The modules below are the next generator's unless noted:
//!
//! * [`atlas`]: the planetary atlas: world-scale fields (climate with rain shadows and
//!   seasons, plates, lithology, glaciation, cultures) precomputed once per world on a cube map;
//! * [`world`]: the macro-scale world model ("pass A"): continents, relief, hydrology, climate;
//! * [`surface`], [`stack`], [`layers`]: the fine-scale surface ("pass B"): albedo, DSM height,
//!   land cover, as a fixed stack of composited layers;
//! * [`registry`], [`kernels`], [`eco`], [`kits`]: biomes as data over a kernel library,
//!   ecoregions and cultures, and the kits that add biomes, kernels and layers;
//! * [`noise`], [`landcover`], [`config`]: noise primitives, land-cover classes, the world config;
//! * [`tile`]: the [`NextGenerator`] producing [`TileData`] (elevation, albedo, satellite rgb,
//!   normals, land cover, night lights);
//! * `gpu` (feature `gpu`, default): the same generator as wgpu compute shaders, used when a
//!   device with 64-bit float / integer shader support is available; its device, compile
//!   notices and pipeline cache serve both generators;
//! * [`generator`]: [`Generator`], the world's generator (classic or next);
//! * [`store`]: the generators' [`tilestore`] binding (config, generator and version stored
//!   with the tiles).

pub mod atlas;
pub mod classic;
pub mod config;
pub mod eco;
pub mod features;
pub mod generator;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod instances;
pub mod kernels;
pub mod kits;
pub mod landcover;
pub mod layers;
pub mod memo;
pub mod noise;
pub(crate) mod pool;
pub mod registry;
pub mod stack;
pub mod store;
pub mod surface;
pub mod tile;
pub mod world;

pub use config::{Config, GeneratorKind};
pub use generator::Generator;
pub use store::{CLASSIC_GENERATOR_VERSION, GENERATOR_VERSION};
pub use tile::{Backend, NextGenerator, PointTerrain, TileData, TILE_SIZE};
