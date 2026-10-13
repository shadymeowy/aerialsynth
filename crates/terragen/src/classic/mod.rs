//! The classic terrain generator (`world.generator: classic`, the default): the generator of
//! aerialsynth v0.2.0, kept unchanged so that its tiles stay bit-identical to v0.2.0's for the
//! same config and seed (tile stores record generator version 3, see
//! [`crate::CLASSIC_GENERATOR_VERSION`]).
//!
//! The terrain code (CPU and WGSL) is v0.2.0's (`git show v0.2.0:crates/terragen/src/…`) with
//! its module paths moved under `classic::`. What it shares with the next generator is
//! infrastructure only, so the robustness work since v0.2.0 applies to both:
//!
//! * the config type ([`crate::Config`], a superset of v0.2.0's; classic reads the settings
//!   v0.2.0 had and ignores the rest);
//! * the land-cover ids ([`crate::landcover`]: v2 keeps the v0.2.0 ids 0–17 unchanged, which
//!   are the only ones classic emits, so the mapping is the identity);
//! * the GPU device and its handling ([`crate::gpu::device`]: adapter selection
//!   `AERIALSYNTH_GPU`, device-lost / stall errors, concurrent read-backs, the exit fixes),
//!   strict mode (`AERIALSYNTH_GPU_STRICT`), and the compile session (on-disk pipeline cache
//!   under its own `classic-` file names, compile notices, the device-wide compile lock);
//! * the backends and point results ([`crate::tile::Backend`], [`crate::tile::PointTerrain`]).
//!
//! [`crate::Generator`] picks this generator or the next one from `world.generator`.
//! `tests/classic_v020.rs` compares its tiles with those of the v0.2.0 binary.

pub mod world;
pub mod surface;
pub mod noise;
pub mod tile;
#[cfg(feature = "gpu")]
pub mod gpu;

pub use tile::Generator;
