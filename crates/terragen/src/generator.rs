//! The world's tile generator: the classic or the next one, as `world.generator` says.
//!
//! [`Generator`] is what the rest of aerialsynth uses (CLI, renderer, viewer, survey,
//! bindings). It holds either generator and forwards to it; both have the same interface
//! (tiles, point queries, backends, the background GPU setup) and the same infrastructure (GPU
//! device, strict mode, compile notices, pipeline cache). Tile stores record which generator
//! wrote them (`store.rs`).

use crate::config::GeneratorKind;
use crate::tile::{Backend, NextGenerator, PointTerrain, TileData};
use crate::Config;
use geodesy::tiles::TileId;
use std::sync::Arc;

/// The tile generator of a world: the classic generator (`world.generator: classic`, the
/// default, see [`crate::classic`]) or the next one (`next`, [`NextGenerator`]).
pub struct Generator {
    imp: Imp,
}

enum Imp {
    Classic(Arc<crate::classic::Generator>),
    Next(Arc<NextGenerator>),
}

/// Forward a call to the generator.
macro_rules! each {
    ($self:expr, $g:ident => $e:expr) => {
        match &$self.imp {
            Imp::Classic($g) => $e,
            Imp::Next($g) => $e,
        }
    };
}

impl Generator {
    /// The generator of `cfg.generator`, on the GPU when there is one (`Backend::Auto`). The
    /// config is not checked: use [`Generator::try_new`] for configs from users.
    pub fn new(cfg: Config) -> Self {
        Self::with_backend(cfg, Backend::Auto)
    }

    /// The generator of `cfg.generator` on `backend` (the GPU generator is compiled on first use).
    pub fn with_backend(cfg: Config, backend: Backend) -> Self {
        let imp = match cfg.generator {
            GeneratorKind::Classic => Imp::Classic(Arc::new(crate::classic::Generator::with_backend(cfg, backend))),
            GeneratorKind::Next => Imp::Next(Arc::new(NextGenerator::with_backend(cfg, backend))),
        };
        Generator { imp }
    }

    /// The generator of `cfg.generator` on the GPU when there is one, for a validated config
    /// ([`Config::validate`]).
    pub fn try_new(cfg: Config) -> anyhow::Result<Self> {
        Self::try_with_backend(cfg, Backend::Auto)
    }

    /// The generator of `cfg.generator` on `backend` for a validated config
    /// ([`Config::validate`]); `Backend::Gpu` fails without a suitable GPU.
    pub fn try_with_backend(cfg: Config, backend: Backend) -> anyhow::Result<Self> {
        let imp = match cfg.generator {
            GeneratorKind::Classic => Imp::Classic(Arc::new(crate::classic::Generator::try_with_backend(cfg, backend)?)),
            GeneratorKind::Next => Imp::Next(Arc::new(NextGenerator::try_with_backend(cfg, backend)?)),
        };
        Ok(Generator { imp })
    }

    /// Which generator this is.
    pub fn kind(&self) -> GeneratorKind {
        match self.imp {
            Imp::Classic(_) => GeneratorKind::Classic,
            Imp::Next(_) => GeneratorKind::Next,
        }
    }

    /// The version of this generator's output (what its tile stores record).
    pub fn version(&self) -> u32 {
        self.kind().version()
    }

    /// The next generator, if this is it.
    pub fn next(&self) -> Option<&NextGenerator> {
        match &self.imp {
            Imp::Next(g) => Some(g),
            Imp::Classic(_) => None,
        }
    }

    /// The classic generator, if this is it.
    pub fn classic(&self) -> Option<&crate::classic::Generator> {
        match &self.imp {
            Imp::Classic(g) => Some(g),
            Imp::Next(_) => None,
        }
    }

    /// The world's ellipsoid.
    pub fn ell(&self) -> geodesy::Ellipsoid {
        each!(self, g => g.world.ell)
    }

    /// The world config (`generator` set to this generator).
    pub fn config(&self) -> &Config {
        each!(self, g => g.config())
    }

    /// With `Backend::Auto`, set the GPU generator up on a background thread (see
    /// [`NextGenerator::prepare_gpu_in_background`]); the CPU generates until it is ready.
    pub fn prepare_gpu_in_background(&self) {
        each!(self, g => g.prepare_gpu_in_background())
    }

    /// `"GPU (<adapter>)"` or `"CPU"`: where tiles are generated.
    pub fn backend_name(&self) -> String {
        each!(self, g => g.backend_name())
    }

    /// Generate tiles: on the GPU in batches, or on the CPU in parallel. With `Backend::Auto`,
    /// a failed GPU batch is generated on the CPU (and so is everything after it).
    pub fn tiles(&self, ids: &[TileId]) -> anyhow::Result<Vec<TileData>> {
        each!(self, g => g.tiles(ids))
    }

    /// Generate one tile (see [`Generator::tiles`]; panics if the GPU fails).
    pub fn tile(&self, id: TileId) -> TileData {
        each!(self, g => g.tile(id))
    }

    /// Generate one tile on the CPU.
    pub fn tile_cpu(&self, id: TileId) -> TileData {
        each!(self, g => g.tile_cpu(id))
    }

    /// Pass A at points (lat, lon in radians, pixel size in metres): the bare ground and the
    /// standing water (on the GPU when tiles are generated there).
    pub fn terrain_points(&self, pts: &[(f64, f64, f64)]) -> anyhow::Result<Vec<PointTerrain>> {
        each!(self, g => g.terrain_points(pts))
    }

    /// The world at one point (lat, lon in radians) seen with pixels of `gsd` metres: pass A,
    /// the surface height and the land-cover class. The classic generator's pass A fills the
    /// fields it has (the next generator's ecoregion and atlas fields stay 0).
    pub fn probe(&self, lat: f64, lon: f64, gsd: f64) -> (crate::world::Terrain, f64, u8) {
        match &self.imp {
            Imp::Next(g) => g.probe(lat, lon, gsd),
            Imp::Classic(g) => {
                let (t, h, c) = g.probe(lat, lon, gsd);
                (classic_terrain(&t), h, c)
            }
        }
    }

    /// The biome of the ecoregion at pass-A point `t` (index into the registry, its name); `None`
    /// for the classic generator (it has no biome registry).
    pub fn biome(&self, t: &crate::world::Terrain) -> Option<(u16, &str)> {
        self.next().map(|g| g.biome(t))
    }
}

/// The classic generator's pass A as the next generator's `Terrain` (the fields both have).
fn classic_terrain(t: &crate::classic::world::Terrain) -> crate::world::Terrain {
    let site =
        |s: &crate::classic::world::Site| crate::world::Site { id: s.id, id2: s.id2, center: s.center, dist: s.dist, edge: s.edge, ..Default::default() };
    crate::world::Terrain {
        ground: t.ground,
        water: t.water,
        water_kind: t.water_kind,
        river_d: t.river_d,
        river_hw: t.river_hw,
        river_level: t.river_level,
        river_wet: t.river_wet,
        temp: t.temp,
        moist: t.moist,
        mountain: t.mountain,
        rock_expect: t.rock_expect,
        sand: t.sand,
        floodplain: t.floodplain,
        mesa: t.mesa,
        cont: t.cont,
        agri: t.agri,
        habit: t.habit,
        gully: t.gully,
        style: t.style,
        road_major: t.road_major,
        road_minor: t.road_minor,
        region: site(&t.region),
        town: site(&t.town),
        ..Default::default()
    }
}
