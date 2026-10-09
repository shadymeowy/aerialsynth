//! Tile access and camera rendering shared by the C (`bindings/c`) and Python
//! (`bindings/python`) bindings, so both behave identically.
//!
//! A [`World`] is a tile store opened for one world (seed + world config). [`World::tile`] reads
//! a layer of a tile from the store; a tile that is not stored yet is generated (on the GPU when
//! there is a suitable one, else on the CPU), written to the store, and returned. A [`Camera`]
//! renders images of the world (the renderer of `terrain run`), generating the tiles it needs
//! into the store the same way.
//!
//! [`World::tiles_into`] gets many tiles at once (parallel decompression, missing tiles generated
//! in batches), and a world keeps the tiles it reads or generates in an in-memory cache
//! ([`World::set_cache_mb`]).
//!
//! The world is given like the CLI's `terrain -c FILE --seed N`: an optional scenario YAML
//! (only its `world:` section is used) or a bare world config, and an optional seed override.
//! A store holds exactly one world: opening a store of another world (or of another generator
//! version) fails, naming the settings that differ.

use anyhow::Context;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use terragen::{Config, Generator};
use tilestore::{TileData, TileStore, TILE_SIZE};

mod cache;
mod camera;

pub use cache::CacheStats;
use cache::{Bytes, TileCache};

pub use camera::{Backend, Camera, CameraDef, Exposure, Frame, Mount, Outputs, Pinhole, Pose, MAX_IMAGE_SIZE};
pub use terragen::GENERATOR_VERSION;
pub use tilestore::{Layer, TileId};

/// Highest zoom level of a tile (`x, y < 2^z`).
pub const MAX_ZOOM: u32 = geodesy::MAX_ZOOM as u32;

/// Width and height of a tile in pixels.
pub const TILE_PX: usize = TILE_SIZE;

/// Default size of a world's tile cache in MiB ([`World::set_cache_mb`]).
pub const DEFAULT_CACHE_MB: usize = 256;

/// Most tiles [`World::prefetch`] takes (its box over its zooms).
pub const MAX_PREFETCH_TILES: u64 = 1_000_000;

/// Tiles generated per [`Generator::tiles`] call (and stored per write) by [`World::tiles_into`]
/// and [`World::prefetch`]: the GPU generates many tiles per dispatch, and a batch of all layers
/// takes about 70 MB.
const GEN_BATCH: usize = 64;

/// Top-level keys of a scenario YAML (`render::scenario::Scenario`). A config file with any of
/// them is a scenario (its world is the `world:` section); otherwise it is a bare world config.
const SCENARIO_KEYS: [&str; 7] = ["world", "tiles", "trajectory", "render", "cameras", "imu", "output"];

/// Errors of the bindings.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A bad argument: tile coordinates, layer name, buffer size, seed...
    #[error("{0}")]
    InvalidArgument(String),
    /// A file could not be read (the config file).
    #[error("{context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    /// Anything else: an invalid config, a store of another world, HDF5 or generation errors.
    #[error("{0:#}")]
    Failed(#[from] anyhow::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Element type of a layer's pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    U8,
    I8,
    /// little-endian IEEE 754 binary32
    F32,
}

impl Dtype {
    /// numpy-style type string: `"|u1"`, `"|i1"`, `"<f4"`.
    pub fn numpy(self) -> &'static str {
        match self {
            Dtype::U8 => "|u1",
            Dtype::I8 => "|i1",
            Dtype::F32 => "<f4",
        }
    }
    pub fn size(self) -> usize {
        match self {
            Dtype::U8 | Dtype::I8 => 1,
            Dtype::F32 => 4,
        }
    }
}

/// Pixel format and meaning of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerInfo {
    pub layer: Layer,
    pub name: &'static str,
    pub dtype: Dtype,
    /// 1 (shape 256 × 256) or 3 (shape 256 × 256 × 3)
    pub channels: usize,
    /// bytes of one tile of this layer: 256 · 256 · channels · dtype size
    pub size: usize,
    pub description: &'static str,
}

/// Describe a layer.
pub fn layer_info(layer: Layer) -> LayerInfo {
    let dtype = match layer {
        Layer::Elevation => Dtype::F32,
        Layer::Normal => Dtype::I8,
        _ => Dtype::U8,
    };
    debug_assert_eq!(dtype.size(), layer.elem_size());
    let description = match layer {
        Layer::Rgb => "satellite look: the surface lit by a fixed sun with haze, sRGB",
        Layer::Albedo => "unlit surface colour, sRGB encoded",
        Layer::Elevation => "DSM (ground, canopy, buildings, water surface): metres above the WGS84 ellipsoid at pixel centres",
        Layer::Normal => "unit surface normal (east, north, up) * 127",
        Layer::Landcover => "land-cover class id (terragen::landcover::NAMES: 0 unknown, 1 ocean, 2 lake, 3 river, ...)",
        Layer::Emission => "night-time artificial light: linear radiance = 16 * (v/255)^3",
    };
    LayerInfo { layer, name: layer.name(), dtype, channels: layer.channels(), size: layer.tile_bytes(), description }
}

/// Every layer, in the order of the C `AS_LAYER_*` values.
pub fn layers() -> [LayerInfo; 6] {
    Layer::ALL.map(layer_info)
}

/// A layer by name (`"rgb"`, `"albedo"`, `"elevation"`, `"normal"`, `"landcover"`, `"emission"`).
pub fn layer_by_name(name: &str) -> Result<Layer> {
    Layer::from_name(name).ok_or_else(|| {
        let names: Vec<&str> = Layer::ALL.iter().map(|l| l.name()).collect();
        Error::InvalidArgument(format!("unknown layer {name:?} (layers: {})", names.join(", ")))
    })
}

/// Default zoom limit of a world: the scenario default `tiles.max_zoom`.
pub const DEFAULT_MAX_ZOOM: u32 = 18;

/// A world: its config and the highest zoom its tiles are served at.
#[derive(Clone, Debug)]
pub struct WorldSpec {
    pub config: Config,
    /// Tiles above this zoom are refused (`tiles.max_zoom` of a scenario, default 18; at most
    /// [`MAX_ZOOM`]).
    pub max_zoom: u32,
}

impl Default for WorldSpec {
    fn default() -> Self {
        WorldSpec { config: Config::default(), max_zoom: DEFAULT_MAX_ZOOM }
    }
}

/// The world of a scenario or world config file (`None`: the default world), with an optional
/// seed override.
///
/// A file whose top level has a scenario section (`world`, `tiles`, `render`, ...) is a scenario:
/// its `world:` section is the world and its `tiles.max_zoom` the zoom limit (the other sections
/// are ignored). Otherwise the file is a bare world config (the contents of a `world:` section)
/// and the zoom limit is [`DEFAULT_MAX_ZOOM`]. An empty file is the default world.
pub fn world_spec(config: Option<&Path>, seed: Option<u64>) -> Result<WorldSpec> {
    let mut spec = match config {
        None => WorldSpec::default(),
        Some(p) => {
            let s = std::fs::read_to_string(p).map_err(|source| Error::Io { context: format!("reading {}", p.display()), source })?;
            parse_spec(&s).with_context(|| format!("parsing {}", p.display()))?
        }
    };
    if let Some(seed) = seed {
        spec.config.seed = seed;
    }
    Ok(spec)
}

fn parse_spec(s: &str) -> anyhow::Result<WorldSpec> {
    use serde_yaml::Value;
    let v: Value = serde_yaml::from_str(s)?;
    let (world, max_zoom) = match v {
        Value::Null => return Ok(WorldSpec::default()),
        Value::Mapping(ref m) if SCENARIO_KEYS.iter().any(|k| m.contains_key(*k)) => {
            let max_zoom = match m.get("tiles").and_then(|t| t.get("max_zoom")) {
                None => DEFAULT_MAX_ZOOM,
                Some(z) => match z.as_u64() {
                    Some(z) if z <= MAX_ZOOM as u64 => z as u32,
                    _ => anyhow::bail!("tiles.max_zoom must be an integer in 0..={MAX_ZOOM}"),
                },
            };
            (m.get("world").cloned().unwrap_or(Value::Null), max_zoom)
        }
        v => (v, DEFAULT_MAX_ZOOM),
    };
    let config = if world.is_null() { Config::default() } else { serde_yaml::from_value(world).context("world")? };
    Ok(WorldSpec { config, max_zoom })
}

/// A tile id from binding arguments: `z <= max_zoom` (at most [`MAX_ZOOM`]), `x, y < 2^z`.
pub fn tile_id(z: u32, x: u32, y: u32, max_zoom: u32) -> Result<TileId> {
    let max_zoom = max_zoom.min(MAX_ZOOM);
    if z > max_zoom {
        return Err(Error::InvalidArgument(format!("zoom {z} out of range 0..={max_zoom} (the world's tiles.max_zoom)")));
    }
    let n = 1u64 << z;
    if x as u64 >= n || y as u64 >= n {
        return Err(Error::InvalidArgument(format!("tile {z}/{x}/{y}: x and y must be < 2^{z} = {n}")));
    }
    Ok(TileId::new(z as u8, x, y))
}

/// A tile store of one world, generating missing tiles on demand, with an in-memory cache of the
/// decoded tiles.
///
/// `World` is `Send + Sync`: tiles can be requested from several threads at once (store reads
/// run in parallel, writes are serialized; the cache's lock is held only to look up and insert).
/// Two threads asking for the same missing tile may both generate it; the result is identical
/// (generation is deterministic) and stored once.
///
/// A tiles file is open by at most one `World` per process: a second open fails while the
/// first is alive (share the handle instead).
pub struct World {
    /// (shared with the tile caches of cameras)
    gen: Arc<Generator>,
    store: Arc<TileStore>,
    path: PathBuf,
    max_zoom: u32,
    cache: TileCache,
    /// The store's entry in [`OPEN`]; after `store`, so it is dropped after the store is closed.
    _open: OpenFile,
}

/// Tiles files open in this process (canonical paths). A second `TileStore` of the same file
/// would keep its own index of the file's rows: the two would overwrite each other's tiles.
static OPEN: std::sync::Mutex<Vec<PathBuf>> = std::sync::Mutex::new(Vec::new());

fn open_files() -> std::sync::MutexGuard<'static, Vec<PathBuf>> {
    OPEN.lock().unwrap_or_else(|e| e.into_inner())
}

struct OpenFile(PathBuf);

impl Drop for OpenFile {
    fn drop(&mut self) {
        open_files().retain(|k| *k != self.0);
    }
}

const _: () = {
    const fn send_sync<T: Send + Sync>() {}
    send_sync::<World>()
};

impl World {
    /// Open (or create) the tile store `tiles_file` for the world of `config` (a scenario or world
    /// config YAML; `None`: the default world) with an optional seed override.
    ///
    /// A new store (or an existing empty one) is stamped with this world; an existing store must
    /// hold this world and generator version.
    pub fn open(tiles_file: &Path, config: Option<&Path>, seed: Option<u64>) -> Result<World> {
        Self::with_spec(tiles_file, world_spec(config, seed)?)
    }

    /// [`World::open`] with a world spec.
    pub fn with_spec(tiles_file: &Path, spec: WorldSpec) -> Result<World> {
        if spec.max_zoom > MAX_ZOOM {
            return Err(Error::InvalidArgument(format!("max_zoom {} > {MAX_ZOOM}", spec.max_zoom)));
        }
        if tiles_file.as_os_str().is_empty() {
            return Err(Error::InvalidArgument("empty tiles file path".into()));
        }
        // (validated: a bad config value is an error, not a broken world)
        let gen = Generator::try_new(spec.config).map_err(Error::Failed)?;
        // (the registry stays locked until the store is open: two threads opening the same new
        // file must not both create it)
        let mut open = open_files();
        if let Ok(key) = tiles_file.canonicalize() {
            if open.contains(&key) {
                return Err(Error::Failed(anyhow::anyhow!(
                    "{} is already open in this process: use that handle (it can be shared between threads) or close it first",
                    tiles_file.display()
                )));
            }
        }
        // (the store's errors name the file and lead with the reason: "… is a directory")
        let store = gen.open_store_rw(tiles_file)?;
        let key = tiles_file.canonicalize().with_context(|| format!("resolving {}", tiles_file.display()))?;
        open.push(key.clone());
        Ok(World {
            gen: Arc::new(gen),
            store: Arc::new(store),
            path: tiles_file.to_path_buf(),
            max_zoom: spec.max_zoom,
            cache: TileCache::new(DEFAULT_CACHE_MB << 20),
            _open: OpenFile(key),
        })
    }

    /// The tiles file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The world's seed.
    pub fn seed(&self) -> u64 {
        self.gen.config().seed
    }

    /// The highest zoom served (tiles above it are refused).
    pub fn max_zoom(&self) -> u32 {
        self.max_zoom
    }

    /// The world config (as stored in the tile store).
    pub fn config(&self) -> &Config {
        self.gen.config()
    }

    /// The DSM height (ground, canopy, buildings, water surface; metres above the WGS84
    /// ellipsoid) at a point (degrees), evaluated by the generator at the ground resolution of
    /// the max zoom: about what the tiles hold there. For heights above ground.
    pub fn surface_height(&self, lat_deg: f64, lon_deg: f64) -> Result<f64> {
        if !(lat_deg.is_finite() && lon_deg.is_finite() && lat_deg.abs() <= 90.0) {
            return Err(Error::InvalidArgument(format!("latitude {lat_deg}° / longitude {lon_deg}°: finite values with |latitude| <= 90 expected")));
        }
        let lat = lat_deg.to_radians();
        let ell = self.store.meta().ellipsoid();
        let gsd = geodesy::tiles::gsd_ew(lat, self.max_zoom as u8, TILE_SIZE as u32, &ell).max(0.1);
        Ok(self.gen.probe(lat, lon_deg.to_radians(), gsd).1)
    }

    /// Is the tile stored (i.e. would [`World::tile`] read it rather than generate it)?
    pub fn contains(&self, z: u32, x: u32, y: u32) -> Result<bool> {
        Ok(self.store.contains(tile_id(z, x, y, self.max_zoom)?))
    }

    /// The raw pixels of a layer of tile `z/x/y` ([`LayerInfo::size`] bytes: row-major,
    /// 256 rows (row 0 = north) × 256 columns × channels, little-endian). Generates and stores the
    /// tile (all layers) if it is missing.
    pub fn tile(&self, z: u32, x: u32, y: u32, layer: Layer) -> Result<Vec<u8>> {
        let mut out = vec![0u8; layer.tile_bytes()];
        self.tile_into(z, x, y, layer, &mut out)?;
        Ok(out)
    }

    /// [`World::tile`] into `out` (at least [`LayerInfo::size`] bytes; only those are written).
    pub fn tile_into(&self, z: u32, x: u32, y: u32, layer: Layer, out: &mut [u8]) -> Result<()> {
        let id = tile_id(z, x, y, self.max_zoom)?;
        let n = layer.tile_bytes();
        if out.len() < n {
            return Err(Error::InvalidArgument(format!("buffer of {} bytes is too small for layer {} ({n} bytes)", out.len(), layer.name())));
        }
        let v = match self.cache.get(id, layer) {
            Some(v) => v,
            None => match self.read_stored(id, layer)? {
                Some(v) => {
                    self.cache.insert([(id, layer, v.clone())]);
                    v
                }
                None => self.generate(&[id], Some(layer))?.remove(0).expect("the requested layer"),
            },
        };
        out[..n].copy_from_slice(&v);
        Ok(())
    }

    /// A layer of many tiles: tile `i` (`tiles[i]` = z, x, y) into `out[i * size ..][.. size]`,
    /// `size` = [`LayerInfo::size`]; `out` must hold `tiles.len() * size` bytes (only those are
    /// written).
    ///
    /// Every coordinate is checked before any work. Cached tiles are copied, the stored ones read
    /// and decompressed in parallel, the missing ones generated together (in batches of up to
    /// 64 tiles; one store write per batch). A tile listed several times is read or generated
    /// once. On an error the contents of `out` are unspecified (the tiles generated so far are
    /// stored).
    pub fn tiles_into(&self, tiles: &[[u32; 3]], layer: Layer, out: &mut [u8]) -> Result<()> {
        let ids = tiles.iter().map(|&[z, x, y]| tile_id(z, x, y, self.max_zoom)).collect::<Result<Vec<_>>>()?;
        let size = layer.tile_bytes();
        let need = size.checked_mul(ids.len()).ok_or_else(|| Error::InvalidArgument(format!("{} tiles are too many", ids.len())))?;
        if out.len() < need {
            return Err(Error::InvalidArgument(format!(
                "buffer of {} bytes is too small for {} tiles of layer {} ({need} bytes)",
                out.len(),
                ids.len(),
                layer.name()
            )));
        }
        // the distinct tiles, and which of them each requested tile is
        let mut index: HashMap<TileId, usize> = HashMap::with_capacity(ids.len());
        let mut uniq: Vec<TileId> = Vec::with_capacity(ids.len());
        let slot: Vec<usize> = ids
            .iter()
            .map(|&id| {
                *index.entry(id).or_insert_with(|| {
                    uniq.push(id);
                    uniq.len() - 1
                })
            })
            .collect();
        let mut vals: Vec<Option<Bytes>> = uniq.iter().map(|&id| self.cache.get(id, layer)).collect();
        // stored: read and decompressed in parallel
        let todo: Vec<usize> = (0..uniq.len()).filter(|&i| vals[i].is_none()).collect();
        let read = |&i: &usize| self.read_stored(uniq[i], layer);
        let read: Vec<Result<Option<Bytes>>> = if todo.len() > 1 { todo.par_iter().map(read).collect() } else { todo.iter().map(read).collect() };
        let mut missing = Vec::new();
        let mut cached = Vec::new();
        for (&i, r) in todo.iter().zip(read) {
            match r? {
                Some(v) => {
                    cached.push((uniq[i], layer, v.clone()));
                    vals[i] = Some(v);
                }
                None => missing.push(i),
            }
        }
        self.cache.insert(cached);
        // missing: generated in batches
        for chunk in missing.chunks(GEN_BATCH) {
            let ids: Vec<TileId> = chunk.iter().map(|&i| uniq[i]).collect();
            for (&i, v) in chunk.iter().zip(self.generate(&ids, Some(layer))?) {
                vals[i] = v;
            }
        }
        let vals: Vec<Bytes> = vals.into_iter().map(|v| v.expect("every tile read or generated")).collect();
        let out = &mut out[..need];
        if ids.len() > 1 {
            out.par_chunks_mut(size).zip(&slot).for_each(|(o, &s)| o.copy_from_slice(&vals[s]));
        } else {
            for (o, &s) in out.chunks_mut(size).zip(&slot) {
                o.copy_from_slice(&vals[s]);
            }
        }
        Ok(())
    }

    /// Generate and store the missing tiles of a box (`[lat_min, lon_min, lat_max, lon_max]`,
    /// degrees; `lon_min > lon_max` is a box across the antimeridian) at zooms `z_min..=z_max`,
    /// in batches of up to 64 tiles; the number of tiles generated. They are not cached.
    ///
    /// The box must hold at most [`MAX_PREFETCH_TILES`] tiles over those zooms (stored ones
    /// included); this and the arguments are checked before any work.
    pub fn prefetch(&self, bbox: [f64; 4], z_min: u32, z_max: u32) -> Result<usize> {
        let [lat_min, lon_min, lat_max, lon_max] = bbox;
        if !bbox.iter().all(|v| v.is_finite()) || lat_min.abs() > 90.0 || lat_max.abs() > 90.0 || lon_min.abs() > 180.0 || lon_max.abs() > 180.0 {
            return Err(Error::InvalidArgument(format!("box {bbox:?}: finite degrees with |latitude| <= 90 and |longitude| <= 180 expected")));
        }
        if lat_min >= lat_max || lon_min == lon_max {
            return Err(Error::InvalidArgument(format!("box {bbox:?}: lat_min < lat_max and lon_min != lon_max expected")));
        }
        if z_min > z_max || z_max > self.max_zoom {
            return Err(Error::InvalidArgument(format!("zooms {z_min}..={z_max}: needs z_min <= z_max <= {} (the world's tiles.max_zoom)", self.max_zoom)));
        }
        let b = geodesy::tiles::LatLonBounds {
            lat_min: lat_min.to_radians(),
            lon_min: lon_min.to_radians(),
            lat_max: lat_max.to_radians(),
            lon_max: lon_max.to_radians(),
        };
        let n: u64 = (z_min..=z_max).map(|z| geodesy::tiles::count_tiles_in_bounds(&b, z as u8)).sum();
        if n > MAX_PREFETCH_TILES {
            return Err(Error::InvalidArgument(format!(
                "the box holds {n} tiles at zooms {z_min}..={z_max} (at most {MAX_PREFETCH_TILES}): use a smaller box or fewer zooms"
            )));
        }
        let todo: Vec<TileId> = (z_min..=z_max).flat_map(|z| geodesy::tiles::tiles_in_bounds(&b, z as u8)).filter(|&id| !self.store.contains(id)).collect();
        for chunk in todo.chunks(GEN_BATCH) {
            self.generate(chunk, None)?;
        }
        Ok(todo.len())
    }

    /// Set the size of the cache of decoded tiles to `mb` MiB (default [`DEFAULT_CACHE_MB`];
    /// 0 turns it off and empties it). Tiles read or generated are cached (a generated tile with
    /// all its layers, about 1.1 MiB), the least recently used dropped beyond the size.
    pub fn set_cache_mb(&self, mb: usize) {
        self.cache.set_capacity(mb.saturating_mul(1 << 20));
    }

    /// The size of the tile cache in MiB.
    pub fn cache_mb(&self) -> usize {
        self.cache.stats().capacity >> 20
    }

    /// Usage of the tile cache.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// A layer of a stored tile (`None`: not stored).
    fn read_stored(&self, id: TileId, layer: Layer) -> Result<Option<Bytes>> {
        let t = self.store.read_tile(id, &[layer]).with_context(|| format!("reading tile {id} from {}", self.path.display()))?;
        t.map(|mut t| take_layer(&mut t, layer).map(Arc::new)).transpose()
    }

    /// Generate tiles (one batch) and store them; with `layer`, cache them (all their layers, the
    /// requested one last: the most recently used) and return that layer of each.
    fn generate(&self, ids: &[TileId], layer: Option<Layer>) -> Result<Vec<Option<Bytes>>> {
        let what = || if ids.len() == 1 { format!("tile {}", ids[0]) } else { format!("{} tiles", ids.len()) };
        let tiles = self.gen.tiles(ids).with_context(|| format!("generating {}", what()))?;
        if tiles.len() != ids.len() || tiles.iter().zip(ids).any(|(t, id)| t.id != *id) {
            return Err(Error::Failed(anyhow::anyhow!("the generator returned other tiles than asked for")));
        }
        self.store.write_tiles(&tiles).with_context(|| format!("writing {} to {}", what(), self.path.display()))?;
        let Some(layer) = layer else { return Ok(vec![None; tiles.len()]) };
        let all = self.cache.stats().capacity > 0;
        let mut out = Vec::with_capacity(tiles.len());
        let mut cached = Vec::new();
        for mut t in tiles {
            if all {
                for l in Layer::ALL.into_iter().filter(|&l| l != layer) {
                    cached.push((t.id, l, Arc::new(take_layer(&mut t, l)?)));
                }
            }
            let v = Arc::new(take_layer(&mut t, layer)?);
            cached.push((t.id, layer, v.clone()));
            out.push(Some(v));
        }
        self.cache.insert(cached);
        Ok(out)
    }
}

/// A layer of `t` as little-endian bytes (taken out of `t`).
fn take_layer(t: &mut TileData, layer: Layer) -> Result<Vec<u8>> {
    let v: Vec<u8> = match layer {
        Layer::Elevation => std::mem::take(&mut t.elevation).iter().flat_map(|v| v.to_le_bytes()).collect(),
        Layer::Normal => std::mem::take(&mut t.normal).into_iter().map(|v| v as u8).collect(),
        Layer::Rgb => std::mem::take(&mut t.rgb),
        Layer::Albedo => std::mem::take(&mut t.albedo),
        Layer::Landcover => std::mem::take(&mut t.landcover),
        Layer::Emission => std::mem::take(&mut t.emission),
    };
    if v.len() != layer.tile_bytes() {
        return Err(Error::Failed(anyhow::anyhow!("tile {}: layer {} has {} bytes, expected {}", t.id, layer.name(), v.len(), layer.tile_bytes())));
    }
    Ok(v)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A fresh temporary directory (removed on drop).
    pub(crate) struct TempDir(pub PathBuf);
    impl TempDir {
        pub(crate) fn new(name: &str) -> Self {
            let d = std::env::temp_dir().join(format!("aerialsynth-core-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            TempDir(d)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A cheap world (one sample per pixel).
    fn fast_config(dir: &Path) -> PathBuf {
        let p = dir.join("world.yaml");
        std::fs::write(&p, "world:\n  tile_supersample: 1\ntiles:\n  max_zoom: 12\n").unwrap();
        p
    }

    #[test]
    fn tiles_are_generated_once_then_read() {
        let d = TempDir::new("gen");
        let cfg = fast_config(&d.0);
        let path = d.0.join("w.h5");
        let w = World::open(&path, Some(&cfg), Some(5)).unwrap();
        assert_eq!(w.seed(), 5);
        assert_eq!(w.config().tile_supersample, 1);
        assert!(!w.contains(4, 9, 6).unwrap());
        let elev = w.tile(4, 9, 6, Layer::Elevation).unwrap();
        assert_eq!(elev.len(), 256 * 256 * 4);
        assert!(w.contains(4, 9, 6).unwrap());
        // every layer of the generated tile was stored: the second call reads the same bytes
        assert_eq!(w.tile(4, 9, 6, Layer::Elevation).unwrap(), elev);
        let h: Vec<f32> = elev.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        assert!(h.iter().all(|v| v.is_finite() && v.abs() < 12_000.0), "elevation out of range");
        for info in layers() {
            let b = w.tile(4, 9, 6, info.layer).unwrap();
            assert_eq!(b.len(), info.size);
        }
        drop(w);
        // reopened: still the same world and tile
        let w = World::open(&path, Some(&cfg), Some(5)).unwrap();
        assert_eq!(w.tile(4, 9, 6, Layer::Elevation).unwrap(), elev);
        // a second handle of the same file (also by another path) is refused while one is open
        let other = d.0.join(".").join("w.h5");
        let e = World::open(&other, Some(&cfg), Some(5)).err().expect("a second handle of an open store");
        assert!(e.to_string().contains("already open"), "{e}");
        let mut small = vec![0u8; 100];
        assert!(matches!(w.tile_into(4, 9, 6, Layer::Rgb, &mut small), Err(Error::InvalidArgument(_))));
    }

    #[test]
    fn the_cache_serves_the_same_bytes() {
        let d = TempDir::new("cache");
        let cfg = fast_config(&d.0);
        let w = World::open(&d.0.join("w.h5"), Some(&cfg), Some(5)).unwrap();
        assert_eq!(w.cache_mb(), DEFAULT_CACHE_MB);
        let rgb = w.tile(4, 9, 6, Layer::Rgb).unwrap(); // generated: all its layers are cached
        let s = w.cache_stats();
        assert_eq!(s.entries, 6);
        assert_eq!(s.bytes, Layer::ALL.iter().map(|l| l.tile_bytes()).sum::<usize>());
        let cached: Vec<Vec<u8>> = Layer::ALL
            .iter()
            .map(|&l| {
                let before = w.cache_stats().hits;
                let b = w.tile(4, 9, 6, l).unwrap();
                assert_eq!(w.cache_stats().hits, before + 1, "{l:?} from the cache");
                b
            })
            .collect();
        assert_eq!(cached[0], rgb);
        // 0: off and empty, every read goes to the store
        w.set_cache_mb(0);
        let s = w.cache_stats();
        assert_eq!((s.capacity, s.bytes, s.entries), (0, 0, 0));
        for (l, b) in Layer::ALL.iter().zip(&cached) {
            // the same bytes as read from the store
            assert_eq!(w.tile(4, 9, 6, *l).unwrap(), *b, "{l:?}");
        }
        assert_eq!((w.cache_stats().entries, w.cache_stats().hits), (0, s.hits));
        // 1 MiB holds 5 rgb layers: the least recently used go first
        w.set_cache_mb(1);
        let ids: Vec<[u32; 3]> = (0..6).map(|x| [3, x, 2]).collect();
        let mut all = vec![0u8; 6 * Layer::Rgb.tile_bytes()];
        w.tiles_into(&ids, Layer::Rgb, &mut all).unwrap(); // generated (with the cache on)
        w.set_cache_mb(0);
        w.set_cache_mb(1);
        for &[z, x, y] in &ids {
            w.tile(z, x, y, Layer::Rgb).unwrap();
        }
        let s = w.cache_stats();
        assert_eq!((s.entries, s.bytes), (5, 5 * Layer::Rgb.tile_bytes()));
        let hits = s.hits;
        w.tile(3, 5, 2, Layer::Rgb).unwrap(); // the most recent: cached
        assert_eq!(w.cache_stats().hits, hits + 1);
        w.tile(3, 0, 2, Layer::Rgb).unwrap(); // the oldest: evicted
        assert_eq!(w.cache_stats().hits, hits + 1);
    }

    #[test]
    fn batches_equal_single_tiles() {
        let d = TempDir::new("batch");
        let cfg = fast_config(&d.0);
        let one = World::open(&d.0.join("one.h5"), Some(&cfg), Some(5)).unwrap();
        let many = World::open(&d.0.join("many.h5"), Some(&cfg), Some(5)).unwrap();
        many.set_cache_mb(0);
        // stored before (3/4/2), missing, repeated
        let ids: [[u32; 3]; 6] = [[3, 4, 2], [3, 5, 2], [3, 4, 2], [4, 9, 6], [2, 1, 1], [3, 5, 2]];
        many.tile(3, 4, 2, Layer::Landcover).unwrap();
        for layer in [Layer::Elevation, Layer::Rgb, Layer::Normal] {
            let n = layer.tile_bytes();
            let mut out = vec![7u8; ids.len() * n + 5];
            many.tiles_into(&ids, layer, &mut out).unwrap();
            for (i, &[z, x, y]) in ids.iter().enumerate() {
                assert!(out[i * n..(i + 1) * n] == one.tile(z, x, y, layer).unwrap()[..], "{layer:?} {z}/{x}/{y}");
            }
            assert_eq!(out[ids.len() * n..], [7; 5], "only the tiles are written");
        }
        assert_eq!(many.store.len(), 4);
        // with the cache: the same
        many.set_cache_mb(64);
        let mut a = vec![0u8; 6 * Layer::Rgb.tile_bytes()];
        let mut b = a.clone();
        many.tiles_into(&ids, Layer::Rgb, &mut a).unwrap();
        many.tiles_into(&ids, Layer::Rgb, &mut b).unwrap();
        assert!(a == b);
        assert!(many.cache_stats().hits >= 4);
        // nothing: fine
        many.tiles_into(&[], Layer::Rgb, &mut []).unwrap();
        // a bad coordinate: refused before 3/6/2 (valid, missing) is made
        for bad in [[3, 8, 0], [13, 0, 0], [3, 0, 8]] {
            let e = many.tiles_into(&[[3, 6, 2], bad], Layer::Rgb, &mut vec![0u8; 2 * Layer::Rgb.tile_bytes()]);
            assert!(matches!(e, Err(Error::InvalidArgument(_))), "{bad:?}");
        }
        assert!(!many.contains(3, 6, 2).unwrap());
        // a short buffer: refused before any work
        let e = many.tiles_into(&[[3, 6, 2], [3, 7, 2]], Layer::Rgb, &mut vec![0u8; 2 * Layer::Rgb.tile_bytes() - 1]);
        assert!(matches!(e, Err(Error::InvalidArgument(ref m)) if m.contains("too small")), "{e:?}");
        assert!(!many.contains(3, 6, 2).unwrap());
    }

    #[test]
    fn prefetch_generates_the_missing_tiles() {
        let d = TempDir::new("prefetch");
        let cfg = fast_config(&d.0); // tiles.max_zoom 12
        let w = World::open(&d.0.join("w.h5"), Some(&cfg), Some(5)).unwrap();
        w.tile(4, 8, 5, Layer::Rgb).unwrap(); // stored already
        let bbox = [44.0, 9.0, 46.0, 11.0];
        let entries = w.cache_stats().entries;
        assert_eq!(w.prefetch(bbox, 0, 5).unwrap(), 5); // one tile per zoom, 4/8/5 stored
        assert!(w.contains(5, 16, 11).unwrap() && w.contains(0, 0, 0).unwrap());
        assert_eq!(w.cache_stats().entries, entries, "not cached");
        assert_eq!(w.prefetch(bbox, 0, 5).unwrap(), 0);
        // across the antimeridian: both sides
        assert_eq!(w.prefetch([-10.0, 170.0, 10.0, -170.0], 2, 2).unwrap(), 4);
        for (b, z0, z1) in [
            (bbox, 0, 13),
            (bbox, 3, 2),
            ([46.0, 9.0, 44.0, 11.0], 3, 3),
            ([44.0, 9.0, 95.0, 11.0], 3, 3),
            ([44.0, 9.0, f64::NAN, 11.0], 3, 3),
            ([44.0, 9.0, 46.0, 9.0], 3, 3),
            ([-80.0, -180.0, 80.0, 180.0], 0, 12), // millions of tiles
        ] {
            assert!(matches!(w.prefetch(b, z0, z1), Err(Error::InvalidArgument(_))), "{b:?} {z0}..={z1}");
        }
    }

    #[test]
    fn another_world_is_refused() {
        let d = TempDir::new("world");
        let cfg = fast_config(&d.0);
        let path = d.0.join("w.h5");
        let w = World::open(&path, Some(&cfg), Some(3)).unwrap();
        w.tile(3, 4, 2, Layer::Landcover).unwrap();
        drop(w);
        let e = World::open(&path, Some(&cfg), Some(4)).err().expect("a store of seed 3 opened with seed 4");
        assert!(matches!(e, Error::Failed(_)));
        assert!(e.to_string().contains("world.seed (3 → 4)"), "{e}");
        // another setting than the seed: the default supersampling
        assert!(World::open(&path, None, Some(3)).is_err());
    }

    #[test]
    fn bad_arguments_are_rejected() {
        let d = TempDir::new("args");
        let cfg = fast_config(&d.0); // tiles.max_zoom 12
        let w = World::open(&d.0.join("w.h5"), Some(&cfg), None).unwrap();
        assert_eq!(w.max_zoom(), 12);
        for (z, x, y) in [(3, 8, 0), (3, 0, 8), (13, 0, 0), (MAX_ZOOM + 1, 0, 0), (u32::MAX, 0, 0), (0, 1, 0)] {
            assert!(matches!(w.tile(z, x, y, Layer::Rgb), Err(Error::InvalidArgument(_))), "{z}/{x}/{y}");
            assert!(matches!(w.contains(z, x, y), Err(Error::InvalidArgument(_))), "{z}/{x}/{y}");
        }
        assert!(tile_id(MAX_ZOOM, (1 << MAX_ZOOM) - 1, 0, MAX_ZOOM).is_ok());
        assert!(tile_id(19, 0, 0, DEFAULT_MAX_ZOOM).is_err() && tile_id(18, 0, 0, DEFAULT_MAX_ZOOM).is_ok());
        assert!(layer_by_name("height").is_err());
        assert_eq!(layer_by_name("normal").unwrap(), Layer::Normal);
        assert!(matches!(World::open(&d.0.join("x.h5"), Some(&d.0.join("missing.yaml")), None), Err(Error::Io { .. })));
        assert!(matches!(World::open(Path::new(""), None, None), Err(Error::InvalidArgument(_))));
        // invalid world values are refused before a store is made
        let bad = d.0.join("bad.yaml");
        std::fs::write(&bad, "world: { planet: { a: 0 } }\n").unwrap();
        let e = World::open(&d.0.join("bad.h5"), Some(&bad), None).err().expect("planet.a = 0");
        assert!(e.to_string().contains("planet.a"), "{e}");
        assert!(!d.0.join("bad.h5").exists());
    }

    #[test]
    fn a_directory_or_another_file_is_not_a_store() {
        let d = TempDir::new("notastore");
        let text = d.0.join("notes.txt");
        std::fs::write(&text, "hello").unwrap();
        for (p, want) in [(&d.0, "is a directory"), (&text, "is not an HDF5 file")] {
            let e = World::open(p, None, None).err().expect("not a store, opened").to_string();
            assert!(e.starts_with(&format!("{} {want}", p.display())), "{e}");
            assert_eq!(e.lines().count(), 1, "{e}");
        }
    }

    #[test]
    fn config_files() {
        let d = TempDir::new("cfg");
        let spec = |s: &str, seed: Option<u64>| {
            let p = d.0.join("c.yaml");
            std::fs::write(&p, s).unwrap();
            world_spec(Some(&p), seed)
        };
        // a scenario's world section, a bare world config, an empty file, a scenario without world
        let a = spec("world: { seed: 9 }\nrender: {}\n", None).unwrap();
        assert_eq!((a.config.seed, a.max_zoom), (9, DEFAULT_MAX_ZOOM));
        let b = spec("seed: 11\ntile_supersample: 1\n", None).unwrap();
        assert_eq!((b.config.seed, b.config.tile_supersample, b.max_zoom), (11, 1, DEFAULT_MAX_ZOOM));
        assert_eq!(spec("", None).unwrap().config.seed, Config::default().seed);
        let d4 = spec("tiles: { max_zoom: 14 }\n", Some(3)).unwrap();
        assert_eq!((d4.config.seed, d4.max_zoom), (3, 14));
        assert!(spec("world: { no_such_setting: 1 }\n", None).is_err());
        assert!(spec("tiles: { max_zoom: 31 }\n", None).is_err());
        assert!(spec("tiles: { max_zoom: -1 }\n", None).is_err());
        // the repository's scenarios load
        let configs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs");
        for f in ["quick.yaml", "dataset.yaml"] {
            world_spec(Some(&configs.join(f)), None).unwrap_or_else(|e| panic!("{f}: {e}"));
        }
    }
}
