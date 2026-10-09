//! Tile access and camera rendering shared by the C (`bindings/c`) and Python
//! (`bindings/python`) bindings, so both behave identically.
//!
//! A [`World`] is a tile store opened for one world (seed + world config). [`World::tile`] reads
//! a layer of a tile from the store; a tile that is not stored yet is generated (on the GPU when
//! there is a suitable one, else on the CPU), written to the store, and returned. A [`Camera`]
//! renders images of the world (the renderer of `terrain run`), generating the tiles it needs
//! into the store the same way.
//!
//! The world is given like the CLI's `terrain -c FILE --seed N`: an optional scenario YAML
//! (only its `world:` section is used) or a bare world config, and an optional seed override.
//! A store holds exactly one world: opening a store of another world (or of another generator
//! version) fails, naming the settings that differ.

use anyhow::Context;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use terragen::{Config, Generator};
use tilestore::{TileStore, TILE_SIZE};

mod camera;

pub use camera::{Backend, Camera, CameraDef, Exposure, Frame, Mount, Outputs, Pinhole, Pose, MAX_IMAGE_SIZE};
pub use terragen::GENERATOR_VERSION;
pub use tilestore::{Layer, TileId};

/// Highest zoom level of a tile (`x, y < 2^z`).
pub const MAX_ZOOM: u32 = geodesy::MAX_ZOOM as u32;

/// Width and height of a tile in pixels.
pub const TILE_PX: usize = TILE_SIZE;

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

/// A tile store of one world, generating missing tiles on demand.
///
/// `World` is `Send + Sync`: tiles can be requested from several threads at once (store reads
/// run in parallel, writes are serialized). Two threads asking for the same missing tile may
/// both generate it; the result is identical (generation is deterministic) and stored once.
///
/// A tiles file is open by at most one `World` per process: a second open fails while the
/// first is alive (share the handle instead).
pub struct World {
    /// (shared with the tile caches of cameras)
    gen: Arc<Generator>,
    store: Arc<TileStore>,
    path: PathBuf,
    max_zoom: u32,
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
        let store = gen.open_store_rw(tiles_file).with_context(|| format!("opening the tile store {}", tiles_file.display()))?;
        let key = tiles_file.canonicalize().with_context(|| format!("resolving {}", tiles_file.display()))?;
        open.push(key.clone());
        Ok(World { gen: Arc::new(gen), store: Arc::new(store), path: tiles_file.to_path_buf(), max_zoom: spec.max_zoom, _open: OpenFile(key) })
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
        let out = &mut out[..n];
        if let Some(t) = self.store.read_tile(id, &[layer]).with_context(|| format!("reading tile {id} from {}", self.path.display()))? {
            copy_layer(&t, layer, out)?;
            return Ok(());
        }
        let t = self.gen.tiles(&[id]).with_context(|| format!("generating tile {id}"))?.remove(0);
        self.store.write_tile(&t).with_context(|| format!("writing tile {id} to {}", self.path.display()))?;
        copy_layer(&t, layer, out)
    }
}

/// A layer of `t` into `out` as little-endian bytes.
fn copy_layer(t: &tilestore::TileData, layer: Layer, out: &mut [u8]) -> Result<()> {
    match layer {
        Layer::Elevation => {
            if t.elevation.len() * 4 != out.len() {
                return Err(Error::Failed(anyhow::anyhow!("tile {}: elevation layer has {} values", t.id, t.elevation.len())));
            }
            for (o, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(&t.elevation) {
                *o = v.to_le_bytes();
            }
        }
        _ => {
            let b = t.layer_bytes(layer);
            if b.len() != out.len() {
                return Err(Error::Failed(anyhow::anyhow!("tile {}: layer {} has {} bytes, expected {}", t.id, layer.name(), b.len(), out.len())));
            }
            out.copy_from_slice(b);
        }
    }
    Ok(())
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
