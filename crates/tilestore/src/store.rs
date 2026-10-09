use crate::codec;
use crate::{Layer, TileData, TileId, FORMAT, FORMAT_VERSION, TILE_SIZE};
use anyhow::{bail, Context, Result};
use h5::Attrs;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

const DEFLATE_LEVEL: u32 = 4;

/// File-level metadata.
#[derive(Clone, Debug)]
pub struct StoreMeta {
    pub ellipsoid_a: f64,
    pub ellipsoid_b: f64,
    /// Generator configuration (YAML) that produced the tiles, if any.
    pub generator_config: String,
    pub seed: u64,
    /// Version of the generator that produced the tiles (0 = not recorded). Tiles of different
    /// versions differ slightly and must not be mixed in one store.
    pub generator_version: u32,
    /// Layers stored in this file.
    pub layers: Vec<Layer>,
}

impl Default for StoreMeta {
    fn default() -> Self {
        let e = geodesy::Ellipsoid::WGS84;
        StoreMeta { ellipsoid_a: e.a, ellipsoid_b: e.b, generator_config: String::new(), seed: 0, generator_version: 0, layers: Layer::ALL.to_vec() }
    }
}

impl StoreMeta {
    pub fn ellipsoid(&self) -> geodesy::Ellipsoid {
        geodesy::Ellipsoid { a: self.ellipsoid_a, b: self.ellipsoid_b }
    }
}

struct Level {
    rows: Vec<(u32, u32)>,
    index: HashMap<(u32, u32), usize>,
    ranges: Vec<(f32, f32)>,
    idx_ds: h5::Dataset,
    range_ds: h5::Dataset,
    layers: HashMap<Layer, h5::Dataset>,
}

/// A tile pyramid in an HDF5 file. Safe to share between threads: reads decode in parallel,
/// writes are serialized.
pub struct TileStore {
    file: h5::File,
    path: PathBuf,
    writable: bool,
    meta: StoreMeta,
    levels: RwLock<BTreeMap<u8, Level>>,
}

fn layer_shape(l: Layer, n: usize) -> Vec<usize> {
    if l.channels() > 1 {
        vec![n, TILE_SIZE, TILE_SIZE, l.channels()]
    } else {
        vec![n, TILE_SIZE, TILE_SIZE]
    }
}

impl TileStore {
    /// Create (replace) a store. The file is built as `<path>.tmp` and renamed into place once
    /// its metadata is on disk, so a process killed during creation never leaves a store that
    /// later runs cannot open.
    pub fn create(path: impl AsRef<Path>, meta: StoreMeta) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(p) = path.parent() {
            if !p.as_os_str().is_empty() {
                std::fs::create_dir_all(p)?;
            }
        }
        let tmp = {
            let mut s = path.clone().into_os_string();
            s.push(".tmp");
            PathBuf::from(s)
        };
        let file = h5::File::create(&tmp).map_err(|e| open_error(&tmp, e, "creating"))?;
        let res = (move || -> Result<()> {
            file.set_attr_str("format", FORMAT)?;
            file.set_attr("format_version", FORMAT_VERSION)?;
            file.set_attr("tile_size", TILE_SIZE as i32)?;
            file.set_attr_str("scheme", "xyz")?;
            file.set_attr_str("projection", "EPSG:3857")?;
            file.set_attr_str("pixel_registration", "center")?;
            file.set_attr_str("vertical_datum", "ellipsoid")?;
            file.set_attr("ellipsoid_a", meta.ellipsoid_a)?;
            file.set_attr("ellipsoid_b", meta.ellipsoid_b)?;
            file.set_attr_str("generator_config", &meta.generator_config)?;
            file.set_attr("seed", meta.seed)?;
            file.set_attr("generator_version", meta.generator_version)?;
            let names: Vec<&str> = meta.layers.iter().map(|l| l.name()).collect();
            file.set_attr_str("layers", &names.join(","))?;
            file.ensure_group("levels")?;
            file.flush()?;
            Ok(())
            // (the file closes here: every handle into it is dropped)
        })();
        if let Err(e) = res {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.context(format!("creating {}", path.display())));
        }
        std::fs::rename(&tmp, &path).with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
        Self::open_impl(&path, true)
    }

    /// Open read-only.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_impl(path.as_ref(), false)
    }

    /// Open for appending tiles.
    pub fn open_rw(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_impl(path.as_ref(), true)
    }

    /// Open for appending, or create with `meta` if it does not exist.
    pub fn open_or_create(path: impl AsRef<Path>, meta: StoreMeta) -> Result<Self> {
        if path.as_ref().exists() {
            Self::open_rw(path)
        } else {
            Self::create(path, meta)
        }
    }

    fn open_impl(path: &Path, writable: bool) -> Result<Self> {
        let file = if writable { h5::File::open_rw(path) } else { h5::File::open(path) }.map_err(|e| open_error(path, e, "opening"))?;
        let format = file.attr_str("format").unwrap_or_default();
        if format != FORMAT {
            bail!("{} is not a terrain tile store (format attr = {format:?})", path.display());
        }
        let layers = file.attr_str("layers")?.split(',').filter_map(Layer::from_name).collect::<Vec<_>>();
        let meta = StoreMeta {
            ellipsoid_a: file.attr("ellipsoid_a")?,
            ellipsoid_b: file.attr("ellipsoid_b")?,
            generator_config: file.attr_str("generator_config").unwrap_or_default(),
            seed: file.attr("seed").unwrap_or(0),
            generator_version: file.attr("generator_version").unwrap_or(0),
            layers,
        };
        let mut levels = BTreeMap::new();
        let lg = file.group("levels")?;
        for name in lg.member_names()? {
            let Ok(z) = name.parse::<u8>() else { continue };
            let g = lg.group(&name)?;
            let missing: Vec<&str> = ["index", "elev_range"].into_iter().chain(meta.layers.iter().map(|l| l.name())).filter(|n| !g.exists(n)).collect();
            if !missing.is_empty() {
                // a level whose creation was interrupted (it holds no tiles; a write to this zoom
                // completes it)
                eprintln!("warning: tile store {}: skipping incomplete level {z} (no {})", path.display(), missing.join(", "));
                continue;
            }
            let idx_ds = g.dataset("index")?;
            let range_ds = g.dataset("elev_range")?;
            let n = idx_ds.shape()?[0];
            let idx: Vec<i32> = if n > 0 { idx_ds.read_slice(&[0, 0], &[n, 2])? } else { vec![] };
            let rng: Vec<f32> = if n > 0 { range_ds.read_slice(&[0, 0], &[n, 2])? } else { vec![] };
            let rows: Vec<(u32, u32)> = idx.as_chunks::<2>().0.iter().map(|c| (c[0] as u32, c[1] as u32)).collect();
            // rows with a negative index were never completed (interrupted write): skip them
            let index =
                idx.as_chunks::<2>().0.iter().enumerate().filter(|(_, c)| c[0] >= 0 && c[1] >= 0).map(|(i, c)| ((c[0] as u32, c[1] as u32), i)).collect();
            let ranges = rng.as_chunks::<2>().0.iter().map(|c| (c[0], c[1])).collect();
            let mut lds = HashMap::new();
            for l in &meta.layers {
                if g.exists(l.name()) {
                    lds.insert(*l, g.dataset(l.name())?);
                }
            }
            levels.insert(z, Level { rows, index, ranges, idx_ds, range_ds, layers: lds });
        }
        Ok(TileStore { file, path: path.to_path_buf(), writable, meta, levels: RwLock::new(levels) })
    }

    /// Record the world (ellipsoid, config YAML, seed, generator version) of a writable store,
    /// e.g. of an empty store about to be filled with another world. The store's layers are kept.
    pub fn set_generator(&mut self, meta: &StoreMeta) -> Result<()> {
        if !self.writable {
            bail!("{} is open read-only", self.path.display());
        }
        self.file.set_attr("ellipsoid_a", meta.ellipsoid_a)?;
        self.file.set_attr("ellipsoid_b", meta.ellipsoid_b)?;
        self.file.set_attr_str("generator_config", &meta.generator_config)?;
        self.file.set_attr("seed", meta.seed)?;
        self.file.set_attr("generator_version", meta.generator_version)?;
        self.file.flush()?;
        self.meta.ellipsoid_a = meta.ellipsoid_a;
        self.meta.ellipsoid_b = meta.ellipsoid_b;
        self.meta.generator_config = meta.generator_config.clone();
        self.meta.seed = meta.seed;
        self.meta.generator_version = meta.generator_version;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Opened for writing (`create` / `open_rw`)?
    pub fn writable(&self) -> bool {
        self.writable
    }
    pub fn meta(&self) -> &StoreMeta {
        &self.meta
    }
    pub fn file(&self) -> &h5::File {
        &self.file
    }

    pub fn contains(&self, id: TileId) -> bool {
        self.levels.read().get(&id.z).is_some_and(|l| l.index.contains_key(&(id.x, id.y)))
    }

    pub fn zooms(&self) -> Vec<u8> {
        self.levels.read().keys().copied().collect()
    }

    pub fn tiles_at(&self, z: u8) -> Vec<TileId> {
        self.levels.read().get(&z).map(|l| l.rows.iter().filter(|k| l.index.contains_key(k)).map(|&(x, y)| TileId::new(z, x, y)).collect()).unwrap_or_default()
    }

    pub fn tiles(&self) -> Vec<TileId> {
        let lv = self.levels.read();
        lv.iter().flat_map(|(z, l)| l.rows.iter().filter(|k| l.index.contains_key(k)).map(move |&(x, y)| TileId::new(*z, x, y))).collect()
    }

    /// Number of stored tiles (rows of interrupted writes excluded).
    pub fn len(&self) -> usize {
        self.levels.read().values().map(|l| l.index.len()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// (min, max) elevation of a stored tile.
    pub fn elev_range(&self, id: TileId) -> Option<(f32, f32)> {
        let lv = self.levels.read();
        let l = lv.get(&id.z)?;
        l.index.get(&(id.x, id.y)).and_then(|&r| l.ranges.get(r).copied())
    }

    /// Create the datasets of level `z`. The layers come first and the index last, so a level
    /// with all of them is complete. The datasets of a level whose creation was interrupted
    /// (empty; skipped on open) are replaced.
    fn create_level(&self, z: u8) -> Result<Level> {
        let g = self.file.ensure_group(&format!("levels/{z}"))?;
        let names = self.meta.layers.iter().map(|l| l.name()).chain(["elev_range", "index"]);
        for name in names {
            if g.exists(name) {
                if g.dataset(name)?.shape()?.first().copied().unwrap_or(0) != 0 {
                    bail!("{}: level {z} is damaged ({name} has rows, but the level is incomplete); use a new tiles file", self.path.display());
                }
                g.delete(name)?;
            }
        }
        let mut layers = HashMap::new();
        for &l in &self.meta.layers {
            let shape = layer_shape(l, 0);
            let mut max: Vec<Option<usize>> = shape.iter().map(|&s| Some(s)).collect();
            max[0] = None;
            let mut chunk = shape.clone();
            chunk[0] = 1;
            let ds = match l {
                Layer::Elevation => {
                    g.new_dataset::<f32>().shape(&shape).max_shape(&max).chunk(&chunk).shuffle(true).deflate(DEFLATE_LEVEL as u8).create(l.name())?
                }
                Layer::Normal => {
                    g.new_dataset::<i8>().shape(&shape).max_shape(&max).chunk(&chunk).shuffle(true).deflate(DEFLATE_LEVEL as u8).create(l.name())?
                }
                _ => g.new_dataset::<u8>().shape(&shape).max_shape(&max).chunk(&chunk).shuffle(true).deflate(DEFLATE_LEVEL as u8).create(l.name())?,
            };
            ds.set_attr_str(
                "description",
                match l {
                    Layer::Rgb => "satellite-look imagery (baked lighting), sRGB u8",
                    Layer::Albedo => "surface albedo, sRGB-encoded u8",
                    Layer::Elevation => "DSM height above the ellipsoid (m) at pixel centres",
                    Layer::Normal => "unit surface normal (east, north, up) * 127, i8",
                    Layer::Landcover => "land-cover class id (terragen::landcover)",
                    Layer::Emission => "night-time artificial light, linear radiance = 16 * (v/255)^3",
                },
            )?;
            layers.insert(l, ds);
        }
        let range_ds = g.new_dataset::<f32>().shape(&[0, 2]).max_shape(&[None, Some(2)]).chunk(&[1024, 2]).deflate(4).create("elev_range")?;
        let idx_ds = g
            .new_dataset::<i32>()
            .shape(&[0, 2])
            .max_shape(&[None, Some(2)])
            .chunk(&[1024, 2])
            .deflate(4)
            .fill_value(-1) // rows never written (interrupted write) are recognisable
            .create("index")?;
        Ok(Level { rows: vec![], index: HashMap::new(), ranges: vec![], idx_ds, range_ds, layers })
    }

    /// Write (append or overwrite) tiles. Compression runs in parallel outside the HDF5 lock.
    pub fn write_tiles(&self, tiles: &[TileData]) -> Result<()> {
        if !self.writable {
            bail!("tile store {} opened read-only", self.path.display());
        }
        let layers = self.meta.layers.clone();
        // every tile must be valid and carry all of the store's layers at full size (a short
        // buffer would be stored as a corrupt chunk; a missing layer would leave an unallocated
        // chunk that fails every later read of that tile)
        for t in tiles {
            if !t.id.is_valid() {
                bail!("invalid tile id {:?}", t.id);
            }
            for &l in &layers {
                let n = t.layer_bytes(l).len();
                if n != l.tile_bytes() {
                    bail!("tile {:?}: layer {} has {n} bytes, expected {}", t.id, l.name(), l.tile_bytes());
                }
            }
        }
        let encoded: Vec<Vec<(Layer, Vec<u8>)>> = tiles
            .par_iter()
            .map(|t| layers.iter().filter(|l| t.has(**l)).map(|&l| (l, codec::encode(t.layer_bytes(l), l.elem_size(), DEFLATE_LEVEL))).collect())
            .collect();
        let mut lv = self.levels.write();
        // group by level so each level is resized once
        let mut by_level: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
        for (i, t) in tiles.iter().enumerate() {
            by_level.entry(t.id.z).or_default().push(i);
        }
        for (z, idxs) in by_level {
            let level = match lv.entry(z) {
                std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::btree_map::Entry::Vacant(e) => e.insert(self.create_level(z)?),
            };
            let mut rows = Vec::with_capacity(idxs.len());
            let n0 = level.rows.len();
            let ranges0: Vec<(f32, f32)> = level.ranges.clone();
            for &i in &idxs {
                let id = tiles[i].id;
                let key = (id.x, id.y);
                let row = match level.index.get(&key) {
                    Some(&r) => r,
                    None => {
                        let r = level.rows.len();
                        level.rows.push(key);
                        level.ranges.push((0.0, 0.0));
                        level.index.insert(key, r);
                        r
                    }
                };
                level.ranges[row] = (tiles[i].elev_min, tiles[i].elev_max);
                rows.push(row);
            }
            let n = level.rows.len();
            let res = (|| -> Result<()> {
                if n > n0 {
                    for (&l, ds) in &level.layers {
                        ds.resize(&layer_shape(l, n))?;
                    }
                    level.idx_ds.resize(&[n, 2])?;
                    level.range_ds.resize(&[n, 2])?;
                }
                // chunks first, the index last: rows of an interrupted or failed write keep the
                // index fill value (-1) and are skipped on open, so they are regenerated
                for (k, &i) in idxs.iter().enumerate() {
                    let row = rows[k];
                    for (l, bytes) in &encoded[i] {
                        let ds = level.layers.get(l).with_context(|| format!("{}: level {z} has no {} dataset", self.path.display(), l.name()))?;
                        let off = if l.channels() > 1 { vec![row, 0, 0, 0] } else { vec![row, 0, 0] };
                        ds.write_chunk_raw(&off, 0, bytes)?;
                    }
                }
                let lo = rows.iter().copied().min().unwrap_or(0);
                let idx: Vec<i32> = level.rows[lo..n].iter().flat_map(|&(x, y)| [x as i32, y as i32]).collect();
                let rng: Vec<f32> = level.ranges[lo..n].iter().flat_map(|&(a, b)| [a, b]).collect();
                level.range_ds.write_slice(&rng, &[lo, 0], &[n - lo, 2])?;
                level.idx_ds.write_slice(&idx, &[lo, 0], &[n - lo, 2])?;
                Ok(())
            })();
            if let Err(e) = res {
                // roll back the in-memory view of this level (new rows are dropped) and, best
                // effort, the file's (new index rows back to -1)
                if n > n0 {
                    let _ = level.idx_ds.write_slice(&vec![-1i32; (n - n0) * 2], &[n0, 0], &[n - n0, 2]);
                }
                for key in level.rows.drain(n0..) {
                    level.index.remove(&key);
                }
                level.ranges = ranges0;
                return Err(e);
            }
        }
        // bound what an abrupt end (kill, crash) can lose to the current batch
        self.file.flush()?;
        Ok(())
    }

    pub fn write_tile(&self, t: &TileData) -> Result<()> {
        self.write_tiles(std::slice::from_ref(t))
    }

    /// Read selected layers of a tile; `None` if the tile is not stored.
    pub fn read_tile(&self, id: TileId, layers: &[Layer]) -> Result<Option<TileData>> {
        // (layer, bytes, still compressed?)
        let raw: Vec<(Layer, Vec<u8>, bool)> = {
            let lv = self.levels.read();
            let Some(level) = lv.get(&id.z) else { return Ok(None) };
            let Some(&row) = level.index.get(&(id.x, id.y)) else { return Ok(None) };
            let mut out = Vec::with_capacity(layers.len());
            for &l in layers {
                let Some(ds) = level.layers.get(&l) else { continue };
                let off = if l.channels() > 1 { vec![row, 0, 0, 0] } else { vec![row, 0, 0] };
                let (mask, bytes) = ds.read_chunk_raw(&off)?;
                if mask != 0 {
                    // filters skipped: fall back to the regular (filtered) read path
                    let mut count = layer_shape(l, 1);
                    count[0] = 1;
                    let data = match l {
                        Layer::Elevation => {
                            let v: Vec<f32> = ds.read_slice(&off, &count)?;
                            v.iter().flat_map(|f| f.to_le_bytes()).collect()
                        }
                        // i8 normals must be read as i8 (an i8 → u8 conversion clips negatives)
                        Layer::Normal => ds.read_slice::<i8>(&off, &count)?.into_iter().map(|v| v as u8).collect(),
                        _ => ds.read_slice::<u8>(&off, &count)?,
                    };
                    out.push((l, data, false));
                    continue;
                }
                out.push((l, bytes, true));
            }
            out
        };
        let mut t = TileData { id, ..Default::default() };
        if let Some((a, b)) = self.elev_range(id) {
            t.elev_min = a;
            t.elev_max = b;
        }
        for (l, bytes, compressed) in raw {
            let data = if compressed {
                let d = codec::decode(&bytes, l.elem_size(), l.tile_bytes()).with_context(|| format!("decoding {id} layer {}", l.name()))?;
                if d.len() != l.tile_bytes() {
                    bail!("corrupt chunk for {id} layer {}", l.name());
                }
                d
            } else {
                bytes
            };
            t.set_layer_bytes(l, data);
        }
        Ok(Some(t))
    }

    pub fn flush(&self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }
}

/// A short error for the common ways of failing to open (or create) a store: the file is locked
/// by another process, or it is truncated (its creation was interrupted).
fn open_error(path: &Path, e: h5::Error, doing: &str) -> anyhow::Error {
    let stack = match &e {
        h5::Error::Hdf5 { stack, .. } => stack.as_str(),
        _ => "",
    };
    let name = path.display();
    if stack.contains("unable to lock file") && (stack.contains("errno = 11") || stack.contains("temporarily unavailable")) {
        return anyhow::anyhow!("{name} is open in another process (e.g. terrain view or terrain tiles); close it or use another tiles.file");
    }
    if stack.contains("truncated file") {
        let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if len < 4096 {
            return anyhow::anyhow!("{name} looks incomplete ({len} bytes: the process creating it was killed?); it holds no tiles and can be deleted");
        }
        return anyhow::Error::new(e).context(format!("{name} is truncated (a write was interrupted?); delete it to start over"));
    }
    anyhow::Error::new(e).context(format!("{doing} {name}"))
}
