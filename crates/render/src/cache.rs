//! Tile cache over a `TileStore`, with optional lazy generation of missing tiles.

use crate::lod::TileOracle;
use anyhow::Result;
use geodesy::tiles::TileId;
use parking_lot::Mutex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use terragen::Generator;
use tilestore::{Layer, TileData, TileStore};

/// Where progress messages go (one line each, without a newline).
pub type Log = Arc<dyn Fn(&str) + Send + Sync>;

pub struct TileCache {
    pub store: Arc<TileStore>,
    /// When set, tiles missing from the store are generated on demand (and written back if the
    /// store is writable and `write_back` is true).
    pub generator: Option<Arc<Generator>>,
    pub write_back: bool,
    /// Zoom range that lazy generation may create.
    pub lazy_max_zoom: u8,
    layers: Vec<Layer>,
    capacity: usize,
    inner: Mutex<Inner>,
    /// progress of lazy generation (None: quiet)
    log: Mutex<Option<Log>>,
}

#[derive(Default)]
struct Inner {
    map: HashMap<TileId, (Arc<TileData>, u64)>,
    tick: u64,
    /// tiles generated so far
    generated: usize,
    /// elevation ranges of the tiles generated without writing them back (the store knows the
    /// others): an evicted tile keeps its range, so the LOD selection does not depend on what
    /// the cache holds
    ranges: HashMap<TileId, (f32, f32)>,
}

impl TileCache {
    pub fn new(store: Arc<TileStore>, layers: Vec<Layer>, capacity: usize) -> Self {
        TileCache {
            store,
            generator: None,
            write_back: false,
            lazy_max_zoom: 0,
            layers,
            capacity: capacity.max(1),
            inner: Mutex::new(Inner::default()),
            log: Mutex::new(None),
        }
    }

    pub fn with_generator(mut self, g: Arc<Generator>, max_zoom: u8, write_back: bool) -> Self {
        self.generator = Some(g);
        self.lazy_max_zoom = max_zoom;
        self.write_back = write_back;
        self
    }

    /// Report lazy generation (a line per batch of tiles) to `log`; None: quiet (the default).
    pub fn set_log(&self, log: Option<Log>) {
        *self.log.lock() = log;
    }

    /// Does the cache generate missing tiles?
    pub fn lazy(&self) -> bool {
        self.generator.is_some()
    }

    /// Is the tile available (stored, cached, or generatable)?
    pub fn available(&self, id: TileId) -> bool {
        self.store.contains(id) || self.inner.lock().map.contains_key(&id) || self.generatable(id)
    }

    fn generatable(&self, id: TileId) -> bool {
        self.generator.is_some() && id.z <= self.lazy_max_zoom
    }

    /// Tiles held in memory: at most the capacity (`tiles.cache_tiles`).
    pub fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, id: TileId) -> Option<Arc<TileData>> {
        {
            let mut g = self.inner.lock();
            g.tick += 1;
            let tick = g.tick;
            if let Some(e) = g.map.get_mut(&id) {
                e.1 = tick;
                return Some(e.0.clone());
            }
        }
        let t = match self.store.read_tile(id, &self.layers).ok().flatten() {
            Some(t) => t,
            None if self.generatable(id) => self.generate(&[id]).ok()?.pop()?,
            None => return None,
        };
        let t = Arc::new(t);
        self.insert(id, t.clone());
        Some(t)
    }

    /// Generate tiles in batches (the memory they take is bounded by the batch size), each
    /// written back right away (an evicted tile is read again instead of generated again);
    /// returns them reduced to the cached layers, as a store read would.
    fn generate(&self, ids: &[TileId]) -> Result<Vec<TileData>> {
        let Some(g) = &self.generator else { return Ok(vec![]) };
        if ids.is_empty() {
            return Ok(vec![]);
        }
        let batch = (rayon::current_num_threads() * 4).clamp(8, 256);
        let log = self.log.lock().clone();
        let t0 = std::time::Instant::now();
        if let Some(log) = &log {
            let (z0, z1) = ids.iter().fold((u8::MAX, 0), |(a, b), id| (a.min(id.z), b.max(id.z)));
            let zooms = if z0 == z1 { format!("z{z0}") } else { format!("z{z0}-{z1}") };
            let n = ids.len();
            log(&format!("generating {n} missing tile{} ({zooms}) on the {}", if n == 1 { "" } else { "s" }, g.backend_name()));
        }
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(batch) {
            let mut tiles = g.tiles(chunk)?;
            if self.write_back {
                self.store.write_tiles(&tiles)?;
            }
            {
                let mut inner = self.inner.lock();
                inner.generated += tiles.len();
                if !self.write_back {
                    inner.ranges.extend(tiles.iter().map(|t| (t.id, (t.elev_min, t.elev_max))));
                }
            }
            for t in &mut tiles {
                t.retain_layers(&self.layers);
            }
            out.extend(tiles);
            if let Some(log) = &log {
                let s = t0.elapsed().as_secs_f64();
                if out.len() < ids.len() {
                    log(&format!("  {}/{} tiles ({s:.1} s)", out.len(), ids.len()));
                } else {
                    log(&format!("  {} tiles in {s:.1} s", ids.len()));
                }
            }
        }
        Ok(out)
    }

    fn insert(&self, id: TileId, t: Arc<TileData>) {
        let mut g = self.inner.lock();
        g.tick += 1;
        let tick = g.tick;
        g.map.insert(id, (t, tick));
        if g.map.len() > self.capacity {
            // evict the least recently used quarter
            let mut v: Vec<(u64, TileId)> = g.map.iter().map(|(k, e)| (e.1, *k)).collect();
            v.sort_unstable();
            for (_, k) in v.iter().take(g.map.len() - self.capacity * 3 / 4) {
                g.map.remove(k);
            }
        }
    }

    /// Load many tiles in parallel (decode / generation outside the cache lock). A failed
    /// generation is reported as a warning (those tiles stay missing).
    pub fn prefetch(&self, ids: &[TileId]) {
        if let Err(e) = self.try_prefetch(ids) {
            eprintln!("warning: generating tiles failed: {e:#}");
        }
    }

    /// [`TileCache::prefetch`], failing when generating or writing back tiles does.
    pub fn try_prefetch(&self, ids: &[TileId]) -> Result<()> {
        let missing: Vec<TileId> = {
            let g = self.inner.lock();
            ids.iter().copied().filter(|id| !g.map.contains_key(id)).collect()
        };
        // stored tiles read in parallel, the rest generated in batches
        let read: Vec<(TileId, Option<TileData>)> = missing.par_iter().map(|&id| (id, self.store.read_tile(id, &self.layers).ok().flatten())).collect();
        let mut todo = Vec::new();
        for (id, t) in read {
            match t {
                Some(t) => self.insert(id, Arc::new(t)),
                None if self.generatable(id) => todo.push(id),
                None => {}
            }
        }
        for t in self.generate(&todo)? {
            self.insert(t.id, Arc::new(t));
        }
        Ok(())
    }

    /// Flush the store; returns the number of tiles generated and written back so far.
    pub fn flush_generated(&self) -> Result<usize> {
        let n = if self.write_back { self.inner.lock().generated } else { 0 };
        if n > 0 {
            self.store.flush()?;
        }
        Ok(n)
    }
}

impl TileOracle for TileCache {
    fn range(&self, id: TileId) -> Option<(f32, f32)> {
        if let Some(r) = self.store.elev_range(id) {
            return Some(r);
        }
        let g = self.inner.lock();
        g.map.get(&id).map(|e| (e.0.elev_min, e.0.elev_max)).or_else(|| g.ranges.get(&id).copied())
    }
    fn exists(&self, id: TileId) -> bool {
        self.available(id)
    }
    /// With lazy generation a tile of unknown range is generated (and the view selected again,
    /// `Renderer::select_units`) rather than refined on a guessed range.
    fn refine_unknown(&self) -> bool {
        !self.lazy()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A directory removed on drop.
    pub(crate) struct TempDir(pub PathBuf);
    impl TempDir {
        pub(crate) fn new(name: &str) -> Self {
            let d = std::env::temp_dir().join(format!("render-{name}-{}", std::process::id()));
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

    /// A cheap world (one sample per pixel) generated on the CPU.
    pub(crate) fn cheap_generator() -> Arc<Generator> {
        let cfg = terragen::Config { tile_supersample: 1, ..Default::default() };
        Arc::new(Generator::try_with_backend(cfg, terragen::Backend::Cpu).unwrap())
    }

    /// Lazily generated tiles are written back, and the cache holds at most its capacity however
    /// many tiles one prefetch generates.
    #[test]
    fn generated_tiles_are_bounded_by_the_capacity() {
        let d = TempDir::new("cache-bounded");
        let gen = cheap_generator();
        let store = Arc::new(gen.open_store_rw(&d.0.join("t.h5")).unwrap());
        let cache = TileCache::new(store.clone(), vec![Layer::Elevation], 6).with_generator(gen, 12, true);
        let ids: Vec<TileId> = (0..4).flat_map(|y| (0..5).map(move |x| TileId::new(10, 600 + x, 380 + y))).collect();
        cache.try_prefetch(&ids).unwrap();
        assert!(cache.len() <= 6, "{} tiles held", cache.len());
        assert!(ids.iter().all(|id| store.contains(*id)));
        // evicted tiles are read back (not generated again), their ranges known
        assert!(ids.iter().all(|id| cache.range(*id).is_some() && cache.get(*id).is_some()));
        assert!(cache.len() <= 6, "{} tiles held", cache.len());
        assert_eq!(cache.flush_generated().unwrap(), ids.len());
    }

    /// A tile generated by the cache is what reading the stored tile gives.
    #[test]
    fn generated_equals_stored() {
        let d = TempDir::new("cache-same");
        let gen = cheap_generator();
        let store = Arc::new(gen.open_store_rw(&d.0.join("t.h5")).unwrap());
        let layers = vec![Layer::Elevation, Layer::Landcover, Layer::Normal];
        let cache = TileCache::new(store.clone(), layers.clone(), 10).with_generator(gen, 12, true);
        let id = TileId::new(11, 1200, 770);
        let fresh = cache.get(id).unwrap();
        let stored = store.read_tile(id, &layers).unwrap().unwrap();
        assert_eq!((fresh.elev_min, fresh.elev_max), (stored.elev_min, stored.elev_max));
        assert_eq!(fresh.elevation, stored.elevation);
        assert_eq!(fresh.landcover, stored.landcover);
        assert_eq!(fresh.normal, stored.normal);
        assert!(fresh.rgb.is_empty() && fresh.albedo.is_empty());
    }
}
