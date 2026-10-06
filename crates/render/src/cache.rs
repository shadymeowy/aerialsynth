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
}

#[derive(Default)]
struct Inner {
    map: HashMap<TileId, (Arc<TileData>, u64)>,
    tick: u64,
    /// tiles generated (and written back) so far
    generated: usize,
}

impl TileCache {
    pub fn new(store: Arc<TileStore>, layers: Vec<Layer>, capacity: usize) -> Self {
        TileCache { store, generator: None, write_back: false, lazy_max_zoom: 0, layers, capacity, inner: Mutex::new(Inner::default()) }
    }

    pub fn with_generator(mut self, g: Arc<Generator>, max_zoom: u8, write_back: bool) -> Self {
        self.generator = Some(g);
        self.lazy_max_zoom = max_zoom;
        self.write_back = write_back;
        self
    }

    /// Is the tile available (stored, cached, or generatable)?
    pub fn available(&self, id: TileId) -> bool {
        self.store.contains(id)
            || self.inner.lock().map.contains_key(&id)
            || (self.generator.is_some() && id.z <= self.lazy_max_zoom)
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
        let (t, gen) = self.load(id).ok().flatten()?;
        let t = if gen { self.store_generated(vec![t]).ok()?.pop()? } else { t };
        let t = Arc::new(t);
        self.insert(id, t.clone());
        Some(t)
    }

    /// Read a tile from the store, or generate it (second value = generated).
    fn load(&self, id: TileId) -> Result<Option<(TileData, bool)>> {
        if let Some(t) = self.store.read_tile(id, &self.layers)? {
            return Ok(Some((t, false)));
        }
        if let Some(g) = &self.generator {
            if id.z <= self.lazy_max_zoom {
                return Ok(Some((g.tile(id), true)));
            }
        }
        Ok(None)
    }

    /// Write generated tiles back right away (so memory stays bounded and an evicted tile is
    /// re-read instead of re-generated); returns them reduced to the cached layers.
    fn store_generated(&self, mut tiles: Vec<TileData>) -> Result<Vec<TileData>> {
        if tiles.is_empty() {
            return Ok(tiles);
        }
        if self.write_back {
            self.store.write_tiles(&tiles)?;
            self.inner.lock().generated += tiles.len();
        }
        for t in &mut tiles {
            t.retain_layers(&self.layers);
        }
        Ok(tiles)
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

    /// Load many tiles in parallel (decode / generation outside the cache lock).
    pub fn prefetch(&self, ids: &[TileId]) {
        let missing: Vec<TileId> = {
            let g = self.inner.lock();
            ids.iter().copied().filter(|id| !g.map.contains_key(id)).collect()
        };
        let loaded: Vec<(TileData, bool)> = missing.par_iter().filter_map(|&id| self.load(id).ok().flatten()).collect();
        let (gen, stored): (Vec<_>, Vec<_>) = loaded.into_iter().partition(|(_, g)| *g);
        let gen = match self.store_generated(gen.into_iter().map(|(t, _)| t).collect()) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("warning: writing generated tiles back failed: {e:#}");
                vec![]
            }
        };
        for t in stored.into_iter().map(|(t, _)| t).chain(gen) {
            self.insert(t.id, Arc::new(t));
        }
    }

    /// Flush the store; returns the number of tiles generated and written back so far.
    pub fn flush_generated(&self) -> Result<usize> {
        let n = self.inner.lock().generated;
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
        g.map.get(&id).map(|e| (e.0.elev_min, e.0.elev_max))
    }
    fn exists(&self, id: TileId) -> bool {
        self.available(id)
    }
}
