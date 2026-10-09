//! The decoded-tile cache of a [`World`](crate::World): the bytes of a layer of a tile (as the
//! bindings return them), least recently used first out, within a byte budget.
//!
//! One mutex guards the index; it is held only to look up, insert or evict entries (no copying
//! or decoding under it). Values are shared (`Arc`): a hit hands out the buffer and the caller
//! copies from it after the lock is released, so evicting an entry in use is harmless.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard};
use tilestore::{Layer, TileId};

/// The bytes of one layer of one tile.
pub(crate) type Bytes = Arc<Vec<u8>>;

type Key = (TileId, Layer);

#[derive(Default)]
struct Lru {
    /// budget in bytes (0: disabled)
    capacity: usize,
    /// bytes held
    used: usize,
    /// last use of every entry (increasing)
    tick: u64,
    map: HashMap<Key, (Bytes, u64)>,
    /// entries by last use: the first is the least recently used
    order: BTreeMap<u64, Key>,
    hits: u64,
    misses: u64,
}

impl Lru {
    /// Drop entries, least recently used first, until at most `capacity` bytes are held. The
    /// dropped values are returned (to be freed after the lock is released).
    fn evict(&mut self, capacity: usize) -> Vec<Bytes> {
        let mut dropped = Vec::new();
        while self.used > capacity {
            let Some((_, key)) = self.order.pop_first() else { break };
            if let Some((v, _)) = self.map.remove(&key) {
                self.used -= v.len();
                dropped.push(v);
            }
        }
        dropped
    }
}

/// Usage of a tile cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// budget in bytes (0: the cache is off)
    pub capacity: usize,
    /// bytes held
    pub bytes: usize,
    /// layers of tiles held
    pub entries: usize,
    /// lookups that found the layer since the world was opened
    pub hits: u64,
    /// lookups that did not
    pub misses: u64,
}

pub(crate) struct TileCache {
    inner: Mutex<Lru>,
}

impl TileCache {
    pub(crate) fn new(capacity: usize) -> Self {
        TileCache { inner: Mutex::new(Lru { capacity, ..Default::default() }) }
    }

    fn lock(&self) -> MutexGuard<'_, Lru> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The layer of a tile, if cached (it becomes the most recently used).
    pub(crate) fn get(&self, id: TileId, layer: Layer) -> Option<Bytes> {
        let mut c = self.lock();
        if c.capacity == 0 {
            return None;
        }
        c.tick += 1;
        let tick = c.tick;
        let found = c.map.get_mut(&(id, layer)).map(|(v, last)| (v.clone(), std::mem::replace(last, tick)));
        match found {
            Some((v, last)) => {
                c.hits += 1;
                c.order.remove(&last);
                c.order.insert(tick, (id, layer));
                Some(v)
            }
            None => {
                c.misses += 1;
                None
            }
        }
    }

    /// Cache the layers of tiles (in this order: the last is the most recently used), evicting
    /// the least recently used entries beyond the budget. A layer larger than the budget is not
    /// cached.
    pub(crate) fn insert(&self, entries: impl IntoIterator<Item = (TileId, Layer, Bytes)>) {
        let dropped = {
            let mut c = self.lock();
            if c.capacity == 0 {
                return;
            }
            let mut dropped = Vec::new();
            for (id, layer, v) in entries {
                if v.len() > c.capacity {
                    continue;
                }
                c.tick += 1;
                let tick = c.tick;
                c.used += v.len();
                if let Some((old, last)) = c.map.insert((id, layer), (v, tick)) {
                    c.used -= old.len();
                    c.order.remove(&last);
                    dropped.push(old);
                }
                c.order.insert(tick, (id, layer));
                let cap = c.capacity;
                dropped.extend(c.evict(cap));
            }
            dropped
        };
        drop(dropped);
    }

    /// Change the budget (0 turns the cache off and empties it).
    pub(crate) fn set_capacity(&self, capacity: usize) {
        let dropped = {
            let mut c = self.lock();
            c.capacity = capacity;
            c.evict(capacity)
        };
        drop(dropped);
    }

    pub(crate) fn stats(&self) -> CacheStats {
        let c = self.lock();
        CacheStats { capacity: c.capacity, bytes: c.used, entries: c.map.len(), hits: c.hits, misses: c.misses }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: usize, b: u8) -> Bytes {
        Arc::new(vec![b; n])
    }

    #[test]
    fn least_recently_used_out() {
        let c = TileCache::new(300);
        let id = |x| TileId::new(3, x, 0);
        c.insert([(id(0), Layer::Rgb, v(100, 0)), (id(1), Layer::Rgb, v(100, 1)), (id(2), Layer::Rgb, v(100, 2))]);
        assert_eq!(c.stats().bytes, 300);
        assert_eq!(*c.get(id(0), Layer::Rgb).unwrap(), vec![0; 100]); // 0 is now the most recent
        assert!(c.get(id(0), Layer::Elevation).is_none());
        c.insert([(id(3), Layer::Rgb, v(100, 3))]); // evicts 1
        assert!(c.get(id(1), Layer::Rgb).is_none());
        assert!(c.get(id(0), Layer::Rgb).is_some() && c.get(id(2), Layer::Rgb).is_some() && c.get(id(3), Layer::Rgb).is_some());
        // replacing an entry keeps the byte count right
        c.insert([(id(3), Layer::Rgb, v(50, 4))]);
        assert_eq!((c.stats().bytes, c.stats().entries), (250, 3));
        // too large for the budget: not cached, nothing evicted
        c.insert([(id(5), Layer::Rgb, v(301, 5))]);
        assert!(c.get(id(5), Layer::Rgb).is_none());
        assert_eq!(c.stats().entries, 3);
        // shrinking evicts the oldest (0, 2 were used before 3)
        c.set_capacity(60);
        assert_eq!((c.stats().bytes, c.stats().entries), (50, 1));
        assert!(c.get(id(3), Layer::Rgb).is_some());
        c.set_capacity(0);
        assert_eq!((c.stats().bytes, c.stats().entries), (0, 0));
        c.insert([(id(0), Layer::Rgb, v(10, 0))]);
        assert!(c.get(id(0), Layer::Rgb).is_none());
        let s = c.stats();
        assert_eq!((s.capacity, s.entries), (0, 0));
        assert!(s.hits >= 5 && s.misses >= 3);
    }
}
