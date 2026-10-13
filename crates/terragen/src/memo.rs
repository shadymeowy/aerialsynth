//! Shared memo tables for host-side pure functions of the world (drainage nodes, query
//! results, sites …): sharded read-write locked hash maps, bounded per shard (a full shard is
//! cleared), shared by all threads. Values must be pure functions of their key (include the
//! world's `cache_key` in it), so a value computed twice by two threads is the same and what is
//! cached never changes a result, only its cost.
//!
//! A lookup costs one uncontended read lock (~20 ns): for values that are cheap to recompute,
//! a thread-local map in front of it saves that (see `world::hydro`).

use crate::noise::{FxHashMap, FxHasher};
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

const SHARDS: usize = 64;

/// A bounded, sharded memo table (`static` friendly: [`Memo::new`] is const).
pub struct Memo<K, V> {
    shards: [RwLock<Option<FxHashMap<K, V>>>; SHARDS],
    /// entries per shard before it is cleared
    cap: usize,
}

impl<K: Hash + Eq, V: Clone> Memo<K, V> {
    /// A table of at most about `cap` entries.
    pub const fn new(cap: usize) -> Self {
        Memo { shards: [const { RwLock::new(None) }; SHARDS], cap: cap.div_ceil(SHARDS) }
    }

    #[inline]
    fn shard(&self, k: &K) -> &RwLock<Option<FxHashMap<K, V>>> {
        let mut h = FxHasher::default();
        k.hash(&mut h);
        // (the high bits: FxHash mixes upwards)
        &self.shards[(h.finish() >> 58) as usize % SHARDS]
    }

    /// The cached value of `k`.
    #[inline]
    pub fn get(&self, k: &K) -> Option<V> {
        self.shard(k).read().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(k).cloned())
    }

    /// Cache `v` for `k`.
    pub fn insert(&self, k: K, v: V) {
        let mut g = self.shard(&k).write().unwrap_or_else(|e| e.into_inner());
        let m = g.get_or_insert_with(FxHashMap::default);
        if m.len() >= self.cap {
            m.clear();
        }
        m.insert(k, v);
    }

    /// The cached value of `k`, else `f()` (computed without holding a lock, then cached).
    #[inline]
    pub fn get_or(&self, k: K, f: impl FnOnce() -> V) -> V {
        if let Some(v) = self.get(&k) {
            return v;
        }
        let v = f();
        self.insert(k, v.clone());
        v
    }

    /// Drop every entry (tests).
    pub fn clear(&self) {
        for s in &self.shards {
            *s.write().unwrap_or_else(|e| e.into_inner()) = None;
        }
    }
}
