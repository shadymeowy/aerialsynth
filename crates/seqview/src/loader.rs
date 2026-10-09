//! Frames on demand: a background thread reads what the view wants into a small cache (LRU by
//! bytes). The view says what it wants each frame ([`Loader::want`], the newest list replaces
//! the old, so a scrubbed-past frame is never read) and shows what is in the cache. A
//! blocking loader (snapshots, export) reads on the calling thread instead.

use crate::seq::{FrameData, Key, Sequence};
use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;
use std::sync::Arc;

/// A read result (errors kept, so a broken frame is not read again and again).
pub type Loaded = Arc<Result<FrameData, String>>;

struct Entry {
    data: Loaded,
    used: u64,
    bytes: usize,
}

#[derive(Default)]
struct State {
    wanted: Vec<Key>,
    cache: HashMap<Key, Entry>,
    bytes: usize,
    tick: u64,
    quit: bool,
    busy: bool,
}

struct Inner {
    seq: Arc<Sequence>,
    state: Mutex<State>,
    cv: Condvar,
    budget: usize,
    notify: Box<dyn Fn() + Send + Sync>,
}

impl Inner {
    fn insert(&self, st: &mut State, key: Key, data: Loaded) {
        let bytes = data.as_ref().as_ref().map_or(0, |d| d.bytes()) + 64;
        st.tick += 1;
        let used = st.tick;
        if let Some(old) = st.cache.insert(key, Entry { data, used, bytes }) {
            st.bytes -= old.bytes;
        }
        st.bytes += bytes;
        // evict the least recently used, but never what is wanted now
        while st.bytes > self.budget && st.cache.len() > 1 {
            let wanted = &st.wanted;
            let Some(k) = st.cache.iter().filter(|(k, _)| **k != key && !wanted.contains(k)).min_by_key(|(_, e)| e.used).map(|(k, _)| *k) else { break };
            let e = st.cache.remove(&k).unwrap();
            st.bytes -= e.bytes;
        }
    }

    fn load(&self, key: Key) -> Loaded {
        Arc::new(self.seq.read(key).map_err(|e| format!("{e:#}")))
    }
}

/// The frame cache of one sequence.
pub struct Loader {
    inner: Arc<Inner>,
    blocking: bool,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Loader {
    /// A loader with a background reader; `notify` is called after each read (a repaint).
    pub fn background(seq: Arc<Sequence>, budget: usize, notify: impl Fn() + Send + Sync + 'static) -> Loader {
        let inner = Arc::new(Inner { seq, state: Mutex::new(State::default()), cv: Condvar::new(), budget, notify: Box::new(notify) });
        let worker = inner.clone();
        let thread = std::thread::Builder::new()
            .name("seqview-loader".into())
            .spawn(move || loop {
                let key = {
                    let mut st = worker.state.lock();
                    loop {
                        if st.quit {
                            return;
                        }
                        if let Some(k) = st.wanted.iter().find(|k| !st.cache.contains_key(k)).copied() {
                            st.busy = true;
                            break k;
                        }
                        st.busy = false;
                        worker.cv.wait(&mut st);
                    }
                };
                let data = worker.load(key);
                {
                    let mut st = worker.state.lock();
                    worker.insert(&mut st, key, data);
                    st.busy = false;
                }
                (worker.notify)();
            })
            .expect("starting the loader thread");
        Loader { inner, blocking: false, thread: Some(thread) }
    }

    /// A loader that reads on the calling thread ([`Loader::get`] always returns the data).
    pub fn blocking(seq: Arc<Sequence>, budget: usize) -> Loader {
        Loader {
            inner: Arc::new(Inner { seq, state: Mutex::new(State::default()), cv: Condvar::new(), budget, notify: Box::new(|| {}) }),
            blocking: true,
            thread: None,
        }
    }

    pub fn seq(&self) -> &Arc<Sequence> {
        &self.inner.seq
    }

    /// What the view wants now, most important first (replaces the previous list).
    pub fn want(&self, keys: Vec<Key>) {
        let mut st = self.inner.state.lock();
        if st.wanted != keys {
            st.wanted = keys;
            self.inner.cv.notify_one();
        }
    }

    /// The data of `key` if it is read (blocking loader: read now).
    pub fn get(&self, key: Key) -> Option<Loaded> {
        {
            let mut st = self.inner.state.lock();
            st.tick += 1;
            let tick = st.tick;
            if let Some(e) = st.cache.get_mut(&key) {
                e.used = tick;
                return Some(e.data.clone());
            }
        }
        if !self.blocking {
            return None;
        }
        let data = self.inner.load(key);
        let mut st = self.inner.state.lock();
        self.inner.insert(&mut st, key, data.clone());
        Some(data)
    }

    /// Whether a wanted item is still to be read (never for a blocking loader: it reads what
    /// is asked for).
    pub fn pending(&self) -> bool {
        if self.blocking {
            return false;
        }
        let st = self.inner.state.lock();
        st.busy || st.wanted.iter().any(|k| !st.cache.contains_key(k))
    }

    /// Bytes in the cache.
    pub fn cached_bytes(&self) -> usize {
        self.inner.state.lock().bytes
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.inner.state.lock().quit = true;
        self.inner.cv.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
