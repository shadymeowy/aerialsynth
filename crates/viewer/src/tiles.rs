//! Tile service: reads tiles from the store and generates missing ones on background threads,
//! and turns them into GPU-ready payloads (colour + land cover, elevation, slope).
//!
//! The view sends the tiles it wants every frame (most important first); workers take the first
//! ones that are not in flight. The base levels (z0..=base) are generated unconditionally;
//! deeper tiles only while dynamic generation is on.

use geodesy::tiles::{gsd_ew, gsd_ns, pixel_to_latlon, TileId};
use glam::DVec2;
use parking_lot::{Condvar, Mutex};
use rayon::prelude::*;
use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use terragen::tile::Generator;
use tilestore::{Layer, TileData, TileStore, TILE_SIZE};

pub const N: usize = TILE_SIZE;

/// A tile ready for upload.
pub struct Payload {
    pub id: TileId,
    /// albedo (sRGB) + land-cover class in alpha
    pub color: Vec<u8>,
    /// DSM height (m)
    pub elev: Vec<f32>,
    /// slope (east, north) of the DSM, m/m
    pub grad: Vec<half::f16>,
    pub elev_min: f32,
    pub elev_max: f32,
}

#[derive(Default)]
struct State {
    want_load: Vec<TileId>,
    want_gen: Vec<TileId>,
    base: VecDeque<TileId>,
    in_flight: HashSet<TileId>,
}

#[derive(Default)]
pub struct Stats {
    pub loaded: AtomicU64,
    pub generated: AtomicU64,
    /// thread-seconds spent generating (µs)
    pub gen_us: AtomicU64,
    pub base_total: AtomicUsize,
    pub base_done: AtomicUsize,
    pub gen_busy: AtomicUsize,
    pub errors: AtomicU64,
}

struct Shared {
    state: Mutex<State>,
    cv: Condvar,
    results: Mutex<Vec<Payload>>,
    stop: AtomicBool,
    stats: Stats,
    repaint: Box<dyn Fn() + Send + Sync>,
}

pub struct Service {
    pub store: Arc<TileStore>,
    pub gen: Arc<Generator>,
    shared: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Service {
    /// Start the workers; `base` lists the base tiles to generate if missing.
    pub fn start(store: Arc<TileStore>, gen: Arc<Generator>, base: Vec<TileId>, gen_batch: usize, repaint: impl Fn() + Send + Sync + 'static) -> Service {
        let missing: VecDeque<TileId> = base.into_iter().filter(|t| !store.contains(*t)).collect();
        let shared = Arc::new(Shared {
            state: Mutex::new(State { base: missing.clone(), ..Default::default() }),
            cv: Condvar::new(),
            results: Mutex::new(Vec::new()),
            stop: AtomicBool::new(false),
            stats: Stats::default(),
            repaint: Box::new(repaint),
        });
        shared.stats.base_total.store(missing.len(), Ordering::Relaxed);
        let mut threads = Vec::new();
        for k in 0..2 {
            let (sh, st) = (shared.clone(), store.clone());
            threads.push(std::thread::Builder::new().name(format!("tile-load-{k}")).spawn(move || loader(sh, st)).unwrap());
        }
        {
            let (sh, st, g) = (shared.clone(), store.clone(), gen.clone());
            threads.push(std::thread::Builder::new().name("tile-gen".into()).spawn(move || generator(sh, st, g, gen_batch.max(1))).unwrap());
        }
        Service { store, gen, shared, threads }
    }

    /// The tiles the view wants now, most important first (replaces the previous lists).
    pub fn want(&self, load: Vec<TileId>, gen: Vec<TileId>) {
        let mut s = self.shared.state.lock();
        s.want_load = load;
        s.want_gen = gen;
        drop(s);
        self.shared.cv.notify_all();
    }

    pub fn take_results(&self) -> Vec<Payload> {
        std::mem::take(&mut *self.shared.results.lock())
    }

    pub fn stats(&self) -> &Stats {
        &self.shared.stats
    }

    pub fn in_flight(&self) -> usize {
        self.shared.state.lock().in_flight.len()
    }

    pub fn base_pending(&self) -> usize {
        self.shared.state.lock().base.len()
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        self.shared.cv.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        let _ = self.store.flush();
    }
}

fn loader(sh: Arc<Shared>, store: Arc<TileStore>) {
    loop {
        let id = {
            let mut s = sh.state.lock();
            loop {
                if sh.stop.load(Ordering::Relaxed) {
                    return;
                }
                let pick = s.want_load.iter().position(|t| !s.in_flight.contains(t));
                if let Some(i) = pick {
                    let id = s.want_load.remove(i);
                    s.in_flight.insert(id);
                    break id;
                }
                sh.cv.wait(&mut s);
            }
        };
        match store.read_tile(id, &[Layer::Albedo, Layer::Elevation, Layer::Landcover]) {
            Ok(Some(t)) => {
                let p = payload(&t, &store);
                sh.results.lock().push(p);
                sh.stats.loaded.fetch_add(1, Ordering::Relaxed);
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("reading tile {id}: {e:#}");
                sh.stats.errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        sh.state.lock().in_flight.remove(&id);
        (sh.repaint)();
    }
}

fn generator(sh: Arc<Shared>, store: Arc<TileStore>, gen: Arc<Generator>, batch: usize) {
    loop {
        let (ids, n_base) = {
            let mut s = sh.state.lock();
            loop {
                if sh.stop.load(Ordering::Relaxed) {
                    return;
                }
                let mut ids = Vec::new();
                // the base levels first (coarse to fine), then what the view asks for
                while ids.len() < batch {
                    let Some(t) = s.base.pop_front() else { break };
                    if !store.contains(t) && !s.in_flight.contains(&t) {
                        ids.push(t);
                    } else {
                        sh.stats.base_done.fetch_add(1, Ordering::Relaxed);
                    }
                }
                let n_base = ids.len();
                while ids.len() < batch && !s.want_gen.is_empty() {
                    let t = s.want_gen.remove(0);
                    if !s.in_flight.contains(&t) && !store.contains(t) {
                        ids.push(t);
                    }
                }
                if !ids.is_empty() {
                    for t in &ids {
                        s.in_flight.insert(*t);
                    }
                    break (ids, n_base);
                }
                sh.cv.wait(&mut s);
            }
        };
        sh.stats.gen_busy.store(ids.len(), Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        let tiles: Vec<TileData> = ids.par_iter().map(|&id| gen.tile(id)).collect();
        sh.stats.gen_us.fetch_add((t0.elapsed().as_secs_f64() * 1e6) as u64, Ordering::Relaxed);
        if let Err(e) = store.write_tiles(&tiles) {
            eprintln!("writing tiles: {e:#}");
            sh.stats.errors.fetch_add(1, Ordering::Relaxed);
        }
        let payloads: Vec<Payload> = tiles.par_iter().map(|t| payload(t, &store)).collect();
        sh.results.lock().extend(payloads);
        sh.stats.generated.fetch_add(ids.len() as u64, Ordering::Relaxed);
        sh.stats.base_done.fetch_add(n_base, Ordering::Relaxed);
        sh.stats.gen_busy.store(0, Ordering::Relaxed);
        let mut s = sh.state.lock();
        for t in &ids {
            s.in_flight.remove(t);
        }
        drop(s);
        (sh.repaint)();
    }
}

/// GPU payload of a tile: colour + class, elevation, and its slope by central differences
/// (one-sided at the tile edges) with the pixel size of each row.
fn payload(t: &TileData, store: &TileStore) -> Payload {
    let ell = store.meta().ellipsoid();
    let mut color = vec![0u8; N * N * 4];
    for k in 0..N * N {
        if !t.albedo.is_empty() {
            color[4 * k..4 * k + 3].copy_from_slice(&t.albedo[3 * k..3 * k + 3]);
        }
        color[4 * k + 3] = t.landcover.get(k).copied().unwrap_or(0);
    }
    let e = &t.elevation;
    let mut grad = vec![half::f16::ZERO; N * N * 2];
    if e.len() == N * N {
        let id = t.id;
        for j in 0..N {
            let py = (id.y as usize * N + j) as f64 + 0.5;
            let (lat, _) = pixel_to_latlon(DVec2::new(id.x as f64 * N as f64 + 128.0, py), id.z, N as u32);
            let (gx, gy) = (gsd_ew(lat, id.z, N as u32, &ell), gsd_ns(lat, id.z, N as u32, &ell));
            for i in 0..N {
                let (i0, i1) = (i.saturating_sub(1), (i + 1).min(N - 1));
                let (j0, j1) = (j.saturating_sub(1), (j + 1).min(N - 1));
                let de = (e[j * N + i1] - e[j * N + i0]) as f64 / ((i1 - i0) as f64 * gx);
                // rows run south: north is towards j0
                let dn = (e[j0 * N + i] - e[j1 * N + i]) as f64 / ((j1 - j0) as f64 * gy);
                grad[2 * (j * N + i)] = half::f16::from_f64(de);
                grad[2 * (j * N + i) + 1] = half::f16::from_f64(dn);
            }
        }
    }
    // the mesh takes a box-filtered height (the vertices are 8 px apart: single buildings and
    // trees made needles); the full-resolution slope above still shades them
    let elev = if e.len() == N * N { box_filter(e, 4) } else { vec![0.0; N * N] };
    Payload { id: t.id, color, elev, grad, elev_min: t.elev_min, elev_max: t.elev_max }
}

/// Separable (2r+1)² box filter, clamped at the edges.
fn box_filter(e: &[f32], r: usize) -> Vec<f32> {
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0f32; N * N];
        for a in 0..N {
            for b in 0..N {
                let (lo, hi) = (b.saturating_sub(r), (b + r).min(N - 1));
                let mut sum = 0.0;
                for k in lo..=hi {
                    sum += if horizontal { src[a * N + k] } else { src[k * N + a] };
                }
                let v = sum / (hi - lo + 1) as f32;
                if horizontal {
                    out[a * N + b] = v;
                } else {
                    out[b * N + a] = v;
                }
            }
        }
        out
    };
    pass(&pass(e, true), false)
}
